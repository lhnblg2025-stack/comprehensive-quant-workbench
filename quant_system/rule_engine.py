"""
可配置止盈止损规则引擎 — 支持移动止盈/时间止损/板块联动止损/ATR止损

功能:
  1. 规则注册: 用户可配置多种风控规则
  2. 规则执行: 对持仓逐一检查是否触犯规则
  3. 规则热加载: 运行时修改 JSON 配置文件即生效
  4. 内置规则库: 移动止盈/固定止损/时间止损/板块联动/ATR跟踪

用法:
  python3 -m quant_system.rule_engine                   # 检查所有持仓
  python3 -m quant_system.rule_engine --portfolio       # 显示规则运行状态
  python3 -m quant_system.rule_engine --rules           # 列出当前所有规则
  python3 -m quant_system.rule_engine --add NAME PARAMS # 添加规则(快速)
"""

from __future__ import annotations

import json
import logging
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))

# 默认规则配置路径
CONFIG_DIR = ROOT.parent / "config"
RULES_FILE = CONFIG_DIR / "trading_rules.json"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ───────── 规则定义 ─────────

class Rule:
    """一条交易规则的基类"""

    def __init__(self, rule_id: str, name: str, enabled: bool = True, **params):
        self.rule_id = rule_id
        self.name = name
        self.enabled = enabled
        self.params = params
        self.created_at = datetime.now(CST).isoformat()

    def check(self, position: dict, market: dict) -> dict | None:
        """检查此规则是否触发

        Args:
            position: {'stock':, 'cost':, 'qty':, 'current_price':, 'entry_date':,
                       'pnl_pct':, 'high_since_entry':}
            market:   {'sector_pnl':, 'index_pnl':}

        Returns:
            None(未触发) 或 dict{'action': 'sell'/'reduce'/'alert',
                                  'reason': str, 'priority': int}
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {
            'type': getattr(self, 'type_name', type(self).__name__),
            'rule_id': self.rule_id,
            'name': self.name,
            'enabled': self.enabled,
            'params': self.params,
            'created_at': self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        rule_cls = RULE_REGISTRY.get(d.get('type', ''))
        if not rule_cls:
            # P2-Q26-fix: 未知规则类型显式报错，不再静默返回 None 丢弃规则
            raise ValueError(f"未知规则类型 {d.get('type')!r}（可用: {sorted(RULE_REGISTRY)}）")
        return rule_cls(d['rule_id'], d['name'], d.get('enabled', True), **d.get('params', {}))


# ───────── 内置规则 ─────────

class FixedStopLoss(Rule):
    """固定百分比止损"""
    type_name = "fixed_stop_loss"

    def check(self, position: dict, market: dict) -> dict | None:
        loss_pct = self.params.get('loss_pct', 0.08)
        if position.get('pnl_pct', 0) <= -loss_pct:
            return {
                'action': 'sell',
                'reason': f"固定止损触发: 亏损{position['pnl_pct']*100:.1f}% ≤ -{loss_pct*100:.0f}%",
                'priority': 1,
            }
        return None


class TrailingStop(Rule):
    """移动止盈止损（跟踪最高价回撤）"""
    type_name = "trailing_stop"

    def check(self, position: dict, market: dict) -> dict | None:
        trail_pct = self.params.get('trail_pct', 0.10)
        high = position.get('high_since_entry', position.get('current_price', 0))
        current = position.get('current_price', 0)
        if high > 0 and current > 0:
            drawdown = (high - current) / high
            if drawdown >= trail_pct:
                return {
                    'action': 'sell',
                    'reason': f"移动止盈触发: 从高点回撤{drawdown*100:.1f}% ≥ {trail_pct*100:.0f}%",
                    'priority': 2,
                }
        return None


class TimeStop(Rule):
    """时间止损（持仓超过N天无条件卖出）"""
    type_name = "time_stop"

    def check(self, position: dict, market: dict) -> dict | None:
        max_days = self.params.get('max_days', 60)
        entry_date = position.get('entry_date')
        if entry_date:
            try:
                if isinstance(entry_date, str):
                    entry_dt = datetime.fromisoformat(entry_date)
                else:
                    entry_dt = entry_date
                # P1-Q26-fix: 统一时区，避免 naive/aware 相减抛 TypeError 导致永不触发
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=CST)
                days_held = (datetime.now(CST) - entry_dt).days
                if days_held >= max_days:
                    return {
                        'action': 'sell',
                        'reason': f"时间止损触发: 持仓{days_held}天 ≥ {max_days}天上限",
                        'priority': 3,
                    }
            except (ValueError, TypeError):
                pass
        return None


class SectorLinkStop(Rule):
    """板块联动止损（同板块股票跌超X%→减仓）"""
    type_name = "sector_link_stop"

    def check(self, position: dict, market: dict) -> dict | None:
        sector_threshold = self.params.get('sector_threshold', 0.05)
        sector_pnl = market.get('sector_pnl', 0)
        if sector_pnl <= -sector_threshold:
            return {
                'action': 'reduce',
                'reason': f"板块联动止损触发: 板块跌幅{-sector_pnl*100:.1f}% ≥ {sector_threshold*100:.0f}%",
                'priority': 4,
            }
        return None


class ProfitLock(Rule):
    """利润锁定（盈利超过X%后锁定Y%利润）"""
    type_name = "profit_lock"

    def check(self, position: dict, market: dict) -> dict | None:
        profit_target = self.params.get('profit_target', 0.20)
        lock_ratio = self.params.get('lock_ratio', 0.5)
        cost = position.get('cost', 0)
        current = position.get('current_price', 0)
        high = position.get('high_since_entry', position.get('current_price', 0))
        if cost > 0 and current > 0 and high > 0:
            # P2-Q26-fix: 统一以成本为基准——峰值/当前盈利均相对成本计算，
            # 不再混用分母（原实现回撤相对持仓高点、盈利相对成本）。
            # 且只要曾触及 profit_target，跳空下跌跌破目标价仍会触发锁定。
            peak_pnl = (high - cost) / cost
            current_pnl = (current - cost) / cost
            if peak_pnl >= profit_target and peak_pnl > 0:
                retained = current_pnl / peak_pnl  # 峰值利润保留比例
                if retained <= lock_ratio:
                    return {
                        'action': 'reduce',
                        'reason': f"利润锁定触发: 峰值盈利{peak_pnl*100:.1f}%→当前{current_pnl*100:.1f}%，"
                                  f"保留{retained*100:.0f}%≤{lock_ratio*100:.0f}%",
                        'priority': 2,
                    }
        return None


class ATRTrailingStop(Rule):
    """ATR移动止损（海龟交易法改良版）"""
    type_name = "atr_trailing_stop"

    def check(self, position: dict, market: dict) -> dict | None:
        atr_multiple = self.params.get('atr_multiple', 3)
        atr = position.get('atr', 0)
        cost = position.get('cost', 0)
        current = position.get('current_price', 0)
        if atr > 0 and cost > 0:
            stop_price = position.get('high_since_entry', cost) - atr_multiple * atr
            if current <= stop_price:
                return {
                    'action': 'sell',
                    'reason': f"ATR跟踪止损: 现价{current:.2f} ≤ 止损线{stop_price:.2f} (ATR×{atr_multiple})",
                    'priority': 1,
                }
        return None


# 规则注册表
RULE_REGISTRY: dict[str, type[Rule]] = {
    FixedStopLoss.type_name: FixedStopLoss,
    TrailingStop.type_name: TrailingStop,
    TimeStop.type_name: TimeStop,
    SectorLinkStop.type_name: SectorLinkStop,
    ProfitLock.type_name: ProfitLock,
    ATRTrailingStop.type_name: ATRTrailingStop,
}

# 默认规则配置
DEFAULT_RULES = [
    {"type": "fixed_stop_loss", "rule_id": "default_sl_8pct", "name": "固定止损8%",
     "enabled": True, "params": {"loss_pct": 0.08}},
    {"type": "trailing_stop", "rule_id": "default_ts_10pct", "name": "移动止盈10%",
     "enabled": True, "params": {"trail_pct": 0.10}},
    {"type": "time_stop", "rule_id": "default_time_60d", "name": "时间止损60天",
     "enabled": True, "params": {"max_days": 60}},
    {"type": "profit_lock", "rule_id": "default_pl_20_50", "name": "利润锁定(20%→50%)",
     "enabled": True, "params": {"profit_target": 0.20, "lock_ratio": 0.5}},
    {"type": "atr_trailing_stop", "rule_id": "default_atr_3x", "name": "ATR跟踪止损3x",
     "enabled": False, "params": {"atr_multiple": 3}},
]


# ───────── 规则引擎 ─────────

class RuleEngine:
    """规则引擎：管理规则加载、执行、热重载"""

    def __init__(self):
        self.rules: list[Rule] = []
        self.load()

    def load(self):
        """从配置文件加载规则"""
        rebuild = False  # 是否因 JSON 损坏而重建默认规则（会落盘覆盖）
        if RULES_FILE.exists():
            try:
                data = json.loads(RULES_FILE.read_text())
            except json.JSONDecodeError as e:
                # P2-Q26-fix: 仅 JSON 语法错误才重建默认规则（覆盖损坏文件）
                logger.warning(f"规则配置 JSON 解析失败({e})，重建默认规则")
                data = None
                rebuild = True
            except Exception as e:
                # P2-Q26-fix: 其他异常（权限/磁盘/编码等）保留原文件并告警，不覆盖
                logger.error(f"规则配置读取失败({e})，保留原文件并回退默认规则")
                data = None
                rebuild = False

            if isinstance(data, dict):
                rules_raw = data.get('rules', [])
                if isinstance(rules_raw, list):
                    self.rules = []
                    for item in rules_raw:
                        try:
                            rule = Rule.from_dict(item)
                        except (ValueError, KeyError, TypeError) as e:
                            # P2-Q26-fix: 单条规则无效（未知类型/缺字段）报错并跳过，
                            # 不再静默丢弃；单条错误不触发整文件覆盖
                            logger.warning(f"跳过无效规则 {item!r}: {e}")
                            continue
                        if rule:
                            self.rules.append(rule)
                    return
                logger.warning("规则配置缺少 rules 列表，使用默认规则（保留原文件）")
                self._load_defaults(save=False)
                return
            # data 为 None：JSON 损坏或其他读取异常
            self._load_defaults(save=rebuild)
            return

        # 首次运行：写入默认规则
        self._load_defaults(save=True)

    def _load_defaults(self, save: bool = True) -> None:
        """加载内置默认规则；save=True 时落盘覆盖配置文件。"""
        self.rules = []
        for item in DEFAULT_RULES:
            rule_cls = RULE_REGISTRY.get(item['type'])
            if rule_cls:
                self.rules.append(rule_cls(
                    item['rule_id'], item['name'],
                    item['enabled'], **item['params']
                ))
        if save:
            self.save()

    def save(self):
        """保存规则到配置文件"""
        data = {'rules': [r.to_dict() for r in self.rules], 'updated_at': datetime.now(CST).isoformat()}
        RULES_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def add_rule(self, rule_type: str, rule_id: str, name: str, **params) -> bool:
        """添加一条规则"""
        rule_cls = RULE_REGISTRY.get(rule_type)
        if not rule_cls:
            return False
        self.rules.append(rule_cls(rule_id, name, True, **params))
        self.save()
        return True

    def remove_rule(self, rule_id: str) -> bool:
        """删除一条规则"""
        before = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        if len(self.rules) != before:
            self.save()
            return True
        return False

    def toggle_rule(self, rule_id: str, enabled: bool = None) -> bool:
        """启用/禁用规则"""
        for r in self.rules:
            if r.rule_id == rule_id:
                if enabled is not None:
                    r.enabled = enabled
                else:
                    r.enabled = not r.enabled
                self.save()
                return True
        return False

    def check_all(self, positions: list[dict], market: dict) -> list[dict]:
        """检查所有持仓是否触发了任何规则

        Args:
            positions: 持仓列表 [{'stock':, 'cost':, 'qty':, ...}]
            market: 市场数据 {'sector_pnl':, 'index_pnl':, ...}

        Returns:
            list[dict]: 触发的告警 [{'stock':, 'action':, 'reason':, 'rule':, ...}]

        P2-Q26-fix: 同一持仓可能同时触发多条规则（如 FixedStopLoss sell +
        ProfitLock reduce + TrailingStop sell）。按 stock 聚合后只保留最高优先级
        动作（priority 值最小；同优先级 sell > reduce > alert），且同动作去重，
        避免执行端逐条执行导致重复卖出/减仓。
        """
        alerts: list[dict] = []
        by_stock: dict[str, dict] = {}
        for pos in positions:
            # P1-Q26-fix: 字段适配层，兼容 trade_db 返回的 symbol/shares/cost_price/...
            pos = _adapt_position(pos)
            # 板块联动：优先按持仓自身 sector 取真实板块盈亏
            m = market
            sector_map = market.get('sector_pnl_map')
            if sector_map and pos.get('sector'):
                m = dict(market)
                m['sector_pnl'] = sector_map.get(pos['sector'], market.get('sector_pnl', 0.0))
            best: dict | None = None
            for rule in self.rules:
                if not rule.enabled:
                    continue
                result = rule.check(pos, m)
                if not result:
                    continue
                alert = {
                    'stock': pos.get('stock', ''),
                    'stock_name': pos.get('stock_name', ''),
                    'action': result['action'],
                    'reason': result['reason'],
                    'rule_id': rule.rule_id,
                    'rule_name': rule.name,
                    'priority': result['priority'],
                    'current_price': pos.get('current_price', 0),
                    'cost': pos.get('cost', 0),
                    'pnl_pct': pos.get('pnl_pct', 0),
                }
                if best is None or _higher_priority(alert, best):
                    best = alert
            if best is not None:
                by_stock[best.get('stock', pos.get('stock', ''))] = best
        # 按优先级排序
        alerts = list(by_stock.values())
        alerts.sort(key=lambda x: x.get('priority', 99))
        return alerts

    def format_rules(self) -> str:
        """格式化规则列表"""
        lines = ["\n## 📋 交易规则列表\n"]
        lines.append(f"{'ID':<20}{'名称':<20}{'类型':<18}{'状态'}")
        lines.append("-" * 65)
        for r in self.rules:
            status = "✅ 启用" if r.enabled else "⏸️ 停用"
            type_name = getattr(r, 'type_name', type(r).__name__)
            lines.append(f"  {r.rule_id:<18} {r.name[:16]:<18} {type_name:<16} {status}")
        return "\n".join(lines)


# ───────── P2-Q26-fix: 规则触发优先级比较 ─────────

def _higher_priority(a: dict, b: dict) -> bool:
    """a 是否比 b 更优先：priority 值小优先；同 priority 时 sell > reduce > alert。"""
    pa, pb = a.get('priority', 99), b.get('priority', 99)
    if pa != pb:
        return pa < pb
    severity = {'sell': 0, 'reduce': 1, 'alert': 2}
    return severity.get(a.get('action'), 3) < severity.get(b.get('action'), 3)


# ───────── P1-Q26-fix: 持仓字段适配层 ─────────

_ATR_CACHE: dict[str, tuple[float, float]] = {}  # code -> (atr, 时间戳)


def _compute_atr(code: str, period: int = 14) -> float:
    """用日K计算 ATR(14)。数据源优先 easy-tdx，失败返回0（规则不触发）。"""
    now = _time.time()
    cached = _ATR_CACHE.get(code)
    if cached and now - cached[1] < 3600:
        return cached[0]
    atr = 0.0
    try:
        from quant_system.integrations.tdx_adapter import historical_bars
        bars = historical_bars(code, freq="daily", count=period * 3)
        if bars is not None and len(bars) >= period + 1:
            high = bars["high"].astype(float)
            low = bars["low"].astype(float)
            close = bars["close"].astype(float)
            prev_close = close.shift(1)
            tr = pd.concat([high - low,
                            (high - prev_close).abs(),
                            (low - prev_close).abs()], axis=1).max(axis=1)
            atr = float(tr.iloc[-period:].mean())
    except Exception:
        atr = 0.0
    _ATR_CACHE[code] = (atr, now)
    return atr


def _adapt_position(p: dict) -> dict:
    """把 trade_db 持仓字段适配为规则引擎字段。

    trade_db:    symbol/name/shares/cost_price/current_price/pnl_pct(百分数)/buy_date/highest_price
    rule_engine: stock/stock_name/qty/cost/current_price/pnl_pct(小数)/entry_date/high_since_entry/atr
    """
    # 已是规则引擎格式（含 stock 键）直接返回
    if "stock" in p and ("qty" in p or "cost" in p):
        return p

    cost = float(p.get("cost_price") or 0)
    current = float(p.get("current_price") or 0) or cost
    high = float(p.get("highest_price") or 0)
    if high <= 0:
        high = current
    high = max(high, current)
    # trade_db 的 pnl_pct 为百分数（如 -8.0 表示 -8%），规则引擎用小数（-0.08）
    pnl_pct = float(p.get("pnl_pct") or 0) / 100.0
    return {
        "stock": p.get("symbol", ""),
        "stock_name": p.get("name", ""),
        "qty": p.get("shares", 0),
        "cost": cost,
        "current_price": current,
        "entry_date": p.get("buy_date") or p.get("entry_date", ""),
        "high_since_entry": high,
        "pnl_pct": pnl_pct,
        "atr": float(p.get("atr") or 0) or _compute_atr(p.get("symbol", "")),
        "sector": p.get("sector", ""),
    }


def _fetch_index_pnl() -> float:
    """获取上证指数当日涨跌幅(小数)。真实数据源：腾讯行情；失败返回0（不硬编码）。"""
    try:
        import requests
        r = requests.get(
            "http://qt.gtimg.cn/q=sh000001",
            headers={"Referer": "https://gu.qq.com", "User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        line = r.text.strip().split(";\n")[0]
        parts = line.split("~")
        if len(parts) > 32 and parts[32]:
            return float(parts[32]) / 100.0
    except Exception as e:
        logger.error(f"[rule_engine] 操作失败: {e}", exc_info=True)
    return 0.0


def _load_market_data(positions: list[dict]) -> dict:
    """从真实数据源加载市场数据，禁止硬编码虚构值。

    - index_pnl: 上证指数真实涨跌幅（腾讯行情）
    - sector_pnl: 按持仓实际盈亏分板块聚合（真实持仓数据），供 SectorLinkStop 使用
    """
    market: dict[str, Any] = {"sector_pnl": 0.0, "index_pnl": _fetch_index_pnl()}
    sector_map: dict[str, list[float]] = {}
    for p in positions:
        sec = p.get("sector", "") or "未分类"
        # 兼容 trade_db(百分数) 与已适配(小数) 两种 pnl_pct
        val = float(p.get("pnl_pct") or 0)
        if val == 0 or abs(val) > 1:  # 百分数形态 → 转小数
            val = val / 100.0
        sector_map.setdefault(sec, []).append(val)
    sector_pnl = {sec: sum(v) / len(v) for sec, v in sector_map.items() if v}
    if sector_pnl:
        market["sector_pnl_map"] = sector_pnl
        market["sector_pnl"] = sum(sector_pnl.values()) / len(sector_pnl)
    return market


# ───────── 格式化告警 ─────────

def format_alerts(alerts: list[dict]) -> str:
    """格式化规则告警"""
    if not alerts:
        return "\n✅ 所有持仓无规则触发"

    lines = ["\n🚨 规则触发告警\n"]
    lines.append(f"{'股票':<12}{'动作':<8}{'规则':<16}{'理由'}")
    lines.append("-" * 60)
    for a in alerts:
        stock = a.get('stock_name', a.get('stock', ''))
        action = a.get('action', '')
        rule = a.get('rule_name', '')
        reason = a.get('reason', '')
        action_emoji = "🔴卖出" if action == 'sell' else "🟡减仓"
        lines.append(f"  {stock:<10} {action_emoji:<8} {rule:<14} {reason[:40]}")
    return "\n".join(lines)


# ───────── 报告格式 ─────────

def format_full_report(positions: list[dict] = None) -> str:
    """生成完整风控报告"""
    engine = RuleEngine()
    parts = []
    now = datetime.now(CST)
    parts.append(f"# 🛡️ 风控规则引擎 ({now.strftime('%Y-%m-%d %H:%M')})")
    parts.append(f"{'='*50}")

    # 规则列表
    parts.append(engine.format_rules())

    # 如果提供了持仓，检查规则
    if positions:
        # P1-Q26-fix: 市场数据从真实源获取，禁止硬编码虚构值
        market = _load_market_data(positions)
        alerts = engine.check_all(positions, market)
        parts.append(format_alerts(alerts))

    return "\n".join(parts)


# ───────── CLI ─────────

def main():
    args = sys.argv[1:]
    engine = RuleEngine()

    if "--rules" in args:
        print(engine.format_rules())
    elif "--add" in args:
        idx = args.index("--add")
        if idx + 2 < len(args):
            rule_type = args[idx + 1]
            rule_id = args[idx + 2]
            params = {}
            if idx + 3 < len(args):
                import json as _json
                try:
                    params = _json.loads(args[idx + 3])
                except (_json.JSONDecodeError, IndexError):
                    pass
            if engine.add_rule(rule_type, rule_id, rule_id, **params):
                print(f"✅ 已添加规则: {rule_id} ({rule_type})")
            else:
                print(f"❌ 未知规则类型: {rule_type}，可用: {list(RULE_REGISTRY.keys())}")
        else:
            print("用法: --add TYPE RULE_ID [JSON_PARAMS]")
    elif "--remove" in args:
        idx = args.index("--remove")
        if idx + 1 < len(args):
            if engine.remove_rule(args[idx + 1]):
                print(f"✅ 已删除规则: {args[idx + 1]}")
            else:
                print(f"❌ 未找到规则: {args[idx + 1]}")
    elif "--toggle" in args:
        idx = args.index("--toggle")
        if idx + 1 < len(args):
            if engine.toggle_rule(args[idx + 1]):
                print(f"✅ 已切换规则状态: {args[idx + 1]}")
            else:
                print(f"❌ 未找到规则: {args[idx + 1]}")
    elif "--portfolio" in args:
        # 从 trade_db 加载持仓
        try:
            from quant_system.trade_db import get_positions
            positions = get_positions()
            # P1-Q26-fix: 市场数据从真实源获取，禁止硬编码虚构值
            market = _load_market_data(positions)
            alerts = engine.check_all(positions, market)
            print(format_alerts(alerts))
        except ImportError:
            print("⚠️ 无法加载持仓数据，请先建仓")
    else:
        print(format_full_report())


if __name__ == "__main__":
    main()
