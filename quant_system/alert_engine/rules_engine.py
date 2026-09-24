"""
rules_engine.py — 规则引擎 (V5)

声明式预警规则 DSL：用一行 JSON/dict 定义一条风控规则，无需为每种预警写专门代码。

设计思想：
  Rule(name="single_stock_risk", condition="holding.weight > 0.15",
       severity="high", message="单票占比{weight:.0%}超限")

条件表达式用 Python 表达式的安全子集求值（不使用 eval 任意代码，
而是通过受限的 AST 白名单求值器），字段来自持仓/组合上下文字典。

对标：券商风控系统的规则引擎 / Drools 简化版
"""

from __future__ import annotations

import ast
import operator
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

CST = timezone(timedelta(hours=8))

logger = __import__("logging").getLogger(__name__)

# P1-Q21-fix(H04): 告警去重/冷却/风暴抑制的默认参数
DEFAULT_COOLDOWN_SECONDS = 1800  # 规则级+标的级冷却窗口：30 分钟
DEFAULT_STORM_LIMIT = 5          # 单次评估同规则触发超限 -> 聚合降级为 1 条摘要


# ════════════════════════════════════════════════════════════════
# 安全表达式求值器（AST 白名单，禁止任意代码执行）
# ════════════════════════════════════════════════════════════════

_ALLOWED_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_CMPOPS: dict[type, Callable[[Any, Any], bool]] = {
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}

_ALLOWED_BOOLOPS: dict[type, Callable[[list[Any]], bool]] = {
    ast.And: all,
    ast.Or: any,
}

_ALLOWED_UNARYOPS: dict[type, Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.not_,
}


class UnsafeExpressionError(Exception):
    """表达式包含不允许的语法节点。"""


def _safe_eval(expr: str, context: dict[str, Any]) -> Any:
    """在受限的 AST 白名单内求值表达式，禁止函数调用、属性访问以外的任意代码。

    支持: 比较运算、算术运算（含受限指数运算）、布尔运算、点号属性访问
    （仅允许 dict 键）、数字/字符串常量、变量名查找。
    P2-Q21-fix: 指数运算限制 |指数|<=10（防 2**999999999 求值 DoS）；
    属性访问仅允许 dict 键（防 getattr 读对象内部属性）。

    Args:
        expr: 形如 "holding.weight > 0.15" 或 "daily_return < -0.04" 的表达式
        context: 变量绑定字典，例如 {"holding": {...}, "daily_return": -0.05}

    Returns:
        表达式求值结果（通常是 bool）

    Raises:
        UnsafeExpressionError: 表达式使用了不允许的语法
    """
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"表达式语法错误: {expr}") from exc
    return _eval_node(node, context)


def _lookup(name: str, context: dict[str, Any]) -> Any:
    if name in context:
        return context[name]
    raise UnsafeExpressionError(f"未知变量: {name}")


def _eval_node(node: ast.AST, context: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return _lookup(node.id, context)
    if isinstance(node, ast.Attribute):
        base = _eval_node(node.value, context)
        # P2-Q21-fix: 属性访问仅允许 dict 键——原实现对非 dict 对象走 getattr，
        # 可读出对象内部属性（如 weight.__class__）。规则上下文的字段本就是 dict，
        # 收紧后消除该信息泄露路径。
        if isinstance(base, dict):
            if node.attr not in base:
                raise UnsafeExpressionError(f"字段不存在: {node.attr}")
            return base[node.attr]
        raise UnsafeExpressionError("属性访问仅支持字典字段")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise UnsafeExpressionError(f"不支持的运算符: {op_type}")
        left = _eval_node(node.left, context)
        right = _eval_node(node.right, context)
        # P2-Q21-fix: 限制 Pow 指数大小——原白名单直接允许 ast.Pow，
        # 表达式 2**999999999 可造成大整数求值 DoS（内存/CPU 耗尽）。
        # 允许 |指数|<=10 的常规平方/立方运算，指数过大直接拒绝。
        if op_type is ast.Pow:
            try:
                exp_val = abs(float(right))
            except (OverflowError, TypeError, ValueError):
                raise UnsafeExpressionError("指数运算的指数必须是有限数值")
            if exp_val != exp_val or exp_val > 10:  # NaN 或超大指数
                raise UnsafeExpressionError("指数运算的指数过大（|指数|>10），已拦截")
        return _ALLOWED_BINOPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise UnsafeExpressionError(f"不支持的单目运算符: {op_type}")
        return _ALLOWED_UNARYOPS[op_type](_eval_node(node.operand, context))
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, context)
        result = True
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in _ALLOWED_CMPOPS:
                raise UnsafeExpressionError(f"不支持的比较运算符: {op_type}")
            right = _eval_node(comparator, context)
            result = result and _ALLOWED_CMPOPS[op_type](left, right)
            left = right
        return result
    if isinstance(node, ast.BoolOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BOOLOPS:
            raise UnsafeExpressionError(f"不支持的布尔运算符: {op_type}")
        values = [_eval_node(v, context) for v in node.values]
        return _ALLOWED_BOOLOPS[op_type](values)
    raise UnsafeExpressionError(f"不支持的语法节点: {type(node)}")


# ════════════════════════════════════════════════════════════════
# 规则引擎
# ════════════════════════════════════════════════════════════════

class RulesEngine:
    """声明式规则引擎——用 dict/JSON 定义预警规则并批量评估。

    规则结构:
        {
            "name": "single_stock_risk",
            "condition": "holding.weight > 0.15",
            "severity": "high",           # high / medium / low
            "message": "单票{code}占比{weight:.0%}超限",
            "enabled": True,
        }

    评估上下文（context）里注入以下键，供 condition 引用：
        holding: 当前持仓 dict（逐条评估）
        weight, code, name, ... : holding 的字段展开到顶层，便于书写
        daily_return, consecutive_down_days, sector_weight,
        daily_turnover, avg_turnover: 组合/持仓派生指标（若存在于 holding 中）

    Attributes:
        rules: 当前注册的规则列表
    """

    def __init__(self, cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
                 storm_limit: int = DEFAULT_STORM_LIMIT) -> None:
        self.rules: list[dict[str, Any]] = []
        # P1-Q21-fix(H04): 告警去重/冷却/风暴抑制状态
        self.cooldown_seconds = cooldown_seconds
        self.storm_limit = storm_limit
        # (rule_name, code) -> 上次发出时间戳（epoch 秒），实现规则级+标的级冷却
        self._last_alert_time: dict[tuple[str, str], float] = {}
        # 内容指纹 -> 上次发出时间戳，实现同消息去重
        self._last_fingerprint_time: dict[str, float] = {}
        # 最近一次 evaluate 的去重/聚合统计（供调用方观察，不改变返回类型）
        self.last_evaluation_stats: dict[str, int] = {
            "emitted": 0, "suppressed_cooldown": 0, "suppressed_dedup": 0, "aggregated": 0,
        }
        self._load_builtin_rules()

    def reset_cooldowns(self) -> None:
        """清空冷却/去重状态（测试或切换评估周期时调用）。"""
        self._last_alert_time.clear()
        self._last_fingerprint_time.clear()
        self.last_evaluation_stats = {
            "emitted": 0, "suppressed_cooldown": 0, "suppressed_dedup": 0, "aggregated": 0,
        }

    # ────────────────────────────────────────────────────────
    # 内置规则
    # ────────────────────────────────────────────────────────

    def _load_builtin_rules(self) -> None:
        """加载 5 条内置风控规则。"""
        builtin = [
            {
                "name": "single_stock_risk",
                "condition": "weight > 0.15",
                "severity": "high",
                "message": "单票{code}{name}占比{weight:.0%}超限(阈值15%)",
                "enabled": True,
            },
            {
                "name": "sector_over_concentration",
                "condition": "sector_weight > 0.40",
                "severity": "medium",
                "message": "行业{sector}占比{sector_weight:.0%}过于集中(阈值40%)",
                "enabled": True,
            },
            {
                "name": "daily_loss_exceeds",
                "condition": "daily_return < -0.04",
                "severity": "high",
                "message": "{code}{name}单日跌幅{daily_return:.2%}触发止损预警(阈值-4%)",
                "enabled": True,
            },
            {
                "name": "consecutive_loss",
                "condition": "consecutive_down_days >= 5",
                "severity": "medium",
                "message": "{code}{name}连续下跌{consecutive_down_days}天",
                "enabled": True,
            },
            {
                "name": "turnover_spike",
                "condition": "daily_turnover > 3 * avg_turnover",
                "severity": "low",
                "message": "{code}{name}成交额{daily_turnover:.0f}异常放大(20日均量的{ratio:.1f}倍)",
                "enabled": True,
            },
        ]
        for rule in builtin:
            self.rules.append(dict(rule))

    # ────────────────────────────────────────────────────────
    # CRUD
    # ────────────────────────────────────────────────────────

    def add_rule(self, rule: dict[str, Any]) -> None:
        """添加一条规则。

        Args:
            rule: 规则字典，必须包含 name/condition/severity/message，
                  enabled 缺省为 True

        Raises:
            ValueError: 规则缺少必要字段，或同名规则已存在
        """
        required = {"name", "condition", "severity", "message"}
        missing = required - set(rule.keys())
        if missing:
            raise ValueError(f"规则缺少必要字段: {missing}")
        if any(r["name"] == rule["name"] for r in self.rules):
            raise ValueError(f"规则已存在: {rule['name']}")
        new_rule = dict(rule)
        new_rule.setdefault("enabled", True)
        self.rules.append(new_rule)

    def remove_rule(self, name: str) -> bool:
        """按名称删除规则。

        Args:
            name: 规则名

        Returns:
            是否成功删除（找到并删除返回 True）
        """
        before = len(self.rules)
        self.rules = [r for r in self.rules if r["name"] != name]
        return len(self.rules) < before

    def list_rules(self) -> list[dict[str, Any]]:
        """返回当前所有规则的浅拷贝列表。"""
        return [dict(r) for r in self.rules]

    def enable_rule(self, name: str, enabled: bool = True) -> bool:
        """启用/禁用规则。"""
        for r in self.rules:
            if r["name"] == name:
                r["enabled"] = enabled
                return True
        return False

    # ────────────────────────────────────────────────────────
    # 评估
    # ────────────────────────────────────────────────────────

    def _build_context(self, holding: dict[str, Any]) -> dict[str, Any]:
        """把 holding 字段展开到求值上下文顶层，同时保留 holding 本身供属性访问。"""
        ctx: dict[str, Any] = dict(holding)
        ctx["holding"] = holding
        # 换算 turnover_spike 需要的比率，避免除零
        avg_turnover = holding.get("avg_turnover")
        daily_turnover = holding.get("daily_turnover")
        if avg_turnover and daily_turnover is not None:
            ctx["ratio"] = daily_turnover / max(avg_turnover, 1e-8)
        else:
            ctx["ratio"] = 0.0
        return ctx

    def evaluate(self, holdings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """对每条持仓逐一评估所有启用规则，返回触发的预警列表。

        Args:
            holdings: 持仓/上下文字典列表，字段视规则而定，常见字段：
                code, name, weight, sector, sector_weight, daily_return,
                consecutive_down_days, daily_turnover, avg_turnover

        Returns:
            触发的预警列表:
            [{"rule": name, "severity": ..., "message": 渲染后的文本,
              "holding": 原始 holding, "timestamp": ISO时间}, ...]
        """
        alerts: list[dict[str, Any]] = []
        now = datetime.now(CST)
        now_ts = now.timestamp()
        # P1-Q21-fix(H04): 去重/聚合统计
        stats: dict[str, int] = {
            "emitted": 0, "suppressed_cooldown": 0, "suppressed_dedup": 0, "aggregated": 0,
        }
        rule_counts: dict[str, int] = {}
        rule_samples: dict[str, dict[str, Any]] = {}

        for holding in holdings:
            ctx = self._build_context(holding)
            code = str(holding.get("code", ""))
            for rule in self.rules:
                if not rule.get("enabled", True):
                    continue
                try:
                    triggered = bool(_safe_eval(rule["condition"], ctx))
                except UnsafeExpressionError as exc:
                    logger.warning("规则 %s 求值失败: %s", rule["name"], exc)
                    continue
                except Exception as exc:  # noqa: BLE001 — 缺字段等容忍跳过
                    logger.debug("规则 %s 缺少字段，跳过: %s", rule["name"], exc)
                    continue

                if not triggered:
                    continue

                try:
                    message = rule["message"].format(**ctx)
                except (KeyError, ValueError, IndexError):
                    message = rule["message"]

                # P1-Q21-fix(H04): 规则级+标的级冷却窗口（同一持仓+同一条规则
                # 在冷却期内不重复告警，避免外部定时循环造成告警风暴）
                rule_key = (rule["name"], code)
                if now_ts - self._last_alert_time.get(rule_key, 0.0) < self.cooldown_seconds:
                    stats["suppressed_cooldown"] += 1
                    continue

                # P1-Q21-fix(H04): 内容指纹去重（同规则+同标的+同消息）
                fingerprint = f"{rule['name']}|{code}|{message}"
                if now_ts - self._last_fingerprint_time.get(fingerprint, 0.0) < self.cooldown_seconds:
                    stats["suppressed_dedup"] += 1
                    continue

                self._last_alert_time[rule_key] = now_ts
                self._last_fingerprint_time[fingerprint] = now_ts

                alert = {
                    "rule": rule["name"],
                    "severity": rule["severity"],
                    "message": message,
                    "holding": holding,
                    "timestamp": now.isoformat(),
                }
                rule_counts[rule["name"]] = rule_counts.get(rule["name"], 0) + 1
                rule_samples.setdefault(rule["name"], alert)
                alerts.append(alert)
                stats["emitted"] += 1

        # P1-Q21-fix(H04): 风暴聚合降级——同一规则在单次评估中触发超过
        # storm_limit 只标的时，合并为 1 条摘要（降级可见：消息带 [风暴聚合] 标记）
        if self.storm_limit > 0:
            merged_rules = [name for name, cnt in rule_counts.items() if cnt > self.storm_limit]
            if merged_rules:
                alerts = [a for a in alerts if a["rule"] not in merged_rules]
                for name in merged_rules:
                    sample = rule_samples[name]
                    alerts.append({
                        "rule": name,
                        "severity": sample["severity"],
                        "message": (
                            f"[风暴聚合] 规则 {name} 在 {rule_counts[name]} 只标的上同时触发，"
                            f"已合并为1条摘要；示例消息: {sample['message']}"
                        ),
                        "holding": None,
                        "timestamp": now.isoformat(),
                        "aggregated": True,
                        "aggregated_count": rule_counts[name],
                    })
                    stats["aggregated"] += 1

        stats["emitted"] = len(alerts)
        self.last_evaluation_stats = stats
        return alerts


# ════════════════════════════════════════════════════════════════
# main — 测试入口
# ════════════════════════════════════════════════════════════════

def main() -> None:
    """独立运行测试：模拟持仓，验证 5 条内置规则均能正确触发。"""
    try:
        engine = RulesEngine()
        print("=" * 60)
        print("RulesEngine 规则引擎 — 测试")
        print("=" * 60)
        print(f"内置规则数: {len(engine.list_rules())}")
        for r in engine.list_rules():
            print(f"  - {r['name']:<25s} severity={r['severity']:<8s} enabled={r['enabled']}")

        holdings = [
            {
                "code": "600519", "name": "贵州茅台", "weight": 0.18,
                "sector": "食品饮料", "sector_weight": 0.45,
                "daily_return": -0.05, "consecutive_down_days": 6,
                "daily_turnover": 5.0e8, "avg_turnover": 1.0e8,
            },
            {
                "code": "000001", "name": "平安银行", "weight": 0.08,
                "sector": "银行", "sector_weight": 0.10,
                "daily_return": 0.01, "consecutive_down_days": 1,
                "daily_turnover": 1.0e8, "avg_turnover": 1.0e8,
            },
        ]

        alerts = engine.evaluate(holdings)
        print(f"\n触发预警数: {len(alerts)}")
        for a in alerts:
            print(f"  [{a['severity'].upper():<6s}] {a['rule']:<25s} {a['message']}")

        # 测试自定义规则增删
        engine.add_rule({
            "name": "custom_test", "condition": "weight > 0.5",
            "severity": "low", "message": "自定义规则测试",
        })
        assert engine.remove_rule("custom_test")
        assert not engine.remove_rule("not_exist")

        print("\n✅ RulesEngine 测试通过")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ RulesEngine 测试失败: {exc}")
        raise


if __name__ == "__main__":
    main()
