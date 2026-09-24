"""
多策略组合引擎 — 策略注册/信号融合/资金分配/组合优化

功能:
  1. 策略注册: 将各信号源注册为标准策略单元
  2. 信号融合: 多策略加权投票/净权重差置信度聚合
  3. 资金分配: 静态默认权重(0.35/0.25/0.20) + set_weight 调整 + 每期归一化再平衡
  4. 执行建议: 生成最终交易指令

注意(P2-Q17-fix M141):
  等权/风险平价/凯利公式/马科维茨组合优化并未在本实现中提供——
  原文档虚构功能声明已移除, 与实际实现保持一致。

策略架构:
  Strategy (基类)
  ├── MLSignalStrategy    ml_signals.py 的ML预测
  ├── OpportunityStrategy opportunity.py 的机会扫描
  ├── TechnicalStrategy   indicators.py 的技术指标
  ├── SectorRotationStrat sector_rotation.py 的行业轮动
  └── SentimentStrategy   (预留) 情感分析

用法:
  python3 -m quant_system.strategy_engine              # 运行组合策略
  python3 -m quant_system.strategy_engine --list       # 列出已注册策略
  python3 -m quant_system.strategy_engine --backtest   # 组合回测
  python3 -m quant_system.strategy_engine --weights    # 当前权重
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant_system.context import AsOfContext, RunContext, coerce_as_of, coerce_run_context
from quant_system.market_clock import latest_completed_trading_day, prev_trading_day

logger = __import__('logging').getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))
CONFIG_DIR = ROOT.parent / "config"


# P2-Q17-fix(M137): 统一股票键规范化函数——剥离 .SH/.SZ/.BJ 后缀与字母前缀,
# 三策略(ML/机会/技术)共用, 避免"同股异键"导致信号被拆分、权重稀释。
def normalize_stock_key(stock: str) -> str:
    s = str(stock or '').strip().upper()
    if not s:
        return ''
    for suf in ('.SH', '.SZ', '.BJ', '.XSHG', '.XSHE', '.XSHZ'):
        if s.endswith(suf):
            s = s[:-len(suf)]
            break
    digits = ''.join(ch for ch in s if ch.isdigit())
    return digits.zfill(6) if digits else s


# P2-Q17-fix(M140): 硬编码阈值全部外置为可配置参数(ENGINE_CONFIG),
# 避免参数过拟合; 可用 configure_engine() 运行时覆盖, 亦可在敏感性分析后调整。
ENGINE_CONFIG = {
    'net_score_buy': 0.15,           # 融合净评分买入阈值
    'confidence_buy': 0.30,          # 融合置信度买入阈值
    'max_single_weight': 0.15,       # 单票仓位上限
    'expected_return_per_net': 0.01,  # 预期收益 = net_score * 系数
    'ml_buy_threshold': 65,          # ML 评分买入分档
    'ml_sell_threshold': 35,         # ML 评分卖出分档
    'rsi_oversold1': 35,             # RSI 超卖一档
    'rsi_oversold2': 25,             # RSI 超卖二档
    'rsi_overbought': 70,            # RSI 超买
    'ma_discount': 0.95,             # 收盘 < MA20*0.95 加分
    'ma_premium': 1.05,              # 收盘 > MA20*1.05 减分
    'min_signal_score': 2,           # 技术策略最低 |score|
    'top_candidates': 20,            # 单策略候选数上限
    'enable_slow_strategies': False,  # 慢网络策略(ML/机会/技术)默认关闭，避免境外IP+网络拖垮扫描
}


def configure_engine(**overrides) -> None:
    """P2-Q17-fix(M140): 运行时覆盖引擎参数(供敏感性分析/回测扫描使用)。"""
    ENGINE_CONFIG.update(overrides)


# ───────── 数据模型 ─────────

@dataclass
class StrategySignal:
    """一条策略信号"""
    stock: str                     # 股票代码
    stock_name: str = ""
    direction: int = 0            # 1=买入, -1=卖出, 0=持有
    confidence: float = 0.0       # 置信度 [0, 1]
    score: float = 0.0            # 原始分数
    reason: str = ""
    strategy: str = ""            # 来源策略名
    target_price: float = 0.0
    stop_loss: float = 0.0
    timestamp: str = ""


@dataclass
class StrategyAllocation:
    """策略对某只股票的配置建议"""
    stock: str
    stock_name: str = ""
    weight: float = 0.0           # 建议仓位比例 [0, 1]
    expected_return: float = 0.0
    expected_risk: float = 0.0
    confidence: float = 0.0
    signals: list[StrategySignal] = field(default_factory=list)


# ───────── 策略基类 ─────────

class Strategy(ABC):
    """策略基类"""

    fast = False  # True=本地快速策略，主流程串行先跑，不被慢网络策略拖死

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight          # 策略在组合中的权重
        self.enabled = True
        self.signals: list[StrategySignal] = []

    @abstractmethod
    def generate_signals(self, stocks: list[str]) -> list[StrategySignal]:
        """生成信号

        Args:
            stocks: 需要评估的股票列表

        Returns:
            list[StrategySignal]: 信号列表
        """
        pass

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'weight': self.weight,
            'enabled': self.enabled,
            'type': type(self).__name__,
        }


# ───────── 内置策略实现 ─────────

class MLSignalStrategy(Strategy):
    """ML模型预测信号"""

    def __init__(self, weight: float = 0.3):
        super().__init__("ML信号", weight)

    def generate_signals(self, stocks: list[str]) -> list[StrategySignal]:
        signals = []
        try:
            from quant_system.ml_signals import predict_stock
            for stock in stocks[:20]:  # 限制数量
                try:
                    result = predict_stock(stock)
                    if result and 'ml_score' in result:
                        score = result['ml_score']
                        # P2-Q17-fix(M140): ML 分档阈值外置到 ENGINE_CONFIG
                        ml_buy = ENGINE_CONFIG['ml_buy_threshold']
                        ml_sell = ENGINE_CONFIG['ml_sell_threshold']
                        direction = 1 if score >= ml_buy else (-1 if score <= ml_sell else 0)
                        confidence = abs(score - 50) / 50
                        signals.append(StrategySignal(
                            # P2-Q17-fix(M137): 统一股票键规范化, 避免同股异键拆分信号
                            stock=normalize_stock_key(stock),
                            direction=direction,
                            confidence=confidence,
                            score=score,
                            reason=f"ML评分{score:.0f}/100: {result.get('recommendation', '')[:40]}",
                            strategy=self.name,
                        ))
                except Exception as e:
                    logger.error(f"[strategy_engine] 操作失败: {e}", exc_info=True)
                    continue
        except ImportError:
            pass
        self.signals = signals
        return signals


class OpportunityStrategy(Strategy):
    """买入机会扫描"""

    def __init__(self, weight: float = 0.2):
        super().__init__("机会扫描", weight)

    def generate_signals(self, stocks: list[str]) -> list[StrategySignal]:
        signals = []
        try:
            from quant_system.opportunity import scan_opportunities
            opps = scan_opportunities(top_n=30, min_score=2)
            for opp in opps:
                sym = str(opp.get('symbol', ''))
                count = opp.get('signal_count', 0)
                if count >= 3:
                    signals.append(StrategySignal(
                        # P2-Q17-fix(M137): opp['symbol'] 可能带后缀, 统一规范化
                        stock=normalize_stock_key(sym),
                        stock_name=str(opp.get('name', '')),
                        direction=1,
                        confidence=min(1.0, count / 10),
                        score=count,
                        reason=f"机会扫描: {opp.get('signal_summary', '')}",
                        strategy=self.name,
                    ))
        except ImportError:
            pass
        self.signals = signals
        return signals


class TechnicalStrategy(Strategy):
    """技术指标策略（RSI/CCI/MA突破等）"""

    def __init__(self, weight: float = 0.2):
        super().__init__("技术指标", weight)

    def generate_signals(self, stocks: list[str]) -> list[StrategySignal]:
        signals = []
        try:
            import akshare as ak
            for stock in stocks[:ENGINE_CONFIG['top_candidates']]:
                try:
                    code = normalize_stock_key(stock)  # P2-Q17-fix(M137): 统一股票键规范化
                    if code.startswith(('60', '00', '30')):
                        hist = ak.stock_zh_a_hist(symbol=code, period="daily",
                                                   start_date=(datetime.now(CST) - timedelta(days=120)).strftime('%Y%m%d'),
                                                   end_date=datetime.now(CST).strftime('%Y%m%d'),
                                                   # P1-Q17-fix(H11): 后复权避免前复权重述历史引入前视偏差
                                                   adjust="hfq")
                        if hist is not None and len(hist) > 20:
                            close = hist['收盘'].values
                            # RSI
                            delta = np.diff(close)
                            gain = np.where(delta > 0, delta, 0)
                            loss = np.where(delta < 0, -delta, 0)
                            avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
                            avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
                            rsi = 100 - 100 / (1 + avg_gain / avg_loss) if avg_loss != 0 else 100
                            # MA
                            ma20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]

                            score = 0
                            votes = 0
                            # P2-Q17-fix(M138): 方向由净投票决定(不再被最后一次赋值覆盖),
                            # 且 score 与 direction 符号必须一致后才发信号,
                            # 避免"正向评分+卖出方向"的矛盾信号。
                            # P2-Q17-fix(M140): RSI/MA 阈值外置到 ENGINE_CONFIG
                            if rsi < ENGINE_CONFIG['rsi_oversold1']:
                                score += 2
                                votes += 1
                            if rsi < ENGINE_CONFIG['rsi_oversold2']:
                                score += 2
                                votes += 1
                            if close[-1] < ma20 * ENGINE_CONFIG['ma_discount']:
                                score += 1
                                votes += 1
                            if close[-1] > ma20 * ENGINE_CONFIG['ma_premium']:
                                score -= 1
                                votes -= 1
                            if rsi > ENGINE_CONFIG['rsi_overbought']:
                                score -= 2
                                votes -= 1

                            direction = 1 if votes > 0 else (-1 if votes < 0 else 0)
                            score_dir_ok = (score > 0) == (direction > 0)
                            if direction != 0 and abs(score) >= ENGINE_CONFIG['min_signal_score'] and score_dir_ok:
                                signals.append(StrategySignal(
                                    stock=code,
                                    direction=direction,
                                    confidence=min(abs(score) / 6, 1.0),
                                    score=score,
                                    reason=f"RSI={rsi:.0f}, MA20={ma20:.2f}",
                                    strategy=self.name,
                                ))
                except Exception as e:
                    logger.error(f"[strategy_engine] 操作失败: {e}", exc_info=True)
                    continue
        except ImportError:
            pass
        self.signals = signals
        return signals


class FactorStrategy(Strategy):
    """OOS 验证核心因子截面策略。

    把「因子研究」接进「组合决策」的桥：从 generated/factor_quality_registry.json
    读取 tier=core 且含样本外 ICIR 的因子，用 direction 校准方向、|oos_icir| 作为
    权重，对候选池做截面多因子打分——分数最高的个股发买入信号、最低的发卖出信号。
    只使用数据仓库已落盘的日K（无网络），因子口径与 scripts/ic_vectorized.py 的
    IC/OOS 报告完全一致，避免另起炉灶产生"假因子"。
    """

    REGISTRY_PATH = ROOT.parent / "generated" / "factor_quality_registry.json"
    KLINE_DIR = ROOT.parent / "data_warehouse" / "kline"
    TOP_K = 8          # 买入候选数
    BOTTOM_K = 5       # 卖出候选数
    MIN_OOS_ICIR = 0.5  # 只保留样本外 ICIR 绝对值达标的因子

    fast = True  # 本地K线截面因子，无网络，主流程串行先跑

    def __init__(self, weight: float = 0.25):
        super().__init__("核心因子", weight)
        self._core: dict[str, dict] | None = None

    def _load_core(self) -> dict[str, dict]:
        """载入 OOS 有效核心因子 {factor: {direction, weight}}，进程内缓存。"""
        if self._core is not None:
            return self._core
        core: dict[str, dict] = {}
        try:
            data = json.loads(self.REGISTRY_PATH.read_text(encoding="utf-8"))
            for f in data.get("factors", []):
                if f.get("tier") != "core":
                    continue
                oos_icir = f.get("oos_icir")
                if oos_icir is None:
                    continue
                w = abs(float(oos_icir))
                if w < self.MIN_OOS_ICIR:
                    continue
                # 方向以实测全样本 IC 为准（sign(ic_mean)），避免登记表先验方向
                # 与实测相反而把有效因子用反（如动量因子登记为正向、全样本 IC 为负）。
                ic_mean = f.get("ic_mean")
                if ic_mean is not None and float(ic_mean) != 0:
                    direction = 1 if float(ic_mean) > 0 else -1
                else:
                    direction = int(f.get("direction") or 1)
                core[f["factor"]] = {
                    "direction": direction,
                    "weight": w,
                }
        except Exception as exc:  # noqa: BLE001
            logger.error("[strategy_engine] 核心因子登记读取失败: %s", exc)
        self._core = core
        return self._core

    MAX_STALE_TRADING_DAYS = 3

    @staticmethod
    def _load_klines(stocks: list[str], as_of: str | datetime | None = None) -> dict[str, pd.DataFrame]:
        """读取严格截至最近已完成交易日且足够新鲜的日 K。"""
        cutoff = latest_completed_trading_day(as_of)
        oldest = cutoff
        for _ in range(FactorStrategy.MAX_STALE_TRADING_DAYS):
            oldest = prev_trading_day(oldest) or oldest
        kl: dict[str, pd.DataFrame] = {}
        for code in stocks:
            code = normalize_stock_key(code)
            # 用户只能买主板：仅沪市60/深市00；排除科创板688、创业板30、北交所8/4。
            if not code or not code[0] in ("6", "0"):
                continue
            if code.startswith(("688", "8", "4")):
                continue
            p = FactorStrategy.KLINE_DIR / f"{code}.parquet"
            if not p.exists():
                continue
            try:
                df = pd.read_parquet(p)
                date_col = next((c for c in ("date", "日期", "trade_date") if c in df.columns), None)
                if date_col is None:
                    continue
                dates = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
                df = df.assign(date=dates).dropna(subset=["date"])
                df = df[df["date"].dt.date <= cutoff].sort_values("date").drop_duplicates("date", keep="last")
                if len(df) < 60 or df["date"].iloc[-1].date() < oldest:
                    continue
                kl[code] = df.reset_index(drop=True)
            except Exception:  # noqa: BLE001
                continue
        return kl

    def _factor_scores(self, stocks: list[str], as_of: str | datetime | None = None) -> pd.DataFrame:
        """返回 DataFrame(index=symbol, columns=核心因子)，截面原始因子值。"""
        import sys as _sys
        scripts_dir = str(ROOT.parent / "scripts")
        if scripts_dir not in _sys.path:
            _sys.path.insert(0, scripts_dir)
        import ic_vectorized as icv

        kl = self._load_klines(stocks, as_of=as_of)
        rows: dict[str, dict[str, float]] = {}
        for code, df in kl.items():
            try:
                series = icv._zoo_series_all(df)
                series.update(icv._gtja_series_all(df))
                latest = {}
                for name, s in series.items():
                    vals = s.dropna()
                    if len(vals):
                        latest[name] = float(vals.iloc[-1])
                if latest:
                    rows[code] = latest
            except Exception:  # noqa: BLE001
                continue
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame.from_dict(rows, orient="index")
        return frame

    def generate_signals(self, stocks: list[str], as_of: str | datetime | None = None) -> list[StrategySignal]:
        core = self._load_core()
        if not core:
            return []
        frame = self._factor_scores(stocks, as_of=as_of)
        if frame.empty:
            return []

        # 只保留登记里存在的因子列
        cols = [c for c in core if c in frame.columns]
        if len(cols) < 2:
            return []
        sub = frame[cols].astype(float)

        # 截面 z-score（按因子标准化），缺失值填 0（中性）
        z = (sub - sub.mean()) / (sub.std().replace(0, np.nan))
        z = z.fillna(0.0)

        # 方向校准 + OOS ICIR 加权合成
        direction = pd.Series({c: core[c]["direction"] for c in cols})
        weight = pd.Series({c: core[c]["weight"] for c in cols})
        composite = (z * direction * weight).sum(axis=1) / weight.sum()
        composite = composite.sort_values(ascending=False)

        signals: list[StrategySignal] = []
        top = composite.head(self.TOP_K)
        bottom = composite.tail(self.BOTTOM_K)
        max_abs = max(float(composite.abs().max()), 1e-6)
        for code, score in top.items():
            if score <= 0:
                continue
            signals.append(StrategySignal(
                stock=code,
                direction=1,
                confidence=min(0.9, float(score) / max_abs * 0.7),
                score=float(score),
                reason=f"核心因子合成{score:+.2f}（{len(cols)}因子OOS加权）",
                strategy=self.name,
            ))
        for code, score in bottom.items():
            if score >= 0:
                continue
            signals.append(StrategySignal(
                stock=code,
                direction=-1,
                confidence=min(0.9, abs(float(score)) / max_abs * 0.7),
                score=float(score),
                reason=f"核心因子合成{score:+.2f}（弱侧回避）",
                strategy=self.name,
            ))
        self.signals = signals
        return signals


# ───────── 组合引擎 ─────────

class StrategyEngine:
    """多策略组合引擎"""

    def __init__(self):
        self.strategies: list[Strategy] = []
        self._register_defaults()

    def _register_defaults(self):
        """注册默认策略"""
        self.strategies = [
            MLSignalStrategy(0.30),
            OpportunityStrategy(0.20),
            TechnicalStrategy(0.20),
            FactorStrategy(0.30),
        ]

    def register(self, strategy: Strategy):
        """注册一个策略"""
        self.strategies.append(strategy)

    def unregister(self, name: str) -> bool:
        """注销一个策略"""
        before = len(self.strategies)
        self.strategies = [s for s in self.strategies if s.name != name]
        return len(self.strategies) != before

    def set_weight(self, name: str, weight: float):
        """调整策略权重"""
        for s in self.strategies:
            if s.name == name:
                s.weight = weight
                break

    def get_all_stocks(self) -> list[str]:
        """获取所有候选股票"""
        try:
            from quant_system.watchlist import get_watchlist
        except ImportError as e:
            # P2-Q17-fix(L143): ImportError 单独捕获(模块缺失), 与运行期异常分开记录,
            # 不再使用冗余的 (ImportError, Exception) 元组
            logger.warning("Q17-L143: watchlist 模块不可用(%s)，回退到默认候选池", e)
            return ["600519", "601288", "600036", "600900", "601166"]
        try:
            stocks = get_watchlist()
            # P1-Q17-fix: get_watchlist() 返回键为 'symbol'（409只均为 symbol 键，'code' 恒为空），
            # 统一改用 symbol 并剔除空值；候选为空时显式告警而非静默返回空串列表。
            codes = [str(s.get('symbol', '') or s.get('code', '')).strip() for s in stocks[:50]]
            codes = [c for c in codes if c]
            if not codes:
                logger.warning("Q17-H01: get_watchlist() 未返回有效股票代码，回退到默认候选池")
                return ["600519", "601288", "600036", "600900", "601166"]
            return codes
        except Exception as e:
            logger.warning("Q17-L143: get_watchlist() 运行异常(%s)，回退到默认候选池", e)
            return ["600519", "601288", "600036", "600900", "601166"]

    def run(self, stocks: list[str] = None, *, as_of: str | datetime | None = None, context: RunContext | None = None, release_id: str | None = None) -> dict:
        """运行所有策略，生成组合建议

        A same-day data release is required for production runs. Passing
        ``release_id`` binds the resulting allocations to immutable inputs.

        Returns:
            dict: {allocation: [StrategyAllocation], meta: {...}}
        """
        if stocks is None:
            stocks = self.get_all_stocks()
        run_context = coerce_run_context(context, as_of=as_of)
        cutoff = run_context.iso_date
        if release_id is None and cutoff:
            try:
                from quant_system.data_release import load_release
                release = load_release(ROOT.parent, expected_day=cutoff)
                release_id = release["release_id"]
            except Exception as exc:
                raise RuntimeError(f"strategy data release required for {cutoff}: {exc}") from exc

        # 1. 各策略生成信号：fast 策略(本地核心因子)串行先跑保证必出信号；
        #    慢网络策略(ML/机会/技术)默认关闭，需显式 enable_slow_strategies 才跑，
        #    避免境外IP访问国内源(东财)被代理阻断反复重试拖垮整条扫描(用户"卡"的根因)。
        all_signals: list[StrategySignal] = []
        strategy_results = {}
        enabled = [s for s in self.strategies if s.enabled]
        fast_strats = [s for s in enabled if getattr(s, 'fast', False)]
        slow_strats = ([s for s in enabled if not getattr(s, 'fast', False)]
                       if ENGINE_CONFIG.get('enable_slow_strategies') else [])

        for strategy in fast_strats:
            try:
                if cutoff and isinstance(strategy, FactorStrategy):
                    sigs = strategy.generate_signals(stocks, as_of=cutoff)
                else:
                    sigs = strategy.generate_signals(stocks)
                all_signals.extend(sigs)
                strategy_results[strategy.name] = {'count': len(sigs), 'weight': strategy.weight}
            except Exception as e:  # noqa: BLE001
                strategy_results[strategy.name] = {'error': str(e)}

        for strategy in slow_strats:
            try:
                if cutoff and isinstance(strategy, FactorStrategy):
                    sigs = strategy.generate_signals(stocks, as_of=cutoff)
                else:
                    sigs = strategy.generate_signals(stocks)
                all_signals.extend(sigs)
                strategy_results[strategy.name] = {'count': len(sigs), 'weight': strategy.weight}
            except Exception as e:  # noqa: BLE001
                strategy_results[strategy.name] = {'error': str(e)}

        # P2-Q17-fix(L142): StrategySignal.timestamp 各策略构造时均未赋值, 在此统一填充
        now_iso = datetime.now(CST).isoformat()
        for sig in all_signals:
            if not sig.timestamp:
                sig.timestamp = now_iso

        # 2. 按股票聚合信号
        stock_signals: dict[str, list[StrategySignal]] = {}
        for sig in all_signals:
            if sig.stock not in stock_signals:
                stock_signals[sig.stock] = []
            stock_signals[sig.stock].append(sig)

        # 3. 计算每只股票的融合评分
        allocations = []
        for stock, sigs in stock_signals.items():
            if not sigs:
                continue

            # 加权投票
            # P2-Q17-fix(M139): total_weight 只计"实际产出信号"的策略(排除运行报错/无信号者),
            # 避免稀释 net_score; confidence 改用净权重差占比, 弱单边信号不再获得高置信度。
            active_strategies = [
                s for s in self.strategies
                if s.enabled and strategy_results.get(s.name, {}).get('count', 0) > 0
            ]
            active_weights = {s.name: s.weight for s in active_strategies}
            total_weight = sum(s.weight for s in active_strategies)
            # 加权投票: 用"信号自身置信度 sig.confidence"×产出策略权重;
            # 原实现误用策略对象上的 s.confidence(Strategy 无此属性), 策略一产出信号即 AttributeError。
            buy_weight = sum(active_weights.get(sig.strategy, 0.0) * sig.confidence
                             for sig in sigs if sig.direction == 1)
            sell_weight = sum(active_weights.get(sig.strategy, 0.0) * sig.confidence
                              for sig in sigs if sig.direction == -1)

            net_score = (buy_weight - sell_weight) / max(total_weight, 0.01)
            # V11 审计修复（Medium）: 原实现 confidence = |buy-sell|/total ≡ |net_score|，
            # 分子完全相同导致 confidence_buy 阈值覆盖 net_score_buy（0.15 参数永不生效）。
            # 修正: confidence = 产生信号的策略权重占比（参与度/一致性），
            # 信号参与面越广（多策略共振）confidence 越高，与方向倾向 net_score 正交。
            voting_weight = sum(active_weights.get(sig.strategy, 0.0) for sig in sigs)
            confidence = voting_weight / max(total_weight, 0.01)

            # 如果多方信号占优且置信度够高(P2-Q17-fix(M140): 阈值来自 ENGINE_CONFIG)
            if (net_score > ENGINE_CONFIG['net_score_buy']
                    and confidence > ENGINE_CONFIG['confidence_buy']):
                weight = min(net_score * 0.3, ENGINE_CONFIG['max_single_weight'])  # 单票上限15%
                name = sigs[0].stock_name if sigs[0].stock_name else ""
                allocations.append(StrategyAllocation(
                    stock=stock,
                    stock_name=name,
                    weight=round(weight, 4),
                    confidence=round(confidence, 3),
                    expected_return=round(net_score * ENGINE_CONFIG['expected_return_per_net'], 4),
                    expected_risk=round(0.02, 4),
                    signals=sigs,
                ))

        # 4. 归一化权重（确保总仓位≤1）
        # P2-Q17-fix(M141): 每期归一化即再平衡机制——按当期信号强度把组合暴露
        # 缩放到≤1, 实现文档所述的"组合权重再平衡"。
        total_weight = sum(a.weight for a in allocations)
        if total_weight > 0:
            scale = min(1.0 / total_weight, 1.0)
            for a in allocations:
                a.weight = round(a.weight * scale, 4)

        # 5. 排序
        allocations.sort(key=lambda x: x.confidence * x.weight, reverse=True)

        return {
            'allocations': allocations[:20],
            'meta': {
                'timestamp': datetime.now(CST).isoformat(),
                'release_id': release_id,
                'candidates_scanned': len(stocks),
                'total_signals': len(all_signals),
                'buy_signals': sum(1 for s in all_signals if s.direction == 1),
                'sell_signals': sum(1 for s in all_signals if s.direction == -1),
                'strategies': strategy_results,
                'recommended_positions': len(allocations),
                'total_exposure': round(total_weight * scale if total_weight > 0 else 0, 4),
            }
        }


# ───────── 格式化 ─────────

def format_result(result: dict) -> str:
    """格式化组合运行结果"""
    meta = result.get('meta', {})
    allocs = result.get('allocations', [])

    lines = [f"# 🧩 多策略组合引擎 ({meta.get('timestamp', '')[:16]})"]
    lines.append(f"{'='*50}")

    # 策略状态
    lines.append(f"\n## 策略运行概览")
    lines.append(f"扫描{meta.get('candidates_scanned', 0)}只 | "
                 f"{meta.get('total_signals', 0)}信号 "
                 f"(买{meta.get('buy_signals', 0)}/卖{meta.get('sell_signals', 0)}) | "
                 f"推荐{meta.get('recommended_positions', 0)}仓")
    lines.append(f"\n策略权重:")
    for sname, sinfo in meta.get('strategies', {}).items():
        count = sinfo.get('count', 0)
        weight = sinfo.get('weight', 0)
        lines.append(f"  {sname}: {weight*100:.0f}% → {count}个信号")

    # 推荐持仓
    lines.append(f"\n## 📋 推荐组合\n")
    if allocs:
        lines.append(f"{'股票':<12}{'权重%':>8}{'置信度':>8}{'预期收益':>10}{'信号来源'}")
        lines.append("-" * 60)
        for a in allocs[:15]:
            name = a.stock_name or a.stock[:6]
            signals_str = "+".join(set(s.strategy for s in a.signals))
            lines.append(f"  {name:<10} {a.weight*100:>7.1f}% {a.confidence:>7.0%} "
                         f"{a.expected_return*100:>+8.2f}% {signals_str}")
        lines.append(f"\n总暴露: {meta.get('total_exposure', 0)*100:.1f}%")
    else:
        lines.append("当前无推荐买入信号")

    return "\n".join(lines)


def format_strategies(strategies: list[Strategy]) -> str:
    """格式化已注册策略"""
    lines = ["\n## 📋 已注册策略\n"]
    lines.append(f"{'名称':<16}{'权重':>8}{'状态'}")
    lines.append("-" * 35)
    for s in strategies:
        status = "✅" if s.enabled else "⏸️"
        lines.append(f"  {status} {s.name:<14} {s.weight*100:>6.0f}%")
    return "\n".join(lines)


# ───────── CLI ─────────

def main():
    args = set(sys.argv[1:])

    if "--list" in args:
        engine = StrategyEngine()
        print(format_strategies(engine.strategies))
    elif "--run" in args or not args:
        engine = StrategyEngine()
        result = engine.run()
        print(format_result(result))
    elif "--backtest" in args:
        print("\n## 🔬 组合回测")
        print("⚠️ 回测功能开发中，当前可用单策略回测:")
        print("  python3 -m quant_system.signal_backtest")
    elif "--weights" in args:
        engine = StrategyEngine()
        for s in engine.strategies:
            print(f"  {s.name}: {s.weight*100:.0f}% {'✅' if s.enabled else '⏸️'}")
    elif "--export" in args:
        engine = StrategyEngine()
        result = engine.run()
        out = CONFIG_DIR / "strategy_result.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        print(f"✅ 已导出到 {out}")
    else:
        engine = StrategyEngine()
        result = engine.run()
        print(format_result(result))


if __name__ == "__main__":
    main()
