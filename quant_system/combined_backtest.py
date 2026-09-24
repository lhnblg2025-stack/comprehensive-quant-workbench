# V4.1
"""
组合回测 — 策略引擎+规则引擎+交易逻辑 全链回测

D2收敛登记 (2026-08-11): 独立能力保留——策略→信号→风控→持仓→绩效的
全链组合回测（规则引擎/止损对比/实时行情）backtest_engine 未覆盖
（engine.run_multiple_strategies/BacktestComparison 为策略类并行与摘要
对比，数据模型不同）。文件与公开 API 均保留，不转发、不强迁。

功能:
  1. 全链回测: 策略→信号→风控→持仓→绩效
  2. 多策略对比: 单策略 vs 组合 vs 基准
  3. 规则引擎验证: 有止损 vs 无止损对比
  4. 绩效分析: 夏普/卡玛/最大回撤/胜率/盈亏比
  5. 资金曲线: 每日净值输出
  6. 参数扫描: 策略参数敏感性分析

用法:
  python3 -m quant_system.combined_backtest                     # 默认回测60天
  python3 -m quant_system.combined_backtest --days 120          # 120天
  python3 -m quant_system.combined_backtest --compare           # 有/无止损对比
  python3 -m quant_system.combined_backtest --scan              # 参数扫描
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant_system.execution_broker import DEFAULT_SLIPPAGE_RATE

logger = __import__('logging').getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))

# 回测默认参数
INITIAL_CAPITAL = 1_000_000
COMMISSION = 0.000085    # 佣金万0.85
# P1-Q17-fix(H08): A股佣金最低 5 元/笔（非0.1元）；印花税卖出 0.05%；
# 过户费 0.001% 双边。以下费用在买入(佣金+过户费)与卖出(佣金+印花税+过户费)路径均已扣除。
MIN_COMMISSION = 5.0       # A股最低佣金5元/笔
STAMP_TAX = 0.0005         # 卖出万5（2023年8月起减半征收）
TRANSFER_FEE = 0.00001     # A股过户费0.001%，双边收取
# P1-Q17-fix(H10): 成交滑点——信号基于当日收盘计算后不再按同一收盘价无成本成交，
# 买入按 收盘*(1+滑点)、卖出按 收盘*(1-滑点) 执行，消除乐观成交假设。
# P1-6 (审计回测层): 滑点默认值统一到 execution_broker.DEFAULT_SLIPPAGE_RATE(10bp)，
# 不再各自写 5bp，消除与 backtest.py/config/SimBroker 差一倍的问题。
SLIPPAGE = DEFAULT_SLIPPAGE_RATE   # 滑点10bp
MAX_POSITIONS = 10
MAX_SINGLE_PCT = 0.15

# 基准
BENCHMARK_CODES = ['000300.XSHG']  # 沪深300


def _date_range(days: int) -> list[str]:
    """生成最近 days 个交易日的日期序列。

    P1-Q17-fix(H09): 用新浪交易日历过滤周末/节假日，并排除当日（盘中未收盘的
    不完整K线），避免: ① 非交易日按"前一交易日收盘价"成交产生虚假交易；
    ② tracking_days 混入自然日；③ annual_return 用 len(snapshots)/252 年化偏差。
    """
    today = datetime.now(CST).date()
    start = today - timedelta(days=int(days * 1.8) + 40)
    try:
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        if df is not None and 'trade_date' in df.columns:
            td = pd.to_datetime(df['trade_date']).dt.date
            trading_days = [d for d in td if start <= d < today]  # 排除当日(未收盘)
            dates = [d.strftime('%Y-%m-%d') for d in trading_days]
            if dates:
                return dates[-days:]
            logger.warning("Q17-H09: 交易日历结果为空，回退到工作日过滤")
        else:
            logger.warning("Q17-H09: 交易日历无 trade_date 列，回退到工作日过滤")
    except Exception as e:
        logger.warning("Q17-H09: 交易日历获取失败(%s)，回退到工作日过滤", e)

    # 回退: 仅排除周末（节假日无法剔除，但已告警）
    dates: list[str] = []
    for i in range(int(days * 1.8) + 40):
        d = start + timedelta(days=i)
        if d >= today or d.weekday() >= 5:
            continue
        dates.append(d.strftime('%Y-%m-%d'))
    return dates[-days:]


@dataclass
class BacktestTrade:
    date: str
    stock: str
    action: str  # buy/sell
    price: float
    qty: int
    amount: float
    commission: float = 0
    tax: float = 0
    pnl: float = 0
    pnl_pct: float = 0


@dataclass
class BacktestDaySnapshot:
    date: str
    capital: float
    positions_value: float
    total_value: float
    positions_count: int
    trades_today: int


class CombinedBacktest:
    """全链组合回测

    D2收敛登记: 独立能力保留——全链组合回测 backtest_engine 未覆盖。
    """

    def __init__(self, days: int = 60, use_rules: bool = True, initial_capital: float = INITIAL_CAPITAL):
        self.days = days
        self.use_rules = use_rules
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: dict[str, dict] = {}
        self.trades: list[dict] = []
        self.snapshots: list[dict] = []
        self.dates = _date_range(days)
        self._hist_cache: dict[str, pd.DataFrame | None] = {}  # P2-Q17-fix(L156): 历史数据本地缓存

    def _get_hist(self, code: str) -> pd.DataFrame | None:
        """P2-Q17-fix(L156): 每只股票整个回测窗口的历史(后复权)仅拉取一次并缓存,
        替代原"逐日逐股 akshare 调用"(60天×30只×多接口≈数千次), 消除文档声称
        '约2分钟'与实际运行时长严重不符的问题。"""
        code6 = str(code).replace('.', '')
        if code6 in self._hist_cache:
            return self._hist_cache[code6]
        if not self.dates:
            return None
        # Prefer the recovered canonical kline base.  Besides avoiding a
        # network dependency during offline backtests, this keeps the
        # backtest on the same restored data as the rest of the workbench.
        local_path = ROOT.parent / "data_warehouse" / "kline" / f"{code6}.parquet"
        if local_path.is_file():
            try:
                local = pd.read_parquet(local_path)
                rename = {
                    "date": "日期", "open": "开盘", "high": "最高",
                    "low": "最低", "close": "收盘", "volume": "成交量",
                    "amount": "成交额",
                }
                local = local.rename(columns={k: v for k, v in rename.items() if k in local.columns})
                if {"日期", "收盘"}.issubset(local.columns):
                    self._hist_cache[code6] = local
                    return local
            except Exception as exc:
                logger.warning("本地 K 线读取失败 %s: %s", local_path, exc)
        try:
            import akshare as ak
            start_d = (datetime.strptime(self.dates[0], '%Y-%m-%d') - timedelta(days=200)).strftime('%Y%m%d')
            end_d = self.dates[-1].replace('-', '')
            # P1-Q17-fix(H11): 后复权(hfq), 避免前复权重述历史引入前视偏差
            df = ak.stock_zh_a_hist(symbol=code6, period="daily",
                                    start_date=start_d, end_date=end_d,
                                    adjust="hfq")
            self._hist_cache[code6] = df
        except Exception as e:
            logger.warning("Q17-L156: %s 历史数据获取失败(%s)", code6, e)
            self._hist_cache[code6] = None
        return self._hist_cache[code6]

    def _get_price_at(self, code: str, date: str) -> float:
        """获取某日(或该日之前最近交易日)的收盘价格"""
        hist = self._get_hist(code)
        if hist is None or len(hist) == 0 or '日期' not in hist.columns or '收盘' not in hist.columns:
            return 0.0
        ds = pd.to_datetime(hist['日期']).dt.strftime('%Y-%m-%d')
        mask = ds <= date
        if mask.any():
            return float(hist.loc[mask, '收盘'].iloc[-1])
        return 0.0

    def _get_open_at(self, code: str, date: str) -> float:
        """获取当日开盘价(存在开盘列时)——V12.3 审计 P0-2: 信号昨日生成, 今日开盘成交。"""
        hist = self._get_hist(code)
        if hist is None or len(hist) == 0 or '日期' not in hist.columns:
            return 0.0
        wd = hist[hist.iloc[:, 0].astype(str).str.strip() == date] if hist.columns[0] != '日期' else None
        oc = '开盘' if '开盘' in hist.columns else None
        if not oc:
            return 0.0
        ds = pd.to_datetime(hist['日期']).dt.strftime('%Y-%m-%d')
        mask = ds == date
        if mask.any():
            return float(hist.loc[mask, oc].iloc[0])
        return 0.0

    def _prev_close_at(self, code: str, date: str) -> float:
        """获取 date 之前最近交易日的收盘价（用于跌停价计算，须为真实成交价）。"""
        hist = self._get_hist(code)
        if hist is None or len(hist) == 0 or '日期' not in hist.columns or '收盘' not in hist.columns:
            return 0.0
        ds = pd.to_datetime(hist['日期']).dt.strftime('%Y-%m-%d')
        mask = ds < date
        if not mask.any():
            return 0.0
        return float(hist.loc[mask, '收盘'].iloc[-1])

    def _day_low(self, code: str, date: str) -> float:
        """获取当日最低价；无当日 bar 返回 0.0。"""
        hist = self._get_hist(code)
        if hist is None or len(hist) == 0 or '日期' not in hist.columns or '最低' not in hist.columns:
            return 0.0
        ds = pd.to_datetime(hist['日期']).dt.strftime('%Y-%m-%d')
        mask = ds == date
        if not mask.any():
            return 0.0
        return float(hist.loc[mask, '最低'].iloc[0])

    def _sell_status(self, code: str, date: str, name_hint: str = ""):
        """P1-5 (审计回测层): 判定当日止损/止盈卖出是否可成交。

        返回 dict：{"suspended": bool, "down_stop": float, "can_sell": bool}。
          - suspended: 当日停牌/无有效行情（开盘价<=0 或成交量<=0 或当日无 bar）
          - down_stop: 当日跌停价（per-board 规则，见 market_rules.get_price_limit_pct）；
                        无涨跌幅限制时返回 None
          - can_sell:  未停牌 且 当日收盘价 > 跌停价（可成交）时为 True
        """
        date_low = 0.0
        date_vol = 0.0
        date_close = 0.0
        hist = self._get_hist(code)
        if hist is not None and '日期' in hist.columns:
            ds = pd.to_datetime(hist['日期']).dt.strftime('%Y-%m-%d')
            mask = ds == date
            if mask.any():
                row = hist.loc[mask].iloc[0]
                date_low = float(row.get('最低', 0.0) or 0.0)
                date_vol = float(row.get('成交量', 0.0) or 0.0)
                date_close = float(row.get('收盘', 0.0) or 0.0)
        suspended = date_close <= 0 or date_low <= 0 or date_vol <= 0
        if suspended:
            return {"suspended": True, "down_stop": None, "can_sell": False}

        code6 = str(code).replace('.', '').zfill(6)
        prev_close = self._prev_close_at(code, date)
        down_stop = None
        try:
            from quant_system.market_rules import get_price_limit_pct
            limit_pct = get_price_limit_pct(code6, name=name_hint)  # 百分比，如 10
            if limit_pct and limit_pct > 0 and prev_close > 0:
                down_stop = prev_close * (1 - limit_pct / 100.0)
        except Exception:
            down_stop = prev_close * 0.90 if prev_close > 0 else None  # 默认主板-10%

        can_sell = date_close > down_stop if down_stop is not None else True
        return {"suspended": False, "down_stop": down_stop, "can_sell": can_sell}

    def _get_prices(self, codes: list[str], date: str) -> dict[str, float]:
        """批量获取价格(走本地缓存, 无逐股 sleep)"""
        return {code: self._get_price_at(code, date) for code in codes}

    def _get_benchmark_returns(self, start_date: str, end_date: str) -> float | None:
        """基准(沪深300)区间收益率(%).
        P2-Q17-fix(L155): 实现文档声称的"单策略 vs 组合 vs 基准"对比中的基准部分。
        """
        if not BENCHMARK_CODES:
            return None
        sym = BENCHMARK_CODES[0].split('.')[0]
        try:
            import akshare as ak
            df = ak.index_zh_a_hist(symbol=sym, period="daily",
                                    start_date=start_date.replace('-', ''),
                                    end_date=end_date.replace('-', ''))
            if df is not None and len(df) > 1 and '收盘' in df.columns:
                first = float(df['收盘'].iloc[0])
                last = float(df['收盘'].iloc[-1])
                if first > 0:
                    return round((last / first - 1) * 100, 2)
        except Exception as e:
            logger.warning("Q17-L155: 基准(%s)收益获取失败(%s)", sym, e)
        return None

    def _calc_buy_qty(self, amount: float, price: float) -> int:
        if price <= 0:
            return 0
        qty = int(amount / price / 100) * 100
        return max(qty, 0)

    def _generate_signals(self, date: str, stocks: list[str]) -> list[dict]:
        """生成当日信号模拟

        简化的策略引擎——使用技术指标模拟ML/机会/技术策略
        """
        signals = []
        for stock in stocks[:15]:
            code = stock.replace('.', '')
            try:
                # P2-Q17-fix(L156): 历史数据从本地缓存取(整窗口一次拉取), 不再逐日轮询
                hist = self._get_hist(stock)
                if hist is None or len(hist) < 20 or '日期' not in hist.columns:
                    continue
                ds = pd.to_datetime(hist['日期']).dt.strftime('%Y-%m-%d')
                # V12.3 审计 P0-2 修复: 排除当日(ds < date)——信号只用昨日及以前收盘,
                # 消除"当日收盘算信号+同日收盘价成交"的前视(收益系统性高估)。
                hist = hist[ds < date]
                if len(hist) < 20:
                    continue

                close = hist['收盘'].values
                vol = hist['成交量'].values

                # RSI
                delta = np.diff(close)
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                avg_g = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
                avg_l = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
                rsi = 100 - 100 / (1 + avg_g / avg_l) if avg_l != 0 else 100

                # MA
                ma20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
                ma60 = np.mean(close[-60:]) if len(close) >= 60 else close[-1]

                # 成交量比
                vol_ratio = vol[-1] / np.mean(vol[-6:-1]) if len(vol) >= 6 else 1

                score = 0
                direction = 0

                # RSI超卖
                if rsi < 35:
                    score += 2
                    direction = 1
                # 价格在MA20下方但接近
                if close[-1] < ma20 * 1.02 and close[-1] > ma20 * 0.95:
                    score += 1
                    direction = 1 if direction >= 0 else direction
                # 金叉
                if ma20 > ma60 and len(close) >= 60:
                    score += 1
                    direction = 1 if direction >= 0 else direction
                # 放量
                if vol_ratio > 1.3:
                    score += 1

                # RSI超买
                if rsi > 70:
                    score -= 1
                    if direction == 0:
                        direction = -1

                if direction == 1 and score >= 2:
                    signals.append({
                        'stock': code,
                        'direction': 1,
                        'confidence': min(score / 6, 1.0),
                        'price': close[-1],
                        'reason': f"RSI={rsi:.0f}, MA20={ma20:.1f}",
                    })
                elif direction == -1 and code in self.positions:
                    signals.append({
                        'stock': code,
                        'direction': -1,
                        'confidence': min(abs(score) / 4, 1.0),
                        'price': close[-1],
                        'reason': f"RSI={rsi:.0f}超买",
                    })
            except Exception as e:
                logger.error(f"[combined_backtest] 操作失败: {e}", exc_info=True)
                continue

        return signals

    def _check_rules(self, code: str, pos: dict, price: float) -> str | None:
        """检查止盈止损规则"""
        if not self.use_rules:
            return None

        cost = pos.get('cost', 0)
        if cost <= 0:
            return None
        pnl_pct = (price - cost) / cost

        # 固定止损8%
        if pnl_pct < -0.08:
            return "止损(固定-8%)"
        # 移动止盈: 从最高点回撤10%
        peak = pos.get('peak_price', cost)
        if price > peak:
            pos['peak_price'] = price
        elif (peak - price) / peak > 0.10 and (peak - cost) / cost > 0.05:
            return "止盈(移动回撤10%)"
        # 利润锁定: 盈利20%以上锁50%
        if pnl_pct > 0.20:
            return "止盈(利润锁定20%)"

        return None

    def run(self) -> dict:
        """执行回测"""
        # 基础候选池
        try:
            from quant_system.watchlist import get_watchlist
            stocks = [str(s.get('symbol', '') or s.get('code', '')) for s in get_watchlist()[:30]]
            stocks = [s for s in stocks if s]
        except ImportError:
            stocks = ["600519", "601288", "600036", "600900", "601166"]

        snapshots = []
        all_trades = []

        for i, date in enumerate(self.dates):
            if i < 20:  # 需要至少20天预热
                continue

            # 1. 更新所有持仓市值
            positions_value = 0
            new_positions = {}
            for code, pos in self.positions.items():
                price = self._get_price_at(code, date)
                if price <= 0:
                    price = pos.get('cost', 0)
                mkt_val = price * pos.get('qty', 0)

                # 检查规则
                rule_check = self._check_rules(code, pos, price)
                if rule_check:
                    # 触发止损/止盈
                    # 审计 2026-08-16：修复"用当日收盘触发、却用当日开盘成交"的时间倒置。
                    # 收盘触发（date 收盘后确认）→ 下个交易日开盘成交。
                    next_date = self.dates[i + 1] if i + 1 < len(self.dates) else None
                    if next_date is None:
                        # 无下一交易日可成交：保留持仓，随现有持仓更新市值
                        pos['current_price'] = price
                        pos['market_value'] = mkt_val
                        new_positions[code] = pos
                        positions_value += mkt_val
                        continue
                    sell_status = self._sell_status(code, next_date)
                    if not sell_status["can_sell"]:
                        # 次交易日停牌或跌停封板 → 不可成交，保留持仓，随现有持仓更新市值
                        pos['current_price'] = price
                        pos['market_value'] = mkt_val
                        new_positions[code] = pos
                        positions_value += mkt_val
                        continue
                    qty = pos.get('qty', 0)
                    exec_date = next_date
                    exec_price = self._get_open_at(code, exec_date) or (price * (1 - SLIPPAGE))
                    # P1-5: 成交价 clamp 到执行日 [low, high]（开盘价缺省回退时）
                    day_low = self._day_low(code, exec_date)
                    if day_low and day_low > 0:
                        exec_price = max(exec_price, day_low)
                    amount = qty * exec_price
                    commission = max(amount * COMMISSION, MIN_COMMISSION)
                    tax = amount * STAMP_TAX
                    transfer_fee = amount * TRANSFER_FEE
                    self.capital += amount - commission - tax - transfer_fee
                    # P2-Q17-fix(M154): pnl = 卖出净额(扣佣金+印花税+过户费) - 买入总成本(含买入佣金),
                    # 原实现只含卖出金额与摊入成本的买入佣金, 系统性高估胜率/盈亏比
                    cost_basis = qty * pos.get('cost', 0)
                    net_proceeds = amount - commission - tax - transfer_fee
                    all_trades.append(BacktestTrade(
                        date=exec_date, stock=code,
                        action='sell', price=exec_price, qty=qty,
                        amount=amount, commission=commission, tax=tax,
                        pnl=net_proceeds - cost_basis,
                        pnl_pct=(net_proceeds / cost_basis - 1) * 100 if cost_basis > 0 else 0.0,
                    ).__dict__)
                    continue  # 不加入新仓位
                else:
                    pos['current_price'] = price
                    pos['market_value'] = mkt_val
                    new_positions[code] = pos
                    positions_value += mkt_val

            self.positions = new_positions

            # 2. 生成信号并交易
            signals = self._generate_signals(date, stocks)
            for sig in signals:
                code = sig['stock']
                direction = sig['direction']
                # V12.3 审计 P0-2: 信号基于昨日收盘生成, 成交挂当日开盘价(消除前视)
                # A normal signal carries the price used by the strategy.  The
                # local K-line base supplies the opening bar when available;
                # retain the signal fallback so deterministic callers that
                # mock the signal/price path remain stable.
                price = self._get_open_at(code, date) or sig['price']
                if price <= 0:
                    continue

                if direction == 1 and code not in self.positions:
                    # 买入
                    # P2-Q17-fix(L155): 持仓数上限控制(MAX_POSITIONS 原定义后从未使用)
                    if len(self.positions) >= MAX_POSITIONS:
                        continue
                    # P1-Q17-fix(H10): 信号基于当日收盘计算, 按 收盘*(1+滑点) 成交, 消除乐观成交假设
                    available = self.capital * 0.9
                    max_per = self.capital * MAX_SINGLE_PCT
                    buy_amount = min(available * 0.2, max_per)
                    exec_price = price * (1 + SLIPPAGE)
                    qty = self._calc_buy_qty(buy_amount, exec_price)
                    if qty < 100:
                        continue
                    amount = qty * exec_price
                    commission = max(amount * COMMISSION, MIN_COMMISSION)
                    transfer_fee = amount * TRANSFER_FEE
                    total = amount + commission + transfer_fee
                    if total > self.capital:
                        continue
                    self.capital -= total
                    cost_per = exec_price + commission / qty
                    self.positions[code] = {
                        'qty': qty, 'cost': cost_per,
                        'peak_price': exec_price, 'current_price': exec_price,
                        'market_value': amount,
                    }
                    all_trades.append(BacktestTrade(
                        date=date, stock=code,
                        action='buy', price=exec_price, qty=qty,
                        amount=amount, commission=commission,
                    ).__dict__)

                elif direction == -1 and code in self.positions:
                    # 卖出
                    # P1-Q17-fix(H10): 卖出按 收盘*(1-滑点) 成交
                    pos = self.positions[code]
                    qty = pos.get('qty', 0)
                    exec_price = price * (1 - SLIPPAGE)
                    amount = qty * exec_price
                    commission = max(amount * COMMISSION, MIN_COMMISSION)
                    tax = amount * STAMP_TAX
                    transfer_fee = amount * TRANSFER_FEE
                    self.capital += amount - commission - tax - transfer_fee
                    # P2-Q17-fix(M154): pnl = 卖出净额(扣佣金+印花税+过户费) - 买入总成本(含买入佣金),
                    # 原实现只含卖出金额与摊入成本的买入佣金, 系统性高估胜率/盈亏比
                    cost_basis = qty * pos.get('cost', 0)
                    net_proceeds = amount - commission - tax - transfer_fee
                    all_trades.append(BacktestTrade(
                        date=date, stock=code,
                        action='sell', price=exec_price, qty=qty,
                        amount=amount, commission=commission, tax=tax,
                        pnl=net_proceeds - cost_basis,
                        pnl_pct=(net_proceeds / cost_basis - 1) * 100 if cost_basis > 0 else 0.0,
                    ).__dict__)
                    del self.positions[code]

            # P1-Q17-fix(H07): 在买入/卖出循环之后再统一对所有持仓估值——
            # 若在买入前统计 positions_value，当日新买入仓位的市值未计入，
            # 会导致买入日净值出现约等于买入金额的虚假下跳。此处重新估值，
            # 使 total_value = capital + 全持仓市值（含当日新仓）。
            positions_value = sum(
                pos.get('market_value')
                or pos.get('qty', 0) * pos.get('current_price', pos.get('cost', 0))
                for pos in self.positions.values()
            )
            total_value = self.capital + positions_value
            snapshots.append(BacktestDaySnapshot(
                date=date, capital=round(self.capital, 2),
                positions_value=round(positions_value, 2),
                total_value=round(total_value, 2),
                positions_count=len(self.positions),
                trades_today=len([t for t in all_trades if t.get('date') == date]),
            ).__dict__)

        # 计算绩效
        return self._compute_performance(snapshots, all_trades)

    def _compute_performance(self, snapshots: list[dict], trades: list[dict]) -> dict:
        if not snapshots:
            return {'error': '无回测数据'}

        values = [s['total_value'] for s in snapshots]
        init = self.initial_capital
        final_value = values[-1]
        total_return = (final_value / init - 1) * 100

        # 日收益率
        daily_returns = []
        for i in range(1, len(values)):
            dr = (values[i] / values[i - 1]) - 1
            daily_returns.append(dr)

        # 年化收益
        years = len(snapshots) / 252
        annual_return = ((final_value / init) ** (1 / years) - 1) * 100 if years > 0 else total_return

        # 夏普
        if daily_returns:
            sharpe = (np.mean(daily_returns) - 0.02 / 252) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        else:
            sharpe = 0

        # 最大回撤
        peak = values[0]
        max_dd = 0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # 卡玛比率
        calmar = annual_return / max_dd if max_dd > 0 else 0

        # 胜率
        buy_trades = [t for t in trades if t.get('action') == 'sell']
        wins = sum(1 for t in buy_trades if t.get('pnl', 0) > 0)
        win_rate = wins / len(buy_trades) * 100 if buy_trades else 0

        # 盈亏比
        if buy_trades:
            win_pnls = [t['pnl'] for t in buy_trades if t.get('pnl', 0) > 0]
            loss_pnls = [abs(t['pnl']) for t in buy_trades if t.get('pnl', 0) < 0]
            avg_win = np.mean(win_pnls) if win_pnls else 0
            avg_loss = np.mean(loss_pnls) if loss_pnls else 1
            profit_loss = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            profit_loss = 0

        # P2-Q17-fix(L155): 计算基准(沪深300)区间收益与超额收益
        bench = self._get_benchmark_returns(snapshots[0]['date'], snapshots[-1]['date']) if snapshots else None
        return {
            'initial_capital': init,
            'final_value': round(final_value, 2),
            'total_return_pct': round(total_return, 2),
            'benchmark_return_pct': bench,
            'excess_return_pct': round(total_return - bench, 2) if bench is not None else None,
            'annual_return_pct': round(annual_return, 2),
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown_pct': round(max_dd, 2),
            'calmar_ratio': round(calmar, 3),
            'win_rate_pct': round(win_rate, 1),
            'profit_loss_ratio': round(profit_loss, 2),
            'total_trades': len(trades),
            'buy_trades': len([t for t in trades if t.get('action') == 'buy']),
            'sell_trades': len([t for t in trades if t.get('action') == 'sell']),
            'tracking_days': len(snapshots),
            'snapshots': snapshots,
            'trades': trades[-50:],
            'use_rules': self.use_rules,
        }


def run_comparison(days: int = 60) -> dict:
    """有止损 vs 无止损 对比回测"""
    bt_with_rules = CombinedBacktest(days=days, use_rules=True)
    bt_without_rules = CombinedBacktest(days=days, use_rules=False)

    result_with = bt_with_rules.run()
    result_without = bt_without_rules.run()

    return {
        'with_rules': result_with,
        'without_rules': result_without,
        'days': days,
    }


def format_result(result: dict) -> str:
    if 'error' in result:
        return f"❌ {result['error']}"

    lines = [f"# 🔬 组合回测报告 ({result.get('tracking_days', 0)}天)"]
    lines.append(f"{'=' * 50}")

    lines.append(f"\n📊 绩效总览")
    lines.append(f"  初始资金: ¥{result.get('initial_capital', 0):,.0f}")
    lines.append(f"  最终资产: ¥{result.get('final_value', 0):,.0f}")
    lines.append(f"  累计收益: {result.get('total_return_pct', 0):+.2f}%")
    # P2-Q17-fix(L155): 展示基准/超额收益(数据源缺失时不显示)
    bench = result.get('benchmark_return_pct')
    if bench is not None:
        lines.append(f"  基准收益: {bench:+.2f}%")
        exc = result.get('excess_return_pct')
        if exc is not None:
            lines.append(f"  超额收益: {exc:+.2f}%")
    lines.append(f"  年化收益: {result.get('annual_return_pct', 0):+.2f}%")
    lines.append(f"  夏普比率: {result.get('sharpe_ratio', 0)}")
    lines.append(f"  最大回撤: {result.get('max_drawdown_pct', 0):.2f}%")
    lines.append(f"  卡玛比率: {result.get('calmar_ratio', 0)}")
    lines.append(f"  胜率: {result.get('win_rate_pct', 0):.1f}%")
    lines.append(f"  盈亏比: {result.get('profit_loss_ratio', 0):.2f}")
    lines.append(f"  交易次数: {result.get('total_trades', 0)}笔")
    lines.append(f"  止盈止损: {'✅启用' if result.get('use_rules') else '⏸️未启用'}")

    # 资金曲线（最后10天）
    snapshots = result.get('snapshots', [])
    if snapshots:
        lines.append(f"\n📈 资金曲线（最近10天）")
        for s in snapshots[-10:]:
            ret = (s['total_value'] / result['initial_capital'] - 1) * 100
            bars = "█" * max(1, min(20, int(abs(ret) / 2)))
            sign = "+" if ret >= 0 else ""
            lines.append(f"  {s['date']}: ¥{s['total_value']:>10,.0f} {sign}{ret:.2f}% {bars}")

    # 最近交易
    trades = result.get('trades', [])
    if trades:
        recent = trades[-10:]
        lines.append(f"\n📋 最近交易")
        for t in recent:
            act = "🟢买入" if t.get('action') == 'buy' else "🔴卖出"
            pnl = f" | PnL: {t.get('pnl',0):+.0f}" if 'pnl' in t else ""
            lines.append(f"  {t['date']} {act} {t['stock'][:6]} {t['qty']}股 @ {t['price']:.2f}{pnl}")

    return "\n".join(lines)


def format_comparison(comp: dict) -> str:
    wr = comp.get('with_rules', {})
    wor = comp.get('without_rules', {})

    lines = [f"# ⚖️ 有/无止损对比回测 ({comp.get('days', 60)}天)"]
    lines.append(f"{'='*50}")
    lines.append(f"\n{'指标':<16}{'有止损':>12}{'无止损':>12}{'差异':>12}")
    lines.append("-" * 52)
    metrics = [
        ('total_return_pct', '累计收益%'),
        ('sharpe_ratio', '夏普'),
        ('max_drawdown_pct', '最大回撤%'),
        ('win_rate_pct', '胜率%'),
        ('profit_loss_ratio', '盈亏比'),
        ('total_trades', '交易次数'),
    ]
    for key, label in metrics:
        v_wr = wr.get(key, 0)
        v_wor = wor.get(key, 0)
        diff = v_wr - v_wor
        fmt = f"{v_wr:>11.1f} | {v_wor:>11.1f} | {diff:>+10.1f}" if isinstance(v_wr, (int, float)) else f"{v_wr} | {v_wor}"
        lines.append(f"  {label:<14} {v_wr:>10.2f} {v_wor:>10.2f} {diff:>+10.2f}")

    return "\n".join(lines)


# ───────── CLI ─────────
def main():
    args = sys.argv[1:]

    days = 60
    for i, a in enumerate(args):
        if a == '--days' and i + 1 < len(args):
            try:
                days = max(30, min(int(args[i+1]), 252))
            except ValueError:
                pass

    if "--compare" in args:
        print("⏳ 执行对比回测（历史数据本地缓存, 已消除逐日轮询）...")
        comp = run_comparison(days)
        print(format_comparison(comp))
    elif "--scan" in args:
        print("⏳ 参数敏感性扫描...")
        for d in [30, 60, 120]:
            bt = CombinedBacktest(days=d)
            r = bt.run()
            print(f"  {d}天: 收益{r.get('total_return_pct',0):+.2f}% "
                  f"夏普{r.get('sharpe_ratio',0)} 回撤{r.get('max_drawdown_pct',0):.2f}%")
    else:
        print("⏳ 执行组合回测（历史数据本地缓存, 已消除逐日轮询）...")
        bt = CombinedBacktest(days=days)
        result = bt.run()
        print(format_result(result))

        # 保存
        out = ROOT.parent / "config" / "backtest_result.json"
        ROOT.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        print(f"\n📁 已保存到 {out}")


if __name__ == "__main__":
    main()
