"""
market_warehouse.py — V7.0 全市场历史数据仓库（分批抓取 + 断点续传）
=====================================================================
目标: 把全市场 ~5500 只 A 股的历史数据（K线/财务/估值）落盘为本地 Parquet，
     回测/因子计算直接读仓库，零网络请求。

设计:
  - 分批抓取: 每批 200 只，失败自动跳过（断点续传），可多次运行续传
  - 数据域: kline（日K qfq）/ financial（新浪财务指标）/ valuation（baostock估值）
  - 落盘: data_warehouse/<domain>/<code>.parquet（按 code 分文件，天然续传）
  - 全链路走 cache.py + usage.py + 熔断（与 data_loader_v7 一致）
  - 进度: data_warehouse/progress.json 记录已成功 code，重跑跳过
  - 回测读取: load_warehouse(domain, codes) → dict[code -> DataFrame]

用法:
  python -m quant_system.market_forecast._support.data.market_warehouse build --domains kline financial valuation --batch 200
  python -m quant_system.market_forecast._support.data.market_warehouse build --domains kline --limit 500   # 先试 500 只
  python -m quant_system.market_forecast._support.data.market_warehouse status
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from quant_system.market_forecast._support.common.cache import DataCache
from quant_system.market_forecast._support.common.logger import get_logger
from quant_system.market_forecast._support.common.usage import get_tracker

# ── 全局兜底超时：akshare 内部 requests.get 不带 timeout，
#    新浪/东财挂起时会无限等待导致并发卡死。这里统一加 20s 超时。──
import requests as _requests
_orig_request = _requests.sessions.Session.request


def _patched_request(self, method, url, **kwargs):
    kwargs.setdefault("timeout", 20)
    return _orig_request(self, method, url, **kwargs)


_requests.sessions.Session.request = _patched_request

log = get_logger("qv6.market_warehouse")

# baostock 全局会话锁（login/logout 不能并发）
_BS_LOCK = threading.Lock()

WAREHOUSE_DIR = Path(__file__).resolve().parents[4] / "data_warehouse"
PROGRESS_FILE = WAREHOUSE_DIR / "progress.json"
DEFAULT_DOMAINS = ["kline", "financial", "valuation"]
# 全局市场域（市场级单份数据，非按 code 分文件）
MARKET_KEYS = [
    "futures_basis", "bond_futures", "qvix", "cb_spot", "repo_rate",
    "shibor", "macro_monthly", "rates", "commodity", "index_pe",
    "market_heat", "sw_industry", "boxoffice",
]
MARKET_DONE_FILE = "market_done.json"  # 记录已抓取成功的市场子键


def get_stock_list() -> list[str]:
    """全市场 A 股代码清单（baostock，剔除退市/指数）。"""
    import baostock as bs
    lg = bs.login()
    try:
        rs = bs.query_stock_basic()
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
    finally:
        bs.logout()
    if df.empty:
        return []
    # type=1 股票 + 未退市（status=1）
    df = df[(df["type"] == "1") & (df["status"] == "1")]
    codes = [c.split(".")[1] for c in df["code"]]
    return sorted(codes)


class MarketWarehouse:
    """全市场数据仓库。"""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root or WAREHOUSE_DIR)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = DataCache()
        self.usage = get_tracker()
        self.progress_file = self.root / "progress.json"
        self.progress: dict[str, list[str]] = self._load_progress()
        # 东财失败快速熔断：连续 3 次 → 后续直接切新浪
        self._em_fails = 0
        # 并发模式：跳过东财直接新浪（东财限流时提速）
        self._skip_em = False

    # ── 通用抓取封装（缓存 + 用量 + 熔断，与 data_loader_v7 一致）──
    def _fetch(self, source: str, key: str, ttl_hours: float, fetcher,
               *args, **kwargs) -> pd.DataFrame:
        if not self.usage.is_allowed(source):
            log.info(f"[{source}] 预算/熔断拦截，走缓存")
            hit = self.cache.get(key)
            return hit if hit is not None else pd.DataFrame()
        try:
            df = self.cache.cached_fetch(key, ttl_hours, fetcher, *args,
                                         source=source, **kwargs)
            self.usage.record(source)
            self.usage.record_success(source)
            return df
        except Exception as e:  # noqa: BLE001
            self.usage.record_failure(source, str(e))
            log.warning(f"[{source}] 抓取失败 {e}")
            hit = self.cache.get(key)
            return hit if hit is not None else pd.DataFrame()

    # ── 进度管理（断点续传）──────────────────────────────
    def _load_progress(self) -> dict[str, list[str]]:
        if self.progress_file.exists():
            try:
                return json.loads(self.progress_file.read_text())
            except Exception:  # noqa: BLE001
                return {}
        return {}

    def _save_progress(self) -> None:
        # 原子写：先写临时文件再 rename，避免并发读半写文件
        tmp = self.progress_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.progress, ensure_ascii=False, indent=1))
        tmp.replace(self.progress_file)

    def _domain_dir(self, domain: str) -> Path:
        d = self.root / domain
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _done_codes(self, domain: str) -> set[str]:
        return set(self.progress.get(domain, []))

    def _mark_done(self, domain: str, code: str) -> None:
        # 并发安全：写前重读合并（多进程各自有内存副本，落盘前合并）
        try:
            if self.progress_file.exists():
                disk = json.loads(self.progress_file.read_text())
                for d in list(disk):
                    self.progress.setdefault(d, [])
                    merged = sorted(set(self.progress[d]) | set(disk[d]))
                    self.progress[d] = merged
        except Exception as e:  # noqa: BLE001
            log.error(f"[market_warehouse] 操作失败: {e}", exc_info=True)
        self.progress.setdefault(domain, [])
        if code not in self.progress[domain]:
            self.progress[domain].append(code)
        # 每 10 只存一次进度（防中断丢进度，并发下也更及时）
        if len(self.progress[domain]) % 10 == 0:
            self._save_progress()

    # ── 抓取单只 ────────────────────────────────────────
    def _fetch_kline(self, code: str) -> pd.DataFrame | None:
        import akshare as ak
        import datetime as _dt
        # V10 审计 M1：end 动态化（原硬编码 20261231，跨年即断）
        start, end = "20200101", _dt.date.today().strftime("%Y%m%d")
        df = None
        # 东财连续失败 → 跳过东财直接新浪（并发模式或已熔断时直接新浪）
        if self._em_fails < 3 and not self._skip_em:
            try:
                key = self.cache.make_key("wh_kline", code, start)
                df = self._fetch("wh_kline_em", key, 24 * 30,
                                 ak.stock_zh_a_hist, symbol=code,
                                 period="daily", start_date=start,
                                 end_date=end, adjust="qfq")
                if df is None or df.empty:
                    self._em_fails += 1
                else:
                    self._em_fails = 0
            except Exception as e:  # noqa: BLE001
                self._em_fails += 1
                log.warning(f"K线东财 {code}: {e}")
        if df is None or df.empty:
            try:
                prefix = "sh" if code.startswith(("6", "9")) else \
                         "bj" if code.startswith(("4", "8")) else "sz"
                key2 = self.cache.make_key("wh_kline_sina", code, start)
                df = self._fetch("wh_kline_sina", key2, 24 * 30,
                                 ak.stock_zh_a_daily,
                                 symbol=f"{prefix}{code}",
                                 start_date=start, end_date=end,
                                 adjust="qfq")
            except Exception as e:  # noqa: BLE001
                log.warning(f"K线新浪 {code}: {e}")
        if df is None or df.empty:
            return None
        df = df.copy()
        # 列名统一为英文
        rename = {"日期": "date", "开盘": "open", "收盘": "close",
                  "最高": "high", "最低": "low", "成交量": "volume",
                  "成交额": "amount", "换手率": "turnover"}
        df = df.rename(columns=rename)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def _fetch_financial(self, code: str) -> pd.DataFrame | None:
        import akshare as ak
        try:
            key = self.cache.make_key("wh_financial", code)
            # 必须带 start_year（不带则返回空表）
            df = self._fetch("wh_financial", key, 24 * 30,
                             ak.stock_financial_analysis_indicator,
                             symbol=code, start_year="2020")
            if df is not None and not df.empty:
                return df
        except Exception as e:  # noqa: BLE001
            log.warning(f"财务新浪 {code}: {e}")
        return None

    def _fetch_valuation(self, code: str) -> pd.DataFrame | None:
        # 主源：东财 stock_value_em（PE/PB/PS/PCF + 市值），4 线程并发
        # 回退：baostock（北交所 bj. 前缀等东财不支持的代码）
        import akshare as ak
        try:
            key = self.cache.make_key("wh_valuation_em", code)
            df = self._fetch("wh_valuation_em", key, 24 * 30,
                             ak.stock_value_em, symbol=code)
            if df is not None and not df.empty:
                df = df.copy()
                rename = {"数据日期": "date", "当日收盘价": "close",
                          "总市值": "total_mv", "流通市值": "float_mv",
                          "PE(TTM)": "pe_ttm", "PE(静)": "pe_lyr",
                          "市净率": "pb", "PEG值": "peg",
                          "市现率": "pcf", "市销率": "ps"}
                df = df.rename(columns=rename)
                keep = [c for c in ["date", "close", "total_mv", "float_mv",
                                    "pe_ttm", "pe_lyr", "pb", "peg",
                                    "pcf", "ps"] if c in df.columns]
                df = df[keep]
                df["date"] = pd.to_datetime(df["date"])
                return df
        except Exception as e:  # noqa: BLE001
            log.warning(f"估值东财 {code}: {e}")
        # 回退 baostock（北交所等）——全局会话，需锁串行
        import baostock as bs
        bscode = ("sh." if code.startswith(("6", "9")) else
                  "bj." if code.startswith(("4", "8")) else "sz.") + code
        try:
            key = self.cache.make_key("wh_valuation", code)
            key_source = "wh_valuation"
            def _bs_fetch() -> pd.DataFrame:
                import datetime as _dt
                with _BS_LOCK:
                    lg = bs.login()
                    try:
                        rs = bs.query_history_k_data_plus(
                            bscode, "date,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                            start_date="2020-01-01",
                            end_date=_dt.date.today().strftime("%Y-%m-%d"),
                            frequency="d", adjustflag="3")
                        rows = []
                        while rs.error_code == "0" and rs.next():
                            rows.append(rs.get_row_data())
                        return pd.DataFrame(rows, columns=rs.fields)
                    finally:
                        bs.logout()
            df = self._fetch(key_source, key, 24 * 30, _bs_fetch)
            if df is not None and not df.empty:
                for c in ["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df["date"] = pd.to_datetime(df["date"])
                return df
        except Exception as e:  # noqa: BLE001
            log.warning(f"估值baostock {code}: {e}")
        return None

    # ── 批处理 ──────────────────────────────────────────
    def build(self, domains: list[str] | None = None, codes: list[str] | None = None,
              batch: int = 200, limit: int | None = None,
              sleep: float = 0.1, workers: int = 4) -> dict[str, int]:
        """分批抓取入库（kline/financial 并发，valuation 串行因 baostock 全局会话）。
        返回 {domain: 新增数}。断点续传：已完成的 code 自动跳过。"""
        domains = domains or DEFAULT_DOMAINS
        all_codes = codes or get_stock_list()
        if limit:
            all_codes = all_codes[:limit]
        self._skip_em = workers > 1  # 并发时跳过东财（限流中）
        stats: dict[str, int] = {}
        for domain in domains:
            done = self._done_codes(domain)
            todo = [c for c in all_codes if c not in done]
            log.info(f"[{domain}] 总 {len(all_codes)} / 已完成 {len(done)} / 待抓 {len(todo)}")
            fetch = getattr(self, f"_fetch_{domain}")
            n_ok, n_fail = 0, 0
            t0 = time.time()
            lock = threading.Lock()

            def _save_one(code: str) -> bool:
                nonlocal n_ok, n_fail
                try:
                    df = fetch(code)
                    if df is not None and not df.empty:
                        df.to_parquet(self._domain_dir(domain) / f"{code}.parquet")
                        with lock:
                            self._mark_done(domain, code)
                            n_ok += 1
                        return True
                    with lock:
                        n_fail += 1
                except Exception as e:  # noqa: BLE001
                    with lock:
                        n_fail += 1
                        if n_fail % 20 == 1:
                            log.warning(f"[{domain}] {code} 失败({n_fail}): {e}")
                return False

            if workers <= 1:
                # 单线程（baostock 回退时也安全）
                for i, code in enumerate(todo):
                    _save_one(code)
                    if sleep:
                        time.sleep(sleep)
                    if (i + 1) % 50 == 0:
                        log.info(f"[{domain}] {i+1}/{len(todo)} 成功{n_ok} 失败{n_fail} "
                                 f"{time.time()-t0:.0f}s")
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(fetch, code): code for code in todo}
                    for i, fut in enumerate(as_completed(futs)):
                        code = futs[fut]
                        try:
                            df = fut.result()
                            if df is not None and not df.empty:
                                df.to_parquet(
                                    self._domain_dir(domain) / f"{code}.parquet")
                                with lock:
                                    self._mark_done(domain, code)
                                    n_ok += 1
                            else:
                                with lock:
                                    n_fail += 1
                        except Exception as e:  # noqa: BLE001
                            with lock:
                                n_fail += 1
                                if n_fail % 20 == 1:
                                    log.warning(
                                        f"[{domain}] {code} 失败({n_fail}): {e}")
                        if (i + 1) % 50 == 0:
                            with lock:
                                log.info(
                                    f"[{domain}] {i+1}/{len(todo)} 成功{n_ok} "
                                    f"失败{n_fail} {time.time()-t0:.0f}s")
            with lock:
                self._save_progress()
            stats[domain] = n_ok
        return stats

    # ── 全局市场域（市场级，单份数据）───────────────────────
    def build_market(self, keys: list[str] | None = None) -> dict[str, str]:
        """抓取全局市场数据（futures_basis/qvix/rates/commodity 等）落盘。
        每 key 一个 parquet；dict 类型的展开为 {key}__{sub}.parquet。
        断点续传：已抓成功的子键跳过（market_done.json）。"""
        from quant_system.market_forecast._support.data.factors.market_data_v7 import MarketDataV7

        md = MarketDataV7()
        keys = keys or MARKET_KEYS
        done_file = self.root / MARKET_DONE_FILE
        done: set[str] = set()
        if done_file.exists():
            try:
                done = set(json.loads(done_file.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                done = set()
        d = self._domain_dir("market")
        d.mkdir(parents=True, exist_ok=True)
        stats: dict[str, str] = {}
        for k in keys:
            if k in done:
                stats[k] = "skip"
                continue
            try:
                v = md.load([k]).get(k)
                if v is None:
                    stats[k] = "empty"
                    continue
                if isinstance(v, dict):
                    n = 0
                    for sub, df in v.items():
                        if df is None or (hasattr(df, "empty") and df.empty):
                            continue
                        df.to_parquet(d / f"{k}__{sub}.parquet")
                        n += 1
                    stats[k] = f"ok({n}子表)" if n else "empty"
                elif hasattr(v, "empty") and not v.empty:
                    v.to_parquet(d / f"{k}.parquet")
                    stats[k] = "ok"
                else:
                    stats[k] = "empty"
                if not stats[k].startswith("empty"):
                    done.add(k)
                    done_file.write_text(
                        json.dumps(sorted(done), ensure_ascii=False, indent=1),
                        encoding="utf-8")
                log.info(f"[market] {k}: {stats[k]}")
            except Exception as e:  # noqa: BLE001
                stats[k] = f"err:{e}"
                log.warning(f"[market] {k} 失败: {e}")
        return stats

    def load_market(self, keys: list[str] | None = None) -> dict:
        """读全局市场域（无网络）。返回与 MarketDataV7.load 同构的 dict。"""
        d = self._domain_dir("market")
        keys = keys or MARKET_KEYS
        out: dict = {}
        for k in keys:
            p = d / f"{k}.parquet"
            if p.exists():
                try:
                    out[k] = pd.read_parquet(p)
                    continue
                except Exception as e:  # noqa: BLE001
                    log.error(f"[market_warehouse] 操作失败: {e}", exc_info=True)
            subs = sorted(d.glob(f"{k}__*.parquet"))
            if subs:
                out[k] = {}
                for sp in subs:
                    try:
                        out[k][sp.stem.split("__", 1)[1]] = pd.read_parquet(sp)
                    except Exception as e:  # noqa: BLE001
                        log.error(f"[market_warehouse] 操作失败: {e}", exc_info=True)
                        continue
        return out

    # ── 个股事件/资金域（margin/lhb/north 等，全市场一张表）────────
    def _event_last_date(self, path: Path, date_col: str) -> pd.Timestamp | None:
        """读事件文件最新日期（用于增量起点）。"""
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            if date_col in df.columns:
                v = pd.to_datetime(df[date_col], errors="coerce").max()
                return v if pd.notna(v) else None
        except Exception as e:  # noqa: BLE001
            log.error(f"[market_warehouse] 操作失败: {e}", exc_info=True)
        return None

    def _recent_trade_days(self, n: int = 10) -> list[str]:
        """最近 n 个交易日（≤今天，Sina 日历含未来日期需过滤）。"""
        import akshare as ak
        try:
            cal = ak.tool_trade_date_hist_sina()
            days = pd.to_datetime(cal["trade_date"])
            today = pd.Timestamp.now().normalize()
            return days[days <= today].dt.strftime("%Y%m%d").tolist()[-n:]
        except Exception:  # noqa: BLE001
            import datetime as _dt
            out = []
            d = _dt.date.today()
            while len(out) < n:
                if d.weekday() < 5:
                    out.append(d.strftime("%Y%m%d"))
                d -= _dt.timedelta(days=1)
            return out

    def _merge_events(self, path: Path, df: pd.DataFrame,
                      dedup_cols: list[str] | None = None,
                      date_col: str | None = None) -> pd.DataFrame:
        """事件文件增量合并（保留历史 + 去重 + 原子写）。幂等：重复跑不产生重复行。
        dedup_cols 缺省按整行去重（龙虎榜同日同股多上榜原因合法）。"""
        if df is None or len(df) == 0:
            return pd.DataFrame()
        if path.exists():
            try:
                old = pd.read_parquet(path)
                if len(old):
                    df = pd.concat([old, df], ignore_index=True)
            except Exception as e:  # noqa: BLE001
                log.error(f"[market_warehouse] 操作失败: {e}", exc_info=True)
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if dedup_cols:
            df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
        else:
            df = df.drop_duplicates().reset_index(drop=True)
        if date_col and date_col in df.columns:
            df = df.sort_values(date_col).reset_index(drop=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
        return df

    def _refresh_event(self, key: str, d: Path) -> pd.DataFrame:
        """增量刷新单个事件 key（保留历史 + 只补最新缺口，幂等）。"""
        import akshare as ak
        import datetime as _dt

        today = _dt.date.today().strftime("%Y%m%d")
        path = d / f"{key}.parquet"
        if key == "north":
            df = ak.stock_hsgt_hist_em(symbol="北向资金")
            return self._merge_events(path, df, dedup_cols=["日期"], date_col="日期")
        if key == "lhb":
            last = self._event_last_date(path, "上榜日")
            start = (last or (pd.Timestamp.now() - pd.Timedelta(days=120))).strftime("%Y%m%d")
            df = ak.stock_lhb_detail_em(start_date=start, end_date=today)
            return self._merge_events(path, df, date_col="上榜日")
        if key == "margin":
            # 旧文件是单日无日期快照（schema 无日期列）→ 无法合并，首次刷新直接重建为多日长表
            if path.exists():
                try:
                    old = pd.read_parquet(path)
                    if "日期" not in old.columns:
                        path.unlink()
                except Exception as e:  # noqa: BLE001
                    log.error(f"[market_warehouse] 操作失败: {e}", exc_info=True)
            frames = []
            for day in self._recent_trade_days(10):
                for fn in (ak.stock_margin_detail_szse, ak.stock_margin_detail_sse):
                    try:
                        sub = fn(date=day)
                        if sub is not None and not sub.empty:
                            sub = sub.copy()
                            sub["日期"] = pd.Timestamp(day)
                            frames.append(sub)
                    except Exception as e:  # noqa: BLE001
                        log.error(f"[market_warehouse] 操作失败: {e}", exc_info=True)
                        continue
            if not frames:
                return pd.DataFrame()
            df = pd.concat(frames, ignore_index=True)
            return self._merge_events(path, df, dedup_cols=["日期", "证券代码"], date_col="日期")
        if key == "block":
            last = self._event_last_date(path, "交易日期")
            start = (last or pd.Timestamp("2020-01-01")).strftime("%Y%m%d")
            df = ak.stock_dzjy_mrmx(symbol="基金", start_date=start, end_date=today)
            return self._merge_events(path, df, date_col="交易日期")
        if key == "etf_flow":
            df = ak.fund_etf_hist_sina(symbol="sh510300")
            return self._merge_events(path, df, dedup_cols=["date"], date_col="date")
        raise KeyError(key)

    def build_events(self, keys: list[str] | None = None,
                     update: bool = False) -> dict[str, str]:
        """抓取个股事件/资金数据（lhb/north/margin/block/pledge/unlock/
        etf/holders/mainflow/industry），每 key 一张 parquet 落盘。
        与 build_market 同构：{key}.parquet 或 {key}__{sub}.parquet。
        update=True 时滚动刷新：保留历史 + 只补最新缺口（幂等，供 cron 每日调用）。"""
        import akshare as ak

        d = self._domain_dir("events")
        d.mkdir(parents=True, exist_ok=True)
        done_file = self.root / "events_done.json"
        done: set[str] = set()
        if done_file.exists():
            try:
                done = set(json.loads(done_file.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                done = set()
        fetchers: dict[str, callable] = {
            "north": lambda: ak.stock_hsgt_hist_em(symbol="北向资金"),
            "lhb": lambda: (lambda s, e: ak.stock_lhb_detail_em(
                start_date=s, end_date=e))(
                (dt.date.today() - dt.timedelta(days=120)).strftime("%Y%m%d"),
                dt.date.today().strftime("%Y%m%d")),
            "margin": lambda: ak.stock_margin_detail_szse(
                date=dt.date.today().strftime("%Y%m%d")),
            "block": lambda: ak.stock_dzjy_mrmx(
                symbol="基金", start_date="20200101",
                end_date=dt.date.today().strftime("%Y%m%d")),
            "etf_flow": lambda: ak.fund_etf_hist_sina(symbol="sh510300"),
        }
        # 以下域为当日快照/逐股慢抓，仅实盘截面用，不参与 IC 回测：
        #   pledge: 逐股 253 只 ~12min（stock_gpzy_pledge_ratio_detail_em）
        #   unlock: 逐股全市场数小时，且 unlock_pressure_30d 为未来窗口（前视）
        #   holders/mainflow: 逐股 500 只 + 东财限流，仅实盘用
        stats: dict[str, str] = {}
        for k in keys or list(fetchers):
            if k in done and not update:
                stats[k] = "skip"
                continue
            fn = fetchers.get(k)
            if fn is None:
                stats[k] = "nokey"
                continue
            try:
                if update:
                    df = self._refresh_event(k, d)
                else:
                    df = self._fetch("wh_event", f"event_{k}", 24, fn)
                if df is not None and not df.empty:
                    if not update:
                        df.to_parquet(d / f"{k}.parquet")
                    done.add(k)
                    done_file.write_text(
                        json.dumps(sorted(done), ensure_ascii=False, indent=1),
                        encoding="utf-8")
                    stats[k] = f"ok({df.shape[0]}行)"
                else:
                    stats[k] = "empty"
                log.info(f"[events] {k}: {stats[k]}")
            except Exception as e:  # noqa: BLE001
                stats[k] = f"err:{str(e)[:60]}"
                log.warning(f"[events] {k} 失败: {e}")
        # holders / mainflow 逐股抓（慢，取前 500 只代表性样本）
        for k in (keys or list(fetchers)):
            if k not in ("holders", "mainflow"):
                continue
            if k in done:
                continue
            out = []
            codes = get_stock_list()[:500]
            for i, c in enumerate(codes):
                try:
                    if k == "holders":
                        df = ak.stock_zh_a_gdhs_detail_em(symbol=c)
                    else:
                        mkt = "sh" if c.startswith("6") else "sz"
                        df = ak.stock_individual_fund_flow(stock=c, market=mkt)
                    if df is not None and not df.empty:
                        df = df.copy()
                        df["code"] = c
                        out.append(df)
                except Exception as e:  # noqa: BLE001
                    log.error(f"[market_warehouse] 操作失败: {e}", exc_info=True)
                    continue
                if (i + 1) % 50 == 0:
                    log.info(f"[events] {k} {i+1}/{len(codes)} 成功{len(out)}")
            if out:
                pd.concat(out, ignore_index=True).to_parquet(d / f"{k}.parquet")
                done.add(k)
                done_file.write_text(
                    json.dumps(sorted(done), ensure_ascii=False, indent=1),
                    encoding="utf-8")
                stats[k] = f"ok({len(out)}只)"
            else:
                stats[k] = "empty"
            log.info(f"[events] {k}: {stats[k]}")
        return stats

    def load_events(self, keys: list[str] | None = None) -> dict:
        """读事件域（无网络）。返回 {key: DataFrame}。"""
        d = self._domain_dir("events")
        out: dict = {}
        for k in (keys or ["north", "lhb", "margin", "block",
                           "etf_flow"]):
            p = d / f"{k}.parquet"
            if p.exists():
                try:
                    out[k] = pd.read_parquet(p)
                except Exception as e:  # noqa: BLE001
                    log.error(f"[market_warehouse] 操作失败: {e}", exc_info=True)
                    continue
        return out

    # ── 状态 ────────────────────────────────────────────
    def status(self) -> dict:
        out = {}
        for domain in DEFAULT_DOMAINS + ["market", "events"]:
            d = self._domain_dir(domain)
            n = len(list(d.glob("*.parquet")))
            mb = sum(f.stat().st_size for f in d.glob("*.parquet")) / 1e6
            out[domain] = {"files": n, "mb": round(mb, 1)}
        return out

    # ── 读取 ────────────────────────────────────────────
    def load_warehouse(self, domain: str, codes: list[str]) -> dict[str, pd.DataFrame]:
        """读仓库（无网络）。codes 缺省读全部已入库。"""
        d = self._domain_dir(domain)
        out: dict[str, pd.DataFrame] = {}
        for code in codes:
            p = d / f"{code}.parquet"
            if p.exists():
                try:
                    out[code] = pd.read_parquet(p)
                except Exception as e:  # noqa: BLE001
                    log.error(f"[market_warehouse] 操作失败: {e}", exc_info=True)
                    continue
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="全市场历史数据仓库")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="分批抓取入库")
    b.add_argument("--domains", default=",".join(DEFAULT_DOMAINS),
                   help="kline,financial,valuation")
    b.add_argument("--batch", type=int, default=200)
    b.add_argument("--limit", type=int, default=None, help="只抓前 N 只（测试用）")
    b.add_argument("--sleep", type=float, default=0.1, help="每只间隔秒（东财限流）")
    sub.add_parser("status", help="查看进度")
    sub.add_parser("list", help="获取全市场代码数")
    bm = sub.add_parser("market", help="抓取全局市场数据（一次全市场共享）")
    bm.add_argument("--keys", default="", help="逗号分隔，缺省全部")
    be = sub.add_parser("events", help="抓取个股事件/资金数据（全市场一张表）")
    be.add_argument("--keys", default="", help="逗号分隔，缺省全部")
    be.add_argument("--update", action="store_true",
                    help="滚动增量刷新：保留历史 + 只补最新缺口（幂等，供 cron 每日调用）")

    args = ap.parse_args()
    wh = MarketWarehouse()
    if args.cmd == "status":
        st = wh.status()
        print("仓库状态:")
        for k, v in st.items():
            print(f"  {k}: {v['files']} 文件 {v['mb']} MB")
        total = sum(v["files"] for v in st.values())
        print(f"  合计: {total} 文件")
    elif args.cmd == "list":
        codes = get_stock_list()
        print(f"全市场股票数: {len(codes)}")
    elif args.cmd == "market":
        keys = [x.strip() for x in args.keys.split(",") if x.strip()] or None
        stats = wh.build_market(keys=keys)
        print("市场数据抓取:", stats)
    elif args.cmd == "events":
        keys = [x.strip() for x in args.keys.split(",") if x.strip()] or None
        stats = wh.build_events(keys=keys, update=args.update)
        print("事件数据抓取:", stats)
    elif args.cmd == "build":
        domains = [x.strip() for x in args.domains.split(",") if x.strip()]
        stats = wh.build(domains=domains, batch=args.batch,
                         limit=args.limit, sleep=args.sleep)
        print("抓取完成:", stats)
        st = wh.status()
        for k, v in st.items():
            print(f"  {k}: {v['files']} 文件 {v['mb']} MB")


if __name__ == "__main__":
    main()
