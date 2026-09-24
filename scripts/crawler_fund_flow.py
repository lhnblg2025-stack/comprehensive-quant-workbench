# -*- coding: utf-8 -*-
"""crawler_fund_flow.py - 东财个股资金流每日抓取 (腾讯云 国内IP 固定通道, 18:45)

本机(境外IP)直连 push2delay.eastmoney.com 被断连；本脚本部署在腾讯云
(国内IP, C:\\quant\\crawler_fund_flow.py) 由计划任务 crawler_fund_flow 触发，
把当日个股资金流落盘为 data/fund_flow/{YYYYMMDD}.parquet，
再由本机 pull_cloud_data.py --only fund_flow 增量回传。

数据契约（与 quant_system.analysis_core.fund_flow_divergence 对齐）:
  东财 fflow/kline/get (klt=101 日线, lmt=0 全量)
  klines 每行: date,主力净额,小单,中单,大单,超大单,主力净占比,小单占比,
               中单占比,大单占比,超大单占比,收盘价,涨跌幅
  单位=元（主力/超大单净额列）；比例列为 %。
  取每只股票最后一行（最近交易日）= 当日主力/超大单净额。

池子:
  - watchlist.json 自选 (永远包含)
  - 当日涨停池成分 (zt_pool_em_daily.parquet 当日) → 活跃资金关注
  - 北向/热门榜 top (hot_rank_*.parquet 当日) → 市场焦点
  去重后 ≤ 400 只, 单只 <2s, 总耗时 <10min

输出: C:\\quant\\data\\fund_flow\\fund_flow_YYYYMMDD.parquet (schema 见下)
  columns: code, date(交易日), main_net(元), super_net(元), close, pct_chg
"""
from __future__ import annotations  # 2026-08-22 兼容: PEP604 注解在 py3.9 需此
import sys, io, json, datetime, os, logging, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import requests, pandas as pd
from pathlib import Path

# 2026-08-22 稳定化: 路径参数化(环境变量可覆盖), 默认仓库内; 云端/本机通用
_WS = Path(__file__).resolve().parent.parent
_LOGF = os.environ.get("FUND_FLOW_LOG", str(_WS / "generated" / "logs" / "fund_flow.log"))
OUT_DIR = os.environ.get("FUND_FLOW_DIR", str(_WS / "data_warehouse" / "fund_flow"))
Path(_LOGF).parent.mkdir(parents=True, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
try:
    logging.basicConfig(filename=_LOGF, level=logging.INFO,
                        format="%(asctime)s %(message)s", force=True)
except Exception as e:  # noqa: BLE001
    print(f"[crawler_fund_flow] 日志初始化失败: {e}", file=sys.stderr)

FFLOW_URL = ("https://push2delay.eastmoney.com/api/qt/stock/fflow/kline/get"
             "?secid={secid}&fields1=f1,f2,f3,f7"
             "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"
             "&klt=101&lmt=0&ut=b2884a393a59ad64002292a3e90d46a5")
HDRS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Referer": "https://data.eastmoney.com/",
    "Accept": "application/json, text/plain, */*",
}


def normalize(code: str) -> str | None:
    """6位代码 → 东财 secid: 6/9开头=沪(1.), 0/3开头=深(0.), 4/8/92北交所跳过."""
    c = str(code).strip().upper().replace(".", "")
    if c.startswith(("SH", "SZ", "BJ")):
        c = c[2:]
    if not c.isdigit() or len(c) != 6:
        return None
    if c.startswith(("4", "8", "92")):
        return None
    if c.startswith(("6", "9")):
        return f"1.{c}"
    return f"0.{c}"


def fetch_one(code: str, timeout: float = 10.0) -> dict | None:
    secid = normalize(code)
    if secid is None:
        return {"code": code, "skip": True, "note": "北交所/非法代码"}
    try:
        url = FFLOW_URL.format(secid=secid)
        r = requests.get(url, headers=HDRS, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        klines = ((j.get("data") or {}).get("klines")) or []
        if not klines:
            return {"code": code, "skip": True, "note": "接口无数据"}
        last = klines[-1]
        parts = last.split(",")
        # 实测接口 klines 行仅 6 字段: date,主力净额,小单,中单,大单,超大单 (与
        # fund_flow_divergence IDX_MAIN=1 / IDX_SUPER=5 对齐); 收盘/涨跌幅由本地 kline 补。
        if len(parts) < 6:
            return {"code": code, "skip": True, "note": "行字段不足"}
        return {
            "code": code,
            "date": parts[0],              # f51 日期
            "main_net": float(parts[1]),   # f52 主力净额(元)
            "super_net": float(parts[5]),  # f56 超大单净额(元)
        }
    except Exception as e:
        logging.error("fetch %s FAIL %r", code, e)
        return {"code": code, "skip": True, "note": str(e)[:60]}


def build_pool() -> list[str]:
    """池子: watchlist + 当日涨停池 + 当日人气榜, 去重."""
    codes: list[str] = []
    try:
        wl = json.load(open(_WS / "config" / "watchlist.json", encoding="utf-8"))
        codes += [str(c).zfill(6) for c in wl.get("watch", [])]
    except Exception as e:
        logging.warning("watchlist 读取失败: %r", e)
    try:
        zp = _WS / "data_warehouse" / "market" / "zt_pool_em_daily.parquet"
        if os.path.exists(zp):
            try:
                df = pd.read_parquet(zp)
            except Exception as pe:
                logging.warning("zt_pool_em_daily 解析失败(可能损坏), 跳过: %r", pe)
                df = pd.DataFrame()
            if not df.empty and "code" in df.columns and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                last = df["date"].max()
                d0 = df[df["date"] == last]
                codes += [str(c).zfill(6) for c in d0["code"].tolist()]
    except Exception as e:
        logging.warning("涨停池读取失败: %r", e)
    try:
        import glob
        hr = sorted(glob.glob(str(_WS / "data_warehouse" / "hot_rank" / "hot_rank_*.parquet")))
        if hr:
            hdf = pd.read_parquet(hr[-1])
            if "code" in hdf.columns:
                codes += [str(c).zfill(6) for c in hdf["code"].head(60).tolist()]
    except Exception as e:
        logging.warning("人气榜读取失败: %r", e)
    # 去重保序
    seen = set()
    out = []
    for c in codes:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out[:400]


def main() -> None:
    t0 = time.time()
    pool = build_pool()
    if not pool:
        logging.error("池子为空, 退出")
        sys.exit(1)
    rows: list[dict] = []
    ok = 0
    for i, code in enumerate(pool, 1):
        r = fetch_one(code)
        if r and not r.get("skip"):
            rows.append(r)
            ok += 1
        if i % 50 == 0:
            logging.info("progress %d/%d ok=%d", i, len(pool), ok)
    if not rows:
        logging.error("全部抓取失败")
        sys.exit(1)
    df = pd.DataFrame(rows)
    today = datetime.date.today().strftime("%Y%m%d")
    path = os.path.join(OUT_DIR, f"fund_flow_{today}.parquet")
    df.to_parquet(path, index=False)
    logging.info("fund_flow %s saved rows=%d ok=%d elapsed=%.1fs", today, len(rows), ok,
                 time.time() - t0)
    print(f"fund_flow {today} rows={len(rows)} ok={ok} elapsed={time.time()-t0:.1f}s -> {path}")


if __name__ == "__main__":
    main()