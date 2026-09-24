# -*- coding: utf-8 -*-
"""crawler_hot_rank.py - 东财人气榜每日抓取 (腾讯云 18:35)

2026-08-22 稳定化修复:
  - 路径参数化: 写死的 C:\\quant 改为环境变量可覆盖(默认仓库内), 云端/本地通用
  - 失败可观测: 网络/落盘失败打印明确错误到 stderr, 退出码语义化(1=空数据 2=落盘/初始化失败)
用法:
  python3 scripts/crawler_hot_rank.py
  HOT_RANK_DIR=/path python3 scripts/crawler_hot_rank.py
"""
import sys, io, json, datetime, os, logging
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import requests, pandas as pd
from pathlib import Path
_WS = Path(__file__).resolve().parent.parent
_LOG_FILE = os.environ.get("HOT_RANK_LOG", str(_WS / "generated" / "logs" / "hot_rank.log"))
_DATA_DIR = os.environ.get("HOT_RANK_DIR", str(_WS / "data_warehouse" / "hot_rank"))
try:
    Path(_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(_DATA_DIR).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=_LOG_FILE, level=logging.INFO,
                        format="%(asctime)s %(message)s", force=True)
except Exception as e:  # noqa: BLE001
    print(f"[crawler_hot_rank] 日志/目录初始化失败: {e}", file=sys.stderr)
    sys.exit(2)
HDRS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Referer": "https://guba.eastmoney.com/rank/",
    "Accept": "application/json, text/plain, */*",
}
url = "https://emappdata.eastmoney.com/stockrank/getAllCurrentList"
rows = []
fail_reason = ""
for page in range(1, 12):
    try:
        r = requests.post(url, headers=HDRS, json={"appId":"appId01","globalId":"786e4c21-70dc-435a-93bb-38","marketCode":"","pageNo":page,"pageSize":100}, timeout=15)
        j = r.json()
        data = j.get("data", [])
        if not data: break
        for it in data:
            rows.append({"code": it.get("sc"), "rank": it.get("rk"), "rank_change": it.get("rc"), "hist_rank": it.get("hisRc")})
        if len(data) < 100: break
    except Exception as e:
        fail_reason = f"{type(e).__name__}: {str(e)[:120]}"
        logging.error("page %s FAIL %r", page, e)
        print(f"[crawler_hot_rank] page {page} 失败: {fail_reason}", file=sys.stderr)
        break
if not rows:
    logging.error("hot_rank empty")
    print(f"[crawler_hot_rank] 空数据, 失败原因: {fail_reason or '无数据返回'}", file=sys.stderr)
    sys.exit(1)
today = datetime.date.today().strftime("%Y%m%d")
try:
    df = pd.DataFrame(rows)
    df.to_parquet(str(Path(_DATA_DIR) / f"hot_rank_{today}.parquet"))
    df.to_json(str(Path(_DATA_DIR) / f"hot_rank_{today}.json"), orient="records", force_ascii=False)
    logging.info("hot_rank %s rows=%s saved -> %s", today, len(rows), _DATA_DIR)
    print(f"[crawler_hot_rank] 已保存 {today} rows={len(rows)} -> {_DATA_DIR}")
except Exception as e:  # noqa: BLE001
    print(f"[crawler_hot_rank] 落盘失败: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
    sys.exit(2)
