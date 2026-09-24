"""
data_sources — 统一数据源注册表（V11 底层重构）

原则（用户硬规则 2026-08-09）:
  - 限额/低速/无权限的源一律不启用（tushare 财务接口无权限→禁用；iFinD 无账号→禁用）
  - 只保留: 免费 + 快 + 稳定的源，多源互为降级
  - 每个源: availability 自检 + fetch 封装 + 滚动落盘

源注册表:
  S1 东财   akshare(已用主力)     行情/涨停池/龙虎榜/两融/公告情绪
  S2 同花顺 akshare THS 46接口    概念375/行业90/财务摘要1200行/排名
  S3 问财   pywencai              自然语言选股（免费无限额）
  S4 通达信 pytdx                 本地行情源（备用/分钟线）
  S5 巨潮   cninfo(akshare)       公告/财报官方源
  S6 baostock                     免费K线/财务（降级备用）

用法:
  python3 -m quant_system.analysis_core.data_sources --check   # 全源自检
  python3 -m quant_system.analysis_core.data_sources --ths-concepts   # 抓同花顺375概念成分
  python3 -m quant_system.analysis_core.data_sources --ths-members --limit 10   # 增量抓同花顺概念成分（断点续传）
  python3 -m quant_system.analysis_core.data_sources --ths-members --refresh    # 强制重抓已完成的板块
"""

from __future__ import annotations
import logging

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.config import MARKET_DIR  # noqa: E402

try:
    from quant_system.rate_limiter import get_limiter  # noqa: E402
except ImportError:  # rate_limiter 缺失时只降级为不限流，不阻断
    get_limiter = None

CST = timezone(timedelta(hours=8))

# 源分类（用户 2026-08-09 硬规则：删的是不可用 API，不删数据源分类；独特源即使慢/限额也保留）
SOURCE_STATUS = {
    "eastmoney": {"desc": "东财(akshare)", "status": "active", "scope": ["global"],
                  "note": "主力源，本机直连正常"},
    "ths": {"desc": "同花顺(akshare THS 46接口)", "status": "active", "scope": ["global"],
            "note": "概念375/行业90/财务摘要1200行，已验证"},
    "wencai": {"desc": "问财(pywencai)", "status": "unique", "scope": ["global"],
              "note": "自然语言选股；境外IP被风控→建议腾讯云跑"},
    "pytdx": {"desc": "通达信(pytdx)", "status": "unique", "scope": ["global"],
             "note": "分钟线/盘口独特用途；境外连不上→建议腾讯云跑"},
    "cninfo": {"desc": "巨潮(akshare)", "status": "unique", "scope": ["global"],
              "note": "官方公告/财报权威源；境外连不上→建议腾讯云跑"},
    "baostock": {"desc": "Baostock", "status": "active", "scope": ["global"],
                "note": "免费降级备用，已验证"},
    "tushare": {"desc": "Tushare", "status": "partial", "scope": ["global"],
                "note": "日线行情可用；财务接口无权限→不接入API层", "api_enabled": False},
    "ifind": {"desc": "同花顺iFinD", "status": "classified", "scope": ["global"],
             "note": "无账号（境外），仅保留分类待日后开通", "api_enabled": False},
    "hot_rank": {"desc": "东财热榜云快照(腾讯云18:35)", "status": "active", "scope": ["global"],
                 "note": "data_warehouse/hot_rank/hot_rank_*.parquet（top100）"},
    "social": {"desc": "社交快照(Vultr 18:40)", "status": "active", "scope": ["global"],
               "note": "data_warehouse/social/weibo_*.parquet + baidu_hot_*.json"},
    "cninfo_cloud": {"desc": "巨潮公告云快照(腾讯云18:30)", "status": "active", "scope": ["global"],
                     "note": "data_warehouse/cninfo/cninfo_*.json"},
    "ths_concept": {"desc": "同花顺概念成分云快照", "status": "active", "scope": ["global"],
                    "note": "data_warehouse/classification/concept_member_ths.parquet"},
    "industry_sw": {"desc": "申万行业", "status": "active", "scope": ["global"],
                    "note": "data_warehouse/industry/sw_first*.parquet"},
    "industry_csi": {"desc": "中证行业", "status": "active", "scope": ["global"],
                     "note": "data_warehouse/industry/csi_industry*.parquet"},
    "industry_car": {"desc": "汽车销量", "status": "active", "scope": ["chain_auto"],
                     "note": "data_warehouse/industry/car_*.parquet"},
}


def _cloud_snapshot_status(data_dir: Path, glob_pattern: str, fresh_days: int = 3) -> dict:
    """云快照自检（不联网）。

    as_of 优先取文件名 YYYYMMDD，其次 parquet 的 date/日期 列最大值，最后退化为文件 mtime。
    返回 {"ok": bool, "note": str, "as_of": str|None, "n_files": int}。
    """
    if not data_dir.exists():
        return {"ok": False, "note": f"目录缺失 {data_dir}", "as_of": None, "n_files": 0}
    files = sorted(data_dir.glob(glob_pattern))
    if not files:
        return {"ok": False, "note": f"无匹配文件 {data_dir / glob_pattern}", "as_of": None, "n_files": 0}
    latest = files[-1]
    as_of: str | None = None
    as_of_src = "file"
    m = re.search(r"(\d{8})", latest.name)
    if m:
        as_of = m.group(1)
    elif latest.suffix == ".parquet":
        for date_col in ("date", "日期"):
            try:
                df = pd.read_parquet(latest, columns=[date_col])
                v = pd.to_datetime(df[date_col], errors="coerce").dropna()
                if len(v):
                    as_of = v.max().strftime("%Y%m%d")
                    as_of_src = "date_col"
                    break
            except Exception:  # noqa: BLE001
                # 该表无 date/日期 列（如 concept_member，只有 concept/code/name）→ 静默降级 mtime
                continue
    if as_of is None:
        try:
            as_of = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y%m%d")
            as_of_src = "mtime"
        except OSError:
            as_of = None
    if as_of is None:
        return {"ok": False, "note": f"{latest.name} 无法解析 as_of", "as_of": None, "n_files": len(files)}
    age_days = (datetime.now(CST) - datetime.strptime(as_of, "%Y%m%d").replace(tzinfo=CST)).days
    ok = age_days <= fresh_days
    note = f"{latest.name} as_of={as_of}({as_of_src}) 共{len(files)}个文件"
    note += "，新鲜" if ok else f"，陈旧{age_days}日(阈值{fresh_days})"
    return {"ok": ok, "note": note, "as_of": as_of, "n_files": len(files)}


def _cloud_snapshots_status() -> dict:
    """全部云快照自检（不联网）。"""
    dw = ROOT / "data_warehouse"
    checks = {
        "hot_rank": _cloud_snapshot_status(dw / "hot_rank", "hot_rank_*.parquet"),
        "social": _cloud_snapshot_status(dw / "social", "weibo_*.parquet"),
        "cninfo_cloud": _cloud_snapshot_status(dw / "cninfo", "cninfo_*.json"),
        "ths_concept": _cloud_snapshot_status(dw / "classification", "concept_member_ths.parquet"),
        "industry_sw": _cloud_snapshot_status(dw / "industry", "sw_first*.parquet"),
        "industry_csi": _cloud_snapshot_status(dw / "industry", "csi_industry*.parquet"),
        "industry_car": _cloud_snapshot_status(dw / "industry", "car_*.parquet"),
    }
    # social 补充 baidu 快照（weibo+baidu 任一新鲜即算 ok）。
    # 2026-08-29 修复: 本机 social_realtime 实际落盘 guba_*/bilibili_*/baidu_*/rank_*，
    # 旧 weibo_* 已停更到 08-11；按全部 social 快照取最新，避免误报陈旧。
    social_snapshots = []
    for pattern in ("weibo_*.parquet", "guba_*.parquet", "bilibili_*.parquet",
                    "baidu_*.parquet", "rank_*.parquet", "baidu_hot_*.json"):
        social_snapshots.append(_cloud_snapshot_status(dw / "social", pattern))
    newest = max(social_snapshots, key=lambda s: (s.get("as_of") or "0"))
    checks["social"] = {
        "ok": any(s["ok"] for s in social_snapshots),
        "note": " | ".join(f"{s['note']}" for s in social_snapshots if s.get("as_of")),
        "as_of": newest["as_of"],
        "n_files": sum(s["n_files"] for s in social_snapshots),
    }
    return checks

def check_availability(verbose: bool = True) -> dict:
    """全源自检。返回 {源: {ok, note}}。"""
    res = {}
    import akshare as ak

    # S1 东财
    try:
        df = ak.stock_zt_pool_em(date=datetime.now(CST).strftime("%Y%m%d"))
        res["eastmoney"] = {"ok": True, "note": f"涨停池{len(df)}行"}
    except Exception as e:
        res["eastmoney"] = {"ok": False, "note": str(e)[:60]}

    # S2 同花顺
    try:
        df = ak.stock_board_concept_name_ths()
        res["ths"] = {"ok": True, "note": f"概念板块{len(df)}个"}
    except Exception as e:
        res["ths"] = {"ok": False, "note": str(e)[:60]}

    # S3 问财（独特用途：自然语言选股）
    try:
        import pywencai
        r = pywencai.get(query="今天涨停的股票", loop=True)
        n = len(r) if r is not None else 0
        res["wencai"] = {"ok": n > 0, "note": f"返回{n}条"}
    except Exception as e:
        res["wencai"] = {"ok": False, "note": str(e)[:60]}

    # S4 通达信（独特用途：分钟线/盘口，免费无限额，稍慢可接受）
    try:
        from pytdx.hq import TdxHq_API
        api = TdxHq_API()
        api.connect('119.147.212.81', 7709, time_out=5)
        n = api.get_security_count(1)
        api.disconnect()
        res["pytdx"] = {"ok": True, "note": f"深市证券数{n}"}
    except Exception as e:
        res["pytdx"] = {"ok": False, "note": str(e)[:60]}

    # S5 巨潮（独特用途：官方公告/财报，权威源）
    try:
        df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol="最新", market="沪深",
            start_date=(datetime.now(CST) - timedelta(days=3)).strftime("%Y%m%d"),
            end_date=datetime.now(CST).strftime("%Y%m%d"))
        res["cninfo"] = {"ok": True, "note": f"公告{len(df)}条"}
    except Exception as e:
        res["cninfo"] = {"ok": False, "note": str(e)[:60]}

    # S6 baostock
    try:
        import baostock as bs
        lg = bs.login()
        res["baostock"] = {"ok": lg.error_code == "0", "note": lg.error_msg}
        bs.logout()
    except Exception as e:
        res["baostock"] = {"ok": False, "note": str(e)[:60]}

    # 分类状态（active/unique/partial/classified）
    for k, info in SOURCE_STATUS.items():
        if k not in res:
            res[k] = {"ok": info.get("api_enabled", True), "note": info["note"]}

    # ── 云快照自检（不联网，每源一条 note）────────────────────
    for k, st in _cloud_snapshots_status().items():
        res[k] = {"ok": st["ok"], "note": st["note"]}

    if verbose:
        icons = {"active": "✅", "unique": "🔶", "partial": "⚠️", "classified": "📋"}
        for k, v in res.items():
            st = SOURCE_STATUS.get(k, {}).get("status", "active")
            icon = icons.get(st, "❌")
            okmark = "可用" if v["ok"] else "不可用"
            print(f"{icon} {k:<10} [{st}] {okmark} | {v['note']}")
    return res


# ── 同花顺概念板块成分（升级 theme_cycle / 传导图用）────────
# 2026-08-10 审计：akshare 本版本无 stock_board_concept_cons_ths，
# 直连同花顺 10jqka 板块成分页抓取（分页 HTML），增量合并落盘，幂等。
THS_MEMBER_FILE = MARKET_DIR / "concept_member_ths.parquet"
THS_MEMBER_DONE = MARKET_DIR / "concept_member_ths_done.json"
THS_MEMBER_ALT = ROOT / "data_warehouse" / "classification" / "concept_member_ths.parquet"
THS_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Referer": "https://q.10jqka.com.cn/",
}


def _ths_page_text(url: str, timeout: int = 15) -> str:
    """拉取 10jqka 页面文本（utf-8 优先，GBK 兜底）。"""
    import requests as _req
    if get_limiter is not None:
        get_limiter("10jqka.com.cn").wait()
    r = _req.get(url, headers=THS_HEADERS, timeout=timeout)
    r.raise_for_status()
    try:
        return r.text
    except Exception:  # noqa: BLE001
        return r.content.decode("gbk", errors="ignore")


def _ths_parse_members(text: str) -> list[tuple[str, str]]:
    """从成分页 HTML/JSON 中解析 (code, name)。

    2026-08-11 修复: 页面实际结构是 <a href="http://stockpage.10jqka.com.cn/002298/"
    （旧正则 /stock/code/XXX.html 解析 0 行，导致 THS 成分从未抓到）。
    兼容两种结构 + 名称从相邻 td/span 提取（取代码即可，名称可后续回填）。
    """
    import re as _re
    html = text
    if text.lstrip().startswith("{"):
        try:
            import json as _json
            payload = _json.loads(text)
            html = payload.get("data") or payload.get("result") or ""
        except Exception:  # noqa: BLE001
            html = text
    # 主结构: http://stockpage.10jqka.com.cn/002298/
    codes = _re.findall(r'stockpage\.10jqka\.com\.cn/(\d{6})/', html)
    # 兜底: /stock/code/002298.html
    if not codes:
        codes = _re.findall(r'/stock/code/(\d{6})\.html', html)
    return [(c, "") for c in codes]


def fetch_ths_concept_members(limit: int = 0, refresh: bool = False,
                              save: bool = True) -> pd.DataFrame:
    """增量抓同花顺概念成分（断点续传 + 合并去重，幂等）。
    输出 market/concept_member_ths.parquet（short_term_extra 读取路径）
    并同步 data_warehouse/classification/concept_member_ths.parquet。
    limit>0 只抓前 N 个板块（调试）；refresh=True 强制重抓已完成板块。"""
    import json as _json
    import time as _t
    import akshare as ak

    boards = ak.stock_board_concept_name_ths()  # name, code
    if boards is None or boards.empty:
        print("[ths-members] 板块列表获取失败，跳过", flush=True)
        return pd.DataFrame()
    done: dict = {}
    if THS_MEMBER_DONE.exists():
        try:
            done = _json.loads(THS_MEMBER_DONE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            done = {}
    targets = []
    for _, r in boards.iterrows():
        name, code = str(r["name"]).strip(), str(r["code"]).strip()
        if not code:
            continue
        if code in done and not refresh:
            continue
        targets.append((name, code))
        if limit > 0 and len(targets) >= limit:
            break
    print(f"[ths-members] 待抓板块 {len(targets)}（已完成 {len(done)}）", flush=True)
    rows: list[dict] = []
    fails: list[str] = []
    for i, (name, code) in enumerate(targets, 1):
        page = 1
        got = 0
        while page <= 200:
            # 2026-08-11 修复: 实测可用 URL 为 /gn/detail/code/{code}/page/{n}/
            # （旧带 field/addtime/order/desc/ajax/1 的 URL 返回 401/403 需登录）
            url = f"https://q.10jqka.com.cn/gn/detail/code/{code}/page/{page}/"
            try:
                text = _ths_page_text(url)
                pairs = _ths_parse_members(text)
            except Exception as e:  # noqa: BLE001
                fails.append(f"{name}:{type(e).__name__}")
                break
            if not pairs:
                break
            seen: set[str] = set()
            for c, n in pairs:
                if c in seen:
                    continue
                seen.add(c)
                rows.append({"concept": name, "code": c, "name": n})
            got += len(seen)
            page += 1
            _t.sleep(0.3)
        if got:
            done[code] = True
            THS_MEMBER_DONE.write_text(
                _json.dumps(done, ensure_ascii=False), encoding="utf-8")
        if i % 20 == 0 or i == len(targets):
            print(f"  [ths-members] {i}/{len(targets)} 累计 {len(rows)} 行 失败 {len(fails)}",
                  flush=True)
    if fails:
        print(f"  ⚠️ 失败板块 {len(fails)}（下次重跑自动补）: {fails[:5]}", flush=True)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["code"] = df["code"].astype(str).str.zfill(6)
    if save:
        for path in (THS_MEMBER_FILE, THS_MEMBER_ALT):
            old = None
            if path.exists():
                try:
                    old = pd.read_parquet(path)
                except Exception:  # noqa: BLE001
                    old = None
            out = pd.concat([old, df], ignore_index=True) if old is not None and len(old) else df
            out = out.drop_duplicates(subset=["concept", "code"]).reset_index(drop=True)
            tmp = path.with_suffix(".parquet.tmp")
            out.to_parquet(tmp, index=False)
            tmp.replace(path)
            print(f"[ths-members] {len(out)} 行 → {path}", flush=True)
    return df


def fetch_ths_concepts(save: bool = True) -> pd.DataFrame:
    """保存同花顺 375 概念板块列表（成分见 concept_member_ths.parquet，由 --ths-members 增量抓取）。
    输出: concept_ths_boards.parquet —— 供题材关键词/板块热度增强。
    """
    import akshare as ak
    boards = ak.stock_board_concept_name_ths()  # name, code
    out = boards.copy()
    out.columns = ["concept", "code"]
    if save:
        p = MARKET_DIR / "concept_ths_boards.parquet"
        out.to_parquet(p, index=False)
        print(f"[ths] 概念板块列表 {len(out)} 个 → {p}")
    return out



def ths_members_status() -> dict:
    """离线查看 THS 概念成分抓取进度（不联网）。"""
    import json as _json
    total = 0
    if (MARKET_DIR / "concept_ths_boards.parquet").exists():
        try:
            total = len(pd.read_parquet(MARKET_DIR / "concept_ths_boards.parquet"))
        except Exception:  # noqa: BLE001
            total = 0
    done = 0
    if THS_MEMBER_DONE.exists():
        try:
            done = len(_json.loads(THS_MEMBER_DONE.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            done = 0
    rows = 0
    for path in (THS_MEMBER_FILE, THS_MEMBER_ALT):
        if path.exists():
            try:
                rows = len(pd.read_parquet(path))
                break
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).error(f"[data_sources] 操作失败: {e}", exc_info=True)
                continue
    return {"boards_total": total, "boards_done": done, "member_rows": rows,
            "pct": round(done / total * 100, 1) if total else 0.0}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="V11 统一数据源")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cloud-status", action="store_true", help="只打印云快照状态（不联网）")
    ap.add_argument("--ths-concepts", action="store_true")
    ap.add_argument("--ths-members", action="store_true", help="增量抓同花顺概念成分（断点续传）")
    ap.add_argument("--ths-status", action="store_true", help="离线查看 THS 成分抓取进度（不联网）")
    ap.add_argument("--limit", type=int, default=0, help="--ths-members 只抓前 N 板块（调试）")
    ap.add_argument("--refresh", action="store_true", help="--ths-members 强制重抓已完成板块")
    args = ap.parse_args()
    if args.check:
        check_availability()
    if args.cloud_status:
        import json as _json
        print(_json.dumps(_cloud_snapshots_status(), ensure_ascii=False, indent=2))
    if args.ths_concepts:
        fetch_ths_concepts()
    if args.ths_members:
        fetch_ths_concept_members(limit=args.limit, refresh=args.refresh)
    if args.ths_status:
        import json as _json
        print(_json.dumps(ths_members_status(), ensure_ascii=False, indent=2))
    if not (args.check or args.cloud_status or args.ths_concepts or args.ths_members or args.ths_status):
        ap.print_help()
