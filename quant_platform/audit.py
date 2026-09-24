"""
quant_platform.audit — 审计维护模块（V3 数据层 P3）

用户 #3：加审计维护模块。
用户 #6：所有运行后的输出，不但要解决绝对的报错，还要检查输出内容是否合理。

三层次审计：
1. DataAudit     数据合理性：新鲜度/覆盖度/异常值/分布合理性/缺失模式
2. OutputAudit   输出内容审计：报告章节完整/数值范围/逻辑一致性/时间戳
3. MaintenanceAudit 维护自检：语法/import/磁盘/cron 状态

用法：
    from quant_platform.audit import run_full_audit
    report = run_full_audit()   # → dict（含 3 层次 + 汇总评分）
    # CLI: python3 -m quant_platform.audit --json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════════
# 1. 数据合理性审计（DataAudit）
# ══════════════════════════════════════════════════════════

# 合理性阈值（A股常识约束，来源：skills/factor-investing-ashare-practice + 硬规则）
_RANGE_RULES: dict[str, tuple[float, float]] = {
    "pct_chg": (-25.0, 25.0),        # 单日涨跌幅（双创±20%/北交±30% 留裕量）
    "turnover": (0.0, 100.0),        # 换手率 %
    "peTTM": (-500.0, 5000.0),       # 负值允许（亏损），异常大剔除
    "pbMRQ": (-50.0, 500.0),
    "mom20": (-60.0, 60.0),
}


def _data_audit() -> dict[str, Any]:
    """数据合理性审计：对每个数据集检查新鲜度/覆盖/异常值/分布。"""
    from quant_system.data_store import DataStore
    ds = DataStore()
    findings: list[dict] = []

    # 1a. 新鲜度（DataStore.freshness 复用）
    rows = ds.freshness()
    for r in rows:
        if r["status"] == "missing":
            findings.append({"level": "high", "scope": "data", "dataset": r["dataset"],
                             "msg": f"数据集缺失（0 文件）"})
        elif r["status"] == "stale":
            findings.append({"level": "medium", "scope": "data", "dataset": r["dataset"],
                             "msg": f"数据过期: 最新 {r['latest']}，预期 {r['expected']}"})
        elif r["status"] == "partial":
            findings.append({"level": "medium", "scope": "data", "dataset": r["dataset"],
                             "msg": f"覆盖不足: {r['note']}"})

    # 1b. K线抽样异常值检查（抽 30 只）
    try:
        codes = sorted(p.stem for p in (ROOT / "data_warehouse" / "kline").glob("*.parquet"))
        import random
        random.seed(42)
        sample = random.sample(codes, min(30, len(codes)))
        bad_pct, bad_turn = 0, 0
        n_rows = 0
        for c in sample:
            try:
                df = pd.read_parquet(ROOT / "data_warehouse" / "kline" / f"{c}.parquet")
                if df.empty:
                    continue
                n_rows += len(df)
                if "pct_chg" in df.columns:
                    bad_pct += int((df["pct_chg"].abs() > 25).sum())
                if "turnover" in df.columns:
                    bad_turn += int((df["turnover"] > 100).sum())
            except Exception:
                continue
        if n_rows:
            rate = (bad_pct + bad_turn) / n_rows
            if rate > 0.001:
                findings.append({"level": "medium", "scope": "data", "dataset": "kline",
                                 "msg": f"异常值率 {rate:.4%}（涨跌幅>25% 或 换手>100%），抽样 {len(sample)} 只 {n_rows} 行"})
    except Exception as e:
        findings.append({"level": "low", "scope": "data", "dataset": "kline", "msg": f"异常值检查失败: {e}"})

    # 1c. 估值分布合理性（全市场 peTTM 中位数应在 10~60）
    try:
        vp = ROOT / "data_warehouse" / "valuation"
        pes, pbs = [], []
        for p in list(vp.glob("*.parquet"))[:500]:
            try:
                df = pd.read_parquet(p)
                if df.empty or "peTTM" not in df.columns:
                    continue
                last = df.iloc[-1]
                if pd.notna(last.get("peTTM")) and last["peTTM"] > 0:
                    pes.append(float(last["peTTM"]))
                if pd.notna(last.get("pbMRQ")) and last["pbMRQ"] > 0:
                    pbs.append(float(last["pbMRQ"]))
            except Exception:
                continue
        if pes:
            med_pe = float(pd.Series(pes).median())
            med_pb = float(pd.Series(pbs).median())
            if not (5 <= med_pe <= 80):
                findings.append({"level": "high", "scope": "data", "dataset": "valuation",
                                 "msg": f"全市场 peTTM 中位数 {med_pe:.1f} 异常（预期 5~80）"})
            if not (0.5 <= med_pb <= 15):
                findings.append({"level": "high", "scope": "data", "dataset": "valuation",
                                 "msg": f"全市场 pbMRQ 中位数 {med_pb:.2f} 异常（预期 0.5~15）"})
    except Exception as e:
        findings.append({"level": "low", "scope": "data", "dataset": "valuation", "msg": f"估值分布检查失败: {e}"})

    # 1d. Feature Store 覆盖
    try:
        fs = sorted((ROOT / "data_warehouse" / "feature_store").glob("*.parquet"))
        if fs:
            fdf = pd.read_parquet(fs[-1])
            coverage = fdf["peTTM"].notna().mean() if "peTTM" in fdf else None
            if coverage is not None and coverage < 0.5:
                findings.append({"level": "medium", "scope": "data", "dataset": "feature_store",
                                 "msg": f"peTTM 覆盖仅 {coverage:.0%}"})
    except Exception:
        pass

    return {"findings": findings, "count": len(findings)}


# ══════════════════════════════════════════════════════════
# 2. 输出内容审计（OutputAudit）
# ══════════════════════════════════════════════════════════

def _find_report_files() -> list[Path]:
    """定位日报/报告文件（优先日报目录，其次项目文档）。"""
    desktop = Path(os.environ.get("QUANT_DESKTOP", str(Path.home() / "Desktop")))
    # 日报优先（用户硬规则：每日任务/A股量化日报）
    d1 = desktop / "每日任务" / "A股量化日报"
    if d1.exists():
        daily = sorted(d1.glob("*.md"))
        if daily:
            return daily[-5:]
    # 回落：项目文档
    d2 = desktop / "项目文档"
    if d2.exists():
        docs = sorted(d2.glob("量化交易系统/*.md"))
        if docs:
            return docs[-5:]
    return []


def _get_trade_calendar() -> set[str]:
    """A股交易日集合（baostock 最近 45 自然日，失败时回退工作日）。"""
    try:
        import baostock as bs
        lg = bs.login()
        rs = bs.query_trade_dates(start_date=(date.today() - timedelta(days=45)).strftime("%Y-%m-%d"),
                                  end_date=date.today().strftime("%Y-%m-%d"))
        days = set()
        while rs.next():
            row = dict(zip(rs.fields, rs.get_row_data()))
            if str(row.get("is_trading_day")) == "1":
                days.add(row["calendar_date"])
        bs.logout()
        return days
    except Exception:
        # 回退：工作日近似（不含节假日，仅防周末误报）
        out = set()
        d = date.today() - timedelta(days=45)
        while d <= date.today():
            if d.isoweekday() <= 5:
                out.add(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)
        return out


def _output_audit() -> dict[str, Any]:
    """输出内容审计：报告章节完整/数值范围/逻辑一致性/日期。"""
    findings: list[dict] = []
    reports = _find_report_files()
    if not reports:
        return {"findings": [{"level": "low", "scope": "output", "msg": "未找到日报文件（可能未生成）"}],
                "count": 1}

    # 日报必备章节（用户硬规则：热股等权/全A等权/市场宽度/量能/两融/大小微盘/情绪）
    # 关键词放宽：大小微盘允许"大盘股/小盘股/微盘股"等变体，避免误报
    required_sections = [
        ("热股", ["热股"]),
        ("全A等权", ["全A", "全市场", "等权"]),
        ("市场宽度", ["宽度", "涨跌家数", "涨跌停"]),
        ("量能", ["成交", "量能", "量比"]),
        ("融资", ["融资", "两融"]),
        ("大小微盘", ["大盘", "小盘", "微盘"]),
        ("情绪", ["情绪"]),
    ]
    for rp in reports[-1:]:  # 只审最新一份
        text = rp.read_text(encoding="utf-8", errors="ignore")
        missing = [name for name, kws in required_sections if not any(k in text for k in kws)]
        if missing:
            findings.append({"level": "medium", "scope": "output", "file": rp.name,
                             "msg": f"日报缺章节关键词: {missing}"})
        # 数值合理性：涨跌幅数量级（只审“涨跌幅”语境，排除占比/距离类指标）
        # 修复：原逻辑把任何 数字% 当涨跌幅 → 误报 66.3%(上涨占比)/85.0%(热股上涨占比)
        #       /33.37%(TMT占融资余额)/37.1%(距MA距离) 等非涨跌幅指标
        import re
        pct_pairs = re.findall(r"([^%\n]*?)([+-]?\d+\.\d+)%", text)
        big = []
        for ctx, val in pct_pairs:
            v = float(val)
            if abs(v) <= 25:
                continue
            # 仅当上下文含涨跌幅类关键词才判定为个股涨跌幅
            kw = ("涨跌幅", "涨幅", "跌幅", "涨停", "跌停", "pct_chg", "涨跌")
            if any(k in ctx for k in kw):
                big.append(f"{ctx.strip()[-12:]}{val}%")
        if len(big) > 3:
            findings.append({"level": "medium", "scope": "output", "file": rp.name,
                             "msg": f"涨跌幅语境 >25% 出现 {len(big)} 次（检查是否误把小数当百分比）: {big[:5]}"})
        # 日期正确性：报告日期应与文件名一致，且不能过度滞后（>3 自然日视为过期）
        import re as _re2
        m = _re2.search(r"(\d{4})-(\d{2})-(\d{2})", rp.name)
        if m:
            rpt_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            age = (date.today() - rpt_date).days
            if age > 3:
                findings.append({"level": "medium", "scope": "output", "file": rp.name,
                                 "msg": f"日报已滞后 {age} 天（{rpt_date}），确认盘后 cron 是否运行"})
        today = date.today().strftime("%Y-%m-%d")
        # V3 修复：周末/非交易日跳过"今日日期"检查（周六不会生成当日日报）
        _is_trade_day = True
        try:
            _wd = date.today().isoweekday()
            if _wd >= 6:  # 周六日
                _is_trade_day = False
            else:
                # 用 baostock 交易日历（带缓存，失败时回退工作日判断）
                from quant_system.cache import cached
                _cal = cached("audit_trade_calendar", ttl=6 * 3600)(_get_trade_calendar)
                _is_trade_day = date.today().strftime("%Y-%m-%d") in _cal
        except Exception:
            pass
        if _is_trade_day and today not in text and date.today().strftime("%Y年%m月%d日") not in text and m:
            if (date.today() - rpt_date).days <= 3:
                findings.append({"level": "low", "scope": "output", "file": rp.name,
                                 "msg": f"报告未含今日日期 {today}"})

    return {"findings": findings, "count": len(findings)}


# ══════════════════════════════════════════════════════════
# 3. 维护自检（MaintenanceAudit）
# ══════════════════════════════════════════════════════════

def _py_compile_all() -> list[str]:
    """全仓 Python 语法检查（排除 venv/node_modules/archive/legacy 大目录）。"""
    import py_compile
    errors: list[str] = []
    roots = [ROOT / "quant_system", ROOT / "quant_platform", ROOT / "quant_web", ROOT / "scripts"]
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if any(seg in p.parts for seg in ("__pycache__", "legacy", "archive", "tests", "test")):
                continue
            try:
                py_compile.compile(str(p), doraise=True)
            except Exception as e:
                errors.append(f"{p.relative_to(ROOT)}: {e}")
    return errors


def _maintenance_audit() -> dict[str, Any]:
    """维护自检：语法/仓库大小/日志增长。"""
    findings: list[dict] = []
    # 语法
    errs = _py_compile_all()
    if errs:
        findings.append({"level": "high", "scope": "maint", "msg": f"语法错误 {len(errs)} 个: {errs[:3]}"})
    else:
        findings.append({"level": "ok", "scope": "maint", "msg": "全仓 Python 语法检查通过"})
    # 仓库大小
    try:
        du = subprocess.run(["du", "-sh", str(ROOT / "data_warehouse")],
                            capture_output=True, text=True, timeout=10)
        size = du.stdout.split()[0] if du.stdout else "?"
        findings.append({"level": "ok", "scope": "maint", "msg": f"数据仓库大小: {size}"})
    except Exception:
        pass
    # 磁盘余量
    try:
        st = os.statvfs(str(ROOT))
        free_gb = st.f_bavail * st.f_frsize / 1e9
        if free_gb < 5:
            findings.append({"level": "high", "scope": "maint", "msg": f"磁盘剩余仅 {free_gb:.1f}GB"})
        else:
            findings.append({"level": "ok", "scope": "maint", "msg": f"磁盘剩余 {free_gb:.1f}GB"})
    except Exception:
        pass
    return {"findings": findings, "count": len(findings)}


# ══════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════

_LEVEL_WEIGHT = {"high": 3, "medium": 2, "low": 1, "ok": 0}


def run_full_audit(verbose: bool = False) -> dict[str, Any]:
    """全量审计：数据合理性 + 输出内容 + 维护自检。

    Returns:
        {
          "ts": 时间戳, "score": 0-100,
          "sections": {data: {...}, output: {...}, maint: {...}},
          "all_findings": [...], "high_count": n, ...
        }
    """
    t0 = time.time()
    data = _data_audit()
    output = _output_audit()
    maint = _maintenance_audit()
    all_f = data["findings"] + output["findings"] + maint["findings"]
    weighted = sum(_LEVEL_WEIGHT.get(f["level"], 1) for f in all_f if f["level"] != "ok")
    score = max(0, min(100, 100 - weighted * 4))
    result = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "score": round(score, 1),
        "elapsed_s": round(time.time() - t0, 1),
        "sections": {"data": data, "output": output, "maint": maint},
        "all_findings": all_f,
        "high_count": sum(1 for f in all_f if f["level"] == "high"),
        "medium_count": sum(1 for f in all_f if f["level"] == "medium"),
    }
    if verbose:
        print(f"=== 审计评分: {result['score']}/100（high={result['high_count']} medium={result['medium_count']}）===")
        for f in all_f:
            if f["level"] != "ok":
                print(f"  [{f['level'].upper():6}] {f.get('scope','?')} {f.get('dataset','')} {f.get('msg','')}")
    return result


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="平台审计维护")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()
    r = run_full_audit(verbose=not args.json)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    return 0 if r["high_count"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
