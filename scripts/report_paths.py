#!/usr/bin/env python3
"""Shared report path rules for OpenClaw cron reports.

Cron prompts and report scripts should use these helpers instead of inventing
Desktop paths. This keeps monthly macro reports out of weekly report folders and
prevents reports from being saved to shared folders.

Folder Taxonomy (also documented in 桌面分类说明.md):
  See get_taxonomy() below for the full hierarchy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import re

# 支持环境变量覆盖（云端部署时 QUANT_DESKTOP=/root/quant_desktop 等，本机默认桌面）
# strip()：run_task.bat 的 `set QUANT_DESKTOP=C:\quant\out ` 行尾空格会进入环境变量值，
# 若不清理，Path("C:\quant\out ") 拼接子目录会产生 "out \宏观数据" 的错误路径。
DESKTOP = Path(os.environ.get("QUANT_DESKTOP", "${QUANT_DESKTOP:-.}").strip())


# ── Full Folder Taxonomy ──────────────────────────────────────────────────


def get_taxonomy() -> dict[str, dict]:
    """Return the complete desktop folder taxonomy.

    Each entry: { "path": Path, "description": str, "type": str }
    Used by audit/validation scripts.
    """
    D = DESKTOP
    return {
        # ── 每日任务 ──
        "每日任务-根": {"path": D / "每日任务", "description": "每日生成的任务报告", "type": "root"},
        "每日任务-A股量化日报": {"path": D / "每日任务" / "A股量化日报", "description": "A股量化日报", "type": "leaf"},
        # ── 周报 ──
        "周报-根": {"path": D / "周报", "description": "按 ISO 周次归档的周报", "type": "root"},
        "周报-周次子目录": {"path": D / "周报" / "{week}", "description": "每周周报文件", "type": "dynamic_leaf"},
        # ── 宏观数据 ──
        "宏观数据-根": {"path": D / "宏观数据", "description": "宏观月度/专题数据与解读", "type": "root"},
        "宏观数据-PMI": {"path": D / "宏观数据" / "PMI", "description": "PMI 数据报告", "type": "leaf"},
        "宏观数据-CPI_PPI": {"path": D / "宏观数据" / "CPI_PPI", "description": "CPI/PPI 数据报告", "type": "leaf"},
        "宏观数据-海关进出口": {"path": D / "宏观数据" / "海关进出口", "description": "海关进出口数据报告", "type": "leaf"},
        "宏观数据-社融M2": {"path": D / "宏观数据" / "社融M2", "description": "社融/M2 数据报告", "type": "leaf"},
        "宏观数据-信贷": {"path": D / "宏观数据" / "信贷", "description": "信贷数据报告", "type": "leaf"},
        "宏观数据-经济数据": {"path": D / "宏观数据" / "经济数据", "description": "综合经济数据、半年报与专题解读", "type": "leaf"},
        "宏观数据-地产消费": {"path": D / "宏观数据" / "地产消费", "description": "地产/消费数据报告", "type": "leaf"},
        "宏观数据-国际原油": {"path": D / "宏观数据" / "国际原油", "description": "国际原油数据报告", "type": "leaf"},
        "宏观数据-生猪产能出清": {"path": D / "宏观数据" / "生猪产能出清", "description": "生猪产能出清数据", "type": "leaf"},
        "宏观数据-牧原产能出清": {"path": D / "宏观数据" / "牧原产能出清", "description": "牧原产能出清数据", "type": "leaf"},
        # ── 行业景气数据（单一行业专题分析用，不进因子流水线）──
        "宏观数据-汽车景气": {"path": D / "宏观数据" / "汽车", "description": "汽车行业景气（中汽协产销/新能源渗透率）", "type": "leaf"},
        "宏观数据-快递物流": {"path": D / "宏观数据" / "快递物流", "description": "快递物流景气（邮政业务量/货运量）", "type": "leaf"},
        "宏观数据-电力": {"path": D / "宏观数据" / "电力", "description": "电力景气（全社会用电量/分产业）", "type": "leaf"},
        "宏观数据-半导体": {"path": D / "宏观数据" / "半导体", "description": "半导体景气（电子指数/工业增加值）", "type": "leaf"},
        "宏观数据-煤炭钢铁水泥": {"path": D / "宏观数据" / "煤炭钢铁水泥", "description": "煤炭/钢铁/水泥景气（统计局产量，待爬）", "type": "leaf"},
        "宏观数据-白酒": {"path": D / "宏观数据" / "白酒", "description": "白酒景气（统计局产量，待爬）", "type": "leaf"},
        "宏观数据-黄金有色": {"path": D / "宏观数据" / "黄金有色", "description": "黄金有色景气（上金所行情）", "type": "leaf"},
        "宏观数据-生猪景气": {"path": D / "宏观数据" / "生猪", "description": "生猪景气（搜猪网现货猪价/饲料）", "type": "leaf"},
        # ── 黄金分析 ──
        "黄金分析-根": {"path": D / "黄金分析", "description": "黄金及贵金属相关分析", "type": "root"},
        "黄金分析-日报": {"path": D / "黄金分析" / "日报", "description": "黄金每日分析", "type": "leaf"},
        "黄金分析-周报": {"path": D / "黄金分析" / "周报", "description": "黄金每周分析", "type": "leaf"},
        "黄金分析-专题": {"path": D / "黄金分析" / "专题", "description": "黄金专题报告", "type": "leaf"},
        # ── 会议纪要 ──
        "会议纪要-根": {"path": D / "会议纪要", "description": "会议纪要、研报纪要", "type": "root"},
        "会议纪要-本周研报": {"path": D / "会议纪要" / "本周研报", "description": "本周研报纪要汇总", "type": "leaf"},
        "会议纪要-日常会议": {"path": D / "会议纪要" / "日常会议", "description": "日常会议纪要", "type": "leaf"},
        "会议纪要-研报纪要": {"path": D / "会议纪要" / "研报纪要", "description": "中长期研报纪要存档", "type": "leaf"},
        # ── 内资研报 ──
        "内资研报-根": {"path": D / "内资研报", "description": "券商/内资研报汇总", "type": "root"},
        "内资研报-周次子目录": {"path": D / "内资研报" / "{week}", "description": "按周的研报汇总", "type": "dynamic_leaf"},
        # ── 项目文档 ──
        "项目文档-根": {"path": D / "项目文档", "description": "非定时报表类项目资料", "type": "root"},
        "项目文档-量化交易系统": {"path": D / "项目文档" / "量化交易系统", "description": "量化交易系统设计文档", "type": "leaf"},
        "项目文档-学术文档": {"path": D / "项目文档" / "学术文档", "description": "学术论文/文档", "type": "leaf"},
        "项目文档-ETF套利回测": {"path": D / "项目文档" / "量化交易系统", "description": "ETF套利回测分析报告", "type": "leaf"},
        # ── 运维记录 ──
        "运维记录-根": {"path": D / "运维记录", "description": "cron、系统、插件、配置等运维记录", "type": "root"},
        "运维记录-cron任务清单": {"path": D / "运维记录" / "cron任务清单", "description": "Cron 任务清单盘点", "type": "leaf"},
        "运维记录-脚本修复": {"path": D / "运维记录" / "脚本修复", "description": "脚本修复/补丁记录", "type": "leaf"},
        "运维记录-配置变更": {"path": D / "运维记录" / "配置变更", "description": "系统/配置变更记录", "type": "leaf"},
        "运维记录-数据源维护": {"path": D / "运维记录" / "数据源维护", "description": "数据源问题修复/切换", "type": "leaf"},
        "运维记录-系统检查": {"path": D / "运维记录" / "系统检查", "description": "系统健康检查/核验", "type": "leaf"},
        # ── 书籍 ──
        "books": {"path": D / "books", "description": "原始电子书文件", "type": "leaf"},
        "books-skills": {"path": D / "books-skills", "description": "用于技能构建的书籍材料", "type": "leaf"},
        "books-rag": {"path": D / "books-rag", "description": "书籍RAG知识库（全文分块+向量索引+检索）", "type": "root"},
        # ── 共享文件夹（禁止写入报告） ──
        "共享文件夹": {"path": D / "共享文件夹", "description": "VM 共享文件夹快捷方式（禁止写入报告）", "type": "forbidden"},
    }


# ── Report Path Rules ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReportPathRule:
    key: str
    root: Path
    filename_template: str


RULES: dict[str, ReportPathRule] = {
    # ── 每日任务 ──
    "a_share_daily": ReportPathRule(
        "a_share_daily",
        DESKTOP / "每日任务" / "A股量化日报",
        "{date}-A股量化日报.md",
    ),
    # ── 宏观月度数据 ──
    "pmi": ReportPathRule("pmi", DESKTOP / "宏观数据" / "PMI", "{month}_PMI.md"),
    "cpi_ppi": ReportPathRule("cpi_ppi", DESKTOP / "宏观数据" / "CPI_PPI", "{month}_CPI_PPI.md"),
    "customs": ReportPathRule("customs", DESKTOP / "宏观数据" / "海关进出口", "{month}_海关进出口.md"),
    "social_finance_m2": ReportPathRule("social_finance_m2", DESKTOP / "宏观数据" / "社融M2", "{month}_社融M2.md"),
    "credit": ReportPathRule("credit", DESKTOP / "宏观数据" / "信贷", "{month}_信贷.md"),
    "real_estate_consumption": ReportPathRule("real_estate_consumption", DESKTOP / "宏观数据" / "地产消费", "{month}_地产消费.md"),
    "international_oil": ReportPathRule("international_oil", DESKTOP / "宏观数据" / "国际原油", "{date}_国际原油.md"),
    "oil_monthly": ReportPathRule("oil_monthly", DESKTOP / "宏观数据" / "国际原油", "{date}_国际原油.md"),
    "pig_capacity": ReportPathRule("pig_capacity", DESKTOP / "宏观数据" / "生猪产能出清", "{month}_生猪产能出清.md"),
    "muyuan_capacity": ReportPathRule("muyuan_capacity", DESKTOP / "宏观数据" / "牧原产能出清", "{month}_牧原产能出清检查.md"),
    # ── 行业景气数据 ──
    "industry_auto": ReportPathRule("industry_auto", DESKTOP / "宏观数据" / "汽车", "{month}_汽车景气.md"),
    "industry_express": ReportPathRule("industry_express", DESKTOP / "宏观数据" / "快递物流", "{month}_快递物流景气.md"),
    "industry_power": ReportPathRule("industry_power", DESKTOP / "宏观数据" / "电力", "{month}_电力景气.md"),
    "industry_semiconductor": ReportPathRule("industry_semiconductor", DESKTOP / "宏观数据" / "半导体", "{month}_半导体景气.md"),
    "industry_heavy": ReportPathRule("industry_heavy", DESKTOP / "宏观数据" / "煤炭钢铁水泥", "{month}_煤炭钢铁水泥景气.md"),
    "industry_baijiu": ReportPathRule("industry_baijiu", DESKTOP / "宏观数据" / "白酒", "{month}_白酒景气.md"),
    "industry_gold": ReportPathRule("industry_gold", DESKTOP / "宏观数据" / "黄金有色", "{date}_黄金有色景气.md"),
    "industry_pig": ReportPathRule("industry_pig", DESKTOP / "宏观数据" / "生猪", "{date}_生猪景气.md"),
    "industry_shipping": ReportPathRule("industry_shipping", DESKTOP / "宏观数据" / "航运", "{month}_航运景气.md"),
    "industry_steel_coal": ReportPathRule("industry_steel_coal", DESKTOP / "宏观数据" / "煤炭钢铁水泥", "{date}_煤钢价格景气.md"),
    "industry_insurance": ReportPathRule("industry_insurance", DESKTOP / "宏观数据" / "保险", "{month}_保险景气.md"),
    "industry_silver": ReportPathRule("industry_silver", DESKTOP / "宏观数据" / "黄金有色" / "白银", "{date}_白银基准价.md"),
    "industry_pulse": ReportPathRule("industry_pulse", DESKTOP / "宏观数据", "{month}_行业景气速览.md"),
    # ── 周报 ──
    "macro_weekly": ReportPathRule("macro_weekly", DESKTOP / "周报" / "{week}", "宏观数据周报.md"),
    "dividend_weekly": ReportPathRule("dividend_weekly", DESKTOP / "周报" / "{week}", "红利指数.md"),
    "zijin_weekly": ReportPathRule("zijin_weekly", DESKTOP / "周报" / "{week}", "紫金矿业.md"),
    "gold_weekly": ReportPathRule("gold_weekly", DESKTOP / "周报" / "{week}", "黄金宏观.md"),
    "china_internet_weekly": ReportPathRule("china_internet_weekly", DESKTOP / "周报" / "{week}", "159792中概互联网.md"),
    "muyuan_weekly": ReportPathRule("muyuan_weekly", DESKTOP / "周报" / "{week}", "牧原股份.md"),
    "pig_weekly": ReportPathRule("pig_weekly", DESKTOP / "周报" / "{week}", "生猪周期.md"),
    "complete_weekly": ReportPathRule("complete_weekly", DESKTOP / "周报" / "{week}", "完整周报.md"),
    # ── 黄金分析 ──
    "gold_daily": ReportPathRule("gold_daily", DESKTOP / "黄金分析" / "日报", "{date}_黄金日报.md"),
    "gold_weekly_analysis": ReportPathRule("gold_weekly_analysis", DESKTOP / "黄金分析" / "周报", "{week}_黄金周报.md"),
    "gold_special": ReportPathRule("gold_special", DESKTOP / "黄金分析" / "专题", "{date}_黄金专题.md"),
    # ── 会议纪要 ──
    "meeting_weekly_research": ReportPathRule("meeting_weekly_research", DESKTOP / "会议纪要" / "本周研报", "{week}_本周研报纪要.md"),
    "meeting_daily": ReportPathRule("meeting_daily", DESKTOP / "会议纪要" / "日常会议", "{date}_会议纪要.md"),
    "meeting_research": ReportPathRule("meeting_research", DESKTOP / "会议纪要" / "研报纪要", "{date}_研报纪要.md"),
    # ── 内资研报 ──
    "domestic_research_weekly": ReportPathRule("domestic_research_weekly", DESKTOP / "内资研报" / "{week}", "本周研报汇总_{week}.md"),
    # ── 项目文档 ──
    "project_trading_system": ReportPathRule("project_trading_system", DESKTOP / "项目文档" / "量化交易系统", "{date}_量化系统设计.md"),
    "project_academic": ReportPathRule("project_academic", DESKTOP / "项目文档" / "学术文档", "{filename}"),
    "project_etf_arbitrage": ReportPathRule("project_etf_arbitrage", DESKTOP / "项目文档" / "量化交易系统", "{date}_ETF套利回测.md"),
    # ── 运维记录 ──
    "ops_cron_inventory": ReportPathRule("ops_cron_inventory", DESKTOP / "运维记录" / "cron任务清单", "cron任务清单-{date}.md"),
    "ops_script_fix": ReportPathRule("ops_script_fix", DESKTOP / "运维记录" / "脚本修复", "{topic}-{date}.md"),
    "ops_config_change": ReportPathRule("ops_config_change", DESKTOP / "运维记录" / "配置变更", "{topic}-{date}.md"),
    "ops_data_source": ReportPathRule("ops_data_source", DESKTOP / "运维记录" / "数据源维护", "{topic}-{date}.md"),
    "ops_system_check": ReportPathRule("ops_system_check", DESKTOP / "运维记录" / "系统检查", "{topic}-{date}.md"),
}

# ── Cron Name → Rule Key ────────────────────────────────────────────────

ALIASES = {
    # 每日任务
    "A股量化日报": "a_share_daily",
    # 宏观数据
    "宏观-PMI数据解读": "pmi",
    "宏观-CPI-PPI数据解读": "cpi_ppi",
    "宏观-海关进出口数据解读": "customs",
    "宏观-社融M2数据解读": "social_finance_m2",
    "宏观-信贷数据解读": "credit",
    "宏观-地产消费数据解读": "real_estate_consumption",
    "国际原油官方数据报告": "international_oil",
    "月度原油市场报告": "oil_monthly",
    "生猪产能出清-每月10号": "pig_capacity",
    "牧原股份-产能出清月度检查": "muyuan_capacity",
    # 行业景气
    "行业景气-汽车": "industry_auto",
    "行业景气-快递物流": "industry_express",
    "行业景气-电力": "industry_power",
    "行业景气-半导体": "industry_semiconductor",
    "行业景气-煤炭钢铁水泥": "industry_heavy",
    "行业景气-白酒": "industry_baijiu",
    "行业景气-黄金有色": "industry_gold",
    "行业景气-生猪": "industry_pig",
    "行业景气-航运": "industry_shipping",
    "行业景气-煤钢": "industry_steel_coal",
    "行业景气-保险": "industry_insurance",
    "行业景气-白银": "industry_silver",
    "行业景气-速览": "industry_pulse",
    # 周报
    "5-宏观数据周报": "macro_weekly",
    "1-红利指数周报": "dividend_weekly",
    "3-紫金矿业周报": "zijin_weekly",
    "4-黄金宏观周报": "gold_weekly",
    "159792-定投周报": "china_internet_weekly",
    "2-牧原股份周报": "muyuan_weekly",
    "生猪周期周报": "pig_weekly",
    # 黄金分析
    "黄金日报": "gold_daily",
    "黄金周报-分析": "gold_weekly_analysis",
    "黄金专题": "gold_special",
    # 会议纪要
    "本周研报纪要": "meeting_weekly_research",
    "会议纪要": "meeting_daily",
    "研报纪要": "meeting_research",
    # 内资研报
    "内资研报周汇总": "domestic_research_weekly",
    # 运维记录
    "运维-cron清单": "ops_cron_inventory",
    "运维-脚本修复": "ops_script_fix",
    "运维-配置变更": "ops_config_change",
    "运维-数据源维护": "ops_data_source",
    "运维-系统检查": "ops_system_check",
    # 项目文档
    "ETF套利回测": "project_etf_arbitrage",
}


# ── Helper Functions ─────────────────────────────────────────────────────


def iso_week(d: date | datetime | None = None) -> str:
    d = (d or date.today())
    if isinstance(d, datetime):
        d = d.date()
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def month_key(d: date | datetime | str | None = None) -> str:
    if isinstance(d, str):
        m = re.match(r"^(\d{4})[-/]?(\d{2})", d)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
        raise ValueError(f"cannot parse month from {d!r}")
    d = d or date.today()
    if isinstance(d, datetime):
        d = d.date()
    return d.strftime("%Y-%m")


def date_key(d: date | datetime | str | None = None) -> str:
    if isinstance(d, str):
        m = re.match(r"^(\d{4})[-/]?(\d{2})[-/]?(\d{2})", d)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        raise ValueError(f"cannot parse date from {d!r}")
    d = d or date.today()
    if isinstance(d, datetime):
        d = d.date()
    return d.strftime("%Y-%m-%d")


def report_path(
    key_or_name: str,
    *,
    d: date | datetime | str | None = None,
    month: str | None = None,
    topic: str = "",
    filename: str = "",
    ensure_dir: bool = True,
) -> Path:
    """Resolve a report file path by key or alias.

    Parameters:
        key_or_name: rule key or cron alias
        d: date override (default today)
        month: month string override (e.g. "2026-06")
        topic: placeholder for ops reports ({topic})
        filename: placeholder for project academic ({filename})
        ensure_dir: auto-create parent dir (default True)
    """
    key = ALIASES.get(key_or_name, key_or_name)
    if key not in RULES:
        raise KeyError(f"unknown report path key: {key_or_name}")
    rule = RULES[key]
    values = {
        "date": date_key(d),
        "month": month_key(month or d),
        "week": iso_week(d if not isinstance(d, str) else None),
        "topic": topic or "unknown",
        "filename": filename or "unknown",
    }
    root = Path(str(rule.root).format(**values))
    fname = rule.filename_template.format(**values)
    if ensure_dir:
        root.mkdir(parents=True, exist_ok=True)
    return root / fname


def assert_not_shared_path(path: str | Path) -> None:
    p = str(path)
    if "/mnt/hgfs" in p or "/share/" in p:
        raise ValueError(f"shared folder paths are forbidden for report output: {p}")


def assert_expected_folder(
    key_or_name: str,
    path: str | Path,
    *,
    d: date | datetime | str | None = None,
    month: str | None = None,
) -> None:
    expected = report_path(key_or_name, d=d, month=month, ensure_dir=False).parent
    p = Path(path)
    assert_not_shared_path(p)
    if p.parent != expected:
        raise ValueError(
            f"wrong report folder for {key_or_name}: {p.parent}; expected {expected}"
        )


# ── Audit Helpers ────────────────────────────────────────────────────────


def expected_allowed_subdirs() -> set[str]:
    """Return the set of allowed top-level directory names under Desktop."""
    return {
        "每日任务", "周报", "宏观数据", "黄金分析", "会议纪要",
        "内资研报", "项目文档", "运维记录",
        "books", "books-skills", "books-rag", "共享文件夹",
    }


def validate_desktop_structure() -> list[str]:
    """Scan Desktop and return list of issues found."""
    issues: list[str] = []
    allowed = expected_allowed_subdirs()
    taxonomy = get_taxonomy()

    for entry in DESKTOP.iterdir():
        if entry.is_symlink() and entry.resolve().as_posix().startswith("/mnt/hgfs/"):
            # VMware shared-folder mount symlinks are host-managed; skip audit
            continue
        if not entry.is_dir():
            # .desktop files on root: may need moving to project docs
            if entry.suffix == ".desktop":
                issues.append(f"Loose .desktop file on Desktop root: {entry.name} → 建议移入项目文档/")
            else:
                issues.append(f"Loose file on Desktop root: {entry.name}")
            continue
        if entry.name not in allowed:
            issues.append(f"Unexpected directory on Desktop: {entry.name}")

    # Check empty required directories
    for name, info in taxonomy.items():
        p = info["path"]
        if info["type"] == "leaf" and p.exists() and not any(p.iterdir()):
            issues.append(f"Empty leaf directory: {p.relative_to(DESKTOP)}")

    return issues


if __name__ == "__main__":
    # Self-test
    print("=== report_paths.py self-test ===")
    print(f"Today: {date_key()}, Month: {month_key()}, Week: {iso_week()}")
    for key in sorted(RULES):
        try:
            p = report_path(key, ensure_dir=False)
            print(f"  {key:30s} → {p}")
        except Exception as e:
            print(f"  {key:30s} → ERROR: {e}")
    print()
    issues = validate_desktop_structure()
    if issues:
        print("=== Desktop Structure Issues ===")
        for i in issues:
            print(f"  ⚠ {i}")
    else:
        print("=== Desktop Structure OK ===")
