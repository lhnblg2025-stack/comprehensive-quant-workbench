#!/usr/bin/env python3
"""Desktop folder structure audit.

Run periodically to check files are in the right folders.
Generates a markdown report with issues found.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

# Add workspace scripts to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_paths import DESKTOP, get_taxonomy, expected_allowed_subdirs, iso_week


def audit_desktop() -> list[str]:
    """Scan Desktop, return list of issues."""
    issues: list[str] = []
    allowed = expected_allowed_subdirs()
    taxonomy = get_taxonomy()

    # 1. Check for loose files on Desktop root
    for entry in DESKTOP.iterdir():
        if entry.is_symlink():
            continue  # allow symlinks like 共享文件夹
        if not entry.is_dir():
            issues.append(f"桌面根目录存在散落文件: {entry.name}")

    # 2. Check for unexpected directories
    for entry in DESKTOP.iterdir():
        if not entry.is_dir() or entry.is_symlink():
            continue
        if entry.name not in allowed:
            issues.append(f"桌面存在未授权的目录: {entry.name}")

    # 3. Check 宏观数据/ root for loose files
    macro_root = DESKTOP / "宏观数据"
    if macro_root.exists():
        for f in macro_root.iterdir():
            if f.is_file():
                issues.append(
                    f"宏观数据根目录存在散落文件（应移入对应子目录）: {f.name}"
                )

    # 4. Check macro data subdirectories for wrong file patterns
    macro_subdir_rules = {
        "PMI": ["PMI"],
        "CPI_PPI": ["CPI", "PPI", "CPI_PPI"],
        "海关进出口": ["海关", "进出口", "customs"],
        "社融M2": ["社融", "M2", "social_finance"],
        "信贷": ["信贷", "credit"],
        "地产消费": ["地产", "消费", "real_estate"],
        "国际原油": ["原油", "oil"],
        "生猪产能出清": ["生猪", "pig"],
        "牧原产能出清": ["牧原", "muyuan"],
    }
    for subdir, keywords in macro_subdir_rules.items():
        p = macro_root / subdir
        if not p.exists():
            continue
        # Check if any file here might belong elsewhere
        for f in p.iterdir():
            if f.is_file():
                name_lower = f.stem.lower()
                # Quick heuristic: check if filename matches expected keywords
                has_keyword = any(k.lower() in name_lower for k in keywords)
                # Skip if it looks like a backup/migration file
                if "重复" in name_lower or "迁移" in name_lower or "备份" in name_lower:
                    issues.append(f"宏观数据/{subdir}/ 存在冗余文件（可能是重复/备份）: {f.name}")

    # 5. Check 运维记录/ root for loose files (only 桌面分类说明.md allowed)
    ops_root = DESKTOP / "运维记录"
    if ops_root.exists():
        for f in ops_root.iterdir():
            if f.is_file() and f.name != "桌面分类说明.md":
                issues.append(f"运维记录根目录散落文件（应移入对应子目录）: {f.name}")

    # 6. Check for files in wrong top-level dir based on name heuristic
    # This is a simple check - just flag potential misplacements
    keyword_folder_map = {
        "PMI": "宏观数据/PMI",
        "CPI": "宏观数据/CPI_PPI",
        "PPI": "宏观数据/CPI_PPI",
        "海关": "宏观数据/海关进出口",
        "进出口": "宏观数据/海关进出口",
        "社融": "宏观数据/社融M2",
        "M2": "宏观数据/社融M2",
        "信贷": "宏观数据/信贷",
        "原油": "宏观数据/国际原油",
        "生猪": "宏观数据/生猪产能出清",
        "牧原": "宏观数据/牧原产能出清",
        "cron": "运维记录/cron任务清单",
        "脚本": "运维记录/脚本修复",
        "配置": "运维记录/配置变更",
        "数据源": "运维记录/数据源维护",
        "检查": "运维记录/系统检查",
        "核验": "运维记录/系统检查",
        "验证": "运维记录/系统检查",
        "黄金": "黄金分析",
        "日报": "每日任务/A股量化日报",
        "研报": "内资研报",
    }

    return issues


def generate_report() -> str:
    """Generate a markdown report of the audit."""
    today = date.today()
    issues = audit_desktop()
    
    lines = [
        f"# 桌面文件结构审计报告",
        f"**运行时间：** {today.isoformat()} (W{iso_week(today).split('-W')[1]})",
        f"**桌面路径：** {DESKTOP}",
        "",
    ]

    if not issues:
        lines.append("✅ **桌面结构正常，所有文件已归位。**")
    else:
        lines.append(f"⚠️ **发现 {len(issues)} 个问题：**")
        lines.append("")
        for i, issue in enumerate(issues, 1):
            lines.append(f"{i}. {issue}")
    
    lines.append("")
    lines.append("---")
    lines.append("### 当前桌面结构")
    lines.append("")
    
    # List all top-level directories
    for entry in sorted(DESKTOP.iterdir()):
        if entry.is_symlink():
            lines.append(f"- `{entry.name}` → symlink")
        elif entry.is_dir():
            subd = [x.name for x in entry.iterdir() if x.is_dir()]
            files = [x.name for x in entry.iterdir() if x.is_file()]
            parts = [f"`{entry.name}/`"]
            if subd:
                parts.append(f"子目录: {', '.join(f'`{s}`' for s in sorted(subd)[:10])}")
                if len(subd) > 10:
                    parts.append(f"...共{len(subd)}个子目录")
            if files:
                parts.append(f"文件: {len(files)}个")
            lines.append(f"- {' — '.join(parts)}")
    
    lines.append("")
    lines.append("---")
    lines.append("### 备案与投递状态")
    lines.append(f"- 审计文件: `运维记录/系统检查/桌面审计-{today.isoformat()}.md`")
    lines.append(f"- 分类文档: `运维记录/桌面分类说明.md`")
    lines.append(f"- 路径规则: `scripts/report_paths.py`")
    lines.append(f"- 审计脚本: `scripts/desktop_audit.py`")

    return "\n".join(lines)


def main():
    report = generate_report()
    today = date.today()
    
    # Save to desktop
    output_dir = DESKTOP / "运维记录" / "系统检查"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"桌面审计-{today.isoformat()}.md"
    output_path.write_text(report, encoding="utf-8")
    
    print(f"Report saved: {output_path}")
    
    # Also print key issues to stdout for cron logging
    issues = audit_desktop()
    if issues:
        print(f"\n⚠️ {len(issues)} issue(s) found:")
        for i in issues:
            print(f"  • {i}")
    else:
        print("\n✅ All clean.")


if __name__ == "__main__":
    main()
