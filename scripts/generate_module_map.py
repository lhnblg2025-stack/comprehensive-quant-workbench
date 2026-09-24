#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate_module_map — 模块索引生成器（D1 收敛前置）

产出: 项目文档/量化交易系统/MODULE_MAP.md（目录分组模块清单 + 统计摘要）

实现: 纯 AST + 标准库（ast/pathlib/json/re/datetime/collections），
      不 import、不执行被扫描代码。

扫描范围: quant_system/ 与 scripts/ 下全部 *.py，
          跳过 __pycache__/archive/archived/legacy/垃圾文件夹 及隐藏目录。

每个模块提取:
  - 相对路径、行数
  - 职责: 模块 docstring 首个非空行（截断 80 字）
  - 顶层函数数、类数
  - 被 import 次数: 全仓 import 语句引用计数（AST 解析 import/import-from 并解析到本仓模块）
  - 重复标记: 复用 generated/audit/static_scan_2026-08-11.json 的
    duplicate_function 规则（若存在），标注同名函数跨文件出现的模块

用法:
  python3 scripts/generate_module_map.py
  python3 scripts/generate_module_map.py --out /tmp/MODULE_MAP.md
"""

from __future__ import annotations

import argparse
import ast
import collections
import datetime as _dt
import json
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("quant_system", "scripts")
AUDIT_JSON = ROOT / "generated" / "audit" / "static_scan_2026-08-11.json"
DEFAULT_OUT = ROOT / "项目文档" / "量化交易系统" / "MODULE_MAP.md"
TMP_FALLBACK = Path("/tmp/MODULE_MAP.md")

SKIP_DIR_NAMES = {"__pycache__", "archive", "archived", "legacy", "垃圾文件夹"}
RULE_DUP = "duplicate_function"
DOC_TRUNC = 80
TOP_MODULES = 10
TOP_DUPS = 20


def collect_py_files(root: Path) -> list[Path]:
    """递归收集 *.py，跳过垃圾目录与隐藏目录。"""
    files = []
    for p in sorted(root.rglob("*.py")):
        parents = p.relative_to(root).parts[:-1]
        if any(part in SKIP_DIR_NAMES or part.startswith(".") for part in parents):
            continue
        files.append(p)
    return files


def load_sources(
    files: list[Path], root: Path, root_name: str
) -> dict[str, tuple[ast.Module | None, str, str]]:
    """一次性读取并 AST 解析全部文件，返回 {rel: (tree, text, err_msg)}。

    用 tokenize.open 按 PEP 263 coding cookie 自动解码（替代硬编码
    utf-8 读取，避免非 utf-8 源码 docstring 乱码）；解析失败时 tree 为
    None，err_msg 记录失败原因。
    """
    cache: dict[str, tuple[ast.Module | None, str, str]] = {}
    for p in files:
        rel = f"{root_name}/{p.relative_to(root).as_posix()}"
        try:
            with tokenize.open(p) as fh:
                text = fh.read()
        except (OSError, SyntaxError, LookupError, ValueError):
            text = p.read_text(encoding="utf-8", errors="replace")
        err_msg = ""
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            tree = None
            err_msg = exc.msg
        cache[rel] = (tree, text, err_msg)
    return cache


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def extract_docstring(tree: ast.Module) -> str:
    """模块 docstring 首个非空行，截断 80 字。"""
    doc = ast.get_docstring(tree) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return truncate(line, DOC_TRUNC)
    return ""


def count_toplevel(tree: ast.Module) -> tuple[int, int]:
    funcs = classes = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs += 1
        elif isinstance(node, ast.ClassDef):
            classes += 1
    return funcs, classes


def rel_dotted(rel: str) -> str:
    """相对路径 -> 点分模块名。quant_system/data/__init__.py -> quant_system.data"""
    if rel.endswith("__init__.py"):
        return rel[: -len("__init__.py")].rstrip("/").replace("/", ".")
    return rel[:-3].replace("/", ".")


def build_module_map(files: list[Path], root: Path, root_name: str) -> dict[str, set[str]]:
    """点分模块名 -> 存在的相对模块路径集合（本仓可被 import 的目标）。"""
    module_map: dict[str, set[str]] = collections.defaultdict(set)
    for p in files:
        rel = f"{root_name}/{p.relative_to(root).as_posix()}"
        module_map[rel_dotted(rel)].add(rel)
    return module_map


def package_dotted(importer_dotted: str, is_package_init: bool) -> list[str]:
    """当前文件所在包的点分名（相对 import 的基准）。"""
    parts = importer_dotted.split(".")
    if not is_package_init:
        parts = parts[:-1]
    return parts


def resolve_dotted(dotted: str, module_map: dict[str, set[str]]) -> set[str]:
    return module_map.get(dotted, set())


def count_imports(
    cache: dict[str, tuple[ast.Module | None, str, str]],
    module_map: dict[str, set[str]],
) -> collections.Counter:
    """全仓 import 引用计数: 相对模块路径 -> 引用它的文件数（按文件去重）。

    语义与审计一致:
      - `from X import name`: 计入解析到的 X 及其子模块 X.name（若存在）
      - `import X[.Y]`: 计入解析到的模块
    """
    counter: collections.Counter = collections.Counter()

    for rel, (tree, _, _) in cache.items():
        if tree is None:
            continue
        is_init = rel.endswith("__init__.py")
        base_parts = package_dotted(rel_dotted(rel), is_init)
        targets: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    targets |= resolve_dotted(alias.name, module_map)
            elif isinstance(node, ast.ImportFrom):
                level = node.level or 0
                if level > 0:
                    parts = list(base_parts)
                    for _ in range(level - 1):
                        if parts:
                            parts.pop()
                    if node.module:
                        parts.extend(node.module.split("."))
                    x_dotted = ".".join(parts)
                else:
                    x_dotted = node.module or ""
                targets |= resolve_dotted(x_dotted, module_map)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    targets |= resolve_dotted(f"{x_dotted}.{alias.name}", module_map)
        for target in targets:
            counter[target] += 1
    return counter


def normalize_audit_path(path: str, scanned_rels: set[str]) -> str:
    """审计 JSON 的 file 路径归一化为脚本统一格式（相对路径含 root 前缀）。

    处理反斜杠、`./`、前导 `/`，以及绝对路径中嵌有 `quant_system/`
    / `scripts/` 前缀的形态；仍无法识别时按后缀匹配已扫描模块。
    """
    p = (path or "").replace("\\", "/").strip().lstrip("/")
    while p.startswith("./"):
        p = p[2:]
    for root_name in SCAN_ROOTS:
        prefix = f"{root_name}/"
        if p == root_name or p.startswith(prefix):
            return p
        idx = p.find(prefix)
        if idx >= 0:
            return p[idx:]
    for rel in scanned_rels:
        if p == rel or p.endswith(f"/{rel}"):
            return rel
    return p


def load_dup_map(
    audit_path: Path, scanned_rels: set[str]
) -> tuple[dict[str, set[str]], dict[str, int], set[str]]:
    """复用审计 JSON 的 duplicate_function 规则。

    返回:
      dup_funcs_per_file: 相对模块路径 -> 该模块参与的重复函数名集合
      dup_group_sizes:    函数名 -> 跨文件出现数
      unmatched:          审计中出现但未匹配到扫描模块的路径集合
    """
    dup_funcs_per_file: dict[str, set[str]] = collections.defaultdict(set)
    dup_group_sizes: dict[str, int] = {}
    unmatched: set[str] = set()
    if not audit_path.exists():
        return dup_funcs_per_file, dup_group_sizes, unmatched
    try:
        data = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dup_funcs_per_file, dup_group_sizes, unmatched
    if not isinstance(data, dict):
        return dup_funcs_per_file, dup_group_sizes, unmatched
    issues = data.get("issues")
    if not isinstance(issues, list):
        return dup_funcs_per_file, dup_group_sizes, unmatched

    groups: dict[str, set[str]] = collections.defaultdict(set)
    fn_re = re.compile(r"重复实现:\s*([A-Za-z_]\w*)\(")
    peer_re = re.compile(r"与\s+(\S+\.py):\d+")
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        if issue.get("rule") != RULE_DUP:
            continue
        msg = issue.get("msg", "")
        if not isinstance(msg, str):
            continue
        m = fn_re.search(msg)
        if not m:
            continue  # 匹配不到函数名则跳过该条，避免 '?' 伪重复组
        fn = m.group(1)
        file_path = normalize_audit_path(issue.get("file", ""), scanned_rels)
        if not file_path:
            continue
        groups[fn].add(file_path)
        for peer in peer_re.findall(msg):  # 取全部 peer，而非仅首个
            peer_path = normalize_audit_path(peer, scanned_rels)
            if peer_path:
                groups[fn].add(peer_path)
    for fn, files in groups.items():
        files = {f for f in files if f}
        if len(files) < 2:
            continue
        dup_group_sizes[fn] = len(files)
        for f in files:
            dup_funcs_per_file[f].add(fn)
            if f not in scanned_rels:
                unmatched.add(f)
    return dup_funcs_per_file, dup_group_sizes, unmatched


def group_of(rel: str) -> str:
    """按目录分组: quant_system 顶层 / quant_system/analysis_core / scripts 顶层 ..."""
    parts = rel.split("/")
    if len(parts) == 2:
        return f"{parts[0]} 顶层"
    return f"{parts[0]}/{parts[1]}"


def build_rows(
    cache: dict[str, tuple[ast.Module | None, str, str]],
    imports: collections.Counter,
    dup_funcs_per_file: dict[str, set[str]],
    dup_group_sizes: dict[str, int],
) -> list[dict]:
    rows = []
    for rel, (tree, text, err_msg) in cache.items():
        lines = len(text.splitlines())
        doc = ""
        funcs = classes = 0
        if tree is not None:
            doc = extract_docstring(tree)
            funcs, classes = count_toplevel(tree)
        else:
            doc = f"⚠ 语法解析失败: {err_msg}"
        dup_fns = sorted(dup_funcs_per_file.get(rel, set()))
        dup_mark = ", ".join(f"{fn}×{dup_group_sizes[fn]}" for fn in dup_fns[:3])
        if len(dup_fns) > 3:
            dup_mark += f" +{len(dup_fns) - 3} 项"
        rows.append(
            {
                "rel": rel,
                "name": rel.split("/")[-1],
                "lines": lines,
                "doc": doc or "—",
                "funcs": funcs,
                "classes": classes,
                "imports": imports.get(rel, 0),
                "dup_mark": dup_mark or "—",
                "dup_fns": dup_fns,
            }
        )
    rows.sort(key=lambda r: r["name"])
    return rows


def render_doc(
    rows_by_group: list[tuple[str, list[dict]]],
    totals: dict,
    top_modules: list[dict],
    top_dups: list[tuple[str, int, list[str]]],
    audit_used: bool,
    skipped_dirs: tuple[str, ...],
) -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 模块索引（MODULE_MAP）",
        "",
        "> 自动生成: `scripts/generate_module_map.py`（纯 AST 静态扫描，不执行被扫描代码）",
        f"> 生成时间: {now}",
        f"> 扫描范围: `quant_system/` + `scripts/` 全部 *.py；跳过目录: {', '.join(skipped_dirs)} 及隐藏目录",
        f"> 重复标记来源: `generated/audit/static_scan_2026-08-11.json` duplicate_function 规则{'（已复用）' if audit_used else '（未找到，跳过）'}",
        "> 列说明: 函数列 = 顶层函数数/顶层类数；被引用 = 引用该模块的文件数（全仓按文件去重）；重复标记 = 跨文件同名函数×涉及文件数",
        "",
        "## 总览",
        "",
        "| 统计项 | 数值 |",
        "|---|---|",
        f"| 模块总数 | {totals['modules']} |",
        f"| 总行数 | {totals['lines']} |",
        f"| 顶层函数总数 | {totals['funcs']} |",
        f"| 顶层类总数 | {totals['classes']} |",
        f"| 被引用总次数 | {totals['imports']} |",
        f"| 重复函数组数 | {totals['dup_groups']} |",
        f"| 涉及重复模块数 | {totals['dup_files']} |",
        f"| 语法解析失败 | {totals['parse_errors']} |",
        "",
    ]
    for group, rows in rows_by_group:
        lines += [
            f"## {group}",
            "",
            "| 模块 | 行数 | 职责 | 函数 | 被引用 | 重复标记 |",
            "|---|---|---|---|---|---|",
        ]
        for r in rows:
            doc = r["doc"].replace("|", "\\|").replace("\n", " ")
            dup = r["dup_mark"].replace("|", "\\|")
            lines.append(
                f"| {r['name']} | {r['lines']} | {doc} | {r['funcs']}/{r['classes']} | {r['imports']} | {dup} |"
            )
        lines.append("")
    lines += [
        "## 统计摘要",
        "",
        f"- 总模块: **{totals['modules']}** 个",
        f"- 总行数: **{totals['lines']}** 行",
        f"- 顶层函数/类: {totals['funcs']} / {totals['classes']}",
        f"- 被引用总次数: {totals['imports']}",
        f"- 重复函数组: {totals['dup_groups']} 组（涉及 {totals['dup_files']} 个模块）",
        "",
        f"### TOP{TOP_MODULES} 最大模块（按行数）",
        "",
        "| # | 模块 | 行数 | 职责 | 函数/类 | 被引用 |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(top_modules, 1):
        doc = truncate(r["doc"], 60).replace("|", "\\|")
        lines.append(
            f"| {i} | {r['rel']} | {r['lines']} | {doc} | {r['funcs']}/{r['classes']} | {r['imports']} |"
        )
    lines += [
        "",
        f"### 重复函数 TOP 清单（{len(top_dups)} 组）",
        "",
        "| 函数 | 跨文件数 | 示例模块 |",
        "|---|---|---|",
    ]
    for fn, size, examples in top_dups:
        lines.append(f"| `{fn}()` | {size} | {', '.join(examples[:4])} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成模块索引 MODULE_MAP.md（D1 收敛前置）")
    parser.add_argument("--out", type=Path, default=None, help="输出 Markdown 路径（默认 项目文档/量化交易系统/MODULE_MAP.md）")
    parser.add_argument("--audit-json", type=Path, default=AUDIT_JSON, help="静态审计 JSON（duplicate_function 规则来源）")
    args = parser.parse_args()

    # 一次性收集文件并读取/解析（缓存复用，避免重复 collect/parse）
    root_files: list[tuple[str, Path, list[Path]]] = []
    cache: dict[str, tuple[ast.Module | None, str, str]] = {}
    for root_name in SCAN_ROOTS:
        root = ROOT / root_name
        files = collect_py_files(root)
        root_files.append((root_name, root, files))
        cache.update(load_sources(files, root, root_name))

    scanned_rels = set(cache)
    dup_funcs_per_file, dup_group_sizes, unmatched = load_dup_map(
        args.audit_json, scanned_rels
    )
    if unmatched:
        shown = ", ".join(sorted(unmatched)[:5])
        suffix = " ..." if len(unmatched) > 5 else ""
        print(
            f"警告: 审计重复路径 {len(unmatched)} 个未匹配扫描模块（已忽略）: {shown}{suffix}",
            file=sys.stderr,
        )

    module_map: dict[str, set[str]] = {}
    for root_name, root, files in root_files:
        module_map.update(
            build_module_map(files, root, root_name)
        )
    imports: collections.Counter = count_imports(cache, module_map)
    all_rows = build_rows(cache, imports, dup_funcs_per_file, dup_group_sizes)

    # 按目录分组
    grouped: dict[str, list[dict]] = collections.OrderedDict()
    for r in all_rows:
        grouped.setdefault(group_of(r["rel"]), []).append(r)
    order = sorted(
        grouped.keys(),
        key=lambda g: (g.split("/", 1)[0].split(" ", 1)[0] != "quant_system", g),
    )
    rows_by_group = [(g, grouped[g]) for g in order]

    totals = {
        "modules": len(all_rows),
        "lines": sum(r["lines"] for r in all_rows),
        "funcs": sum(r["funcs"] for r in all_rows),
        "classes": sum(r["classes"] for r in all_rows),
        "imports": sum(imports.values()),
        "dup_groups": len(dup_group_sizes),
        "dup_files": len({r["rel"] for r in all_rows if r["dup_fns"]}),
        "parse_errors": sum(1 for r in all_rows if r["doc"].startswith("⚠")),
    }
    top_modules = sorted(all_rows, key=lambda r: -r["lines"])[:TOP_MODULES]
    top_dups = sorted(
        dup_group_sizes.items(), key=lambda kv: (-kv[1], kv[0])
    )[:TOP_DUPS]
    files_per_dup_func: dict[str, set[str]] = collections.defaultdict(set)
    for f, fns in dup_funcs_per_file.items():
        for fn in fns:
            files_per_dup_func[fn].add(f)
    top_dups_rendered = [
        (fn, size, sorted(files_per_dup_func.get(fn, set()) & scanned_rels))
        for fn, size in top_dups
    ]

    audit_used = args.audit_json.exists()
    doc = render_doc(
        rows_by_group, totals, top_modules, top_dups_rendered, audit_used,
        tuple(sorted(SKIP_DIR_NAMES)),
    )

    out = args.out or DEFAULT_OUT
    fallback_note = ""
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(doc, encoding="utf-8")
    except OSError as exc:
        out = TMP_FALLBACK
        out.write_text(doc, encoding="utf-8")
        fallback_note = f"（目标路径不可写: {exc}，已改输出到 {out}）"

    print(f"模块索引已生成 → {out}{fallback_note}")
    print(f"统计: 模块 {totals['modules']} | 总行数 {totals['lines']} | "
          f"顶层函数 {totals['funcs']} | 类 {totals['classes']} | 被引用 {totals['imports']}")
    print(f"重复函数 TOP{min(5, len(top_dups_rendered))}:")
    for fn, size, _ in top_dups_rendered[:5]:
        print(f"  {fn}() ×{size} 个文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
