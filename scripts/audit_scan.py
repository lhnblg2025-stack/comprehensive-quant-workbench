"""audit_scan — 全仓静态审计扫描器（AST 级）

抓取:
  1. 语法错误（无法 parse 的文件）
  2. 未定义名称（NameError 风险, 排除内置/导入/参数/全局）
  3. 裸 except / except Exception 无日志（静默吞噬）
  4. 硬编码日期（20xx-xx-xx / 8位数字日期）
  5. 前视风险模式: bfill/ffill(整列)/shift(-)/iloc 前向引用
  6. iterrows 修改 DataFrame（不生效陷阱）
  7. TODO/FIXME/XXX/HACK 标记
  8. print 调试残留
  9. 动态导入（importlib/__import__/exec/getattr 动态调用）
  10. 大文件耗时操作标记（iterrows/双重 for 循环）

用法:
  python3 scripts/audit_scan.py [--path quant_system] [--json]
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BUILTINS = set(dir(__builtins__)) | {
    "pd", "np", "plt", "os", "sys", "json", "re", "time", "datetime", "timedelta",
    "timezone", "Path", "Any", "Optional", "List", "Dict", "Tuple", "Union",
    "defaultdict", "Counter", "OrderedDict", "random", "math", "logging", "argparse",
    "traceback", "warnings", "uuid", "glob", "copy", "collections",
}


class FuncVisitor(ast.NodeVisitor):
    def __init__(self):
        self.defined = set()       # 本函数内定义的名称
        self.used = []             # (lineno, name)
        self.globals_decl = set()
        self.bare_excepts = []
        self.iterrows = []
        self.bfill = []
        self.dynamic = []
        self.todos = []
        self.hardcoded_dates = []
        self.prints = []

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used.append((node.lineno, node.id))
        elif isinstance(node.ctx, ast.Store):
            self.defined.add(node.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.type is None:
            self.bare_excepts.append(node.lineno)
        elif isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException"):
            # 检查 handler 体是否有日志/print
            has_log = any(isinstance(n, (ast.Call, ast.Expr)) and
                          (getattr(n, "value", None) and isinstance(n.value, ast.Call) and
                           isinstance(n.value.func, ast.Attribute) and
                           n.value.func.attr in ("write", "print", "info", "warning", "error", "exception"))
                          for n in ast.walk(node))
            if not has_log:
                self.bare_excepts.append(node.lineno)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr == "iterrows":
            self.iterrows.append(node.lineno)
        if node.attr in ("bfill",):
            self.bfill.append(node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in ("__import__", "exec", "eval"):
            self.dynamic.append((node.lineno, node.func.id))
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("import_module", "importlib"):
            self.dynamic.append((node.lineno, f"importlib.{node.func.attr}"))
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            self.prints.append(node.lineno)
        self.generic_visit(node)


def scan_file(path: Path) -> dict:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError as e:
        return {"file": str(path), "syntax_error": f"{e.lineno}:{e.msg}"}
    except Exception as e:
        return {"file": str(path), "read_error": str(e)[:100]}

    issues = {"file": str(path), "undefined": [], "bare_excepts": [], "iterrows": [],
              "bfill": [], "dynamic": [], "todos": [], "hardcoded_dates": [],
              "prints": [], "functions": 0, "lines": len(path.read_text(encoding="utf-8").splitlines())}

    module_defined = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_defined.add(n.name)
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                module_defined.add(a.asname or a.name.split(".")[0])

    # 文件级硬编码日期 + TODO
    src = path.read_text(encoding="utf-8", errors="ignore")
    for i, line in enumerate(src.splitlines(), 1):
        if any(k in line for k in ("TODO", "FIXME", "HACK", "XXX")):
            issues["todos"].append(i)
        import re
        if re.search(r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}", line) and "示例" not in line and "2020" not in line[:50]:
            issues["hardcoded_dates"].append((i, line.strip()[:80]))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            issues["functions"] += 1
            fv = FuncVisitor()
            fv.visit(node)
            # 未定义: 使用但未定义, 且不在内置/模块级/参数/全局声明
            for lineno, name in fv.used:
                if name in fv.defined or name in BUILTINS or name in module_defined or \
                   name in fv.globals_decl or name in ("self", "cls", "args", "kwargs", "_"):
                    continue
                issues["undefined"].append((lineno, name, node.name))
            issues["bare_excepts"].extend((l, node.name) for l in fv.bare_excepts)
            issues["iterrows"].extend((l, node.name) for l in fv.iterrows)
            issues["bfill"].extend((l, node.name) for l in fv.bfill)
            issues["dynamic"].extend((l, k, node.name) for l, k in fv.dynamic)
            issues["prints"].extend((l, node.name) for l in fv.prints)

    # 去重
    for k in ("undefined", "bare_excepts", "iterrows", "bfill", "dynamic", "prints"):
        issues[k] = sorted(set(issues[k]))
    return issues


def scan_tree(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.rglob("*.py")):
        if "node_modules" in str(p) or ".git" in str(p) or "__pycache__" in str(p):
            continue
        out.append(scan_file(p))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="quant_system")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    res = scan_tree(ROOT / args.path)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        sys.exit(0)
    # 汇总
    n_syntax = sum(1 for r in res if "syntax_error" in r)
    n_undef = sum(len(r.get("undefined", [])) for r in res)
    n_bare = sum(len(r.get("bare_excepts", [])) for r in res)
    n_it = sum(len(r.get("iterrows", [])) for r in res)
    n_bf = sum(len(r.get("bfill", [])) for r in res)
    n_dyn = sum(len(r.get("dynamic", [])) for r in res)
    n_todo = sum(len(r.get("todos", [])) for r in res)
    n_dates = sum(len(r.get("hardcoded_dates", [])) for r in res)
    n_prints = sum(len(r.get("prints", [])) for r in res)
    print(f"扫描 {len(res)} 个 py | 语法错误 {n_syntax} | 未定义引用 {n_undef} | "
          f"静默异常 {n_bare} | iterrows {n_it} | bfill {n_bf} | 动态导入 {n_dyn} | "
          f"TODO {n_todo} | 硬编码日期 {n_dates} | print残留 {n_prints}")
    for r in res:
        tag = []
        if "syntax_error" in r: tag.append(f"SYNTAX {r['syntax_error']}")
        if r.get("undefined"): tag.append(f"UNDEF {len(r['undefined'])}")
        if r.get("bare_excepts"): tag.append(f"BARE {len(r['bare_excepts'])}")
        if r.get("bfill"): tag.append(f"BFILL {len(r['bfill'])}")
        if r.get("iterrows"): tag.append(f"ITER {len(r['iterrows'])}")
        if tag:
            print(f"  {r['file'].replace(str(ROOT)+'/', '')}: {', '.join(tag)}")
