#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_static_scan — 全系统 L1 静态审计扫描器（AST 级，不执行被扫描代码）

设计依据: 项目文档/量化交易系统/2026-08-11_全系统审计方案.md（L1 静态扫描章节）

扫描范围: 递归扫描 quant_system/ 与 scripts/ 下全部 *.py（跳过 __pycache__）。
实现: 纯 AST + 标准库（ast/pathlib/json/datetime/re/time），不 import、不执行被扫描模块。

扫描项（分级 critical/major/minor/info）:
  a. 未定义引用: 收集模块内所有 Name 使用，减去 import/builtin/参数/局部赋值后报可疑
  b. 硬编码易腐值: 字符串/数字字面量含年份(2015-2029)、IP 地址、
     eastmoney/10jqka/baostock 域名、push2 残留
  c. 吞异常: except (Exception|BaseException|裸 except) 的 body 仅 pass/continue（无日志）
  d. 前视偏差模式（启发式标记 + 建议人工确认）: shift(-、bfill、method='bfill'、
     sort(ascending=False) 后取行、因子/标签上下文内 iloc[负索引]
  e. 超长结构: 函数 >500 行、文件 >2000 行
  f. 重复实现: 不同文件顶层函数同名 + 同参数数量（纯转发薄壳除外，见 thin_shell）
  g. 生产代码 print(（排除 __main__/argparse 块与注释）、TODO/FIXME 注释、
     eval(/exec(/os.system(
  h. 未使用 import
  i. quant_system 内互相 import 计数（JSON 输出每模块被引用次数）

级别约定:
  major  未定义引用 / 硬编码端点(IP|域名|push2) / 吞异常 / 前视偏差(强信号) /
         eval|exec|os.system / 生产代码 print / 语法解析失败
  minor  硬编码年份 / 未使用 import / 重复实现 / TODO/FIXME / 超长结构 /
         sort 降序标记 / scripts 内 print
  info   非因子上下文 iloc[负索引] / 星号导入提示 / 薄壳转发 thin_shell
  critical 保留（当前无扫描项可确凿判定为致命；未定义引用为 AST 启发式归入 major）

薄壳转发 (thin_shell): 函数体（去 docstring 后）仅 return 调用同仓公共函数
  （utils.py 或其他扫描模块），或 docstring 标注 D1-D6 收敛且仅转发同仓公共函数
  （含 import 失败兜底保留原实现的情形）。此类函数与同仓同名同签名函数共享实现，
  不计为重复实现（降级为 info；真正重复仅统计双方均为完整实现者）。

输出: generated/audit/static_scan_{date}.json（含 scan_time/文件数/行数/问题列表/
     模块元信息/import 图谱），并打印人类可读摘要（按 level 汇总 + TOP10 问题文件）。

用法:
  python3 scripts/audit_static_scan.py
  python3 scripts/audit_static_scan.py --root quant_system --out /tmp/scan.json
"""

from __future__ import annotations

import argparse
import ast
import builtins as _builtins
import collections
import datetime as _dt
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("quant_system", "scripts")
OUTPUT_DIR = ROOT / "generated" / "audit"

YEAR_MIN, YEAR_MAX = 2015, 2029

STR_YEAR_RE = re.compile(r"(?<!\d)(20(?:1[5-9]|2[0-9]))")
IPV4_RE = re.compile(r"(?<!\d)(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?!\d)")
DOMAIN_RE = re.compile(r"eastmoney|10jqka|baostock|push2", re.IGNORECASE)
TODO_RE = re.compile(r"\b(TODO|FIXME)\b")
CONTEXT_RE = re.compile(r"factor|ic|label|feature|signal", re.IGNORECASE)

LEVELS = ("critical", "major", "minor", "info")

BUILTIN_NAMES = (
    set(dir(_builtins))
    | {
        "__name__", "__file__", "__doc__", "__package__", "__loader__", "__spec__",
        "__builtins__", "__annotations__", "__cached__", "__debug__", "__class__",
        "__qualname__", "__module__", "__dict__", "__all__", "__future__",
    }
)

_SORT_ATTRS = ("sort_values", "sort_index")
_TAKE_ATTRS = ("head", "tail", "first", "last")
_STMT_NODES = (
    ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Return,
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith,
    ast.Try, ast.Raise, ast.Assert, ast.Delete, ast.Global, ast.Nonlocal,
    ast.Import, ast.ImportFrom, ast.Pass, ast.Break, ast.Continue,
) + ((ast.Match,) if hasattr(ast, "Match") else ())


# ---------------------------------------------------------------- 小工具

def read_text(path: Path):
    """读取源码文本，依次尝试 utf-8/gb18030/latin-1，保证不因编码中断扫描。"""
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def discover(root: Path):
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _collect_assigned(nodes):
    """收集某作用域内被绑定/赋值的名字（不进入嵌套作用域）。

    覆盖: 赋值/解包/for/with/except-as/import/def/class/walrus/match 捕获等。
    """
    names = set()

    def rec(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            return
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
            return
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.asname or a.name.split(".")[0])
            return
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name != "*":
                    names.add(a.asname or a.name)
            return
        if isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif getattr(ast, "MatchAs", None) is not None and isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        for child in ast.iter_child_nodes(node):
            rec(child)

    for n in nodes:
        rec(n)
    return names


def _collect_target(target, into):
    """收集推导式/解包目标名。"""
    if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
        into.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for e in target.elts:
            _collect_target(e, into)
    elif isinstance(target, ast.Starred):
        _collect_target(target.value, into)


def _param_count(args):
    n = len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
    if args.vararg:
        n += 1
    if args.kwarg:
        n += 1
    return n


def _module_file_candidates(dotted):
    """点分模块名 -> 仓库内可能的 .py 相对路径（模块文件或包 __init__.py）。"""
    p = "/".join(dotted.split("."))
    return {p + ".py", p + "/__init__.py"}


def _module_dotted(rel):
    """模块相对路径 -> 点分模块名（quant_system/utils.py -> quant_system.utils）。"""
    if rel.endswith("__init__.py"):
        rel = rel[: -len("/__init__.py")]
    else:
        rel = rel[: -len(".py")]
    return rel.replace("/", ".")


def _build_import_map(tree, rel, repo_files):
    """模块内 import 别名 -> {(目标模块相对路径, 公共函数名或 None)}。

    func 为 None 表示别名绑定的是模块本身（后续经属性访问），如 utils.safe_float；
    否则别名绑定该模块的公共函数，如 from quant_system.utils import safe_float as _impl。
    rel 为当前文件相对路径，用于把相对导入（node.level）折算到当前文件所在包。
    """
    mapping = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                alias = a.asname or a.name.split(".")[0]
                # 无 as 时别名绑定顶层包/模块（import a.b → a），属性链后续再叠加；
                # 有 as 时别名直接绑定完整模块（import a.b as x → x 指向 a.b）。
                mod_name = a.name if a.asname else alias
                for cand in _module_file_candidates(mod_name):
                    if cand in repo_files:
                        mapping.setdefault(alias, set()).add((cand, None))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                dir_parts = rel.split("/")[:-1]
                up = node.level - 1
                if up:
                    dir_parts = dir_parts[:-up] if len(dir_parts) >= up else []
                mod_parts = node.module.split(".") if node.module else []
                base = ".".join(dir_parts + mod_parts)
            else:
                base = node.module or ""
            for a in node.names:
                if a.name == "*":
                    continue
                alias = a.asname or a.name
                sub_cands = _module_file_candidates((base + "." + a.name) if base else a.name)
                if sub_cands & repo_files:
                    for cand in sub_cands & repo_files:
                        mapping.setdefault(alias, set()).add((cand, None))
                    continue
                for cand in _module_file_candidates(base) & repo_files:
                    mapping.setdefault(alias, set()).add((cand, a.name))
    return mapping


def _resolve_forward_target(callee, import_map, module_funcs, repo_files):
    """被调用对象是否指向同仓模块的公共函数；是则返回 'module.func'，否则 None。"""
    if isinstance(callee, ast.Name):
        for mod_rel, func in import_map.get(callee.id, ()):
            if func and not func.startswith("_") and func in module_funcs.get(mod_rel, ()):
                return f"{_module_dotted(mod_rel)}.{func}"
        return None
    if isinstance(callee, ast.Attribute):
        parts = []
        node = callee
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        parts.reverse()
        for mod_rel, func in import_map.get(node.id, ()):
            if func is not None:
                continue
            mod_dotted = _module_dotted(mod_rel)
            func_name = parts[-1]
            for cand in _module_file_candidates(".".join([mod_dotted] + parts[:-1])):
                if cand in repo_files and not func_name.startswith("_") \
                        and func_name in module_funcs.get(cand, ()):
                    return ".".join([mod_dotted] + parts[:-1] + [func_name])
        return None
    return None


_CONVERGED_RE = re.compile(r"D\s*[1-6]\s*收敛")
_FORWARD_HINTS = ("转发", "复用", "回退")


def _thin_shell_info(node, import_map, module_funcs, repo_files):
    """判定'纯转发薄壳'，返回 (is_shell, 转发目标描述或 None)。

    两种形态:
      A. 函数体（去 docstring 后）仅一个 return，直接调用同仓模块公共函数:
         return _safe_float_impl(...) / return utils.safe_float(...) /
         return _to_float_impl(...)
      B. D1-D6 收敛 wrapper（docstring 标注收敛 + 转发/复用/回退），体内含转发调用，
         如 import 失败兜底保留原实现的 _safe_float / now_cst。
    """
    body = node.body
    doc = None
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        doc = body[0].value.value
        body = body[1:]
    if len(body) == 1 and isinstance(body[0], ast.Return) \
            and isinstance(body[0].value, ast.Call):
        target = _resolve_forward_target(body[0].value.func, import_map, module_funcs, repo_files)
        if target:
            return True, target
    if doc and _CONVERGED_RE.search(doc) and any(k in doc for k in _FORWARD_HINTS):
        for stmt in body:
            for n in ast.walk(stmt):
                if isinstance(n, ast.Call):
                    target = _resolve_forward_target(n.func, import_map, module_funcs, repo_files)
                    if target:
                        return True, target
    return False, None


def _is_neg_const(node):
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float)))


def _is_main_guard(stmt):
    return (isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Compare)
            and len(stmt.test.ops) == 1
            and isinstance(stmt.test.ops[0], ast.Eq)
            and isinstance(stmt.test.left, ast.Name)
            and stmt.test.left.id == "__name__"
            and len(stmt.test.comparators) == 1
            and isinstance(stmt.test.comparators[0], ast.Constant)
            and stmt.test.comparators[0].value == "__main__")


def _uses_argparse(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in ("argparse", "ArgumentParser"):
            return True
    return False


def _sort_desc_node(node):
    """是否为 sort_values/sort_index 且显式 ascending=False 的调用。"""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr not in _SORT_ATTRS:
        return False
    for kw in node.keywords:
        if (kw.arg == "ascending" and isinstance(kw.value, ast.Constant)
                and kw.value.value is False):
            return True
    return False


def _take_node(node):
    """是否为"取行"模式: head/tail/first/last 调用或下标 0 / 负索引。"""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr in _TAKE_ATTRS
    if isinstance(node, ast.Subscript):
        sl = node.slice
        elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
        return any(
            (isinstance(e, ast.Constant) and e.value == 0) or _is_neg_const(e)
            for e in elts
        )
    return False


def _file_candidates(parts):
    if not parts:
        return set()
    base = "/".join(parts)
    return {base + ".py", base + "/__init__.py"}


def _resolve_import_targets(rel, node):
    """把 import 节点解析为可能的 quant_system 内模块相对路径（用于 import 图谱）。"""
    targets = set()
    if isinstance(node, ast.Import):
        for a in node.names:
            parts = a.name.split(".")
            if parts and parts[0] == "quant_system":
                targets |= _file_candidates(parts)
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            dir_parts = rel.split("/")[:-1]
            up = node.level - 1
            if up:
                dir_parts = dir_parts[:-up] if len(dir_parts) >= up else []
            mod_parts = node.module.split(".") if node.module else []
            base = dir_parts + mod_parts
            targets |= _file_candidates(base)
            for a in node.names:
                if a.name != "*":
                    targets |= _file_candidates(base + a.name.split("."))
        else:
            mod_parts = (node.module or "").split(".")
            if mod_parts and mod_parts[0] == "quant_system":
                targets |= _file_candidates(mod_parts)
                for a in node.names:
                    if a.name != "*":
                        targets |= _file_candidates(mod_parts + a.name.split("."))
    return targets


def _collect_all_names(tree):
    names = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and node.targets
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "__all__"
                and isinstance(node.value, (ast.List, ast.Tuple))):
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    names.add(elt.value)
    return names


def _comment_todo(line):
    """在行内注释中查找 TODO/FIXME（跳过字符串字面量里的 '#'）。"""
    in_s = in_d = esc = False
    for i, ch in enumerate(line):
        if in_s:
            if ch == "\\" and not esc:
                esc = True
                continue
            if ch == "'" and not esc:
                in_s = False
            esc = False
        elif in_d:
            if ch == "\\" and not esc:
                esc = True
                continue
            if ch == '"' and not esc:
                in_d = False
            esc = False
        else:
            if ch == "'":
                in_s = True
            elif ch == '"':
                in_d = True
            elif ch == "#":
                return TODO_RE.search(line[i + 1:])
    return None


# ---------------------------------------------------------------- 作用域

class _Scope:
    __slots__ = ("parent", "kind", "assigned")

    def __init__(self, parent, kind):
        self.parent = parent
        self.kind = kind
        self.assigned = set()


# ---------------------------------------------------------------- 单文件扫描器

class _ModuleWalker(ast.NodeVisitor):
    """单文件 AST 扫描器: 未定义引用 / 字面量 / 吞异常 / 前视模式 / 危险调用 / print。"""

    def __init__(self, module_rel, emit, is_production):
        self.module_rel = module_rel
        self.emit = emit
        self.is_production = is_production
        self.stack = []                 # 作用域链，stack[0] 为 module 作用域
        self.factor_context = 0         # 是否处于因子/标签构建上下文
        self.print_exempt = 0           # __main__/argparse 上下文内 print 豁免
        self.used_names = set()
        self.func_count = 0
        self.longest = None             # (name, lineno, lines)
        self.star_import = False
        self.imports = []               # ast.Import / ast.ImportFrom 节点列表
        self._seen = set()

    def emit1(self, rule, lineno, level, msg):
        key = (rule, lineno, msg)
        if key in self._seen:
            return
        self._seen.add(key)
        self.emit(rule, lineno, level, msg)

    # ---- 作用域解析 ----

    def _push(self, scope):
        self.stack.append(scope)

    def _pop(self):
        self.stack.pop()

    def _resolve(self, name):
        for scope in reversed(self.stack):
            if name in scope.assigned:
                return True
        return name in BUILTIN_NAMES

    # ---- 模块 ----

    def visit_Module(self, node):
        scope = _Scope(None, "module")
        scope.assigned |= _collect_assigned(node.body)
        self.stack.append(scope)
        for stmt in node.body:
            self.visit(stmt)
        self._scan_sort_take(node.body)

    # ---- 函数 / 类 / lambda / 推导式 ----

    def _visit_function(self, node):
        args = node.args
        for d in node.decorator_list:
            self.visit(d)
        for d in args.defaults:
            self.visit(d)
        for d in args.kw_defaults:
            if d is not None:
                self.visit(d)
        for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            if a.annotation is not None:
                self.visit(a.annotation)
        if args.vararg and args.vararg.annotation is not None:
            self.visit(args.vararg.annotation)
        if args.kwarg and args.kwarg.annotation is not None:
            self.visit(args.kwarg.annotation)
        if node.returns is not None:
            self.visit(node.returns)

        scope = _Scope(self.stack[-1], "function")
        for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            scope.assigned.add(a.arg)
        if args.vararg:
            scope.assigned.add(args.vararg.arg)
        if args.kwarg:
            scope.assigned.add(args.kwarg.arg)
        scope.assigned |= _collect_assigned(node.body)
        self._push(scope)

        self.func_count += 1
        end = getattr(node, "end_lineno", None) or node.lineno
        f_lines = end - node.lineno + 1
        if self.longest is None or f_lines > self.longest[2]:
            self.longest = (node.name, node.lineno, f_lines)
        if f_lines > 500:
            self.emit1("overlong_function", node.lineno, "minor",
                       f"函数 {node.name}() 超长: {f_lines} 行（>500，建议重构）")

        prev_factor = self.factor_context
        if CONTEXT_RE.search(node.name):
            self.factor_context += 1
        argparse_ctx = _uses_argparse(node)
        if argparse_ctx:
            self.print_exempt += 1
        for stmt in node.body:
            self.visit(stmt)
        self._scan_sort_take(node.body)
        if argparse_ctx:
            self.print_exempt -= 1
        self.factor_context = prev_factor
        self._pop()

    def visit_FunctionDef(self, node):
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._visit_function(node)

    def visit_Lambda(self, node):
        args = node.args
        for d in args.defaults:
            self.visit(d)
        for d in args.kw_defaults:
            if d is not None:
                self.visit(d)
        scope = _Scope(self.stack[-1], "lambda")
        for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            scope.assigned.add(a.arg)
        if args.vararg:
            scope.assigned.add(args.vararg.arg)
        if args.kwarg:
            scope.assigned.add(args.kwarg.arg)
        scope.assigned |= _collect_assigned([node.body])
        self._push(scope)
        self.visit(node.body)
        self._pop()

    def visit_ClassDef(self, node):
        for d in node.decorator_list:
            self.visit(d)
        for b in node.bases:
            self.visit(b)
        for k in node.keywords:
            self.visit(k)
        scope = _Scope(self.stack[-1], "class")
        scope.assigned |= _collect_assigned(node.body)
        self._push(scope)
        prev_factor = self.factor_context
        if "factor" in node.name.lower():
            self.factor_context += 1
        for stmt in node.body:
            self.visit(stmt)
        self._scan_sort_take(node.body)
        self.factor_context = prev_factor
        self._pop()

    def _visit_comprehension(self, node, elt):
        scope = _Scope(self.stack[-1], "comp")
        for gen in node.generators:
            _collect_target(gen.target, scope.assigned)
        if node.generators:
            self.visit(node.generators[0].iter)
        self._push(scope)
        for gen in node.generators:
            if gen is not node.generators[0]:
                self.visit(gen.iter)
            for cond in gen.ifs:
                self.visit(cond)
        elt()
        self._pop()

    def visit_ListComp(self, node):
        self._visit_comprehension(node, lambda: self.visit(node.elt))

    def visit_SetComp(self, node):
        self._visit_comprehension(node, lambda: self.visit(node.elt))

    def visit_GeneratorExp(self, node):
        self._visit_comprehension(node, lambda: self.visit(node.elt))

    def visit_DictComp(self, node):
        self._visit_comprehension(node, lambda: (self.visit(node.key), self.visit(node.value)))

    # ---- 名字 / import ----

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
            if not self.star_import and not self._resolve(node.id):
                self.emit1("undefined_name", node.lineno, "major",
                           f"未定义引用: {node.id}（AST 启发式，需人工确认）")
        self.generic_visit(node)

    def visit_Import(self, node):
        self.imports.append(node)

    def visit_ImportFrom(self, node):
        self.imports.append(node)
        for a in node.names:
            if a.name == "*":
                self.star_import = True
                self.emit1("star_import", node.lineno, "info",
                           "星号导入 import *（本模块未定义引用检查已跳过）")

    # ---- 吞异常 ----

    def visit_ExceptHandler(self, node):
        names = set()

        def add_type(n):
            if isinstance(n, ast.Name):
                names.add(n.id)
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
            elif isinstance(n, ast.Tuple):
                for e in n.elts:
                    add_type(e)

        if node.type is not None:
            add_type(node.type)
        swallow = bool(node.body) and all(
            isinstance(s, (ast.Pass, ast.Continue)) for s in node.body)
        if swallow and (node.type is None or names & {"Exception", "BaseException"}):
            self.emit1("swallow_exception", node.lineno, "major",
                       "吞异常: except 无日志，body 仅 pass/continue")
        self.generic_visit(node)

    # ---- 字面量（硬编码易腐值） ----

    def visit_Constant(self, node):
        v = node.value
        if isinstance(v, bool):
            pass
        elif isinstance(v, int):
            if YEAR_MIN <= v <= YEAR_MAX:
                self.emit1("hardcoded_year", node.lineno, "minor",
                           f"硬编码年份字面量: {v}（易腐值）")
            elif 20150100 <= v <= 20291299:
                self.emit1("hardcoded_year", node.lineno, "minor",
                           f"硬编码日期数字字面量: {v}（疑似 yyyymmdd，易腐值）")
            elif 201501 <= v <= 202912:
                self.emit1("hardcoded_year", node.lineno, "minor",
                           f"硬编码日期数字字面量: {v}（疑似 yyyymm，易腐值）")
        elif isinstance(v, str):
            m = STR_YEAR_RE.search(v)
            if m:
                self.emit1("hardcoded_year", node.lineno, "minor",
                           f"字符串含硬编码年份 {m.group(1)}: {v[:60]!r}")
            for mm in IPV4_RE.finditer(v):
                octets = [int(x) for x in mm.groups()]
                if all(0 <= o <= 255 for o in octets):
                    self.emit1("hardcoded_ip", node.lineno, "major",
                               f"硬编码 IP 地址: {mm.group(0)}")
            if DOMAIN_RE.search(v):
                self.emit1("hardcoded_domain", node.lineno, "major",
                           f"硬编码域名/push2 残留: {v[:80]!r}")
        self.generic_visit(node)

    # ---- 调用 / 下标（前视模式 + 危险调用 + print） ----

    @staticmethod
    def _is_bfill_value(v):
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return v.value == "bfill"
        if isinstance(v, ast.Name):
            return v.id == "bfill"
        if isinstance(v, ast.Attribute):
            return v.attr == "bfill"
        return False

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Name) and func.id in ("eval", "exec"):
            self.emit1("dangerous_call", node.lineno, "major",
                       f"危险动态执行: {func.id}(")
        elif (isinstance(func, ast.Attribute) and func.attr == "system"
              and isinstance(func.value, ast.Name) and func.value.id == "os"):
            self.emit1("dangerous_call", node.lineno, "major",
                       "危险命令执行: os.system(")
        if isinstance(func, ast.Name) and func.id == "print" and self.print_exempt == 0:
            self.emit1("print_production", node.lineno,
                       "major" if self.is_production else "minor",
                       "生产代码 print( 调用（非 __main__/argparse 上下文）")
        if isinstance(func, ast.Attribute):
            attr = func.attr
            if attr == "shift":
                neg = any(_is_neg_const(a) for a in node.args)
                neg = neg or any(kw.arg == "periods" and _is_neg_const(kw.value)
                                 for kw in node.keywords)
                if neg:
                    self.emit1("lookahead_shift", node.lineno, "major",
                               "前视偏差候选: shift(负周期)（建议人工确认）")
            elif attr == "bfill":
                self.emit1("lookahead_bfill", node.lineno, "major",
                           "前视偏差候选: bfill()（建议人工确认）")
            elif attr == "fillna":
                for kw in node.keywords:
                    if kw.arg == "method" and self._is_bfill_value(kw.value):
                        self.emit1("lookahead_bfill", node.lineno, "major",
                                   "前视偏差候选: fillna(method=bfill)（建议人工确认）")
            elif attr == "sort_values":
                for kw in node.keywords:
                    if (kw.arg == "ascending" and isinstance(kw.value, ast.Constant)
                            and kw.value.value is False):
                        self.emit1("lookahead_sort_desc", node.lineno, "minor",
                                   "前视偏差候选: sort_values(ascending=False)（建议人工确认）")
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Attribute) and node.value.attr == "iloc":
            sl = node.slice
            elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            if any(_is_neg_const(e) for e in elts):
                if self.factor_context:
                    self.emit1("lookahead_iloc_factor", node.lineno, "major",
                               "前视偏差候选: 因子/标签上下文内 iloc[负索引]（建议人工确认）")
                else:
                    self.emit1("lookahead_iloc", node.lineno, "info",
                               "iloc[负索引] 使用（非因子上下文，关注即可）")
        self.generic_visit(node)

    def visit_If(self, node):
        if _is_main_guard(node):
            self.print_exempt += 1
            self.generic_visit(node)
            self.print_exempt -= 1
        else:
            self.generic_visit(node)

    # ---- 排序后取行（同函数内顺序启发式） ----

    def _scan_sort_take(self, stmts):
        out = []

        def flatten(node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return
            if isinstance(node, _STMT_NODES):
                out.append(node)
            for child in ast.iter_child_nodes(node):
                flatten(child)

        for s in stmts:
            flatten(s)

        last_sort = -1
        for i, n in enumerate(out):
            sub = list(ast.walk(n))
            s = any(_sort_desc_node(x) for x in sub)
            t = any(_take_node(x) for x in sub)
            if s and t:
                self.emit1("lookahead_sort_take", n.lineno, "major",
                           "前视偏差候选: 降序排序后直接取行（建议人工确认）")
                last_sort = -1
            elif s:
                last_sort = i
            elif t and last_sort >= 0 and n.lineno > out[last_sort].lineno:
                self.emit1("lookahead_sort_take", n.lineno, "major",
                           "前视偏差候选: 降序排序后取行（建议人工确认）")
                last_sort = -1

    # ---- 未使用 import ----

    def unused_imports(self, all_strings, all_all):
        result = []
        for node in self.imports:
            if isinstance(node, ast.ImportFrom):
                if node.module == "__future__":
                    continue
                aliases = [a.asname or a.name for a in node.names if a.name != "*"]
            else:
                aliases = [a.asname or a.name.split(".")[0] for a in node.names]
            for alias in aliases:
                if alias in self.used_names or alias in all_all:
                    continue
                if alias in all_strings and re.search(r"\b" + re.escape(alias) + r"\b", all_strings):
                    continue
                result.append((node.lineno, alias))
        return result


# ---------------------------------------------------------------- 主流程

def _flatten_issues(issues):
    return sorted(issues, key=lambda x: (x["file"], x["line"], x["rule"]))


def print_summary(payload, top_files, out_str, quiet):
    if quiet:
        return
    s = payload["summary"]
    total = sum(s.values())
    print("=" * 64)
    print(f"L1 静态扫描摘要 | {payload['date']}")
    print("=" * 64)
    print(f"范围: {' + '.join(payload['roots'])} | 文件 {payload['files_scanned']} | "
          f"总行数 {payload['total_lines']:,}")
    print(f"耗时: {payload['duration_sec']} 秒")
    print(f"问题: 合计 {total}（critical={s['critical']} / major={s['major']} / "
          f"minor={s['minor']} / info={s['info']}）")
    thin = payload.get("rule_counts", {}).get("thin_shell", 0)
    print(f"薄壳转发 thin_shell: {thin}（info 级，仅转发同仓公共函数，不计重复）")
    if top_files:
        print("TOP10 问题文件:")
        width = max(len(f) for f, _ in top_files)
        for i, (f, c) in enumerate(top_files, 1):
            extra = f"critical {c['critical']} / " if c["critical"] else ""
            print(f"  {i:>2}. {f:<{width}}  合计 {c['total']:<4} "
                  f"{extra}major {c['major']} / minor {c['minor']} / info {c['info']}")
    if out_str:
        print(f"JSON 输出: {out_str}")
    elif not quiet:
        print("JSON 输出失败（目录只读？），完整问题清单如下:")
        for iss in payload["issues"]:
            print(f"  {iss['level']:<8} {iss['file']}:{iss['line']} "
                  f"[{iss['rule']}] {iss['msg']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="L1 全系统静态审计扫描器（AST 级，不执行被扫描代码）")
    ap.add_argument("--root", action="append", default=None,
                    help="扫描根目录（相对仓库根），可多次指定；默认: quant_system scripts")
    ap.add_argument("--out", default=None,
                    help="JSON 输出路径；默认 generated/audit/static_scan_{date}.json")
    ap.add_argument("--quiet", action="store_true", help="不打印人类可读摘要")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    roots = [Path(r) for r in (args.root or list(SCAN_ROOTS))]
    scan_date = _dt.date.today().strftime("%Y-%m-%d")
    out_path = Path(args.out) if args.out else OUTPUT_DIR / f"static_scan_{scan_date}.json"

    files_by_root = {}
    files = []
    for r in roots:
        rp = ROOT / r if not Path(r).is_absolute() else Path(r)
        fl = discover(rp)
        files_by_root[str(r)] = len(fl)
        files.extend(fl)
    files = sorted(set(files))

    repo_files = {p.relative_to(ROOT).as_posix() for p in files}
    quant_module_set = set()
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith("quant_system/"):
            quant_module_set.add(rel)

    issues = []
    modules = []
    total_lines = 0
    inbound = collections.defaultdict(set)
    outbound = collections.defaultdict(set)
    func_index = collections.defaultdict(list)

    parsed = {}
    broken = {}
    module_funcs = {}
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = read_text(path)
        except OSError as exc:
            broken[rel] = (1, 0, f"文件读取失败: {exc}")
            continue
        line_count = len(text.splitlines()) if text else 0
        total_lines += line_count
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as exc:
            broken[rel] = (exc.lineno or 1, line_count, f"语法解析失败: {exc.msg}")
            continue
        parsed[rel] = (text, tree)
        module_funcs[rel] = {
            s.name for s in tree.body
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not s.name.startswith("_")
        }

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        is_production = rel.startswith("quant_system/")
        if rel in broken:
            lineno, line_count, msg = broken[rel]
            issues.append({"file": rel, "line": lineno, "level": "major",
                           "rule": "parse_error", "msg": msg})
            modules.append({"file": rel, "lines": line_count, "functions": 0,
                            "longest_function": None})
            continue

        text, tree = parsed[rel]
        line_count = len(text.splitlines()) if text else 0

        file_issues = []

        def emit(rule, lineno, level, msg):
            file_issues.append({"file": rel, "line": lineno,
                                "level": level, "rule": rule, "msg": msg})

        walker = _ModuleWalker(rel, emit, is_production)
        walker.visit(tree)

        all_strings = " ".join(
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str))
        all_all = _collect_all_names(tree)
        for lineno, alias in walker.unused_imports(all_strings, all_all):
            emit("unused_import", lineno, "minor", f"未使用 import: {alias}")

        for node in walker.imports:
            for target in _resolve_import_targets(rel, node):
                if target in quant_module_set:
                    inbound[target].add(rel)
                    outbound[rel].add(target)

        import_map = _build_import_map(tree, rel, repo_files)
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_shell, target = _thin_shell_info(stmt, import_map, module_funcs, repo_files)
                func_index[(stmt.name, _param_count(stmt.args))].append(
                    {"rel": rel, "lineno": stmt.lineno, "shell": is_shell,
                     "target": target})

        if line_count > 2000:
            emit("overlong_file", 1, "minor",
                 f"文件超长: {line_count} 行（>2000，建议拆分）")

        longest = None
        if walker.longest:
            name, lno, fl = walker.longest
            longest = {"name": name, "line": lno, "lines": fl}
        modules.append({"file": rel, "lines": line_count,
                        "functions": walker.func_count, "longest_function": longest})
        issues.extend(file_issues)

    for (name, nparams), entries in func_index.items():
        if len({e["rel"] for e in entries}) < 2:
            continue
        for e in entries:
            if e["shell"]:
                target = e["target"] or "同仓公共函数"
                issues.append({"file": e["rel"], "line": e["lineno"], "level": "info",
                               "rule": "thin_shell",
                               "msg": f"薄壳转发: {name}()（{nparams} 个参数）仅转发 "
                                      f"{target}，与同仓同名同签名函数共享实现，不计重复"})
        fulls = [e for e in entries if not e["shell"]]
        if fulls:
            ref = fulls[0]
            for e in fulls[1:]:
                if e["rel"] != ref["rel"]:
                    issues.append({"file": e["rel"], "line": e["lineno"], "level": "minor",
                                   "rule": "duplicate_function",
                                   "msg": f"重复实现: {name}()（{nparams} 个参数）"
                                          f"与 {ref['rel']}:{ref['lineno']} 签名相同"})

    issues = _flatten_issues(issues)
    counts = {level: 0 for level in LEVELS}
    rule_counts = collections.Counter()
    for iss in issues:
        counts[iss["level"]] += 1
        rule_counts[iss["rule"]] += 1

    import_graph = {}
    for rel in sorted(quant_module_set):
        inb = sorted(inbound.get(rel, ()))
        outb = sorted(outbound.get(rel, ()))
        import_graph[rel] = {
            "imported_by_count": len(inb),
            "imported_by": inb,
            "imports_count": len(outb),
            "imports": outb,
        }
    imports_per_file = {f: sorted(v) for f, v in sorted(outbound.items()) if v}

    duration = time.perf_counter() - t0
    payload = {
        "scan_time": _dt.datetime.now().isoformat(timespec="seconds"),
        "date": scan_date,
        "scanner": "scripts/audit_static_scan.py",
        "roots": [str(r) for r in roots],
        "files_by_root": files_by_root,
        "files_scanned": len(files),
        "total_lines": total_lines,
        "duration_sec": round(duration, 2),
        "summary": counts,
        "rule_counts": dict(rule_counts),
        "issues": issues,
        "modules": modules,
        "import_graph": import_graph,
        "imports_per_file": imports_per_file,
    }

    out_str = None
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        out_str = str(out_path)
    except OSError as exc:
        sys.stderr.write(f"[audit_static_scan] 写入 {out_path} 失败: {exc}\n")

    per_file = collections.defaultdict(collections.Counter)
    for iss in issues:
        per_file[iss["file"]][iss["level"]] += 1
    top_files = sorted(per_file.items(),
                       key=lambda kv: (-sum(kv[1].values()), kv[0]))[:10]
    top_files = [(f, {"total": sum(c.values()),
                      "critical": c["critical"], "major": c["major"],
                      "minor": c["minor"], "info": c["info"]})
                 for f, c in top_files]

    print_summary(payload, top_files, out_str, args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
