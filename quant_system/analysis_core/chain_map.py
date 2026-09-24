# -*- coding: utf-8 -*-
"""
chain_map — 产业链先验因果映射引擎（V11 作战地图联动）

与 industry_graph（数据驱动的实证涨停传导）双图并行、互不覆盖：
  chain_map      = 确定性先验上下游（chain_map_data.CHAINS/RELATIONS）
  industry_graph = 概率性传导（lift）

功能：
  load()                       模块级缓存 ChainMap
  upstream_of(sector)          申万一级 → 上游申万一级列表
  downstream_of(sector)        申万一级 → 下游申万一级列表
  chains_of(sector)            申万一级 → [{chain, layer, node}, ...]
  chain_temperature(chain,date) 链内三层最近 5 日平均涨跌幅 + 温度（层内等权，只读 ≤ date 防前视）
  propagate_from_concept(concept,date) 题材→成分股→申万一级→链/层→温度与信号

数据降级：任何数据源缺失/异常 → 空或 None，不抛异常（供 battle_map try/except 兜底）。

用法：
  python3 -m quant_system.analysis_core.chain_map --chains
  python3 -m quant_system.analysis_core.chain_map --sector 电力设备
  python3 -m quant_system.analysis_core.chain_map --temp 新能源车 --date 2026-08-11
  python3 -m quant_system.analysis_core.chain_map --concept 固态电池 --date 2026-08-11
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analysis_core.chain_map_data import CHAINS, RELATIONS  # noqa: E402

SW_FIRST = ROOT / "data_warehouse" / "industry" / "sw_first.parquet"
SW_FIRST_CONS = ROOT / "data_warehouse" / "industry" / "sw_first_cons.parquet"
SW_FIRST_HIST = ROOT / "data_warehouse" / "industry" / "sw_first_hist.parquet"
CONCEPT_BOARD = ROOT / "data_warehouse" / "classification" / "concept_board.parquet"
CONCEPT_MEMBER = ROOT / "data_warehouse" / "classification" / "concept_member.parquet"

LOOKBACK = 5  # 链温度回看交易日数
LAYERS = ("upstream", "midstream", "downstream")


class ChainMap:
    """先验产业链知识库 + 数据仓库索引（一次性载入，模块级缓存）。"""

    def __init__(self, chains, relations, sw_name2code, sw_code2name,
                 stock_sector, board_name2code, hist):
        self.chains = chains
        self.relations = relations
        self.sw_name2code = sw_name2code          # 行业名称 → 6位代码
        self.sw_code2name = sw_code2name          # 6位代码 → 行业名称
        self.stock_sector = stock_sector          # 证券代码 → 行业名称
        self.board_name2code = board_name2code    # 概念名 → BK代码
        self.hist = hist                          # sw_first_hist（代码/日期/收盘）

    # ── 关系查询 ──────────────────────────────────────────────
    def upstream_of(self, sector: str) -> list[str]:
        """sector 的直接上游申万一级（去重保序）。"""
        seen: set[str] = set()
        out = []
        for up, down, _ in self.relations:
            if down == sector and up not in seen:
                seen.add(up)
                out.append(up)
        return out

    def downstream_of(self, sector: str) -> list[str]:
        """sector 的直接下游申万一级（去重保序）。"""
        seen: set[str] = set()
        out = []
        for up, down, _ in self.relations:
            if up == sector and down not in seen:
                seen.add(down)
                out.append(down)
        return out

    def chains_of(self, sector: str) -> list[dict]:
        """sector 属于哪些链的哪一层：[{chain, layer, node}, ...]。"""
        out = []
        for chain, nodes in self.chains.items():
            for node, layer, sw in nodes:
                if sw == sector:
                    out.append({"chain": chain, "layer": layer, "node": node})
        return out

    # ── 链温度 ────────────────────────────────────────────────
    def chain_temperature(self, chain: str, date: str) -> dict:
        """链内三层最近 5 日平均涨跌幅与温度（层内等权，只读 ≤ date 数据防前视）。"""
        if chain not in self.chains or self.hist is None or self.hist.empty:
            return {}
        try:
            d = pd.to_datetime(date) if date else pd.NaT
        except Exception:
            d = pd.NaT
        if pd.isna(d):
            d = self.hist["日期"].max()  # 未给日期 → 历史最新（仍只读已发生数据）
        h = self.hist[self.hist["日期"] <= d]
        if h.empty:
            return {}

        # 节点 → 行业代码（去重：同一行业出现在多个节点时只计一次，保持层内等权）
        layer_sectors: dict[str, list[str]] = {k: [] for k in LAYERS}
        for node, layer, sw in self.chains[chain]:
            code = self.sw_name2code.get(sw)
            if code and code not in layer_sectors[layer]:
                layer_sectors[layer].append(code)
        if not any(layer_sectors.values()):
            return {}

        result: dict[str, dict] = {"chain": chain, "window": LOOKBACK, "layers": {}}
        code_set = {c for cs in layer_sectors.values() for c in cs}
        sub = h[h["代码"].isin(code_set)].sort_values(["代码", "日期"])
        for layer, codes in layer_sectors.items():
            if not codes:
                continue
            rets, totals = [], []
            for code in codes:
                closes = sub.loc[sub["代码"] == code, "收盘"]
                tail = closes.tail(LOOKBACK + 1)
                if len(tail) >= 2:
                    rets.append(tail.pct_change().dropna().mean())
                if len(tail) == LOOKBACK + 1:
                    totals.append(tail.iloc[-1] / tail.iloc[0] - 1)
            if not rets:
                continue
            avg_ret = float(np.mean(rets))
            result["layers"][layer] = {
                "avg_ret": round(avg_ret, 4),            # 5日平均日涨跌幅（层内等权）
                "total_ret": round(float(np.mean(totals)) if totals else avg_ret, 4),
                "temp": round(float(np.clip(50 + avg_ret * 100, 0, 100)), 1),
                "sectors": [self.sw_code2name.get(c, c) for c in codes],
            }
        if not result["layers"]:
            return {}
        result["as_of"] = str(sub["日期"].max())[:10]
        return result

    # ── 题材传导 ──────────────────────────────────────────────
    def propagate_from_concept(self, concept: str, date: str) -> list[dict]:
        """题材名 → 成分股 → 申万一级 → 链/层 → 温度信号。

        返回 [{chain, layer, node, sectors, up_rets, mid_rets, down_rets, signal}, ...]
        """
        if not concept:
            return []
        board = self._resolve_board(concept)
        if board is None or self.hist is None:
            return []  # 正常空结果：题材未收录 / 历史数据缺失
        if not CONCEPT_MEMBER.exists():
            return []  # 正常空结果：数据源文件不存在
        # 数据源存在但读取/解析异常 → 向上抛出（battle_map try/except 捕获进 degraded）
        mem = pd.read_parquet(CONCEPT_MEMBER)
        mem["code"] = mem["code"].astype(str).str.zfill(6)
        codes = set(mem.loc[mem["concept"] == board, "code"])
        if not codes:
            return []  # 正常空结果：题材无成分股
        sectors = sorted({self.stock_sector[c] for c in codes if c in self.stock_sector})
        if not sectors:
            return []

        # 命中链/层：{chain: {layer: {"nodes": [...], "sectors": set(...)}}}
        hit: dict[str, dict[str, dict]] = {}
        for sec in sectors:
            for entry in self.chains_of(sec):
                hit.setdefault(entry["chain"], {}).setdefault(entry["layer"], {
                    "nodes": [], "sectors": set()})
                hit[entry["chain"]][entry["layer"]]["nodes"].append(entry["node"])
                hit[entry["chain"]][entry["layer"]]["sectors"].add(sec)

        # 按知识库链序 + 层序输出（题材最相关的链排前，供 battle_map 取 props[0]）
        temps: dict[str, dict] = {}
        out = []
        for chain in self.chains:
            layers = hit.get(chain)
            if not layers:
                continue
            if chain not in temps:
                temps[chain] = self.chain_temperature(chain, date) or {}
            t = temps[chain]
            up = (t.get("layers", {}).get("upstream") or {}).get("avg_ret")
            mid = (t.get("layers", {}).get("midstream") or {}).get("avg_ret")
            down = (t.get("layers", {}).get("downstream") or {}).get("avg_ret")
            for layer in LAYERS:
                info = layers.get(layer)
                if not info:
                    continue
                out.append({
                    "chain": chain,
                    "layer": layer,
                    "node": sorted(set(info["nodes"])),
                    "sectors": sorted(info["sectors"]),
                    "up_rets": up,
                    "mid_rets": mid,
                    "down_rets": down,
                    "signal": _signal(layer, up, mid, down),
                })
        return out

    def _resolve_board(self, concept: str) -> str | None:
        """概念名 → BK 代码；输入已是 BK 代码则原样返回；查不到返回 None。"""
        c = str(concept).strip()
        if c[:2].upper() == "BK" and c[2:].isdigit():
            return c
        return self.board_name2code.get(c)


def _signal(layer: str, up, mid, down) -> str:
    """层温度信号：层内涨跌 + 跨层传导方向。"""
    rets = {"upstream": up, "midstream": mid, "downstream": down}
    r = rets.get(layer)
    if r is None:
        return "数据不足"
    base = "升温" if r > 0 else ("降温" if r < 0 else "走平")
    if up is not None and down is not None and mid is not None:
        if up > 0 and mid > 0 and down > 0:
            return f"{base}（全链正向传导）"
        if up > 0 and down < 0:
            return "上游强·下游弱"
        if up < 0 and down > 0:
            return "下游独立走强"
    return base


# ── 载入与缓存 ───────────────────────────────────────────────
_CACHE: ChainMap | None = None
_LOCK = threading.Lock()


def _read_parquet(path: Path, **kw) -> pd.DataFrame | None:
    """读取 parquet：文件不存在 → None（正常空结果）；存在但读取出错 → 抛异常（可运维）。"""
    if not path.exists():
        return None
    return pd.read_parquet(path, **kw)


def _build() -> ChainMap:
    sw = _read_parquet(SW_FIRST)
    cons = _read_parquet(SW_FIRST_CONS)
    hist = _read_parquet(SW_FIRST_HIST)
    cb = _read_parquet(CONCEPT_BOARD)

    sw_name2code: dict[str, str] = {}
    sw_code2name: dict[str, str] = {}
    if sw is not None and "行业代码" in sw and "行业名称" in sw:
        sw = sw.copy()
        sw["_code6"] = sw["行业代码"].astype(str).str[:6]
        sw_name2code = dict(zip(sw["行业名称"], sw["_code6"]))
        sw_code2name = dict(zip(sw["_code6"], sw["行业名称"]))

    stock_sector: dict[str, str] = {}
    if cons is not None and {"证券代码", "行业代码"} <= set(cons.columns):
        cc = cons[["证券代码", "行业代码"]].drop_duplicates("证券代码")
        cc["证券代码"] = cc["证券代码"].astype(str).str.zfill(6)
        cc["行业代码"] = cc["行业代码"].astype(str).str[:6]
        cc["_sector"] = cc["行业代码"].map(sw_code2name)
        for code, sector in zip(cc["证券代码"], cc["_sector"]):
            if sector is not None and not pd.isna(sector):
                stock_sector[code] = sector

    board_name2code: dict[str, str] = {}
    if cb is not None and {"board_name", "board_code"} <= set(cb.columns):
        board_name2code = dict(zip(cb["board_name"], cb["board_code"]))

    hist_df: pd.DataFrame | None = None
    if hist is not None and {"代码", "日期", "收盘"} <= set(hist.columns):
        hist_df = hist[["代码", "日期", "收盘"]].copy()
        hist_df["代码"] = hist_df["代码"].astype(str).str.zfill(6)
        hist_df["日期"] = pd.to_datetime(hist_df["日期"])

    return ChainMap(CHAINS, RELATIONS, sw_name2code, sw_code2name,
                    stock_sector, board_name2code, hist_df)


def load() -> ChainMap:
    """模块级缓存加载（线程安全双重检查锁，并发首调只 _build 一次）。

    数据源文件缺失 → 空映射（正常空结果，不抛异常）；
    数据源存在但读取/解析出错 → 抛异常（供 battle_map try/except 捕获进 degraded，可运维）。
    """
    global _CACHE
    if _CACHE is None:
        with _LOCK:
            if _CACHE is None:
                _CACHE = _build()
    return _CACHE


# 模块级便捷函数（供 battle_map 等调用）
def upstream_of(sector: str) -> list[str]:
    return load().upstream_of(sector)


def downstream_of(sector: str) -> list[str]:
    return load().downstream_of(sector)


def chains_of(sector: str) -> list[dict]:
    return load().chains_of(sector)


def chain_temperature(chain: str, date: str) -> dict:
    return load().chain_temperature(chain, date)


def propagate_from_concept(concept: str, date: str) -> list[dict]:
    return load().propagate_from_concept(concept, date)


# ── CLI ──────────────────────────────────────────────────────
def _cli() -> None:
    ap = argparse.ArgumentParser(description="产业链先验映射（chain_map）")
    ap.add_argument("--chains", action="store_true", help="列出全部链与节点数")
    ap.add_argument("--sector", help="查申万一级行业属于哪些链/层")
    ap.add_argument("--temp", help="查链温度（如 新能源车）")
    ap.add_argument("--concept", help="查题材→链/层传导")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD")
    args = ap.parse_args()

    if args.chains:
        for chain, nodes in CHAINS.items():
            layers = {}
            for _, layer, _ in nodes:
                layers[layer] = layers.get(layer, 0) + 1
            print(f"{chain}: {len(nodes)} 节点 {layers}")
        return
    if args.sector:
        cm = load()
        print(f"{args.sector} 上游: {cm.upstream_of(args.sector)}")
        print(f"{args.sector} 下游: {cm.downstream_of(args.sector)}")
        print(f"{args.sector} 链归属: {json.dumps(cm.chains_of(args.sector), ensure_ascii=False)}")
        return
    if args.temp:
        print(json.dumps(load().chain_temperature(args.temp, args.date), ensure_ascii=False, indent=1))
        return
    if args.concept:
        print(json.dumps(load().propagate_from_concept(args.concept, args.date),
                         ensure_ascii=False, indent=1))
        return
    ap.print_help()


if __name__ == "__main__":
    _cli()
