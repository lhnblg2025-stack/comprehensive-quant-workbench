# -*- coding: utf-8 -*-
"""concept_lifecycle — 概念生命周期知识图谱（V11）

数据源（全部本地 parquet）:
  data_warehouse/market/zt_pool_history.parquet   全量涨停历史 2020→今
  data_warehouse/classification/concept_member.parquet  code→概念
  data_warehouse/classification/concept_board.parquet  board_code→board_name/leader
  data_warehouse/market/theme_cycle.parquet       概念×日期×zt_cnt/stage

功能:
  concept_archive(concept_code)    单概念炒作档案（次数/时长/龙头/更替/间隔）
  similar_episodes(concept_code,k) 当前炒作 vs 历史段形态相似度 + 退潮深度
  concept_report(date)             当日活跃概念逐个出档案 → generated/concept_lifecycle_{date}.json

性能: 中间表一次构建并缓存到 generated/cache/（源文件变化才重建）。
      热路径只读小 parquet + 向量化 groupby；龙头统计用 numpy 掩码按概念切片，
      不做长表×段交叉 join，避免内存爆炸；单概念查询 <2s。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CONCEPT_MEMBER = ROOT / "data_warehouse" / "classification" / "concept_member.parquet"
CONCEPT_BOARD = ROOT / "data_warehouse" / "classification" / "concept_board.parquet"
ZT_HISTORY = ROOT / "data_warehouse" / "market" / "zt_pool_history.parquet"
THEME_CYCLE = ROOT / "data_warehouse" / "market" / "theme_cycle.parquet"
GEN_DIR = ROOT / "generated"
CACHE_DIR = GEN_DIR / "cache"
PANEL_CACHE = CACHE_DIR / "concept_lifecycle_panel.parquet"
LONG_CACHE = CACHE_DIR / "concept_lifecycle_long.parquet"
EPISODES_CACHE = CACHE_DIR / "concept_lifecycle_episodes.parquet"
LEADERS_CACHE = CACHE_DIR / "concept_lifecycle_leaders.parquet"
RETREAT_CACHE = CACHE_DIR / "concept_lifecycle_retreat.parquet"
FEATURES_CACHE = CACHE_DIR / "concept_lifecycle_features.parquet"
NAMES_CACHE = CACHE_DIR / "concept_lifecycle_names.parquet"
META_CACHE = CACHE_DIR / "concept_lifecycle_meta.json"

SCHEMA_VERSION = 3

ACTIVE_MIN = 3      # zt_cnt>=3 视为活跃日
GAP_DAYS = 5        # 间隔>=5个交易日算新一次炒作
SIM_FEATURES = 8    # zt_cnt 序列下采样特征长度


def _cache_fresh() -> bool:
    need = [PANEL_CACHE, LONG_CACHE, EPISODES_CACHE, LEADERS_CACHE,
            RETREAT_CACHE, FEATURES_CACHE, NAMES_CACHE, META_CACHE]
    if any(not x.exists() for x in need):
        return False
    try:
        meta = json.loads(META_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return False
    if meta.get("schema_version") != SCHEMA_VERSION:
        return False
    for k, path in (("zt", ZT_HISTORY), ("member", CONCEPT_MEMBER)):
        if meta.get(f"{k}_mtime") != path.stat().st_mtime:
            return False
    return True


class _Cache:
    """懒加载缓存：中间表优先读盘，源文件变化时重建。"""

    def __init__(self, use_disk: bool = True) -> None:
        self._seq_parsed: Optional[dict[int, list[int]]] = None
        if use_disk and _cache_fresh():
            self._load_disk()
        else:
            self._build()

    # ── 热路径: 读盘 ──────────────────────────────────────
    def _load_disk(self) -> None:
        meta = json.loads(META_CACHE.read_text(encoding="utf-8"))
        self.concepts: list[str] = meta["concepts"]
        self.n_days: int = meta["n_days"]
        self.zt_latest: str = meta["zt_latest"]

        panel = pd.read_parquet(PANEL_CACHE)
        self.panel = panel
        self.dates = np.sort(pd.to_datetime(panel["date"].unique()).to_numpy())
        self.day_index = {d: i for i, d in enumerate(self.dates)}
        self.zt_cnt_mat = panel["zt_cnt"].to_numpy().reshape(len(self.concepts), self.n_days)
        self.max_board_mat = panel["max_board"].to_numpy().reshape(len(self.concepts), self.n_days)

        self.long = None

        nm = pd.read_parquet(NAMES_CACHE)
        self.name_map: dict[str, str] = dict(zip(nm["code"], nm["name"]))
        self.member_names: dict[str, dict[str, str]] = {}
        for c, g in nm[nm["concept"].notna()].groupby("concept"):
            self.member_names[c] = dict(zip(g["code"], g["name"]))

        cb = pd.read_parquet(CONCEPT_BOARD)
        self.board_names: dict[str, str] = dict(zip(cb["board_code"], cb["board_name"]))

        ep = pd.read_parquet(EPISODES_CACHE)
        ep["start_date"] = pd.to_datetime(ep["start_date"])
        ep["end_date"] = pd.to_datetime(ep["end_date"])
        self.episodes = ep
        self._seq_parsed: Optional[dict[int, list[int]]] = None
        self.leaders = pd.read_parquet(LEADERS_CACHE)
        self.retreat = pd.read_parquet(RETREAT_CACHE)

        feat = pd.read_parquet(FEATURES_CACHE)
        feat["concept"] = feat["concept"].astype(str)
        features: dict[str, np.ndarray] = {}
        for c, g in feat.groupby("concept"):
            cols = [f"f{i}" for i in range(SIM_FEATURES + 2)]
            features[c] = g[cols].to_numpy(dtype=float)
        self.features = features

    # ── 冷路径: 重建（源文件变化才触发）──────────────────
    def _build(self) -> None:
        cm = pd.read_parquet(CONCEPT_MEMBER)
        cm["code"] = cm["code"].astype(str).str.zfill(6)
        zt = pd.read_parquet(ZT_HISTORY)
        zt = zt[zt["is_zt"]][["date", "code", "board_count", "name"]].copy()
        zt["code"] = zt["code"].astype(str).str.zfill(6)
        zt["date"] = zt["date"].dt.normalize()

        self.dates = np.sort(zt["date"].unique())
        self.day_index = {d: i for i, d in enumerate(self.dates)}
        self.n_days = len(self.dates)
        self.zt_latest = str(pd.Timestamp(self.dates[-1]).date())
        self.concepts = sorted(cm["concept"].unique())

        self.name_map: dict[str, str] = (
            zt.dropna(subset=["name"]).groupby("code")["name"].last().to_dict()
        )
        self.member_names: dict[str, dict[str, str]] = {
            c: dict(zip(g["code"], g["name"])) for c, g in cm.groupby("concept")
        }

        # 涨停股先按 (date,code) 聚合，再展开到概念（只留龙头统计所需列）
        d1 = zt.groupby(["date", "code"], as_index=False).agg(
            board_count=("board_count", "max"))
        cm["_map"] = 1
        long = d1.merge(cm[["concept", "code", "_map"]], on="code", how="inner") \
                 .drop(columns="_map")
        self.long = long

        # 每日每概念 zt_cnt / max_board
        daily = long.groupby(["concept", "date"]).agg(
            zt_cnt=("code", "size"), max_board=("board_count", "max")).reset_index()
        base = pd.DataFrame({"concept": np.repeat(self.concepts, self.n_days),
                             "d_idx": np.tile(np.arange(self.n_days), len(self.concepts))})
        base["date"] = base["d_idx"].map(self.dates.__getitem__)
        daily["d_idx"] = daily["date"].map(self.day_index)
        panel = base.merge(daily[["concept", "d_idx", "zt_cnt", "max_board"]],
                           on=["concept", "d_idx"], how="left")
        panel["zt_cnt"] = panel["zt_cnt"].fillna(0).astype(int)
        panel["max_board"] = panel["max_board"].fillna(0).astype(int)
        panel = panel.drop(columns="d_idx")
        self.panel = panel
        self.zt_cnt_mat = panel["zt_cnt"].to_numpy().reshape(len(self.concepts), self.n_days)
        self.max_board_mat = panel["max_board"].to_numpy().reshape(len(self.concepts), self.n_days)

        cb = pd.read_parquet(CONCEPT_BOARD)
        self.board_names: dict[str, str] = dict(zip(cb["board_code"], cb["board_name"]))

        # 落盘缓存（前置：即使后续构建失败也可复用中间表）
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(PANEL_CACHE, index=False)
        long.to_parquet(LONG_CACHE, index=False)
        nm_rows = [{"code": cd, "name": n, "concept": None} for cd, n in self.name_map.items()]
        for c, m in self.member_names.items():
            nm_rows.extend({"code": cd, "name": n, "concept": c} for cd, n in m.items())
        pd.DataFrame(nm_rows).to_parquet(NAMES_CACHE, index=False)
        META_CACHE.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "concepts": self.concepts,
            "n_days": self.n_days,
            "zt_latest": self.zt_latest,
            "zt_mtime": ZT_HISTORY.stat().st_mtime,
            "member_mtime": CONCEPT_MEMBER.stat().st_mtime,
        }, ensure_ascii=False), encoding="utf-8")

        self._build_episodes_and_leaders()
        self._persist_artifacts()

    # ── 共享: 段检测 + 龙头 + 退潮深度（向量化）──────────
    def _build_episodes_and_leaders(self) -> None:
        zt_cnt = self.zt_cnt_mat
        max_board = self.max_board_mat
        active = zt_cnt >= ACTIVE_MIN
        active_idx = np.nonzero(active.any(axis=1))[0]

        ep_rows = []
        for ci in active_idx:
            c = self.concepts[ci]
            idxs = np.nonzero(active[ci])[0]
            ep_idx = 0
            start = idxs[0]
            prev = idxs[0]
            for j in range(1, len(idxs)):
                if idxs[j] - prev >= GAP_DAYS:
                    ep_rows.append((c, ep_idx, int(start), int(prev),
                                    int(prev - start + 1),
                                    [int(x) for x in zt_cnt[ci, start:prev + 1]],
                                    int(max_board[ci, start:prev + 1].max())))
                    ep_idx += 1
                    start = idxs[j]
                prev = idxs[j]
            ep_rows.append((c, ep_idx, int(start), int(prev), int(prev - start + 1),
                            [int(x) for x in zt_cnt[ci, start:prev + 1]],
                            int(max_board[ci, start:prev + 1].max())))
        ep = pd.DataFrame(ep_rows, columns=["concept", "ep_idx", "start_d", "end_d",
                                            "dur", "seq", "max_board"])
        if len(ep):
            ep["start_date"] = ep["start_d"].map(self.dates.__getitem__)
            ep["end_date"] = ep["end_d"].map(self.dates.__getitem__)
        else:
            ep["start_date"] = pd.Series(dtype="datetime64[ns]")
            ep["end_date"] = pd.Series(dtype="datetime64[ns]")
        self.episodes = ep

        # 每段龙头: 段内 board_count 最高且出现次数最多。
        # 先向量化算每日龙头（concept×date 内按 maxb desc, cnt desc 取首），
        # 再取段内最高板日的当日龙头作为段龙头。
        long = self.long
        if len(ep):
            dl = long.groupby(["concept", "date", "code"]).agg(
                maxb=("board_count", "max"), cnt=("code", "size")).reset_index()
            dl = dl.sort_values(["concept", "date", "maxb", "cnt", "code"],
                                ascending=[True, True, False, False, True])
            dl = dl.groupby(["concept", "date"], as_index=False).first()
            dl["d_idx"] = dl["date"].map(self.day_index)
            dl = dl.rename(columns={"maxb": "max_board", "cnt": "cnt"})

            # 段内最高板日的 d_idx（向量化: 段内首个 max_board==段最高板的活跃日）
            mb = self.max_board_mat
            ci_map = {c: i for i, c in enumerate(self.concepts)}
            ci_arr = np.array([ci_map[c] for c in ep["concept"]])
            st = ep["start_d"].to_numpy()
            en = ep["end_d"].to_numpy()
            seg_mb = np.where(
                np.arange(self.n_days)[None, :] >= st[:, None],
                np.where(np.arange(self.n_days)[None, :] <= en[:, None], mb[ci_arr], -1),
                -1)
            peak_hit = (seg_mb == ep["max_board"].to_numpy()[:, None])
            peak_d = st + np.argmax(peak_hit, axis=1)
            ep = ep.assign(peak_d=peak_d)

            dl2 = dl[["concept", "d_idx", "code", "cnt"]].rename(
                columns={"code": "l_code", "cnt": "l_cnt"})
            lk = ep.merge(dl2, left_on=["concept", "peak_d"], right_on=["concept", "d_idx"],
                          how="left").drop(columns="d_idx")
            msk = lk["l_code"].notna()
            leader_rows = [{
                "concept": r.concept, "ep_idx": int(r.ep_idx), "code": r.l_code,
                "name": self.name_map.get(r.l_code) or self.member_names.get(r.concept, {}).get(r.l_code, ""),
                "max_board": int(r.max_board),
                "cnt": int(r.l_cnt) if pd.notna(r.l_cnt) else 0,
            } for r in lk[msk].itertuples()]
            # 兜底: 峰值日不在每日龙头表里（成员映射变化等）→ 回退到段内全量统计
            missing = ep[~msk]
            if len(missing):
                fallback = []
                for c, g in missing.groupby("concept"):
                    seg = long[long["concept"] == c]
                    if not len(seg):
                        continue
                    s_code = seg["code"].to_numpy()
                    s_date = seg["date"].to_numpy()
                    s_bc = seg["board_count"].to_numpy()
                    for r in g.itertuples():
                        mm = (s_date >= r.start_date) & (s_date <= r.end_date)
                        if not mm.any():
                            continue
                        codes = s_code[mm]
                        bcs = s_bc[mm]
                        top_bc = int(bcs.max())
                        m2 = bcs == top_bc
                        uniq, counts = np.unique(codes[m2], return_counts=True)
                        fallback.append({
                            "concept": c, "ep_idx": int(r.ep_idx), "code": uniq[np.argmax(counts)],
                            "name": self.name_map.get(uniq[np.argmax(counts)])
                                    or self.member_names.get(c, {}).get(uniq[np.argmax(counts)], ""),
                            "max_board": top_bc, "cnt": int(counts.max()),
                        })
                leader_rows.extend(fallback)
            self.leaders = pd.DataFrame(leader_rows, columns=[
                "concept", "ep_idx", "code", "name", "max_board", "cnt"])
        else:
            self.leaders = pd.DataFrame(columns=["concept", "ep_idx", "code", "name",
                                                 "max_board", "cnt"])

        # 段后 5/10 日 zt_cnt 均值（退潮深度）—— 前缀和向量化
        n_c, n_d = zt_cnt.shape
        if len(ep):
            csum = np.concatenate([np.zeros((n_c, 1), dtype=np.int64),
                                   np.cumsum(zt_cnt, axis=1)], axis=1)
            ci_map = {c: i for i, c in enumerate(self.concepts)}
            e = ep["end_d"].to_numpy() + 1
            ci = np.array([ci_map[c] for c in ep["concept"]])
            w = np.minimum(e[:, None] + np.array([5, 10]), n_d)
            s = csum[ci[:, None], w] - csum[ci[:, None], e[:, None]]
            cnt = (w - e[:, None]).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                r5 = np.where(cnt[:, 0] > 0, s[:, 0] / cnt[:, 0], np.nan)
                r10 = np.where(cnt[:, 1] > 0, s[:, 1] / cnt[:, 1], np.nan)
            self.retreat = pd.DataFrame({
                "concept": ep["concept"], "ep_idx": ep["ep_idx"],
                "retreat_5": r5, "retreat_10": r10})
        else:
            self.retreat = pd.DataFrame(columns=["concept", "ep_idx",
                                                 "retreat_5", "retreat_10"])

        # 每概念特征矩阵 [n_ep, SIM_FEATURES+2]: 下采样 zt_cnt + 时长 + 最高板
        features: dict[str, np.ndarray] = {}
        if len(ep):
            for c, g in ep.groupby("concept"):
                seqs = [np.array(x, dtype=float) for x in g["seq"]]
                fixed = np.array([self._resample(s) for s in seqs])
                dur = g["dur"].to_numpy(dtype=float).reshape(-1, 1)
                maxb = g["max_board"].to_numpy(dtype=float).reshape(-1, 1)
                features[c] = np.hstack([fixed, dur, maxb])
        self.features = features

    def _persist_artifacts(self) -> None:
        """把段/龙头/退潮/特征 持久化，热路径直接读盘。"""
        ep = self.episodes.copy()
        ep["seq"] = ep["seq"].apply(json.dumps)
        ep.to_parquet(EPISODES_CACHE, index=False)
        self.leaders.to_parquet(LEADERS_CACHE, index=False)
        self.retreat.to_parquet(RETREAT_CACHE, index=False)

        rows = []
        for c, mat in self.features.items():
            df = pd.DataFrame(mat, columns=[f"f{i}" for i in range(SIM_FEATURES + 2)])
            df.insert(0, "concept", c)
            rows.append(df)
        feat_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["concept"] + [f"f{i}" for i in range(SIM_FEATURES + 2)])
        feat_df.to_parquet(FEATURES_CACHE, index=False)

    def _get_seq(self) -> dict[int, list[int]]:
        """惰性解析段内 zt_cnt 序列（similar_episodes 才需要）。"""
        if self._seq_parsed is None:
            self._seq_parsed = {
                i: json.loads(x) for i, x in zip(self.episodes.index,
                                                 self.episodes["seq"].tolist())
            }
        return self._seq_parsed

    @staticmethod
    def _resample(s: np.ndarray, n: int = SIM_FEATURES) -> np.ndarray:
        if len(s) == n:
            return s
        if len(s) > n:
            idx = np.linspace(0, len(s) - 1, n).round().astype(int)
            return s[idx]
        out = np.zeros(n)
        out[:len(s)] = s
        return out


_CACHE: Optional[_Cache] = None


def _get_cache() -> _Cache:
    global _CACHE
    if _CACHE is None:
        _CACHE = _Cache()
    return _CACHE


def _eps_of(c: str, cache: _Cache) -> pd.DataFrame:
    ep = cache.episodes
    if not len(ep):
        return ep
    return ep[ep["concept"] == c]


def _leader_history(c: str, cache: _Cache) -> list[dict]:
    rows = []
    ldf = cache.leaders
    if len(ldf):
        sub = ldf[ldf["concept"] == c].sort_values("ep_idx")
        for r in sub.itertuples():
            rows.append({"ep_idx": int(r.ep_idx), "code": r.code, "name": r.name,
                         "max_board": int(r.max_board), "count": int(r.cnt)})
    return rows


def _interval_dist(ep: pd.DataFrame) -> tuple[list[dict], list[int]]:
    """炒作间隔分布（交易日数）+ 原始间隔列表。"""
    if len(ep) < 2:
        return [], []
    order = np.argsort(ep["start_d"].to_numpy())
    starts = ep["start_d"].to_numpy()[order]
    ends = ep["end_d"].to_numpy()[order]
    gaps = [int(starts[i] - ends[i - 1] - 1) for i in range(1, len(order))]
    out = []
    for lo, hi, label in ((1, 4, "1-4"), (5, 9, "5-9"), (10, 19, "10-19"), (20, 10 ** 9, "20+")):
        out.append({"interval": label, "count": int(sum(1 for g in gaps if lo <= g <= hi))})
    return out, gaps


def concept_archive(concept_code: str) -> dict:
    """单概念炒作档案。无历史记录 → {"concept":..., "note":"首次活跃"}。"""
    cache = _get_cache()
    ep = _eps_of(concept_code, cache)
    name = cache.board_names.get(concept_code, "")
    if len(ep) == 0:
        return {"concept": concept_code, "name": name, "note": "首次活跃"}

    ep = ep.sort_values("start_d").reset_index(drop=True)
    leaders = _leader_history(concept_code, cache)
    change = []
    for i in range(1, len(leaders)):
        prev_l, cur_l = leaders[i - 1], leaders[i]
        if prev_l["code"] != cur_l["code"]:
            change.append({"from_ep": prev_l["ep_idx"], "to_ep": cur_l["ep_idx"],
                           "from": prev_l["name"] or prev_l["code"],
                           "to": cur_l["name"] or cur_l["code"]})
    intervals, gaps = _interval_dist(ep)

    return {
        "concept": concept_code,
        "name": name,
        "episode_count": int(len(ep)),
        "avg_duration": round(float(ep["dur"].mean()), 1),
        "max_duration": int(ep["dur"].max()),
        "latest_episode": {
            "start": str(ep["start_date"].iloc[-1].date()),
            "end": str(ep["end_date"].iloc[-1].date()),
            "duration": int(ep["dur"].iloc[-1]),
            "max_board": int(ep["max_board"].iloc[-1]),
            "ongoing": bool(ep["end_d"].iloc[-1] == cache.n_days - 1),
        },
        "leader": leaders[-1] if leaders else None,
        "historical_leader": leaders,
        "leader_changes": change,
        "interval_dist": intervals,
        "avg_interval": round(float(np.mean(gaps)), 1) if gaps else None,
    }


def similar_episodes(concept_code: str, k: int = 3) -> list[dict]:
    """当前这轮炒作与历史段的形态相似度，输出历史段后续退潮深度。"""
    cache = _get_cache()
    ep = _eps_of(concept_code, cache)
    if len(ep) < 2:
        return []
    ep = ep.sort_values("start_d").reset_index(drop=True)
    feat = cache.features.get(concept_code)
    if feat is None:
        return []
    cur = feat[-1]
    pool = feat[:-1]

    mu = pool.mean(axis=0)
    sd = pool.std(axis=0)
    sd[sd == 0] = 1.0
    dist = np.linalg.norm((pool - mu) / sd - (cur - mu) / sd, axis=1)
    order = np.argsort(dist)[:k]
    mx = float(dist.max()) if len(dist) else 1.0
    if mx == 0:
        mx = 1.0

    retreat = cache.retreat.set_index(["concept", "ep_idx"]) if len(cache.retreat) else pd.DataFrame()
    seq_map = cache._get_seq()
    out = []
    for i in order:
        row = ep.iloc[int(i)]
        key = (concept_code, int(row["ep_idx"]))
        r5 = r10 = None
        if len(retreat) and key in retreat.index:
            r5 = retreat.loc[key, "retreat_5"]
            r10 = retreat.loc[key, "retreat_10"]
        out.append({
            "similarity": round(float(1.0 - dist[int(i)] / mx), 3),
            "start": str(row["start_date"].date()),
            "end": str(row["end_date"].date()),
            "duration": int(row["dur"]),
            "max_board": int(row["max_board"]),
            "zt_seq": seq_map.get(row.name, []),
            "retreat_5d": None if r5 is None or pd.isna(r5) else round(float(r5), 2),
            "retreat_10d": None if r10 is None or pd.isna(r10) else round(float(r10), 2),
        })
    return out


def concept_report(date: Optional[str] = None) -> dict:
    """当日活跃概念逐个输出 archive 摘要，标记'再次炒作'，落盘 JSON。"""
    cache = _get_cache()
    tc = pd.read_parquet(THEME_CYCLE)
    tc["date"] = tc["date"].dt.normalize()
    latest = tc["date"].max() if date is None else pd.Timestamp(date)
    day = tc[tc["date"] == latest]
    active = day[day["zt_cnt"] >= ACTIVE_MIN]
    if len(active) == 0:
        return {"date": str(latest.date()), "note": "当日无活跃概念", "concepts": []}

    entries = []
    for r in active.itertuples():
        code = r.concept
        arc = concept_archive(code)
        sims = similar_episodes(code, k=3)
        again = bool(arc.get("episode_count", 0) >= 2)
        entries.append({
            "concept": code,
            "board_name": arc.get("name") or r.board_name,
            "zt_cnt": int(r.zt_cnt),
            "max_board": int(r.max_board),
            "stage": r.stage,
            "again": again,
            "archive": arc,
            "similar_episodes": sims,
        })

    again_list = [e["concept"] for e in entries if e["again"]]
    report = {
        "date": str(latest.date()),
        "active_concept_count": len(entries),
        "again_count": len(again_list),
        "again_concepts": again_list,
        "concepts": entries,
    }
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GEN_DIR / f"concept_lifecycle_{latest.date()}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                        encoding="utf-8")
    report["_saved"] = str(out_path)
    return report


if __name__ == "__main__":
    import argparse
    import time

    p = argparse.ArgumentParser(description="概念生命周期知识图谱")
    p.add_argument("--concept", default="BK0899")
    p.add_argument("--report", action="store_true", help="生成当日概念生命周期报告")
    p.add_argument("--date", default=None, help="报告日期 YYYY-MM-DD")
    args = p.parse_args()

    t0 = time.perf_counter()
    a = concept_archive(args.concept)
    t1 = time.perf_counter()
    print(f"[concept_archive] {args.concept} 耗时 {t1 - t0:.3f}s")
    print(json.dumps(a, ensure_ascii=False, indent=1, default=str))

    sim = similar_episodes(args.concept)
    print(f"\n[similar_episodes] k={len(sim)}")
    print(json.dumps(sim, ensure_ascii=False, indent=1, default=str))

    if args.report:
        r = concept_report(args.date)
        print(f"\n[concept_report] 保存至 {r.get('_saved')}")
