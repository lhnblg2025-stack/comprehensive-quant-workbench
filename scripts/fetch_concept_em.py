#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch Eastmoney concept boards from push2delay into the live warehouse.

The former implementation lived under the archive tree, so its ROOT pointed
at the wrong checkout and it was never part of the production data path.
This copy preserves incremental member fetching and has a small --limit mode
for source verification before a full refresh.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CLS_DIR = ROOT / "data_warehouse" / "classification"
CLS_DIR.mkdir(parents=True, exist_ok=True)
DONE_FILE = CLS_DIR / "concept_done.json"
BOARD_FILE = CLS_DIR / "concept_board.parquet"
MEMBER_FILE = CLS_DIR / "concept_member.parquet"
HOST = "https://push2delay.eastmoney.com/api/qt/clist/get"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}


def fetch(params: dict, timeout: int = 12) -> dict | None:
    for attempt in range(3):
        try:
            r = requests.get(HOST, params=params, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            data = r.json().get("data")
            if data is not None:
                return data
        except Exception as exc:
            if attempt == 2:
                print(f"source failure: {type(exc).__name__}: {str(exc)[:120]}", file=sys.stderr)
        time.sleep(1 + attempt)
    return None


def board_list() -> pd.DataFrame:
    rows, page = [], 1
    while True:
        data = fetch({"pn": page, "pz": 100, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                      "fid": "f3", "fs": "m:90+t:3",
                      "fields": "f12,f14,f2,f3,f20,f8,f104,f105,f128,f140"})
        diff = (data or {}).get("diff") or []
        if not diff:
            break
        for x in diff:
            rows.append({"board_code": str(x.get("f12", "")), "board_name": x.get("f14", ""),
                         "price": x.get("f2"), "pct_chg": x.get("f3"), "total_mv": x.get("f20"),
                         "turnover": x.get("f8"), "up_count": x.get("f104"), "down_count": x.get("f105"),
                         "leader_name": x.get("f128", ""), "leader_code": str(x.get("f140", "")),
                         "ts": pd.Timestamp.now().strftime("%Y-%m-%d")})
        if len(rows) >= int((data or {}).get("total") or len(rows)):
            break
        page += 1
    return pd.DataFrame(rows)


def members(board: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Fetch members with the board code required by Eastmoney (fs=b:BKxxxx)."""
    done = json.loads(DONE_FILE.read_text(encoding="utf-8")) if DONE_FILE.exists() else {}
    rows = []
    pairs = [(str(r.get("board_code", "")), str(r.get("board_name", "")))
             for r in board[["board_code", "board_name"]].to_dict("records")]
    pairs = [(code, name) for code, name in pairs if code and name]
    if limit:
        pairs = pairs[:limit]
    for i, (code, name) in enumerate(pairs, 1):
        # Legacy done files keyed by Chinese names are not trusted: the old
        # worker queried fs=b:<name>, which returned invalid/empty members.
        if done.get(f"code:{code}"):
            continue
        data = fetch({"pn": 1, "pz": 500, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                      "fid": "f3", "fs": f"b:{code}", "fields": "f12,f14"})
        if data is None:
            print(f"member source failed: {code} {name}", file=sys.stderr)
            continue
        diff = data.get("diff") or []
        for x in diff:
            rows.append({"concept": code, "concept_name": name,
                         "code": str(x.get("f12", "")).zfill(6), "name": x.get("f14", "")})
        # Empty but valid responses are complete; failed responses are retryable.
        done[f"code:{code}"] = True
        DONE_FILE.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
        if i % 10 == 0:
            print(f"members {i}/{len(pairs)}", flush=True)
        time.sleep(.25)
    new = pd.DataFrame(rows, columns=["concept", "concept_name", "code", "name"])
    if MEMBER_FILE.exists():
        old = pd.read_parquet(MEMBER_FILE)
        aliases = {"board_code": "concept", "板块代码": "concept", "代码": "code", "名称": "name"}
        old = old.rename(columns={k: v for k, v in aliases.items() if k in old.columns})
        if "concept" in old.columns and "code" in old.columns:
            if "concept_name" not in old.columns:
                old["concept_name"] = old["concept"].astype(str)
            old = old[["concept", "concept_name", "code", "name"]].copy()
            new = pd.concat([old, new], ignore_index=True)
    if not new.empty:
        new["concept"] = new["concept"].astype(str)
        new["code"] = new["code"].astype(str).str.zfill(6)
        new = new.drop_duplicates(["concept", "code"])
    return new


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="board count for verification; 0 means all")
    ap.add_argument("--no-members", action="store_true")
    args = ap.parse_args()
    boards = board_list()
    if boards.empty:
        print("no board data returned", file=sys.stderr)
        return 2
    boards.to_parquet(BOARD_FILE, index=False)
    print(f"boards={len(boards)} as_of={boards['ts'].max()}", flush=True)
    if not args.no_members:
        mem = members(boards, args.limit or None)
        if not mem.empty:
            mem.to_parquet(MEMBER_FILE, index=False)
            print(f"members={len(mem)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
