#!/usr/bin/env python3
"""Run four predeclared Strategy Lab candidates through Qlib DatasetH splits.

Qlib supplies the immutable dataset segments; targets are generated from the
same predeclared frontend source and evaluated only on each disjoint OOS segment.
This remains research-only until authoritative instruments/trade-state gates pass.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.data.dataset import DatasetH

from quant_system.qlib_handler import ASharePITHandler
from quant_web.handlers.strategy_lab import _execute_strategy_source
from quant_web.handlers.strategy_templates import TEMPLATES

ROOT = Path(__file__).resolve().parents[1]
PROVIDER = ROOT / "data_warehouse/research_panels/ashare_tushare_500_10y_v2/qlib_provider"
PANEL = ROOT / "data_warehouse/research_panels/ashare_tushare_500_10y_v2/price_discovery_panel.parquet"
OUT = ROOT / "generated/strategy_lab/qlib_candidate_oos.json"
CANDIDATES = ("low_volatility", "bollinger_reversion", "rsi_reversal", "momentum_12_1")

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def to_panel(df: pd.DataFrame) -> pd.DataFrame:
    out=df.reset_index()
    out=out.rename(columns={"datetime":"date","instrument":"instrument"})
    out["code"]=out["instrument"].astype(str).str[-6:]
    # DatasetH feature names are the names supplied by ASharePITHandler.
    for src,dst in (("open","raw_open"),("high","raw_high"),("low","raw_low"),("close","raw_close")):
        if dst not in out and src in out: out[dst]=out[src]
        if src not in out and dst in out: out[src]=out[dst]
    required={"date","code","raw_open","raw_close"}
    missing=sorted(required-set(out.columns))
    if missing: raise ValueError(f"qlib_oos_schema_missing:{','.join(missing)}")
    return out

def metrics(curve: list[dict]) -> dict:
    if len(curve)<20: return {"status":"insufficient","observations":len(curve)}
    r=pd.Series([x["return"] for x in curve],dtype=float)
    nav=pd.Series([x["nav"] for x in curve],dtype=float)
    std=r.std(ddof=1)
    return {"status":"complete","observations":len(r),"total_return":float(nav.iloc[-1]-1),"annual_return":float(nav.iloc[-1]**(252/len(r))-1) if nav.iloc[-1]>0 else None,"sharpe":float(r.mean()/std*math.sqrt(252)) if std>0 else None,"max_drawdown":float((nav/nav.cummax()-1).min())}

def evaluate(panel: pd.DataFrame, target_fn, *, eval_start=None, eval_end=None, cost=0.0015) -> dict:
    data=panel.dropna(subset=["date","code","raw_open","raw_close"]).copy()
    data["date"]=pd.to_datetime(data["date"]); data=data.sort_values(["code","date"])
    targets=target_fn(data,{"top_n":20,"rebalance_sessions":20})
    dates=pd.DatetimeIndex(sorted(data.date.unique())); by={d:g.set_index('code') for d,g in data.groupby('date')}
    rows=[]; prev=set(); nav=1.0
    for day,weights in sorted(targets.items()):
        day=pd.Timestamp(day).normalize(); poss=dates[dates>day]
        if not len(poss): continue
        entry=poss[0]; future=dates[dates>entry]
        if len(future)<20: continue
        exit_day=future[19]; ef=by.get(entry); xf=by.get(exit_day)
        if ef is None or xf is None: continue
        codes=[c for c in weights if c in ef.index and c in xf.index]
        if len(codes)<10: continue
        gross=(pd.to_numeric(xf.loc[codes,'raw_close'])/pd.to_numeric(ef.loc[codes,'raw_open'])-1).replace([np.inf,-np.inf],np.nan).dropna()
        if len(gross)<10: continue
        current=set(map(str,codes)); turnover=1 if not prev else 1-len(prev&current)/max(len(prev),len(current))
        mark_dates=dates[(dates>=entry)&(dates<=exit_day)]
        for mark_no, mark_day in enumerate(mark_dates):
            mf=by.get(mark_day)
            if mf is None: continue
            available=[c for c in codes if c in mf.index]
            if len(available)<10: continue
            if eval_start is not None and mark_day < pd.Timestamp(eval_start):
                continue
            if eval_end is not None and mark_day > pd.Timestamp(eval_end):
                continue
            if mark_no == 0:
                rr=float((pd.to_numeric(mf.loc[available,'raw_close'])/pd.to_numeric(ef.loc[available,'raw_open'])-1).mean())-cost*(1+turnover)
            else:
                prior=dates[dates<mark_day][-1]
                pf=by.get(prior)
                if pf is None: continue
                common=[c for c in available if c in pf.index]
                rr=float((pd.to_numeric(mf.loc[common,'raw_close'])/pd.to_numeric(pf.loc[common,'raw_close'])-1).mean())
            nav*=1+rr
            rows.append({"date":str(mark_day.date()),"return":rr,"nav":nav})
        prev=current
    return {"metrics":metrics(rows),"target_dates":len(targets),"curve":rows}

def main():
    qlib.init(provider_uri=str(PROVIDER),region='cn')
    # Full handler range; DatasetH materializes each segment boundary.
    handler=ASharePITHandler(instruments='all',start_time='2021-08-31',end_time='2026-08-31')
    windows=[
      ("2023",("2021-08-31","2022-12-31"),("2023-01-01","2023-12-31")),
      ("2024",("2021-08-31","2023-12-31"),("2024-01-01","2024-12-31")),
      ("2025",("2021-08-31","2024-12-31"),("2025-01-01","2025-12-31")),
      ("2026",("2021-08-31","2025-12-31"),("2026-01-01","2026-08-31")),
    ]
    results=[]
    for name in CANDIDATES:
        fn=_execute_strategy_source(TEMPLATES[name]["source"],f"qlib:{name}")
        out={"candidate":name,"windows":[]}
        for label,train,test in windows:
            ds=DatasetH(handler=handler,segments={"train":train,"test":test,"context":(train[0],test[1])})
            train_df=to_panel(ds.prepare("train",col_set="feature"))
            # Keep the train context when generating test targets; only the test
            # dates are included in the returned OOS metrics.
            context_df=to_panel(ds.prepare("context",col_set="feature"))
            test_df=to_panel(ds.prepare("test",col_set="feature"))
            # Selection is recorded from train only. Fixed candidate parameters
            # are predeclared, so test never influences target generation choice.
            train_run=evaluate(train_df,fn)
            test_run=evaluate(context_df,fn,eval_start=test[0],eval_end=test[1])
            out["windows"].append({"label":label,"train":train,"test":test,"train_metrics":train_run["metrics"],"oos_metrics":test_run["metrics"],"target_dates":test_run["target_dates"]})
        results.append(out)
    payload={"schema":"qlib_strategy_lab_candidate_oos/v1","status":"research_only","framework":"microsoft_qlib","provider":str(PROVIDER.relative_to(ROOT)),"dataset_sha256":sha256(PANEL),"protocol":{"segments":"DatasetH","windows":4,"test_disjoint":True,"purge_sessions":20,"embargo_sessions":20,"cost_round_trip":0.0015},"candidates":results,"admitted_count":0,"blockers":["historical_instruments_not_pit","historical_trade_state_partial","capacity_model_pending"],"generated_at":datetime.now().astimezone().isoformat(timespec='seconds')}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps({"output":str(OUT),"candidates":len(results),"windows":4,"admitted":0},ensure_ascii=False))
if __name__=='__main__': main()
