#!/usr/bin/env python3
"""Persist normalized daily industry-chain layer metrics."""
from __future__ import annotations
import argparse,json,os,sys
from datetime import datetime
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
OUT=ROOT/'data_warehouse/industry_chain/history.parquet';LATEST=ROOT/'generated/industry_chain_latest.json'
LAYERS=('upstream','midstream','downstream')
def atomic(df,p):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');df.to_parquet(t,index=False);os.replace(t,p)
def collect(date=None):
 from quant_system.analysis_core.chain_map_data import CHAINS
 from quant_system.analysis_core.chain_map import chain_temperature
 day=date or datetime.now().strftime('%Y-%m-%d');rows=[]
 for chain,nodes in CHAINS.items():
  try: temp=chain_temperature(chain,day) or {}
  except Exception as exc: temp={'error':str(exc)[:120]}
  ls=temp.get('layers') or {}; vals=[float(v.get('temp')) for v in ls.values() if v.get('temp') is not None]
  row={'date':pd.Timestamp(day),'as_of':temp.get('as_of'),'chain':chain,'window':temp.get('window'),'chain_temperature':round(sum(vals)/len(vals),2) if vals else None,'signal':'偏热' if vals and sum(vals)/len(vals)>=55 else '偏冷' if vals and sum(vals)/len(vals)<=45 else '中性','node_count':len(nodes),'source':'chain_map+sw_first_hist','status':'available' if vals else 'missing','schema_version':'industry-chain.v1','generated_at':datetime.now().isoformat()}
  for layer in LAYERS:
   v=ls.get(layer) or {}; row[layer+'_nodes']=len(v.get('sectors') or []);row[layer+'_temp']=v.get('temp');row[layer+'_avg_ret']=v.get('avg_ret');row[layer+'_total_ret']=v.get('total_ret')
  row['detail_json']=json.dumps(temp,ensure_ascii=False,default=str);rows.append(row)
 df=pd.DataFrame(rows)
 if OUT.exists(): df=pd.concat([pd.read_parquet(OUT),df],ignore_index=True)
 df['date']=pd.to_datetime(df['date'],errors='coerce');df=df.drop_duplicates(['date','chain'],keep='last').sort_values(['date','chain']);atomic(df,OUT)
 payload={'ok':True,'date':day,'rows':rows,'counts':{'chains':len(rows),'available':sum(r['status']=='available' for r in rows)},'history_path':str(OUT)};LATEST.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding='utf-8');return payload
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--date');a=ap.parse_args();r=collect(a.date);print(json.dumps({'ok':r['ok'],'counts':r['counts'],'path':str(OUT)},ensure_ascii=False));raise SystemExit(0 if r['ok'] else 1)
