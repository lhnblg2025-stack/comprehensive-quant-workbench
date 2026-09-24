"""Fetch annual historical Baostock industry snapshots for a fixed universe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _rows(query):
    rows=[]
    while query.error_code == "0" and query.next(): rows.append(query.get_row_data())
    return rows


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--price-panel',required=True); p.add_argument('--output',required=True); p.add_argument('--start-year',type=int,default=2016); p.add_argument('--end-year',type=int,default=2019)
    a=p.parse_args(); codes=set(pd.read_parquet(a.price_panel,columns=['code']).code.astype(str).str.zfill(6)); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    import baostock as bs
    login=bs.login()
    if login.error_code!='0': raise RuntimeError(login.error_msg)
    parts=[]
    try:
        for year in range(a.start_year,a.end_year+1):
            query=bs.query_stock_industry(date=f'{year}-01-04'); values=_rows(query)
            frame=pd.DataFrame(values,columns=query.fields)
            if frame.empty: continue
            frame=frame.rename(columns={'updateDate':'effective_date','industryClassification':'industry_code'})
            frame['code']=frame.code.astype(str).str.replace(r'^(sh|sz)\.','',regex=True).str.zfill(6)
            frame=frame[frame.code.isin(codes)][['code','industry','industry_code','effective_date']]
            frame['source_as_of']='baostock_stock_industry'
            parts.append(frame)
            print(json.dumps({'year':year,'rows':len(frame)},ensure_ascii=True),flush=True)
    finally: bs.logout()
    history=pd.concat(parts,ignore_index=True).sort_values(['code','effective_date'])
    history['effective_date']=pd.to_datetime(history.effective_date); history['end_date']=history.groupby('code').effective_date.shift(-1)-pd.Timedelta(days=1)
    history.to_parquet(out,index=False); print(json.dumps({'rows':len(history),'codes':history.code.nunique(),'output':str(out)},ensure_ascii=True)); return 0
if __name__=='__main__': raise SystemExit(main())
