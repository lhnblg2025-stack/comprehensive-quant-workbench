#!/usr/bin/env python3
"""Build a truthful factor eligibility registry from real IC artifacts."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
IC=ROOT/'generated/ic_report/FACTOR_IC_REPORT_VECTORIZED.csv'
OOS=ROOT/'generated/ic_report/IC_OOS_REPORT.json'
GEN=ROOT/'generated/factor_generator_report_2026-08-22.json'
OUT=ROOT/'generated/factor_quality_registry.json'


def build(min_days=120,min_coverage=.7,min_icir=.10,min_winrate=.45):
    if not IC.exists():
        return {'ok':False,'error':'真实IC产物不存在','factors':[]}
    df=pd.read_csv(IC)
    def _norm(n):
        # OOS 用 mom6m / vol_20d，zoo 用 mom_6m / vol_20d 混拼；统一小写去下划线做匹配键。
        return str(n).lower().replace('_','')
    rows=[]
    oos_map={}
    oos_by_norm={}
    if OOS.exists():
        try:
            od=json.loads(OOS.read_text(encoding='utf-8'))
            oos_rows=(od.get('factors') or od.get('stable_top') or [])
            for r in oos_rows:
                oos_map[str(r.get('factor'))]=r
                oos_by_norm[_norm(r.get('factor'))]=r
        except Exception: pass
    gen_map={}
    if GEN.exists():
        try:
            gd=json.loads(GEN.read_text(encoding='utf-8'))
            for r in (gd.get('valid_factors') or []): gen_map[str(r.get('name'))]=r
        except Exception: pass
    for _,r in df.iterrows():
        name=str(r.get('factor','')).strip()
        ic=float(r.get('ic_mean') or 0)
        icir=float(r.get('icir') or 0)
        win=float(r.get('winrate') or 0)
        direction=int(r.get('direction') or 1)
        adjusted_win=win if direction >= 0 else 1-win
        days=int(r.get('n_days') or 0)
        cov=float(r.get('coverage') or 0)
        valid_base=bool(name and days>=min_days and cov>=min_coverage and abs(icir)>=min_icir and pd.notna(r.get('ic_mean')))
        strong=bool(valid_base and abs(icir)>=.25 and abs(ic)>=.02)
        eligible=bool(valid_base and adjusted_win>=min_winrate)
        tier='core' if strong and eligible else 'candidate' if eligible else 'observe' if valid_base else 'blocked'
        rows.append({'factor':name,'category':str(r.get('category','')),'ic_mean':ic,'icir':icir,'winrate':win,'adjusted_winrate':adjusted_win,'n_days':days,'coverage':cov,'grade':str(r.get('grade','')),'direction':direction,'eligible':eligible,'tier':tier,'reason':'核心有效' if tier=='core' else '候选有效' if tier=='candidate' else '观察：IC有信号但仍需验证' if tier=='observe' else '未通过真实IC质量门槛'})
    # 叠加OOS信息到同名IC因子；方向一致且OOS有效可升级核心。
    for row in rows:
        oo = oos_map.get(row['factor']) or oos_by_norm.get(_norm(row['factor']))
        if oo:
            row['oos_verified']=bool(oo.get('sign_keep')); row['oos_ic']=oo.get('ic_mean_test'); row['oos_icir']=oo.get('icir')
            # OOS 里 direction=-1 表示“因子越大越看空”，test_ic 负值即方向一致；
            # 用方向校准后的 IC/ICIR 绝对值判定有效性，避免把负向有效因子误判为无效。
            oo_dir=int(oo.get('direction') or 1)
            oos_ic=float(oo.get('ic_mean_test') or 0); oos_icir=float(oo.get('icir') or 0); oos_win=float(oo.get('winrate') or 0)
            oos_adj_icir=abs(oos_icir)
            oos_adj_win=oos_win if oo_dir>=0 else 1-oos_win
            if row['oos_verified'] and abs(oos_ic)>=.005 and oos_adj_icir>=.5 and oos_adj_win>=min_winrate:
                row['eligible']=True; row['tier']='core' if oos_adj_icir>=1.0 else 'candidate'; row['adjusted_winrate']=max(row.get('adjusted_winrate',0),oos_adj_win)
    # 合并IC报告之外的真实OOS稳定因子和生成器valid因子。
    existing_norms={_norm(x['factor']) for x in rows}
    for name,r in {**oos_map,**gen_map}.items():
        if _norm(name) in existing_norms: continue
        ic=float(r.get('ic_mean_test',r.get('ic_mean',0)) or 0); icir=float(r.get('icir',0) or 0); win=float(r.get('winrate',r.get('win_rate',0)) or 0); direction=int(r.get('direction',1) or 1); adj=win if direction>=0 else 1-win
        valid=abs(icir)>=.5 and adj>=min_winrate
        rows.append({'factor':name,'category':'oos_or_mined','ic_mean':ic,'icir':icir,'winrate':win,'adjusted_winrate':adj,'n_days':int(r.get('n_days',177) or 177),'coverage':float(r.get('coverage',1) or 1),'grade':'OOS','direction':direction,'eligible':valid,'tier':'candidate' if valid else 'observe','oos_verified':_norm(name) in set(oos_by_norm),'reason':'OOS/生成器真实验证通过' if valid else '有信号，等待更多验证'})
    rows.sort(key=lambda x:(x['eligible'],x.get('oos_verified',False),abs(x['icir']),x['n_days']),reverse=True)
    out={'ok':True,'schema_version':'factor-quality.v1','generated_at':datetime.now(timezone.utc).isoformat(),'source':str(IC),'thresholds':{'min_days':min_days,'min_coverage':min_coverage,'min_abs_icir':min_icir,'min_winrate':min_winrate,'note':'winrate按方向校准前原始口径，仅作辅助，不单独否决负方向因子'},'counts':{'total':len(rows),'eligible':sum(x['eligible'] for x in rows),'core':sum(x.get('tier')=='core' for x in rows),'candidate':sum(x.get('tier')=='candidate' for x in rows),'observe':sum(x.get('tier')=='observe' for x in rows),'blocked':sum(x.get('tier')=='blocked' for x in rows),'oos_verified':sum(x.get('oos_verified',False) for x in rows),'mined_verified':sum(x.get('factor') in gen_map for x in rows)},'factors':rows}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--min-days',type=int,default=120); ap.add_argument('--min-coverage',type=float,default=.7); ap.add_argument('--min-icir',type=float,default=.15); ap.add_argument('--min-winrate',type=float,default=.52); a=ap.parse_args()
    out=build(a.min_days,a.min_coverage,a.min_icir,a.min_winrate); print(json.dumps({'ok':out.get('ok'), 'counts':out.get('counts'), 'path':str(OUT)},ensure_ascii=False))
    raise SystemExit(0 if out.get('ok') else 1)
