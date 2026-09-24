#!/usr/bin/env python3
"""Incremental factor revalidation and reversible negative-IC mining."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/'generated/factor_quality_registry.json'
OOS=ROOT/'generated/ic_report/IC_OOS_REPORT.json'
GEN=ROOT/'generated/factor_generator_report_2026-08-22.json'
OUT=ROOT/'generated/factor_revalidation.json'
HIST=ROOT/'generated/factor_revalidation_history.jsonl'


def norm(x): return str(x or '').lower().replace('_','')
def load(p):
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception: return {}

def run():
    reg=load(REG); oos=load(OOS); gen=load(GEN)
    if not reg.get('ok'): return {'ok':False,'error':'factor quality registry unavailable'}
    oos_map={norm(x.get('factor')):x for x in (oos.get('factors') or oos.get('stable_top') or [])}
    previous=load(OUT)
    prev_map={norm(x.get('factor')):x for x in previous.get('revalidated',[])}
    revalidated=[]
    for f in reg.get('factors',[]):
        if f.get('tier')!='observe': continue
        name=f.get('factor'); oo=oos_map.get(norm(name)); prev=prev_map.get(norm(name),{})
        if not oo:
            action='observe'; reason='本轮OOS无同名证据'; streak=int(prev.get('consistent_streak',0))
        else:
            direction=int(oo.get('direction') or f.get('direction') or 1)
            train=float(oo.get('ic_mean_train') or 0); test=float(oo.get('ic_mean_test') or 0)
            icir=abs(float(oo.get('icir') or 0)); raw_win=float(oo.get('winrate') or 0)
            adj_win=raw_win if direction>=0 else 1-raw_win
            sign_keep=bool(oo.get('sign_keep'))
            strong=sign_keep and abs(test)>=.005 and icir>=.5 and adj_win>=.52
            streak=(int(prev.get('consistent_streak',0))+1) if strong else 0
            if strong and streak>=2: action='promote_candidate'; reason='连续两轮OOS方向一致且质量达标'
            elif not sign_keep and abs(test)>=.02: action='retire'; reason='OOS严重方向翻转'
            elif strong: action='observe'; reason='本轮OOS达标，等待第二轮确认'
            else: action='observe'; reason='OOS证据不足，继续观察'
        revalidated.append({'factor':name,'action':action,'reason':reason,'consistent_streak':streak,
                            'oos':oo or None})

    reverse=[]
    for f in gen.get('factors',[]):
        ic=float(f.get('ic_mean') or 0); icir=float(f.get('icir') or 0); win=float(f.get('win_rate') or 0)
        days=int(f.get('n_days') or 0); cov=float(f.get('coverage') or 0)
        if ic>=0: continue
        adjusted_win=1-win
        quality=abs(ic)>=.02 and abs(icir)>=.3 and days>=120 and cov>=.8 and adjusted_win>=.55
        reverse.append({'factor':f.get('name'),'formula':f.get('formula'),'family':f.get('family'),
                        'raw_ic':ic,'raw_icir':icir,'raw_winrate':win,'proposed_direction':-1,
                        'adjusted_winrate':round(adjusted_win,4),
                        'action':'oos_candidate' if quality else 'discard',
                        'reason':'负IC稳定，反向后具备复验价值' if quality else '反向后仍未通过基础质量门槛'})
    reverse.sort(key=lambda x:(x['action']=='oos_candidate',abs(x['raw_icir'])),reverse=True)
    result={'ok':True,'schema_version':'factor-revalidation.v1','generated_at':datetime.now(timezone.utc).isoformat(),
            'oos_generated_at':oos.get('generated_at'),'counts':{'observe_checked':len(revalidated),
            'promote':sum(x['action']=='promote_candidate' for x in revalidated),
            'retire':sum(x['action']=='retire' for x in revalidated),
            'reverse_candidates':sum(x['action']=='oos_candidate' for x in reverse),
            'reverse_discarded':sum(x['action']=='discard' for x in reverse)},
            'revalidated':revalidated,'negative_ic_mining':reverse}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with HIST.open('a',encoding='utf-8') as fh: fh.write(json.dumps(result,ensure_ascii=False)+'\n')
    return result

if __name__=='__main__':
    out=run(); print(json.dumps({'ok':out.get('ok'),'counts':out.get('counts'),'path':str(OUT)},ensure_ascii=False)); raise SystemExit(0 if out.get('ok') else 1)
