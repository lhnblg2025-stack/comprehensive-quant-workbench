#!/usr/bin/env python3
"""Backtrader order-level reconciliation for the four predeclared candidates."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from quant_system.backtest_protocol import BacktestRequest
from quant_system.backtest_service import execute
from quant_web.handlers.strategy_lab import _execute_strategy_source, _load_asset_panel, _research_data_gate
from quant_web.handlers.strategy_templates import TEMPLATES
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'generated/strategy_lab/candidate_order_reconciliation.json'
CANDIDATES=('low_volatility','bollinger_reversion','rsi_reversal','momentum_12_1')

def main():
 panel,state,path=_load_asset_panel('stock')
 source,cutoff,gate=_research_data_gate(panel,state,path,years=5)
 source=source[source['_history_sessions']>=252].copy()
 context_start=pd.Timestamp('2021-08-31'); oos_start=pd.Timestamp('2023-01-01')
 context=source[(source.date>=context_start)].copy(); execution=source[source.date>=oos_start].copy()
 exec_state=None
 if state is not None:
  exec_state=state.copy(); exec_state['date']=pd.to_datetime(exec_state['date'],errors='coerce'); exec_state=exec_state[exec_state.date>=oos_start]
 results=[]
 for name in CANDIDATES:
  row={'candidate':name,'status':'failed','error':None,'admitted':False}
  try:
   fn=_execute_strategy_source(TEMPLATES[name]['source'],f'reconciliation:{name}')
   all_targets=fn(context,{'top_n':20,'rebalance_sessions':20})
   targets={pd.Timestamp(d):w for d,w in all_targets.items() if pd.Timestamp(d)>=oos_start}
   if not targets: raise ValueError('no_targets_in_oos')
   request=BacktestRequest.from_dict({'strategy':{'kind':'target_weights','family':'candidate','parameters':{'candidate':name}},'dataset_id':gate['dataset_sha256'],'mode':'research'})
   run=execute(request,execution,trade_state=exec_state,targets=targets,quality_path=path.parent/'pit_data_quality_report.json')
   if not run.get('ok'): raise ValueError(run.get('error') or 'canonical_backtest_failed')
   result,equity,trades=run['result'],run['equity'],run['trades']
   row.update({'status':'executed','canonical':True,'schema':run['schema'],'spec_hash':run['spec_hash'],'engine':result.engine,'execution_status':result.status,'target_dates':len(targets),'observations':int(result.observations),'orders':int(result.orders),'trades':int(result.trades),'rejected_orders':int(result.rejected_orders),'metrics':result.to_dict()})
  except Exception as exc: row['error']=f'{type(exc).__name__}:{str(exc)[:300]}'
  results.append(row)
 out={'schema':'strategy_lab_order_reconciliation/v1','status':'research_only','dataset_gate':gate,'protocol':{'context_start':str(context_start.date()),'oos_start':str(oos_start.date()),'fill':'next_open','lot_size':100,'trade_state':'merged_cloud_partial','candidates':len(CANDIDATES)},'results':results,'admitted_count':0,'blockers':['historical_trade_state_not_authoritative','historical_instruments_not_pit','capacity_model_pending'],'generated_at':datetime.now().astimezone().isoformat(timespec='seconds')}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding='utf-8');print(json.dumps({'output':str(OUT),'results':len(results),'executed':sum(r['status']=='executed' for r in results)},ensure_ascii=False))
if __name__=='__main__':main()
