from __future__ import annotations

import pandas as pd

from quant_system.broad_sample_backtest import BroadCase, run_case


def test_single_weight_cap_is_applied_without_negative_cash():
    dates = pd.date_range('2024-01-01', periods=45, freq='B')
    rows=[]
    for d in dates:
        for i in range(20):
            p=10+i/10
            rows.append({'date':d,'code':f'{i:06d}','open':p,'high':p*1.01,'low':p*.99,'close':p,'volume':100000,'amount':1000000,'inv_vol20':-0.01})
    panel=pd.DataFrame(rows)
    result=run_case(panel,BroadCase('cap','inv_vol20',1,.2,'monthly','none',0,1,0,.02))
    assert result['audit']=='PASS'
    assert result['negative_cash']==0
