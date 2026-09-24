#!/usr/bin/env python3
"""Audit coverage and freeze the next factor study before any 2026 read."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant_system.research_upgrade import ResearchBoundaries, audit_point_in_time, freeze

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', default='generated/long_price_panel/price_panel.parquet')
    ap.add_argument('--pit-panel', default='data_warehouse/research_panels/ashare_tushare_500_10y_v2/annual_pit_research_panel.parquet')
    ap.add_argument('--out', default='generated/research_upgrade_2026')
    args = ap.parse_args()
    panel_path = ROOT / args.panel; pit_path = ROOT / args.pit_panel; out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(panel_path, columns=['date','code','raw_close','is_tradable'])
    boundaries = ResearchBoundaries()
    audit = audit_point_in_time(panel, boundaries)
    coverage = {'rows': len(panel), 'symbols': int(panel.code.nunique()), 'start': str(pd.to_datetime(panel.date).min().date()), 'end': str(pd.to_datetime(panel.date).max().date()), 'pit_panel_exists': pit_path.is_file()}
    if pit_path.is_file():
        pit = pd.read_parquet(pit_path)
        coverage.update({'pit_rows': len(pit), 'pit_symbols': int(pit.code.nunique()) if 'code' in pit else None, 'pit_columns': list(pit.columns)[:50]})
    report = {'schema':'research_upgrade_audit.v1','boundaries':boundaries.__dict__,'coverage':coverage,'quality':audit,'holdout_note':'2026 data was previously observed in the earlier study; this manifest protects future changes but cannot retroactively restore blindness.'}
    (out/'coverage_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    freeze(out/'freeze_manifest.json', boundaries=boundaries, inputs=[panel_path, pit_path], config={'coverage':coverage})
    print(json.dumps(report,ensure_ascii=False,indent=2,default=str))
    return 0
if __name__ == '__main__': raise SystemExit(main())
