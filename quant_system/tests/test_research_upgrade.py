import json
import pandas as pd
import pytest

from quant_system.research_upgrade import ResearchBoundaries, evaluate_holdout, performance_report, robustness_report, select_candidates, split_frame
from quant_system.build_lifecycle_price_panel import build_lifecycle_panel
from quant_system.pit_factor_engine import build_pit_factors


def test_holdout_requires_freeze(tmp_path):
    b = ResearchBoundaries(); frame = pd.DataFrame({'date': ['2026-01-02'], 'code': ['000001'], 'score': [1.]})
    with pytest.raises(PermissionError): split_frame(frame, b, partition='holdout')
    with pytest.raises(PermissionError): evaluate_holdout(frame, b, frozen_manifest=tmp_path/'missing.json')
    p = tmp_path/'m.json'; p.write_text(json.dumps({'frozen': True, 'boundaries': {'holdout_start': b.holdout_start}}))
    assert len(evaluate_holdout(frame, b, frozen_manifest=p)) == 1


def test_candidate_selection_excludes_validation_and_holdout():
    b = ResearchBoundaries(); frame = pd.DataFrame({'date': ['2024-12-31','2025-01-02','2026-01-02'], 'score': [1, 9, 99]})
    assert select_candidates(frame, b, score_col='score', n=1).iloc[0].score == 1


def test_pit_factor_rejects_future_and_builds_factors():
    f = pd.DataFrame({'date': ['2024-01-01', '2024-01-01'], 'code': ['000001', '000002'], 'available_date': ['2023-12-01', '2023-12-01'], 'pe_ttm':[10, 20], 'pb':[1, 2], 'ps':[2, 4], '净资产收益率(%)':[10, 10], '资产的经营现金流量回报率(%)':[5, 5], '净利润增长率(%)':[8, 8], '资产负债率(%)':[40, 40], '流动比率':[2, 2], '加权每股收益(元)':[1, 1]})
    built = build_pit_factors(f)
    assert 'factor_quality' in built
    assert built.loc[built.code == '000001', 'factor_value'].iloc[0] > built.loc[built.code == '000002', 'factor_value'].iloc[0]
    f.loc[0,'available_date']='2025-01-01'
    with pytest.raises(ValueError): build_pit_factors(f)


def test_metrics_and_bootstrap_have_expected_contract():
    r = performance_report([.01, -.01, .005] * 20); assert {'sharpe','sortino','calmar','max_drawdown'}.issubset(r)
    x = robustness_report(pd.Series([.01, -.005, .002] * 40), blocks=20, block_length=3); assert x['status'] == 'PASS'


def test_metrics_keep_first_loss_and_expose_drawdown_durations():
    x = performance_report([-0.20, 0.0, 0.25], dates=['2024-01-01', '2024-06-01', '2025-01-01'])
    assert x['total_return'] == pytest.approx(0.0)
    assert x['max_drawdown'] < 0
    assert x['max_drawdown_duration_days'] >= 1
    assert x['max_drawdown_recovery_days'] >= 1


def test_lifecycle_builder_keeps_membership_and_rejects_qfq(tmp_path):
    panel, report = build_lifecycle_panel('data_warehouse/kline_raw', 'data_warehouse/kline_hfq', start='2018-01-01', end='2018-01-10', codes=['600236'])
    assert report['symbols'] == 1
    assert {'start_date', 'end_date', 'listing_status', 'turnover_rate'}.issubset(panel.columns)
    with pytest.raises(ValueError):
        build_lifecycle_panel('data_warehouse/kline', 'data_warehouse/kline_hfq', codes=['600236'])
