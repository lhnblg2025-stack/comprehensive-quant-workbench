# Open-Source Backtest Integration

## Engine Roles

- `factor_backtest_runner.py`: batch cross-sectional factor discovery and IC diagnostics.
- `portfolio_holding_backtest.py`: canonical non-overlapping holding-period replay with explicit ADV, lot size, fees, slippage, blocked exits and incomplete-execution status.
- `backtrader_adapter.py`: independent Backtrader order-level cross-check using the same RAW OHLCV panel and target weights.
- `backtest_protocol.py`: engine-neutral result schema and reconciliation comparison.

Backtrader is optional at runtime because it is already installed in the current environment. It is an independent execution check, not a replacement for the factor engine or a historical data provider.

## Workflow

```text
1. Build factors / ML predictions
2. Convert signals to target weights
3. Run portfolio_holding_backtest.py for non-overlapping research replay
4. Run backtrader_adapter.py on the same target weights
5. Compare UnifiedBacktestResult values
6. Investigate gaps, rejected orders and blocked exits
7. Only then run rolling OOS and promotion gates
```

## Cross-Check Evidence

`generated/runs/backtest_engine_crosscheck.json` records a deterministic 2024 Q1, 20-stock comparison:

- portfolio-holding total return: -7.4740%
- Backtrader total return: -7.2839%
- return gap: 0.1901 percentage points
- relative final-value gap: 0.2054%
- Earlier bounded comparison (2024 Q1, 20 stocks) status: `PASS` within a deliberately broad diagnostic tolerance.
- Current UI recent-window comparison uses same-window targets and a 5% tolerance; it currently returns `BLOCK` with a 19.09 percentage-point return gap and 33.12% final-value gap. This is an unresolved engine reconciliation issue, not a pass.

Backtrader reported rejected orders in the bounded test. This is retained as execution evidence and must not be converted into assumed fills. Production promotion remains blocked until the strict current-window comparison passes.

## Production Data Gate

The integrated engines still require an authoritative daily trade-state table before production promotion:

```text
code, date, suspended, suspension_reason, st_flag,
limit_up_price, limit_down_price, limit_up_locked, limit_down_locked,
source_document_id, source_as_of
```

The current dataset has prices, annual PIT financials, official historical SW industries, corporate actions, delisting references and capacity, but the Tushare token lacks `suspend_d`, `stk_limit`, and `stock_st`. The quality gate therefore remains `DATA_BLOCKED` until this source is supplied.
