# PIT Low-Frequency Factor Research Program

## 1. Scope And Non-Goals

This program evaluates A-share value, quality, and value-quality factors on a point-in-time (PIT) basis. The primary portfolio is long-only, monthly rebalanced, and formed from a historical investable universe of 500 to 1000 securities.

The research output is a factor validation result, not an investment recommendation. Existing OHLCV-only momentum, volatility, reversal, liquidity, and breakout signals remain separate research baselines; they must not be presented as evidence for a fundamental factor.

## 2. End-To-End Workflow

The stages below are ordered gates. A failure blocks the next stage and produces a machine-readable reason in the run report.

| Stage | Input | Required Work | Pass Output | Block Conditions |
|---|---|---|---|---|
| 0. Freeze research contract | Configuration and code version | Freeze factor definitions, costs, universe rules, benchmark, validation windows, and random seed | Versioned config and manifest template | Unversioned input, undefined timing, or mutable factor formula |
| 1. Acquire source data | Vendor files and source metadata | Ingest prices, membership, corporate actions, release-level financials, industry history, and benchmark | Raw source inventory with hashes | Missing source, missing coverage period, or unverifiable source date |
| 2. Validate PIT data | Canonical source tables | Validate schemas, uniqueness, date intervals, announcement dates, and historical membership | PIT data-quality report | Report period used as availability date, overlapping industry intervals, duplicate release, or missing benchmark date |
| 3. Build research panel | Validated sources and trade calendar | Join price, membership, corporate actions, financial releases, and industry by date-effective keys | Daily PIT panel and coverage report | Fewer than 500 usable names, future release observed, or current industry snapshot used |
| 4. Build factors | PIT panel | Calculate raw value/quality factors, winsorization/standardization, and industry-neutral residuals | Factor panel with factor lineage | Missing required inputs, insufficient industry group, or undefined factor direction |
| 5. Run signal research | Factor panel | Monthly formation, quantile portfolios, rolling IS/validation/OOS, cost and quantile stress | Signal-level OOS report | Fewer than three OOS windows, missing benchmark alignment, or industry-cap infeasibility |
| 6. Run execution validation | Frozen final OOS signal | Trade at next available open using raw execution prices, lot sizes, costs, corporate actions, and tradability flags | Order-level equity curve and reconciliation | Fill before signal availability, negative cash, unmatched corporate action, or reconciliation breach |
| 7. Decide lifecycle | All reports | Apply promotion gate and publish candidate/reject/archive state | Signed decision record | Any mandatory gate fails |

No stage may silently substitute a missing PIT table with a current snapshot, a report period, or a technical proxy.

## 3. Canonical Data Contracts

### 3.1 Price And Tradability Panel

One row per `date, code`; code is zero-padded six digits. Required columns are:

```text
date, code, raw_open, raw_high, raw_low, raw_close, close,
volume, amount, is_tradable, is_suspended, limit_up, limit_down
```

`close` is the factor/label price. `raw_*` fields are used only for executable order simulation. A record is investable only when it is inside the date-effective membership interval, is not suspended, has valid prices, and passes configured liquidity and listing-age filters. Delisted securities remain included until their own delisting or membership end date.

### 3.2 Historical Membership

Required columns:

```text
code, start_date, end_date, universe_id, source_as_of
```

Intervals for the same `code, universe_id` must not overlap. The production universe is a date-effective broad A-share universe, not current CSI constituents. Its coverage rule is 500 to 1000 valid names at every rebalance date. A run with fewer than 500 valid names is blocked; a run with more than 1000 names must use the frozen, deterministic sampling or liquidity-ranking rule recorded in the manifest.

### 3.3 Financial Release Table

Required columns:

```text
code, report_period, announcement_date, source_document_id,
source_as_of, eps_ttm, book_value_per_share,
operating_cashflow_per_share, roe, net_margin,
cfo_to_net_income, debt_to_assets
```

`announcement_date` is a public filing/release date, never an accounting period end. A release becomes available on the first trade session after `announcement_date`, with `availability_lag_sessions: 1`. A restatement is a new release and supersedes a previous value only from its own availability date. A null announcement date blocks that record from the PIT panel.

### 3.4 Historical Industry Table

Required columns:

```text
code, industry, industry_code, effective_date, end_date, source_as_of
```

Industry intervals must not overlap for an individual code. The assigned industry is the value whose effective interval contains the signal date. `market/sw_industry_map.parquet` is a current snapshot and is not a valid source for historical neutralization.

### 3.5 Corporate Actions And Benchmark

Corporate actions require `code, ex_date, action_type, ratio, source_as_of`. The benchmark requires a complete daily `date, return` return series. The configured CSI 300 series must cover every signal-return date; `require_alignment: true` blocks on any unmatched date.

## 4. PIT Transformation Rules

1. Normalize source identifiers and dates before joins; reject invalid or duplicate keys.
2. Filter each price row by historical membership before factor construction.
3. Join a financial item with an as-of join by code and `available_date`, never by report period.
4. Join industry through its date-effective interval.
5. Preserve `report_period`, `announcement_date`, `available_date`, and `source_document_id` in the materialized factor panel for audit.
6. Do not forward-fill a financial value beyond its next valid release if the previous release was withdrawn or invalidated.
7. Require at least five usable names in an industry when calculating industry residual factors.
8. Record raw input hashes, row counts, date ranges, coverage, and data-quality failures in the experiment manifest.

## 5. Factor Catalogue

All directions below mean that a higher score is preferred. Raw factors are calculated from only the latest available release at each date.

| Family | Factor | Formula | Direction |
|---|---|---|---:|
| Value | `value_earnings_yield` | `eps_ttm / close` | + |
| Value | `value_book_to_price` | `book_value_per_share / close` | + |
| Value | `value_ocf_to_price` | `operating_cashflow_per_share / close` | + |
| Value | `value_composite` | Equal-weight complete-case cross-sectional z-score of the three value factors | + |
| Quality | `quality_roe` | `roe` | + |
| Quality | `quality_net_margin` | `net_margin` | + |
| Quality | `quality_cash_conversion` | `cfo_to_net_income` | + |
| Quality | `quality_low_leverage` | `-debt_to_assets` | + |
| Quality | `quality_composite` | Equal-weight complete-case cross-sectional z-score of the four quality factors | + |
| Composite | `value_quality_composite` | Equal-weight z-score of value and quality composites | + |

For every raw factor, the production candidate is its contemporaneous industry residual, named `<factor>_industry_neutral`. Factor neutralization eliminates average industry-level exposure in the score; it does not replace a portfolio industry cap.

## 6. Portfolio And Strategy Matrix

The primary strategy matrix is fixed before seeing OOS results.

| Strategy family | Factor input | Rebalance | Quantiles | Purpose |
|---|---|---|---|---|
| Value | Three raw value residuals and value composite residual | Monthly | 10%, 20%, 30% | Test independent and composite value effects |
| Quality | Four raw quality residuals and quality composite residual | Monthly | 10%, 20%, 30% | Test profitability, cash, and leverage effects |
| Combined | Value-quality composite residual | Monthly | 10%, 20%, 30% | Test diversified fundamental signal |
| Robustness | The same frozen signals | Weekly | 10%, 20%, 30% | Frequency sensitivity only |

Primary portfolio rules: long-only, equal weight, default 20% quantile, 2% maximum single-name weight, 15% maximum industry weight, and uninvested cash when a constraint prevents full allocation. When equal weighting makes an industry cap infeasible, the run fails rather than relaxing the cap.

The matrix is evaluated as fixed rules. There is no unconstrained parameter search. A validation window can choose only among the predeclared 10%, 20%, and 30% quantiles for each factor.

## 7. Backtest Timing And Accounting

### Signal Timing

At session `t` close, calculate factors from prices through `t`, financial releases with `available_date <= t`, and industries effective on `t`. A rebalance order can be submitted only after this snapshot is sealed.

### Signal Research Return

The fast signal-research layer must use the same timing as the order layer: selection at `t` close, entry at session `t+1` raw open, and marking/exit at the configured holding-period close or the next rebalance. Close-to-next-close results are diagnostic only and cannot satisfy the promotion gate.

### Order-Level Return

The order-level layer uses next open fills, board lot size, commission, sell stamp duty, transfer fee, slippage, suspension/limit checks, volume participation, and corporate actions. It records submitted orders, fills, rejects, cash, positions, turnover, and daily equity.

### Costs

The base cost is the configured commission, stamp duty, transfer fee, and slippage. Each surviving candidate is also tested at 2x and 4x base costs. Cost scenarios are fixed before research begins.

## 8. Validation Protocol

The primary configuration is `configs/research/pit_value_quality_500_monthly.yaml`.

```text
Train:       756 trading days
Validation:  252 trading days
OOS:         126 trading days
Step:         63 trading days
Purge:        60 trading days
Embargo:       5 trading days
Minimum:       3 OOS windows
```

For each rolling window, factor formulas, industry neutralization, portfolio constraints, and transaction-cost assumptions remain frozen. Validation may select the quantile from the declared set. OOS performance is never used to choose a factor, direction, weighting, industry cap, or transaction-cost parameter.

Required report sections are coverage, IC and ICIR, gross and net return, annual return, volatility, Sharpe, maximum drawdown, turnover, benchmark excess return, tracking error, information ratio, cost stress, quantile sensitivity, rolling-window results, industry exposures, and execution reconciliation.

## 9. Promotion Gate And Lifecycle

A factor is `candidate` only when all requirements pass:

1. PIT data quality, announcement-date checks, industry history, membership history, and benchmark alignment are all `PASS`.
2. Every accepted rebalance date has at least 500 usable securities; each factor has documented median and minimum coverage.
3. At least three OOS windows exist and aggregate OOS excess return is positive.
4. The factor retains the configured direction in at least two of the three portfolio quantiles.
5. Net performance survives 2x costs; 4x-cost results are disclosed even when they fail.
6. No portfolio breaches single-name or industry caps, and no industry-cap feasibility exception occurs.
7. Order-level next-open performance reconciles to the signal layer within a predeclared tolerance. A material gap creates a `reconcile_failed` state.

Lifecycle states are `draft`, `data_blocked`, `research`, `candidate`, `paper`, `retired`, and `rejected`. Only `candidate` factors proceed to paper trading; no factor proceeds directly from a backtest to production.

## 10. Required Artifacts

Every completed run publishes:

```text
experiment_manifest.json
pit_data_quality_report.json
coverage_report.json
materialized_pit_factor_panel.parquet
factor_backtest_report.json
factor_backtest_report.md
rolling_oos_report.json
benchmark_alignment_report.json
industry_exposure_report.csv
order_level_backtest.json
signal_execution_reconciliation.json
lifecycle_decision.json
```

The manifest hashes every input and output artifact, records code version and configuration, and binds the decision record to the exact dataset and calculation.

## 11. Delivery Sequence

1. Deliver and validate the release-level financial table, historical industry intervals, historical membership, corporate actions, and daily benchmark.
2. Materialize the 500-1000-stock PIT price panel and run stage-2 data checks.
3. Produce the PIT factor panel and coverage report; review factor lineage before any performance analysis.
4. Run the frozen primary monthly matrix, then the predeclared robustness matrix.
5. Run order-level next-open validation for only factors that pass signal-level OOS gates.
6. Produce a lifecycle decision record; archive rejected factors with their exact failure reason.

## 12. Current Status

The implementation has PIT financial and industry contracts, factor construction, industry residuals, sample-floor enforcement, strict custom-benchmark alignment, and industry-cap selection tests. The formal production run remains `data_blocked` until the canonical historical release, industry, membership, corporate-action, and 500-1000-stock panel data are delivered. Existing `data_warehouse/financial/*.parquet` files lack announcement dates and are intentionally ineligible.
