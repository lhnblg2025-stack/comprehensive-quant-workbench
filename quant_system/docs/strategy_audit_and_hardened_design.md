# Strategy Audit And Hardened Research Design

## Verdict On Existing Results

The completed 90-strategy matrix is a useful broad proxy-data screen, not a tradable strategy set and not evidence that all 90 signals work. It must remain `research_only` for four reasons:

1. The source label is `deadline_pit_proxy_and_current_industry_proxy`, not actual release-date financial PIT and historical industry PIT.
2. The original signal return layer used close-to-next-close returns while the research contract declared next-open execution. This timing mismatch has now been corrected in code; prior rankings are invalid for promotion and must be rerun.
3. The original batch execution target was created at a session close and traded at the same session open, a direct look-ahead fill. Targets now execute at the next trading session open. Prior order-level outcomes are invalid for promotion and must be rerun.
4. Selecting from 90 correlated strategy variants after observing their results creates multiple-testing bias. A positive result in every variation is a data-quality warning, not a confirmation.

## What The Existing Screen Did Establish

The proxy screen did establish that low leverage, book-to-price, operating-cash-flow-to-price, and a value-quality composite are sensible candidates for a strictly controlled replication. It also established that 700-source / 500-minimum cross-sectional processing, industry caps, strict CSI300 alignment, cost stress, and raw next-open order accounting can run at scale.

## Corrected Timing Contract

At session `t` close:

1. Seal price, PIT financial availability, industry assignment, and factor values.
2. Rank the eligible universe and write an immutable signal artifact with `signal_date=t`.
3. Submit target orders only for session `t+1` open.
4. Use raw open prices for entry and raw close prices for marking.
5. Calculate fast signal labels as `open(t+2) / open(t+1) - 1`; use a holding-period label matching each rebalance horizon for research summaries.

No target may be filled at `t` open from information observed at `t` close.

## Data And Universe Requirements

Production research must use `actual_pubdate_and_historical_industry` quality:

- A release-level financial table with actual announcement dates, source document IDs, restatement/vintage identifiers, and at least 500 valid securities at each rebalance.
- Date-effective historical industry classification; a current snapshot is not permitted.
- Historical membership, listing-age, suspension, limit-lock, ST, delisting, liquidity, and corporate-action records.
- CSI300 total-return or price-return series aligned exactly to strategy dates.
- A separate liquidity/capacity report using amount, turnover, participation, and estimated market impact.

## Frozen Candidate Families

Before the next actual-PIT run, freeze only these three primary candidates:

| Candidate | Signal | Rebalance | Quantile | Rationale |
|---|---|---:|---:|---|
| Value core | book-to-price industry residual | monthly | 30% | Strong proxy IR, stable across monthly/weekly variants |
| Quality core | low leverage industry residual | monthly | 20% | Strong proxy screen and low turnover |
| Balanced core | value-quality composite industry residual | monthly | 20% | Better diversified exposure and drawdown profile in proxy screen |

All other factor/frequency/quantile combinations are discovery tests. They cannot be promoted based on the completed matrix.

## Statistical Protocol

1. Use at least three non-overlapping rolling OOS windows; report each window, not only aggregate OOS.
2. Select the quantile on validation only. The OOS period must not choose factor, direction, quantile, frequency, industry cap, or cost model.
3. Use a deflated-Sharpe or White reality-check family adjustment across every tested candidate and configuration.
4. Require factor IC direction stability by year and by market regime.
5. Require monthly and weekly sensitivity reports, but promote only the frozen monthly configuration unless both remain stable.
6. Separate development, validation, final OOS, and paper-trading periods. No reoptimization after final OOS begins.

## Execution Protocol

Order-level validation must include:

- Next-open target execution after sealed t-close signals.
- 100-share lot size, buy/sell commission, sell stamp duty, transfer fee, slippage, market-impact participation cap, suspended and limit-up/limit-down rejection.
- Corporate actions, delisting liquidation rules, cash interest, and unfilled-order persistence.
- A reconciliation report between signal and execution equity. Large differences block promotion rather than being explained away.
- Capacity tests at 10m, 50m, 100m, and 500m CNY. A candidate with more than 10% ADV participation is rejected for that capital level.

## Promotion Gate

A candidate is `candidate` only if all conditions pass:

```text
actual_pubdate_and_historical_industry data quality
raw_next_open_to_following_open signal basis
at least 3 rolling OOS windows
positive aggregate and majority-window OOS excess return
positive return under 2x costs
same direction in at least 2 declared quantiles
no industry/single-name/cash/capacity breach
execution reconciliation within a predeclared tolerance
multiple-testing adjusted significance passes
```

Otherwise its lifecycle state is `research_only`, `data_blocked`, or `rejected`.

## Immediate Implementation Changes

- Corrected `_prepare_panel()` to label raw next-open to following-open returns when raw opens exist.
- Corrected order targets to execute one session after the signal date.
- Added timing regression tests and a deterministic promotion-gate module.
- Invalidated the existing proxy screen for promotion. It remains archived as a discovery artifact.
- The next rerun must use actual release dates and historical industry intervals; the 2016-2019 data acquisition status remains `DATA_BLOCKED` until release coverage reaches 500 securities.
