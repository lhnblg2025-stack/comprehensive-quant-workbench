# Production Backtest Data-Source Plan

## Implemented

- Tushare `daily`: ten-year raw OHLCV execution prices for 500 securities.
- Tencent direct QFQ: adjusted signal-price augmentation for the same universe.
- AkShare `stock_report_disclosure`: actual annual disclosure dates.
- AkShare Sina financial statements: annual raw balance sheet, income statement and cash-flow values.
- Official SW workbook `StockClassifyUse_stock.xls`: historical effective industry intervals; six-digit leaves retained and level-one parent used for neutralization.
- Eastmoney `stock_fhps_detail_em`: implemented dividends and splits with ex-dates.
- SSE/SZSE delisting references.
- Historical amount-derived ADV and participation/impact stress fields.

## Required remaining source

The production engine requires an independent daily trade-state table with:

```text
code, date, suspended, suspension_reason, st_flag,
limit_up_price, limit_down_price, limit_up_locked, limit_down_locked,
source_document_id, source_as_of
```

Current Tushare token lacks `suspend_d`, `stk_limit`, and `stock_st` permissions. Public current ST and OHLC endpoints are insufficient for historical state. Therefore the gate remains blocked and the order-level replay cannot be promoted.

## Correct replay behavior

The common engine is `quant_system/portfolio_holding_backtest.py`:

- signal at t close;
- target order at t+1 open;
- non-overlapping holding intervals;
- 100-share lots;
- historical ADV participation cap;
- commission, stamp duty, transfer fee and slippage;
- limit-up entry and suspension rejection;
- limit-down/suspension exit marked as blocked;
- insufficient history or blocked execution produces `annual_return=null`, `sharpe=null`, and `execution_status=incomplete_execution`.

No proxy trade-state table may be used to change this status.
