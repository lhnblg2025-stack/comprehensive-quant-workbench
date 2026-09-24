# Quant system

The public quant engine provides data contracts, feature pipelines, risk controls, backtesting, and research governance. Private datasets and generated artifacts are intentionally excluded.

## Quick start

```bash
python3 -m quant_system.cli --help
python3 -m quant_system.strategy_engine --help
python3 -m quant_system.strategy_matrix_backtest --help
```

The approved concrete strategy surface is exposed by `quant_system.public_strategies`: `rsi_reversal`, `low_volatility`, and `momentum_12_1`. All other strategy research implementations and result artifacts are omitted from this release.
