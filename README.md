# comprehensive-quant-workbench — declassified public release

A research-only quantitative workbench with a Python backend, browser frontend,
strategy engine, audited A-share backtest accounting, deployment templates,
reusable financial-services skills, and optional RAG adapters.

The public strategy surface is deliberately limited to `rsi_reversal`,
`low_volatility`, and `momentum_12_1`. Generic engines and data gates remain
available for extension, while private concrete strategy implementations,
credentials, runtime data, personal information, and recovery directories are
excluded. See [DECLASSIFICATION_NOTICE_FULL.md](DECLASSIFICATION_NOTICE_FULL.md)
and [docs/RELEASE_SCOPE.md](docs/RELEASE_SCOPE.md).

## Run

```bash
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests/test_public.py tests/test_extended.py
bash start_quant.sh
```

The service binds to loopback by default. Deployment files are templates and
require credentials through environment variables or a secret manager. No
credentials or market data are bundled. RAG modules accept user-supplied local
sources; no personal knowledge export is included.

Research outputs are not investment advice or a promise of returns.
