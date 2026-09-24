# Frontend–Backend Contract & Data-Veracity Audit — quant_web

Delivered by delegated audit subagent. Read-only; no files modified (except this notes file).

Scope: server.py (4211 lines), handlers/v11.py, handlers/risk.py, stock_analysis.py,
static/app.js (+ trader.js, research_dashboard.html, task-dashboard.js).

## Verified base contract
- `app.js` `api()` (L203–234) returns the FULL JSON payload (`return data`), NOT `.data`.
  Callers read `d.data` or `d.<bare>`. So contract consistency is per-caller.
- `api()` throws when `data.ok===false` (L232). Errors must carry `ok:false`.
- README stipulates every endpoint returns `{"ok":true,"data":{...}}`.

---

## P0 — none

---

## P1 — confirmed, real frontend–backend breaks or mass fake data

### B1. `/api/factor_ic` — backend nests payload under `data`, frontend reads bare.
- Backend: server.py:1241-1244 return `{"ok": true, "data": {factors, range, chart, n, source}}`.
- Frontend: app.js:4011-4026 reads `d.factors`, `d.range`, `d.chart` (bare top-level),
  `d.factors?.length`. `api()` returns the full payload, so `d.factors` is `undefined`.
- Result: factor-IC table always shows empty / "无数据"; range line shows "undefined";
  chart never renders. CONFIRMED P1 contract break.

### B2. `/api/factor_combine` — same nesting mismatch.
- Backend: server.py:1300-1303 return `{"ok": true, "data": {method, ic_mean, icir, ...}}`.
- Frontend: app.js:4041-4045 reads `d.method`, `d.ic_mean`, `d.icir`, `d.chart` (bare).
- Result: "方法=undefined, IC=undefined, ICIR=undefined"; chart never renders.
  CONFIRMED P1 contract break.

### B3. `/api/news_sentiment` — handler returns fields the frontend never reads; frontend
    expects news headline data the handler never produces.
- Backend server.py:690-697 returns `{symbol, sentiment, score, date, signals, source}`.
- Frontend app.js:2816-2824 reads `data.total` ("新闻条数" metric) and
  `data.headlines` (rendered into a news table). Neither key ever returned.
- Result: "新闻条数" always shows 0 and news table always empty.
- Also category 2 note: the handler is named "news_sentiment" but derives sentiment from
  factor means (mom/quality/valuation), NOT from news — so no news data is ever produced.
  CONFIRMED P1.

### B4. `/api/reports?mode=read` — report body key mismatch (frontend never finds content).
- Backend: server.py:1987 returns `{"ok": True, "report": {"path","name","content"}}`
  (verified _read_report at 3842 puts text under `content`, nested under `report`).
- Frontend: app.js:2664 `readReport` reads `d.content || d.data?.content || d.data || ''`.
  `d.content` and `d.data` are both undefined (real body lives at `d.report.content`),
  so `content=''` → the report body always renders blank.
- Result: report title/path shows but body is empty. CONFIRMED P1 contract break.

### B5. `/api/global/quotes`-cache stale partial write — server.py:1852.
- Inside the first iteration of the per-symbol loop, `_cache_set("global_quotes",
  {"ok":True,"data":result})` caches a 1-row partial payload under the real cache key
  (TTL 300). Next call can return the truncated 1-element payload. The correct full
  payload is only cached after the loop (L1868). CONFIRMED P1 (robustness / cache poison).

### F1. `/api/v4/portfolio/optimize` — random synthetic returns presented as real, no flag.
- server.py:2677-2685: when `<2` cached price series, generates
  `np.random.seed(42); np.random.normal(base, vol, 120)` for 5 hardcoded symbols and
  feeds into `compute_portfolio`, returning real-looking `expected_return/sharpe_ratio/
  weights`. The returned payload carries NO "demo"/"synthetic" marker (the `fallback` note
  is only set for the separate scipy-import fallback). CONFIRMED P1 fake metric.

### F2. `/api/v4/performance/factor_attribution` — hardcoded constants pass as computed.
- server.py:2834-2845 returns fixed weights 动量0.15/价值0.20/质量0.25/成长0.10/低波0.20/
  技术0.10 based only on `n=len(positions)`. Docstring says "Estimate factor attribution
  from current holdings" but no attribution is computed. CONFIRMED P1 fake metric.

### F3. `stock_analysis.py:323` — `total_mv_yi` always `0.0`.
- `"total_mv_yi": round(float(last_v.get("total_mv", 0) or 0) / 1e8, 1)`.
- Current `valuation` schema has NO `total_mv` column (verified on valuation/000001.parquet
  = date,peTTM,pbMRQ,psTTM,pcfNcfTTM). Rename at L174 covers PE/PB/PS but not total_mv.
  Market cap always 0.0 亿元. CONFIRMED P1 fake metric (dead field).

### F4. `stock_analysis.py:651` — M2 value mislabeled (reads a growth % as money supply).
- `m2_yearly.parquet`/`money_supply.parquet` are descending. `last=m2.iloc[0]` takes the
  newest row, then `last.iloc[-1]` reads the LAST COLUMN of that row =
  `流通中的现金(M0)-同比增长` = 11.8. Actual M2 quantity (3567108 亿元) never read.
  API reports `m2.value=11.8` — a percentage mislabeled as M2 money supply.
  CONFIRMED P1 wrong-unit/data-source.

### M1. `_handle_ml_signals` (plural, `/api/ml_signals`) fabricated `ml_score` — CONFIRMED,
    but DEAD CODE.
- server.py:2367-2382: on 15s thread timeout / predict failure, falls back to
  `data_store.get(code, days=120)` and synthesizes `ml_score =
  round(3.0-min(3.0,max(0,abs(_pct)/5)),2)` from 5-day price change %, a `recommendation`
  (+emojis) and `signals:["5日变化=...%(缓存估计)"]`, `top_contributors:[]`, returned in
  the same `rows` array as real predictions with `{"ok":True,"data":...}`.
- Honesty: ONLY the `signals` string carries "(缓存估计)". The UI-rendered `ml_score`/
  `recommendation` carry no provenance and ok=true.
- REACHABILITY vs user question: `grep ml_signals static/*` = ZERO. app.js calls only the
  SINGULAR `/api/ml_signal` (modes scan_batch/predict/scan/importance at 2143,2218,2238,
  2515). The plural route exists only at server.py:295-296. So `/api/ml_signals` is NOT
  reached by the current frontend (dead code) — the fabricated fallback is only hit via a
  direct HTTP request. Confirmed.

---

## P1 — confirmed, error paths that return ok:true (mask failures as success)
- server.py:998 `_handle_ml_model_info` except → `{"ok":True,"data":{"error":...,"trace":...}}`
- server.py:1055 `_handle_cache_status` except → `{"ok":True,"data":{"error":str(e)}}`
- server.py:1143 `_handle_factor_report` except →
  `{"ok":True,"data":{"error":...,"trace":...}}`
- handlers/risk.py:162-163 `handle_cache_status` except → `{"ok":True,"data":{"error":...}}`
  (the only exception-with-ok:true in risk.py; all other risk handlers are clean)

## P1 — potential (input crash / latent)
- handlers/v11.py:84 `int(k)` parsed before try → malformed ?k= crashes (frontend sends no k)
- handlers/v11.py:110 `float(capital)` parsed before try → malformed capital crashes

## P1 — potential (look-ahead / as-of)
- server.py:1123-1134 `_handle_factor_report` takes first non-empty date (usually today)
  as the report "date"
- server.py:1369/1394 `_handle_fama_macbeth` cache key bounds to today; hardcoded start
  "2026-05-01"; on cache-hit-or-failure returns STALE cached `ok:true` payload
- server.py:2055-2062 `_handle_fundamental` default date as-of = datetime.now()

---

## P2 — confirmed
- **Contract (key != data), each still works with its specific consumer:**
  /realtime (server.py:1511 `quotes/signals/count`; fr reads d.quotes) — consistent;
  /intraday (1524/1528 rows); /search_stock (1534/1540 results; trader.js d.results);
  /stock_profile (1552 profile); /ai_decision (1564); /warehouse_status (1572);
  /health (1602-1613 top-level); /audit (1631-1639); /macro_snapshot (1681 macro);
  /ai_analyze (1805 reply; fr d.reply — consistent); /factor_library (1819 **lib; fr
  d.factors/total/summary — consistent); /factor_scan (1834 scan; fr d.scan — consistent);
  /global_kline (1878); /alert_push (1940); /market_regime (1958); /portfolio (1966);
  /reports (1987/1989 report/reports; fr d.reports — consistent); /history (768 symbol/
  period/snapshot/rows); /scan (790); /backtest (806/816); /chat (827 reply); /optimize
  (853); /sector (888); /skills (2483-2498 skills/skill/result; fr d.skills — consistent);
  /watchlist_save (2513); /tasks (2573 stats/recent); /freshness (2595-2602);
  /paper/positions (2614); /paper/trades (2625); /paper/summary (2634); /paper/buy|sell
  (2893/2909 **result); /rag_search (3129 hits; fr d.hits — consistent);
  /watchlist GET (3875 symbols); /system/status (3896-3910 top-level).
- **ok:true on degraded:** server.py:2155 sector_rotation empty-data; server.py:2876 v4_health
  except; handlers/v11.py:56-69 handler_v11_daily except → returns ok:true with `emo={}`
  `card={error}`; handlers/risk.py:186 data_freshness="normal" hardcoded + uptime_hours:None.
- **Fake/placeholder in demo handlers (dishonest even within "demo"):**
  server.py:543-556 backtest_dashboard: strategy numbers are hardcoded base/dd arrays even
  when `source` claims a real engine; server.py:601-614 factor_analysis: quality/momentum
  are fabricated arithmetic even in the "real" branch (demo flag present).
  server.py:1298 factor_combine: bar-series data all constant `0.05`.
- **Robustness:** stock_analysis.py:274-276,307,393-402 NaN leaks → `_json_safe`→null and
  `_make_advice` renders "近5日日均成交额 NaN 亿元"; server.py global_kline:1875 `int(count)`
  unguarded; server.py:2374 `iloc[-1]["close"]` KeyError if no close column.
- stock_analysis.py:186 `load_financial` sorts by "date" which is only present when a
  "日期" column exists → KeyError on schema change.

## P2 — potential
- server.py:305-308 style sensors: /macro_snapshot ordering-heuristic slicing (pmi/m2
  descending vs shibor/lpr iloc[-1]) — stock_analysis.py:641-666. handlers/v11.py:151-178
  "index_daily.parquet" fallback claims 沪深300 but is a different/stale series;
  /v12/kline adjust=qfq echoed but never applied; chip profile ignores `days` window.
  server.py:905 portfolio_optimizer `fetch_daily(start="20260101")`.

## Clean / no defect found
- /market_attribution, /ml_ensemble, /ml_retrain, /cache_refresh, /financial_data,
  /feature_panel, /feature_store, /drift_check, /asset_allocation (server.py range 855-1457);
  /market_temp, /north_flow, /fundamental, /chart_patterns (1457-2109);
  /indices, /ml_signal (singular, honest), /v4_performance_summary (real FIFO metrics),
  /v4_risk_check, /alerts_status, /data_health, /tasks_status, /decision_chain,
  /market_pulse, /opportunity, /portfolio_risk, /macro_calendar, /earnings (2109-3206);
  /stock_lens single path.
- handlers/risk.py: handle_fama_macbeth/handle_risk_model/handle_asset_allocation/
  handle_task_result (correct ok:false on error).
- stock_analysis.py: _read_parquet, load_kline, add_ma, _score_profile (documented rules),
  _make_advice, turnover *100 (correct).

## Highest-impact fixes (recommended)
1. factor_ic / factor_combine: make backend+frontend agree (either strip `data` on backend
   or change app.js to `d.data.*`) — currently silently blank UI.
2. Remove /api/ml_signals fabricated fallback, or surface "(缓存估计)" at top level and use
   ok:false for the cache path.
3. v4/portfolio/optimize: add `"synthetic": true`/`"demo": true` when random returns used.
4. v4/performance/factor_attribution: compute real attribution or explicitly label constants.
5. stock_analysis: total_mv_yi read from a real column or drop; fix M2 to read the actual
   money-supply value; guard NaN (finite checks) before _make_advice.
6. Fix ok:true-on-exception handlers (ml_model_info/cache_status/factor_report/risk cache)
   to emit ok:false.
