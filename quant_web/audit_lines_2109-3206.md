# Audit — server.py lines 2109-3206 (all listed handlers), README, static/* caller verification

Read server.py:2109-3206, README.md, static/app.js, static/paper_trading.html,
static/v4_dashboard.html, static/task-dashboard.js, static/sse-client.js, static/ui9.js.
No files modified.

## SPECIFIC CASE: `_handle_ml_signals` (plural, /api/ml_signals)

**(a) Fabricated ML score — CONFIRMED.** On thread timeout (15s `_t.join(timeout=15)`) or
any predict exception, the fallback (server.py:2367-2385) calls
`data_store.get_store().get(code, days=120)` and synthesizes a row from cached price
history, returned inside the SAME `rows` array as real `predict()` output with
`{"ok": True, "data": ...}`:

```
ml_score:       round(3.0 - min(3.0, max(0, abs(_pct) / 5)), 2)   # 0-3 scale from 5-day % move
recommendation: "🟢 持有观察" if _pct>0 else "🟡 谨慎" if _pct>-3 else "🔴 回避"
signals:        [f"5日变化={_pct:+.1f}%(缓存估计)"]
top_contributors: []
```

A real→fake blend in one list; a caller cannot tell which rows are model output vs this
cache estimate except by parsing the "(缓存估计)" marker inside the signals string.

**(b) Dead code — CONFIRMED.** `grep "ml_signals" static/` returns ZERO hits in any
js/html. app.js only calls the SINGULAR `/api/ml_signal` (modes scan_batch/predict/scan/
importance at lines 2143, 2218, 2515, 2238). The plural route exists only at
server.py:295-296. **So `/api/ml_signals` is dead code reached by no caller today.**
The fabricated fallback is only reachable via a direct HTTP request.

**Honesty of fallback labeling — PARTIAL.** The signals string literally contains
"缓存估计" (honest marker), but the fields the UI actually renders (`ml_score`,
`recommendation`) carry NO provenance marker and ok is `true`, so the synthetic
0-3 score is presented as a real ML prediction. Also, if plural WERE wired to
`renderMLSignalPredict`/`renderMLSignalList` (app.js:2249-2274), the 0-3 `ml_score`
would be rendered `score.toFixed(1)+"%"` and used as a bar-width percentage — a fake
value AND unit/scale mismatch.

## Full structured findings

### FAKE / INVENTED METRICS
- **server.py:2677-2685 — `_handle_v4_portfolio_optimize` synthetic "demo" returns. CONFIRMED. P1.**
  When fewer than 2 real cached price series exist, it fabricates
  `np.random.seed(42); all_rets[sym]=np.random.normal(base, vol, 120)` and feeds it into
  `compute_portfolio`, returning `{"ok":True,"data":{expected_return,sharpe_ratio,weights,...}}`.
  The returned payload carries NO flag indicating synthetic/demo. `payload["fallback"]` is only
  set when the scipy import fails, NOT when the input data is random noise. A user sees optim
  numbers derived from seeded random noise presented as real-price analysis.
- **server.py:2834-2845 — `_handle_v4_factor_attribution` hardcoded constants. CONFIRMED. P1.**
  Returns fixed weights `{"动量":0.15,"价值":0.20,"质量":0.25,"成长":0.10,"低波":0.20,"技术":0.10}`
  (sum=1.0; all 0 when n==0). The docstring says "Estimate factor attribution from current
  holdings" but the only real input is `"n": len(positions)`. No computed attribution.
- **server.py:2375-2382 — fabricated `ml_score` fallback (dead endpoint). CONFIRMED. P1-code/P2-reachability.**
- **server.py:2704-2723 — `_numpy_portfolio_fallback` sets `payload["fallback"]=note` when scipy
  unavailable; properly labeled. NO defect.**

### CONTRACT BREAK — returned key != "data" (against documented `{ok,data:{...}}` envelope)
All `ok` truthful; the envelope shape deviates. P2 each (each still works with its specific consumer):
- 2513 `_handle_watchlist_save` → `{"ok":True,"symbols","count"}` (app.js:297 reads d.count)
- 2573 `_handle_tasks` → `{"ok":True,"stats","recent"}` (task-dashboard reads .stats/.recent)
- 2595-2602 `_handle_freshness` → `{ok,"entries","total_entries","hits","misses","hit_rate_pct"}`
- 2614 `_handle_paper_positions` → `{ok, positions, summary}`
- 2625 `_handle_paper_trades` → `{ok, trades}`
- 2634 `_handle_paper_summary` → `{ok, summary}`
- 2893/2909 `_handle_paper_buy/sell` → `{"ok":True, **result}`
- 2483-2498 `_handle_skills` → `{ok,count,skills,featured}` / `{ok,skill}` / `{ok,result}`
- 3129-3132 `_handle_rag_search` → `{ok,"query","hits"}`

### CONTRACT BREAK — `ok:true` on error/degraded
- **server.py:2155 `_handle_sector_rotation`** → `{"ok":True,"data":[],"note":"数据不可用"}` —
  ok:true with empty data on total failure. P2 (data genuinely empty, not fabricated; borderline).
- **server.py:2876 `_handle_v4_health` except** → `{"ok":True,"data":{"status":"unknown","error":str(e)}}`
  swallows exception as ok:true. P2.

### CRASH / ROBUSTNESS
- **server.py:2374 — POTENTIAL. P3.** Fallback uses `_df.iloc[-1]["close"]` and `_df.iloc[-5]["close"]`
  by literal column; a frame without a `close` column raises KeyError → caught by outer try (2392)
  → 500 ok:false. The `len(_df)>=5` ternary correctly prevents the iloc[-5] IndexError.
- server.py:2557-2562 `_handle_sse` — `time.sleep(30)` ties one thread per SSE client forever. P3 potential.
- Cleared (no defect): `_exc_holder[0]` (guarded), `_handle_indices` per-source try/except,
  `_handle_alerts_status/_data_health/_tasks_status/_decision_chain` file/DB reads,
  `_handle_v4_performance_summary` (real FIFO/trade metrics, honest), `_handle_v4_risk_check`
  (passes real keys through), `_handle_ml_signal` SINGULAR (real predict; connection failures →
  honest `{"ok":False,"error":"模型服务未连接"}` status 200, no fabricated fallback),
  `_handle_market_pulse/opportunity/portfolio_risk/macro_calendar/earnings/regime`.

## Recommended priority
1. Remove `/api/ml_signals` or expose its "(缓存估计)" provenance at top level and not as a fake
   `ml_score`; add ok:false for the empty/cache path if kept reachable.
2. `/api/v4/portfolio/optimize`: surface a `"demo":true`/`"synthetic":true` flag whenever the
   random-data fallback is used (currently silent).
3. `/api/v4/performance/factor_attribution`: either compute real attribution or label the
   constants explicitly as placeholder/estimate.
4. (Optional) standardize the many non-`data` envelopes to `{ok,data}` per README, or update README.

---

## 修复状态（2026-08-15 更新）

已按本审计结果修改 `quant_web/server.py`：

- [x] `/api/ml_signals`：移除缓存价格伪造 `ml_score`/`recommendation`；失败如实进 `failures`，全失败时返回 `ok:false`。
- [x] `/api/v4/portfolio/optimize`：随机收益回退时返回 `"synthetic": true` 并带 `fallback` 说明。
- [x] `/api/v4/performance/factor_attribution`：常量占位显式标注 `is_placeholder: true` / `mode: "placeholder"` / `note`。
- [x] `/api/sector_rotation`：数据源全部不可用时返回 `ok:false`，不再以空数组冒充成功。
- [x] `/api/v4/health`：异常时返回 `ok:false`，不再吞错为 `ok:true`。

新增回归测试：`tests/test_audit_lines_fix.py`（6 用例）。
