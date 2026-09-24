# -*- coding: utf-8 -*-
"""盘中监控 API handler（quant_web）

GET /api/intraday → 六合一盘中快照（指数/涨停强度/主线/资金/风险）
复用 scripts/intraday_monitor_light.monitor_snapshot（纯本地轻量）。
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime as _datetime, timezone as _timezone, timedelta as _timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CST = _timezone(_timedelta(hours=8))


def handler_intraday(query: dict, send_json):
    try:
        import importlib.util as _ilu
        _p = ROOT / "scripts" / "intraday_monitor_light.py"
        _spec = _ilu.spec_from_file_location("intraday_monitor_light", _p)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        # 先确定请求/回放日期，再让监控采集器按该日期截断所有源；不能先取当前实时快照再拼历史链。
        requested_date = str((query.get('date') or [''])[0]).strip()
        if requested_date and not re.fullmatch(r'20\d{2}-\d{2}-\d{2}', requested_date):
            send_json({'ok': False, 'error': 'date 必须为 YYYY-MM-DD'})
            return
        if requested_date:
            today = requested_date
        else:
            try:
                from quant_system.market_clock import is_market_open_now, latest_completed_trading_day
                today = (_datetime.now(CST).date().isoformat() if is_market_open_now()
                         else latest_completed_trading_day().isoformat())
            except Exception:
                today = _datetime.now(CST).strftime('%Y-%m-%d')
        try:
            snap = _mod.monitor_snapshot(today)
        except TypeError:
            # 兼容旧采集器签名；有显式历史请求时不把无日期实时结果当成历史事实。
            snap = _mod.monitor_snapshot()
            if requested_date:
                snap = {"date": requested_date, "data_as_of": requested_date,
                        "status": "date_mismatch", "date_mismatch": True,
                        "error": "旧版监控采集器不支持历史日期，仅返回历史决策链"}
        chain = None
        try:
            paths = sorted(ROOT.glob('generated/intraday_chain_*.json'))
            eligible = [p for p in paths if p.stem[-10:] <= today]
            if eligible:
                selected = eligible[-1]
                chain = json.loads(selected.read_text(encoding='utf-8'))
                source_date = selected.stem[-10:]
                chain['_source_date'] = source_date
                chain['_date_mismatch'] = source_date != today
                chain['_date_status'] = 'stale' if source_date != today else 'available'
        except Exception:
            chain = None
        if chain:
            rows = (chain.get('decision_rows') or chain.get('opportunities') or chain.get('rows') or [])[:50]
            scanned = int(chain.get('n_scanned', 0) or 0)
            raw_coverage = chain.get('coverage') or {}
            evaluated = int(raw_coverage.get('signal_evaluated', scanned) or 0)
            snap['decision_chain'] = {
                'bias': chain.get('bias', {}),
                'opportunities': rows,
                'scanned': scanned,
                'candidates': int(chain.get('n_candidates', chain.get('n_opportunities', 0)) or 0),
                'as_of': chain.get('time') or chain.get('date'),
                'source_date': chain.get('_source_date'),
                'requested_date': requested_date or None,
                'date_mismatch': bool(chain.get('_date_mismatch')),
                'status': chain.get('_date_status') or ('date_mismatch' if chain.get('_date_mismatch') else 'available'),
                'coverage': {
                    'snapshot_rows': int(raw_coverage.get('snapshot_rows', scanned) or 0),
                    'signal_evaluated': evaluated,
                    'candidate_count': int(raw_coverage.get('candidate_count', chain.get('n_candidates', 0)) or 0),
                    'evidence_evaluated': int(raw_coverage.get('evidence_evaluated', 0) or 0),
                    'displayed': len(rows),
                    'ratio': round(evaluated / scanned, 4) if scanned else 0.0,
                    'status': raw_coverage.get('status') or ('complete' if scanned and evaluated == scanned else 'unavailable'),
                },
            }
        try:
            compact_path = ROOT / 'generated' / f"decision_snapshot_intraday_{today}.json"
            if compact_path.exists():
                snap['decision_snapshot'] = json.loads(compact_path.read_text(encoding='utf-8'))
            else:
                import sys as _sys
                _sys.path.insert(0, str(ROOT / 'scripts'))
                from unified_decision_snapshot import build_snapshot
                snap['decision_snapshot'] = build_snapshot(mode='intraday', date=today)
        except Exception as exc:
            snap['decision_snapshot_error'] = str(exc)[:120]
        send_json({'ok': True, 'data': snap})
    except Exception as e:  # noqa: BLE001
        send_json({"ok": False, "error": str(e)[:150]})