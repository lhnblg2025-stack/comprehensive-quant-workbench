#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本机海外跑 macro_overseas → 产物落盘(云端读)。2026-08-22"""
import sys, json
from pathlib import Path
sys.path.insert(0, "quant_system")
from quant_system.analysis_core.macro_overseas import analyze
r = analyze()
out = Path("generated/overseas_macro.json")
out.write_text(json.dumps(r, ensure_ascii=False, default=str), encoding="utf-8")
print(f"✅ macro产物已落盘: {out} ({len(str(r))}B)")
