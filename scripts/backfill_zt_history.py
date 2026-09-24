#!/usr/bin/env python3
"""Backfill exact daily limit-up partitions on the domestic data node."""
from quant_system.analysis_core.zt_pool_history import fetch_em_window

if __name__ == "__main__":
    raise SystemExit(0 if fetch_em_window(days=12) >= 0 else 1)
