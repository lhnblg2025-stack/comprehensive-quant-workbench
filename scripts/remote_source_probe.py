#!/usr/bin/env python3
"""remote_source_probe.py — 腾讯云远程数据源探测（pytdx/问财/巨潮 国内 IP）

本机(境外/虚拟机)连不上通达信/巨潮/问财 → 在腾讯云(124.223.219.237, 国内 IP)上探测。
由 push_to_server.py 推送到远程后执行:
  python C:\\quant\\scripts\\remote_source_probe.py
"""

from __future__ import annotations
import logging

import sys

# Windows GBK 兼容（远程服务器打印中文/符号不崩）
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception as e:
        logging.getLogger(__name__).error(f"[remote_source_probe] 操作失败: {e}", exc_info=True)


def probe() -> None:
    print("=" * 40)
    print("远程数据源探测 (腾讯云 国内IP)")
    print("=" * 40)

    # 1. pytdx 通达信
    try:
        from pytdx.hq import TdxHq_API
        servers = [("119.147.212.81", 7709), ("114.80.63.12", 7709)]
        ok = False
        for ip, port in servers:
            try:
                api = TdxHq_API()
                api.connect(ip, port, time_out=5)
                n = api.get_security_count(1)
                api.disconnect()
                print(f"[pytdx] ✅ {ip}:{port} 证券数={n}")
                ok = True
                break
            except Exception as e:
                print(f"[pytdx] ❌ {ip}:{port} {str(e)[:60]}")
        if not ok:
            print("[pytdx] ❌ 全部服务器失败")
    except ImportError:
        print("[pytdx] ❌ pytdx 未安装: pip install pytdx")

    # 2. 问财 pywencai
    try:
        import pywencai
        r = pywencai.get(query="市盈率小于20且ROE大于15%")
        if r is None:
            print("[wencai] ❌ 返回 None")
        else:
            n = len(r) if hasattr(r, "__len__") else 1
            print(f"[wencai] ✅ 返回 {n} 条")
    except ImportError:
        print("[wencai] ❌ pywencai 未安装: pip install pywencai")
    except Exception as e:
        print(f"[wencai] ❌ {str(e)[:80]}")

    # 3. 巨潮 cninfo
    try:
        import akshare as ak
        import datetime as _dt
        _s = (_dt.date.today() - _dt.timedelta(days=7)).strftime("%Y%m%d")
        _e = _dt.date.today().strftime("%Y%m%d")
        df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol="最新", market="沪深",
            start_date=_s, end_date=_e)
        print(f"[cninfo] ✅ 公告 {len(df)} 条")
    except Exception as e:
        print(f"[cninfo] ❌ {str(e)[:100]}")

    # 4. 东财人气榜（本机被风控，看国内 IP 是否通）
    try:
        import akshare as ak
        df = ak.stock_hot_rank_em()
        print(f"[guba热榜] ✅ 人气榜 {len(df)} 条")
    except Exception as e:
        print(f"[guba热榜] ❌ {str(e)[:80]}")


if __name__ == "__main__":
    probe()
