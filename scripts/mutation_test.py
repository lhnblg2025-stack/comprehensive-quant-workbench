#!/usr/bin/env python3
"""突变测试 v3 — 重指向 quant_system（V11 项7b，用户硬需求#6）

向 quant_system/quant_platform 核心函数注入典型 bug，验证现有测试能否抓出。
安全: 每次突变前备份文件，测试后恢复原文件。
用法: python3 scripts/mutation_test.py [--quick]
"""
import shutil, subprocess, sys, os, time, shlex
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
os.chdir(ROOT)

MUTATIONS = [
    # ── ic_factors registry: 方向/激活翻转 ──
    ("quant_system/ic_factors/registry.py",
     "description=description, direction=direction, active=active,",
     "description=description, direction=-direction, active=active,  # 突变: 因子方向翻转",
     "因子方向翻转（方向+1→-1）",
     "python3 -m pytest tests/test_ic_factors_v7.py -q -p no:cacheprovider"),
    ("quant_system/ic_factors/registry.py",
     "_REGISTRY[fname] = meta",
     "pass  # 突变: 因子不再注册",
     "因子注册被移除（注册表缺因子）",
     "python3 -m pytest tests/test_ic_factors_v7.py -q -p no:cacheprovider"),
    # ── data_store: get_dataset 日期过滤 ──
    ("quant_system/data_store.py",
     'date_cols = [c for c in df.columns if str(c) in (',
     'date_cols = []  # 突变: 日期列探测失效',
     "日期过滤失效（返回全量数据）",
     "python3 -m pytest tests/test_v3_new_modules.py -q -p no:cacheprovider -k dataset"),
    # ── market_segments: 涨跌停阈值 ──
    ("quant_platform/market_segments.py",
     "0.098",
     "0.099  # 突变: 主板涨停阈值 9.8%→9.9%",
     "主板涨停阈值偏移（9.8%→9.9%）",
     "python3 -m pytest tests/test_v3_new_modules.py -q -p no:cacheprovider -k segment"),
    # ── openclaw_api: 代码归一化 ──
    ("quant_platform/openclaw_api.py",
     '    for pre in ("sh", "sz", "bj"):\n        if c.startswith(pre) and c[len(pre):].isdigit():\n            c = c[len(pre):]\n            break',
     "    pass  # 突变: 前缀剥离逻辑被移除",
     "代码前缀剥离失效（sh600519 不再归一化）",
     "python3 -m pytest tests/test_v3_new_modules.py -q -p no:cacheprovider -k norm_code"),
    # ── audit: 涨跌幅异常阈值 ──
    ("quant_platform/audit.py",
     "abs(float(p)) > 25",
     "abs(float(p)) > 250  # 突变: 阈值放宽 10 倍",
     "涨跌幅异常阈值 25%→250%",
     "python3 -m pytest tests/test_v3_new_modules.py -q -p no:cacheprovider -k audit"),
    # ── lhb_analyst: 净买额列 ──
    ("quant_platform/openclaw_api.py",
     'buy_col = next((c for c in ("龙虎榜净买额", "净买额", "净买入额", "净额", "总买卖净额", "net_buy") if c in sub.columns), None)',
     "buy_col = None  # 突变: 净买额列探测失效",
     "龙虎榜净买额列探测失效",
     "python3 -m pytest tests/test_v3_new_modules.py -q -p no:cacheprovider -k lhb_recent"),
]


def run_cmd(cmd, timeout=260):
    try:
        p = subprocess.run(shlex.split(cmd), capture_output=True, text=True,
                           timeout=timeout, cwd=ROOT,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        tail = (p.stdout or p.stderr).strip().splitlines()
        return p.returncode, (tail[-1][:110] if tail else "")
    except subprocess.TimeoutExpired:
        return 99, "TIMEOUT"


def main():
    quick = "--quick" in sys.argv
    results = []
    for i, (fpath, old, new, desc, cmd) in enumerate(MUTATIONS):
        if quick and i % 2 == 1:
            continue
        if not os.path.exists(fpath):
            print(f"[{i:02d}] ⚠️ 文件不存在: {fpath} 跳过")
            continue
        bak = fpath + ".mutbak"
        shutil.copy2(fpath, bak)
        try:
            src = open(fpath, encoding="utf-8").read()
            if old not in src:
                print(f"[{i:02d}] ⚠️ 突变片段未找到: {desc[:28]}... 跳过")
                continue
            open(fpath, "w", encoding="utf-8").write(src.replace(old, new, 1))
            t0 = time.time()
            code, tail = run_cmd(cmd)
            dt = time.time() - t0
            killed = (code != 0)
            status = "✅ 抓到" if killed else "❌ 漏掉"
            results.append((killed, desc, cmd, code, dt))
            print(f"[{i:02d}] {status} ({dt:.0f}s, exit={code}) {desc}")
            print(f"       尾部: {tail}")
        finally:
            # W2.5 P2: 恢复放 finally，中途异常/被杀也不残留突变源码
            if os.path.exists(bak):
                shutil.copy2(bak, fpath)
                os.remove(bak)
    killed_n = sum(1 for r in results if r[0])
    total = len(results)
    print(f"\n=== 突变测试汇总: {killed_n}/{total} 被测试抓到 ({killed_n/total*100:.0f}%) ===")
    for k, desc, cmd, code, dt in results:
        if not k:
            print(f"  ❌ 漏网突变: {desc}")
            print(f"     测试命令: {cmd}")
    if total and killed_n == total:
        print("  🎉 全部突变被测试抓到——测试有效！")


if __name__ == "__main__":
    main()
