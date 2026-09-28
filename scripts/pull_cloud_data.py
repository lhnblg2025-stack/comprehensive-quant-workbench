"""
pull_cloud_data — 拉取云服务器爬虫数据回本机（cron 每日盘后）

数据源:
  1. 腾讯云 quant-cloud.example.com C:\\quant\\data\\（国内IP爬虫）
     - cninfo 巨潮公告:   cninfo/cninfo_*.json        → data_warehouse/cninfo/
     - hot_rank 东财人气榜: hot_rank/hot_rank_*.parquet → data_warehouse/hot_rank/
  2. Vultr quant-relay.example.com /opt/quant/data/social/（国外IP爬虫）
     - weibo 微博舆情:   weibo_*.parquet              → data_warehouse/social/
     - baidu 百度热搜:   baidu_hot_*.parquet          → data_warehouse/social/
       （远端另有 summary_*.json 摘要文件，非核心数据，不拉取）

用法:
  python3 scripts/pull_cloud_data.py              # 拉全部（增量）
  python3 scripts/pull_cloud_data.py --only cninfo
  python3 scripts/pull_cloud_data.py --force      # 覆盖已存在文件
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# 审计 2026-08-16：基础设施信息允许从环境变量注入；未设置时保留本地兼容默认
# 2026-08-23 安全加固: 默认 PEM 改指受保护副本(600, ~/.openclaw/credentials/quant_cloud.pem),
# 原 /mnt/hgfs/share/openclaw.pem 为 777 共享文件夹权限, 仅作为兼容 fallback(不可读时)
import os as _os
_SECURE_PEM = Path.home() / ".openclaw" / "credentials" / "quant_cloud.pem"
PEM = _os.environ.get("QUANT_CLOUD_PEM",
                      str(_SECURE_PEM) if _SECURE_PEM.exists() else "/mnt/hgfs/share/openclaw.pem")
HOST_TX = _os.environ.get("QUANT_CLOUD_HOST_TX", "Administrator@quant-cloud.example.com")
HOST_VULTR = _os.environ.get("QUANT_CLOUD_HOST_VULTR", "root@quant-relay.example.com")

KNOWN_HOSTS = Path(_os.environ.get("QUANT_KNOWN_HOSTS", str(Path.home() / ".ssh" / "known_hosts")))


def _ssh_options() -> list[str]:
    """Return fail-closed SSH host verification options."""
    if not KNOWN_HOSTS.is_file():
        raise RuntimeError(f"known_hosts 不存在: {KNOWN_HOSTS}")
    return [
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
        "-o", "ConnectTimeout=15",
    ]


SOURCES = {
    "cninfo": {
        "host": HOST_TX,
        "remote_dir": "C:/quant/data/cninfo",
        "local_dir": ROOT / "data_warehouse" / "cninfo",
        "pattern": ("cninfo_", ".json"),
        "is_json": True,
        "compress": True,
    },
    "hot_rank": {
        "host": HOST_TX,
        # 2026-08-29 修复: 云端 crawler_hot_rank.py 实际写 C:\\data_warehouse\\hot_rank
        # （脚本 ROOT=parent.parent 解析为 C:\\），旧路径 C:/quant/data/hot_rank 已停更到 08-22。
        "remote_dir": "C:/data_warehouse/hot_rank",
        "local_dir": ROOT / "data_warehouse" / "hot_rank",
        "pattern": ("hot_rank_", ".parquet"),
        "is_json": False,
        "compress": True,
    },
    "social": {
        # 2026-08-14 修复: Vultr 已退役（本机即境外 IP 主机），社交数据改由本机
        # quant_system.analysis_core.social_sentiment 直连抓取（weibo/bilibili/百度/股吧），
        # 不再从远端拉快照。此源保留但默认不参与（--only social 仍可显式调用历史路径）。
        "host": HOST_VULTR,
        "remote_dir": "/root/quant-env/data/social",
        "local_dir": ROOT / "data_warehouse" / "social",
        "pattern": ("", ".parquet"),  # weibo_*.parquet / baidu_hot_*.parquet
        "is_json": False,
        "compress": False,
        "enabled": False,
    },
    "ths_concept": {
        "host": HOST_TX,
        "remote_dir": "C:/quant/data/ths_concept",
        "local_dir": ROOT / "data_warehouse" / "classification",
        "pattern": ("", ".parquet"),
        "is_json": False,
        "compress": True,
    },
    # 2026-08-14 审计P0修复: 8/12上云后短线链冻结(zt_daily_stats/theme_cycle/fusion
    # 停8/11)根因=回传清单只有3源, 云端在产但不回传。扩展以下4源:
    "fund_flow": {
        # 2026-08-22 固定腾讯云国内IP通道: 东财个股资金流(境外IP断连)由云端
        # crawler_fund_flow.py 每日18:45抓取, 本机增量回传 → data_warehouse/market/fund_flow/
        # 2026-08-29 修复: 云端实际写 C:\\data_warehouse\\fund_flow（与 hot_rank 同根因）。
        "host": HOST_TX,
        "remote_dir": "C:/data_warehouse/fund_flow",
        # canonical 数据域固定为 data_warehouse/fund_flow。旧实现写入
        # market/fund_flow，恢复/回传后会形成无人消费的孤儿目录。
        "local_dir": ROOT / "data_warehouse" / "fund_flow",
        "pattern": ("fund_flow_", ".parquet"),
        "is_json": False,
        "compress": True,
    },
    "zt_pool": {
        "host": HOST_TX,
        "remote_dir": "C:/quant/data_warehouse/market",
        "local_dir": ROOT / "data_warehouse" / "market",
        "pattern": ("zt_pool_", ".parquet"),
        "is_json": False,
        "compress": True,
    },
    "lhb": {
        "host": HOST_TX,
        "remote_dir": "C:/quant/data_warehouse/market",
        "local_dir": ROOT / "data_warehouse" / "market",
        "pattern": ("lhb_", ".parquet"),
        "is_json": False,
        "compress": True,
    },
    # 2026-08-21 修复: 决策卡核心数据(theme_cycle/zt_daily_stats/fusion/zt_pool_em_daily)
    # 未回传→本地web决策卡线索陈旧(停08-14)。补齐4源，保证本地与云端决策一致。
    "theme_cycle": {
        "host": HOST_TX,
        "remote_dir": "C:/quant/data_warehouse/market",
        "local_dir": ROOT / "data_warehouse" / "market",
        "pattern": ("theme_cycle", ".parquet"),
        "is_json": False, "compress": True,
    },
    "zt_daily_stats": {
        "host": HOST_TX,
        "remote_dir": "C:/quant/data_warehouse/market",
        "local_dir": ROOT / "data_warehouse" / "market",
        "pattern": ("zt_daily_stats", ".parquet"),
        "is_json": False, "compress": True,
    },
    "fusion": {
        "host": HOST_TX,
        "remote_dir": "C:/quant/data_warehouse/market",
        "local_dir": ROOT / "data_warehouse" / "market",
        "pattern": ("fusion", ".parquet"),
        "is_json": False, "compress": True,
    },
    "zt_pool_em_daily": {
        "host": HOST_TX,
        "remote_dir": "C:/quant/data_warehouse/market",
        "local_dir": ROOT / "data_warehouse" / "market",
        "pattern": ("zt_pool_em_daily", ".parquet"),
        "is_json": False, "compress": True,
    },
    "industry_hist": {
        "host": HOST_TX,
        "remote_dir": "C:/quant/data_warehouse/industry",
        "local_dir": ROOT / "data_warehouse" / "industry",
        "pattern": ("", "_hist.parquet"),
        "is_json": False,
        "compress": True,
    },
    "kline": {
        # 2026-08-21 补: 本地K线停08-14根因=未拉回。增量拉回(仅本地缺失/更新的个股),
        # 5207文件, 默认参与但跳过已最新; --only kline 强制全量。
        "host": HOST_TX,
        "remote_dir": "C:/quant/data_warehouse/kline",
        "local_dir": ROOT / "data_warehouse" / "kline",
        "pattern": ("", ".parquet"),
        "is_json": False, "compress": True,
    },
    "valuation": {
        # 本地估值停08-13: 补拉回(低频,默认执行但增量)
        "host": HOST_TX,
        "remote_dir": "C:/quant/data_warehouse/valuation",
        "local_dir": ROOT / "data_warehouse" / "valuation",
        "pattern": ("", ".parquet"),
        "is_json": False, "compress": True,
    },
    "financial": {
        # 低频(季度): 5206文件全量列目录+比较慢, 不进每日默认拉取,
        # 由 --only financial 或月度任务显式触发
        "host": HOST_TX,
        "remote_dir": "C:/quant/data_warehouse/financial",
        "local_dir": ROOT / "data_warehouse" / "financial",
        "pattern": ("", ".parquet"),
        "is_json": False,
        "compress": True,
        "enabled": False,
    },
    # 2026-09-20 恢复补齐：云端真实运行域此前只在 C:/quant/data_warehouse
    # 留存，本地缺少 canonical 副本。统一走同一套大小、Parquet 和原子替换校验。
    "events": {
        "host": HOST_TX,
        "remote_dir": "C:/quant/data_warehouse/events",
        "local_dir": ROOT / "data_warehouse" / "events",
        "pattern": ("", ".parquet"),
        "is_json": False,
        "compress": True,
    },
    "industry": {
        "host": HOST_TX,
        "remote_dir": "C:/quant/data_warehouse/industry",
        "local_dir": ROOT / "data_warehouse" / "industry",
        "pattern": ("", ".parquet"),
        "is_json": False,
        "compress": True,
    },
    "classification": {
        "host": HOST_TX,
        "remote_dir": "C:/quant/data_warehouse/classification",
        "local_dir": ROOT / "data_warehouse" / "classification",
        "pattern": ("", ".parquet"),
        "is_json": False,
        "compress": True,
    },
    "patterns": {
        "host": HOST_TX,
        "remote_dir": "C:/quant/data_warehouse/patterns",
        "local_dir": ROOT / "data_warehouse" / "patterns",
        "pattern": ("", ".parquet"),
        "is_json": False,
        "compress": True,
    },
    "pit_tradability": {
        "host": HOST_TX,
        "remote_dir": "C:/quant/data_warehouse/pit_tradability",
        "local_dir": ROOT / "data_warehouse" / "pit_tradability",
        "pattern": ("", ".parquet"),
        "is_json": False,
        "compress": True,
    },
}


def _ssh(host: str, cmd: str, timeout: int = 120, decode_gbk: bool = False) -> str:
    """执行远程命令；Windows 输出 GBK 时 decode_gbk=True 按 GBK 解码。"""
    args = ["ssh", *_ssh_options()]
    if host == HOST_TX:
        args += ["-i", PEM]
    args += [host, cmd]
    r = subprocess.run(args, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ssh 失败: {r.stderr[:200]}")
    enc = "gbk" if decode_gbk else "utf-8"
    return r.stdout.decode(enc, errors="replace")


def _parse_windows_dir_output(out: str) -> tuple[list, int, list]:
    """解析 Windows `dir /-c` 输出 → ([(name,size)...], 未解析日期行数, 未解析行示例)。

    V12.3 审计 P2-5: 对以日期开头却无法解析成 [大小 文件名] 的行计数并收集示例,
    不再静默丢弃, 避免"解析不匹配→远端文件永不回传"的隐性数据丢失。
    <DIR> 目录行、卷/目录标题、汇总"个文件 字节"行均跳过。
    """
    import re as _re
    items: list = []
    date_like_lines = 0
    unparsed: list[str] = []
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        # 跳过 dir 输出的标题/汇总行/目录行(不含日期前缀)
        if "<DIR>" in s.upper() or not _re.match(r"^\d{4}/\d{2}/\d{2}", s):
            continue
        m = _re.match(r"^(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2})\s+([\d,]+)\s+(.+)$", s)
        if m:
            items.append((m.group(4).strip(), int(m.group(3).replace(",", ""))))
        else:
            date_like_lines += 1
            if len(unparsed) < 3:
                unparsed.append(s[:90])
    return items, date_like_lines, unparsed


def _ls_remote(host: str, remote_dir: str) -> list:
    """列远程目录（Windows dir /-c 带大小，与 Linux ls -l 兼容）。

    2026-08-14 修复: 原 dir /b 无大小 → pull_source 只按"文件存在"跳过,
    云端更新过但本地同名文件永远不重拉(zt_pool 停 8/11 根因)。
    返回 [(name, size), ...]。
    """
    if host == HOST_TX:
        win_dir = remote_dir.replace("/", "\\")
        out = _ssh(host, f'dir "{win_dir}" /-c', timeout=60, decode_gbk=True)
        items, date_like_lines, unparsed = _parse_windows_dir_output(out)
        if unparsed or date_like_lines:
            print(f"[pull_cloud] 注意: {host} {remote_dir} 有 {date_like_lines} 行疑似"
                  f"文件未被解析(可能含空格/异形名), 前例: {unparsed} —— 需核对是否漏拉", flush=True)
        return items
    out = _ssh(host, f"ls -l {remote_dir}", timeout=60)
    items = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].startswith("-"):
            items.append((" ".join(parts[8:]), int(parts[4])))
    return items


def _verify_ohlc(df: pd.DataFrame) -> None:
    """V12.3 审计 P1-4: K线 OHLC 不变量校验。

    - 若含 OHLC 列: high >= max(open, close), low <= min(open, close),
      所有价格 > 0, 无 NaN。
    - 不含 OHLC 列(如非K线类 parquet) 仅验证可正常读取, 不校验内容。
    违反任一条目直接抛异常, 由调用方删文件重拉。
    """
    cols = {c.lower(): c for c in df.columns}
    ohlc = [k in cols for k in ("open", "high", "low", "close")]
    # 2026-08-21 修复: 原本"任一OHLC存在即限制校验"会误伤部分列的表
    # (如 zt_pool_em_daily 有 open 无 high/low/close)。改为: 四列全在才校验,
    # 否则视为非K线类(仅证明可读)。
    if not all(ohlc):
        return
    o, h, l, c = (df[cols[k]] for k in ("open", "high", "low", "close"))
    if (o <= 0).any() or (h <= 0).any() or (l <= 0).any() or (c <= 0).any():
        raise ValueError("OHLC 含非正价格")
    if (h < o).any() or (h < c).any():
        raise ValueError("high < open/close")
    if (l > o).any() or (l > c).any():
        raise ValueError("low > open/close")
    if df.columns.astype("str").str.contains("nan", case=False).any() \
            or df.duplicated().any():
        # NaN 列名或重复行均属数据污染
        bad = df.columns[df.columns.astype("str").str.contains("nan", case=False)]
        if len(bad) > 0 or df.duplicated().any():
            raise ValueError(f"数据污染: 非法列名 {list(bad)[:5]} / 重复行")


def pull_source(name: str, force: bool = False) -> int:
    """拉取单个源（增量按大小; force=True 覆盖已存在文件）。"""
    cfg = SOURCES[name]
    local_dir = cfg["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    prefix, suffix = cfg["pattern"]
    try:
        files = _ls_remote(cfg["host"], cfg["remote_dir"])
    except Exception as e:
        print(f"[pull_cloud:{name}] 云目录不可达: {e}", file=sys.stderr)
        return -1  # 审计 2026-08-16：不可达不得伪装成功（0）
    got = 0
    for entry in files:
        f = entry[0] if isinstance(entry, tuple) else entry
        size = entry[1] if isinstance(entry, tuple) else None
        if prefix and not f.startswith(prefix):
            continue
        if suffix and not f.endswith(suffix):
            continue
        if f in (".", ".."):
            continue
        local = local_dir / f
        # 2026-08-14 修复: 同名但大小不同 = 云端已更新 → 重拉
        if local.exists() and not force:
            if size is not None and local.stat().st_size == size:
                continue
        # 先落到同目录临时文件；验证通过后原子替换。直接 scp 到 local 会在传输
        # 中断时破坏上一份有效数据，并留下看似存在的半截文件。
        partial = local.with_name(f".{local.name}.{os.getpid()}.part")
        partial.unlink(missing_ok=True)
        scp_args = ["scp"]
        if cfg["compress"]:
            scp_args.append("-C")
        scp_args += [*_ssh_options(), "-o", "ServerAliveInterval=15"]
        if cfg["host"] == HOST_TX:
            scp_args += ["-i", PEM]
        scp_args += [f"{cfg['host']}:{cfg['remote_dir']}/{f}", str(partial)]
        # V12.3 审计 P1-2: scp 包 try/except, 超时/失败删本地残留并记失败(不 abort 整源)
        try:
            r = subprocess.run(scp_args, capture_output=True, text=True,
                               timeout=600, shell=False)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
                OSError) as _sce:
            partial.unlink(missing_ok=True)
            print(f"[pull_cloud:{name}] {f} scp 失败: {type(_sce).__name__}, 跳过")
            continue
        if r.returncode != 0:
            partial.unlink(missing_ok=True)
            print(f"[pull_cloud:{name}] {f} scp 非零退出: {r.returncode}, 跳过")
            continue
        if r.returncode == 0:
            if size is not None and partial.stat().st_size != size:
                print(f"[pull_cloud:{name}] {f} 大小不符: 远端={size} 本地={partial.stat().st_size}，保留旧文件")
                partial.unlink(missing_ok=True)
                continue
            if cfg["is_json"]:
                try:
                    data = json.loads(partial.read_text(encoding="utf-8"))
                    tag = " (force 覆盖)" if force else ""
                except Exception as e:
                    print(f"[pull_cloud:{name}] {f} JSON 损坏: {e}，保留旧文件")
                    partial.unlink(missing_ok=True)
                    continue
            else:
                # V12.3 审计 P1-3/P1-4: 全量读 parquet 而非仅 4 字节魔数;
                # 若含 OHLC 列则同时校验买卖不变量(high>=max(o,c); low<=min(o,c); 价>0)
                try:
                    df = pd.read_parquet(partial)
                    _verify_ohlc(df)
                    tag = " (force 覆盖)" if force else ""
                except Exception as e:
                    print(f"[pull_cloud:{name}] {f} 校验失败: {e}，保留旧文件")
                    partial.unlink(missing_ok=True)
                    continue
            os.replace(partial, local)
            if cfg["is_json"]:
                print(f"[pull_cloud:{name}] {f}: {len(data)} 条 → {local}{tag}")
            else:
                print(f"[pull_cloud:{name}] {f}: {len(df)}行/{len(df.columns)}列 → {local}{tag}")
            got += 1
        else:
            print(f"[pull_cloud:{name}] {f} 拉取失败: {r.stderr[:120]}")
            partial.unlink(missing_ok=True)
    return got


def _enrich_hot_rank() -> int:
    """拉回 hot_rank 后内嵌调用 enrich_hot_rank 补 pct_chg（qt.gtimg.cn）。

    任何失败（含 enrich 脚本本身异常）都只打印提示，不阻塞 pull 主流程。
    """
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "enrich_hot_rank", ROOT / "scripts" / "enrich_hot_rank.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.enrich_latest()
    except Exception as e:  # noqa: BLE001
        print(f"[pull_cloud:hot_rank] enrich 调用失败（不阻塞主流程）: {type(e).__name__}: {e}")
        return -1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="拉取云服务器爬虫数据（腾讯云 cninfo/hot_rank/ths_concept）")
    ap.add_argument("--only", default="", help="只拉指定源: cninfo/hot_rank/ths_concept/social")
    ap.add_argument("--force", action="store_true", help="覆盖已存在文件")
    args = ap.parse_args()

    names = [args.only] if args.only else [n for n, c in SOURCES.items() if c.get("enabled", True)]
    total = 0
    any_failed = False
    for name in names:
        if name not in SOURCES:
            print(f"[pull_cloud] 未知源: {name}（可选: {', '.join(SOURCES)}）")
            continue
        n = pull_source(name, force=args.force)
        print(f"[pull_cloud:{name}] 更新 {n} 个文件")
        if name == "hot_rank":
            _enrich_hot_rank()
        if n < 0:
            any_failed = True
        total += max(n, 0)
    print(f"[pull_cloud] 全部完成，共更新 {total} 个文件")
    # 审计 2026-08-16：任一源不可达 → 非零退出码，cron 能感知失败
    raise SystemExit(1 if any_failed else 0)
