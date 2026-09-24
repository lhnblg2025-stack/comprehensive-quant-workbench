#!/usr/bin/env python3
"""Collect cloud-side domestic PIT tradability observations and pull them back.

The Tencent Cloud node is the domestic-network acquisition worker. This script
uses the existing SSH helper, runs bounded AkShare calls remotely, stores raw
parquet shards, and records source coverage. Absence from a suspension list is
never converted into a tradable=true claim.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.pull_cloud_data import HOST_TX, PEM, _ssh_options, KNOWN_HOSTS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2" / "cloud_pit_tradability"
REMOTE_ROOT = "C:/quant/data_warehouse/pit_tradability"


def remote_python(code: str, timeout: int = 180) -> str:
    encoded = base64.b64encode(code.encode()).decode()
    command = f"python -c \"import base64;exec(base64.b64decode('{encoded}'))\""
    args = ["ssh", *_ssh_options(), "-i", PEM, HOST_TX, command]
    result = subprocess.run(args, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"cloud_command_failed:{result.stderr.decode('utf-8', 'replace')[:500]}")
    return result.stdout.decode("gbk", errors="replace")


def collect_suspension(dates: list[str], remote_root: str = REMOTE_ROOT) -> dict:
    # Windows command lines have a finite length; send short batches so the
    # acquisition can resume after any batch and never encode 1,600 dates at once.
    summary = {"requested": len(dates), "stored": 0, "empty": 0, "failures": []}
    for offset in range(0, len(dates), 30):
        batch = dates[offset:offset + 30]
        code = '''import akshare as ak, json
from pathlib import Path
import pandas as pd
root=Path(r"%s"); root.mkdir(parents=True, exist_ok=True)
summary={"requested":%d,"stored":0,"empty":0,"failures":[]}
for date in %s:
 path=root/("suspend_"+date+".parquet")
 if path.exists(): summary["stored"]+=1; continue
 try:
  raw=ak.stock_tfp_em(date=date)
  frame=pd.DataFrame() if raw is None else raw.copy()
  frame["query_date"]=date; frame["source"]="tencent_cloud_akshare_stock_tfp_em"
  frame.to_parquet(path,index=False)
  summary["empty"] += int(frame.empty); summary["stored"] += int(not frame.empty)
 except Exception as e:
  summary["failures"].append({"date":date,"error":type(e).__name__+":"+str(e)[:240]})
print(json.dumps(summary,ensure_ascii=True))
''' % (remote_root, len(batch), repr(batch))
        output = remote_python(code, timeout=300)
        lines = [line for line in output.splitlines() if line.strip().startswith("{")]
        if lines:
            part = json.loads(lines[-1])
            summary["stored"] += part.get("stored", 0)
            summary["empty"] += part.get("empty", 0)
            summary["failures"].extend(part.get("failures", []))
        else:
            summary["failures"].append({"batch": batch, "error": output[-500:]})
        print(json.dumps({"processed": min(offset + 30, len(dates)), **summary}, ensure_ascii=True), flush=True)
    return summary


def collect_lifecycle(remote_root: str = REMOTE_ROOT) -> dict:
    """Collect exchange-published delisting tables plus a bounded code/name map."""
    code = '''import akshare as ak, json
from pathlib import Path
root=Path(r"%s"); root.mkdir(parents=True, exist_ok=True)
summary={}
for name, fn in [("sh_delist", ak.stock_info_sh_delist), ("sz_delist", ak.stock_info_sz_delist), ("code_name", ak.stock_info_a_code_name)]:
 try:
  frame=fn()
  frame.to_parquet(root/(name+".parquet"),index=False)
  summary[name]={"status":"ok","rows":int(len(frame)),"columns":[str(c) for c in frame.columns]}
 except Exception as e:
  summary[name]={"status":"error","error":type(e).__name__+":"+str(e)[:240]}
print(json.dumps(summary,ensure_ascii=True))
''' % remote_root
    output = remote_python(code, timeout=240)
    lines = [line for line in output.splitlines() if line.strip().startswith("{")]
    return json.loads(lines[-1]) if lines else {"error": output[-1000:]}


def pull_lifecycle(output_dir: Path = OUTPUT) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ["sh_delist.parquet", "sz_delist.parquet", "code_name.parquet"]
    pulled = 0
    for name in names:
        target = output_dir / name
        if target.exists():
            pulled += 1
            continue
        args = ["scp", "-C", *_ssh_options(), "-o", "ServerAliveInterval=15", "-i", PEM, f"{HOST_TX}:{REMOTE_ROOT}/{name}", str(target)]
        result = subprocess.run(args, capture_output=True, timeout=300)
        if result.returncode:
            target.unlink(missing_ok=True)
            continue
        try:
            pd.read_parquet(target); pulled += 1
        except Exception:
            target.unlink(missing_ok=True)
    return {"expected": len(names), "local_files": pulled, "local_dir": str(output_dir)}


def pull_remote_files(output_dir: Path = OUTPUT) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = REMOTE_ROOT
    listing = remote_python(f'''from pathlib import Path
import json
root=Path(r"{remote_dir}")
print(json.dumps([p.name for p in root.glob("suspend_*.parquet")], ensure_ascii=True))
''', timeout=60)
    names = json.loads([line for line in listing.splitlines() if line.strip().startswith("[")][-1])
    local = 0
    for name in names:
        target = output_dir / name
        if target.exists():
            local += 1
            continue
        args = ["scp", "-C", *_ssh_options(), "-o", "ServerAliveInterval=15", "-i", PEM, f"{HOST_TX}:{REMOTE_ROOT}/{name}", str(target)]
        result = subprocess.run(args, capture_output=True, timeout=300)
        if result.returncode:
            target.unlink(missing_ok=True)
            continue
        try:
            pd.read_parquet(target)
            local += 1
        except Exception:
            target.unlink(missing_ok=True)
    return {"remote_files": len(names), "local_files": local, "local_dir": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20260831")
    parser.add_argument("--max-dates", type=int, default=3000, help="最多采集多少交易日；默认覆盖所给日期范围")
    args = parser.parse_args()
    # Bounded first acquisition: dates can be expanded on subsequent runs. The
    # local observed calendar is used only to select actual market sessions.
    calendar = ROOT / "data_warehouse" / "research_panels" / "ashare_tushare_500_10y_v2" / "observed_trade_calendar.parquet"
    if calendar.exists():
        frame = pd.read_parquet(calendar)
        dates = pd.to_datetime(frame["date"], errors="coerce")
        dates = dates[(dates >= pd.Timestamp(args.start)) & (dates <= pd.Timestamp(args.end))].dt.strftime("%Y%m%d").tolist()
    else:
        dates = pd.bdate_range(args.start, args.end).strftime("%Y%m%d").tolist()
    dates = dates[-args.max_dates:]
    remote = collect_suspension(dates)
    local = pull_remote_files()
    lifecycle_remote = collect_lifecycle()
    lifecycle_local = pull_lifecycle()
    manifest = {"schema":"cloud_pit_tradability/v2","source":"tencent_cloud_akshare","start":args.start,"end":args.end,"queried_dates":dates,"remote":remote,"local":local,"lifecycle_remote":lifecycle_remote,"lifecycle_local":lifecycle_local,"interpretation":"suspension_observations_only; delisting_tables_are_exchange_published; absence_is_not_tradable_true","generated_at":datetime.now().astimezone().isoformat(timespec="seconds")}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False))

if __name__ == "__main__": main()
