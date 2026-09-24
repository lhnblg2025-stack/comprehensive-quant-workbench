import json, os, subprocess, sys
cfg = json.load(open(os.path.expanduser("~/.youdaonote.json")))
key = cfg.get("mcp", {}).get("apiKey", "").strip()
if not key:
    print("no key found in ~/.youdaonote.json"); sys.exit(2)
env = dict(os.environ)
env["YOUDAONOTE_API_KEY"] = key
r = subprocess.run(["python3", "scripts/save_to_youdao.py", "--jsonl",
    "data_warehouse/ima_export/media/extracted_new_0903.jsonl", "--path-key", "extracted",
    "--prefix", "IMA研报-", "--limit", "50", "--sleep", "0.6"], cwd=".",
    env=env, timeout=1700)
sys.exit(r.returncode)
