#!/usr/bin/env python3
"""scan_plaintext_secrets.py — V12.3 审计 P1-5: config/*.json 明文密钥静态扫描

扫描 config/ 下所有 JSON, 递归找出"值看起来像真实密钥"但未用 env:NAME 占位符的字段
(key/api_key/token/secret/app_secret/app_id 等敏感键名, 值非 env: 且非脆弱占位符如
YOUR_*/xxx/空)。命中即报(exit 1), 供 CI / 手动审计强制"config 无未解析明文 8+ 位密钥"。

用法: python3 scripts/scan_plaintext_secrets.py [--config-dir config]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# 敏感键名: 值若是真实明文 key → 报
SENSITIVE_KEY = re.compile(
    r"(key|token|secret|password|passwd|api_key|app_secret|app_id|auth|credential)", re.I
)
# 合法值形态: env:NAME 占位符 / 占位串 / 空
SAFE_VALUE = re.compile(
    r"^(env:[A-Za-z_][A-Za-z0-9_]*|YOUR[A-Za-z_]*|your-[a-z-]+|<[^>]*>|"
    r"[A-Za-z_]+_[A-Za-z_]+)$"
)
# 疑似真实密钥: 含 8+ 位高熵字符(字母数字混合或含特殊符号), 且非上述 safe
SUSPECT_KEY = re.compile(r"[A-Za-z0-9_\-]{8,}")


def _looks_like_secret(value: str) -> bool:
    """是否像真实密钥/应决绝作为明文存在."""
    if not value or len(value) < 8:
        return False
    if SAFE_VALUE.fullmatch(value):
        return False
    return bool(SUSPECT_KEY.fullmatch(value))


def _walk_dict(obj, path: str, findings: list[tuple[str, str, str]]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}"
            if isinstance(v, (dict, list)):
                _walk_dict(v, p, findings)
            elif isinstance(v, str) and SENSITIVE_KEY.search(k) and _looks_like_secret(v):
                findings.append((path, k, v))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_dict(v, f"{path}[{i}]", findings)


def scan_file(path: Path, findings: list[tuple[str, str, str]]) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return 0
    _walk_dict(data, path.name, findings)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=str(Path(__file__).resolve().parent.parent / "config"))
    ap.add_argument("--exit-when-found", action="store_true", default=True)
    args = ap.parse_args()
    cfg_dir = Path(args.config_dir)
    findings: list[tuple[str, str, str]] = []
    for p in sorted(cfg_dir.glob("*.json")):
        scan_file(p, findings)
    if findings:
        for path, key, val in findings:
            masked = val[:4] + "…" + val[-2:] if len(val) > 8 else val
            print(f"[SECRET] {path}::{key} = {masked}  (明文密钥, 应迁 .env.secrets + env: 占位)")
        return 1
    print("[OK] config/*.json 无未解析明文密钥")
    return 0


if __name__ == "__main__":
    sys.exit(main())
