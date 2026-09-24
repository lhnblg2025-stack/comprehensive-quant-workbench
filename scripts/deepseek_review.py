#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deepseek_review.py — 代码审查直连脚本（DeepSeek 通道，2026-08-12 起替代 claude_review.py）

背景: Claude 中转站 your-llm-relay.example.com 于 2026-08-12 故障（/v1/models 200 但 chat/completions 403/超时），
按用户指示"中转站有问题就换 deepseek"，审查通道改为 DeepSeek 官方 API（api.deepseek.com）。
DeepSeek key: ~/.deepseek_key（chmod 600）或 config/design_proxy_keys.json['deepseek']。

用法: python3 scripts/deepseek_review.py <diff_file> <module_file> <out_file> [model]
      model 默认 deepseek-v4-flash（快/省，审查主力）；可用 deepseek-v4-pro（更强）
"""
from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_key() -> str:
    kf = Path.home() / ".deepseek_key"
    if kf.exists():
        return kf.read_text().strip()
    try:
        cfg = json.load(open(ROOT / "config" / "design_proxy_keys.json"))
        return cfg["deepseek"]
    except Exception:
        cfg = json.load(open(Path.home() / ".openclaw" / "openclaw.json"))
        return cfg["models"]["providers"]["deepseek"]["apiKey"]


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    diff_file, module_file, out_file = sys.argv[1], sys.argv[2], sys.argv[3]
    model = sys.argv[4] if len(sys.argv) > 4 else "deepseek-v4-flash"
    diff = open(diff_file, encoding="utf-8").read()
    module = open(module_file, encoding="utf-8").read()
    prompt = f"""你是资深量化系统代码审查员。审查以下 diff 与模块，输出结构化审查报告。

# 审查重点
1. Critical（阻断合并）：前视偏差/数据泄漏、逻辑错误、崩溃风险、循环导入、静默吞异常导致降级链失效
2. Major（应修复）：行为语义变更、边界条件、线程安全、IO 损坏处理、变量遮蔽
3. Minor：风格、docstring、命名
4. 逐条给：位置 | 问题 | 修复建议 | 证据（代码片段）
5. 结论：通过 / 需修复后提交（列出必改项）

# diff
{diff[:30000]}

# 模块文件片段
{module[:15000]}
"""
    body = json.dumps({
        "model": model,
        "max_tokens": 8000,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + get_key()},
    )
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=600))
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read()[:300]}", file=sys.stderr)
        return 1
    msg = resp["choices"][0].get("message", {})
    # DeepSeek 推理模型: 最终答案在 content, 推理过程在 reasoning_content
    text = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    if not text and reasoning:
        text = reasoning
    if not text:
        print("⚠️ 模型返回空内容", file=sys.stderr)
        return 1
    open(out_file, "w", encoding="utf-8").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
