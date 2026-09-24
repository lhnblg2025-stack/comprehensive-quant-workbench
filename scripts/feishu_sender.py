#!/usr/bin/env python3
"""
feishu_sender.py — 通过飞书开放 API 直接发送消息（无需 OpenClaw）。

用法:
  python3 scripts/feishu_sender.py --text "你好飞书"
  python3 scripts/feishu_sender.py --title "标题" --text "正文内容"
  echo "消息内容" | python3 scripts/feishu_sender.py

配置: config/private_data_sources.json 下的 feishu 段
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

# 飞书凭据从 private_data_sources.json 读取
FEISHU_CONFIG_KEY = "feishu_bot"
_TEST_MESSAGE_RE = re.compile(r"(?:复盘测试|测试复盘|单元测试|冒烟测试|pytest|test report)", re.IGNORECASE)
_INLINE_MD_RE = re.compile(r"(\*\*.+?\*\*|\[[^\]]+\]\(https?://[^)]+\))")


def _inline_post_nodes(text: str, *, bold: bool = False) -> list[dict]:
    """把一行简化 Markdown 转成飞书 post 节点。"""
    nodes: list[dict] = []
    pos = 0
    for match in _INLINE_MD_RE.finditer(text):
        if match.start() > pos:
            node = {"tag": "text", "text": text[pos:match.start()]}
            if bold:
                node["style"] = ["bold"]
            nodes.append(node)
        token = match.group(0)
        link = re.fullmatch(r"\[([^\]]+)\]\((https?://[^)]+)\)", token)
        if link:
            nodes.append({"tag": "a", "text": link.group(1), "href": link.group(2)})
        else:
            nodes.append({"tag": "text", "text": token[2:-2], "style": ["bold"]})
        pos = match.end()
    if pos < len(text):
        node = {"tag": "text", "text": text[pos:]}
        if bold:
            node["style"] = ["bold"]
        nodes.append(node)
    return nodes or [{"tag": "text", "text": " "}]


def _markdown_to_post_content(text: str) -> list[list[dict]]:
    """将常用 Markdown 映射为飞书富文本行，保留可扫描层级。"""
    rows: list[list[dict]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            rows.append([{"tag": "text", "text": " "}])
            continue
        if re.fullmatch(r"[-—─]{3,}", line):
            rows.append([{"tag": "text", "text": "────────────────", "style": ["bold"]}])
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            prefix = "▰ " if len(heading.group(1)) == 1 else "▌ "
            rows.append(_inline_post_nodes(prefix + heading.group(2), bold=True))
            continue
        if line.startswith(">"):
            rows.append(_inline_post_nodes("│ " + line.lstrip("> ")))
            continue
        bullet = re.match(r"^[-*+]\s+(.+)$", line)
        numbered = re.match(r"^(\d+)[.)]\s+(.+)$", line)
        if bullet:
            line = "• " + bullet.group(1)
        elif numbered:
            line = numbered.group(1) + ". " + numbered.group(2)
        rows.append(_inline_post_nodes(line))
    return rows or [[{"tag": "text", "text": " "}]]


def _blocked_test_message(text: str, title: str | None = None) -> bool:
    """生产通道默认拒绝测试消息；显式 FEISHU_ALLOW_TEST_MESSAGES=1 才放行。"""
    if os.environ.get("FEISHU_ALLOW_TEST_MESSAGES") == "1":
        return False
    sample = f"{title or ''}\n{text or ''}"
    if _TEST_MESSAGE_RE.search(sample):
        print("[feishu] 已拦截测试/复盘测试消息；生产通道不外发", file=sys.stderr)
        return True
    return False


def _default_chat_id() -> str:
    """兜底 chat_id: 从 .env.secrets 读取, 缺失时返回空(发送方自行降级)。"""
    try:
        from secret_loader import get_secret  # noqa: PLC0415
        return get_secret("FEISHU_CHAT_ID") or ""
    except Exception:  # noqa: BLE001
        return ""


def _load_config() -> dict:
    """加载飞书凭据 (V12.3 密钥治理: env:NAME 占位符经 secret_loader 解析)"""
    pds_path = ROOT / "config" / "private_data_sources.json"
    if not pds_path.exists():
        return {}
    try:
        from secret_loader import load_config_with_secrets  # noqa: PLC0415
        data = load_config_with_secrets(pds_path)
        return data.get(FEISHU_CONFIG_KEY) or {}
    except Exception:
        return {}


def get_tenant_token(app_id: str, app_secret: str) -> str | None:
    """获取飞书 tenant_access_token"""
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    try:
        resp = requests.post(url, json={
            "app_id": app_id,
            "app_secret": app_secret,
        }, timeout=15)
        data = resp.json()
        if data.get("code") == 0:
            return data.get("tenant_access_token")
        print(f"[feishu] 获取 token 失败: {data}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[feishu] 获取 token 异常: {e}", file=sys.stderr)
        return None


def send_text(chat_id: str, text: str, title: str | None = None, token: str | None = None) -> bool:
    """发送富文本消息到飞书"""
    if _blocked_test_message(text, title):
        return False
    if not token:
        cfg = _load_config()
        app_id = cfg.get("app_id") or cfg.get("appId")
        app_secret = cfg.get("app_secret") or cfg.get("appSecret")
        if not app_id or not app_secret:
            print("[feishu] 缺少 app_id/app_secret 配置", file=sys.stderr)
            return False
        token = get_tenant_token(app_id, app_secret)
        if not token:
            return False

    # 构建飞书 post 消息：真正渲染标题、粗体、链接、分节和列表。
    content = {
        "zh_cn": {
            "title": title or "量化预警",
            "content": _markdown_to_post_content(text),
        }
    }

    # 如果文本较长，分段发送
    url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "receive_id": chat_id,
        "msg_type": "post",
        "content": json.dumps(content, ensure_ascii=False),
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        data = resp.json()
        ok = data.get("code") == 0
        if not ok:
            print(f"[feishu] 发送失败: {json.dumps(data, ensure_ascii=False)[:300]}", file=sys.stderr)
        return ok
    except Exception as e:
        print(f"[feishu] 发送异常: {e}", file=sys.stderr)
        return False


def send_simple_text(text: str, title: str | None = None) -> bool:
    """简便发送：从配置读取 chat_id 并发消息。

    V12.3 密钥治理: 凭据一律走 config/.env.secrets(env:占位符);
    缺失时显式失败(fail-loud), 不再硬编码兜底, 避免真实密钥入 git。"""
    if _blocked_test_message(text, title):
        return False
    cfg = _load_config()
    chat_id = cfg.get("chat_id") or cfg.get("target") or _default_chat_id()
    # Remove "user:" prefix if present
    if chat_id.startswith("user:"):
        chat_id = chat_id[5:]
    app_id = cfg.get("app_id") or cfg.get("appId")
    app_secret = cfg.get("app_secret") or cfg.get("appSecret")
    if not app_id or not app_secret:
        # V12.3 安全修复: 移除硬编码密钥兜底——凭据缺失必须显式暴露,
        # 否则静默用旧密钥运行, 掩盖配置缺失并泄露真实凭据。
        print("[feishu] 配置缺失: app_id/app_secret 未提供(检查 config/.env.secrets)", file=sys.stderr)
        return False
    if not chat_id:
        print("[feishu] 配置缺失: chat_id 未提供(检查 FEISHU_CHAT_ID)", file=sys.stderr)
        return False

    token = get_tenant_token(app_id, app_secret)
    if not token:
        return False
    return send_text(chat_id, text, title=title, token=token)


def send_markdown(text: str, title: str | None = None) -> bool:
    """发送飞书消息，文本超过2000字时分段"""
    MAX_LEN = 1800
    if len(text) <= MAX_LEN:
        return send_simple_text(text, title=title)

    # 分段发送
    ok = True
    lines = text.split("\n")
    chunks = []
    current = []
    for line in lines:
        current.append(line)
        if len("\n".join(current)) > MAX_LEN:
            chunks.append("\n".join(current[:-1]))
            current = [line]
    if current:
        chunks.append("\n".join(current))

    for i, chunk in enumerate(chunks):
        t = f"{title} ({i+1}/{len(chunks)})" if title and len(chunks) > 1 else title
        if not send_simple_text(chunk, title=t):
            ok = False
    return ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="发送飞书消息")
    parser.add_argument("--text", help="消息正文")
    parser.add_argument("--title", default="量化预警", help="消息标题")
    parser.add_argument("--stdin", action="store_true", help="从 stdin 读取")
    args = parser.parse_args()

    if args.stdin or (not args.text and not sys.stdin.isatty()):
        text = sys.stdin.read().strip()
    else:
        text = args.text or ""

    if not text:
        print("请输入消息内容", file=sys.stderr)
        sys.exit(1)

    ok = send_markdown(text, title=args.title)
    sys.exit(0 if ok else 1)
