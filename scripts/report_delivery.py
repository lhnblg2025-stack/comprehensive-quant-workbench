#!/usr/bin/env python3
"""Shared report archive/delivery helpers for OpenClaw cron reports.

Hard rules:
- Every report/file finalized by this module is backed up to Youdao YNote by
  default unless caller passes backup_youdao=False / CLI --no-youdao-backup.
- Feishu report bodies are sent as plain chat NEW messages only:
  `openclaw message send --channel feishu --target <chat_id> -m <body>`.
  Never use replies, quotes, thread-reply, --reply-to, or --thread-id.
- Chat targets live in config/report_delivery.json. Cron prompts/scripts must not
  hardcode Feishu/Weixin/QQ targets outside this bottom layer.
- Desktop/archive file is preserved even if Youdao or chat delivery fails.
"""
from __future__ import annotations
import logging

import json
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from urllib import error, parse as urlparse, request

# 2026-08-14 修复: 原硬编码本机路径 → 云端(Windows)读不到 config, feishu 段空。
# 动态: 脚本在 scripts/ 下, ROOT=上级(workspace 或 C:\quant)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "report_delivery.json"
PREFERRED_OPENCLAW_BIN = "openclaw"
OPENCLAW_BIN = os.environ.get("OPENCLAW_BIN") or (
    PREFERRED_OPENCLAW_BIN if Path(PREFERRED_OPENCLAW_BIN).exists() else "openclaw"
)


def delivery_suppressed() -> bool:
    return os.environ.get("OPENCLAW_REPORT_NO_DELIVER", "").lower() in {"1", "true", "yes", "on"}


def mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 10:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"version": 0, "defaults": {}, "channels": {}, "youdao": {"enabled": False}}
    try:
        # V12.3 密钥治理: env:NAME 占位符经 secret_loader 解析
        from secret_loader import load_config_with_secrets  # noqa: PLC0415
        return load_config_with_secrets(p)
    except Exception:
        return json.loads(p.read_text(encoding="utf-8"))


def expand_headers(headers: dict[str, Any], values: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (headers or {}).items():
        s = str(v)
        for name, val in values.items():
            s = s.replace("${" + name + "}", str(val))
        out[str(k)] = s
    return out


def redact(text: str, secrets: Sequence[str]) -> str:
    out = text
    for sec in secrets:
        if sec:
            out = out.replace(sec, "***")
    return out


@dataclass
class ChunkResult:
    ok: bool
    message_id: str = ""
    error: str = ""


@dataclass
class DeliveryStatus:
    name: str
    status: str
    reason: str = ""
    target: str = ""
    chunks: list[ChunkResult] = field(default_factory=list)

    def to_line(self) -> str:
        extra = f"，原因：{self.reason}" if self.reason else ""
        target = f"，目标：{self.target}" if self.target else ""
        chunk_info = ""
        if self.chunks:
            ok = sum(1 for c in self.chunks if c.ok)
            total = len(self.chunks)
            chunk_info = f"，分段：{ok}/{total}"
        return f"- {self.name}状态：{self.status}{target}{chunk_info}{extra}。"


# ----------------------------- message chunking -----------------------------


def chunk_message(text: str, max_chars: int = 3500) -> list[str]:
    if not text:
        return []
    text = text.rstrip() + "\n"
    if len(text) <= max_chars:
        return [text.rstrip()]
    chunks: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(buf.rstrip())
        buf = ""

    for para in re.split(r"\n{2,}", text):
        piece = para.rstrip()
        if not piece:
            continue
        candidate = piece if not buf else buf + "\n\n" + piece
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        flush()
        if len(piece) <= max_chars:
            buf = piece
            continue
        line_buf = ""
        for line in piece.split("\n"):
            cand = line if not line_buf else line_buf + "\n" + line
            if len(cand) <= max_chars:
                line_buf = cand
                continue
            if line_buf:
                chunks.append(line_buf.rstrip())
            if len(line) <= max_chars:
                line_buf = line
            else:
                for i in range(0, len(line), max_chars):
                    part = line[i : i + max_chars]
                    if i + max_chars < len(line):
                        chunks.append(part)
                    else:
                        line_buf = part
        if line_buf:
            buf = line_buf
    flush()
    return chunks


def add_chunk_headers(chunks: Sequence[str], title: str | None) -> list[str]:
    if len(chunks) <= 1:
        return list(chunks)
    out: list[str] = []
    for idx, body in enumerate(chunks, 1):
        prefix = f"{title} [{idx}/{len(chunks)}]" if title else f"[{idx}/{len(chunks)}]"
        out.append(f"{prefix}\n\n{body}")
    return out


# -------------------------- OpenClaw chat delivery --------------------------


def _find_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    for start, ch0 in enumerate(text):
        if ch0 != "{":
            continue
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _run_openclaw_message(channel: str, target: str, message: str, account: str | None = None, timeout: int = 90) -> ChunkResult:
    # Feishu hard rule: no --reply-to / --thread-id, always a new chat message.
    cmd = [OPENCLAW_BIN, "message", "send", "--channel", channel, "--target", target, "-m", message, "--json"]
    if account:
        cmd.extend(["--account", account])
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        # 2026-08-14 审计P0修复: 云端无 openclaw → 网关渠道不可投递。
        # feishu 通道降级到独立机器人直发（feishu_sender, 云端也可用）。
        return _feishu_sender_fallback(channel, target, message)
    except subprocess.TimeoutExpired:
        return ChunkResult(False, error=f"timeout after {timeout}s")
    except Exception as e:
        return ChunkResult(False, error=f"exec error: {e}")
    if proc.returncode != 0:
        # 2026-08-14 审计P0修复: openclaw 命令失败(如云端无网关) →
        # feishu 通道降级独立机器人直发, 不静默丢推送
        if channel == "feishu":
            return _feishu_sender_fallback(channel, target, message)
        lines = (proc.stderr or proc.stdout or "").strip().splitlines()
        return ChunkResult(False, error=(lines[-1] if lines else f"rc={proc.returncode}"))
    payload = _find_json_object(proc.stdout)
    return ChunkResult(True, message_id=str((payload or {}).get("messageId") or ""))


def _feishu_sender_fallback(channel: str, target: str, message: str) -> ChunkResult:
    """飞书独立机器人直发降级（云端无 openclaw 时保持投递可达）。"""
    try:
        import sys as _sys
        sys_path = ROOT / "scripts"
        if str(sys_path) not in _sys.path:
            _sys.path.insert(0, str(sys_path))
        from feishu_sender import send_markdown
        ok = send_markdown(message, title="量化报告")
        return ChunkResult(ok, message_id="" if ok else None,
                           error=None if ok else "feishu_sender 发送失败")
    except Exception as e:
        return ChunkResult(False, error=f"feishu fallback: {e}")


def deliver_message(channel: str, target: str, body: str, *, account: str | None = None, title: str | None = None, max_chars: int = 3500, delay_ms: int = 400) -> list[ChunkResult]:
    chunks = add_chunk_headers(chunk_message(body, max_chars=max_chars), title)
    results: list[ChunkResult] = []
    for i, chunk in enumerate(chunks):
        results.append(_run_openclaw_message(channel, target, chunk, account=account))
        if i < len(chunks) - 1 and delay_ms > 0:
            time.sleep(delay_ms / 1000)
    return results


def _build_channel_status(name: str, results: list[ChunkResult], target: str) -> DeliveryStatus:
    ok = sum(1 for r in results if r.ok)
    if not results:
        return DeliveryStatus(name, "未完成", "未发起投递", target)
    if ok == len(results):
        return DeliveryStatus(name, "已投递", "", target, chunks=results)
    first_err = next((r.error for r in results if not r.ok), "未知错误")
    return DeliveryStatus(name, "未完成" if ok == 0 else "部分投递", first_err, target, chunks=results)


def deliver_all(body: str, *, title: str | None = None, config_path: str | Path = DEFAULT_CONFIG, channels: Sequence[str] | None = None, max_chars: int | None = None, include_webchat: bool = False) -> list[DeliveryStatus]:
    cfg = load_config(config_path)
    ch = cfg.get("channels") or {}
    cap = max_chars or int((cfg.get("defaults") or {}).get("max_message_chars") or 3500)
    wanted = set(channels) if channels else {"feishu", "weixin", "qq"}
    if include_webchat:
        wanted.add("webchat")
    statuses: list[DeliveryStatus] = []

    if delivery_suppressed():
        name_map = {"feishu": "飞书", "weixin": "微信", "qq": "QQ", "webchat": "webchat"}
        return [
            DeliveryStatus(name_map.get(channel, channel), "跳过", "OPENCLAW_REPORT_NO_DELIVER=1")
            for channel in sorted(wanted)
        ]

    if "feishu" in wanted:
        feishu = ch.get("feishu") or {}
        chat_id = feishu.get("chat_id")
        if feishu.get("enabled") and chat_id:
            statuses.append(_build_channel_status("飞书", deliver_message("feishu", chat_id, body, title=title, max_chars=cap), chat_id))
        else:
            statuses.append(DeliveryStatus("飞书", "未完成", "缺少 chat_id 或已禁用", chat_id or ""))

    if "weixin" in wanted:
        wx = ch.get("weixin") or {}
        target = wx.get("target")
        if wx.get("enabled") and target:
            statuses.append(_build_channel_status("微信", deliver_message(wx.get("channel") or "openclaw-weixin", target, body, account=wx.get("accountId"), title=title, max_chars=cap), target))
        else:
            statuses.append(DeliveryStatus("微信", "未完成", "缺少 target 或已禁用", target or ""))

    if "qq" in wanted:
        qq = ch.get("qq") or {}
        target = qq.get("target")
        if qq.get("enabled") and target:
            statuses.append(_build_channel_status("QQ", deliver_message(qq.get("channel") or "qqbot", target, body, title=title, max_chars=cap), target))
        else:
            statuses.append(DeliveryStatus("QQ", "未完成", "缺少 target 或已禁用", target or ""))

    if "webchat" in wanted:
        wc = ch.get("webchat") or {}
        mode = wc.get("mode") or "current_session"
        if wc.get("enabled") and mode == "current_session":
            statuses.append(DeliveryStatus("webchat", "待当前会话回显", "webchat 是内部 UI，不支持 openclaw message send 出站；由当前 assistant final 回复承载正文/摘要供检验", mode))
        elif wc.get("enabled"):
            statuses.append(DeliveryStatus("webchat", "未完成", f"不支持的 webchat mode={mode!r}", mode))
        else:
            statuses.append(DeliveryStatus("webchat", "未完成", "webchat 已禁用", mode))
    return statuses


# --------------------------- Youdao SSE MCP client --------------------------


class YoudaoMcpClient:
    def __init__(self, cfg: dict[str, Any], timeout: int = 30) -> None:
        yc = cfg.get("youdao") or {}
        self.url = yc.get("url")
        self.api_key = yc.get("api_key")
        self.base = yc.get("message_url_base") or "https://open.mail.163.com"
        self.timeout = timeout
        self.headers = expand_headers(yc.get("headers") or {}, {"api_key": self.api_key or ""})
        self.headers.update({"Accept": "text/event-stream"})
        self.post_headers = {k: v for k, v in self.headers.items() if k.lower() != "accept"}
        self.post_headers["Content-Type"] = "application/json"
        self.q: queue.Queue[tuple[str | None, str]] = queue.Queue()
        self.post_url = ""
        self._rid = 100

    def _reader(self) -> None:
        req = request.Request(self.url, headers=self.headers)
        with request.urlopen(req, timeout=self.timeout) as resp:
            ev: str | None = None
            data_parts: list[str] = []
            while True:
                line = resp.readline()
                if not line:
                    break
                s = line.decode("utf-8", "replace").rstrip("\n")
                if s.startswith("event:"):
                    ev = s[6:].strip()
                elif s.startswith("data:"):
                    data_parts.append(s[5:].strip())
                elif s.strip() == "" and (ev or data_parts):
                    self.q.put((ev, "\n".join(data_parts)))
                    ev = None
                    data_parts = []

    def connect(self) -> None:
        if not self.url or not self.api_key:
            raise RuntimeError("缺少有道云 MCP url 或 api_key")
        threading.Thread(target=self._reader, daemon=True).start()
        deadline = time.time() + self.timeout
        endpoint = ""
        while time.time() < deadline:
            try:
                _ev, data = self.q.get(timeout=1)
            except queue.Empty:
                continue
            if data and "/api/ynote/mcp/message" in data:
                endpoint = data.strip()
                break
        if not endpoint:
            raise RuntimeError("有道云 MCP SSE 未返回 message endpoint")
        self.post_url = urlparse.urljoin(self.base, endpoint)

    def post(self, obj: dict[str, Any]) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        req = request.Request(self.post_url, data=data, headers=self.post_headers, method="POST")
        with request.urlopen(req, timeout=self.timeout) as resp:
            resp.read()

    def wait_id(self, rid: int, timeout: int | None = None) -> dict[str, Any] | None:
        deadline = time.time() + (timeout or self.timeout)
        while time.time() < deadline:
            try:
                _ev, data = self.q.get(timeout=1)
            except queue.Empty:
                continue
            if data and data.startswith("{"):
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("id") == rid:
                    return obj
        return None

    def initialize(self) -> None:
        self.post({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "openclaw-report-delivery", "version": "1.0"}}})
        obj = self.wait_id(1, self.timeout)
        if not obj or obj.get("error"):
            raise RuntimeError(f"MCP initialize failed: {obj}")
        self.post({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def _call(self, name: str, args: dict[str, Any], timeout: int | None = None) -> dict[str, Any]:
        self._rid += 1
        rid = self._rid
        self.post({"jsonrpc": "2.0", "id": rid, "method": "tools/call", "params": {"name": name, "arguments": args}})
        obj = self.wait_id(rid, timeout or max(self.timeout, 60))
        if not obj:
            raise RuntimeError(f"MCP {name} timeout")
        if obj.get("error"):
            raise RuntimeError(json.dumps(obj.get("error"), ensure_ascii=False)[:500])
        result = obj.get("result") or {}
        if result.get("isError"):
            raise RuntimeError(json.dumps(result, ensure_ascii=False)[:500])
        return result

    @staticmethod
    def _result_json(result: dict[str, Any]) -> dict[str, Any]:
        text = ""
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("text"):
                text += str(item["text"])
        try:
            return json.loads(text) if text.strip().startswith("{") else {"_text": text}
        except json.JSONDecodeError:
            return {"_text": text}

    def list_notes(self, parent_id: str = "0") -> dict[str, Any]:
        return self._result_json(self._call("listNotes", {"parentId": parent_id}))

    def create_dir(self, name: str, parent_id: str = "0") -> dict[str, Any]:
        return self._call("createDir", {"name": name, "parentId": parent_id})

    def resolve_folder(self, name: str, parent_id: str = "0") -> str:
        """Return id of a top-level folder named ``name`` under ``parent_id``.

        Creates it if missing. Falls back to ``parent_id`` when resolution fails.
        """
        data = self.list_notes(parent_id)
        for e in data.get("entries") or []:
            if e.get("dir") and str(e.get("name")) == name:
                return str(e.get("id"))
        try:
            self.create_dir(name, parent_id)
        except Exception:
            return parent_id
        data = self.list_notes(parent_id)
        for e in data.get("entries") or []:
            if e.get("dir") and str(e.get("name")) == name:
                return str(e.get("id"))
        return parent_id

    def create_note(self, title: str, content: str, parent_id: str | None = None) -> dict[str, Any]:
        args: dict[str, Any] = {"title": title, "content": content}
        if parent_id:
            args["parentId"] = parent_id
        return self._call("createNote", args, timeout=max(self.timeout, 60))

    def find_note_by_name(self, name: str, parent_id: str = "0") -> str | None:
        """Return fileId of a non-dir note matching ``name`` under ``parent_id``.

        Youdao appends a ``.note`` suffix to stored note names, so match both
        ``name`` and ``name + '.note'`` case-insensitively.
        """
        wanted = {name.lower(), f"{name}.note".lower()}
        data = self.list_notes(parent_id)
        for e in data.get("entries") or []:
            if e.get("dir"):
                continue
            if str(e.get("name", "")).lower() in wanted:
                return str(e.get("id"))
        return None

    def update_markdown_note(self, file_id: str, content: str, title: str | None = None) -> dict[str, Any]:
        args: dict[str, Any] = {"fileId": file_id, "content": content}
        if title:
            args["title"] = title
        return self._call("updateMarkdownNote", args, timeout=max(self.timeout, 60))

    def delete_note(self, file_id: str) -> dict[str, Any]:
        return self._call("deleteNote", {"fileId": file_id}, timeout=max(self.timeout, 60))


def sanitize_youdao_title(title: str, fallback: str = "OpenClaw报告.md") -> str:
    clean = re.sub(r"[\\/:*?\"<>|]", "-", title or fallback).strip() or fallback
    if not clean.lower().endswith(".md"):
        clean += ".md"
    if len(clean) > 80:
        clean = clean[:77].rstrip("- _.。") + ".md"
    return clean


def backup_to_youdao_note(cfg: dict[str, Any], *, title: str, content: str, timeout: int = 30) -> DeliveryStatus:
    if delivery_suppressed():
        return DeliveryStatus("有道云保存", "跳过", "OPENCLAW_REPORT_NO_DELIVER=1")
    yc = cfg.get("youdao") or {}
    if not yc.get("enabled"):
        return DeliveryStatus("有道云保存", "未启用", "config.report_delivery.youdao.enabled=false")
    tgt = yc.get("target") or {}
    root_folder = tgt.get("root_folder") or "openclaw"
    update_in_place = tgt.get("update_in_place", True)
    try:
        client = YoudaoMcpClient(cfg, timeout=timeout)
        client.connect()
        client.initialize()
        note_title = sanitize_youdao_title(title)
        # All report notes go flat into the top-level ``openclaw`` folder.
        parent_id = client.resolve_folder(root_folder, "0")
        existing = client.find_note_by_name(note_title, parent_id) if update_in_place else None
        if existing:
            # createNote-typed notes are not accepted by updateMarkdownNote, so
            # refresh by deleting the old note then creating it again in place.
            try:
                client.delete_note(existing)
            except Exception as e:
                logging.getLogger(__name__).error(f"[report_delivery] 操作失败: {e}", exc_info=True)
            result = client.create_note(note_title, content, parent_id=parent_id)
            action = "已更新"
        else:
            result = client.create_note(note_title, content, parent_id=parent_id)
            action = "已新建"
        text = ""
        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("text"):
                text += str(item.get("text"))[:180]
        return DeliveryStatus("有道云保存", "已备份", f"{action}：{text or 'ok'}", f"{root_folder}/{note_title}")
    except Exception as e:
        return DeliveryStatus("有道云保存", "未完成", redact(repr(e)[:500], [yc.get("api_key") or ""]), mask_secret(yc.get("api_key")))


def probe_youdao_mcp(cfg: dict[str, Any], timeout: int = 12) -> DeliveryStatus:
    yc = cfg.get("youdao") or {}
    if not yc.get("enabled"):
        return DeliveryStatus("有道云保存", "未启用", "config.report_delivery.youdao.enabled=false")
    url = yc.get("url")
    api_key = yc.get("api_key")
    if not url or not api_key:
        return DeliveryStatus("有道云保存", "未完成", "缺少有道云 MCP url 或 api_key")
    headers = expand_headers(yc.get("headers") or {}, {"api_key": api_key})
    req = request.Request(url, headers={**headers, "Accept": "text/event-stream"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            sample_lines: list[str] = []
            deadline = time.time() + min(timeout, 8)
            while time.time() < deadline and len(sample_lines) < 16:
                line = resp.readline()
                if not line:
                    break
                s = line.decode("utf-8", errors="replace").strip()
                if s:
                    sample_lines.append(s)
                if "/api/ynote/mcp/message" in s or "event:endpoint" in s:
                    break
            sample = "\n".join(sample_lines)
            if code == 200 and ("event:endpoint" in sample or "sessionId=" in sample or "/api/ynote/mcp/message" in sample):
                return DeliveryStatus("有道云保存", "MCP已配置/端点可用", "SSE MCP endpoint 可连接；共用底座可直接调用 createNote 备份", mask_secret(api_key))
            return DeliveryStatus("有道云保存", "待确认", f"HTTP {code}, response={sample[:120]!r}", mask_secret(api_key))
    except error.HTTPError as e:
        body = ""
        try:
            body = e.read(240).decode("utf-8", errors="replace")
        except Exception as e:
            logging.getLogger(__name__).error(f"[report_delivery] 操作失败: {e}", exc_info=True)
        return DeliveryStatus("有道云保存", "未完成", f"HTTP {e.code}: {redact(body[:180], [api_key])}", mask_secret(api_key))
    except Exception as e:
        return DeliveryStatus("有道云保存", "未完成", redact(repr(e)[:220], [api_key]), mask_secret(api_key))


# ----------------------------- footer/finalize -----------------------------


def channel_statuses(cfg: dict[str, Any]) -> list[DeliveryStatus]:
    ch = cfg.get("channels") or {}
    out: list[DeliveryStatus] = []
    feishu = ch.get("feishu") or {}
    out.append(DeliveryStatus("飞书", "已配置" if feishu.get("enabled") and feishu.get("chat_id") else "未完成", "共用底座使用聊天框新消息分段发送；禁止 --reply-to/--thread-id/引用回复", feishu.get("chat_id", "")))
    wx = ch.get("weixin") or {}
    out.append(DeliveryStatus("微信", "已配置" if wx.get("enabled") and wx.get("target") else "未完成", "调用共用底座 deliver_all(...) 实际发送" if wx.get("target") else "缺少 target", wx.get("target", "")))
    qq = ch.get("qq") or {}
    out.append(DeliveryStatus("QQ", "已配置" if qq.get("enabled") and qq.get("target") else "未完成", "调用共用底座 deliver_all(...) 实际发送" if qq.get("target") else "缺少 target", qq.get("target", "")))
    wc = ch.get("webchat") or {}
    wc_mode = wc.get("mode") or "current_session"
    out.append(DeliveryStatus("webchat", "已配置" if wc.get("enabled") and wc_mode == "current_session" else "未完成", "webchat 由当前 assistant 回复承载正文/摘要供用户检验" if wc.get("enabled") and wc_mode == "current_session" else f"不支持的 webchat mode={wc_mode!r}", wc_mode))
    return out


def strip_existing_footer(markdown: str) -> str:
    marker = "\n## 备案与投递状态\n"
    return (markdown.split(marker, 1)[0].rstrip() + "\n") if marker in markdown else (markdown.rstrip() + "\n")


def parse_existing_footer(markdown: str) -> dict[str, str]:
    """Return {channel_name: full_line} from an existing footer, if any.

    Used to preserve prior channel statuses when a later run only re-delivers a
    subset of channels, so partial补发 never erases confirmed history.
    """
    marker = "\n## 备案与投递状态\n"
    if marker not in markdown:
        return {}
    tail = markdown.split(marker, 1)[1]
    out: dict[str, str] = {}
    for raw in tail.splitlines():
        line = raw.strip()
        m = re.match(r"^-\s*([^：:]+?)状态：", line)
        if m:
            out[m.group(1).strip()] = line
    return out


def status_footer(report_path: str | Path, statuses: list[DeliveryStatus], preserved: dict[str, str] | None = None) -> str:
    lines = ["## 备案与投递状态", "", f"- 桌面路径：{Path(report_path)}。"]
    current_names = {s.name for s in statuses}
    lines.extend(s.to_line() for s in statuses)
    # Preserve prior channel lines that this run did not touch.
    for name, line in (preserved or {}).items():
        if name in current_names or name == "桌面路径":
            continue
        lines.append(f"{line}（上次结果保留）")
    lines.append(f"- 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}。")
    return "\n".join(lines) + "\n"


def finalize_report(report_path: str | Path, config_path: str | Path = DEFAULT_CONFIG, probe_youdao: bool = False, *, deliver: bool = False, channels: Sequence[str] | None = None, title: str | None = None, body: str | None = None, max_chars: int | None = None, backup_youdao: bool = True) -> dict[str, Any]:
    p = Path(report_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    cfg = load_config(config_path)
    original = p.read_text(encoding="utf-8")
    report_body = body if body is not None else original
    final_title = title or p.name

    if delivery_suppressed():
        deliver = False
        backup_youdao = False
        probe_youdao = False

    statuses: list[DeliveryStatus] = []
    if backup_youdao:
        statuses.append(backup_to_youdao_note(cfg, title=final_title, content=strip_existing_footer(report_body)))
    elif probe_youdao:
        statuses.append(probe_youdao_mcp(cfg))
    else:
        statuses.append(DeliveryStatus("有道云保存", "跳过", "本次调用显式 --no-youdao-backup"))

    if deliver:
        statuses.extend(deliver_all(report_body, title=final_title, config_path=config_path, channels=channels, max_chars=max_chars, include_webchat=True))
    else:
        statuses.extend(channel_statuses(cfg))

    preserved = parse_existing_footer(original)
    p.write_text(strip_existing_footer(original) + "\n" + status_footer(p, statuses, preserved), encoding="utf-8")
    return {
        "report_path": str(p),
        "config_path": str(config_path),
        "delivered": deliver,
        "youdao_backup": backup_youdao,
        "statuses": [
            {"name": s.name, "status": s.status, "reason": s.reason, "target": s.target, "chunks": [c.__dict__ for c in s.chunks]}
            for s in statuses
        ],
    }


def main() -> int:
    import argparse

    level_channels = {
        "top1": ["feishu", "weixin", "qq"],
        "top2": ["qq", "feishu"],
        "top3": ["feishu"],
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("report_path")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--probe-youdao", action="store_true")
    parser.add_argument("--no-youdao-backup", action="store_true", help="显式跳过有道云备份；默认会备份")
    parser.add_argument("--deliver", action="store_true", help="实际投递到 feishu/weixin/qq，并同步发送到 webchat 供检验")
    parser.add_argument("--channels", default="", help="逗号分隔的渠道白名单，如 feishu,qq。留空表示 feishu+weixin+qq。")
    parser.add_argument("--level", default="", help="按协议分级选择渠道：Top1=飞书+微信+QQ，Top2=QQ+飞书，Top3=飞书；webchat 由当前会话回显检验")
    parser.add_argument("--title", default="", help="投递分段和有道云笔记标题")
    parser.add_argument("--max-chars", type=int, default=0)
    args = parser.parse_args()

    channels = [c.strip() for c in args.channels.split(",") if c.strip()] or None
    if channels is None and args.level:
        level_key = args.level.strip().lower()
        if level_key not in level_channels:
            parser.error("--level must be one of Top1, Top2, Top3")
        channels = level_channels[level_key]
    result = finalize_report(
        args.report_path,
        args.config,
        args.probe_youdao,
        deliver=args.deliver,
        channels=channels,
        title=args.title or None,
        max_chars=args.max_chars or None,
        backup_youdao=not args.no_youdao_backup,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    failed = [s for s in result.get("statuses", []) if s.get("status") in {"未完成", "失败"}]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
