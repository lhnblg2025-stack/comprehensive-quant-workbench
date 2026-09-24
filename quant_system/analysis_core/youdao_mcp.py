# -*- coding: utf-8 -*-
"""有道云笔记 MCP 客户端（轻量 SSE，无第三方 MCP SDK 依赖）。

用于从有道云笔记拉取研报/笔记，交给 research_flow.ingest_youdao_notes() 分析。

用法:
    from quant_system.analysis_core.youdao_mcp import YoudaoMCPClient
    with YoudaoMCPClient() as client:
        tools = client.list_tools()
        notes = client.fetch_research_notes()
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.parse
from typing import Any

import requests

logger = logging.getLogger("youdao_mcp")

DEFAULT_URL = "https://open.mail.163.com/api/ynote/mcp/sse"
DEFAULT_TIMEOUT = 20


class YoudaoMCPClient:
    """极简 MCP over SSE 客户端。

    流程:
      1. GET sse -> 从事件流拿到 message endpoint
      2. 后台线程持续读 SSE, 把 data 事件放入队列
      3. 发送 JSON-RPC POST 到 message endpoint, 响应从队列中按 id 匹配
    """

    def __init__(self, url: str = DEFAULT_URL, headers: dict[str, str] | None = None,
                 timeout: int = DEFAULT_TIMEOUT) -> None:
        self.url = url
        self.headers = headers or {}
        self.headers.setdefault("Accept", "text/event-stream")
        self.timeout = timeout
        self._resp: requests.Response | None = None
        self._endpoint: str | None = None
        self._msg_url: str | None = None
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()

    # ── 连接 ──────────────────────────────────────────────
    def connect(self) -> None:
        if self._resp is not None:
            return
        try:
            resp = requests.get(self.url, headers=self.headers, stream=True,
                                timeout=self.timeout)
            resp.raise_for_status()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"有道云 MCP SSE 连接失败: {e}") from e
        self._resp = resp
        # 读取前若干字节拿 endpoint
        it = resp.iter_lines(decode_unicode=True)
        endpoint = None
        try:
            for line in it:
                if line.startswith("data:"):
                    endpoint = line[5:].strip()
                    break
                if line.startswith("event:endpoint"):
                    continue
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"读取有道云 MCP endpoint 失败: {e}") from e
        if not endpoint:
            raise RuntimeError("有道云 MCP 未返回 endpoint")
        self._endpoint = endpoint
        self._msg_url = urllib.parse.urljoin(self.url, endpoint)
        # 后台线程继续读后续事件
        self._stop.clear()
        self._reader = threading.Thread(target=self._read_loop, args=(it,), daemon=True)
        self._reader.start()
        logger.info("[youdao_mcp] connected endpoint=%s", endpoint)

    def _read_loop(self, lines) -> None:
        """持续读 SSE 行，data: 开头视为 JSON-RPC 事件。"""
        data_buf: list[str] = []
        try:
            for line in lines:
                if self._stop.is_set():
                    break
                if line.startswith("data:"):
                    data_buf.append(line[5:].strip())
                    continue
                if line.strip() == "" and data_buf:
                    raw = "\n".join(data_buf)
                    data_buf = []
                    try:
                        obj = json.loads(raw)
                        self._queue.put(obj)
                    except Exception:
                        logger.debug("[youdao_mcp] 非 JSON 事件: %.80s", raw)
        except Exception as e:  # noqa: BLE001
            logger.warning("[youdao_mcp] SSE 读取结束: %s", e)
        finally:
            if data_buf:
                raw = "\n".join(data_buf)
                try:
                    self._queue.put(json.loads(raw))
                except Exception:
                    pass

    def close(self) -> None:
        self._stop.set()
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:
                pass
            self._resp = None

    def __enter__(self) -> "YoudaoMCPClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── JSON-RPC ──────────────────────────────────────────
    def _request(self, method: str, params: dict[str, Any], req_id: int | None = None) -> Any:
        if self._msg_url is None:
            raise RuntimeError("未连接")
        req_id = req_id or int(time.time() * 1000)
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        headers = {**self.headers, "Content-Type": "application/json"}
        resp = requests.post(self._msg_url, data=json.dumps(payload).encode(),
                             headers=headers, timeout=self.timeout)
        if resp.status_code not in (200, 202):
            raise RuntimeError(f"有道云 MCP 请求失败 HTTP {resp.status_code}: {resp.text[:200]}")
        # 响应可能通过 SSE 推送；先尝试 HTTP body
        if resp.text and resp.text.strip():
            try:
                obj = json.loads(resp.text)
                if obj.get("id") == req_id:
                    return obj
            except Exception:
                pass
        # 从 SSE 队列按 id 等待
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                obj = self._queue.get(timeout=2)
            except queue.Empty:
                continue
            if obj.get("id") == req_id:
                if "error" in obj:
                    raise RuntimeError(f"有道云 MCP 错误: {obj['error']}")
                return obj
        raise TimeoutError("有道云 MCP 响应超时")

    # ── 工具操作 ──────────────────────────────────────────
    def initialize(self) -> dict[str, Any]:
        return self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "quant-research", "version": "1.0"},
        }, req_id=1)

    def list_tools(self) -> list[dict[str, Any]]:
        resp = self._request("tools/list", {}, req_id=2)
        return (resp.get("result") or {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        resp = self._request("tools/call", {"name": name, "arguments": arguments}, req_id=3)
        result = resp.get("result") or {}
        return result

    # ── 研报拉取 ──────────────────────────────────────────
    def fetch_research_notes(self, max_notes: int = 20) -> list[dict[str, Any]]:
        """尝试用有道云 MCP 工具拉取笔记/研报。

        工具名不确定时，先列出工具，再按常见命名调用 list/search/get。
        """
        tools = self.list_tools()
        names = [t.get("name", "") for t in tools]
        logger.info("[youdao_mcp] 可用工具: %s", names)

        # 候选工具名
        list_names = [n for n in names if n.lower() in ("list_notes", "list_ynotes", "notes_list", "list_docs")]
        search_names = [n for n in names if "search" in n.lower() and "note" in n.lower()]
        get_names = [n for n in names if n.lower() in ("get_note", "get_ynote", "note_get", "get_doc")]

        notes: list[dict[str, Any]] = []
        try:
            if list_names:
                res = self.call_tool(list_names[0], {"limit": max_notes, "start": 0})
                notes = _extract_notes(res)
            elif search_names:
                res = self.call_tool(search_names[0], {"query": "研报", "limit": max_notes})
                notes = _extract_notes(res)
            else:
                logger.warning("[youdao_mcp] 未找到 list/search 笔记工具，无法自动拉取；可用工具=%s", names)
                return []
        except Exception as e:  # noqa: BLE001
            logger.warning("[youdao_mcp] 拉取笔记失败: %s", e)
            return []

        # 尝试获取每个笔记正文
        if get_names and notes:
            for n in notes[:max_notes]:
                nid = n.get("id") or n.get("noteId") or n.get("uuid") or n.get("fileId")
                if not nid:
                    continue
                try:
                    body = self.call_tool(get_names[0], {"id": nid})
                    text = _extract_note_text(body)
                    n["content"] = text
                except Exception as e:  # noqa: BLE001
                    n["content"] = ""
                    logger.warning("[youdao_mcp] 获取笔记正文失败 %s: %s", nid, e)
        return notes


def _extract_notes(res: Any) -> list[dict[str, Any]]:
    """从 tools/call 返回中尽力提取笔记列表。"""
    if isinstance(res, dict):
        # 常见结构: content[0].text 可能是 JSON 字符串
        content = res.get("content") or res.get("notes") or res.get("data") or []
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except Exception:
                content = []
        if isinstance(content, list):
            out = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item and isinstance(item["text"], str):
                        try:
                            obj = json.loads(item["text"])
                            if isinstance(obj, list):
                                out.extend(obj)
                            elif isinstance(obj, dict):
                                out.append(obj)
                        except Exception:
                            out.append({"title": item.get("text", "")[:50], "content": item["text"]})
                    else:
                        out.append(item)
                elif isinstance(item, str):
                    out.append({"title": item[:50], "content": item})
            return out
        if isinstance(content, dict):
            return _extract_notes(content)
    return []


def _extract_note_text(res: Any) -> str:
    if isinstance(res, dict):
        content = res.get("content") or res.get("text") or res.get("data") or ""
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict):
                    if isinstance(c.get("text"), str):
                        parts.append(c["text"])
                elif isinstance(c, str):
                    parts.append(c)
            return "\n".join(parts)
        if isinstance(content, str):
            return content
    return ""


if __name__ == "__main__":
    import json as _json
    import os as _os
    cfg = _json.loads(_os.popen("python3 -c \"import json,os;print(json.dumps(json.load(open(os.path.expanduser('${OPENCLAW_CONFIG:-$HOME/.config/openclaw/config.json}')))['mcp']['servers']['youdao-ynote']))\"").read())
    with YoudaoMCPClient(cfg["url"], headers=cfg.get("headers", {})) as client:
        print("tools:", [t.get("name") for t in client.list_tools()])
