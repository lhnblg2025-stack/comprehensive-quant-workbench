"""
quant_web 处理函数模块

每个 handler 接收 (query: dict[str, list[str]], send_json: callable) 并处理请求。
send_json(data, status=200) 是 server 的回传函数。
"""

from __future__ import annotations
