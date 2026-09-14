"""Loopback-only demo server; no filesystem, trading or credential endpoints."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from quant_platform.demo import run

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/livez':
            body, kind = b'{"status":"ok"}', 'application/json'
        elif self.path == '/api/demo':
            body = json.dumps(run(), ensure_ascii=False, allow_nan=False, default=str).encode()
            kind = 'application/json'
        elif self.path == '/':
            body = '''<!doctype html><meta charset="utf-8"><title>综合量化工作台 · 公开核心</title>
            <style>body{max-width:960px;margin:60px auto;font:18px system-ui;background:#101827;color:#e2e8f0}button{padding:12px}pre{white-space:pre-wrap}</style>
            <h1>综合量化工作台 · 公开核心</h1><p>通用回测引擎与均线策略模板。合成数据，无真实账户，无实盘交易。</p>
            <p>本页为公开核心演示，不是原工作台全部功能。仅供研究，不构成投资建议。</p>
            <button onclick="fetch('/api/demo').then(r=>r.json()).then(x=>document.getElementById('result').textContent=JSON.stringify(x,null,2))">运行离线模板回测</button><pre id="result"></pre>'''.encode()
            kind = 'text/html'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', kind + '; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8600)
    args = parser.parse_args()
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
