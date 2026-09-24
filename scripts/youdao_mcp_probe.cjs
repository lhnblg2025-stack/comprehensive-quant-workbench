#!/usr/bin/env node
/* 有道云笔记 MCP 最小客户端：列出工具 / 调用工具（如搜笔记查重）。
用法:
  node youdao_mcp_probe.mjs tools-list
  node youdao_mcp_probe.mjs call <toolName> '<json args>'
  node youdao_mcp_probe.mjs search '<关键词>'        # 便捷：searchNote/搜索笔记
*/
const https = require('https');

const SSE_URL = 'https://open.mail.163.com/api/ynote/mcp/sse';
const API_KEY = process.env.YOUDAONOTE_API_KEY || 'iv1rsUfZy6eV8I6wzwqrEnAzCCsMjxjUvcx-c616c56c3f59bf3a';
const CONNECT_TIMEOUT_MS = 20000;

class SseParser {
  constructor(onEvent) {
    this.buf = ''; this.onEvent = onEvent;
  }
  feed(chunk) {
    this.buf += chunk;
    let idx;
    while ((idx = this.buf.indexOf('\n\n')) >= 0) {
      const block = this.buf.slice(0, idx);
      this.buf = this.buf.slice(idx + 2);
      let type = 'message', data = '';
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) type = line.slice(6).trim();
        else if (line.startsWith('data:')) data += line.slice(5).trim();
      }
      if (data) this.onEvent({ type, data });
    }
  }
}

class McpClient {
  constructor(sseUrl, apiKey) {
    this.sseUrl = new URL(sseUrl); this.apiKey = apiKey;
    this.pending = new Map(); this.nextId = 1;
    this.messageUrl = null; this.res = null; this.req = null; this.connected = false;
  }
  connect() {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.close(); reject(new Error('SSE 连接超时')); }, CONNECT_TIMEOUT_MS);
      const parser = new SseParser((ev) => {
        if (ev.type === 'endpoint') { this.messageUrl = new URL(ev.data, this.sseUrl); this.connected = true; clearTimeout(timer); resolve(); }
        else if (ev.type === 'message') this.handle(ev.data);
      });
      this.req = https.request({
        hostname: this.sseUrl.hostname, port: 443,
        path: this.sseUrl.pathname + this.sseUrl.search,
        method: 'GET', headers: { 'Accept': 'text/event-stream', 'Cache-Control': 'no-cache', 'x-api-key': this.apiKey },
      }, (res) => {
        if (res.statusCode !== 200) { clearTimeout(timer); reject(new Error(`SSE HTTP ${res.statusCode}`)); return; }
        this.res = res; res.setEncoding('utf-8');
        res.on('data', (c) => parser.feed(c));
        res.on('end', () => { this.connected = false; });
        res.on('error', reject);
      });
      this.req.on('error', reject); this.req.end();
    });
  }
  handle(data) {
    try {
      const msg = JSON.parse(data);
      if (msg.id != null && this.pending.has(msg.id)) {
        const p = this.pending.get(msg.id); this.pending.delete(msg.id); p.resolve(msg);
      }
    } catch {}
  }
  async post(body) {
    if (!this.connected) throw new Error('未连接');
    const r = await fetch(this.messageUrl.toString(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream', 'x-api-key': this.apiKey },
      body: JSON.stringify(body),
    });
    return r;
  }
  async send(method, params = {}) {
    const id = this.nextId++;
    const body = { jsonrpc: '2.0', id, method, params };
    let resolveRes, rejectRes;
    const promise = new Promise((res, rej) => { resolveRes = res; rejectRes = rej; });
    const timer = setTimeout(() => { this.pending.delete(id); rejectRes(new Error(`超时: ${method}`)); }, 60000);
    this.pending.set(id, { resolve: (m) => { clearTimeout(timer); resolveRes(m); }, reject: (e) => { clearTimeout(timer); rejectRes(e); } });
    const postRes = await this.post(body);
    try {
      const ct = postRes.headers.get('content-type') || '';
      if (ct.includes('application/json')) {
        const direct = await postRes.json();
        if (direct?.jsonrpc && direct.id === id && this.pending.has(id)) {
          this.pending.delete(id); clearTimeout(timer); resolveRes(direct);
        }
      }
    } catch {}
    return promise;
  }
  async call(method, params) {
    const res = await this.send(method, params);
    return res;
  }
  close() { this.connected = false; if (this.res) this.res.destroy(); if (this.req) this.req.destroy(); for (const p of this.pending.values()) p.reject(new Error('关闭')); this.pending.clear(); }
}

async function main() {
  const [cmd, arg1, arg2] = process.argv.slice(2);
  if (!cmd) { console.error('用法: youdao_mcp_probe.mjs tools-list | call <tool> <json> | search <kw>'); process.exit(1); }
  const client = new McpClient(SSE_URL, API_KEY);
  try {
    await client.connect();
    await client.send('initialize', {
      protocolVersion: '2024-11-05', capabilities: {},
      clientInfo: { name: 'youdao-probe', version: '1.0.0' },
    });
    await client.post({ jsonrpc: '2.0', method: 'notifications/initialized' });
    if (cmd === 'tools-list') {
      const r = await client.send('tools/list');
      const tools = r.result?.tools || [];
      console.log(JSON.stringify(tools.map(t => ({ name: t.name, desc: (t.description || '').slice(0, 120) })), null, 1));
    } else if (cmd === 'call') {
      const args = arg2 ? JSON.parse(arg2) : {};
      const r = await client.send('tools/call', { name: arg1, arguments: args });
      console.log(JSON.stringify(r, null, 1).slice(0, 200000));
    } else if (cmd === 'search') {
      const r = await client.send('tools/call', { name: 'searchNote', arguments: { query_info: { title: arg1 }, start: 0, end: 20 } });
      console.log(JSON.stringify(r, null, 1).slice(0, 200000));
    }
  } catch (e) {
    console.error('ERR', e.message); process.exit(1);
  } finally {
    client.close();
  }
}
main();
