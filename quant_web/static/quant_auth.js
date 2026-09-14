/* quant_auth.js — 上线认证辅助（2026-08-24）
 * 不把 API key 写进源码；由用户在浏览器输入后存于 sessionStorage，
 * 所有 fetch 统一带 X-API-Key。未配置 key 时为回环模式兼容调用。
 */
(function () {
  'use strict';
  var KEY = 'quant_web_api_key';

  window.quantApiKey = function () {
    try { return window.sessionStorage.getItem(KEY) || ''; } catch (e) { return ''; }
  };

  window.quantSetKey = function (k) {
    try {
      if (k) { window.sessionStorage.setItem(KEY, String(k).trim()); }
      else { window.sessionStorage.removeItem(KEY); }
    } catch (e) { /* storage unavailable */ }
  };

  window.quantApiFetch = function (url, options) {
    options = options || {};
    options.headers = Object.assign({}, options.headers || {});
    var key = window.quantApiKey();
    if (key) { options.headers['X-API-Key'] = key; }
    var retried = false;
    return fetch(url, options).then(function (r) {
      if (r.status === 401 && !retried && window.confirm('需要验证身份。是否现在输入访问凭证？')) {
        retried = true;
        var k = window.prompt('请输入访问凭证（仅保存在当前会话）');
        if (k && k.trim()) {
          window.quantSetKey(k);
          options.headers['X-API-Key'] = k.trim();
          return fetch(url, options);
        }
      }
      return r;
    });
  };

  // 统一 JSON 响应：支持 {ok,data}、{ok,status:computing} 和裸数据。
  window.quantApiJson = async function (url, options) {
    var r = await window.quantApiFetch(url, options || {});
    var body;
    try { body = await r.json(); } catch (e) { throw new Error('服务返回格式异常'); }
    if (!r.ok) throw new Error('服务请求失败，请稍后重试');
    return body || {};
  };
  window.quantApiData = function (body) {
    if (!body || typeof body !== 'object') return {};
    if (body.status === 'computing') return body;
    return Object.prototype.hasOwnProperty.call(body, 'data') ? (body.data || {}) : body;
  };
  window.quantApiWait = async function (url, options, maxPolls, delayMs) {
    maxPolls = maxPolls == null ? 60 : maxPolls;
    delayMs = delayMs == null ? 1000 : delayMs;
    var last = {};
    for (var i = 0; i <= maxPolls; i++) {
      last = await window.quantApiJson(url, options || {});
      if (last.status !== 'computing' || i >= maxPolls) return last;
      await new Promise(function (resolve) { setTimeout(resolve, delayMs); });
    }
    return last;
  };

  // 若页面已存在便捷别名，保留兼容
  if (!window.apiGet) {
    window.apiGet = async function (path) {
      var r = await window.quantApiFetch(path);
      return r.json();
    };
  }
  if (!window.apiPost) {
    window.apiPost = async function (path, body) {
      var r = await window.quantApiFetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {}),
      });
      return r.json();
    };
  }

  // 全局包装 fetch：同源 /api/* 自动带 X-API-Key，覆盖所有直接用 fetch 的页面代码。
  // 保留原函数引用，避免重复包装。
  if (!window.__quantFetchWrapped) {
    var _origFetch = window.fetch.bind(window);
    window.fetch = function (input, init) {
      init = init || {};
      init.headers = Object.assign({}, init.headers || {});
      var url = typeof input === 'string' ? input : (input && input.url);
      if (url && url.indexOf('/api/') === 0 && !(init.headers['X-API-Key'] || '')) {
        var key = window.quantApiKey();
        if (key) init.headers['X-API-Key'] = key;
      }
      return _origFetch(input, init).then(function (r) {
        if (r.status === 401 && url && url.indexOf('/api/') === 0 &&
            window.confirm('需要验证身份。是否现在输入访问凭证？')) {
          var k = window.prompt('请输入访问凭证（仅保存在当前会话）');
          if (k && k.trim()) {
            window.quantSetKey(k);
            init.headers['X-API-Key'] = k.trim();
            return _origFetch(input, init);
          }
        }
        return r;
      });
    };
    window.__quantFetchWrapped = true;
  }
})();