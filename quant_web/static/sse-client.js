/*
 * SSE client using fetch + ReadableStream so authenticated deployments can
 * send X-API-Key. The key is read from the current browser session only.
 */
(function () {
  'use strict';

  var SSE_URL = '/api/events';
  var RECONNECT_DELAY_MS = 5000;
  var controller = null;
  var connected = false;
  var listeners = {};
  var reconnectTimer = null;
  var manualDisconnect = false;

  function on(event, fn) {
    if (!listeners[event]) listeners[event] = [];
    listeners[event].push(fn);
  }
  function off(event, fn) {
    if (!listeners[event]) return;
    listeners[event] = listeners[event].filter(function (f) { return f !== fn; });
  }
  function emit(event, data) {
    (listeners[event] || []).slice().forEach(function (fn) {
      try { fn(data); } catch (e) { console.error('[SSE] listener error:', e); }
    });
  }
  function sessionKey() {
    try {
      if (typeof window.quantApiKey === 'function') return window.quantApiKey() || '';
      return window.sessionStorage.getItem('quant_web_api_key') || '';
    } catch (e) { return ''; }
  }
  function dispatch(event, data) {
    var parsed;
    try { parsed = JSON.parse(data); } catch (e) { return; }
    emit(event || 'message', parsed);
  }
  function scheduleReconnect() {
    if (manualDisconnect || reconnectTimer) return;
    reconnectTimer = setTimeout(function () {
      reconnectTimer = null;
      connect();
    }, RECONNECT_DELAY_MS);
  }
  async function connect() {
    if (controller || manualDisconnect) return;
    manualDisconnect = false;
    controller = new AbortController();
    var current = controller;
    var headers = { Accept: 'text/event-stream' };
    var key = sessionKey();
    if (key) headers['X-API-Key'] = key;
    try {
      var response = await fetch(SSE_URL, { headers: headers, cache: 'no-store', signal: current.signal });
      if (!response.ok) throw new Error('HTTP ' + response.status);
      if (!response.body) throw new Error('ReadableStream unavailable');
      connected = true;
      emit('connected', {});
      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = '', event = '', data = [];
      function consume(line) {
        if (line === '') {
          if (data.length) dispatch(event, data.join('\n'));
          event = ''; data = [];
        } else if (line.charAt(0) !== ':') {
          var colon = line.indexOf(':');
          var field = colon < 0 ? line : line.slice(0, colon);
          var value = colon < 0 ? '' : line.slice(colon + 1).replace(/^ /, '');
          if (field === 'event') event = value;
          else if (field === 'data') data.push(value);
        }
      }
      while (true) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        var lines = buffer.split(/\r?\n/);
        buffer = lines.pop();
        lines.forEach(consume);
      }
      if (buffer) consume(buffer);
    } catch (e) {
      if (!current.signal.aborted) console.warn('[SSE] connection error:', e.message || e);
    } finally {
      if (controller === current) controller = null;
      if (connected) { connected = false; emit('disconnected', {}); }
      if (!manualDisconnect) scheduleReconnect();
    }
  }
  function disconnect() {
    manualDisconnect = true;
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
    if (controller) controller.abort();
    controller = null;
    if (connected) { connected = false; emit('disconnected', {}); }
  }
  window.sseClient = {
    connect: function () { manualDisconnect = false; return connect(); },
    disconnect: disconnect,
    on: on,
    off: off,
    isConnected: function () { return connected; }
  };
  function boot() { setTimeout(connect, 500); }
  if (document.readyState === 'complete') boot();
  else document.addEventListener('DOMContentLoaded', boot, { once: true });
  document.addEventListener('visibilitychange', function () {
    if (document.hidden) disconnect();
    else { manualDisconnect = false; setTimeout(connect, 1000); }
  });
})();
