/**
 * 牧云天枢 V5 — 系统初始化模块
 * 负责：环境检测 / 全局错误兜底 / 键盘快捷键 / 服务心跳检测
 */

(function() {
  'use strict';

  // ── 1. 全局未捕获异常兜底 ──
  window.addEventListener('error', function(e) {
    console.error('[quant-web] 未捕获异常:', e.error?.message || e.message);
    const statusEl = document.getElementById('status');
    if (statusEl && !e.target?.closest) {
      statusEl.textContent = '页面暂时遇到问题，请刷新重试';
      statusEl.style.color = '#b42318';
    }
  });

  window.addEventListener('unhandledrejection', function(e) {
    console.error('[V5] 未处理的 Promise 拒绝:', e.reason?.message || e.reason);
  });

  // ── 2. 服务心跳检测 ──
  let heartbeatFailures = 0;
  async function checkHeartbeat() {
    try {
      const request = window.quantApiFetch || fetch;
      const res = await request('/api/market', { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        heartbeatFailures = 0;
        const badge = document.getElementById('v5Heartbeat');
        if (badge) {
          badge.textContent = '🟢';
          badge.title = '服务正常';
        }
      } else {
        throw new Error('服务请求失败');
      }
    } catch(e) {
      heartbeatFailures++;
      const badge = document.getElementById('v5Heartbeat');
      if (badge) {
        if (heartbeatFailures >= 3) {
          badge.textContent = '🔴';
          badge.title = '服务可能异常，连续 ' + heartbeatFailures + ' 次心跳失败';
        } else {
          badge.textContent = '🟡';
          badge.title = '心跳异常 (' + heartbeatFailures + '/3)';
        }
      }
      // 连续10次失败 → 提示用户重启
      if (heartbeatFailures >= 10) {
        const statusEl = document.getElementById('status');
        if (statusEl) {
          statusEl.textContent = '服务暂时不可用，请稍后再试';
          statusEl.style.color = '#b42318';
        }
      }
    }
  }
  // 每30秒心跳
  setInterval(checkHeartbeat, 30000);
  // 立即执行一次
  setTimeout(checkHeartbeat, 1000);

  // ── 3. 键盘快捷键 ──
  document.addEventListener('keydown', function(e) {
    // Ctrl+Shift+R: 强制刷新总览
    if (e.ctrlKey && e.shiftKey && e.key === 'R') {
      e.preventDefault();
      const dashboardBtn = document.querySelector('.nav[data-view="dashboard"]');
      if (dashboardBtn) dashboardBtn.click();
      return;
    }
    // Escape: 关闭下拉框
    if (e.key === 'Escape') {
      const dropdowns = document.querySelectorAll('.autocomplete-dropdown');
      dropdowns.forEach(d => d.style.display = 'none');
    }
    // / 键: 聚焦扫描输入框
    if (e.key === '/' && !e.ctrlKey && !e.metaKey && !e.target?.closest('input,textarea,select')) {
      e.preventDefault();
      const scanInput = document.getElementById('scanSymbols');
      if (scanInput) scanInput.focus();
    }
  });

  // ── 4. 状态栏心跳指示器插入 ──
  const statusEl = document.getElementById('status');
  if (statusEl) {
    const badge = document.createElement('span');
    badge.id = 'v5Heartbeat';
    badge.textContent = '⚪';
    badge.title = '检测中...';
    badge.style.marginLeft = '8px';
    badge.style.fontSize = '12px';
    statusEl.after(badge);
  }

  // ── 5. 页面可见性变化时暂停/恢复后台轮询 ──
  document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
      document.body.dataset.v5Hidden = 'true';
    } else {
      delete document.body.dataset.v5Hidden;
      // 回到前台时刷新总览如果可见
      const activeSection = document.querySelector('.view.active');
      if (activeSection && activeSection.id === 'dashboard') {
        loadMacroOverview().catch(() => {});
      }
    }
  });

  console.log('[V5] 牧云天枢 初始化完成');
})();
