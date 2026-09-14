/**
 * 牧云天枢 3.1 — 任务队列看板
 *
 * 显示后台任务的统计、实时状态、历史记录。
 * 对接 /api/freshness 和 /api/tasks。
 */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);

  async function loadTaskQueue() {
    const statusEl = $('taskQueueStatus');
    const statsEl = $('taskQueueStats');
    const tbody = $('taskQueueTbody');
    if (!statusEl || !statsEl || !tbody) return;

    try {
      statusEl.textContent = '⏳ 加载中...';

      // 并行获取任务数据 + 新鲜度
      const request = window.quantApiFetch || fetch;
      const [taskRes, freshRes] = await Promise.all([
        request('/api/tasks', { signal: AbortSignal.timeout(8000) }),
        request('/api/freshness', { signal: AbortSignal.timeout(8000) }),
      ]);

      const taskData = await taskRes.json();
      const freshData = await freshRes.json();

      if (!taskData.ok) {
        statusEl.textContent = '暂时无法获取任务情况，请稍后重试';
        return;
      }

      // ── 统计概览 ──
      const stats = taskData.stats || {};
      let statsHtml = '';
      const statCards = [
        { label: '已完成', value: stats.completed || 0, color: '#34d399' },
        { label: '运行中', value: stats.running || 0, color: '#60a5fa' },
        { label: '待执行', value: stats.pending || 0, color: '#fbbf24' },
        { label: '失败', value: stats.failed || 0, color: '#f87171' },
        { label: '平均耗时', value: (stats.avg_elapsed_seconds != null ? stats.avg_elapsed_seconds + ' 秒' : '-'), color: '#c8cdd3' },
        { label: '任务总数', value: stats.total || 0, color: '#c8cdd3' },
      ];
      statCards.forEach(function (c) {
        statsHtml += '<div class="macro-card"><div class="name">' + c.label +
          '</div><div class="value" style="color:' + c.color + ';">' +
          c.value + '</div></div>';
      });
      // 并发配置
      if (stats.config) {
        statsHtml += '<div class="macro-card"><div class="name">并发上限</div>' +
          '<div class="value" style="color:#8b8fa3;font-size:14px;">' +
          stats.config.max_concurrent + '</div></div>';
        statsHtml += '<div class="macro-card"><div class="name">超时</div>' +
          '<div class="value" style="color:#8b8fa3;font-size:14px;">' +
          stats.config.task_timeout + 's</div></div>';
      }
      statsEl.innerHTML = statsHtml;

      // ── 最近任务列表 ──
      const recent = taskData.recent || [];
      if (recent.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="color:#657282;text-align:center;">暂无任务记录</td></tr>';
      } else {
        let rowsHtml = '';
        recent.forEach(function (t) {
          const statusMap = {
            'completed': '<span style="color:#34d399;">✅ 完成</span>',
            'running': '<span style="color:#60a5fa;">🔄 运行中</span>',
            'pending': '<span style="color:#fbbf24;">⏳ 等待</span>',
            'failed': '<span style="color:#f87171;">❌ 失败</span>',
          };
          const statusHtml = statusMap[t.status] || t.status;
          const elapsed = t.completed && t.started
            ? (t.completed - t.started).toFixed(1) + ' 秒'
            : (t.status === 'running' ? '...' : '-');
          const retryInfo = (t.retry_count || 0) + '/' + (t.max_retries || 3);
          const createdStr = t.created ? new Date(t.created * 1000).toLocaleTimeString() : '-';
          const kindShort = (t.kind || '').length > 24
            ? t.kind.slice(0, 24) + '…'
            : (t.kind || '-');
          rowsHtml += '<tr><td style="font-family:monospace;font-size:11px;">' +
            (t.id || '').slice(0, 12) + '</td>' +
            '<td>' + kindShort + '</td>' +
            '<td>' + statusHtml + '</td>' +
            '<td>' + createdStr + '</td>' +
            '<td>' + elapsed + '</td>' +
            '<td>' + retryInfo + '</td></tr>';
        });
        tbody.innerHTML = rowsHtml;
      }

      // ── 最近失败（如果有） ──
      const failures = stats.recent_failures || [];
      if (failures.length > 0) {
        statusEl.innerHTML = '✅ ' + stats.completed + ' 完成 · ⚠️ ' + failures.length + ' 失败 ' +
          '<span style="color:#f87171;font-size:11px;">最近: ' +
          failures.slice(0, 2).map(function (f) {
            return f.kind + ' (' + (f.error || '').slice(0, 40) + ')';
          }).join('; ') + '</span>';
      } else {
        statusEl.textContent = '✅ 运行正常';
      }

      // ── 新鲜度显示 ──
      if (freshData.ok && freshData.entries) {
        const entries = Object.entries(freshData.entries).slice(0, 5);
        const freshLabels = {
          'live': '<span style="color:#34d399;">🟢 实时</span>',
          'cached': '<span style="color:#fbbf24;">🟡 缓存</span>',
          'stale': '<span style="color:#f87171;">🔴 过期</span>',
        };
        let freshHtml = '';
        entries.forEach(function (_a) {
          var key = _a[0];
          var info = _a[1];
          var label = freshLabels[info.freshness] || info.freshness;
          freshHtml += '<div style="padding:2px 4px;font-size:12px;color:#9ca3af;">' +
            key.slice(0, 30) + ' → ' + label +
            ' <span style="color:#657282;">' + info.age_seconds + 's</span></div>';
        });
        if (freshHtml) {
          // 如果 stats 区域后面有新鲜度显示位，追加
          var existingFresh = document.querySelector('.freshness-summary');
          if (!existingFresh) {
            statsEl.insertAdjacentHTML('beforeend',
              '<div class="freshness-summary" style="margin-top:8px;">' +
              '<div style="font-size:12px;color:#8b8fa3;margin-bottom:4px;">📡 数据新鲜度</div>' +
              freshHtml + '</div>');
          }
        }
      }

    } catch (e) {
      console.error('[tasks] load error:', e);
      statusEl.textContent = '暂时无法获取任务情况，请稍后重试';
    }
  }

  // ── 初始化 ──
  function init() {
    // 绑定刷新按钮
    var refreshBtn = $('refreshTasksBtn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', loadTaskQueue);
    }

    // 监听子视图激活
    var observer = new MutationObserver(function () {
      var tasksView = $('o-tasks');
      if (tasksView && tasksView.classList.contains('active')) {
        loadTaskQueue();
      }
    });
    var sidebarEl = document.querySelector('.sidebar');
    if (sidebarEl) {
      observer.observe(sidebarEl, { attributes: true, childList: true, subtree: true });
    }

    // 监听 view 切换
    var viewObserver = new MutationObserver(function () {
      var tasksView = $('o-tasks');
      if (tasksView && tasksView.classList.contains('active')) {
        loadTaskQueue();
      }
    });
    var views = document.querySelectorAll('.view');
    views.forEach(function (v) {
      viewObserver.observe(v, { attributes: true, attributeFilter: ['class'] });
    });

    // 收到 SSE 心跳时刷新（如果任务看板可见）
    if (window.sseClient) {
      window.sseClient.on('macro_overview', function () {
        var tasksView = $('o-tasks');
        if (tasksView && tasksView.classList.contains('active')) {
          loadTaskQueue();
        }
      });
    }

    console.log('[tasks] task-dashboard.js loaded');
  }

  if (document.readyState === 'complete') {
    init();
  } else {
    document.addEventListener('DOMContentLoaded', init);
  }

  // 暴露刷新函数供外部调用
  window.loadTaskQueue = loadTaskQueue;
})();
