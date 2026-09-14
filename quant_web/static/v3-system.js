// V3 System Status Module — 系统状态面板，整合cron状态、报告、预警历史
// 注入 index.html 的 dashboard section

async function loadSystemStatus() {
  try {
    const res = await fetch('/api/system/status');
    const data = await res.json();
    if (!data.ok) throw new Error(data.error);

    const el = document.getElementById('systemStatus');
    if (!el) return;

    // ── 系统健康度 ──
    let html = '<div style="display:flex;flex-wrap:wrap;gap:16px;margin:12px 0;">';
    html += `<div class="status-card"><div class="sc-title">服务</div><div class="sc-val ${data.server_alive?'':'sc-warn'}">${data.server_alive?'✅ 运行中':'❌ 异常'}</div><div class="sc-sub">${data.started_at ? '启动于 ' + data.started_at : (data.uptime_seconds != null ? '运行 ' + Math.floor(data.uptime_seconds / 3600) + '小时' : '-')}</div></div>`;
    html += `<div class="status-card"><div class="sc-title">数据源</div><div class="sc-val">${data.data_source_status||'未知'}</div><div class="sc-sub">${data.data_freshness||'-'}</div></div>`;
    html += `<div class="status-card"><div class="sc-title">cron任务</div><div class="sc-val">${data.cron_total||0}个</div><div class="sc-sub">启用${data.cron_enabled||0}</div></div>`;
    html += `<div class="status-card"><div class="sc-title">末次日报</div><div class="sc-val">${data.last_report_date||'未生成'}</div><div class="sc-sub">${data.last_report_size||'-'}</div></div>`;
    html += `<div class="status-card"><div class="sc-title">末次预警</div><div class="sc-val">${data.last_alert||'无'}</div><div class="sc-sub">${data.alert_count||0}条历史</div></div>`;
    html += '</div>';

    // ── cron任务列表 ──
    if (data.cron_jobs && data.cron_jobs.length) {
      html += '<h3 style="margin-top:16px;">📋 cron 任务</h3>';
      html += '<div class="tableWrap"><table><thead><tr><th>任务</th><th>周期</th><th>状态</th><th>下次</th></tr></thead><tbody>';
      for (const j of data.cron_jobs) {
        const ok = j.enabled ? '✅' : '⏸️';
        html += `<tr><td>${j.displayName||j.name}</td><td>${j.schedule||'-'}</td><td>${ok}</td><td>${j.nextRun||'-'}</td></tr>`;
      }
      html += '</tbody></table></div>';
    }

    // ── 最近报告 ├─
    if (data.recent_reports && data.recent_reports.length) {
      html += '<h3 style="margin-top:16px;">📝 最近报告</h3>';
      html += '<div class="tableWrap"><table><thead><tr><th>日期</th><th>路径</th></tr></thead><tbody>';
      for (const r of data.recent_reports) {
        html += `<tr><td>${r.date}</td><td style="font-size:12px;color:#9ca3af;">${r.path}</td></tr>`;
      }
      html += '</tbody></table></div>';
    }

    // ── 预警历史 ──
    if (data.alert_history && data.alert_history.length) {
      html += '<h3 style="margin-top:16px;">🔔 预警历史</h3>';
      html += '<div class="tableWrap"><table><thead><tr><th>时间</th><th>内容</th></tr></thead><tbody>';
      for (const a of data.alert_history.slice(-10)) {
        html += `<tr><td>${a.time}</td><td style="font-size:13px;">${a.text}</td></tr>`;
      }
      html += '</tbody></table></div>';
    }

    el.innerHTML = html;
  } catch(e) {
    console.error('loadSystemStatus error:', e);
  }
}

// 自动刷新：每60秒
let _sysTimer = null;
function startSystemPoll() {
  loadSystemStatus();
  if (_sysTimer) clearInterval(_sysTimer);
  _sysTimer = setInterval(loadSystemStatus, 60000);
}
function stopSystemPoll() {
  if (_sysTimer) { clearInterval(_sysTimer); _sysTimer = null; }
}
