/* Unified decision workspace: one compact view over the canonical snapshot. */
(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };
  var esc = function (v) { return String(v == null ? '' : v).replace(/[&<>\"]/g, function (c) { return ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]); }); };
  var fmt = function (v, suffix) { return v == null || v === '' ? '-' : esc(v) + (suffix || ''); };
  var state = { mode: 'after_close', catalog: null };

  function statusLabel(value) {
    return ({available:'可用', stale:'陈旧', missing:'缺失', partial:'部分可用'}[String(value)] || '待确认');
  }
  function statusClass(value) {
    return ({available:'good', stale:'warn', missing:'bad', partial:'warn'}[String(value)] || 'neutral');
  }
  function showState(kind, text) {
    var el = $('workbenchState');
    if (!el) return;
    el.className = 'workbench-state ' + kind;
    el.textContent = text;
    el.hidden = false;
  }
  function renderSources(rows) {
    var el = $('workbenchSources');
    if (!el) return;
    el.innerHTML = (rows || []).map(function (row) {
      return '<div class="workbench-source ' + statusClass(row.status) + '"><span><b>' + esc(row.label) + '</b><small>' + esc(statusLabel(row.status)) + '</small></span><em>' + esc(row.as_of || '无数据日') + '</em></div>';
    }).join('') || '<div class="workbench-state empty">暂无来源状态</div>';
  }
  function renderMainlines(rows) {
    var el = $('workbenchMainlines');
    if (!el) return;
    el.innerHTML = (rows || []).map(function (row, index) {
      var flow = row.net_yi == null ? '资金未确认' : ('资金 ' + Number(row.net_yi).toFixed(2) + '亿');
      var level = row.level || row.score_status || '观察';
      return '<div class="workbench-line"><span class="rank">' + (index + 1) + '</span><span class="line-copy"><b>' + esc(row.name || row.concept || '-') + '</b><small>' + esc(level) + ' · 涨停 ' + fmt(row.zt) + ' · 扩散 ' + fmt(row.candidate_count) + ' · ' + esc(flow) + '</small></span><strong>' + fmt(row.score != null ? row.score : row.structure_score) + '</strong></div>';
    }).join('') || '<div class="workbench-state empty">暂无主线排行，系统没有用单一涨停数量替代确认。</div>';
  }
  function renderCandidates(data) {
    var el = $('workbenchCandidates');
    if (!el) return;
    var rows = data.candidates || [];
    var mode = data.candidate_mode === 'short_term' ? '可比候选' : '观察候选';
    var meta = $('workbenchCandidateMeta');
    if (meta) meta.textContent = mode + ' ' + rows.length + ' 只 · 执行候选 ' + fmt(data.counts && data.counts.execution_candidates) + ' 只';
    el.innerHTML = rows.map(function (row, index) {
      var allowed = row.trade_allowed === true || row.trade_label === '可执行';
      var reasons = (row.trade_reasons || row.observation_rank_reasons || []).slice(0, 2).join('；');
      return '<div class="workbench-candidate"><span class="rank">' + (index + 1) + '</span><span class="candidate-copy"><b>' + esc(row.name || row.symbol || '-') + ' <small>' + esc(row.symbol || row.code || '') + '</small></b><span>' + esc(row.decision || reasons || '等待触发条件确认') + '</span></span><span class="workbench-badge ' + (allowed ? 'good' : 'warn') + '">' + (allowed ? '可执行' : '观察') + '</span></div>';
    }).join('') || '<div class="workbench-state empty">当前没有可比候选。缺证据时只显示观察态，不为了凑数输出名单。</div>';
  }
  function openView(view, sub, symbol) {
    if (symbol && window.quickViewIntegrated) {
      var input = $('qaSymbol');
      if (input) input.value = symbol;
      window.quickViewIntegrated(view, sub);
      return;
    }
    if (window.quickViewIntegrated) window.quickViewIntegrated(view, sub);
  }
  function renderActionQueue(data) {
    var el = $('workbenchActionsQueue');
    var meta = $('workbenchActionMeta');
    var summary = $('workbenchActionSummary');
    if (!el) return;
    var market = data.market || {}, counts = data.counts || {};
    var candidates = data.candidates || [];
    var executable = candidates.filter(function (row) { return row.trade_allowed === true || row.trade_label === '可执行'; });
    var degraded = (data.source_status || []).filter(function (row) { return row.status === 'stale' || row.status === 'missing' || row.status === 'partial'; });
    var risks = market.risk_flags || [];
    var executionBlocked = degraded.length > 0 || risks.length > 0;
    var actions = [];
    if (degraded.length) actions.push({ tone: 'warn', title: '先核验数据口径', body: degraded.slice(0, 2).map(function (row) { return row.label + '（' + statusLabel(row.status) + '）'; }).join('、') + ' 影响当前判断。', label: '查看数据健康', view: 'ops', sub: 'o-datahealth' });
    if (risks.length) actions.push({ tone: 'warn', title: '处理风险门控', body: risks.slice(0, 2).join('；'), label: '查看组合风险', view: 'risk', sub: 'r-risk' });
    if (executable.length) {
      var first = executable[0];
      actions.push({ tone: executionBlocked ? 'warn' : 'good', title: executionBlocked ? '核验后再处理候选' : '复核执行候选', body: (first.name || first.symbol || '候选') + '：' + (first.decision || (first.trade_reasons || []).slice(0, 1).join('') || '已满足当前执行条件') + (executionBlocked ? '；当前存在数据或风险门控，不可直接进入纸面执行。' : '。'), label: '打开个股决策台', view: 'stock', sub: 's-overview', symbol: first.symbol || first.code });
    } else if (candidates.length) {
      var observed = candidates[0];
      actions.push({ tone: 'neutral', title: '跟踪观察候选', body: (observed.name || observed.symbol || '候选') + '：' + (observed.decision || (observed.observation_rank_reasons || []).slice(0, 1).join('') || '等待触发条件确认') + '。', label: '查看个股全景', view: 'stock', sub: 's-panorama', symbol: observed.symbol || observed.code });
    } else {
      actions.push({ tone: 'neutral', title: '补充机会池', body: '当前没有可比候选，先运行扫描或检查数据状态。', label: '进入策略扫描', view: 'strategy', sub: 'st-scan' });
    }
    actions.push({ tone: 'neutral', title: '完成盘面归因', body: '将主线、资金与情绪放到同一时间轴复核，避免只依据单一信号。', label: '查看市场归因', view: 'market', sub: 'm-attribution' });
    if (meta) meta.textContent = '可执行 ' + executable.length + ' · 观察 ' + Number(counts.observation_candidates || candidates.length || 0) + ' · 待核验 ' + degraded.length;
    if (summary) {
      summary.className = 'workbench-badge ' + (degraded.length || risks.length ? 'warn' : (executable.length ? 'good' : 'neutral'));
      summary.textContent = degraded.length || risks.length ? '需要复核' : (executable.length ? '存在执行候选' : '观察优先');
    }
    el.innerHTML = actions.slice(0, 4).map(function (action, index) {
      return '<article class="workbench-action ' + action.tone + '"><span class="action-index">' + (index + 1) + '</span><div><b>' + esc(action.title) + '</b><p>' + esc(action.body) + '</p></div><button class="btn-sm workbench-action-btn" type="button" data-view="' + esc(action.view) + '" data-sub="' + esc(action.sub) + '" data-symbol="' + esc(action.symbol || '') + '">' + esc(action.label) + '</button></article>';
    }).join('');
    el.querySelectorAll('.workbench-action-btn').forEach(function (button) {
      button.addEventListener('click', function () { openView(button.dataset.view, button.dataset.sub, button.dataset.symbol); });
    });
  }
  function renderLedger(ledger) {
    var el = $('workbenchLedger');
    if (!el) return;
    if (!ledger) { el.innerHTML = '<span class="muted">执行账本不可用</span>'; return; }
    var d = ledger.delivery || {}, s = ledger.skill_runs || {};
    var latest = d.latest;
    var channelName = ({ feishu: '飞书', wechat: '微信' }[latest && latest.channel] || (latest && latest.channel) || '-');
    var deliveryLine = latest
      ? (latest.ok === true ? '✅ 已投递' : (latest.status === 'blocked' ? '⛔ 阻断未投' : (latest.status === 'failed' ? '❌ 投递失败' : '📤 ' + esc(latest.status)))) + ' · ' + esc(latest.data_as_of || latest.date || '-')
      : '📭 暂无投递回执';
    var deliveryMeta = latest ? (channelName + ' · ' + esc((latest.attempted_at || '').slice(0, 16))) : '';
    el.innerHTML =
      '<div class="ledger-row"><b>投递回执</b><span>' + deliveryLine + '</span><small>' + deliveryMeta + '</small></div>' +
      '<div class="ledger-row"><b>技能执行</b><span>' + (s.count || 0) + ' 次已登记</span><small>' + (s.count ? '完整记录见技能执行账本' : '运行技能分析后自动登记') + '</small></div>' +
      '<div class="ledger-row muted"><b>口径</b><span>每笔投递与技能执行均可追溯</span><small>数据缺失或失败会在此显式呈现，不静默</small></div>';
  }
  async function loadLedger() {
    try {
      var request = window.quantApiFetch || window.fetch;
      var response = await request('/api/execution_ledger', { cache: 'no-store' });
      var data = await response.json();
      renderLedger(data.ok ? data.ledger : null);
    } catch (error) { console.error('[ledger]', error); renderLedger(null); }
  }
  function render(data) {
    if (!data || data.ok === false) { showState('error', (data && data.error) || '统一工作台暂不可用'); return; }
    var market = data.market || {}, counts = data.counts || {};
    showState(data.state === 'degraded' ? 'warn' : 'good', data.state_label || '工作台已更新');
    if (window.updateGlobalDataStatus) window.updateGlobalDataStatus(data.state === 'degraded' ? 'warn' : 'good', data.state === 'degraded' ? '决策状态：部分证据不可用 · 当前仅可研究和观察，不可直接进入纸面执行。' : '决策状态：核心证据链可用 · 仍需逐项核验后进入纸面预检。');
    var meta = $('workbenchMeta');
    if (meta) meta.textContent = '数据截至 ' + (data.as_of || '未知') + ' · 生成 ' + (data.generated_at || '未知') + ' · 统一决策快照';
    var values = {
      workbenchTemperature: fmt(market.temperature, '/100'),
      workbenchForce: fmt(market.force_index, '/100'),
      workbenchEmotion: market.emotion_stage || '待确认',
      workbenchBreadth: market.breadth == null ? '-' : (Number(market.breadth) * 100).toFixed(1) + '%',
      workbenchLimits: fmt(market.limit_up) + ' / ' + fmt(market.limit_down),
      workbenchCandidatesCount: fmt(counts.observation_candidates),
    };
    Object.keys(values).forEach(function (id) { var el = $(id); if (el) el.textContent = values[id]; });
    var risk = $('workbenchRisks');
    if (risk) risk.innerHTML = (market.risk_flags || []).map(function (x) { return '<li>' + esc(x) + '</li>'; }).join('') || '<li>暂无额外风险标记</li>';
    renderSources(data.source_status); renderMainlines(data.mainlines); renderCandidates(data); renderActionQueue(data);
   var confidence = $('workbenchConfidence');
   if (confidence) {
     var parts = data.confidence_components || {};
     confidence.textContent = data.confidence == null ? '暂未形成可信度判断：核心证据尚未完成' : '可信度 ' + (Number(data.confidence) * 100).toFixed(0) + '% · 基础判断：' + (parts.base == null ? '-' : parts.base) + ' · 数据影响：' + (parts.degraded_sources == null ? '无' : parts.degraded_sources) + ' · 情绪参考：' + (parts.social_sentiment == null ? '无' : parts.social_sentiment);
   }
  }
  async function load(mode) {
    state.mode = mode || state.mode;
    var modeEl = $('workbenchMode'); if (modeEl) modeEl.value = state.mode;
    showState('loading', '正在读取统一决策快照…');
    try {
      var request = window.quantApiFetch || window.fetch;
      var response = await request('/api/workbench?mode=' + encodeURIComponent(state.mode), { cache: 'no-store' });
      var data = await response.json();
      if (!response.ok || data.ok === false) throw new Error(data.error || ('HTTP ' + response.status));
      render(data);
    } catch (error) { console.error('[workbench]', error); showState('error', '暂时无法获取决策摘要，请稍后重试'); }
  }
  function renderCatalog(catalog) {
    state.catalog = catalog;
    var el = $('apiCatalogList'); if (!el) return;
    el.innerHTML = (catalog.domains || []).map(function (item) {
      return '<article class="api-catalog-item"><div><b>' + esc(item.label) + '</b><small>' + esc(item.description) + '</small></div><span class="catalog-tier">' + esc(item.primary ? '首页核心能力' : '深入分析能力') + '</span></article>';
    }).join('');
  }
  async function openCatalog() {
    var modal = $('apiCatalogModal'); if (modal) modal.hidden = false;
    var el = $('apiCatalogList'); if (el) el.innerHTML = '<div class="workbench-state loading">正在读取接口目录…</div>';
    try {
      var request = window.quantApiFetch || window.fetch;
      var response = await request('/api/catalog', { cache: 'no-store' });
      var data = await response.json();
      if (!response.ok || data.ok === false) throw new Error(data.error || ('HTTP ' + response.status));
      renderCatalog(data);
    } catch (error) { console.error('[catalog]', error); if (el) el.innerHTML = '<div class="workbench-state error">暂时无法获取数据来源，请稍后重试</div>'; }
  }
  function init() {
    var refresh = $('workbenchRefresh'); if (refresh) refresh.addEventListener('click', function () { load(); });
    var recapRefresh = $('marketRecapRefresh'); if (recapRefresh) recapRefresh.addEventListener('click', function () { if (window.loadMarketRecap) window.loadMarketRecap(true); });
    var mode = $('workbenchMode'); if (mode) mode.addEventListener('change', function () { load(mode.value); });
    var catalog = $('apiCatalogBtn'); if (catalog) catalog.addEventListener('click', openCatalog);
    var close = $('apiCatalogClose'); if (close) close.addEventListener('click', function () { $('apiCatalogModal').hidden = true; });
    var modal = $('apiCatalogModal'); if (modal) modal.addEventListener('click', function (e) { if (e.target === modal) modal.hidden = true; });
    load();
    loadLedger();
  }
  window.Workbench = { load: load, openCatalog: openCatalog };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true }); else init();
})();
