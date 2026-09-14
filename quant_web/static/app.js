const state = { market: null, lastScan: [], lastSnapshot: null, lastBacktest: [], klineChart: null, radarChart: null, globalKlineChart: null, watchlist: [], config: null, period: 'daily', intraPeriod: '5', lastKlineSymbol: '002714' };
const COLORS = {
  up:'#b42318', down:'#087443',
  ma20:'#1f7a5c', ma60:'#2563eb', ma144:'#b45309', ma300:'#b42318',
  boll:'#8b5cf6', bollBand:'rgba(139,92,246,0.08)',
  macd:'#2563eb', macdSignal:'#b45309', macdBar:'#6b7280',
  rsi:'#8b5cf6', rsiMid:'#d1d5db', rsiOver:'#fca5a5',
  k:'#2563eb', d:'#b45309', j:'#8b5cf6',
};
const $ = (id) => document.getElementById(id);
const NAV_STATE_KEY = 'quant_web_navigation_state';
function saveNavigationState(view, sub) {
  try { sessionStorage.setItem(NAV_STATE_KEY, JSON.stringify({view, sub, at: Date.now()})); } catch (e) {}
}
function readNavigationState() {
  try { return JSON.parse(sessionStorage.getItem(NAV_STATE_KEY) || '{}'); } catch (e) { return {}; }
}

// ════════════════════════════════════════════════
// 统一导航系统（主Tab + 子导航）
// ════════════════════════════════════════════════

const TAB_LOAD_MAP = {
  'stock': { 's-overview': 'LOAD_STOCK_overview', 's-kline': 'LOAD_STOCK_kline', 's-ml': 'LOAD_STOCK_ml', 's-panorama': 'loadStockLens', 's-intraday': 'LOAD_STOCK_intraday', 's-patterns': 'LOAD_STOCK_patterns', 's-fundamental': 'LOAD_STOCK_fundamental', 's-factors': 'LOAD_STOCK_factors', 's-ai': 'loadStockAiAnalysis' },
  'factors': { 'f-overview': 'loadFactorOverview', 'f-library': 'renderFactorLibrary', 'f-scan': 'loadFactorScan', 'f-ic': 'loadFactorIC', 'f-combine': 'loadFactorCombine' },
  'strategy': { 'st-scan': 'LOAD_STRATEGY_scan', 'st-backtest': 'runBacktest', 'st-optimize': 'runOptimize', 'st-dashboard': 'loadBacktestDashboard', 'st-compare': 'loadStrategyCompare', 'st-skills': 'loadSkills', 'st-event-backtest': 'runEventBacktest' },
  'market': { 'm-sector': 'LOAD_MARKET_sector', 'm-flow': 'loadFlowWorkbench', 'm-temp': 'loadMarketTemp', 'm-regime': 'loadMarketRegime', 'm-margin': 'loadMarginSummary', 'm-north': 'loadNorthboundFlow', 'm-sentiment': 'loadNorthFlow', 'm-global': 'loadGlobalQuotes', 'm-calendar': 'LOAD_MARKET_calendar', 'm-attribution': 'LOAD_MARKET_attribution', 'm-etf': 'loadEtfFlow', 'm-history': 'loadMarketHistoryCatalog' },
  'risk': { 'r-portfolio': 'loadPortfolio', 'r-risk': 'loadPortfolioRisk', 'r-factor': 'loadFactorAnalysis', 'r-financial': 'loadFinancialData', 'r-fama': 'loadFamaMacBeth', 'r-opportunity': 'LOAD_RISK_opportunity', 'r-news': 'loadNewsSentiment', 'r-optimizer': 'LOAD_RISK_optimizer', 'r-allocation': 'loadAssetAllocation' },
  'ops': { 'o-realtime': 'loadRealtime', 'o-reports': 'loadReports', 'o-ml-monitor': 'LOAD_OPS_mlmonitor', 'o-drift': 'loadDriftCheck', 'o-cache': 'loadCacheStatus', 'o-databass': 'loadDataBase', 'o-featurestore': 'loadFeatureStore', 'o-tasks': 'loadTaskQueue', 'o-alerts': 'loadAlertsStatus', 'o-datahealth': 'loadDataHealth', 'o-cron': 'loadCronTasks', 'o-chain': 'loadDecisionChain', 'o-chat': 'focusChatInput' },
  'slip': { 'slip-compare': 'loadSlipCompare' },
};

// 主Tab切换
// 共享代码输入同步
if ($('sSymbol')) {
  $('sSymbol').addEventListener('input', function() {
    const val = this.value.trim();
    state.lastKlineSymbol = val || state.lastKlineSymbol;
    ['chartSymbol','svSymbol','fundSymbol','patternSymbol','earningsSymbol','intraSymbol','factorSymbol','calendarEarningsSymbol','lensCode','newsSymbol'].forEach(id => {
      const el = $(id);
      if (el) el.value = val;
    });
  });
}

document.querySelectorAll('.nav').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav').forEach(n => n.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    btn.classList.add('active');
    const viewId = btn.dataset.view;
    $(viewId).classList.add('active');
    saveNavigationState(viewId, viewId === 'dashboard' ? '' : ($(viewId)?.querySelector('.sub-nav-btn.active')?.dataset.sub || ''));
    // 每次回到总览都强制取一轮最新行情，避免首页显示旧交易日。
    if (viewId === 'dashboard') loadMacroOverview(false);
    // 默认激活第一个sub-view（非dashboard）
    if (viewId !== 'dashboard') {
      const firstSub = $(viewId)?.querySelector('.sub-nav-btn');
      if (firstSub) window.switchSubView(viewId, firstSub.dataset.sub);
    }
    // ECharts resize
    setTimeout(() => {
      if (state.klineChart) state.klineChart.resize();
      if (state.radarChart) state.radarChart.resize();
    }, 150);
    // 停掉实时监控
    if (viewId !== 'ops') stopMonitor();
  });
});

// 侧边栏大类展开/收起
function bindNavGroups() {
  document.querySelectorAll('.nav-group-title').forEach(title => {
    title.addEventListener('click', () => {
      const gid = title.dataset.group;
      const items = $(`grp-${gid}`);
      if (!items) return;
      const open = items.classList.toggle('open');
      const arrow = title.querySelector('.ng-arrow');
      if (arrow) arrow.textContent = open ? '▾' : '▸';
      title.classList.toggle('open', open);
    });
  });
  // 小类按钮：切换 view + sub
  document.querySelectorAll('.nav-sub').forEach(btn => {
    btn.addEventListener('click', () => {
      const view = btn.dataset.view, sub = btn.dataset.sub;
      if (!view) return;
      // 切 view
      document.querySelectorAll('.nav').forEach(n => n.classList.remove('active'));
      document.querySelectorAll('.nav-sub').forEach(n => n.classList.remove('active'));
      document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
      const viewEl = $(view);
      if (!viewEl) return;
      viewEl.classList.add('active');
      btn.classList.add('active');
      // 切 sub
      if (sub) {
        window.switchSubView(view, sub);
      } else {
        const firstSub = viewEl.querySelector('.sub-nav-btn');
        if (firstSub) window.switchSubView(view, firstSub.dataset.sub);
      }
      if (view !== 'ops') stopMonitor();
    });
  });
}
bindNavGroups();

// 子导航切换
// 主工作台只维护一套子导航绑定；统一入口不能以收敛为理由让历史
// 深度功能失去点击能力。
function bindSubNavigation() {
  document.querySelectorAll('.sub-nav-btn').forEach(btn => {
    if (btn.dataset.bound === '1') return;
    btn.dataset.bound = '1';
    btn.addEventListener('click', () => {
      const parent = btn.closest('.view');
      if (parent) window.switchSubView(parent.id, btn.dataset.sub);
    });
  });
}

window.switchSubView = function(parentView, subId) {
  const container = $(parentView);
  saveNavigationState(parentView, subId);
  if (!container) return;
  container.querySelectorAll('.sub-nav-btn').forEach(b => b.classList.remove('active'));
  container.querySelectorAll('.sub-view').forEach(v => v.classList.remove('active'));
  const btn = container.querySelector(`.sub-nav-btn[data-sub="${subId}"]`);
  const view = $(subId);
  if (btn) btn.classList.add('active');
  if (view) view.classList.add('active');
  // 加载数据
  const loadMap = TAB_LOAD_MAP[parentView];
  if (loadMap && loadMap[subId]) {
    const fn = loadMap[subId];
    if (fn.startsWith('LOAD_')) {
      window[fn] && window[fn]();
    } else if (typeof window[fn] === 'function') {
      window[fn]();
    }
  }
  // ECharts resize
  setTimeout(() => { container.querySelectorAll('.sub-view.active canvas').forEach(() => {}); }, 150);
};
bindSubNavigation();

// 快捷跳转（从总览页跳转到指定 tab+sub）
window.quickViewIntegrated = function(parentView, subId) {
  if (!$(parentView)) return;
  // 共享股票代码
  const sym = ($('qaSymbol')?.value || '').trim();
  if (sym) {
    state.lastKlineSymbol = sym;
    if ($('sSymbol')) {
      $('sSymbol').value = sym;
      $('sSymbol').dispatchEvent(new Event('input', { bubbles: true }));
    }
    if ($('chartStart')) { /* keep default */ }
  }
  // 切换到tab
  const navBtn = document.querySelector(`.nav[data-view="${parentView}"]`);
  if (navBtn) navBtn.click();
  else {
    // 直接切view
    document.querySelectorAll('.nav').forEach(n => n.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    $(parentView).classList.add('active');
  }
  // 切sub
  setTimeout(() => window.switchSubView(parentView, subId), 50);
};

// 旧版 switchView 保持兼容
window.switchView = function(view) {
  const btn = document.querySelector(`.nav[data-view="${view}"]`);
  if (btn) { btn.click(); return; }
  const target = $(view);
  if (!target) return;
  document.querySelectorAll('.nav, .nav-sub').forEach(node => node.classList.remove('active'));
  document.querySelectorAll('.view').forEach(node => node.classList.remove('active'));
  target.classList.add('active');
  const firstSub = target.querySelector('.sub-nav-btn');
  if (firstSub) window.switchSubView(view, firstSub.dataset.sub);
};

function quickView(view) {
  const symbol = ($('qaSymbol')?.value || '').trim();
  if (symbol) {
    state.lastKlineSymbol = symbol;
    if ($('sSymbol')) {
      $('sSymbol').value = symbol;
      $('sSymbol').dispatchEvent(new Event('input', { bubbles: true }));
    }
  }
  const btn = document.querySelector(`[data-view="${view}"]`);
  if (btn) btn.click();
}
window.quickView = quickView;
window.openPaperDraft = function () { const symbol = ($('qaSymbol')?.value || sharedSymbol?.() || '').trim(); window.open('/paper_trading.html' + (symbol ? '?symbol=' + encodeURIComponent(symbol) : ''), '_blank', 'noopener'); };

window.toggleSidebar = function(id) {
  const el=$(id), arrow=$('spArrow');
  if (!el) return;
  el.classList.toggle('open');
  arrow.textContent = el.classList.contains('open') ? '▾' : '▸';
};

// ── 增强版 API 调用（自动 Loading + 超时 + 重试 + 内存缓存） ──
// 缓存策略：GET 默认缓存 45s，解决「用一次调用一次」——切换视图/重复点按钮不重复请求。
// 强制刷新：传 {refresh:true} 或 {cache_ttl:0} 跳过缓存。
// 慢端点按需放大超时，显式 options.timeout 优先。
const API_TIMEOUT_MAP = {
  '/api/factor_scan': 150000,     // 实测首调 ~97s
  '/api/opportunity': 240000,     // 实测首调 >180s
  '/api/v1/audit': 60000,         // 实测 ~31s
  '/api/v11/battle': 60000,       // 实测 ~34s
  '/api/portfolio_risk': 60000,   // 实测 ~14s（留余量）
  '/api/v1/health': 60000,        // 实测 ~40s
  '/api/asset_allocation': 90000,
  '/api/fama_macbeth': 180000,
  '/api/v8_backtest': 120000,
  '/api/backtest_almgren': 120000,
  '/api/ai_analyze': 300000,
};
const _apiCache = new Map();      // key -> {t, data}
const _charts = new Map();        // DOM id -> ECharts instance (single owner)
function chartFor(id) {
  const el = $(id);
  if (!el || !window.echarts) return null;
  let chart = _charts.get(id);
  if (!chart || chart.isDisposed()) { chart = window.echarts.init(el); _charts.set(id, chart); }
  return chart;
}
const _apiInflight = new Map();   // key -> Promise（并发去重，同一URL只发一个请求）
const CACHE_DEFAULT_TTL = 45;
window.clearApiCache = function () { _apiCache.clear(); };
async function api(url, body, options = {}) {
  const baseUrl = String(url).split('?')[0];
  const { retries = 0, silent = false, refresh = false, cache_ttl } = options;
  const timeout = options.timeout || API_TIMEOUT_MAP[baseUrl] || 30000;
  const isGet = body === undefined;
  const ttl = cache_ttl !== undefined ? cache_ttl : (isGet ? CACHE_DEFAULT_TTL : 0);
  const key = url + (isGet ? '' : '|' + JSON.stringify(body));
  // 命中缓存
  if (isGet && ttl > 0 && !refresh) {
    const hit = _apiCache.get(key);
    if (hit && (Date.now() - hit.t) < ttl * 1000) return hit.data;
  }
  // 并发去重：同一 GET 同时只发一个请求
  if (isGet && !refresh) {
    const inflight = _apiInflight.get(key);
    if (inflight) { try { return await inflight; } catch (e) { /* 失败落空则重发 */ } }
  }
  let lastErr;
  const run = async () => {
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        const init = isGet
          ? { signal: controller.signal }
          : { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body), signal: controller.signal };
        const res = await fetch(url, init);
        clearTimeout(timeoutId);
        const data = await res.json();
        if (!res.ok || data.ok===false) throw new Error(data.error || `HTTP ${res.status}`);
        // 深度接口统一处理异步响应：后台 computing 时自动轮询，不把半成品交给渲染器。
        const poll = Number(options._poll || 0);
        if (data.status === 'computing' && isGet && poll < 60) {
          setStatus('⏳ 深度计算中，请稍候…');
          await new Promise(r => setTimeout(r, 1000));
          return api(url, body, Object.assign({}, options, { refresh:true, cache_ttl:0, _poll:poll+1 }));
        }
        if (isGet && ttl > 0 && data.status !== 'computing') _apiCache.set(key, { t: Date.now(), data });
        return data;
      } catch(e) {
        lastErr = e;
        if (attempt < retries) {
          await new Promise(r => setTimeout(r, 1000 * (attempt + 1)));
        }
      }
    }
    const msg = lastErr.name === 'AbortError' ? `请求超时 (${timeout}ms): ${url}` : `请求失败: ${lastErr?.message}`;
    if (!silent) setStatus('⚠️ ' + msg);
    throw new Error(msg);
  };
  const p = run();
  if (isGet && !refresh) _apiInflight.set(key, p);
  try {
    return await p;
  } finally {
    if (isGet && !refresh) setTimeout(() => _apiInflight.delete(key), 50);
  }
}

function readParams() {
  const r = (id) => { const e=$(id); return e ? parseFloat(e.value)||0 : 0; };
  return {
    strategy: {
      fast_ma: Math.round(r('sp_fast_ma')||20), slow_ma: Math.round(r('sp_slow_ma')||60),
      trend_ma: Math.round(r('sp_trend_ma')||144), long_trend_ma: Math.round(r('sp_long_trend_ma')||300),
      volume_ma: Math.round(r('sp_volume_ma')||20),
      stop_loss_pct: (r('sp_stop_loss_pct')||8)/100, take_profit_pct: (r('sp_take_profit_pct')||24)/100,
      trail_stop_pct: (r('sp_trail_stop_pct')||12)/100, max_risk_score_for_new_buy: Math.round(r('sp_max_risk_score')||5),
    },
    portfolio: {
      initial_cash: (r('sp_initial_cash')||100)*10000, max_position_pct: (r('sp_max_position_pct')||20)/100,
      risk_per_trade_pct: (r('sp_risk_per_trade_pct')||1)/100, commission_pct: (r('sp_commission_pct')||0.03)/100,
      slippage_pct: (r('sp_slippage_pct')||0.1)/100,
    },
  };
}

async function loadConfig() {
  try {
    const data = await api('/api/config'); state.config = data;
    const s=data.strategy||{}, p=data.portfolio||{};
    const set=(id,val)=>{const e=$(id);if(e&&val!=null)e.value=val;};
    set('sp_fast_ma',s.fast_ma); set('sp_slow_ma',s.slow_ma); set('sp_trend_ma',s.trend_ma); set('sp_long_trend_ma',s.long_trend_ma); set('sp_volume_ma',s.volume_ma);
    set('sp_stop_loss_pct',(s.stop_loss_pct||0.08)*100); set('sp_take_profit_pct',(s.take_profit_pct||0.24)*100); set('sp_trail_stop_pct',(s.trail_stop_pct||0.12)*100); set('sp_max_risk_score',s.max_risk_score_for_new_buy);
    set('sp_initial_cash',(p.initial_cash||1000000)/10000); set('sp_max_position_pct',(p.max_position_pct||0.2)*100); set('sp_risk_per_trade_pct',(p.risk_per_trade_pct||0.01)*100);
    set('sp_commission_pct',(p.commission_pct||0.0003)*100); set('sp_slippage_pct',(p.slippage_pct||0.001)*100);
  } catch(e) { console.warn('Config load failed',e); }
}

async function loadWatchlist() {
  try { const data=await api('/api/watchlist'); state.watchlist=data.symbols||[]; renderWatchlist(); } catch(e) { console.warn('Watchlist failed',e); }
}

function renderWatchlist() {
  const el=$('watchlistDisplay'); if(!el)return;
  if(!state.watchlist.length) { el.innerHTML='<span class="empty">暂无看板。在右侧保存即可添加。</span>'; return; }
  el.innerHTML = state.watchlist.map(s=>`<span class="wl-tag">${esc(s)}<span class="wl-del" data-symbol="${esc(s)}">\u00D7</span></span>`).join('');
}

function saveWatchlist() {
  setStatus('保存看板...');
  api('/api/watchlist',{symbols:state.watchlist}).then(d=>{setStatus(`看板已保存 (${d.count}只)`);}).catch(e=>setStatus('保存失败:'+e.message));
}

$('scanBtn').addEventListener('click', runScan);
$('segRefreshBtn').addEventListener('click', loadSegments);
$('loadChartBtn').addEventListener('click', loadChart);
$('backtestBtn').addEventListener('click', runBacktest);
$('btPortfolioBtn').addEventListener('click', runPortfolioBacktest);
$('askBtn').addEventListener('click', askOpenClaw);
$('alertFeishuBtn').addEventListener('click', ()=>sendAlert('scan',state.lastScan));
$('btAlertFeishuBtn').addEventListener('click', ()=>sendAlert('backtest',state.lastBacktest));
$('wlImportBtn').addEventListener('click',()=>{
  const codes=$('scanSymbols').value.trim();
  if(codes) codes.split(/[,;\s]+/).filter(Boolean).forEach(c=>{const n=c.toUpperCase().slice(0,6);if(!state.watchlist.includes(n))state.watchlist.push(n);});
  renderWatchlist();
});
$('wlAddBtn').addEventListener('click',()=>{
  const input=$('wlInput');
  input.value.split(/[,;\s]+/).filter(Boolean).forEach(c=>{const n=c.toUpperCase().slice(0,6);if(n&&!state.watchlist.includes(n))state.watchlist.push(n);});
  input.value=''; renderWatchlist();
});
$('wlSaveBtn').addEventListener('click', saveWatchlist);
$('optimizeBtn').addEventListener('click', runOptimize);
$('sectorLoadBtn').addEventListener('click', loadSectorView);
$('sectorRefreshBtn').addEventListener('click', ()=>{setStatus('刷新板块数据...');api('/api/sector?refresh=true').then(d=>{setStatus('板块数据已刷新');loadSectorView();}).catch(e=>setStatus('刷新失败:'+e.message));});
$('scanCsvBtn').addEventListener('click', ()=>exportCsv('scan', state.lastScan));
$('btCsvBtn').addEventListener('click', ()=>exportCsv('backtest', state.lastBacktest));
$('rtRefreshBtn').addEventListener('click', loadRealtime);
$('indexTickerRefresh')?.addEventListener('click', () => loadIndexTicker(true));
$('rtStartMonitorBtn').addEventListener('click', toggleMonitor);
$('loadIntradayBtn').addEventListener('click', loadIntradayChart);
$('globalRefreshBtn').addEventListener('click', loadGlobalQuotes);
$('marginRefreshBtn').addEventListener('click', loadMarginSummary);
$('marginQueryBtn').addEventListener('click', loadMarginDetail);
$('mlSignalBtn')?.addEventListener('click', () => loadMLSignal('predict'));
$('mlSignalScanBtn')?.addEventListener('click', () => loadMLSignal('scan'));
$('mlImportanceBtn')?.addEventListener('click', loadMLSignalImportance);
$('btDashboardRefresh')?.addEventListener('click', loadBacktestDashboard);
$('newsRefreshBtn')?.addEventListener('click', loadNewsSentiment);
$('srRefreshBtn')?.addEventListener('click', loadSectorRotation);
$('allDayIndustryBtn')?.addEventListener('click', loadAllDayIndustry);
$('flowLoadBtn')?.addEventListener('click', loadFlowWorkbench);
$('flowLatestBtn')?.addEventListener('click', () => { if ($('flowDate')) $('flowDate').value = ''; loadFlowWorkbench(true); });
$('flowMode')?.addEventListener('change', () => loadFlowWorkbench());
$('fundRefreshBtn')?.addEventListener('click', loadFundamentals);
$('mtRefreshBtn')?.addEventListener('click', loadMarketTemp);
$('regimeRefreshBtn')?.addEventListener('click', loadMarketRegime);
$('oppRefreshBtn')?.addEventListener('click', loadOpportunities);
$('factorModelRefreshBtn')?.addEventListener('click', loadFactorAnalysis);
$('stCompareRefreshBtn')?.addEventListener('click', loadStrategyCompare);
$('patternLoadBtn')?.addEventListener('click', loadPatterns);
$('macroCalRefreshBtn')?.addEventListener('click', loadMacroCal);
$('earningsLoadBtn')?.addEventListener('click', loadEarnings);
document.addEventListener('click',(e)=>{if(e.target.classList.contains('wl-del')){state.watchlist=state.watchlist.filter(s=>s!==e.target.dataset.symbol);renderWatchlist();}});
document.addEventListener('dblclick',(e)=>{if(e.target.classList.contains('wl-tag')){const existing=$('scanSymbols').value.trim(),sym=e.target.textContent.replace('\u00D7','').trim();if(!existing.includes(sym))$('scanSymbols').value=existing?existing+' '+sym:sym;}});

// 周期切换（日/周/月）
document.querySelectorAll('.period-btn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.period-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    state.period=btn.dataset.period;
  });
});
// 日内周期切换（1/5/15/30/60分钟）
document.querySelectorAll('.intra-period-btn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.intra-period-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    state.intraPeriod=btn.dataset.period;
  });
});

// ── 全局视图切换函数（给快速入口按钮用） ──


// ── Quant Terminal Core：统一编排所有工作台能力 ──
// 旧模块只提供能力，不再各自拥有首屏初始化或同一 DOM 的写权限。
window.QUANT_TERMINAL_CORE = window.QUANT_TERMINAL_CORE || {
  version: 'integrated',
  state,
  modules: { market: true, portfolio: true, risk: true, factors: true, tasks: true, reports: true },
};

// ── 初始化：先恢复导航，再分阶段加载，避免刷新时同时触发全部慢接口 ──
function restoreNavigationState() {
  const saved = readNavigationState();
  const view = $(saved.view) ? saved.view : 'dashboard';
  const nav = document.querySelector(`.nav[data-view="${view}"]`);
  if (nav && view !== 'dashboard') nav.click();
  else {
    document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === view));
    document.querySelectorAll('.nav').forEach(n => n.classList.toggle('active', n.dataset.view === view));
    if (view === 'dashboard') saveNavigationState('dashboard', '');
  }
  if (view !== 'dashboard' && saved.sub && $(view)?.querySelector(`.sub-nav-btn[data-sub="${saved.sub}"]`)) {
    window.switchSubView(view, saved.sub);
  }
}

function updateGlobalDataStatus(kind, text) { const el = $('globalDataStatus'); if (!el) return; el.className = 'global-data-status ' + kind; el.textContent = text; }
window.updateGlobalDataStatus = updateGlobalDataStatus;
async function loadIndexTicker(force = false) {
  const host = $('indexTicker');
  if (!host) return;
  try {
    const data = await api('/api/indices?global=false' + (force ? '&refresh=true' : ''), undefined, { refresh: force, cache_ttl: force ? 0 : 45, silent: true, timeout: 8000 });
    const indices = data.data || data;
    const rows = Object.entries(indices.ashare || {}).filter(([, item]) => item && item.index != null);
    const labels = ['上证指数', '深证成指', '沪深300', '创业板指', '科创50'];
    const ordered = labels.map(name => [name, indices.ashare?.[name]]).filter(([, item]) => item && item.index != null);
    const extra = rows.filter(([name]) => !labels.includes(name));
    const display = ordered.concat(extra).slice(0, 7);
    if (!display.length) throw new Error('指数接口未返回有效行情');
    host.innerHTML = display.map(([name, item]) => { const change = Number(item.change_pct ?? item.change); const tone = Number.isFinite(change) ? (change > 0 ? 'up' : change < 0 ? 'down' : '') : ''; return `<article class="index-ticker ${tone}"><b>${esc(name)}</b><strong>${Number(item.index).toFixed(2)}</strong><span>${Number.isFinite(change) ? (change >= 0 ? '+' : '') + change.toFixed(2) + '%' : '涨跌缺失'}</span><small>${esc(item.source || '行情源')} · ${esc(item.time || item.date || '实时')}</small></article>`; }).join('');
    const meta = $('indexTickerMeta'); if (meta) meta.textContent = '独立 /api/indices · ' + new Date().toLocaleTimeString();
    updateGlobalDataStatus('good', '行情状态：A股指数已更新 · 全球指数正在补充 · 可继续进行研究核验；纸面订单仍需通过风险门控。');
    loadGlobalIndexTicker();
  } catch (error) {
    host.innerHTML = `<div class="data-state error">指数行情暂不可用：暂不能形成最新指数判断。请稍后重试。</div>`;
    updateGlobalDataStatus('warn', '行情状态：指数数据暂不可用 · 页面其他历史/研究模块仍可查看，但不应据此判断当前盘面。');
  }
}

async function loadGlobalIndexTicker() {
  const host = $('globalIndexTicker'); if (!host) return;
  try {
    const response = await api('/api/indices?global=true', undefined, { cache_ttl: 60, timeout: 15000, silent: true });
    const data = response.data || response, rows = Object.entries(data.global || {});
    if (!rows.length) { host.innerHTML = '<span class="data-meta">全球指数暂无有效行情</span>'; return; }
    const aliases = {'恒生指数':'港股大盘','恒生科技':'港股科技','日经225':'日本大盘','TOPIX':'日本综合','韩国综合':'韩国大盘','纳斯达克':'美股科技','纳指100':'美股科技100','标普500':'美股大盘'};
    const order = ['恒生指数','恒生科技','日经225','TOPIX','韩国综合','纳斯达克','纳指100','标普500'];
    rows.sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0]));
    host.innerHTML = rows.map(([name, item]) => { const pct = Number(item.change_pct ?? item.change); const tone = pct > 0 ? 'up' : pct < 0 ? 'down' : ''; return `<span class="global-index ${tone}"><b>${esc(aliases[name] || name)}</b><span>${item.index == null ? '-' : Number(item.index).toFixed(2)} · ${Number.isFinite(pct) ? (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%' : '涨跌缺失'}</span><small>${esc(item.source || '行情源')}</small></span>`; }).join('');
  } catch (error) { host.innerHTML = '<span class="data-meta">全球指数暂不可用，不影响 A 股行情。</span>'; }
}

async function loadMarket() {
  setStatus('优先加载指数与市场宽度...');
  // 首屏关键指标先完成，盘后复盘在后台继续，不阻塞用户看到行情。
  const coreResults = await Promise.allSettled([loadMacroOverview(false), loadIndexTicker(false), loadDashboardBreadth()]);
  const rejected = coreResults.filter(result => result.status === 'rejected').length;
  const partial = coreResults.reduce((sum, result) => sum + (result.status === 'fulfilled' ? Number(result.value?.failed || 0) : 0), 0);
  if (rejected || partial) updateGlobalDataStatus('warn', `行情状态：${rejected + partial} 项核心数据不可用 · 当前页面保留有效数据，但不应将其视为完整实时盘面。`);
  setStatus(rejected || partial ? `核心行情部分不可用（${rejected + partial}项）` : '核心行情已更新，正在补充盘后复盘...');
  loadMarketRecap(false).then(recap => {
    const recapAvailable = recap && recap.ok !== false && recap.data_status !== 'error' && recap.data_status !== 'missing';
    if (recapAvailable) renderMarketRecap({ ...recap, indices: state.market?.indices?.data?.ashare || state.market?.indices?.ashare || {} });
    else renderMarketRecap({ ok: false, error: '盘后复盘暂不可用，请点击刷新重试。' });
  }).catch(() => renderMarketRecap({ ok: false, error: '盘后复盘暂不可用，请点击刷新重试。' }));
}

function recapNum(value, suffix = '') {
  return value == null || value === '' ? '-' : `${value}${suffix}`;
}

function renderMarketRecapCandidates(data) {
  const host = $('marketRecapCandidates');
  if (!host) return;
  const rows = (Array.isArray(data?.candidates) ? data.candidates : [])
    .filter(row => row && row.short_term_score != null && row.score_breakdown?._meta?.status !== 'not_comparable')
    .slice(0, 5);
  const meta = $('marketRecapCandidateMeta');
  if (meta) meta.textContent = rows.length ? `完整证据链 ${rows.length} 只 · 评分统一 0–100` : '没有满足同日主线、行业资金和风险门控的可比候选';
  if (!rows.length) {
    host.innerHTML = '<div class="empty">当前没有可比候选。缺少同日资金或结构证据时，系统只保留观察，不为了凑数输出名单。</div>';
    return;
  }
  const list = (value) => Array.isArray(value) ? value.filter(Boolean).map(esc).join('；') : esc(value || '暂无');
  const valueText = value => value == null || value === '' ? '缺失' : esc(value);
  const evidenceRow = (label, value, note = '') => `<div class="candidate-evidence-row"><b>${esc(label)}</b><span>${value}</span>${note ? `<small>${esc(note)}</small>` : ''}</div>`;
  host.innerHTML = rows.map((row, index) => {
    const flow = row.flow_match || {};
    const stockFlow = row.stock_fund_flow || {};
    const concepts = row.concept_context || row.concepts || [];
    const reasons = row.short_term_reasons || row.observation_rank_reasons || row.trade_reasons || [];
    const breakdown = Object.entries(row.score_breakdown || {}).filter(([key, value]) => key !== '_meta' && value && typeof value === 'object')
      .map(([key, value]) => `${esc(key)} ${esc(value.points ?? 0)}/${esc(value.max ?? 0)}（${esc(value.status || '未说明')}）`).join(' · ') || '暂无可核验拆解';
    const invalidators = row.invalidators || row.stop_conditions || row.execution_strategy?.stop || '行业资金转负、跌破关键价位或主线退潮';
    const research = row.research_evidence || {};
    const market = data.market || {};
    const marketForce = row.market_force ?? market.force_index;
    const marketBreadth = row.breadth ?? market.breadth;
    const researchLevel = research.evidence_level || (Number(research.reports || 0) > 0 ? 'A' : 'C');
    const researchNote = researchLevel === 'A' ? `A级代码精确命中，研报${research.reports || 0}篇` : researchLevel === 'B' ? `B级公司名/简称命中，研报${research.reports || 0}篇（仅辅助，不加分）` : 'C级仅主题/行业背景，未归因到该股（不加分）';
    const researchDetail = Number(research.reports || 0) > 0
      ? [research.rating && `评级：${research.rating}`, research.target_price != null && `目标价：${research.target_price}`, (research.top_concepts || []).length && `主题：${research.top_concepts.join('、')}`, (research.top_catalysts || []).length && `催化：${research.top_catalysts.join('、')}`, (research.titles || []).length && `资料：${research.titles.join('；')}`].filter(Boolean).join('；')
      : '暂无代码级研报记录；不使用行业研报数量、公司名模糊命中或全库主题替代个股证据。';
    const evidence = [
      evidenceRow('价格强度', `${valueText(row.change_pct)}%`, '当日涨幅；未封板时属于强势但仍需观察承接'),
      evidenceRow('成交与换手', `成交额 ${valueText(row.amount_yi)}亿 · 换手 ${valueText(row.turnover)}%`, '换手用于判断活跃度与拥挤度'),
      evidenceRow('技术动量', `${valueText(row.composite_conf)} · 20日 ${row.momentum_20d == null ? '缺失' : `${(Number(row.momentum_20d) * 100).toFixed(1)}%`} · 20日量比 ${valueText(row.volume_ratio_20d)}`, '技术动量/量能放大是入选依据，不等于趋势确认'),
      evidenceRow('主线与题材', `${valueText(row.mainline_match)} · ${list(concepts.slice(0, 8))}`, `结构日 ${row.as_of || '当日'}；题材仅作为结构证据`),
      evidenceRow('行业资金', `${valueText(flow.net_yi)}亿 · ${flow.as_of || row.as_of || '日期缺失'}`, '行业资金代理，不等于个股主力净流入'),
      evidenceRow('个股主力', stockFlow.main_net_yi == null ? '缺失·不计分' : `${Number(stockFlow.main_net_yi).toFixed(3)}亿`, stockFlow.as_of || '没有对应当日记录，不代表净流出'),
      evidenceRow('研报与催化', researchNote, `${research.attribution || '未形成股票级归因'}；${researchDetail || '缺失·不计分'}`),
      evidenceRow('市场环境', `资金合力 ${valueText(marketForce)} · 宽度 ${marketBreadth == null ? '缺失' : `${(Number(marketBreadth) * 100).toFixed(1)}%`}`, '环境偏弱会限制执行，不否定个股当日表现'),
      evidenceRow('龙虎榜', row.lhb_feature ? `近60日${valueText(row.lhb_feature.days)}次 · 净买${valueText(row.lhb_feature.positive_days)}次` : '无记录，未加分', '历史席位证据不能替代当日资金'),
    ].join('');
    return `<article class="candidate-card" data-candidate-code="${esc(row.code || row.symbol || '')}">
      <div class="candidate-card-head"><div><span class="candidate-rank">${index + 1}</span><strong>${esc(row.name || row.code || '-')}</strong><span class="candidate-code">${esc(row.code || row.symbol || '')}</span></div><span class="cell-value cell-watch">短线分 ${esc(row.short_term_score)}/100</span></div>
      <p class="candidate-summary">${esc(reasons[0] || row.summary || row.decision || '进入观察范围，但仍需盘中触发')}</p>
      <div class="candidate-reasons"><div><b>为什么入选</b><span>${list(reasons)}</span></div><div><b>资金来自哪里</b><span>行业 ${flow.net_yi == null ? '缺失' : `${Number(flow.net_yi) > 0 ? '+' : ''}${Number(flow.net_yi).toFixed(2)}亿`} · ${esc(flow.as_of || row.as_of || '日期缺失')} · ${esc(flow.status || '同日行业资金代理')}；个股主力 ${stockFlow.main_net_yi == null ? '缺失' : `${Number(stockFlow.main_net_yi) > 0 ? '+' : ''}${Number(stockFlow.main_net_yi).toFixed(3)}亿`} · ${esc(stockFlow.as_of || row.as_of || '日期缺失')}</span></div><div><b>何时失效</b><span>${list(invalidators)}</span></div></div>
      <details open><summary>展开证据明细与评分解释</summary><div class="candidate-detail"><div class="candidate-evidence-grid">${evidence}</div><p class="candidate-data-note">研报状态：${esc(researchNote)}。没有数据只表示本轮未验证，不把空值当作利空；但未验证证据不会增加短线分。</p><p><b>评分拆解：</b>${breakdown}</p><small>阻断/警告：${list(row.short_term_blockers || row.trade_reasons)}</small></div></details>
    </article>`;
  }).join('');
}

function renderMarketRecapFlow(data) {
  const host = $('marketRecapFlowChart');
  if (!host || !window.echarts) return;
  const rows = [...(data?.sector_rotation?.top_up || []), ...(data?.money_flow?.top_in || [])]
    .filter(row => row && Number.isFinite(Number(row.net_yi ?? row.main_net_yi ?? row.net)))
    .slice(0, 10);
  if (!rows.length) { host.innerHTML = '<div class="empty">暂无可绘制的同日资金数据。</div>'; return; }
  let marketRecapFlowChart = echarts.getInstanceByDom ? echarts.getInstanceByDom(host) : null;
  if (!marketRecapFlowChart) marketRecapFlowChart = echarts.init(host, null, { renderer: 'canvas' });
  if (typeof marketRecapFlowChart.setOption !== 'function') { host.innerHTML = '<div class="data-state error">资金图表实例初始化失败，请刷新页面。</div>'; return; }
  const values = rows.map(row => Number(row.net_yi ?? row.main_net_yi ?? row.net));
  const names = rows.map(row => row.name || row.industry || row.sector || row.concept_name || '-');
  marketRecapFlowChart.setOption({tooltip:{trigger:'axis',axisPointer:{type:'shadow'}},grid:{left:100,right:18,top:12,bottom:28},xAxis:{type:'value',axisLabel:{color:'#8b949e'}},yAxis:{type:'category',data:names.slice().reverse(),axisLabel:{color:'#8b949e',width:90,overflow:'truncate'}},series:[{type:'bar',data:values.slice().reverse().map(value=>({value,itemStyle:{color:value >= 0 ? '#f6465d' : '#2ebd85'}}))}]}, true);
  marketRecapFlowChart.off('click');
  marketRecapFlowChart.on('click', params => { const row = rows.slice().reverse()[params.dataIndex] || {}; const detail = $('marketRecapFlowDetail'); if (detail) detail.innerHTML = `<b>${esc(names.slice().reverse()[params.dataIndex] || '-')}</b> · ${Number(values.slice().reverse()[params.dataIndex]).toFixed(2)} 亿 · 数据日 ${esc(data.as_of || '未知')} · 口径：行业/概念主力净流入（以返回字段为准）`; });
}

function renderMarketRecap(data) {
  const stateEl = $('marketRecapState'), content = $('marketRecapContent');
  if (!stateEl || !content) return;
  if (!data || data.ok === false) {
    stateEl.className = 'data-state empty';
    stateEl.textContent = data?.error || '暂无盘后复盘快照，保留市场实时宽度和行情展示。';
    content.hidden = true;
    return;
  }
  const market = data.market || {}, ladder = market.ladder || {}, lines = data.mainlines || [];
  stateEl.hidden = true;
  content.hidden = false;
  $('marketRecapMeta').textContent = `数据日 ${data.as_of || '-'} · ${data.phase || '统一决策快照'} · ${data.date_mismatch ? '请求日期无精确快照，已回退最近可用日' : '按当前交易阶段选取'}`;
  $('marketRecapKpis').innerHTML = [
    ['情绪温度', recapNum(market.temperature, '/100'), market.emotion_stage || '未分类'],
    ['涨停 / 跌停', `${recapNum(market.limit_up)} / ${recapNum(market.limit_down)}`, `炸板 ${recapNum(market.broken_board)}`],
    ['最高连板', recapNum(market.max_board, '板'), `封板率 ${recapNum(market.seal_rate, '%')}`],
    ['资金合力', recapNum(market.force_index, '/100'), `宽度 ${market.breadth == null ? '-' : (Number(market.breadth) * 100).toFixed(1) + '%'}`],
  ].map(([label, value, note]) => stockOverviewKpi(label, value, note)).join('');
  const recapIndices = Object.entries(data.indices || {}).filter(([, item]) => item && item.index != null);
  const indicesEl = $('marketRecapIndices');
  if (indicesEl) indicesEl.innerHTML = recapIndices.length
    ? recapIndices.slice(0, 5).map(([name, item]) => {
        const pct = Number(item.change_pct ?? item.change);
        const tone = Number.isFinite(pct) ? (pct > 0 ? 'up' : pct < 0 ? 'down' : '') : '';
        return `<div class="market-index ${tone}"><b>${esc(name)}</b><strong>${esc(Number(item.index).toFixed(2))}</strong><small>${Number.isFinite(pct) ? `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%` : '涨跌缺失'} · ${item.amount_yi == null ? '成交额缺失' : `成交 ${Number(item.amount_yi).toFixed(0)}亿`}</small></div>`;
      }).join('')
    : '<div class="data-state empty">暂无指数快照，其他盘面证据仍可查看。</div>';
  const lineRows = lines.slice(0, 8).map(line => {
    const flow = line.net_yi == null ? '资金缺失' : `资金 ${Number(line.net_yi).toFixed(2)}亿`;
    const flowState = line.flow_fresh ? '同日' : line.flow_as_of ? `截至${line.flow_as_of}` : '未采集';
    return `<div class="recap-line"><span><b>${esc(line.name || line.concept || '未命名概念')}</b><small>${esc(line.level || '观察')} · 涨停 ${recapNum(line.zt)} · 最高 ${recapNum(line.max_board, '板')} · ${esc(flow)} · ${esc(flowState)}</small><small>结构日 ${esc(line.structure_as_of || '-')}</small></span><strong>${line.score == null ? '-' : Number(line.score).toFixed(1)}</strong></div>`;
  }).join('');
  const riskRows = (Array.isArray(market.risk_flags) ? market.risk_flags : []).slice(0, 6).map(x => `<li>${esc(x)}</li>`).join('');
  $('marketRecapMainlines').innerHTML = lineRows || '<div class="empty">暂无主线排行，未用单一涨停数量替代资金确认。</div>';
  $('marketRecapEmotion').innerHTML = `<div class="ladder-grid">${Object.entries(ladder).sort((a,b) => Number(b[0]) - Number(a[0])).map(([board, count]) => `<span><b>${esc(board)}板</b><small>${esc(count)}只</small></span>`).join('') || '<span class="empty">梯队缺失</span>'}</div><ul class="risk-list">${riskRows || '<li>暂无额外风险标记</li>'}</ul>`;
  renderMarketRecapFlow(data);
  renderMarketRecapCandidates(data);
  $('marketRecapMethod').textContent = `${data.methodology || ''} ${data.scope_detail?.execution || '本页仅作分析与决策辅助，不连接券商执行。'}`;
}

window.loadMarketRecap = loadMarketRecap;
async function loadMarketRecap(force = false) {
  try {
    const data = await api('/api/market_recap' + (force ? '?refresh=true' : ''), undefined, { cache_ttl: force ? 0 : 120, refresh: force });
    renderMarketRecap(data);
    return data;
  } catch (error) {
    renderMarketRecap({ ok: false, error: error.message });
    return { failed: 1 };
  }
}

async function loadDashboardBreadth() {
  try {
    const data = await api('/api/market');
    renderDashboardBreadth(data);
  } catch (e) {
    console.error('loadDashboardBreadth error:', e);
    renderDashboardBreadth({});
  }
}

function renderDashboardBreadth(payload) {
  const data = payload && payload.data ? payload.data : (payload || {});
  const box = $('breadthDisplay');
  if (!box) return;
  const breadth = data.breadth || {};
  const up = Number(breadth['上涨'] || 0);
  const down = Number(breadth['下跌'] || 0);
  const flat = Number(breadth['平盘'] || 0);
  const total = up + down + flat;
  if (!total) {
    box.innerHTML = '<div class="empty">市场宽度接口无有效数据</div>';
    return;
  }
  const pct = value => Math.round(value / total * 100);
  box.innerHTML = `<div class="breadth-bar">
    <div class="b-up" style="flex:${Math.max(up, 0.001)}">${pct(up)}% ↑</div>
    <div class="b-flat" style="flex:${Math.max(flat, 0.0001)}"></div>
    <div class="b-down" style="flex:${Math.max(down, 0.001)}">${pct(down)}% ↓</div>
  </div><div class="breadth-stats">
    <span><span class="up-b">▲ 上涨</span> <b>${up}</b></span>
    <span><span class="down-b">▼ 下跌</span> <b>${down}</b></span>
    <span>平盘 <b>${flat}</b></span>
    <span>上涨占比 <b>${breadth['上涨占比%'] ?? '-'}%</b></span>
    <span>等权涨跌 <b>${breadth['全A等权涨跌幅%'] ?? '-'}%</b></span>
    <span>中位数 <b>${breadth['涨跌幅中位数%'] ?? '-'}%</b></span>
    <span>涨停 <b>${breadth['近似涨停'] ?? '-'} / 跌停 ${breadth['近似跌停'] ?? '-'}</b></span>
  </div>`;
  const fmtDate = value => { const text = String(value || ''); return text.length === 8 ? `${text.slice(0,4)}-${text.slice(4,6)}-${text.slice(6,8)}` : text; };
  const set = (id, value) => { const el = $(id); if (el && value != null && value !== '') el.textContent = value; };
  const effectiveDate = fmtDate(data.trade_date);
  window.__marketTradeDate = effectiveDate;
  set('heroTradeDate', effectiveDate);
  set('heroBreadthPct', `${breadth['上涨占比%'] ?? '-'}%`);
  set('heroMedian', `${breadth['涨跌幅中位数%'] ?? '-'}%`);
  set('heroLimit', `${breadth['近似涨停'] ?? '-'} / ${breadth['近似跌停'] ?? '-'}`);
  const risk = $('heroRiskPill');
  if (risk && data.risk_level) { risk.style.display = ''; risk.classList.toggle('hi', Number(data.risk) >= 4); }
  const riskText = $('heroRiskText');
  if (riskText && data.risk_level) riskText.textContent = `${data.risk_level} · L${data.risk ?? '-'}`;
}

let coreHealthTimer = null;
let coreHeartbeatFailures = 0;

async function loadSystemStatus() {
  const el = $('systemStatus');
  try {
    const data = await api('/api/system/status', undefined, { cache_ttl: 30, silent: true });
    if (!data.ok) throw new Error(data.error || '系统状态不可用');
    coreHeartbeatFailures = 0;
    if (!el) return;
    const source = data.data_source_status ?? '未确认';
    const freshness = data.data_freshness ?? '-';
    const cron = data.cron_enabled == null || data.cron_total == null ? '未知' : `${data.cron_enabled}/${data.cron_total}`;
    el.innerHTML = `<div class="stock-kpi-row">${stockOverviewKpi('服务', data.server_alive === true ? '运行中' : data.server_alive === false ? '异常' : '未知', data.started_at ? `启动于 ${data.started_at}` : (data.uptime_seconds != null ? `运行 ${Math.floor(data.uptime_seconds / 3600)}小时` : '-'), data.server_alive === true ? 'positive' : data.server_alive === false ? 'negative' : '')}${stockOverviewKpi('数据源', source, freshness)}${stockOverviewKpi('定时任务', cron, '已启用 / 总数')}${stockOverviewKpi('最近日报', data.last_report_date ?? '未生成', data.last_report_size ?? '-')}</div>`;
  } catch (error) {
    coreHeartbeatFailures += 1;
    if (el) el.innerHTML = `<div class="data-state ${coreHeartbeatFailures >= 3 ? 'error' : 'empty'}">系统状态暂不可用：${esc(error.message)}</div>`;
  }
}

function startCoreHealth() {
  if (coreHealthTimer) clearInterval(coreHealthTimer);
  loadSystemStatus();
  coreHealthTimer = setInterval(() => {
    if (!document.hidden) loadSystemStatus();
  }, 60000);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && $('dashboard')?.classList.contains('active')) {
      loadSystemStatus();
      loadMacroOverview();
    }
  });
}

/* 统一自动刷新：只在当前视图和子视图可见时运行，页面隐藏时暂停。 */
const AUTO_REFRESH_MAP = [
  ['dashboard', '', 60000, loadIndexTicker],
  ['market', 'm-margin', 300000, loadMarginSummary],
  ['market', 'm-temp', 300000, loadMarketTemp],
  ['market', 'm-sector', 600000, loadSectorRotation],
  ['market', 'm-regime', 300000, loadMarketRegime],
  ['market', 'm-sentiment', 300000, loadNorthFlow],
  ['risk', 'r-portfolio', 300000, loadPortfolio],
  ['ops', 'o-alerts', 60000, loadAlertsStatus],
  ['ops', 'o-datahealth', 60000, loadDataHealth],
  ['ops', 'o-cron', 60000, loadCronTasks],
  ['ops', 'o-chain', 120000, loadDecisionChain],
];

let autoRefreshTimer = null;
function startAutoRefresh() {
  if (autoRefreshTimer) clearInterval(autoRefreshTimer);
  const lastRun = {};   // `${viewId}/${subId}` -> last fire ts
  autoRefreshTimer = setInterval(function () {
    const active = document.querySelector('.view.active');
    const activeId = active && active.id;
    const activeSub = active && active.querySelector('.sub-view.active');
    const activeSubId = activeSub && activeSub.id;
    const now = Date.now();
    AUTO_REFRESH_MAP.forEach(function (item) {
      const viewId = item[0], subId = item[1], ms = item[2], fn = item[3];
      if (activeId !== viewId || activeSubId !== subId) return;
      const key = viewId + '/' + subId;
      if (now - (lastRun[key] || 0) < ms) return;
      lastRun[key] = now;
      Promise.resolve().then(fn).catch(function (e) {
        console.warn('[auto-refresh] ' + key + ' error:', e);
        setStatus('自动刷新失败: ' + key + ' · ' + (e.message || e));
      });
    });
  }, 30000);
  loadSystemStatus();
}

// ════════════════════════════════════════
// 子Tab加载函数
// ════════════════════════════════════════

// 获取共享股票代码
function sharedSymbol() {
  return ($('sSymbol')?.value || '').trim() || state.lastKlineSymbol || '002714';
}

window.LOAD_STOCK_overview = function() {
  setTimeout(loadStockOverview, 80);
};
window.LOAD_STOCK_kline = function() {
  // 如果已经加载过且还在看，不重复加载
  const sym = sharedSymbol();
  if ($('chartStart')) $('chartSymbol') && ($('chartSymbol').value = sym);
  if (!state.lastSnapshot || state.lastSnapshot.symbol !== sym) {
    setTimeout(loadChart, 100);
  }
};
window.LOAD_STOCK_ml = function() {
  const sym = sharedSymbol();
  if ($('mlSignalSymbol')) $('mlSignalSymbol').value = sym;
  if (!$('mlSignalResult')?.dataset?.loaded) setTimeout(() => loadMLSignal('predict'), 100);
};
window.LOAD_STOCK_panorama = function() {
  const sym = sharedSymbol();
  if ($('lensCode')) $('lensCode').value = sym;
  setTimeout(loadStockLens, 100);
};
window.LOAD_STOCK_intraday = function() {
  // 默认不自动加载，用户点击加载按钮
};
window.LOAD_STOCK_patterns = function() {
  const sym = sharedSymbol();
  if ($('patternSymbol')) $('patternSymbol').value = sym;
  // 不自动加载，用户点击按钮
};
window.LOAD_STOCK_fundamental = function() {
  const sym = sharedSymbol();
  if ($('fundSymbol')) $('fundSymbol').value = sym;
  setTimeout(loadFundamentals, 100);
};

window.LOAD_STOCK_factors = function() {
  const sym = sharedSymbol();
  if ($('factorSymbol')) $('factorSymbol').value = sym;
  setTimeout(loadFactorScan, 100);
};

window.LOAD_STRATEGY_scan = function() {
  // 扫描默认显示看板
};

window.LOAD_MARKET_sector = function() {
  setTimeout(loadSectorRotation, 100);
  setTimeout(loadAllDayIndustry, 180);
};
window.LOAD_MARKET_calendar = function() {
  setTimeout(loadMacroCal, 100);
};

window.LOAD_RISK_opportunity = function() {
  setTimeout(loadOpportunities, 100);
};

// 修正：使用sSymbol的各函数
let macroOverviewRequestId = 0;
let macroOverviewPromise = null;
async function loadMacroOverview(forceRefresh = false) {
  if (macroOverviewPromise) return macroOverviewPromise;
  const requestId = ++macroOverviewRequestId;
  macroOverviewPromise = (async function () {
    try {
      const refresh = forceRefresh ? { refresh: true, cache_ttl: 0 } : {};
      const results = await Promise.allSettled([
        api('/api/market_pano' + (forceRefresh ? '?refresh=true' : ''), undefined, {...refresh, timeout:8000, retries:0, silent:true}),
        api('/api/market_temp' + (forceRefresh ? '?refresh=true' : ''), undefined, {...refresh, timeout:8000, retries:0, silent:true}),
        api('/api/market_pulse' + (forceRefresh ? '?refresh=true' : ''), undefined, {...refresh, timeout:8000, retries:0, silent:true}),
        api('/api/sentiment_market' + (forceRefresh ? '?refresh=true' : ''), undefined, {...refresh, timeout:8000, retries:0, silent:true}),
      ]);
      if (requestId !== macroOverviewRequestId) return;
      const names = ['pano','temp','pulse','sentiment'];
      const market = {};
      results.forEach((result, index) => {
        market[names[index]] = result.status === 'fulfilled'
          ? result.value
          : { ok: false, error: result.reason?.message || '数据源不可用' };
      });
      state.market = market;
      renderMarket(state.market);
      const failed = results.filter(result => result.status === 'rejected').length;
      if (failed) setStatus(`宏观总览部分数据不可用（${failed}项），其余模块仍已展示`);
      return { failed };
    } catch(e) {
      console.error('loadMacroOverview error:', e);
      setStatus('宏观总览失败: '+e.message);
    } finally {
      if (requestId === macroOverviewRequestId) macroOverviewPromise = null;
    }
  })();
  return macroOverviewPromise;
}

function renderDashboardCoreKpis(market) {
  const grid = $('kpiGrid');
  if (!grid) return;
  const unwrap = payload => payload && payload.data ? payload.data : (payload || {});
  const rawIndices = unwrap(market.indices);
  const indices = rawIndices.ashare || {};
  const temp = unwrap(market.temp);
  const pulse = unwrap(market.pulse);
  const margin = unwrap(market.margin);
  const cards = [];
  ['上证指数', '深证成指', '创业板指', '沪深300'].forEach(name => {
    const value = indices[name];
    if (!value || value.index == null) return;
    const change = Number(value.change_pct ?? value.change ?? 0);
    cards.push({ label: name, value: Number(value.index).toFixed(2),
      delta: `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`,
      cls: change > 0 ? 'up' : (change < 0 ? 'down' : 'flat'), sub: '指数' });
  });
  if (temp.temperature != null) cards.push({ label: '市场温度', value: `${temp.temperature}°`,
    delta: `上涨 ${temp.advance ?? '-'} / 下跌 ${temp.decline ?? '-'}`,
    cls: Number(temp.temperature) >= 60 ? 'up' : (Number(temp.temperature) <= 40 ? 'down' : 'flat'), sub: '市场宽度' });
  const fearGreed = pulse.index ?? pulse.fear_greed_index ?? pulse.fgi;
  if (fearGreed != null) cards.push({ label: '恐贪指数', value: Number(fearGreed).toFixed(1),
    delta: pulse.level || '市场脉搏', cls: Number(fearGreed) >= 60 ? 'up' : (Number(fearGreed) <= 40 ? 'down' : 'flat'), sub: '0-100' });
  const totalMargin = margin.total_margin_balance ?? margin.total_margin ?? margin.total;
  if (totalMargin != null) cards.push({ label: '两融余额', value: `${formatMacroNumber(totalMargin, 0)} 亿`,
    delta: '沪深全量', cls: 'flat', sub: margin.warning || '融资融券' });
  // 北向资金与舆情不是同一指标：保留旧版资金 KPI，但停更后显示口径状态。
  const north = unwrap(market.north);
  const northRaw = north.raw || north;
  const northSummary = north.summary || north;
  if (northRaw.net_buy_disclosed === false) {
    cards.push({ label: '北向资金', value: '已停更', delta: '净买入不再披露', cls: 'flat', sub: '持股结构仍可查' });
  } else {
    const northNet = northSummary['当日净买入(亿)'] ?? northSummary.total_net_yi;
    if (northNet != null) cards.push({ label: '北向资金', value: `${formatMacroNumber(northNet, 2)} 亿`, delta: '沪深港通', cls: Number(northNet) >= 0 ? 'up' : 'down', sub: northRaw.date || '数据日未知' });
  }
  if (!cards.length) { grid.innerHTML = '<div class="empty">核心行情暂无可用数据</div>'; return; }
  const icons = { '上证指数':'🏛️', '深证成指':'🏙️', '创业板指':'🚀', '沪深300':'💎', '市场温度':'🌡️', '恐贪指数':'💓', '两融余额':'🏦' };
  grid.innerHTML = cards.map(card => `<div class="kpi-card kpi-${card.cls}">
    <div class="kpi-head"><div class="kpi-label">${esc(card.label)}</div><div class="kpi-icon">${icons[card.label] || '📈'}</div></div>
    <div class="kpi-value">${esc(card.value)}</div><div class="kpi-delta ${card.cls}">${card.cls === 'up' ? '▲' : (card.cls === 'down' ? '▼' : '—')} ${esc(card.delta)}</div>
    <div class="kpi-sub">${esc(card.sub)}</div></div>`).join('');
}

function renderMarket(market) {
  // 首屏核心行情由主脚本直接渲染，增强层只能附加，不能阻断。
  const safely = (name, fn) => { try { fn(); } catch (error) { console.error(`render ${name} failed:`, error); } };
  safely('temperature', () => renderTempGauge(market.temp));
  safely('pulse-margin', () => renderPulseMargin(market.pulse, market.margin));
  safely('sentiment', () => renderDashboardSentiment(market.sentiment));
  safely('north-bound', () => renderNorthBoundSummary(market.north));
  safely('core-kpis', () => renderDashboardCoreKpis(market));
  // 图表增强只监听事件，不再二次接管核心 KPI DOM。
}

async function runScan() {
  setBusy('scanBtn',true); setStatus('扫描自选股...');
  try {
    const data = await api('/api/scan', Object.assign({symbols:$('scanSymbols').value,start:$('scanStart').value,refresh:$('scanRefresh').checked}, readParams()));
    state.lastScan = data.rows||[];
    renderTable('scanTable',state.lastScan,scanColumns());
    $('scanActions').style.display = state.lastScan.length?'block':'none';
    renderScanPreview();
    setStatus(`扫描完成：${state.lastScan.length} 条结果`);
  } catch (error) {
    state.lastScan = [];
    renderTable('scanTable', [], scanColumns());
    $('scanActions').style.display = 'none';
    setStatus(`扫描失败：${error.message}`);
  } finally { setBusy('scanBtn',false); }
}

function renderScanPreview() {
  if(!state.lastScan.length)return;
  $('scanPreview').innerHTML='';
  const t=document.createElement('table');
  $('scanPreview').appendChild(t);
  renderTableByEl(t,state.lastScan.slice(0,8),['symbol','action','close','composite_score','trend_score','risk_score','suggested_shares']);
}

let stockFlowOverviewChart = null;
let stockOverviewData = null;

function stockOverviewKpi(label, value, note, tone='') {
  return `<div class="stock-kpi ${tone}"><span>${esc(label)}</span><strong>${esc(value ?? '-')}</strong><small>${esc(note || '')}</small></div>`;
}

function renderStockFlowOverview(flow, mode = 'industry') {
  const dom = $('stockFlowChart');
  const rawRows = Array.isArray(flow?.rows) ? flow.rows : [];
  if (!dom) return;
  const isStock = mode === 'stock';
  const valueKey = isStock ? 'main_net_yi' : 'active_flow_proxy_yi';
  const rows = rawRows.filter(row => row && Number.isFinite(Number(row[valueKey])));
  const dateKey = isStock ? 'date' : 'time';
  const label = isStock ? '个股主力净流入（真实）' : '行业成交额方向代理（非主力净流入）';
  if (!rows.length) {
    if (stockFlowOverviewChart) { stockFlowOverviewChart.dispose(); stockFlowOverviewChart = null; }
    dom.innerHTML = `<div class="data-state empty">暂无${isStock ? '真实个股主力资金' : '可关联的行业资金代理'}时序${isStock ? '，不以行业代理替代。' : '，请检查行业映射或数据采集状态。'}</div>`;
    return;
  }
  stockFlowOverviewChart ||= echarts.init(dom);
  const values = rows.map(r => r[valueKey] == null ? null : Number(r[valueKey]));
  const option = {
    grid:{left:58,right:18,top:28,bottom:38},
    tooltip:{trigger:'axis', valueFormatter:v => v == null ? '-' : `${v} 亿（${isStock ? '真实个股主力净流入' : '行业成交额方向代理，非主力净流入'}）`},
    legend:{data:[label],textStyle:{color:'#8b949e'}},
    xAxis:{type:'category',data:rows.map(r=>r[dateKey] || ''),axisLabel:{color:'#8b949e'}},
    yAxis:{type:'value',name:isStock ? '亿元（真实主力净流入）' : '亿元（成交额方向代理）',axisLabel:{color:'#8b949e'}},
    series:[{name:label,type:'bar',data:values,itemStyle:{color:p=>p.value >= 0 ? '#b42318' : '#087443'}}]
  };
  stockFlowOverviewChart.setOption(applyChartDarkTheme(option), true);
}

function safeHttpUrl(value) {
  const url = String(value || '').trim();
  return /^https?:\/\//i.test(url) ? url : '';
}

function renderStockOverview(data) {
  data = data && typeof data === 'object' ? data : {};
  stockOverviewData = data;
  const profile = data.profile && typeof data.profile === 'object' ? data.profile : {};
  const quote = profile.quote && typeof profile.quote === 'object' ? profile.quote : {};
  const trend = profile.trend && typeof profile.trend === 'object' ? profile.trend : {};
  const liquidity = profile.liquidity && typeof profile.liquidity === 'object' ? profile.liquidity : {};
  const financial = profile.financial && typeof profile.financial === 'object' ? profile.financial : {};
  const valuation = profile.valuation && typeof profile.valuation === 'object' ? profile.valuation : {};
  const decision = data.decision && typeof data.decision === 'object' ? data.decision : {};
  const flow = data.flow && typeof data.flow === 'object' ? data.flow : {};
  const stockFlow = data.stock_flow && typeof data.stock_flow === 'object' ? data.stock_flow : {};
  const stockNews = data.stock_news && typeof data.stock_news === 'object' ? data.stock_news : {};
  const newsItems = Array.isArray(stockNews.items) ? stockNews.items : [];
  const explanation = data.decision_explanation && typeof data.decision_explanation === 'object' ? data.decision_explanation : {};
  const research = Array.isArray(data.research) ? data.research : [];
  const statuses = data.source_status && typeof data.source_status === 'object' ? data.source_status : {};
  const gate = data.decision_gate && typeof data.decision_gate === 'object' ? data.decision_gate : {};
  const hierarchy = data.score_hierarchy && typeof data.score_hierarchy === 'object' ? data.score_hierarchy : {};
  const icEvidence = data.ic_weight_explanation && typeof data.ic_weight_explanation === 'object' ? data.ic_weight_explanation : {};
  const marketContext = data.market_context && typeof data.market_context === 'object' ? data.market_context : {};
  const stateEl = $('stockOverviewState');
  const content = $('stockOverviewContent');
  const hasQuote = quote.close != null && Number.isFinite(Number(quote.close));
  if (stateEl) {
    stateEl.hidden = hasQuote;
    if (!hasQuote) {
      stateEl.className = 'data-state empty';
      const sourceNotes = Object.values(statuses).filter(s => s && typeof s === 'object').map(s => `${s.label || '来源'}：${statusCn(s.status)}`).join(' · ');
      stateEl.innerHTML = `<strong>${esc(data.name || data.symbol || '个股')}已识别，但行情画像不可用</strong><span>不展示空壳评分；请检查数据健康或等待行情仓库更新。</span><small>${esc(sourceNotes)}</small>`;
    }
  }
  if (content) content.hidden = !hasQuote;
  if (!hasQuote) return;

  const change = quote.chg_pct;
  const changeTone = change > 0 ? 'positive' : change < 0 ? 'negative' : '';
  const profileScore = profile.score && typeof profile.score === 'object' ? profile.score : {};
  const commodity = data.commodity && typeof data.commodity === 'object' ? data.commodity : (profile.commodity && typeof profile.commodity === 'object' ? profile.commodity : {});
  const margin = data.margin && typeof data.margin === 'object' ? data.margin : {};
  const scoreCoverage = profileScore.coverage || {};
  const scoreWeights = Object.entries(profileScore.weights || {}).map(([key, value]) => `${uiDimension(key)} ${(Number(value) * 100).toFixed(1)}%`).join(' · ');
  const gateTone = gate.status === 'tradable' ? 'positive' : gate.status === 'blocked' || gate.status === 'insufficient_data' ? 'negative' : '';
  const icSummary = icEvidence.status === 'available' ? `IC校准 ${icEvidence.included_factor_count || 0}因子 · ${icEvidence.as_of || '日期缺失'}` : `IC校准不可用 · ${icEvidence.fallback || '保留业务先验'}`;
  const gateReasons = [...(Array.isArray(gate.blockers) ? gate.blockers : []), ...(Array.isArray(gate.warnings) ? gate.warnings : [])].slice(0,3).join('；');
  const gateLabel = gate.label || uiStatus(gate.status || 'review_only');
  const dimensionContributions = Object.entries(profileScore.dimension_contributions || {}).map(([key, value]) => `${uiDimension(key)} ${value}`).join(' · ');
  const unavailableDimensions = (profileScore.unavailable_dimensions || []).map(uiDimension).join('、');
  $('stockDecisionPanel').innerHTML = `<div class="panel-heading"><h2>${esc(data.name)} <small>${esc(data.symbol)}</small></h2><span class="decision-badge ${gateTone}">${esc(gateLabel)}</span></div>
    <p class="decision-line">${esc(gate.primary_reason || decision.decision || explanation.stance || '当前仅展示客观数据。')}</p>
    <div class="stock-kpi-row">${stockOverviewKpi('基础画像',profileScore.total ?? '-','有效维度 '+(scoreCoverage.available_dimensions ?? '-')+'/'+(scoreCoverage.total_dimensions ?? '-'))}${stockOverviewKpi('分层综合',hierarchy.final_score ?? '-',`门控 ${uiStatus(hierarchy.trade_bucket || gate.status || 'review_only')}`)}${stockOverviewKpi('行业商品',commodity.score ?? '-',commodity.coverage == null ? '覆盖未知' : '覆盖 '+(Number(commodity.coverage) * 100).toFixed(0)+'%')}${stockOverviewKpi('短线评分',decision.short_term_score ?? '-','盘口/主线/资金')}</div>
    <div class="method-note"><b>门控：</b>${esc(uiStatus(gate.status || 'review_only'))} · ${esc(gateReasons || '无额外阻断')}<br><b>画像实际权重：</b>${esc(scoreWeights || '无有效评分维度')}<br><b>原始分 / 覆盖率折减分：</b>${esc(profileScore.raw_score ?? profileScore.total ?? '-')} / ${esc(profileScore.coverage_adjusted_score ?? '-')}（覆盖 ${esc(((Number(scoreCoverage.coverage_ratio || 0)) * 100).toFixed(1))}%）<br><b>维度贡献：</b>${esc(dimensionContributions || '暂无')}<br><b>未参与：</b>${esc(unavailableDimensions || '无')}<br><b>IC/OOS：</b>${esc(icSummary)}<br><b>决策依据：</b>${esc((explanation.supporting_factors || []).slice(0,4).join('；') || '暂无已加载的支持因素')}</div>`;
  $('stockQuotePanel').innerHTML = `<div class="panel-heading"><h2>行情、估值与财务</h2><span class="data-meta">${esc(quote.date || '无行情日期')}</span></div>
    <div class="stock-kpi-row">${stockOverviewKpi('收盘',quote.close ?? '-',change == null?'涨跌缺失':`${change>0?'+':''}${change}%`,changeTone)}${stockOverviewKpi('20日位置',trend.mom20 == null?'-':`${trend.mom20>0?'+':''}${trend.mom20}%`,'相对MA20')}${stockOverviewKpi('成交额',quote.amount?`${(quote.amount/1e8).toFixed(2)}亿`:'-',`20日均值 ${liquidity.amount_ma20_yi ?? '-'}亿`)}${stockOverviewKpi('流通市值',valuation.float_mv_yi == null?'-':`${valuation.float_mv_yi}亿`,valuation.float_mv_status === 'available' ? `股本快照 ${valuation.float_mv_as_of || '日期缺失'}` : '流通股本快照缺失')}${stockOverviewKpi('PE / PB',`${valuation.pe_ttm ?? '-'} / ${valuation.pb ?? '-'}`,`财务覆盖 ${financial.coverage == null ? '-' : (Number(financial.coverage) * 100).toFixed(0) + '%'}`)}</div><div class="method-note">ROE ${esc(financial.roe ?? '-')}% · 营收同比 ${esc(financial.rev_yoy ?? '-')}% · 净利同比 ${esc(financial.profit_yoy ?? '-')}% · 资产负债率 ${esc(financial.debt_ratio ?? '-')}%</div>`;
  const flowRows = Array.isArray(flow.rows) ? flow.rows : [];
  const stockFlowRows = Array.isArray(stockFlow.rows) ? stockFlow.rows : [];
  const latestFlow = flowRows.at(-1) || {};
  const latestStockFlow = stockFlowRows.at(-1) || {};
  const stockFlowStatus = statuses.stock_flow && typeof statuses.stock_flow === 'object' ? statuses.stock_flow : {};
  const industryFlowStatus = statuses.flow && typeof statuses.flow === 'object' ? statuses.flow : {};
  $('stockFlowPanel').innerHTML = `<div class="panel-heading"><h2>资金证据摘要</h2><span class="data-meta">个股 ${esc(stockFlow.as_of || '日期缺失')} · 行业 ${esc(flow.date || '日期缺失')}</span></div>
    <p class="decision-line">个股主力：${latestStockFlow.main_net_yi == null ? '缺失' : `${latestStockFlow.main_net_yi > 0 ? '+' : ''}${Number(latestStockFlow.main_net_yi).toFixed(3)}亿`} · 行业代理：${latestFlow.active_flow_proxy_yi == null ? '缺失' : `${latestFlow.active_flow_proxy_yi > 0 ? '+' : ''}${latestFlow.active_flow_proxy_yi}亿`}</p>
    <div class="stock-kpi-row">${stockOverviewKpi('个股主力净流入',latestStockFlow.main_net_yi == null?'-':`${latestStockFlow.main_net_yi>0?'+':''}${Number(latestStockFlow.main_net_yi).toFixed(3)}亿`,stockFlow.is_real_main_net?'真实个股级口径':stockFlowStatus.message||'未加载',latestStockFlow.main_net_yi>0?'positive':latestStockFlow.main_net_yi<0?'negative':'')}${stockOverviewKpi('行业成交方向参考',latestFlow.active_flow_proxy_yi == null?'-':`${latestFlow.active_flow_proxy_yi>0?'+':''}${latestFlow.active_flow_proxy_yi}亿`,industryFlowStatus.message||'仅作行业趋势参考，不等于个股主力净流入',latestFlow.active_flow_proxy_yi>0?'positive':latestFlow.active_flow_proxy_yi<0?'negative':'')}</div>
    <p class="method-note">${esc(stockFlow.methodology || flow.methodology || stockFlowStatus.message || industryFlowStatus.message || '当前没有可用资金证据，不进入交易结论。')}</p>`;
  const blockers = Array.isArray(decision.short_term_blockers) ? decision.short_term_blockers : (Array.isArray(decision.trade_reasons) ? decision.trade_reasons : []);
  $('stockRiskPanel').innerHTML = `<div class="panel-heading"><h2>风险、宏观与失效条件</h2></div>${blockers.length?`<ul class="risk-list">${blockers.slice(0,5).map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:'<div class="data-state empty">没有候选级风险结论，请结合公告与仓位风控复核。</div>'}<div class="method-note">宏观背景：PMI ${esc(marketContext.pmi?.value ?? '-')} · M2 ${esc(marketContext.m2?.value ?? '-')} ${esc(marketContext.m2?.unit || '')} · SHIBOR ${esc(marketContext.shibor?.value ?? '-')}<br>${esc(marketContext.usage || '宏观数据仅作背景，不能替代个股财务和行业传导证据。')}</div>`;
  const stockMarginBox = $('stockMarginPanel');
  if (stockMarginBox) stockMarginBox.innerHTML = `<div class="panel-heading"><h2>个股两融</h2><span class="data-meta">${esc(margin.as_of || data.data_as_of || '数据日待确认')}</span></div><div class="stock-kpi-row">${stockOverviewKpi('融资余额', margin.rzye == null ? '-' : `${margin.rzye} 亿`, margin.status === 'available' ? '个股级真实数据' : '数据不可用')}${stockOverviewKpi('融券余额', margin.rqye == null ? '-' : `${margin.rqye} ${margin.rqye_unit || '亿元'}`, margin.status === 'available' ? '个股级真实数据' : '数据不可用')}</div>`;
  const commodityItems = Array.isArray(commodity.items) ? commodity.items : [];
  const commodityStatusLabel = item => item.status === 'available' ? `${item.return_20d_pct > 0 ? '+' : ''}${item.return_20d_pct}%` : item.status === 'substitute_only' ? '替代展示' : item.status === 'not_observable' ? '无直接期货' : item.status === 'stale' ? '数据陈旧' : item.status === 'insufficient' ? '历史不足' : item.status === 'missing' ? '未采集' : '不可用';
  const commodityRoleLabel = role => ({product:'产品价格',input_cost:'投入成本',upstream_proxy:'上游代理'}[String(role)] || '产业链参考');
  const commodityDirectionLabel = item => Number(item.direction) > 0 ? '价格上涨正向' : Number(item.direction) < 0 ? '价格上涨成本负向' : '方向未知';
  const statusRows = Object.values(statuses).filter(s => s && typeof s === 'object');
  const dataChainAvailable = statusRows.filter(s => s.status === 'available' || s.status === 'partial').length;
  $('stockCommodityPanel').innerHTML = `<div class="panel-heading"><h2>行业商品 / 期货</h2><span class="data-meta">${esc(commodity.profile || commodity.sw1 || '未映射')}</span></div>
    <p class="decision-line">商品得分：${commodity.score == null ? '不可用' : commodity.score} · 经济暴露覆盖 ${commodity.coverage == null ? '未知' : (Number(commodity.coverage) * 100).toFixed(0) + '%'} · 利润代理 ${commodity.profit_proxy_20d_pct == null ? '不可用' : (Number(commodity.profit_proxy_20d_pct) > 0 ? '+' : '') + commodity.profit_proxy_20d_pct + '%'}</p>
    <div class="commodity-evidence-list">${commodityItems.map(item=>`<div class="commodity-evidence-row"><span><b>${esc(item.name)}</b><small>${commodityRoleLabel(item.role)} · ${commodityDirectionLabel(item)} · 权重 ${item.weight == null ? '未知' : esc(item.weight)} · ${esc(item.as_of || '日期未知')}</small></span><strong class="${item.return_20d_pct > 0 ? 'up' : item.return_20d_pct < 0 ? 'down' : ''}">${commodityStatusLabel(item)}</strong></div>`).join('') || '<div class="data-state empty">该行业没有可审计的直接商品期货暴露。</div>'}</div>
    <p class="method-note">${esc(commodity.profit_proxy_method || commodity.methodology || commodity.profile_reason || '行业映射缺失，不进入评分。')} ${commodity.missing_items?.length ? `未纳入：${commodity.missing_items.join('、')}` : ''}</p>`;

  const stockModeAvailable = stockFlow.status === 'available' && stockFlowRows.length > 0;
  // 默认始终保持个股口径；个股资金缺失时显示空态，行业代理只能由用户主动选择。
  const flowMode = 'stock';
  $('stockFlowSource').textContent = stockModeAvailable
    ? `${data.symbol} 个股主力净流入 · 截至${stockFlow.as_of || '未知'} · 真实个股口径`
    : `${data.symbol} 个股真实资金不可用 · 当前不以行业代理替代`;
  renderStockFlowOverview(stockFlow, flowMode);
  document.querySelectorAll('[data-stock-flow-mode]').forEach(btn => {
    const isStockButton = btn.dataset.stockFlowMode === 'stock';
    btn.classList.toggle('active', btn.dataset.stockFlowMode === flowMode);
    btn.disabled = isStockButton && !stockModeAvailable;
    btn.title = isStockButton && !stockModeAvailable ? (stockFlowStatus.message || '个股真实资金不可用，不能切换到该口径') : '';
  });
  const extracted = research.filter(x=>x.has_decision_extract).length;
  const layers = Array.isArray(hierarchy.layers) ? hierarchy.layers : [];
  const icRows = (Array.isArray(icEvidence.weights) ? icEvidence.weights : []).filter(x => x.included).slice(0, 8);
  $('stockDecisionDetail').innerHTML = `<div class="panel-heading"><h2>决策解释与校准</h2><span class="data-meta">${esc(icEvidence.as_of || 'IC日期缺失')}</span></div>
    <div class="decision-detail-grid"><div><b>分层评分</b>${layers.map(x=>`<div class="detail-row"><span>${esc(x.label)}</span><strong>${x.score == null ? '-' : x.score}</strong><small>${x.eligible ? `权重 ${(Number(x.weight) * 100).toFixed(1)}%` : esc(x.reason || '未参与')}</small></div>`).join('') || '<div class="empty">暂无分层评分</div>'}</div>
    <div><b>昨日IC/OOS校准</b><p class="method-note">${esc(icEvidence.fallback || icEvidence.method || '未声明')}</p>${icRows.map(x=>`<div class="detail-row"><span>${esc(x.factor)} · ${esc(uiFactorCategory(x.category))}</span><strong>${x.ic_mean == null ? '-' : Number(x.ic_mean).toFixed(4)}</strong><small>ICIR ${x.icir == null ? '-' : Number(x.icir).toFixed(2)} · 权重 ${(Number(x.final_weight || 0) * 100).toFixed(1)}%</small></div>`).join('') || '<div class="empty">暂无有效IC因子</div>'}</div></div>`;
  $('stockResearchMeta').textContent = `${research.length}篇命中 · ${extracted}篇含决策字段`;
  $('stockResearchList').innerHTML = `
    <div class="stock-news-summary"><b>个股新闻证据</b><span>${stockNews.total ?? '-'}条 · ${stockNews.status === 'available' ? `情绪 ${fmt(stockNews.sentiment_score)}` : stockNews.status === 'missing' ? '无代码级新闻' : stockNews.status === 'error' ? '接口失败' : '字段/来源不可用'}</span><small>${esc(stockNews.coverage_note || stockNews.error || '新闻不自动进入交易评分，仅供事件复核。')}</small></div>
    ${newsItems.slice(0,5).map(row=>`<article class="evidence-item">
      <div><strong>${esc(row.title || '未命名新闻')}</strong><span>${esc(row.date || '日期缺失')}</span></div>
      <p>新闻情绪：${esc(({positive:'偏积极',negative:'偏谨慎',neutral:'中性'}[row.sentiment] || '待确认'))} · 不改变交易结论</p>
    </article>`).join('') || '<div class="data-state empty">未取得该代码可归因的新闻标题；不使用市场聚合新闻替代。</div>'}
    <div class="research-divider">研报提取证据</div>
    ${research.length ? research.slice(0,8).map(row=>`<article class="evidence-item">
      <div><strong>${esc(row.title || '未命名资料')}</strong><span>${esc(row.date || '日期缺失')}</span></div>
      <p>${row.has_decision_extract ? esc([row.rating&&`评级 ${row.rating}`,row.target_price&&`目标价 ${row.target_price}`,(row.catalysts||[]).length&&`催化 ${(row.catalysts||[]).join('、')}`].filter(Boolean).join(' · ')) : '仅完成代码/公司索引，尚未提取评级、目标价或核心观点，不作为决策加分。'}</p>
    </article>`).join('') : '<div class="data-state empty">未匹配该股研报。系统不会用全库研报数量替代个股证据。</div>'}`;
  $('stockSourceStatus').innerHTML = statusRows.map(s=>`<span class="source-chip ${esc(s.status || 'missing')}"><b>${esc(s.label || '未命名来源')}</b> ${esc(s.as_of || '')}<small>${esc(s.message || '')}</small></span>`).join('') || '<span class="source-chip missing"><b>来源状态</b><small>服务未返回来源状态，不能形成可执行结论。</small></span>';
  $('stockOverviewMeta').textContent = `更新至 ${quote.date || flow.date || '未知'} · ${dataChainAvailable}/${statusRows.length} 数据链可用`;
}

function renderSelectedStockFlow(mode) {
  if (!stockOverviewData) return;
  const stockFlow = stockOverviewData.stock_flow || {};
  const flow = stockOverviewData.flow || {};
  const stockModeAvailable = stockFlow.status === 'available' && Array.isArray(stockFlow.rows) && stockFlow.rows.length > 0;
  const actualMode = mode === 'stock' ? 'stock' : 'industry';
  document.querySelectorAll('[data-stock-flow-mode]').forEach(btn => {
    const isStockButton = btn.dataset.stockFlowMode === 'stock';
    btn.classList.toggle('active', btn.dataset.stockFlowMode === actualMode);
    btn.disabled = isStockButton && !stockModeAvailable;
    btn.title = isStockButton && !stockModeAvailable ? (stockOverviewData.source_status?.stock_flow?.message || '个股真实资金不可用') : '';
  });
  $('stockFlowSource').textContent = actualMode === 'stock'
    ? (stockModeAvailable ? `${stockOverviewData.symbol} 个股主力净流入 · 截至${stockFlow.as_of || '未知'} · 真实个股口径` : `${stockOverviewData.symbol} 个股真实资金不可用 · 不展示行业代理替代`)
    : `${stockOverviewData.symbol} 所属行业方向代理 · ${flow.date || '未知'} · 非个股主力净流入`;
  renderStockFlowOverview(actualMode === 'stock' ? stockFlow : flow, actualMode);
}

async function loadStockOverview(force=false) {
  const symbol = sharedSymbol();
  const stateEl = $('stockOverviewState');
  if (stateEl) { stateEl.hidden = false; stateEl.className = 'data-state loading'; stateEl.textContent = `正在汇总 ${symbol} 的行情、资金与研报证据...`; }
  if ($('stockOverviewContent')) $('stockOverviewContent').hidden = true;
  setBusy('stockOverviewBtn', true);
  try {
    const data = await api(`/api/stock_overview?symbol=${encodeURIComponent(symbol)}`, undefined, force?{refresh:true,cache_ttl:0}:{});
    renderStockOverview(data);
    try { const margin = await api(`/api/margin?symbol=${encodeURIComponent(symbol)}`, undefined, { timeout: 30000, silent: true }); const panel = $('stockMarginPanel'); if (panel) panel.innerHTML = `<div class="panel-heading"><h2>个股两融</h2><span class="data-meta">${esc(margin.as_of || data.data_as_of || '数据日待确认')}</span></div><div class="stock-kpi-row">${stockOverviewKpi('融资余额', margin.rzye == null ? '-' : `${margin.rzye} 亿`, margin.status === 'available' ? '个股级真实数据' : '数据不可用')}${stockOverviewKpi('融券余额', margin.rqye == null ? '-' : `${margin.rqye} ${margin.rqye_unit || '亿元'}`, margin.status === 'available' ? '个股级真实数据' : '数据不可用')}</div>`; } catch (e) { const panel = $('stockMarginPanel'); if (panel) panel.innerHTML = `<div class="panel-heading"><h2>个股两融</h2></div><div class="data-state empty">个股两融暂不可用：请稍后重试。</div>`; }
    setStatus(`${data.name}（${data.symbol}）决策台已更新`);
  } catch (error) {
    if (stateEl) { stateEl.hidden = false; stateEl.className = 'data-state error'; stateEl.innerHTML = `<strong>个股决策台加载失败</strong><span>${esc(error.message)}</span><button type="button" onclick="loadStockOverview(true)">重试</button>`; }
    setStatus(`个股决策台失败：${error.message}`);
  } finally { setBusy('stockOverviewBtn', false); }
}
window.loadStockOverview = loadStockOverview;
$('stockOverviewBtn')?.addEventListener('click', ()=>loadStockOverview(true));

async function loadChart() {
  setBusy('loadChartBtn',true); setStatus('加载K线...');
  try {
    const sym = ($('sSymbol')?.value || '').trim() || $('chartSymbol')?.value || state.lastKlineSymbol || '002714';
    state.lastKlineSymbol = sym;
    const p=readParams();
    const qs=new URLSearchParams({symbol:sym,start:$('chartStart').value,refresh:String($('chartRefresh').checked),period:state.period});
    Object.entries(p.strategy).forEach(([k,v])=>qs.set(k,String(v)));
    const data=await api(`/api/history?${qs.toString()}`);
    state.lastSnapshot=data.snapshot;
    renderSnapshot(data.snapshot);
    renderChartConclusion(data.snapshot);
    renderRadarChart(data.snapshot);
    renderKlineChart('klineChart',data.snapshot,data.rows||[],state.period);
    setStatus(`${sym} K线已更新 · 数据日 ${data.snapshot?.date || data.rows?.at(-1)?.date || '-'}`);
  } catch (error) {
    state.lastSnapshot = null;
    $('klineChart').innerHTML = `<div class="empty">K线加载失败：${esc(error.message)}</div>`;
    $('chartConclusion').style.display = 'none';
    setStatus(`K线加载失败：${error.message}`);
  } finally { setBusy('loadChartBtn',false); }
}

async function loadIntradayChart() {
  setBusy('loadIntradayBtn',true); setStatus('加载日内K线...');
  try {
    const sym=$('sSymbol')?.value.trim()||$('intraSymbol')?.value.trim()||state.lastKlineSymbol||'002714';
    const start=$('intraStart').value||'';
    const end=$('intraEnd').value||'';
    const period=state.intraPeriod;
    const qs=new URLSearchParams({symbol:sym,period,start,end});
    const data=await api(`/api/intraday?${qs.toString()}`);
    renderIntradayChart('intradayChart',data.rows||[],period);
    setStatus(`${sym} 日内K线已更新`);
  } catch (error) {
    $('intradayChart').innerHTML = `<div class="empty">日内K线加载失败：${esc(error.message)}</div>`;
    setStatus(`日内K线加载失败：${error.message}`);
  } finally { setBusy('loadIntradayBtn',false); }
}

function renderChartConclusion(s) {
  const el=$('chartConclusion');
  if(!s||!el) { if(el)el.style.display='none'; return; }
  const score=s.composite_score;
  let verdict, cls;
  if(score>=60){verdict='🔴 关注';cls='cb-buy';}else if(score<=35){verdict='🟢 回避';cls='cb-sell';}else{verdict='🟡 持有观察';cls='cb-hold';}
  let detail='';
  const row=(label,val,extra)=>`<span class="cb-item"><span class="cb-label">${label}</span> <span class="cb-val">${val}</span>${extra||''}</span>`;
  detail+=row('操作',`<span class="cb-badge ${cls}">${verdict}</span>`,'');
  detail+=row('综合',score);
  detail+=row('趋势',s.trend_score??'-');
  detail+=row('动量',s.momentum_score??'-');
  detail+=row('量能',s.volume_score??'-');
  detail+=row('风险',s.risk_score??'-');
  let warns=[];
  if((s.risk_score??0)>=60) warns.push('⚠️高风险');
  if((s.atr_pct??0)>5) warns.push('🔥高波动');
  if(s.rsi_14>80) warns.push('📈超买');
  if(s.rsi_14<20) warns.push('📉超卖');
  if(warns.length) detail+=`<span class="cb-item cb-warn">${warns.join(' ')}</span>`;
  el.style.display='block';
  $('conclusionBody').innerHTML = detail;
}

function renderRadarChart(s) {
  const el=$('radarChart'); if(!el||!s) return;
  const dims=[
    {name:'趋势',max:100,val:s.trend_score??50},
    {name:'动量',max:100,val:s.momentum_score??50},
    {name:'量能',max:100,val:s.volume_score??50},
    {name:'风险(逆)',max:100,val:s.risk_score!=null?100-s.risk_score:50},
    {name:'综合',max:100,val:s.composite_score??50},
  ];
  const option={
    radar:{
      center:['50%','50%'],radius:'65%',
      indicator:dims.map(d=>({name:d.name,max:d.max})),
      shape:'circle',
      name:{textStyle:{fontSize:10,color:'#657282'}},
      splitArea:{areaStyle:{color:['rgba(31,122,92,0.02)','rgba(31,122,92,0.06)']}},
      splitLine:{lineStyle:{color:'#d9dee7'}},
    },
    series:[{
      type:'radar',symbolSize:4,
      areaStyle:{color:'rgba(31,122,92,0.15)'},
      lineStyle:{color:'#1f7a5c',width:1.5},
      data:[dims.map(d=>d.val)],
    }],
    tooltip:{trigger:'item',formatter:p=>{const s=p.value||[];return p.name+'<br/>'+dims.map((d,i)=>`${d.name}: ${s[i]??'-'}`).join('<br/>');}}
  };
  if(!state.radarChart){state.radarChart=echarts.init(el);window.addEventListener('resize',()=>{if(state.radarChart)state.radarChart.resize();});}
  state.radarChart.setOption(option,true);
}

/* ════════════════════════════════════════════════════════════════
   ECharts 暗色主题统一助手
   将 option 中写死的浅色值统一覆盖为深色主题（背景/坐标轴/分割线/滑块/
   图例/tooltip/graphic 状态卡），涨跌色沿用 COLORS.up/down（红涨绿跌）。
   所有图表 setOption 前调用 applyChartDarkTheme(option) 即可。
   ════════════════════════════════════════════════════════════════ */
const DARK_CHART = {
  bg: '#0d1117', panel: '#161b22', border: '#30363d',
  text: '#e6edf3', muted: '#8b949e', gridLine: '#21262d',
};
function applyChartDarkTheme(option) {
  option.backgroundColor = DARK_CHART.bg;
  // tooltip：深色圆角 + 阴影（视觉层 视觉增强，仅视觉参数）
  if (option.tooltip && typeof option.tooltip === 'object') {
    Object.assign(option.tooltip, {
      backgroundColor: DARK_CHART.panel,
      borderColor: DARK_CHART.border,
      borderRadius: 8,
      padding: [8, 12],
      extraCssText: 'box-shadow: 0 10px 28px rgba(0,0,0,0.45);',
      textStyle: Object.assign({ color: DARK_CHART.text, fontSize: 12 }, option.tooltip.textStyle || {}),
    });
  }
  // legend（单个对象）
  if (option.legend && !Array.isArray(option.legend)) {
    option.legend.textStyle = Object.assign({ color: DARK_CHART.muted }, option.legend.textStyle || {});
  }
  // 坐标轴：标签亮色、轴线边框色、分割线暗色
  const fixAxes = (ax) => {
    if (!ax) return;
    if (ax.axisLabel) ax.axisLabel.color = DARK_CHART.muted;
    if (ax.axisLine && ax.axisLine.lineStyle) ax.axisLine.lineStyle.color = DARK_CHART.border;
    if (ax.splitLine && ax.splitLine.lineStyle) ax.splitLine.lineStyle.color = DARK_CHART.gridLine;
  };
  (Array.isArray(option.xAxis) ? option.xAxis : (option.xAxis ? [option.xAxis] : [])).forEach(fixAxes);
  (Array.isArray(option.yAxis) ? option.yAxis : (option.yAxis ? [option.yAxis] : [])).forEach(fixAxes);
  // dataZoom slider：accent 描边 + 手柄发光（视觉层）
  (option.dataZoom || []).forEach(z => {
    if (!z || z.type !== 'slider') return;
    z.borderColor = DARK_CHART.border;
    z.backgroundColor = DARK_CHART.panel;
    z.fillerColor = 'rgba(88,166,255,0.18)';
    z.textStyle = Object.assign({ color: DARK_CHART.muted }, z.textStyle || {});
    z.handleStyle = Object.assign({ color: '#58a6ff', borderColor: '#58a6ff' }, z.handleStyle || {});
    z.moveHandleStyle = Object.assign({ color: 'rgba(88,166,255,0.35)' }, z.moveHandleStyle || {});
  });
  // series：已有 areaStyle 的面积填充升级为纵向渐变（保留原色语义，视觉层）
  if (Array.isArray(option.series) && typeof echarts !== 'undefined' && echarts.graphic && echarts.graphic.LinearGradient) {
    option.series.forEach(s => {
      if (s && s.areaStyle && typeof s.areaStyle === 'object' && !s.areaStyle.colorStops) {
        const base = s.areaStyle.color;
        if (base && typeof base === 'string') {
          s.areaStyle.color = new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: base },
            { offset: 1, color: 'rgba(13,17,23,0)' },
          ]);
        }
      }
    });
  }
  // graphic：右上角状态卡白底→面板色、深色文字→亮色
  if (Array.isArray(option.graphic)) {
    option.graphic.forEach(g => {
      if (!g || !Array.isArray(g.children)) return;
      g.children.forEach(c => {
        if (c.type === 'rect' && c.style) {
          if (c.style.fill === '#ffffff') c.style.fill = DARK_CHART.panel;
          if (c.style.stroke === '#d1d5db') c.style.stroke = DARK_CHART.border;
        }
        if (c.type === 'text' && c.style && c.style.fill === '#17202a') c.style.fill = DARK_CHART.text;
      });
    });
  }
  return option;
}

function renderKlineChart(domId, snapshot, rows, period) {
  const dom = $(domId);
  if(!dom) return;
  if(!rows||rows.length<30){dom.innerHTML='<div class="empty" style="padding:60px;text-align:center;">数据不足，至少需要30个交易日</div>';return;}
  // 动图模式（plotly）：K线滑动动画/资金流动画（用户点名）
  const animEl = $('chartAnimate');
  if (animEl && animEl.checked && window.Plotly && rows.length >= 40) {
    renderKlinePlotly(dom, rows);
    return;
  }
  if(!state.klineChart){state.klineChart=echarts.init(dom);window.addEventListener('resize',()=>state.klineChart.resize());}

  const w=dom.clientWidth, gh=72, sr = [210, 60, 90, 80, 80], gap=15;
  const g=[]; let y=gh; for(let i=0;i<5;i++){g.push({left:50,right:22,top:y,height:sr[i]});y+=sr[i]+gap;}

  const dates=rows.map(r=>r.date);
  const kdata=rows.map(r=>[Number(r.open),Number(r.close),Number(r.low),Number(r.high)]);
  const vols=rows.map(r=>Number(r.volume)||0),maxVol=Math.max(...vols.filter(v=>v>0),1),volSuffix=maxVol>1e8?'亿':maxVol>1e4?'万':'',volUnit=maxVol>1e8?1e8:maxVol>1e4?1e4:1,volsNorm=vols.map(v=>v/volUnit);
  const ma20=rows.map(r=>r.ma_fast),ma60=rows.map(r=>r.ma_slow),ma144=rows.map(r=>r.ma_trend),ma300=rows.map(r=>r.ma_long_trend);
  const bollUpper=rows.map(r=>r.boll_upper),bollMid=rows.map(r=>r.boll_mid),bollLower=rows.map(r=>r.boll_lower);
  const macdDIF=rows.map(r=>r.macd_dif),macdDEA=rows.map(r=>r.macd_dea),macdHist=rows.map(r=>r.macd_hist);
  const rsi14=rows.map(r=>r.rsi_14),kdjK=rows.map(r=>r.kdj_k),kdjD=rows.map(r=>r.kdj_d),kdjJ=rows.map(r=>r.kdj_j);
  const score=snapshot?.composite_score??50;
  let verdict,verdictBg; if(score>=60){verdict='🔴 关注';verdictBg='#b42318';}else if(score<=35){verdict='🟢 回避';verdictBg='#087443';}else{verdict='🟡 持有观察';verdictBg='#92400e';}
  const warns=[]; if((snapshot?.risk_score??0)>=60)warns.push('⚠️高险');if((snapshot?.atr_pct??0)>5)warns.push('🔥高波');

  const option = {
    backgroundColor:'#0d1117', animation:false,
    tooltip:{trigger:'axis',axisPointer:{type:'cross'},confine:true,
      formatter:function(params){
        const p=params.find(x=>x.seriesName==='K线');
        if(!p||p.dataIndex==null)return '';
        const i=p.dataIndex,r=rows[i];
        if(!r)return '';
        let s=`<b>${r.date}</b><br/>开 ${fmt(r.open)} 收 ${fmt(r.close)} 高 ${fmt(r.high)} 低 ${fmt(r.low)}<br/>`;
        if(r.pct_chg!=null)s+=`涨跌 ${fmt(r.pct_chg)}%<br/>`;
        if(r.volume_ratio!=null)s+=`量比 ${fmt(r.volume_ratio)}<br/>`;
        params.filter(p2=>p2.seriesName&&p2.seriesName.startsWith('MA')).forEach(p2=>s+=`${p2.marker} ${p2.seriesName} ${p2.value}<br/>`);
        params.filter(p2=>p2.seriesName==='BOLL中').forEach(p2=>s+=`${p2.marker} BOLL中 ${p2.value}<br/>`);
        s+=`<div style="border-top:1px solid #d1d5db;margin:4px 0;padding-top:4px;">`;
        if(r.amount!=null){const amt=r.amount/10000;s+=`成交额 ${amt.toFixed(1)} 亿<br/>`;}
        if(r.rsi_14!=null)s+=`RSI(14) ${fmt2(r.rsi_14)}<br/>`;
        if(r.macd_dif!=null||r.macd_dea!=null||r.macd_hist!=null)s+=`MACD DIF ${fmt2(r.macd_dif)}  DEA ${fmt2(r.macd_dea)}  柱 ${fmt2(r.macd_hist)}<br/>`;
        if(r.kdj_k!=null)s+=`KDJ-K ${fmt2(r.kdj_k)}  D ${fmt2(r.kdj_d)}  J ${fmt2(r.kdj_j)}<br/>`;
        if(r.boll_upper!=null)s+=`BOLL上 ${fmt2(r.boll_upper)} 中 ${fmt2(r.boll_mid)} 下 ${fmt2(r.boll_lower)}<br/>`;
        s+=`</div>`;
        if(r.patterns){const pl=r.patterns.split(',').filter(Boolean).map(p=>patternLabel(p)).filter(Boolean);if(pl.length)s+=`<div style="border-top:1px solid #d1d5db;margin:4px 0;padding-top:4px;color:#8b5cf6;font-weight:600;">${pl.join(' · ')}</div>`;}
        return s;
      },
    },
    legend:{data:['K线','MA20','MA60','MA144','MA300','BOLL','成交量','MACD','DEA','RSI','KDJ-K','KDJ-D','KDJ-J'],top:2,left:60,textStyle:{fontSize:11},selected:{'KDJ-J':true,'BOLL':false}},
    grid: g,
    xAxis:[
      {type:'category',data:dates,gridIndex:0,axisLine:{onZero:false},axisLabel:{show:false},axisTick:{show:false},splitLine:{show:true,lineStyle:{color:'#d1d5db',width:0.8}}},
      {type:'category',data:dates,gridIndex:1,axisLabel:{show:false},axisTick:{show:false},splitLine:{show:false}},
      {type:'category',data:dates,gridIndex:2,axisLabel:{show:false},axisTick:{show:false},splitLine:{show:false}},
      {type:'category',data:dates,gridIndex:3,axisLabel:{show:false},axisTick:{show:false},splitLine:{show:false}},
      {type:'category',data:dates,gridIndex:4,axisLabel:{rotate:45,fontSize:9,interval:Math.max(1,Math.floor(dates.length/20))},axisTick:{show:false},splitLine:{show:false}},
    ],
    yAxis:[
      {scale:true,gridIndex:0,splitNumber:4,axisLabel:{fontSize:10},splitLine:{lineStyle:{color:'#d1d5db',width:0.8}}},
      {scale:true,gridIndex:1,splitNumber:2,axisLabel:{fontSize:9,formatter:v=>v>=1?v+(volSuffix||''):v}},
      {scale:true,gridIndex:2,splitNumber:3,axisLabel:{fontSize:9}},
      {scale:true,gridIndex:3,min:0,max:100,splitNumber:3,axisLabel:{fontSize:9,formatter:'{value}'}},
      {scale:true,gridIndex:4,splitNumber:3,axisLabel:{fontSize:9}},
    ],
    dataZoom:[
      {type:'inside',xAxisIndex:[0,1,2,3,4],start:Math.max(0,100-Math.floor(120/dates.length*100)),end:100,minSpan:5},
      {type:'slider',xAxisIndex:[0,1,2,3,4],show:true,height:12,bottom:2,borderColor:'#d1d5db',fillerColor:'rgba(31,122,92,0.08)',handleStyle:{color:'#1f7a5c'},start:Math.max(0,100-Math.floor(120/dates.length*100)),end:100,minSpan:5},
    ],
    series:[
      {name:'K线',type:'candlestick',xAxisIndex:0,yAxisIndex:0,data:kdata,itemStyle:{color:COLORS.up,color0:COLORS.down,borderColor:COLORS.up,borderColor0:COLORS.down},
        markLine:buildMarkLines(snapshot,dates),markPoint:buildMarkPoints(snapshot,rows,dates)},
      {name:'MA20',type:'line',xAxisIndex:0,yAxisIndex:0,symbol:'none',smooth:true,lineStyle:{color:COLORS.ma20,width:1.2},data:ma20},
      {name:'MA60',type:'line',xAxisIndex:0,yAxisIndex:0,symbol:'none',smooth:true,lineStyle:{color:COLORS.ma60,width:1.2},data:ma60},
      {name:'MA144',type:'line',xAxisIndex:0,yAxisIndex:0,symbol:'none',smooth:true,lineStyle:{color:COLORS.ma144,width:1.2,type:'dashed'},data:ma144},
      {name:'MA300',type:'line',xAxisIndex:0,yAxisIndex:0,symbol:'none',smooth:true,lineStyle:{color:COLORS.ma300,width:1.2,type:'dashed'},data:ma300},
      // BOLL upper/lower shade (stack trick for translucent band)
      {name:'BOLL',type:'line',xAxisIndex:0,yAxisIndex:0,symbol:'none',lineStyle:{width:0},data:bollUpper,stack:'boll',z:1},
      {name:'BOLL',type:'line',xAxisIndex:0,yAxisIndex:0,symbol:'none',lineStyle:{width:0},data:bollLower,stack:'boll',areaStyle:{color:COLORS.bollBand},z:1},
      {name:'BOLL中',type:'line',xAxisIndex:0,yAxisIndex:0,symbol:'none',smooth:true,lineStyle:{color:COLORS.boll,width:1,type:'dashed'},data:bollMid,z:1},
      {name:'成交量',type:'bar',xAxisIndex:1,yAxisIndex:1,data:rows.map((r,i)=>({value:volsNorm[i],itemStyle:{color:Number(r.close)>=Number(r.open)?COLORS.up:COLORS.down}}))},
      {name:'MACD',type:'bar',xAxisIndex:2,yAxisIndex:2,data:macdHist.map(v=>({value:v,itemStyle:{color:v>=0?'#dc2626':'#16a34a'}}))},
      {name:'MACD',type:'line',xAxisIndex:2,yAxisIndex:2,symbol:'none',smooth:true,lineStyle:{color:COLORS.macd,width:1.2},data:macdDIF},
      {name:'DEA',type:'line',xAxisIndex:2,yAxisIndex:2,symbol:'none',smooth:true,lineStyle:{color:COLORS.macdSignal,width:1.2},data:macdDEA},
      {name:'RSI',type:'line',xAxisIndex:3,yAxisIndex:3,symbol:'none',smooth:true,lineStyle:{color:COLORS.rsi,width:1.2},data:rsi14,
        markLine:{silent:true,data:[{yAxis:80,label:{formatter:'超买 80',fontSize:9,color:'#b42318'},lineStyle:{color:'#fca5a5',type:'dashed',width:0.8}},{yAxis:50,label:{show:false},lineStyle:{color:'#d1d5db',type:'dotted',width:0.5}},{yAxis:20,label:{formatter:'超卖 20',fontSize:9,color:'#087443'},lineStyle:{color:'#bbf7d0',type:'dashed',width:0.8}}]}},
      {name:'KDJ-K',type:'line',xAxisIndex:4,yAxisIndex:4,symbol:'none',smooth:true,lineStyle:{color:COLORS.k,width:1.2},data:kdjK},
      {name:'KDJ-D',type:'line',xAxisIndex:4,yAxisIndex:4,symbol:'none',smooth:true,lineStyle:{color:COLORS.d,width:1.2,type:'dashed'},data:kdjD},
      {name:'KDJ-J',type:'line',xAxisIndex:4,yAxisIndex:4,symbol:'none',smooth:true,lineStyle:{color:COLORS.j,width:1},data:kdjJ},
      {name:'_c',type:'line',xAxisIndex:0,yAxisIndex:0,data:[],symbol:'none',lineStyle:{width:0},tooltip:{formatter:()=>`<b>${verdict}</b><br/>${snapshot?.symbol||''} ${snapshot?.close?fmt(snapshot.close):''} 综${score} 趋${snapshot?.trend_score??'-'} 动${snapshot?.momentum_score??'-'} 量${snapshot?.volume_score??'-'} 险${snapshot?.risk_score??'-'}`}},
    ],
    graphic: [
      {type:'group',right:8,top:6,bounding:'raw',
        children:[
          {type:'rect',shape:{width:164,height:warns.length?62:48,r:4},style:{fill:'#ffffff',stroke:'#d1d5db',lineWidth:0.5,shadowBlur:6,shadowColor:'rgba(0,0,0,0.08)'}},
          {type:'rect',shape:{width:164,height:18,r:[4,4,0,0]},style:{fill:verdictBg}},
          {type:'text',left:8,top:1,style:{text:verdict,fill:'#ffffff',fontSize:11,fontWeight:700}},
          {type:'text',left:8,top:22,style:{text:`${snapshot?.symbol||''} ${snapshot?.close?fmt(snapshot.close):''}`,fill:'#17202a',fontSize:12,fontWeight:600}},
          {type:'text',left:8,top:37,style:{text:`综${score} 趋${snapshot?.trend_score??'-'} 动${snapshot?.momentum_score??'-'}`,fill:'#657282',fontSize:10}},
          ...(warns.length?[{type:'text',left:8,top:51,style:{text:warns.join(' '),fill:'#b42318',fontSize:10,fontWeight:600}}]:[]),
        ],
      },
      ...(()=>{const lines=[];const x2=w-24;for(let i=0;i<4;i++){const yy=g[i].top+g[i].height;lines.push({type:'line',shape:{x1:50,y1:yy,x2:x2,y2:yy},style:{stroke:'#9ca3af',lineWidth:1}});}return lines;})(),
    ],
  };
  state.klineChart.setOption(applyChartDarkTheme(option), true);
  // K线滑动窗口动图（dataZoom + 播放控制）
  if(!option.dataZoom && rows.length > 120){
    const win = Math.min(120, Math.floor(rows.length * 0.8));
    state.klineChart.setOption({
      dataZoom: [
        {type:'inside', xAxisIndex:0, start:100*(rows.length-win)/rows.length, end:100},
        {type:'slider', xAxisIndex:0, start:100*(rows.length-win)/rows.length, end:100, height:18, bottom:4,
         borderColor:'#d1d5db', backgroundColor:'#f9fafb', fillerColor:'rgba(59,130,246,0.12)',
         textStyle:{fontSize:10}}
      ],
      animation:true, animationDurationUpdate:400, animationEasingUpdate:'linear'
    });
    // 自动播放（若未初始化过）
    if(!state.klinePlaying){
      state.klinePlaying = true;
      const total = rows.length;
      let pos = 0;
      state.klineTimer && clearInterval(state.klineTimer);
      state.klineTimer = setInterval(()=>{
        pos = (pos + 10) % (total - win + 10);
        const s = 100 * pos / total, e = 100 * (pos + win) / total;
        state.klineChart && state.klineChart.dispatchAction({type:'dataZoom', start:s, end:e});
        if(pos >= total - win) pos = 0;
      }, 600);
      // 用户交互时停止自动播放
      state.klineChart.on('dataZoom', ()=>{ state.klineTimer && clearInterval(state.klineTimer); state.klineTimer=null; });
    }
  }
}

// plotly K线滑动动画（用户点名：K线滑动动画/资金流动画）
function renderKlinePlotly(dom, rows) {
  const dates = rows.map(r => r.date);
  const ohlc = { x: dates, open: rows.map(r => Number(r.open)), high: rows.map(r => Number(r.high)),
                 low: rows.map(r => Number(r.low)), close: rows.map(r => Number(r.close)) };
  const vol = rows.map(r => Number(r.volume) || 0);
  const win = Math.min(80, Math.floor(rows.length * 0.7));
  // 帧：滑动窗口（每帧推进 5 天），形成“K线播放”动图
  const frames = [];
  for (let s = 0; s + win <= rows.length; s += 5) {
    frames.push({ name: 'f' + s, data: [{ type: 'candlestick', x: dates.slice(s, s + win), open: ohlc.open.slice(s, s + win), high: ohlc.high.slice(s, s + win), low: ohlc.low.slice(s, s + win), close: ohlc.close.slice(s, s + win) }],
                  layout: { xaxis: { range: [s - 2, s + win + 2] } } });
  }
  const last = frames.length - 1;
  const traceCandle = { type: 'candlestick', name: 'K线', x: dates, open: ohlc.open, high: ohlc.high, low: ohlc.low, close: ohlc.close,
                        increasing: { line: { color: '#f6465d' } }, decreasing: { line: { color: '#2ebd85' } } };
  const traceVol = { type: 'bar', name: '成交量', x: dates, y: vol, marker: { color: vol.map((v, i) => (ohlc.close[i] >= ohlc.open[i] ? '#f6465d' : '#2ebd85')) }, yaxis: 'y2' };
  // 统一实现: Plotly 深色主题（与 echarts 暗色一致，避免白底割裂）
  const axisDark = { gridcolor: '#21262d', zerolinecolor: '#30363d', tickfont: { color: '#8b949e', size: 10 } };
  const layout = {
    title: { text: `${rows[rows.length-1]?.symbol || ''} K线动图（窗口${win}日）`, font: { size: 13, color: '#e6edf3' } },
    paper_bgcolor: '#0d1117', plot_bgcolor: '#161b22', font: { color: '#8b949e' },
    dragmode: 'pan', margin: { t: 36, r: 16, b: 40, l: 46 },
    xaxis: Object.assign({ rangeslider: { visible: false }, type: 'category' }, axisDark),
    yaxis: Object.assign({ domain: [0.32, 1], title: { text: '价格', font: { color: '#8b949e' } } }, axisDark),
    yaxis2: Object.assign({ domain: [0, 0.24], title: { text: '量', font: { color: '#8b949e' } }, showgrid: false }, axisDark),
    showlegend: false,
    updatemenus: [{ type: 'buttons', x: 0.02, y: 1.14, buttons: [
      { label: '▶ 播放', method: 'animate', args: [null, { frame: { duration: 120, redraw: false }, transition: { duration: 60 }, fromcurrent: true, mode: 'immediate' }] },
      { label: '⏸ 暂停', method: 'animate', args: [[null], { frame: { duration: 0, redraw: false }, mode: 'immediate' }] }
    ]}],
    sliders: [{ pad: { t: 34 }, x: 0.05, len: 0.9, currentvalue: { prefix: '窗口起点: ', font: { size: 11, color: '#8b949e' } },
      steps: frames.map(f => ({ label: '', method: 'animate', args: [[f.name], { mode: 'immediate', frame: { duration: 0, redraw: false } }] })) }]
  };
  Plotly.newPlot(dom, [traceCandle, traceVol], layout, { responsive: true, displaylogo: false })
    .then(() => Plotly.addFrames(dom, frames));
}

function buildMarkLines(snapshot,dates){const lines=[];if(snapshot?.stop_loss)lines.push({yAxis:snapshot.stop_loss,label:{formatter:'止损 '+snapshot.stop_loss,position:'insideEndTop',fontSize:9,color:'#b42318'},lineStyle:{color:'#b42318',type:'dashed',width:0.8}});if(snapshot?.take_profit_ref)lines.push({yAxis:snapshot.take_profit_ref,label:{formatter:'止盈 '+snapshot.take_profit_ref,position:'insideEndTop',fontSize:9,color:'#087443'},lineStyle:{color:'#087443',type:'dashed',width:0.8}});return{silent:true,data:lines};}
function buildMarkPoints(snapshot,rows,dates){
  const pts=[];
  if(snapshot&&rows.length>=2){const score=snapshot.composite_score;if(score>=60)pts.push({coord:[dates[dates.length-1],rows[rows.length-1].high],symbol:'arrow',symbolSize:24,itemStyle:{color:'#b42318'},label:{formatter:'关注',fontSize:9,color:'#b42318',position:'top'}});else if(score<=35)pts.push({coord:[dates[dates.length-1],rows[rows.length-1].high],symbol:'pin',symbolSize:24,itemStyle:{color:'#087443'},label:{formatter:'回避',fontSize:9,color:'#087443',position:'top'}});}
  const SYM={bullish_engulf:['pin','#b42318'],bearish_engulf:['pin','#087443'],morning_star:['diamond','#b42318'],evening_star:['diamond','#087443'],piercing:['arrow','#b42318'],dark_cloud:['arrow','#087443'],three_white:['rect','#b42318'],three_black:['rect','#087443'],doji:['circle','#9ca3af'],hammer:['pin','#b42318'],shooting_star:['pin','#087443'],harami:['circle','#657282']};
  for(let i=0;i<rows.length;i++){const p=rows[i]?.patterns;if(!p)continue;p.split(',').filter(Boolean).forEach(pat=>{const cfg=SYM[pat];if(!cfg)return;pts.push({coord:[dates[i],rows[i].high||rows[i].close],symbol:cfg[0],symbolSize:16,itemStyle:{color:cfg[1]},label:{show:false}});});}
  return{silent:true,data:pts};
}
function patternLabel(p){const m={doji:'十字星',hammer:'锤子线',shooting_star:'流星/倒锤子',bullish_engulf:'看涨吞没',bearish_engulf:'看跌吞没',morning_star:'启明星',evening_star:'黄昏星',three_white:'三白兵',three_black:'三黑鸦',piercing:'刺透线',dark_cloud:'乌云盖顶',harami:'孕育形态'};return m[p]||p;}

function renderSnapshot(row){const labels={close:'收盘',pct_chg:'涨跌%',pct_20:'20期涨跌%',ma_fast:'MA快',ma_slow:'MA慢',ma_trend:'MA趋势',ma_long_trend:'MA长期',macd_hist:'MACD柱',rsi_14:'RSI14',kdj_j:'KDJ-J',boll_width_pct:'BOLL宽度%',atr_pct:'ATR%',volume_ratio:'量比',drawdown_60_pct:'60期回撤%',trend_score:'趋势分',momentum_score:'动量分',volume_score:'量能分',risk_score:'风险分',composite_score:'综合分'};const el=$('snapshotMetrics');if(!el)return;el.innerHTML=`<div class="snapshot">${Object.entries(labels).map(([k,l])=>`<div class="kv"><span>${l}</span><strong>${fmt(row?.[k])}</strong></div>`).join('')}</div>`;}

async function runBacktest() {
  setBusy('backtestBtn',true); setStatus('运行回测...');
  try{
    const p=readParams();
    const data=await api('/api/backtest',Object.assign({symbols:$('btSymbols').value,start:$('btStart').value,refresh:$('btRefresh').checked}, p));
    state.lastBacktest = data.rows||[];
    // 单票回测 -> 展示详细报告
    if(data.rows&&data.rows.length===1){
      const r=data.rows[0];
      if(r.trades&&r.equity_curve){
        renderBacktestDetail(r);
        $('btOldTable').style.display='none';
      }else{
        $('btSummary').style.display='none';
        $('btOldTable').style.display='block';
        renderTable('backtestTable',data.rows,backtestColumns());
      }
    }else{
      $('btSummary').style.display='none';
      $('btOldTable').style.display='block';
      renderTable('backtestTable',data.rows,backtestColumns());
    }
    $('btActions').style.display = data.rows&&data.rows.length?'block':'none';
    setStatus(`回测完成：${data.rows?.length || 0} 个结果`);
  } catch (error) {
    state.lastBacktest = [];
    $('btSummary').style.display='none'; $('btActions').style.display='none'; $('btOldTable').style.display='block';
    renderTable('backtestTable', [], backtestColumns());
    setStatus(`回测失败：${error.message}`);
  } finally { setBusy('backtestBtn',false); }
}

async function runPortfolioBacktest() {
  setBusy('backtestBtn',true); setStatus('运行组合回测...');
  try{
    const p=readParams();
    const data=await api('/api/backtest',Object.assign({symbols:$('btSymbols').value,start:$('btStart').value,refresh:$('btRefresh').checked,portfolio_mode:true}, p));
    if(data.results && data.summary){
      state.lastBacktest = data.results;
      renderPortfolioBacktest(data);
      $('btActions').style.display = 'block';
    }else{
      state.lastBacktest = data.rows||[];
      $('btSummary').style.display='none';
      $('btOldTable').style.display='block';
      renderTable('backtestTable',data.rows,backtestColumns());
    }
    setStatus('组合回测完成');
  } catch (error) {
    state.lastBacktest = [];
    $('btSummary').style.display='none'; $('btActions').style.display='none';
    renderTable('backtestTable', [], backtestColumns());
    setStatus(`组合回测失败：${error.message}`);
  } finally { setBusy('backtestBtn',false); }
}

function renderBacktestDetail(r) {
  const el=$('btSummary'); el.style.display='block';
  const eq=r.equity_curve||[];
  const bm=r.benchmark_metrics||{};
  // 中英文指标名
  const metrics=[
    ['总收益',pct(r.total_return_pct),Number(r.total_return_pct)>0?'good':'bad'],
    ['年化收益',pct(r.annual_return_pct),Number(r.annual_return_pct)>0?'good':'bad'],
    ['夏普比率',fmt(r.sharpe),(r.sharpe||0)>1?'good':(r.sharpe||0)>0?'':'bad'],
    ['索提诺',fmt(r.sortino_ratio),(r.sortino_ratio||0)>1?'good':''],
    ['卡玛比率',fmt(r.calmar_ratio),(r.calmar_ratio||0)>1?'good':''],
    ['最大回撤',pct(r.max_drawdown_pct),'bad'],
    ['交易次数',r.trade_count??0,''],
    ['胜率',pct(r.win_rate_pct),(r.win_rate_pct||0)>50?'good':'bad'],
    ['盈亏比',fmt(r.profit_factor),(r.profit_factor||0)>1.5?'good':''],
    ['平均盈利',pct(r.avg_win_pct),(r.avg_win_pct||0)>5?'good':''],
    ['平均亏损',pct(r.avg_loss_pct),''],
    ['最终权益',fmt(r.final_equity),''],
    ['持仓',r.open_shares??0,''],
  ];
  if (bm.alpha != null) metrics.push(['Alpha(年化)', (bm.alpha*100).toFixed(1)+'%', bm.alpha>0?'good':'bad']);
  if (bm.beta != null) metrics.push(['Beta', fmt(bm.beta), '']);
  if (bm.information_ratio != null) metrics.push(['信息比率', fmt(bm.information_ratio), (bm.information_ratio||0)>0.5?'good':'']);
  if (bm.excess_sharpe != null) metrics.push(['超额夏普', fmt(bm.excess_sharpe), (bm.excess_sharpe||0)>0?'good':'bad']);
  if (bm.benchmark_sharpe != null) metrics.push(['基准夏普(300)', fmt(bm.benchmark_sharpe), '']);
  $('btMetrics').innerHTML = metrics.map(([l,v,c])=>`<div class="metric"><div class="label">${esc(l)}</div><div class="value ${c}">${esc(v)}</div></div>`).join('');

  // 权益曲线 + 基准双线 + 回撤副图
  if(eq.length>1 && $('btEquityChart')){
    const chart=echarts.init($('btEquityChart'));
    const dates=eq.map(e=>e.date);
    const bench=r.benchmark_curve||[];
    // 基准对齐到交易日(以初始权益为1)
    const eq0=eq[0].equity||1;
    const benchSeries=bench.length?bench.map(b=>{ const i=dates.indexOf(b.date); return i>=0?{value:[i,eq0*b.nav]}:null; }).filter(Boolean):[];
    const dd=r.drawdown_curve||[];
    chart.setOption({
      tooltip:{trigger:'axis',formatter:ps=>{
        if(!ps||!ps[0])return'';const i=ps[0].dataIndex;
        let h=`<b>${eq[i]?.date||''}</b><br/>权益 ${fmt(eq[i]?.equity)}`;
        if(dd[i]!=null) h+=`<br/>回撤 ${dd[i]}%`;
        if(benchSeries.length){ const m=ps.find(s=>s.seriesName==='沪深300'); if(m&&m.value) h+=`<br/>沪深300 ${fmt(m.value[1])}`; }
        return h;
      }},
      legend:{data:['策略','沪深300'],top:0,textStyle:{color:'#8b949e',fontSize:11}},
      grid:[{left:55,right:16,top:24,height:'62%'},{left:55,right:16,top:'76%',height:'18%'}],
      xAxis:[
        {type:'category',data:dates,axisLabel:{rotate:45,fontSize:9,interval:Math.max(1,Math.floor(eq.length/15))},gridIndex:0},
        {type:'category',data:dates,axisLabel:{show:false},gridIndex:1},
      ],
      yAxis:[
        {type:'value',scale:true,splitLine:{lineStyle:{color:'#d1d5db'}},axisLabel:{formatter:v=>fmt(v)},gridIndex:0},
        {type:'value',scale:true,axisLabel:{formatter:v=>v+'%',fontSize:9},splitLine:{show:false},gridIndex:1},
      ],
      dataZoom:[{type:'inside',xAxisIndex:[0,1]}],
      series:[
        {name:'策略',type:'line',data:eq.map(e=>e.equity),smooth:true,
          lineStyle:{color:'#1f7a5c',width:2},areaStyle:{color:'rgba(31,122,92,0.1)'},
          markPoint:{data:[
            {coord:[eq[0]?.date,eq[0]?.equity],symbol:'circle',symbolSize:8,itemStyle:{color:'#1f7a5c'},label:{formatter:'起始',fontSize:9,position:'bottom'}},
            {coord:[eq[eq.length-1]?.date,eq[eq.length-1]?.equity],symbol:'diamond',symbolSize:12,itemStyle:{color:Number(r.total_return_pct)>0?'#b42318':'#087443'},label:{formatter:'结束',fontSize:9,position:'top'}},
          ]}},
        {name:'沪深300',type:'line',data:benchSeries,smooth:true,
          lineStyle:{color:'#2563eb',width:1.5,type:'dashed'},showSymbol:false},
        {name:'回撤',type:'line',data:dd,smooth:true,xAxisIndex:1,yAxisIndex:1,
          lineStyle:{color:'#b45309',width:1},areaStyle:{color:'rgba(180,83,9,0.12)'},showSymbol:false},
      ],
    });
    window.addEventListener('resize',()=>chart.resize());
  }
  // 月度收益热力图
  if($('btMonthlyChart')){
    const mr=r.monthly_returns||[];
    if(mr.length){
      const mc=echarts.init($('btMonthlyChart'));
      const maxAbs=Math.max(...mr.map(m=>Math.abs(m.ret)),1);
      mc.setOption({
        tooltip:{formatter:p=>`<b>${p.data[0]}</b><br/>收益 ${p.data[2]}%`},
        grid:{left:40,right:10,top:10,bottom:20},
        xAxis:{type:'category',data:mr.map(m=>m.month),axisLabel:{rotate:45,fontSize:9,interval:Math.max(0,Math.floor(mr.length/10)-1)}},
        yAxis:{type:'category',data:['收益%']},
        visualMap:{min:-maxAbs,max:maxAbs,calculable:false,orient:'horizontal',left:'center',bottom:0,
          inRange:{color:['#087443','#1f2937','#b42318']},textStyle:{fontSize:9}},
        series:[{type:'heatmap',data:mr.map((m,i)=>[i,0,m.ret]),
          label:{show:true,formatter:p=>p.value[2].toFixed(1),fontSize:9,color:'#e6edf3'}}],
      });
      window.addEventListener('resize',()=>mc.resize());
    }
  }
  // 净值曲线与回撤
  document.dispatchEvent(new CustomEvent('visual:backtest', { detail: r }));

  // 逐笔交易
  const trades=r.trades||[];
  if(trades.length){
    const cols=['date','side','price','shares','pnl'];
    renderTableByEl($('btTradeTable'),trades,cols);
  }
}

function renderPortfolioBacktest(data) {
  const el=$('btSummary'); el.style.display='block';
  const s=data.summary;
  const metrics=[['组合总收益',pct(s.total_return_pct),Number(s.total_return_pct)>0?'good':'bad'],['年化收益',pct(s.annual_return_pct),''],
    ['夏普',fmt(s.sharpe),(s.sharpe||0)>1?'good':''],['最大回撤',pct(s.max_drawdown_pct),'bad'],['最终权益',fmt(s.final_equity),''],['持仓票数',s.open_symbols??0,'']];
  $('btMetrics').innerHTML = metrics.map(([l,v,c])=>`<div class="metric"><div class="label">${esc(l)}</div><div class="value ${c}">${esc(v)}</div></div>`).join('');
  $('btOldTable').style.display='block';
  const rows = Object.values(data.results).map(r=>({symbol:r.symbol||'?',status:r.status,start:r.start,end:r.end,
    total_return_pct:r.total_return_pct,max_drawdown_pct:r.max_drawdown_pct,sharpe:r.sharpe,trade_count:r.trade_count,final_equity:r.final_equity}));
  renderTable('backtestTable',rows,backtestColumns());
}

async function runOptimize() {
  setBusy('optimizeBtn',true); setStatus('运行网格搜索...');
  try{
    const parse=(id)=>$(id).value.split(/[,;，；\s]+/).filter(Boolean).map(Number).filter(n=>!isNaN(n));
    const p=readParams();
    const body={
      symbol:$('optSymbol').value,start:$('optStart').value,refresh:$('optRefresh').checked,
      strategy:p.strategy,portfolio:p.portfolio,
      param_grid:{
        fast_ma:parse('opt_fast'),slow_ma:parse('opt_slow'),
        stop_loss_pct:parse('opt_stop'),take_profit_pct:parse('opt_take'),
      },
      sort_by:$('optSortBy').value,
    };
    const data=await api('/api/optimize',body);
    const rows=data.results||[];
    if(!rows.length){setStatus('无优化结果');return;}
    const cols=['rank','params','total_return_pct','sharpe','max_drawdown_pct','win_rate_pct','trade_count'];
    renderTable('optimizeTable',rows.map((r,i)=>({
      rank:i+1,
      params:Object.entries(r.params).map(([k,v])=>`${shortKey(k)}=${v}`).join(' '),
      total_return_pct:r.result?.total_return_pct,
      sharpe:r.result?.sharpe,
      max_drawdown_pct:r.result?.max_drawdown_pct,
      win_rate_pct:r.result?.win_rate_pct,
      trade_count:r.result?.trade_count,
    })),cols);
    $('optResults').style.display='block';
    setStatus('优化完成，共'+rows.length+'组');
  }finally{setBusy('optimizeBtn',false);}
}

// ════════════════════════════════════════
// 实时监控
// ════════════════════════════════════════

// 注意：rtRefreshBtn / rtStartMonitorBtn 已在顶部统一绑定

let legacyRtTimer = null;
let legacyMonitorRunning = false;

// 停止监控在离开实时tab时
function stopMonitor() { if (legacyRtTimer) { clearInterval(legacyRtTimer); legacyRtTimer = null; } legacyMonitorRunning = false; const btn = $('rtStartMonitorBtn'); if (btn) { btn.textContent = '启动后台监控'; btn.style.background = '#1f7a5c'; } }

// 统一实现: 已删除被覆盖的死代码版本 loadRealtime()/toggleMonitor()（早期实现，后被 rtTimer 版本覆盖）
async function loadSectorView() {
  setBusy('sectorLoadBtn',true); setStatus('加载板块分析...');
  try{
    const data=await api(`/api/sector?symbols=${encodeURIComponent($('sectorSymbols').value)}`);
    const sect=data.sectors||{};
    const names=data.sector_names||{};
    const note=data.note||'';
    const entries=Object.entries(sect);
    if(!entries.length){setStatus('无板块数据');return;}
    if(note) setStatus('⚠️ '+note);
    // 板块分布饼图
    if($('sectorChart')){
      const chart=echarts.init($('sectorChart'));
      chart.setOption({
        tooltip:{trigger:'item',formatter:'{b}: {c}只 ({d}%)'},
        series:[{
          type:'pie',radius:['20%','50%'],center:['35%','50%'],
          data:entries.map(([k,v])=>({name:names[k]||k,value:v.length})),
          label:{fontSize:10},emphasis:{label:{fontSize:12}},
        }],
      });
      window.addEventListener('resize',()=>chart.resize());
    }
    // 表格
    const rows=[];
    entries.forEach(([code,list])=>{
      const name=names[code]||code;
      (list||[]).forEach(s=>{rows.push({sector_name:name,sector_code:code,symbol:s});});
    });
    renderTable('sectorTable',rows,['sector_name','symbol']);
    $('sectorView').style.display='block';
    setStatus('就绪');
  }finally{setBusy('sectorLoadBtn',false);}
}

function exportCsv(type, rows) {
  if(!rows||!rows.length){setStatus('没有数据可导出');return;}
  const cols=type==='scan'?scanColumns():['symbol','date','status','total_return_pct','max_drawdown_pct','sharpe','trade_count','win_rate_pct','final_equity'];
  const header=cols.map(c=>hLabel(c)).join(',');
  const body=rows.map(r=>cols.map(c=>{
    const v=r[c]; return v==null?'':String(v).includes(',')?`"${v}"`:v;
  }).join(',')).join('\n');
  const blob=new Blob(['\ufeff'+header+'\n'+body],{type:'text/csv;charset=utf-8'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download=`${type}_${new Date().toISOString().slice(0,10)}.csv`;a.click();
  URL.revokeObjectURL(a.href);
  setStatus('CSV已导出');
}

async function sendAlert(type,rows){if(!rows||!rows.length){setStatus('没有数据可预警');return;}setStatus('发送预警到飞书...');try{const data=await api('/api/alert',{type,rows:rows.slice(0,20),market:state.market});setStatus(data.message||'预警已发送');}catch(err){setStatus('预警发送失败: '+err.message);}}

function focusChatInput(){ const input=$('chatInput'); if(input){ input.focus(); input.scrollIntoView({behavior:'smooth', block:'center'}); } }
async function askOpenClaw(){const message=$('chatInput').value.trim();if(!message)return;appendMsg('user',message);$('chatInput').value='';setBusy('askBtn',true);appendMsg('bot','OpenClaw 正在思考...');try{const data=await api('/api/chat',{message,model:$('modelInput').value.trim(),context:{market:state.market,lastScan:state.lastScan.slice(0,20),lastSnapshot:state.lastSnapshot,lastBacktest:state.lastBacktest,params:readParams()}});document.querySelector('.msg.bot:last-child').textContent=data.reply?.text||JSON.stringify(data.reply,null,2);}catch(err){document.querySelector('.msg.bot:last-child').textContent=`调用失败：${err.message}`;}finally{setBusy('askBtn',false);}}

const _tables={};
const TABLE_PAGE_SIZE=50;
function renderTable(id,rows,columns){const el=$(id);if(!el)return;if(!rows||!rows.length){el.innerHTML='<tbody><tr><td>暂无数据</td></tr></tbody>';return;}_tables[id]={rows,columns,sortKey:null,sortAsc:true,page:1,pageSize:TABLE_PAGE_SIZE};_renderTable(id);}
function _renderTable(id){const t=_tables[id];if(!t)return;const{rows,columns,sortKey,sortAsc,pageSize}=t;const sorted=[...rows];if(sortKey)sorted.sort((a,b)=>{const va=a[sortKey],vb=b[sortKey];if(va==null)return 1;if(vb==null)return -1;return(typeof va==='number'?va-vb:String(va).localeCompare(String(vb)))*(sortAsc?1:-1);});const pages=Math.max(1,Math.ceil(sorted.length/pageSize));t.page=Math.min(Math.max(1,t.page||1),pages);const start=(t.page-1)*pageSize;const visible=sorted.slice(start,start+pageSize);const el=$(id);el.innerHTML=`<thead><tr>${columns.map(c=>{const arrow=sortKey===c?(sortAsc?' ▲':' ▼'):'';return`<th data-col="${esc(c)}" onclick="window.sortTable('${id}','${esc(c)}')" style="cursor:pointer;">${esc(hLabel(c))}${arrow}</th>`;}).join('')}</tr></thead><tbody>${visible.map(r=>`<tr>${columns.map(c=>`<td>${cell(r[c],c)}</td>`).join('')}</tr>`).join('')}</tbody><tfoot><tr><td colspan="${columns.length}"><div class="tt-pagination"><button type="button" onclick="window.tablePage('${id}',-1)" ${t.page<=1?'disabled':''}>上一页</button><span>第 ${t.page}/${pages} 页 · 共 ${sorted.length} 条</span><button type="button" onclick="window.tablePage('${id}',1)" ${t.page>=pages?'disabled':''}>下一页</button></div></td></tr></tfoot>`;}
window.sortTable=function(id,key){if(!_tables[id])return;if(_tables[id].sortKey===key)_tables[id].sortAsc=!_tables[id].sortAsc;else{_tables[id].sortKey=key;_tables[id].sortAsc=false;}_tables[id].page=1;_renderTable(id);};
window.tablePage=function(id,delta){const t=_tables[id];if(!t)return;t.page=(t.page||1)+delta;_renderTable(id);};
function renderTableByEl(el,rows,columns){if(!rows||!rows.length){el.innerHTML='<tbody><tr><td>暂无数据</td></tr></tbody>';return;}el.innerHTML=`<thead><tr>${columns.map(c=>`<th>${esc(hLabel(c))}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${columns.map(c=>`<td>${cell(r[c],c)}</td>`).join('')}</tr>`).join('')}</tbody>`;}

function hLabel(k){const m={code:'ETF代码',name:'名称',price:'价格',shares_m:'份额（百万份）',shares_delta_m:'相邻快照份额差（百万份）',share_change_notional_yi:'份额变动折算（亿元）',discount_pct:'折溢价%',amount_yi:'成交额（亿元）',amount_delta_yi:'成交额增量（亿元）',main_net_yi:'主力净流入（亿元）',net_yi:'净流入（亿元）',active_flow_proxy_yi:'成交额方向代理（亿元，非主力净流入）',symbol:'代码',date:'日期',timestamp:'采集时点',action:'决策动作',close:'收盘',composite_score:'综合分',trend_score:'趋势分',momentum_score:'动量分',risk_score:'风险分',rsi_14:'RSI14',macd_hist:'MACD柱',atr_pct:'ATR%',volume_ratio:'量比',new_buy_allowed:'新增买入',stop_loss:'止损',take_profit_ref:'止盈参考',suggested_shares:'建议股数',suggested_cash:'建议金额',total_return_pct:'总收益%',max_drawdown_pct:'最大回撤%',sharpe:'夏普',sortino_ratio:'索提诺',calmar_ratio:'卡玛',profit_factor:'盈亏比',avg_win_pct:'平均盈利%',avg_loss_pct:'平均亏损%',annual_return_pct:'年化收益%',trade_count:'交易次数',win_rate_pct:'胜率%',final_equity:'最终权益',open_shares:'持仓',status:'数据状态',start:'开始日期',end:'结束日期',ind_volume_score:'量能分',side:'方向',shares:'份额',pnl:'盈亏',params:'参数',rank:'排名',sector_name:'板块',sector_code:'板块编号',last_price:'最新价',change_pct:'涨跌幅',amplitude:'振幅',prev_close:'昨收',high:'最高',low:'最低',volume:'成交量',breadth:'上涨宽度',advance:'上涨家数',decline:'下跌家数',pattern:'形态',pattern_type:'形态',direction:'方向',confidence:'可信度',description:'形态说明',details:'补充信息',open:'开盘'};return m[k]||'其他信息';}
function uiStatus(value){return ({available:'可用',partial:'部分可用',missing:'暂无数据',error:'读取失败',unavailable:'暂不可用',provider_empty:'暂无记录',stale:'数据较旧',current:'最新',unknown:'待确认',unattributed:'无法归因',mapping_missing:'缺少对应关系',insufficient:'样本不足',complete:'完整',date_mismatch:'日期不一致',research_only:'仅供研究',tradable:'可执行',review_only:'仅复核',blocked:'已拦截',fresh:'数据新鲜',expired:'已过期',idle:'空闲',running:'处理中',ok:'正常'}[String(value)] || '待确认');}
function uiFactorCategory(value){return ({sentiment:'情绪面',industry:'行业/估值',flow:'资金面',macro:'宏观面',technical:'技术面',quality:'质量面',growth:'成长面'}[String(value)] || String(value || '未分类'));}
function uiSource(value){const text=String(value||'');if(!text)return '';if(/sector_fund_flow|industry.*flow/i.test(text))return '行业资金记录';if(/etf_(state|history|flow)/i.test(text))return 'ETF历史记录';if(/kline/i.test(text))return '行情历史记录';if(/stock.*flow|fund_flow/i.test(text))return '个股资金记录';if(/research|report/i.test(text))return '研报记录';if(/akshare|eastmoney|sina|tencent|yahoo/i.test(text))return '行情数据接口';return '本地数据记录';}
function uiDataset(value){const text=String(value||'');if(!text)return '未命名数据集';if(/data_warehouse|generated|parquet|\.json|\.csv/i.test(text))return uiSource(text)||'本地数据集';return text.replace(/[\\/]+/g,' · ');}
function uiDimension(value){return ({trend:'趋势',valuation:'估值',financial:'财务',liquidity:'流动性',events:'事件',profile:'画像',decision:'决策',stock_flow:'个股资金',flow:'行业资金',commodity:'行业商品',research:'研报',news:'个股新闻',mainline:'主线概念'}[String(value)] || String(value||'未知维度'));}
function flowDirectionLabel(value){return ({net_subscription_proxy:'增加方向（相邻份额快照代理）',net_redemption_proxy:'减少方向（相邻份额快照代理）',unknown:'方向未确认'}[String(value)] || uiStatus(value));}
function patternLabelCn(value){const m={doji:'十字星',hammer:'锤子线',shooting_star:'流星线',bullish_engulf:'看涨吞没',bearish_engulf:'看跌吞没',morning_star:'启明星',evening_star:'黄昏星',three_white:'三白兵',three_black:'三只乌鸦',piercing:'刺透线',dark_cloud:'乌云盖顶',harami:'孕线'};return m[String(value)] || String(value||'其他形态');}
function patternRows(patterns){return (patterns||[]).map(p=>{if(typeof p==='string')return {形态:patternLabelCn(p)};return {日期:p.date||p.日期||'—',形态:patternLabelCn(p.pattern||p.pattern_type||p.name||p.形态),方向:Number(p.direction)>0?'看涨':Number(p.direction)<0?'看跌':'中性',置信度:p.confidence==null?'—':`${(Number(p.confidence)*100).toFixed(0)}%`,形态说明:p.description||p.details||'—'};});}
function scanColumns(){return['symbol','date','action','close','composite_score','trend_score','momentum_score','ind_volume_score','risk_score','rsi_14','macd_hist','atr_pct','volume_ratio','new_buy_allowed','stop_loss','take_profit_ref','suggested_shares','suggested_cash'];}
function backtestColumns(){return['symbol','status','start','end','total_return_pct','annual_return_pct','max_drawdown_pct','sharpe','sortino_ratio','calmar_ratio','profit_factor','trade_count','win_rate_pct','final_equity','open_shares'];}
function shortKey(k){const m={fast_ma:'快MA',slow_ma:'慢MA',stop_loss_pct:'止损%',take_profit_pct:'止盈%'};return m[k]||k;}
function appendMsg(role,text){const d=document.createElement('div');d.className='msg '+role;d.textContent=text;$('chatLog').appendChild(d);$('chatLog').scrollTop=$('chatLog').scrollHeight;}
function setStatus(t){$('status').textContent=t;}
function setBusy(id,b){$(id).disabled=b;}
function pct(v){return v==null?'-':`${Number(v).toFixed(2)}%`;}
function fmt(v){return v==null?'-':typeof v==='number'?Number(v).toFixed(2).replace(/\.00$/, ''):String(v);}
function fmt2(v){return v==null?'-':typeof v==='number'?Number(v).toFixed(2):String(v);}
function cell(v,col){
  let s=fmt(v), cls='cell-neutral';
  const directionCols=new Set(['pct_chg','change_pct','main_net_yi','net_yi','active_flow_proxy_yi','amount_delta_yi','pnl','direction']);
  if(col==='action'){
    const labels={BUY_WATCH:'观察后再评估',SELL_OR_AVOID:'回避或减仓',MARKET_RISK_BLOCKED:'市场风险拦截',HOLD:'继续持有'};
    s=labels[String(v)]||uiStatus(v); cls=({BUY_WATCH:'cell-watch',SELL_OR_AVOID:'cell-sell',MARKET_RISK_BLOCKED:'cell-blocked',HOLD:'cell-hold'}[String(v)]||'cell-neutral');
  }else if(col==='status' || col==='trade_label' || col==='decision'){
    const labels={可执行:'可执行',短线候选:'可执行',观察:'观察',高分观察:'观察',条件观察:'条件观察',风险拦截:'风险拦截'};
     s=labels[String(v)]||uiStatus(v); cls=({可执行:'cell-good',短线候选:'cell-good',观察:'cell-watch',高分观察:'cell-watch',条件观察:'cell-watch',风险拦截:'cell-blocked',available:'cell-good',partial:'cell-watch',missing:'cell-blocked',error:'cell-blocked',stale:'cell-watch',complete:'cell-good',date_mismatch:'cell-blocked'}[String(v)]||'cell-neutral');
  }else if(col==='new_buy_allowed'){
    s=v===true?'允许新增':v===false?'暂不新增':uiStatus(v); cls=v===true?'cell-good':'cell-blocked';
  }else if(col==='score' || /score$/.test(String(col)) || col==='short_term_score' || col==='composite_score'){
    const n=Number(v); cls=Number.isFinite(n) ? (n>=70?'cell-good':n>=55?'cell-watch':'cell-blocked') : 'cell-blocked';
  }else if(col==='risk_score'){
    const n=Number(v); cls=Number.isFinite(n) ? (n>=60?'cell-blocked':n>=35?'cell-watch':'cell-good') : 'cell-neutral';
  }else if(directionCols.has(col) && typeof v==='number' && Number.isFinite(v)){
    cls=v>0?'cell-positive':v<0?'cell-negative':'cell-neutral';
  }
  return`<span class="cell-value ${cls}">${esc(s)}</span>`;
}
function esc(v){return String(v).replace(/[&<>"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]));}
// 状态/来源中文映射：界面只显示中文，不暴露后端英文状态码与数据源路径。
function statusCn(v){return ({available:'可用',stale:'陈旧',missing:'缺失',partial:'部分可用',error:'失败',blocked:'阻断',unmapped:'未映射',not_applicable:'不适用',provider_empty:'数据源为空',mapping_missing:'映射缺失',unattributed:'未归属',research_only:'仅供研究',unavailable:'暂不可用',demo:'演示数据',idle:'空闲',running:'处理中',ok:'正常',unknown:'待确认'}[String(v)] || '待确认');}
function srcCn(v){const s=String(v||'');const m={'akshare':'东方财富数据','eastmoney':'东方财富','sina':'新浪财经','tencent':'腾讯行情','yahoo':'雅虎财经','yfinance':'雅虎财经','north_flow':'北向资金','market_margin':'沪深交易所','stock_market_fund_flow':'东方财富资金流','futures_basis':'中金所基差','bond_futures':'中金所国债期货','commodity':'商品期货','parquet':'本地数据仓','fund_etf_hist_em':'基金历史','futures_main_sina':'新浪期货主力'};for(const k in m){if(s.toLowerCase().includes(k))return m[k];}return s.includes('/')||s.includes('.')||s.includes('(')?'本地数据':s;}
// ── 实时行情 ──
let rtTimer = null;
let rtRunning = false;

// Track alerted symbols to avoid duplicate alerts
const alertedSymbols = new Set();

async function loadRealtime() {
  try {
    const syms = state.watchlist && state.watchlist.length
      ? state.watchlist.join(',') : '002714,601899,000858';
    const data = await api('/api/realtime?symbols=' + encodeURIComponent(syms) + '&signals=true');
    renderRealtime(data.quotes || []);
    // 恢复实时信号链，但外发默认关闭，只有用户显式勾选后才推送。
    const signals = data.signals || {};
    const buyable = Object.entries(signals).filter(([symbol, signal]) => signal && signal.new_buy_allowed && !alertedSymbols.has(symbol));
    if (buyable.length) {
      const names = (data.quotes || []).filter(q => buyable.some(([symbol]) => symbol === q.symbol)).map(q => `${q.name}(${q.symbol})`).join('、');
      setStatus(`发现 ${buyable.length} 个待复核信号：${names || buyable.map(([symbol])=>symbol).join('、')}`);
      buyable.forEach(([symbol]) => alertedSymbols.add(symbol));
      if (rtRunning && $('rtSignalAlerts')?.checked) {
        try {
          await api('/api/alert', {type:'scan', rows:buyable.map(([symbol,signal])=>({symbol,...signal})), market:state.market});
          setStatus(`信号已按用户授权推送：${names || buyable.length + '只'}`);
        } catch (alertError) {
          setStatus('信号存在，但推送失败：' + alertError.message);
        }
      }
    }
  } catch(e) {
    setStatus('实时行情加载失败: ' + e.message);
  }
}

function renderRealtime(quotes) {
  const up = quotes.filter(q => q.change_pct > 0).length;
  const down = quotes.filter(q => q.change_pct < 0).length;
  const maxUp = quotes.reduce((a, q) => Math.max(a, q.change_pct), -999);
  const maxDn = quotes.reduce((a, q) => Math.min(a, q.change_pct), 999);
  $('rtMetrics').innerHTML = [
    `<div class="metric-card"><span>监控数</span><strong>${quotes.length}</strong></div>`,
    `<div class="metric-card"><span>上涨</span><strong style="color:#b42318;">${up}</strong></div>`,
    `<div class="metric-card"><span>下跌</span><strong style="color:#087443;">${down}</strong></div>`,
    `<div class="metric-card"><span>最高涨幅</span><strong style="color:#b42318;">${maxUp>-999?maxUp.toFixed(2)+'%':'-'}</strong></div>`,
    `<div class="metric-card"><span>最大跌幅</span><strong style="color:#087443;">${maxDn<999?maxDn.toFixed(2)+'%':'-'}</strong></div>`,
    `<div class="metric-card"><span>状态</span><strong>${rtRunning?'🟢 监控中':'⏸ 就绪'}</strong></div>`,
  ].join('');
  const cols = ['name','symbol','price','change_pct','amplitude','high','low','volume','time'];
  const head = cols.map(c => `<th data-col="${c}" onclick="sortRealTime('${c}')">${hLabel(c)}</th>`).join('');
  $('rtTable').innerHTML = `<thead><tr>${head}</tr></thead><tbody>${
    quotes.map(q => {
      const cls = q.change_pct > 0 ? 'good' : q.change_pct < 0 ? 'bad' : '';
      const vol = q.volume >= 1e8 ? (q.volume/1e8).toFixed(2)+'亿' : q.volume >= 1e4 ? (q.volume/1e4).toFixed(0)+'万' : String(q.volume);
      return `<tr class="${cls}"><td>${esc(q.name)}</td><td>${esc(q.symbol)}</td><td>${q.price.toFixed(2)}</td><td>${q.change_pct.toFixed(2)}%</td><td>${q.amplitude.toFixed(2)}%</td><td>${q.high.toFixed(2)}</td><td>${q.low.toFixed(2)}</td><td>${vol}</td><td>${q.time||''}</td></tr>`;
    }).join('')
  }</tbody>`;
}

function toggleMonitor() {
  if (rtRunning) {
    if (rtTimer) clearInterval(rtTimer);
    rtTimer = null; rtRunning = false;
    $('rtStartMonitorBtn').textContent = '▶ 启动监控';
    $('rtNextPoll').textContent = '已暂停';
  } else {
    rtRunning = true;
    $('rtStartMonitorBtn').textContent = '⏹ 停止';
    loadRealtime();
    scheduleRtPoll();
  }
}

function scheduleRtPoll() {
  if (!rtRunning) return;
  if (rtTimer) clearInterval(rtTimer);
  const interval = (parseInt($('rtInterval').value) || 150) * 1000;
  const auto = $('rtAutoRefresh').checked;
  if (auto) {
    $('rtNextPoll').textContent = '下次刷新: ' + new Date(Date.now()+interval).toLocaleTimeString();
    rtTimer = setInterval(() => { loadRealtime(); scheduleRtPoll(); }, interval);
  } else {
    $('rtNextPoll').textContent = '手动刷新';
  }
}

let rtSortKey = null, rtSortAsc = true;
function sortRealTime(key) {
  if (rtSortKey === key) rtSortAsc = !rtSortAsc;
  else { rtSortKey = key; rtSortAsc = true; }
  loadRealtime();
}

// ── 日内K线图 ──
let intradayChart = null;

function renderIntradayChart(domId, rows, period) {
  const dom = $(domId);
  if(!dom) return;
  if(!rows||rows.length<5){dom.innerHTML='<div class="empty" style="padding:40px;text-align:center;">日内数据不足</div>';return;}
  if(!intradayChart){intradayChart=echarts.init(dom);window.addEventListener('resize',()=>intradayChart.resize());}

  const w=dom.clientWidth;
  const g=[{left:50,right:22,top:40,height:260},{left:50,right:22,top:320,height:80},{left:50,right:22,top:420,height:70},{left:50,right:22,top:510,height:70}];

  const dates=rows.map(r=>{
    const d=new Date(r.date);
    return d.toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit',month:'short',day:'numeric'});
  });
  const kdata=rows.map(r=>[Number(r.open),Number(r.close),Number(r.low),Number(r.high)]);
  const vols=rows.map(r=>Number(r.volume)||0);
  const volMax=Math.max(...vols.filter(v=>v>0),1);
  const volSuffix=volMax>1e8?'亿':volMax>1e4?'万':'';
  const volUnit=volMax>1e8?1e8:volMax>1e4?1e4:1;
  const volsN=vols.map(v=>v/volUnit);

  const ma5=rows.map(r=>r.ma_5),ma10=rows.map(r=>r.ma_10),ma20=rows.map(r=>r.ma_20);
  const macdD=rows.map(r=>r.macd_dif),macdD2=rows.map(r=>r.macd_dea),macdH=rows.map(r=>r.macd_hist);
  const rsi=rows.map(r=>r.rsi_14);

  const option={
    backgroundColor:'#0d1117',animation:false,
    tooltip:{trigger:'axis',axisPointer:{type:'cross'},confine:true,
      formatter:function(params){
        const p=params.find(x=>x.seriesName==='K线');
        if(!p||p.dataIndex==null)return '';
        const i=p.dataIndex,r=rows[i];
        if(!r)return '';
        const d=new Date(r.date);
        let s=`<b>${d.toLocaleString('zh-CN')}</b><br/>开 ${fmt(r.open)} 收 ${fmt(r.close)} 高 ${fmt(r.high)} 低 ${fmt(r.low)}<br/>`;
        params.filter(p2=>p2.seriesName&&p2.seriesName.startsWith('MA')).forEach(p2=>s+=`${p2.marker} ${p2.seriesName} ${p2.value}<br/>`);
        if(r.volume!=null){const vStr=r.volume>=1e8?(r.volume/1e8).toFixed(2)+'亿':r.volume>=1e4?(r.volume/1e4).toFixed(0)+'万':String(r.volume);s+=`量 ${vStr}<br/>`;}
        if(r.volume_ratio!=null)s+=`量比 ${fmt(r.volume_ratio)}<br/>`;
        if(r.rsi_14!=null)s+=`RSI(14) ${fmt(r.rsi_14)}<br/>`;
        return s;
      }
    },
    legend:{data:['K线','MA5','MA10','MA20','成交量','MACD','DEA','RSI(14)'],top:2,left:60,textStyle:{fontSize:11}},
    grid:g,
    xAxis:[
      {type:'category',data:dates,gridIndex:0,axisLine:{onZero:false},axisLabel:{fontSize:9,rotate:30},splitLine:{show:true,lineStyle:{color:'#d1d5db',width:0.8}}},
      {type:'category',data:dates,gridIndex:1,axisLabel:{show:false},splitLine:{show:false}},
      {type:'category',data:dates,gridIndex:2,axisLabel:{show:false},splitLine:{show:false}},
      {type:'category',data:dates,gridIndex:3,axisLabel:{fontSize:9,rotate:30,interval:Math.max(1,Math.floor(dates.length/15))},splitLine:{show:false}},
    ],
    yAxis:[
      {scale:true,gridIndex:0,splitNumber:4,axisLabel:{fontSize:10},splitLine:{lineStyle:{color:'#d1d5db',width:0.8}}},
      {scale:true,gridIndex:1,splitNumber:2,axisLabel:{fontSize:9,formatter:v=>v>=1?v+(volSuffix||''):v}},
      {scale:true,gridIndex:2,splitNumber:3,axisLabel:{fontSize:9}},
      {scale:true,gridIndex:3,min:0,max:100,splitNumber:3,axisLabel:{fontSize:9}},
    ],
    dataZoom:[
      {type:'inside',xAxisIndex:[0,1,2,3],start:80,end:100,minSpan:5},
      {type:'slider',xAxisIndex:[0,1,2,3],show:true,height:12,bottom:4,borderColor:'#d1d5db',fillerColor:'rgba(31,122,92,0.08)',handleStyle:{color:'#1f7a5c'},start:80,end:100,minSpan:5},
    ],
    series:[
      {name:'K线',type:'candlestick',xAxisIndex:0,yAxisIndex:0,data:kdata,itemStyle:{color:COLORS.up,color0:COLORS.down,borderColor:COLORS.up,borderColor0:COLORS.down}},
      {name:'MA5',type:'line',xAxisIndex:0,yAxisIndex:0,symbol:'none',lineStyle:{color:'#2563eb',width:1.2},data:ma5},
      {name:'MA10',type:'line',xAxisIndex:0,yAxisIndex:0,symbol:'none',lineStyle:{color:'#b45309',width:1.2},data:ma10},
      {name:'MA20',type:'line',xAxisIndex:0,yAxisIndex:0,symbol:'none',lineStyle:{color:'#1f7a5c',width:1.2,type:'dashed'},data:ma20},
      {name:'成交量',type:'bar',xAxisIndex:1,yAxisIndex:1,data:volsN.map((v,i)=>{const c=rows[i];return{value:v,itemStyle:{color:c&&Number(c.close)>=Number(c.open)?COLORS.up:COLORS.down}};})},
      {name:'MACD',type:'line',xAxisIndex:2,yAxisIndex:2,symbol:'none',lineStyle:{color:COLORS.macd,width:1.2},data:macdD},
      {name:'DEA',type:'line',xAxisIndex:2,yAxisIndex:2,symbol:'none',lineStyle:{color:COLORS.macdSignal,width:1.2},data:macdD2},
      {name:'RSI(14)',type:'line',xAxisIndex:3,yAxisIndex:3,symbol:'none',lineStyle:{color:COLORS.rsi,width:1.2},data:rsi},
    ]
  };
  intradayChart.setOption(applyChartDarkTheme(option),true);
}

// ═══════════════════════════════════════════════════════════════════════════
// 股票名称自动补全
// ═══════════════════════════════════════════════════════════════════════════

// 需要绑定自动补全的输入框ID列表
const AUTOCOMPLETE_INPUTS = ['scanSymbols', 'chartSymbol', 'intraSymbol', 'btSymbols', 'optSymbol', 'wlInput', 'marginSymbol'];

let _acAbortController = null;  // 用于取消上一次请求
let _acHighlightIdx = -1;       // 当前高亮项索引
let _acActiveInput = null;      // 当前活动的输入框

/** 获取或创建下拉容器 */
function _acDropdown() {
  let el = document.getElementById('stockAutocomplete');
  if (!el) {
    el = document.createElement('div');
    el.id = 'stockAutocomplete';
    el.className = 'autocomplete-dropdown';
    el.style.display = 'none';
    document.body.appendChild(el);
  }
  return el;
}

/** 定位下拉框到输入框下方 */
function _acPosition(input) {
  const dd = _acDropdown();
  const rect = input.getBoundingClientRect();
  dd.style.left = rect.left + 'px';
  dd.style.top = (rect.bottom + 4) + 'px';
  dd.style.width = Math.max(rect.width, 200) + 'px';
}

/** 隐藏下拉框 */
function _acHide() {
  const dd = _acDropdown();
  dd.style.display = 'none';
  dd.innerHTML = '';
  _acActiveInput = null;
  _acHighlightIdx = -1;
}

/** 渲染搜索结果 */
function _acRender(results) {
  const dd = _acDropdown();
  if (!results || !results.length) {
    dd.style.display = 'none';
    dd.innerHTML = '';
    _acHighlightIdx = -1;
    return;
  }
  dd.style.display = 'block';
  dd.innerHTML = results.map((r, i) => {
    const mt = r.match_type || '';
    const mtLabel = mt === 'pinyin_initials' ? '拼音' : mt === 'pinyin_full' ? '全拼' : mt === 'code' ? '代码' : mt === 'cn' || mt === 'chinese' ? '中文' : '';
    const mtHtml = mtLabel ? `<span class="ac-mt">${mtLabel}</span>` : '';
    return `<div class="ac-item${i === 0 ? ' active' : ''}" data-index="${i}" data-code="${esc(r.code)}" data-name="${esc(r.name)}">
      <span class="ac-name">${esc(r.name)}</span>
      <span class="ac-code">${esc(r.code)}</span>${mtHtml}
    </div>`;
  }).join('');
  _acHighlightIdx = 0;
  _acDropdown().scrollTop = 0;
}

/** 点击/回车选中某个结果 */
function _acSelect(index) {
  const dd = _acDropdown();
  const item = dd.querySelector(`.ac-item[data-index="${index}"]`);
  if (!item || !_acActiveInput) return;
  const code = item.dataset.code;
  const input = _acActiveInput;
  // 替换当前输入框内容为此股票的代码
  if (input.id === 'scanSymbols' || input.id === 'wlInput') {
    // 多值输入框：追加在后面
    const current = input.value.trim();
    const parts = current.split(/[,\s]+/).filter(Boolean);
    // 如果最后一个输入是搜索词，替换它
    if (parts.length > 0 && !/^\d{6}$/.test(parts[parts.length - 1])) {
      parts[parts.length - 1] = code;
    } else {
      parts.push(code);
    }
    input.value = parts.join(' ');
  } else {
    // 单值输入框：直接替换
    input.value = code;
  }
  _acHide();
  input.focus();
}

/** 搜索股票名称 */
async function _acSearch(input) {
  const q = input.value.trim();
  if (!q || /^\d{1,6}$/.test(q)) {
    _acHide();
    return;
  }
  if (_acAbortController) {
    _acAbortController.abort();
  }
  _acAbortController = new AbortController();
  try {
    const data = await api(`/api/search_stock?q=${encodeURIComponent(q)}`, undefined, { timeout: 8000, cache_ttl: 30, silent: true });
    if (!data.ok || !data.results || !data.results.length) {
      _acHide();
      return;
    }
    _acActiveInput = input;
    _acPosition(input);
    _acRender(data.results);
  } catch (e) {
    if (e.name !== 'AbortError') {
      _acHide();
    }
  }
}

/** 初始化单个输入框的自动补全 */
function _acBindInput(inputId) {
  const input = document.getElementById(inputId);
  if (!input) return;

  // 输入事件（300ms 防抖）
  let _acTimer = null;
  input.addEventListener('input', () => {
    clearTimeout(_acTimer);
    _acTimer = setTimeout(() => _acSearch(input), 300);
  });

  // 键盘事件
  input.addEventListener('keydown', (e) => {
    const dd = _acDropdown();
    if (dd.style.display === 'none') return;
    const items = dd.querySelectorAll('.ac-item');
    if (!items.length) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      _acHighlightIdx = Math.min(_acHighlightIdx + 1, items.length - 1);
      _acUpdateHighlight(items);
      items[_acHighlightIdx].scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      _acHighlightIdx = Math.max(_acHighlightIdx - 1, 0);
      _acUpdateHighlight(items);
      items[_acHighlightIdx].scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'Enter' || e.key === 'Tab') {
      if (_acHighlightIdx >= 0 && _acHighlightIdx < items.length) {
        e.preventDefault();
        _acSelect(_acHighlightIdx);
      }
    } else if (e.key === 'Escape') {
      _acHide();
    }
  });

  // 失焦关闭（延迟以允许点击选中）
  input.addEventListener('blur', () => {
    setTimeout(_acHide, 200);
  });
}

/** 更新高亮状态 */
function _acUpdateHighlight(items) {
  items.forEach((el, i) => {
    el.classList.toggle('active', i === _acHighlightIdx);
  });
}

// 绑定所有输入框 + 点击委托
AUTOCOMPLETE_INPUTS.forEach(id => _acBindInput(id));

document.addEventListener('click', (e) => {
  const dd = _acDropdown();
  if (dd.style.display === 'none') return;
  const item = e.target.closest('.ac-item');
  if (item) {
    const idx = parseInt(item.dataset.index, 10);
    _acSelect(idx);
  }
});

// 窗口滚动/缩放时重新定位
window.addEventListener('scroll', () => {
  const dd = _acDropdown();
  if (dd.style.display !== 'none' && _acActiveInput) {
    _acPosition(_acActiveInput);
  }
}, true);
window.addEventListener('resize', () => {
  const dd = _acDropdown();
  if (dd.style.display !== 'none' && _acActiveInput) {
    _acPosition(_acActiveInput);
  }
});

// ═══════════════════════════════════════════════════════════════════════════
// 港美股行情
// ═══════════════════════════════════════════════════════════════════════════

function fmtNum(value, digits = 2) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : '—';
}
function fmtSigned(value, digits = 2) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  return (n >= 0 ? '+' : '') + n.toFixed(digits);
}

async function loadGlobalQuotes() {
  const input = $('globalSymbols');
  const symbols = input ? input.value.trim() || 'hk00700,usAAPL' : 'hk00700,usAAPL';
  setBusy('globalRefreshBtn', true); setStatus('加载港美股行情...');
  try {
    const data = await api(`/api/global_quotes?symbols=${encodeURIComponent(symbols)}`);
    renderGlobalQuotes(data.quotes || [], data);
    setStatus(`港美股 ${data.count} 只已刷新`);
  } catch(e) {
    setStatus('港美股加载失败: ' + e.message);
  } finally {
    setBusy('globalRefreshBtn', false);
  }
}

function renderGlobalQuotes(quotes, payload = {}) {
  const el = $('globalTable');
  if (!el) return;
  if (!quotes.length) {
    el.innerHTML = `<thead><tr><th>状态</th></tr></thead><tbody><tr><td class="empty">${payload.error ? esc(payload.error) : '暂无行情数据（检查网络/数据源后重试）'}</td></tr></tbody>`;
    return;
  }
  const thead = `<thead><tr>
    <th>代码</th><th>名称</th><th>现价</th><th>涨跌额</th><th>涨跌幅</th>
    <th>今开</th><th>最高</th><th>最低</th><th>昨收</th>
  </tr></thead>`;
  const tbody = quotes.map(q => {
    const pct = Number(q.change_pct);
    const cls = Number.isFinite(pct) ? (pct >= 0 ? 'rt-up' : 'rt-down') : 'rt-flat';
    return `<tr>
      <td><code>${esc(q.symbol)}</code></td>
      <td><a href="#" class="gl-name" data-symbol="${esc(q.symbol)}" data-name="${esc(q.name || q.symbol)}">${esc(q.name || q.symbol)}</a></td>
      <td class="${cls} rt-change">${fmtNum(q.price)}</td>
      <td class="${cls}">${fmtSigned(q.change)}</td>
      <td class="${cls}">${fmtSigned(q.change_pct)}%</td>
      <td>${fmtNum(q.open)}</td>
      <td>${fmtNum(q.high)}</td>
      <td>${fmtNum(q.low)}</td>
      <td>${fmtNum(q.prev_close)}</td>
    </tr>`;
  }).join('');
  el.innerHTML = thead + '<tbody>' + tbody + '</tbody>';
  el.querySelectorAll('.gl-name').forEach(a => {
    a.addEventListener('click', (e) => {
      e.preventDefault();
      loadGlobalKline(a.dataset.symbol, a.dataset.name);
    });
  });
}

async function loadGlobalKline(symbol, name) {
  const panel = $('globalKlinePanel');
  const title = $('globalKlineTitle');
  const statusEl = $('globalKlineStatus');
  if (statusEl) statusEl.textContent = '加载中…';
  try {
    const data = await api(`/api/global_kline?symbol=${encodeURIComponent(symbol)}&count=1000`);
    if (!data.ok || !data.rows || !data.rows.length) {
      if (panel) panel.hidden = false;
      if (statusEl) statusEl.textContent = data.error || '该标的暂无可用历史数据（数据源无返回）';
      if (title) title.textContent = `${name || symbol} K线（无数据）`;
      return;
    }
    if (title) title.textContent = `${name || symbol} (${data.symbol}) K线`;
    if (panel) panel.hidden = false;
    if (statusEl) statusEl.textContent = `数据日 ${data.rows[data.rows.length-1]?.date || '—'} · ${data.rows.length} 根`;
    renderGlobalKlineChart('globalKlineChart', data.rows);
  } catch(e) {
    if (panel) panel.hidden = false;
    if (statusEl) statusEl.textContent = 'K线加载失败: ' + e.message;
    if (title) title.textContent = `${name || symbol} K线（加载失败）`;
  }
}

function renderGlobalKlineChart(domId, rows) {
  const dom = $(domId);
  if (!dom || !rows.length) {
    if (dom) dom.innerHTML = '<div class="empty">暂无数据</div>';
    return;
  }
  const dates = rows.map(r => r.date);
  const kdata = rows.map(r => [Number(r.open), Number(r.close), Number(r.low), Number(r.high)].map(v => Number.isFinite(v) ? v : null));
  const vols = rows.map(r => Number(r.volume));
  const maxVol = Math.max(...vols.filter(Number.isFinite), 1);

  // MA lines
  function maLine(n) {
    return rows.map((r, i) => {
      if (i < n-1) return null;
      let sum = 0;
      for (let j = i-n+1; j <= i; j++) sum += rows[j].close;
      return sum / n;
    });
  }
  const ma5 = maLine(5), ma10 = maLine(10), ma20 = maLine(20);

  // BOLL(20) 在本地计算，避免依赖未定义的指标数组。
  const bollUpper = [], bollMid = [], bollLower = [];
  for (let i = 0; i < rows.length; i++) {
    if (i < 19) { bollMid.push(null); bollUpper.push(null); bollLower.push(null); continue; }
    let s = 0;
    for (let j = i - 19; j <= i; j++) s += rows[j].close;
    const mid = s / 20;
    let v = 0;
    for (let j = i - 19; j <= i; j++) v += (rows[j].close - mid) * (rows[j].close - mid);
    const sd = Math.sqrt(v / 20);
    bollMid.push(mid);
    bollUpper.push(mid + 2 * sd);
    bollLower.push(mid - 2 * sd);
  }

  const option = {
    backgroundColor: '#0d1117',
    animation: false,
    grid: [
      { left: '6%', right: '3%', top: '8%', height: '60%' },
      { left: '6%', right: '3%', top: '76%', height: '16%' },
    ],
    xAxis: [
      { type: 'category', data: dates, gridIndex: 0, axisLabel: { fontSize: 10, rotate: 30 }, splitLine: { show: true } },
      { type: 'category', data: dates, gridIndex: 1, axisLabel: { show: false } },
    ],
    yAxis: [
      { scale: true, gridIndex: 0 },
      { scale: true, gridIndex: 1 },
    ],
    dataZoom: [
      { type: 'inside', xAxisIndex: [0, 1], start: 70, end: 100 },
      { type: 'slider', xAxisIndex: [0, 1], show: true, height: 10, bottom: 2, start: 70, end: 100 },
    ],
    series: [
      {
        name: 'K线', type: 'candlestick', xAxisIndex: 0, yAxisIndex: 0,
        data: kdata,
        itemStyle: { color: COLORS.up, color0: COLORS.down, borderColor: COLORS.up, borderColor0: COLORS.down },
      },
      { name: 'MA5', type: 'line', xAxisIndex: 0, yAxisIndex: 0, symbol: 'none', lineStyle: { color: '#2563eb', width: 1 }, data: ma5 },
      { name: 'MA10', type: 'line', xAxisIndex: 0, yAxisIndex: 0, symbol: 'none', lineStyle: { color: '#b45309', width: 1 }, data: ma10 },
      { name: 'MA20', type: 'line', xAxisIndex: 0, yAxisIndex: 0, symbol: 'none', lineStyle: { color: '#1f7a5c', width: 1, type: 'dashed' }, data: ma20 },
      { name: 'BOLL上', type: 'line', xAxisIndex: 0, yAxisIndex: 0, symbol: 'none', lineStyle: { color: 'rgba(139,92,246,0.3)', width: 0.8 }, data: bollUpper },
      { name: 'BOLL下', type: 'line', xAxisIndex: 0, yAxisIndex: 0, symbol: 'none', lineStyle: { color: 'rgba(139,92,246,0.3)', width: 0.8 }, data: bollLower },
      { name: 'BOLL', type: 'line', xAxisIndex: 0, yAxisIndex: 0, symbol: 'none', lineStyle: { color: '#8b5cf6', width: 1, type: 'dashed' }, data: bollMid },
      {
        name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1,
        data: vols.map((v, i) => ({ value: Number.isFinite(v) ? v : null, itemStyle: { color: rows[i].close >= rows[i].open ? COLORS.up : COLORS.down } })),
      },
    ]
  };
  if (state.globalKlineChart) {
    state.globalKlineChart.setOption(applyChartDarkTheme(option), true);
  } else {
    state.globalKlineChart = echarts.init(dom);
    state.globalKlineChart.setOption(applyChartDarkTheme(option));
    setTimeout(() => state.globalKlineChart.resize(), 200);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 历史数据中心：多市场历史抓取/落库/显示全链路
// ═══════════════════════════════════════════════════════════════════════════

async function loadMarketHistoryCatalog(refresh = false) {
  setBusy('mhRefreshBtn', true); setStatus(refresh ? '刷新本地历史目录状态...' : '读取历史数据状态...');
  try {
    const d = await api('/api/market_history?asset=catalog' + (refresh ? '&refresh=true' : ''), undefined, { timeout: 600000 });
    renderMarketHistoryCatalog(d.items || []);
    setStatus(refresh ? '历史目录状态已刷新；按资产单独抓取可更新数据' : '历史数据状态已更新');
  } catch(e) {
    setStatus('历史数据中心失败: ' + e.message);
  } finally {
    setBusy('mhRefreshBtn', false);
  }
}

function renderMarketHistoryCatalog(items) {
  const available = items.filter(x => x.status === 'available').length;
  const el = $('mhMetrics');
  if (el) el.innerHTML = [
    ['资产数', items.length], ['已就绪', available], ['缺失/失败', items.length - available],
  ].map(([l,v]) => `<div class="metric"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join('');
  const table = $('mhTable');
  if (!table) return;
  if (!items.length) { table.innerHTML = '<thead><tr><th>状态</th></tr></thead><tbody><tr><td class="empty">历史数据目录为空</td></tr></tbody>'; return; }
  table.innerHTML = `<thead><tr><th>资产</th><th>状态</th><th>开始</th><th>数据日</th><th>行数</th><th>失败原因</th><th>操作</th></tr></thead><tbody>${items.map(x => `
    <tr>
      <td><b>${esc(x.name || x.asset)}</b><small style="display:block;color:var(--muted);">${esc(x.symbol || '')}</small></td>
      <td style="color:${x.status === 'available' ? '#46a758' : '#f06a69'};">${esc(x.status === 'available' ? '已就绪' : '缺失')}</td>
      <td>${esc(x.start || '—')}</td>
      <td>${esc(x.as_of || '—')}</td>
      <td>${esc(x.count ?? '—')}</td>
      <td style="color:#f0b429;font-size:12px;">${esc(x.error || '')}</td>
      <td>${x.status === 'available' ? `<button class="btn-sm" onclick="loadMarketHistoryAsset('${esc(x.asset)}')">历史曲线</button>` : `<button class="btn-sm" onclick="fetchMarketHistoryAsset('${esc(x.asset)}')">抓取</button>`}</td>
    </tr>`).join('')}</tbody>`;
}

async function fetchMarketHistoryAsset(asset) {
  setStatus(`抓取 ${asset} 历史...`);
  try {
    const d = await api(`/api/market_history?asset=${encodeURIComponent(asset)}&refresh=true`, undefined, { timeout: 600000 });
    if (d.ok) { loadMarketHistoryAsset(asset, d); } else { setStatus(`${asset} 抓取失败: ${d.error || '无数据'}`); }
  } catch(e) { setStatus(`抓取失败: ${e.message}`); }
}

let currentHistoryAsset = null;
function reloadCurrentHistory(){ if(currentHistoryAsset) loadMarketHistoryAsset(currentHistoryAsset); }
window.reloadCurrentHistory = reloadCurrentHistory;
async function loadMarketHistoryAsset(asset, preloaded) {
  currentHistoryAsset = asset;
  const days = Number($('mhDays')?.value || 250);
  const chart = $('mhChart'), meta = $('mhChartMeta');
  if (chart) chart.innerHTML = '<div class="empty">加载中…</div>';
  try {
    const d = preloaded || await api(`/api/market_history?asset=${encodeURIComponent(asset)}&days=${days}`, undefined, { cache_ttl: 300 });
    if (!d.ok || !d.rows || !d.rows.length) {
      if (chart) chart.innerHTML = `<div class="empty">${esc(d.error || '暂无数据')}</div>`;
      if (meta) meta.textContent = `${d.name || asset}：${d.status || '缺失'}`;
      return;
    }
    const dates = d.rows.map(r => r.date);
    let series;
    if (asset === 'margin_market') {
      series = [
        ['沪深合计', 'close', '#58a6ff'], ['沪市', 'sh_balance_yi', '#f6465d'], ['深市', 'sz_balance_yi', '#2ebd85'], ['日变动', 'change_yi', '#d29922'],
      ];
    } else if (asset === 'market_fund_flow') {
      series = [
        ['主力净流入', 'main_net', '#58a6ff'], ['超大单', 'super_net', '#f6465d'], ['大单', 'large_net', '#d29922'], ['中单', 'medium_net', '#2ebd85'],
      ];
    } else {
      series = [[d.name || asset, 'close', '#58a6ff']];
    }
    if (meta) meta.textContent = `${d.name} · ${d.start} → ${d.as_of} · ${d.count} 行（显示${d.rows.length}）${d.adjustment ? ' · ' + d.adjustment : ''}`;
    const option = {
      backgroundColor: '#0d1117',
      animation: false,
      tooltip: { trigger: 'axis' },
      legend: { data: series.map(item => item[0]), textStyle: { color: '#8b949e' } },
      grid: { left: 70, right: 24, top: 42, bottom: 60 },
      xAxis: { type: 'category', data: dates, axisLabel: { color: '#9ca3af' } },
      yAxis: { type: 'value', scale: true, axisLabel: { color: '#9ca3af' } },
      dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 8 }],
      series: series.map(([name, field, color]) => ({ name, type: 'line', showSymbol: false, connectNulls: true,
        data: d.rows.map(row => Number.isFinite(Number(row[field])) ? Number(row[field]) : null),
        lineStyle: { color, width: field === 'close' || field === 'main_net' ? 1.8 : 1.1 } })),
    };
    if (window.echarts) {
      const instance = chartFor('mhChart'); if (instance) instance.setOption(option, true);
    } else if (chart) {
      chart.innerHTML = '<div class="empty">ECharts 未加载</div>';
    }
  } catch(e) {
    if (chart) chart.innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
}
window.loadMarketHistoryAsset = loadMarketHistoryAsset;
window.fetchMarketHistoryAsset = fetchMarketHistoryAsset;

// ═══════════════════════════════════════════════════════════════════════════
// 融资融券
// ═══════════════════════════════════════════════════════════════════════════

async function loadMarginSummary() {
  setBusy('marginRefreshBtn', true); setStatus('加载融资融券...');
  try {
    const data = await api('/api/margin');
    renderMarginSummary(data);
    setStatus('两融数据已加载');
  } catch(e) {
    setStatus('两融加载失败: ' + e.message);
  } finally {
    setBusy('marginRefreshBtn', false);
  }
}

function renderMarginSummary(data) {
  const el = $('marginSummary');
  if (!el) return;
  if (!data.ok) {
    el.innerHTML = '<div class="empty">数据不可用: ' + esc(data.error) + '</div>';
    return;
  }
  el.innerHTML = `
    <div class="metric"><div class="label">全市场融资余额</div><div class="value">${data.total_margin_balance?.toFixed(0) || '?'} 亿</div></div>
    <div class="metric"><div class="label">较前日变动</div><div class="value" style="color:${(data.total_margin_change||0) >= 0 ? '#b42318' : '#087443'}">${data.total_margin_change >= 0 ? '+' : ''}${(data.total_margin_change||0).toFixed(2)} 亿</div></div>
    <div class="metric"><div class="label">沪市</div><div class="value">${data.ss_margin_balance?.toFixed(0) || '?'} 亿</div></div>
    <div class="metric"><div class="label">深市</div><div class="value">${data.sz_margin_balance?.toFixed(0) || '?'} 亿</div></div>
    <div class="metric"><div class="label">TMT 融资余额</div><div class="value" title="${esc(data.tmt_note || '')}">${data.tmt_margin_balance ? data.tmt_margin_balance.toFixed(0) + ' 亿' : '暂无公开口径'}</div></div>
    <div class="metric"><div class="label">TMT 较前日</div><div class="value">${data.tmt_note ? '—' : ((data.tmt_margin_change >= 0 ? '+' : '') + (data.tmt_margin_change||0).toFixed(2) + ' 亿')}</div></div>
  `;
}

async function loadMarginDetail() {
  const symbol = ($('marginSymbol')?.value || '').trim() || '002714';
  setStatus('查询两融明细...');
  try {
    const data = await api(`/api/margin?symbol=${encodeURIComponent(symbol)}`);
    const el = $('marginDetail');
    if (!el) return;
    if (data.ok) {
      el.innerHTML = `<div class="split" style="margin-top:10px;">
        <div class="metric"><div class="label">${symbol} 融资余额</div><div class="value">${data.rzye} 亿</div></div>
        <div class="metric"><div class="label">${symbol} 融券${data.rqye_unit === '亿股' ? '余量' : '余额'}</div><div class="value">${data.rqye} ${esc(data.rqye_unit || '亿元')}</div></div>
      </div>`;
    } else {
      el.innerHTML = '<div class="empty">查询失败: ' + esc(data.error) + '</div>';
    }
    setStatus('两融查询完成');
  } catch(e) {
    const el = $('marginDetail');
    if (el) el.innerHTML = `<div class="empty">个股两融查询失败：${esc(e.message)}。全市场两融汇总不受影响，可稍后重试。</div>`;
    setStatus('两融查询失败: ' + e.message);
  }
}



// ═════════════════════════════════════════════════════════════
//  — 新增模块接入
// ═════════════════════════════════════════════════════════════

// ── 沪深港通 / 北向资金（与市场舆情并存，不相互替代） ──
async function loadNorthboundFlow() {
  const stateEl = $('northboundState');
  if (stateEl) { stateEl.hidden = false; stateEl.className = 'data-state loading'; stateEl.textContent = '正在读取沪深港通可用数据…'; }
  try {
    const d = await api('/api/north_flow?refresh=true', undefined, { cache_ttl: 0, refresh: true, silent: true });
    if (!d.ok) throw new Error(d.error || '沪深港通接口不可用');
    const payload = d.data || d;
    const summary = payload.summary || {};
    if (stateEl) { stateEl.className = 'data-state ' + (payload.trading ? '' : 'empty'); stateEl.textContent = payload.trading_status || '历史可用数据'; }
    const entries = Object.entries(summary).filter(([label, value]) => !(payload.net_buy_disclosed === false && /净买入|额度余额/.test(label)) && value !== 0);
    $('northboundMetrics').innerHTML = entries.length ? entries.slice(0, 8).map(([label,value]) => `<div class="metric"><div class="label">${esc(label)}</div><div class="value">${esc(value ?? '-')}</div></div>`).join('') : `<div class="data-state empty">北向净买入已停止披露，不展示 0 值占位；请查看个股持股结构。</div>`;
    $('northboundNote').innerHTML = `<b>口径说明：</b>${esc(payload.note || '沪深港通资金独立展示，不以舆情替代。')}<br><span class="data-meta">来源 ${esc(srcCn(payload.source || '北向资金'))} · ${esc(payload.trading_status || '')}</span>`;
    const rows = Array.isArray(payload.history) ? payload.history : [];
    renderTable('northboundTable', rows, rows.length ? Object.keys(rows[0]).slice(0, 10) : ['date','total_net_yi','sh_net_yi','sz_net_yi']);
  } catch (error) {
    if (stateEl) { stateEl.className = 'data-state error'; stateEl.textContent = '沪深港通当前不可用：' + error.message; }
    if ($('northboundMetrics')) $('northboundMetrics').innerHTML = '';
    if ($('northboundNote')) $('northboundNote').innerHTML = '<b>功能仍保留。</b>当前披露规则或数据源不可用，不以市场舆情、0值或旧日期伪装成当日资金。';
    renderTable('northboundTable', [], ['date','total_net_yi','sh_net_yi','sz_net_yi']);
  }
}
async function loadNorthboundHoldings() {
  const stateEl = $('northboundHoldingsState');
  if (stateEl) { stateEl.hidden = false; stateEl.className = 'data-state loading'; stateEl.textContent = '正在读取北向持股结构…'; }
  try {
    const symbol = ($('sSymbol')?.value || '').trim();
    const endpoint = symbol ? `/api/north_holdings?symbol=${encodeURIComponent(symbol)}&indicator=5日排行&limit=80&refresh=true` : '/api/north_holdings?indicator=5日排行&limit=80&refresh=true';
    const d = await api(endpoint, undefined, { cache_ttl: 0, refresh: true, silent: true });
    const data = d.data || d;
    if (!d.ok && !data.rows?.length) throw new Error(d.error || data.error || '北向持股结构不可用');
    if (stateEl) { stateEl.className = 'data-state ' + (data.status === 'available' ? 'good' : 'empty'); stateEl.textContent = data.status === 'available' ? `${symbol ? symbol + ' 个股' : '北向持股'}结构可用 · 数据日 ${data.as_of || '未知'}` : (data.error || '暂无北向持股结构'); }
    if ($('northboundHoldingsNote')) $('northboundHoldingsNote').innerHTML = `<b>数据分层：</b>持股数量、持股市值、占比和区间变化为持股结构；北向盘中/日频净买入已停止披露，二者不互相替代。<br><span class="data-meta">来源 ${esc(data.source || '北向持股排行')} · 口径 ${esc(data.indicator || '5日排行')} · 数据日 ${esc(data.as_of || '未知')}</span>`;
    const rows = Array.isArray(data.rows) ? data.rows : [];
    const metricKeys = ['holding_market_cap','holding_shares','holding_pct','holding_change_1d','holding_change_5d','change_pct'];
    $('northboundHoldingsMetrics').innerHTML = [['有效记录', data.count ?? rows.length], ['数据日', data.as_of || '-'], ['排行口径', data.indicator || '-'], ['功能状态', '持股结构']].map(([label,value]) => `<div class="metric"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>`).join('');
    renderTable('northboundHoldingsTable', rows, rows.length ? ['symbol','name',...metricKeys.filter(k => k in rows[0]),'date'] : ['symbol','name','holding_pct','holding_market_cap','holding_change_5d','date']);
  } catch (error) {
    if (stateEl) { stateEl.className = 'data-state error'; stateEl.textContent = '北向持股结构当前不可用：' + error.message; }
    if ($('northboundHoldingsNote')) $('northboundHoldingsNote').innerHTML = '<b>功能仍保留。</b>当前数据源未返回有效持股结构，不能用停更的资金流或 0 值替代。';
    if ($('northboundHoldingsMetrics')) $('northboundHoldingsMetrics').innerHTML = '';
    renderTable('northboundHoldingsTable', [], ['symbol','name','holding_pct','holding_market_cap','holding_change_5d','date']);
  }
}

// 进入北向页时同时展示持股结构；刷新按钮重新读取两层数据。
const originalNorthboundLoader = loadNorthboundFlow;
loadNorthboundFlow = async function () { await Promise.allSettled([originalNorthboundLoader(), loadNorthboundHoldings()]); };
$('northboundRefreshBtn')?.addEventListener('click', loadNorthboundFlow);

// ── 市场舆情 ──
async function loadNorthFlow() {
  setStatus('加载市场舆情...');
  try {
    const d = await api('/api/sentiment_market');
    if (!d.ok) throw new Error(d.error);
    const data = d.data || {};
    if (!data || typeof data !== 'object') throw new Error('市场舆情返回格式无效');
    const sentimentStatus = data.status || (data.total != null || Array.isArray(data.items) ? 'partial' : 'missing');
    const sentimentItems = Array.isArray(data.items) ? data.items : [];
    $('northMetrics').innerHTML = [['舆情指数', data.score ?? '-'], ['热度样本', data.total ?? '-'], ['热度上涨', data.hot_up ?? '-'], ['热度下跌', data.hot_down ?? '-'], ['拥挤状态', data.crowded == null ? '未知' : data.crowded?'偏热':'正常'], ['数据状态', sentimentStatus]].map(([l,v]) => `<div class="metric"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join('');
    $('sentimentNarrative').innerHTML = `<div class="empty">${esc(data.note||'市场舆情已更新')}<br>来源：股吧/微博、百度热搜、雪球；B站/小红书按实际采集状态显示，不把缺失填成0。</div>`;
    renderTable('northTable', sentimentItems, ['source','name','heat','change']);
    setStatus('市场舆情已更新');
  } catch(e) {
    if ($('northMetrics')) $('northMetrics').innerHTML = `<div class="data-state error">市场舆情不可用：${esc(e.message)}</div>`;
    if ($('sentimentNarrative')) $('sentimentNarrative').innerHTML = '';
    renderTable('northTable', [], ['source','name','heat','change']);
    setStatus('市场舆情失败: '+e.message);
  }
}
$('northRefreshBtn')?.addEventListener('click', loadNorthFlow);

// ── 板块轮动 ──
async function loadSectorRotation() {
  setStatus('加载板块轮动...');
  try {
    const d = await api('/api/sector_rotation');
    if (!d.ok) throw new Error(d.error);
    const rows = d.data || [];
    const ranked = rows.slice().sort((a,b) => Number(b.change_pct || 0) - Number(a.change_pct || 0));
     const extremes = [...ranked.slice(0, 15), ...ranked.slice(-15)].filter((row, index, all) => all.findIndex(x => x.name === row.name) === index);
     renderTable('srTable', extremes, ['name','change_pct','price','change']);
    const dom = $('srChart');
    if (dom && rows.length) {
      const c = echarts.init(dom);
      const top10 = extremes.slice(0,15);
      const names = top10.map(r=>r.name||'');
      const vals = top10.map(r=>r.change_pct||0);
      c.setOption({
        title:{text:'行业板块涨跌幅',left:'center',textStyle:{color:'#e5e7eb',fontSize:14}},
        tooltip:{trigger:'axis',formatter:function(p){return p[0].name+': '+p[0].value+'%'}},
        xAxis:{type:'category',data:names,axisLabel:{color:'#9ca3af',rotate:30}},
        yAxis:{type:'value',axisLabel:{color:'#9ca3af',formatter:'{value}%'}},
        series:[{type:'bar',data:vals,itemStyle:{color:function(p){return p.value>=0?'#b42318':'#087443'}}}]
      });
      window.addEventListener('resize',()=>c.resize());
    }
    setStatus('板块轮动已更新');
    // 板块热力图
    document.dispatchEvent(new CustomEvent('visual:sector', { detail: rows }));
  } catch(e) { setStatus('板块轮动失败: '+e.message); }
}
let _flowCatalog = null;
const _flowCharts = new Map();

function flowNumber(value, digits = 2) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : '-';
}

function flowChart(id) {
  const element = $(id);
  if (!element || typeof echarts === 'undefined') return null;
  const prior = _flowCharts.get(id);
  if (prior) { prior.dispose(); _flowCharts.delete(id); }
  const chart = echarts.init(element);
  _flowCharts.set(id, chart);
  return chart;
}

function flowMetric(label, value, note = '') {
  return `<div class="metric"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div>${note ? `<div class="sub">${esc(note)}</div>` : ''}</div>`;
}

function flowList(rows, metric, direction, limit) {
  const selected = (rows || []).filter(row => direction === 'in' ? Number(row[metric]) > 0 : Number(row[metric]) < 0)
    .sort((a, b) => direction === 'in' ? Number(b[metric]) - Number(a[metric]) : Number(a[metric]) - Number(b[metric]))
    .slice(0, limit);
  if (!selected.length) return '<div class="empty">当前数据没有可展示的方向性记录</div>';
  return selected.map((row, index) => `<div class="kv"><span>${index + 1}. ${esc(row.name || row.code || '未命名')}</span><strong class="${direction === 'in' ? 'flow-in' : 'flow-out'}">${Number(row[metric]) >= 0 ? '+' : ''}${flowNumber(row[metric])} 亿净额</strong></div>`).join('');
}

async function loadFlowCatalog() {
  if (_flowCatalog) return _flowCatalog;
  const response = await api('/api/intraday_flow_catalog', undefined, { cache_ttl: 300, silent: true });
  _flowCatalog = response.items || [];
  const select = $('flowDate');
  if (select) {
    const current = select.value;
    const options = ['<option value="">最新可用日</option>'].concat(_flowCatalog.map(item => {
      const suffix = item.complete_day ? '完整日' : `快照${item.snapshot_count || 0}`;
      return `<option value="${esc(item.date)}">${esc(item.date)} · ${suffix}</option>`;
    }));
    select.innerHTML = options.join('');
    if (current && _flowCatalog.some(item => item.date === current)) select.value = current;
  }
  return _flowCatalog;
}

function renderFlowChart(series, dates, unit, title) {
  // 零轴条形图：最新时点正/负净额极值，流入向右、流出向左。
  const chart = flowChart('flowChart');
  if (!chart) return;
  const latestValues = series.map(item => ({name: item.name, value: Number(item.data.at(-1)) || 0}))
    .filter(item => Number.isFinite(item.value))
    .sort((a,b) => b.value - a.value);
  const inflows = latestValues.filter(item => item.value > 0).slice(0, 15).reverse();
  const outflows = latestValues.filter(item => item.value < 0).slice(0, 15);
  const names = [...outflows.map(item => item.name).reverse(), ...inflows.map(item => item.name)];
  const values = [...outflows.map(item => item.value).reverse(), ...inflows.map(item => item.value)];
  chart.setOption({
    backgroundColor: 'transparent',
    title: { text: title, left: 8, top: 6, textStyle: { color: '#c9d1d9', fontSize: 13, fontWeight: 600 } },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: params => {
      const row = params && params[0];
      return row ? `${esc(row.name)}<br/>净额 ${row.value > 0 ? '+' : ''}${flowNumber(row.value)} ${unit}` : '';
    }},
    grid: { left: 90, right: 30, top: 60, bottom: 60 },
    xAxis: { type: 'value', name: unit, axisLabel: { color: '#8b949e' }, splitLine: { lineStyle: { color: '#21262d' } } },
    yAxis: { type: 'category', data: names, axisLabel: { color: '#8b949e' }, axisLine: { lineStyle: { color: '#30363d' } } },
    series: [{ name: '净额', type: 'bar', barWidth: '62%', data: values,
      itemStyle: { color: params => params.value >= 0 ? '#f6465d' : '#2ebd85' },
      label: { show: true, position: 'right', formatter: params => `${params.value > 0 ? '+' : ''}${flowNumber(params.value)}`, color: '#8b949e', fontSize: 10 } }]
  }, true);
}

function renderFlowWorkbench(rows, config) {
  const metric = config.metric;
  const topN = Number($('flowTopN')?.value || 12);
  const latestKey = config.latestKey;
  const latestValue = rows.map(row => row[latestKey]).filter(Boolean).sort().at(-1);
  const latest = rows.filter(row => row[latestKey] === latestValue);
  const inflow = latest.filter(row => Number(row[metric]) > 0).sort((a, b) => Number(b[metric]) - Number(a[metric]));
  const outflow = latest.filter(row => Number(row[metric]) < 0).sort((a, b) => Number(a[metric]) - Number(b[metric]));
  const rank = [...inflow, ...outflow].slice(0, topN);
  const labels = [...new Set(rows.map(row => row[latestKey]).filter(Boolean))].sort();
  const names = rank.map(row => row.name || row.code || '未命名');
  const series = names.map(name => ({ name, data: labels.map(label => {
    const row = rows.find(item => item[latestKey] === label && (item.name || item.code || '未命名') === name);
    return row ? Number(row[metric]) : null;
  }) }));
  renderFlowChart(series, labels, config.unit, config.title);
  if ($('flowInList')) $('flowInList').innerHTML = flowList(latest, metric, 'in', topN);
  if ($('flowOutList')) $('flowOutList').innerHTML = flowList(latest, metric, 'out', topN);
  renderTable('flowTable', latest.sort((a, b) => Number(b[metric] || 0) - Number(a[metric] || 0)), config.columns);
  return { latestValue, latest, labels, names };
}

async function loadFlowWorkbench(forceLatest = false) {
  setStatus('加载资金与结构工作台...');
  try {
    const catalog = await loadFlowCatalog();
    const mode = $('flowMode')?.value || 'daily';
    const chosenDate = forceLatest ? '' : ($('flowDate')?.value || '');
    let rows = [], config, meta = '', metrics = [];
    if (mode === 'intraday_real') {
      const payload = await api('/api/intraday_sector_flow?type=行业', undefined, { cache_ttl: 60, silent: true });
      rows = payload.rows || [];
      if (!rows.length) throw new Error(payload.error || '盘中真实行业资金尚未采集');
      config = { metric: 'main_net_yi', latestKey: 'timestamp', unit: '亿', title: '盘中行业真实主力净流入', columns: ['timestamp', 'name', 'main_net_yi', 'type'] };
      metrics = [flowMetric('实际数据日', String(payload.as_of || '-').slice(0, 19)), flowMetric('对象数', rows.length), flowMetric('资金口径', '真实主力净流入', '行业级快照'), flowMetric('数据类型', payload.type || '行业')];
      meta = `来源：${srcCn(payload.source)}。该层为盘中行业主力净流入真实快照，不与成交额方向代理混用；当前仅展示最近一次采集时点。`;
    } else if (mode === 'daily') {
      const snapshot = await api(`/api/research_fusion${chosenDate ? `?date=${encodeURIComponent(chosenDate)}&refresh=true` : '?refresh=true'}`, undefined, { cache_ttl: 0, silent: true, refresh: true });
      const flow = snapshot.sector_rotation || {};
      rows = flow.history || [];
      if (!rows.length) throw new Error('日频行业/概念真实净流入历史不存在');
      config = { metric: flow.metric || 'net_yi', latestKey: 'date', unit: '亿', title: '日频行业/概念真实净流入滚动', columns: ['date', 'name', flow.metric || 'net_yi'] };
      const current = rows.map(row => String(row.date || '').slice(0, 10)).filter(Boolean).sort().at(-1);
      const snapshotDate = String(snapshot.as_of || snapshot.requested_date || '').slice(0, 10);
      const flowDate = String(flow.as_of || current || '').slice(0, 10);
      const lagDays = snapshotDate && flowDate ? Math.round((Date.parse(snapshotDate) - Date.parse(flowDate)) / 86400000) : 0;
      const freshness = lagDays > 0 ? '数据较旧' : statusCn(flow.status || snapshot.data_status?.status);
      metrics = [flowMetric('快照交易日', snapshotDate || '-'), flowMetric('资金数据日', flowDate || '-'), flowMetric('滞后', lagDays > 0 ? `${lagDays} 日` : '0 日'), flowMetric('数据状态', freshness)];
      meta = `来源：${srcCn(flow.source || snapshot.data_status?.sources?.sector_fund_flow?.source)}。行业真实资金为日频数据；快照交易日 ${snapshotDate || '-'}，资金实际数据日 ${flowDate || '-'}。${lagDays > 0 ? '该资金源未更新到快照交易日，不作为当日交易依据。' : ''}`;
    } else {
      const fallbackDate = catalog.find(item => item.status === 'available')?.date || '';
      const date = chosenDate || fallbackDate;
      if (!date) throw new Error('没有可回放的盘中资金产物');
      const payload = await api(`/api/intraday_flow?date=${encodeURIComponent(date)}`, undefined, { cache_ttl: 30, silent: true });
      if (payload.status !== 'available') throw new Error(payload.error || '盘中资金产物不可用');
      if (mode === 'intraday') {
        rows = payload.industry || [];
        config = { metric: 'active_flow_proxy_yi', latestKey: 'time', unit: '亿', title: '盘中行业成交额方向代理', columns: ['time', 'name', 'amount_delta_yi', 'active_flow_proxy_yi', 'avg_pct', 'breadth'] };
        metrics = [flowMetric('回放日', payload.date), flowMetric('快照', `${payload.snapshot_count} 次`), flowMetric('市场覆盖', payload.market_complete_day ? '完整' : '部分'), flowMetric('ETF覆盖', payload.etf_complete_day ? '完整' : '部分/缺失'), flowMetric('资金口径', '方向代理', '非真实主力净流入')];
      } else if (mode === 'market') {
        rows = payload.market || [];
        config = { metric: 'active_flow_proxy_yi', latestKey: 'time', unit: '亿', title: '盘中全市场成交结构', columns: ['time', 'amount_yi', 'amount_delta_yi', 'active_flow_proxy_yi', 'advance', 'decline', 'breadth'] };
        metrics = [flowMetric('回放日', payload.date), flowMetric('快照', `${payload.snapshot_count} 次`), flowMetric('市场覆盖', payload.market_complete_day ? '完整' : '部分'), flowMetric('ETF覆盖', payload.etf_complete_day ? '完整' : '部分/缺失'), flowMetric('上涨广度', `${flowNumber((rows.at(-1) || {}).breadth * 100, 1)}%`)];
      } else {
        rows = (payload.etf || {}).rows || [];
        config = { metric: 'active_flow_proxy_yi', latestKey: 'time', unit: '亿', title: '盘中ETF成交额方向代理', columns: ['time', 'code', 'name', 'amount_yi', 'amount_delta_yi', 'active_flow_proxy_yi', 'pct_chg'] };
        metrics = [flowMetric('回放日', payload.date), flowMetric('ETF状态', statusCn((payload.etf || {}).status)), flowMetric('ETF快照', `${(payload.etf || {}).snapshot_count || 0} 次`), flowMetric('资金口径', '方向代理', '非真实净申购/赎回')];
      }
      if (!rows.length) throw new Error((payload.etf || {}).reason || '该范围没有可用盘中记录');
      meta = `来源：${payload.source}。${payload.methodology?.active_flow_proxy_yi || '成交额方向代理'}；数据日 ${payload.date}，市场${payload.market_complete_day ? '完整' : '部分'}、ETF${payload.etf_complete_day ? '完整' : '部分/缺失'}覆盖。`;
    }
    const rendered = renderFlowWorkbench(rows, config);
    if ($('flowMetrics')) $('flowMetrics').innerHTML = metrics.join('');
    if ($('flowMeta')) $('flowMeta').textContent = `${meta} 最新时点排名先分列流入/流出，再按净额排序；曲线展示流入优先的前 ${rendered.names.length} 个对象，可缩放、悬停比较。`;
    setStatus('资金与结构工作台已更新');
  } catch (error) {
    const message = error.message || String(error);
    if ($('flowMetrics')) $('flowMetrics').innerHTML = flowMetric('状态', '不可用', message);
    if ($('flowMeta')) $('flowMeta').textContent = `资金工作台加载失败：${message}`;
    if ($('flowInList')) $('flowInList').innerHTML = '<div class="empty">无可用数据</div>';
    if ($('flowOutList')) $('flowOutList').innerHTML = '<div class="empty">无可用数据</div>';
    setStatus(`资金工作台加载失败：${message}`);
  }
}

async function loadAllDayIndustry() {
  setStatus('加载盘中行业资金...');
  const dom = $('industryFlowChart');
  try {
    const payload = await api('/api/intraday_sector_flow?type=行业', undefined, { cache_ttl: 60, silent: true });
    const rows = payload.rows || [];
    if (!rows.length) throw new Error(payload.error || '暂无盘中行业资金快照');
    const metric = payload.metric || 'main_net_yi';
    const latest = rows.slice().sort((a,b) => String(b.timestamp || b.time).localeCompare(String(a.timestamp || a.time)))[0];
    const latestTime = latest?.timestamp || latest?.time;
    const current = rows.filter(row => (row.timestamp || row.time) === latestTime).sort((a,b) => Number(b[metric] || 0) - Number(a[metric] || 0));
    const chartRows = current.slice(0, 10);
    renderTable('srTable', chartRows, ['name', metric, 'timestamp']);
     if ($('industryFlowMeta')) $('industryFlowMeta').textContent = `盘中行业真实资金净流向 · 最近时点 ${latestTime || '-'} · 默认展示 ${chartRows.length} 条，可在资金结构页查看历史`;
    if (dom && typeof echarts !== 'undefined') {
      const chart = echarts.init(dom);
      const labels = [...new Set(rows.map(row => row.timestamp || row.time).filter(Boolean))].sort();
      const names = [...new Set(current.slice(0, 12).map(row => row.name || row.code))];
      chart.setOption({tooltip:{trigger:'axis'},legend:{type:'scroll',textStyle:{color:'#9ca3af'}},grid:{left:70,right:20,top:55,bottom:80},xAxis:{type:'category',data:labels,axisLabel:{color:'#9ca3af',rotate:35}},yAxis:{type:'value',name:'净流入(亿)',axisLabel:{color:'#9ca3af'}},dataZoom:[{type:'inside'},{type:'slider',bottom:8}],series:names.map(name=>({name,type:'line',smooth:true,showSymbol:false,data:labels.map(label=>{const row=rows.find(item=>(item.timestamp || item.time)===label&&(item.name||item.code)===name);return row ? Number(row[metric]) : null;})}))},true);
      window.addEventListener('resize', () => chart.resize());
    }
    if ($('industryFlowMeta')) $('industryFlowMeta').textContent = `盘中行业真实资金净流向 · 数据日 ${String(payload.as_of || latestTime || '-').slice(0, 10)} · 最近采集 ${latestTime || '-'} · ${srcCn(payload.source)}`;
    setStatus(`盘中行业资金已更新 · ${latestTime || '-'}`);
  } catch (e) {
    if (dom) dom.innerHTML = `<div class="data-state empty">${esc(e.message)}。日频资金数据不会在此图中冒充盘中数据。</div>`;
    if ($('srTable')) $('srTable').innerHTML = '<tbody><tr><td class="empty">暂无盘中行业快照</td></tr></tbody>';
    if ($('industryFlowMeta')) $('industryFlowMeta').textContent = '当前未取得盘中行业真实资金快照；请检查采集任务状态。';
    setStatus('盘中行业资金不可用: ' + e.message);
  }
}
// btDashboardRefresh / newsRefreshBtn / srRefreshBtn 已在顶部统一绑定

// ── 市场温度 ──
async function loadMarketTemp() {
  setStatus('加载市场温度...');
  try {
    const d = await api('/api/market_temp');
    if (!d.ok) throw new Error(d.error);
    const data = d.data || {};
    const temp = data.temperature;
    if (temp == null || !Number.isFinite(Number(temp))) throw new Error('市场温度缺少有效数值，未用默认值填充');
    // Display key metrics
    const items = [
      ['市场温度', temp+'°'],
      ['上涨占比', data.advance_pct+'%'],
      ['上涨/下跌', data.advance+'/'+data.decline],
      ['上证指数', data.sh_index?.toFixed(1)+' ('+data.sh_pct?.toFixed(2)+'%)'],
      ['深证成指', data.sz_index?.toFixed(1)+' ('+data.sz_pct?.toFixed(2)+'%)'],
      ['创业板指', data.cy_index?.toFixed(1)+' ('+data.cy_pct?.toFixed(2)+'%)'],
      ['成交额(亿)', data.total_amount_yi?.toFixed(0)],
      ['MA300上方%', data.pct_above_ma300+'%'],
      ['MA60方向', data.ma60_direction],
    ];
    $('mtMetrics').innerHTML = items.map(([l,v]) => `<div class="metric"><div class="label">${l}</div><div class="value">${v}</div></div>`).join('');
    const dom = $('mtGauge');
    if (dom) {
      const c = echarts.init(dom);
      c.setOption({
        series:[{
          type:'gauge',center:['50%','60%'],radius:'80%',
          min:0,max:100,splitNumber:5,
          axisLine:{lineStyle:{color:[[0.3,'#087443'],[0.7,'#b45309'],[1,'#b42318']],width:12}},
          detail:{formatter:'{value}',color:'#e5e7eb',fontSize:24},
          data:[{value:temp,name:'市场温度'}],
          title:{color:'#9ca3af',fontSize:14},
        }]
      });
      window.addEventListener('resize',()=>c.resize());
    }
    setStatus('市场温度已更新');
  } catch(e) {
    if ($('mtMetrics')) $('mtMetrics').innerHTML = `<div class="data-state error">市场温度不可用：${esc(e.message)}</div>`;
    if ($('mtGauge')) $('mtGauge').innerHTML = '<div class="data-state empty">未取得有效温度，不展示伪默认值。</div>';
    setStatus('市场温度失败: '+e.message);
  }
}
$('mtRefreshBtn')?.addEventListener('click', loadMarketTemp);

// ── 基本面 ──
async function loadFundamentals() {
  const symbol = ($('fundSymbol')?.value || '').trim() || '002714';
  setStatus('查询基本面...');
  try {
    const d = await api(`/api/fundamental?symbol=${encodeURIComponent(symbol)}`);
    if (!d.ok) throw new Error(d.error);
    const data = d.data || {};
    const items = Object.entries(data).map(([k,v]) => [k, typeof v==='number'?v.toFixed(2):v||'-']);
    $('fundMetrics').innerHTML = items.map(([l,v]) => `<div class="metric"><div class="label">${l}</div><div class="value">${v}</div></div>`).join('');
    setStatus('基本面已更新');
  } catch(e) { setStatus('基本面失败: '+e.message); }
}
// fundRefreshBtn / mtRefreshBtn / regimeRefreshBtn / oppRefreshBtn 已在顶部统一绑定

// ── K线形态 ──
async function loadPatterns() {
  const symbol = sharedSymbol();
  if ($('patternSymbol')) $('patternSymbol').value = symbol;
  const start = ($('patternStart')?.value || '').trim() || '20240601';
  setStatus('加载K线形态...');
  try {
    const d = await api(`/api/chart_patterns?symbol=${encodeURIComponent(symbol)}&start=${start}`);
    if (!d.ok) throw new Error(d.error);
    const data = d.data || {};
    const summary = data.summary || {形态数量:data.count ?? 0, 数据区间:data.range || start, 数据状态:data.status || 'available'};
    const items = Object.entries(summary).map(([k,v]) => [k,typeof v==='number'?v.toFixed(2):v]);
    $('patternMetrics').innerHTML = items.map(([l,v]) => `<div class="metric"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join('');
    const patterns = patternRows(data.patterns || []);
    if (patterns.length) renderTable('patternTable', patterns, ['日期','形态','方向','置信度','形态说明']);
    else $('patternTable').innerHTML = '<tbody><tr><td><div class="data-state empty">所选区间未识别出K线形态。这表示识别结果为空，不是数据请求失败。</div></td></tr></tbody>';
    setStatus(`${symbol} K线形态已更新 · ${patterns.length} 条`);
  } catch(e) {
    $('patternMetrics').innerHTML = '';
    $('patternTable').innerHTML = `<tbody><tr><td><div class="data-state error"><strong>K线形态加载失败</strong><span>${esc(e.message)}</span></div></td></tr></tbody>`;
    setStatus('K线形态失败: '+e.message);
  }
}
$('patternLoadBtn')?.addEventListener('click', loadPatterns);

// ── 宏观日历 ──
async function loadMacroCal() {
  const month = ($('macroCalMonth')?.value || '').trim();
  setStatus('加载宏观日历...');
  try {
    const url = '/api/macro_calendar' + (month ? `?month=${encodeURIComponent(month)}` : '?refresh=true');
    const d = await api(url, undefined, { cache_ttl: 0, refresh: true });
    if (!d.ok) throw new Error(d.error);
    const events = d.data?.events || d.data || [];
    renderTable('macroCalTable', Array.isArray(events) ? events.map(e => ({ date: e.publish_date || e.date, event: e.name || e.event, country: e.source || e.country, importance: e.is_today ? '今日' : e.is_past ? '已发布' : (e.days_until != null ? `${e.days_until}天后` : '-') })) : [events], ['date','event','country','importance']);
    setStatus('宏观日历已更新');
  } catch(e) { setStatus('宏观日历失败: '+e.message); }
}
$('macroCalRefreshBtn')?.addEventListener('click', loadMacroCal);

// ── 财报日历 ──
async function loadEarnings() {
  const symbol = ($('calendarEarningsSymbol')?.value || $('earningsSymbol')?.value || '').trim() || '002714';
  setStatus('加载财报日历...');
  try {
    const d = await api(`/api/earnings?symbol=${encodeURIComponent(symbol)}`);
    if (!d.ok) throw new Error(d.error);
    const data = d.data?.list || d.data || [];
    renderTable('earningsTable', Array.isArray(data) ? data.slice().sort((a,b) => String(b.announce_date || b.date || '').localeCompare(String(a.announce_date || a.date || ''))).map(e => ({ date: e.announce_date || e.date, report_type: e.forecast_type || e.report_type, eps: e.forecast_value ?? e.eps, revenue: e.change_range == null ? '-' : `同比 ${e.change_range}%`, profit: e.forecast_metric || e.profit, summary: e.summary || e.change_type })) : [data], ['date','report_type','eps','revenue','profit','summary']);
    setStatus('财报日历已更新');
  } catch(e) { setStatus('财报日历失败: '+e.message); }
}
$('earningsLoadBtn')?.addEventListener('click', loadEarnings);
$('earningsRefreshBtn')?.addEventListener('click', loadEarnings);

// ── 机会发现 ──
async function loadOpportunities() {
  setStatus('扫描机会...');
  try {
    const d = await api('/api/opportunity');
    if (!d.ok) throw new Error(d.error);
    const rows = d.data?.opportunities || d.data || [];
    renderTable('oppTable', rows, ['symbol','score','reason','timeframe']);
    const dom = $('oppChart');
    if (dom && rows.length) {
      const c = echarts.init(dom);
      c.setOption({
        title:{text:'机会评分Top10',left:'center',textStyle:{color:'#e5e7eb',fontSize:14}},
        xAxis:{type:'category',data:rows.slice(0,10).map(r=>r.symbol||''),axisLabel:{color:'#9ca3af',rotate:30}},
        yAxis:{type:'value',axisLabel:{color:'#9ca3af'}},
        series:[{type:'bar',data:rows.slice(0,10).map(r=>r.score||0),itemStyle:{color:'#1f7a5c'}}]
      });
      window.addEventListener('resize',()=>c.resize());
    }
    setStatus('机会扫描完成');
  } catch(e) { setStatus('机会扫描失败: '+e.message); }
}
$('oppRefreshBtn')?.addEventListener('click', loadOpportunities);

// ── 组合风险 ──
function fmtPct(v) { return (typeof v==='number') ? v.toFixed(2)+'%' : (v||'-'); }
function fmtNum(v) { return (typeof v==='number') ? v.toFixed(2) : (v||'-'); }
function riskBadge(val) {
  if (typeof val!=='number') return '';
  if (val>=8) return '<span class="risk-badge danger">🔴 高</span>';
  if (val>=5) return '<span class="risk-badge warn">🟡 中</span>';
  return '<span class="risk-badge safe">🟢 低</span>';
}

function renderRiskSection(title, icon, fields, level) {
  const cls = level ? ` risk-level-${level}` : '';
  let rows = fields.map(([k,v]) => {
    let disp;
    if (typeof v==='number') disp = v.toFixed(2) + (k.includes('比率')||k.includes('pct')||k.includes('占比')||k.match(/^[a-z_]*[Pp]ct/) ? '%' : '');
    else if (v===null || v===undefined) disp = '-';
    else if (typeof v==='boolean') disp = v ? '是' : '否';
    else disp = String(v);
    return `<tr><td class="rl">${k}</td><td class="rv">${disp}</td></tr>`;
  }).join('');
  return `<div class="risk-section${cls}">
    <div class="risk-section-header" onclick="this.parentElement.classList.toggle('risk-collapsed')">
      <span class="risk-arrow">▾</span>${icon} ${title}
    </div>
    <div class="risk-section-body"><table class="risk-table">${rows}</table></div>
  </div>`;
}

async function loadPortfolioRisk() {
  setStatus('计算组合风险...');
  try {
    const d = await api('/api/portfolio_risk');
    if (!d.ok) throw new Error(d.error);
    const data = d.data || {};
    const el = $('prMetrics');
    if (!el) return;

    let html = '';
    // ── 仓位概览卡片 ──
    const pc = data.position_count ?? '-';
    const ts = data.timestamp ? data.timestamp.replace('T',' ').slice(0,19) : '-';
    html += `<div class="risk-summary">`;
    html += [
      ['📦 持仓数', pc, pc === 0 ? '明确无持仓' : pc === '-' ? '数量未知' : '风险分析覆盖'],
      ['🕐 更新时间', ts, ''],
    ].map(([icon,val,note]) => `<div class="metric"><div class="label">${icon}</div><div class="value">${val}</div><div class="muted">${note}</div></div>`).join('');
    html += `</div>`;

    // ── VaR ──
    const v = data.var || {};
    if (Object.keys(v).length) {
      html += renderRiskSection('VaR 分析', '📉', [
        ['日VaR(95%)', v.var_95], ['日VaR(99%)', v.var_99], ['CVaR(95%)', v.cvar_95],
        ['VaR触发', v.var_threshold_breached], ['波动率(年化)', v.volatility],
      ], v.var_threshold_breached ? 'high' : '');
    }

    // ── 集中度 ──
    const c = data.concentration || {};
    if (Object.keys(c).length) {
      const indRows = Object.entries(c.industry_map||{}).map(([ind,pct]) => [ind, pct+'%']);
      const warnings = (c.warnings||[]).map(w => ['⚠️', w]);
      const fields = [
        ['最重仓', c.max_single_name], ['权重', c.max_single],
        ['前5权重', c.top5_weight], ['行业数', c.n_industries],
        ...indRows, ...warnings
      ];
      const concLevel = (c.max_single||0) > 30 ? 'high' : (c.max_single||0) > 20 ? 'medium' : '';
      html += renderRiskSection('集中度分析', '📊', fields, concLevel);
    }

    // ── 回撤 ──
    const dd = data.drawdown || {};
    if (Object.keys(dd).length) {
      html += renderRiskSection('回撤分析', '📉', [
        ['当前回撤', dd.current_drawdown], ['最大回撤', dd.max_drawdown],
        ['回撤级别', dd.level], ['阈值', dd.threshold],
        ['建议', dd.suggested_action],
      ], dd.level?.includes('警告')||dd.level?.includes('danger') ? 'high' : '');
    }

    // ── 相关性 ──
    const corr = data.correlation || {};
    if (Object.keys(corr).length) {
      const corrLevel = (corr.avg_corr||0) > 0.7 ? 'high' : (corr.avg_corr||0) > 0.5 ? 'medium' : '';
      html += renderRiskSection('相关性分析', '🔗', [
        ['平均', corr.avg_corr], ['最大', corr.max_corr], ['最小', corr.min_corr],
        ...Object.entries(corr.corr_matrix||{}).slice(0,10).map(([k,v]) => [k, fmtNum(v)]),
      ], corrLevel);
    }

    // ── Kelly ──
    const kelly = data.kelly || {};
    if (Object.keys(kelly).length) {
      html += renderRiskSection('Kelly 仓位', '🎯', [
        ['胜率', kelly.win_rate], ['盈亏比', kelly.profit_loss_ratio],
        ['全Kelly', kelly.kelly_pct], ['半Kelly(推荐)', kelly.half_kelly],
        ['¼Kelly(保守)', kelly.quarter_kelly],
      ]);
    }

    // ── ML风险评分 ──
    const mlr = data.ml_risk || {};
    if (mlr.ml_risk_score !== undefined) {
      const score = mlr.ml_risk_score;
      const level = mlr.ml_risk_level || '未知';
      const clr = score >= 55 ? '#b42318' : score >= 40 ? '#d97706' : '#087443';
      const confidence = (mlr.ml_confidence*100).toFixed(0);
      html += `<div class="risk-section risk-level-${score>=55?'high':score>=40?'medium':''}">
        <div class="risk-section-header" onclick="this.parentElement.classList.toggle('risk-collapsed')" style="background:rgba(0,0,0,0.1)">
          <span class="risk-arrow">▾</span>🤖 ML市场风险评分
          <span style="margin-left:auto;font-weight:700;color:${clr};font-size:16px;">${score}分 - ${level}</span>
        </div>
        <div class="risk-section-body">
          <div style="margin:12px 0;display:flex;gap:20px;align-items:center;flex-wrap:wrap;">
            <div style="flex:1;min-width:150px;">
              <div style="height:10px;background:#243041;border-radius:5px;overflow:hidden;">
                <div style="height:100%;width:${score}%;border-radius:5px;background:linear-gradient(90deg,#087443,#d97706,#b42318);transition:width 0.6s;"></div>
              </div>
              <div style="display:flex;justify-content:space-between;margin-top:4px;font-size:11px;color:var(--muted);">
                <span>安全 0</span><span>${score}</span><span>100 高风险</span>
              </div>
            </div>
            <div style="text-align:center;min-width:80px;">
              <div style="color:var(--muted);font-size:11px;">置信度</div>
              <div style="font-size:20px;font-weight:700;">${confidence}%</div>
            </div>
          </div>
          <table class="risk-table">
            <tr><td class="rl">市场温度</td><td class="rv">${mlr.market_temperature||'-'}</td></tr>
            <tr><td class="rl">广度比</td><td class="rv">${mlr.breadth_ratio||'-'}</td></tr>
            <tr><td class="rl">144线上</td><td class="rv">${mlr.pct_above_ma144||'-'}%</td></tr>
            <tr><td class="rl">300线上</td><td class="rv">${mlr.pct_above_ma300||'-'}%</td></tr>
            <tr><td class="rl">市场阶段</td><td class="rv">${esc(mlr.market_regime||'')}</td></tr>
            <tr><td class="rl">触发信号</td><td class="rv" style="font-size:11px;color:var(--muted);">${(mlr.signals||[]).join(', ')}</td></tr>
          </table>
        </div>
      </div>`;
    }

    // ── ML风险评分（整合 ml_signal 到仓位风控） ──
    const positions = data.positions || [];
    if (positions.length) {
      html += `<div class="risk-section">
        <div class="risk-section-header" onclick="this.parentElement.classList.toggle('risk-collapsed')">
          <span class="risk-arrow">▾</span>🤖 ML风险扫描（持仓）
        </div>
        <div class="risk-section-body" id="prMLBody">
          <div class="loading-spinner" style="margin:12px auto"></div>
        </div>
      </div>`;
    }

    el.innerHTML = html;

    // ── 持仓ML并行扫描 ──
    if (positions.length) {
      const codes = positions.map(p => (p.code || p.symbol || '').trim()).filter(Boolean).slice(0,15);
      if (codes.length) {
        try {
          const qs = codes.map(c => encodeURIComponent(c)).join(',');
          const resp = await api(`/api/ml_signal?mode=scan_batch&symbols=${qs}`);
          if (resp.ok && resp.data?.rows?.length) {
            const rows = resp.data.rows;
            rows.sort((a,b) => (b.ml_score||0) - (a.ml_score||0));
            const mlHtml = rows.map(r => {
              const score = Number(r.ml_score || r.confidence || 0);
              const rec = r.recommendation || r.signal || '-';
              const pct = score.toFixed(1);
              const color = score >= 70 ? '#087443' : score >= 40 ? '#d97706' : '#b42318';
              return `<div class="ml-risk-item">
                <span class="ml-risk-sym">${esc(r.symbol||'')}</span>
                <span class="ml-risk-score" style="color:${color}">${pct}%</span>
                <span class="ml-risk-rec">${esc(rec)}</span>
                <div class="ml-risk-bar"><div class="ml-risk-fill" style="width:${pct}%;background:${color}"></div></div>
              </div>`;
            }).join('');
            $('prMLBody').innerHTML = mlHtml;
          } else {
            $('prMLBody').innerHTML = '<div class="empty">暂无ML信号数据</div>';
          }
        } catch(e) {
          $('prMLBody').innerHTML = '<div class="empty">ML扫描暂不可用</div>';
        }
      }
    }

    // ── ECharts 图表 ──
    const dom = $('prChart');
    if (dom) {
      const chartData = [];
      if (v.var_95) chartData.push({name:'日VaR(95%)', value:Math.abs(v.var_95)});
      if (v.var_99) chartData.push({name:'日VaR(99%)', value:Math.abs(v.var_99)});
      if (c.max_single) chartData.push({name:'最重仓', value:c.max_single});
      if (c.top5_weight) chartData.push({name:'前5权重', value:c.top5_weight});
      if (dd.current_drawdown) chartData.push({name:'当前回撤', value:Math.abs(dd.current_drawdown)});

      if (chartData.length) {
        try {
          const chart = echarts.init(dom);
          chart.setOption({
            tooltip:{trigger:'item', formatter:'{b}: {c}'},
            series:[{
              type:'pie', radius:['30%','60%'],
              data:chartData,
              label:{color:'#e5e7eb', formatter:'{b}'},
              itemStyle:{
                color:['#2563eb','#059669','#d97706','#b42318','#8b5cf6','#6b7280']
              }
            }],
            title:{
              text:'风险指标概览', left:'center', top:'bottom',
              textStyle:{color:'#9ca3af', fontSize:12}
            }
          });
          window.addEventListener('resize',()=>chart.resize());
        } catch(e) { /* silent */ }
      }
    }

    setStatus('组合风险已更新 ✓ ML深度分析已集成');
    // 风险指标图表
    document.dispatchEvent(new CustomEvent('visual:risk', { detail: data }));
  } catch(e) { setStatus('组合风险失败: '+e.message); }
}
$('prRefreshBtn')?.addEventListener('click', loadPortfolioRisk);

// ── ML扫描批量入口（支持逗号分隔symbols参数） ──
async function loadMLSignal(mode='predict') {
  // 检查是否为批量模式
  if (mode === 'scan_batch') return; // 由 loadPortfolioRisk 内部调用
  const btn = mode === 'scan' ? 'mlSignalScanBtn' : 'mlSignalBtn';
  setBusy(btn, true); setStatus(mode === 'scan' ? '扫描ML信号...' : '加载ML预测...');
  try {
    const qs = new URLSearchParams({mode});
    if (mode === 'predict') qs.set('symbol', ($('mlSignalSymbol')?.value || '').trim() || ($('sSymbol')?.value || '').trim() || '600519');
    const d = await api(`/api/ml_signal?${qs.toString()}`);
    $('mlSignalResult').dataset.loaded = '1';
    if (mode === 'scan') {
      const data = d.data || {};
      const rows = data.rows || [];
      renderMLSignalList(rows, data.failures || []);
      setStatus(`ML扫描完成 (${rows.length}只)`);
    } else {
      renderMLSignalPredict(d.data || {});
      setStatus('ML预测完成');
    }
  } catch(e) {
    $('mlSignalResult').innerHTML = `<div class="empty">${esc(e.message)}</div>`;
    setStatus('ML信号失败: '+e.message);
  } finally { setBusy(btn, false); }
}

async function loadMLSignalImportance() {
  setBusy('mlImportanceBtn', true); setStatus('加载ML特征重要性...');
  try {
    const d = await api('/api/ml_signal?mode=importance');
    $('mlSignalResult').dataset.loaded = '1';
    renderMLImportance(d.data || {});
    setStatus('特征重要性已更新');
  } catch(e) {
    $('mlSignalResult').innerHTML = `<div class="empty">${esc(e.message)}</div>`;
    setStatus('特征重要性失败: '+e.message);
  } finally { setBusy('mlImportanceBtn', false); }
}

function renderMLSignalPredict(data) {
  const score = Number(data.ml_score ?? data.confidence ?? data.probability ?? 0);
  const rec = data.recommendation || data.signal || data.action || '-';
  const features = data.feature_importance || data.top_contributors || [];
  $('mlSignalMetrics').innerHTML = [
    ['代码', data.symbol || '-'], ['日期', data.date || '-'], ['价格', fmt(data.price)],
    ['信号', rec], ['置信度/评分', `${score.toFixed(1)}%`], ['特征数', data.n_features || '-'],
  ].map(([l,v])=>`<div class="metric"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join('');
  $('mlSignalResult').innerHTML = `
    <h2 style="margin-top:0;">${esc(rec)}</h2>
    <div style="margin:10px 0;color:#657282;">${(data.signals||[]).map(esc).join(' · ') || '暂无触发信号'}</div>
    ${mlFeatureBars(features.slice(0,10))}`;
  renderTable('mlSignalTable', [{
    symbol:data.symbol, date:data.date, price:data.price, recommendation:rec,
    ml_score:score, signals:(data.signals||[]).join(' / '), n_features:data.n_features,
  }], ['symbol','date','price','recommendation','ml_score','signals','n_features']);
}

function renderMLSignalList(rows, failures) {
  $('mlSignalMetrics').innerHTML = [
    ['成功', rows.length], ['失败', failures.length], ['Top评分', rows[0]?.ml_score ?? '-'],
  ].map(([l,v])=>`<div class="metric"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join('');
  $('mlSignalResult').innerHTML = rows.length ? mlFeatureBars((rows[0].top_contributors||[]).slice(0,10), `${rows[0].symbol || ''} Top特征`) : '<div class="empty">暂无ML扫描结果</div>';
  renderTable('mlSignalTable', rows.map(r=>({
    symbol:r.symbol, date:r.date, price:r.price, recommendation:r.recommendation,
    ml_score:r.ml_score, signals:(r.signals||[]).join(' / '), n_features:r.n_features,
  })), ['symbol','date','price','recommendation','ml_score','signals','n_features']);
}

function renderMLImportance(data) {
  const rows = data.feature_importance || [];
  $('mlSignalMetrics').innerHTML = [
    ['样本', data.samples ?? '-'], ['准确率', pct((data.accuracy ?? 0)*100)], ['精准率', pct((data.precision ?? 0)*100)],
    ['F1', pct((data.f1 ?? 0)*100)], ['特征数', data.n_features ?? rows.length], ['模型', data.has_xgboost ? 'XGBoost+RF' : 'RandomForest'],
  ].map(([l,v])=>`<div class="metric"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join('');
  $('mlSignalResult').innerHTML = mlFeatureBars(rows.slice(0,10), 'Top 10 特征重要性');
  renderTable('mlSignalTable', rows.slice(0,30).map(([feature, importance], i)=>({rank:i+1, feature, importance})), ['rank','feature','importance']);
}

function mlFeatureBars(items, title='Top 10 特征贡献') {
  if (!items || !items.length) return '<div class="empty">暂无特征数据</div>';
  const parsed = items.map(([name, value]) => ({name, value:Number(value)||0}));
  const max = Math.max(...parsed.map(x=>Math.abs(x.value)), 0.0001);
  return `<div><h2 style="margin-top:0;">${esc(title)}</h2>${parsed.map(x=>{
    const width = Math.max(3, Math.min(100, Math.abs(x.value)/max*100));
    const color = x.value >= 0 ? '#1f7a5c' : '#b42318';
    return `<div style="display:grid;grid-template-columns:minmax(120px,220px) 1fr 78px;gap:10px;align-items:center;margin:8px 0;">
      <div style="font-size:13px;color:#334155;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${esc(x.name)}</div>
      <div style="height:10px;background:#e5e7eb;border-radius:5px;overflow:hidden;"><div style="height:100%;width:${width}%;background:${color};"></div></div>
      <div style="font-variant-numeric:tabular-nums;text-align:right;color:#657282;">${x.value.toFixed(4)}</div>
    </div>`;
  }).join('')}</div>`;
}

function unwrapApiData(payload) {
  if (!payload) return {};
  if (payload.ok === false) throw new Error(payload.error || 'API error');
  return payload.data ?? payload;
}

function formatMacroNumber(value, digits = 2) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : (value ?? '-');
}

function renderTempGauge(payload) {
  const data = unwrapApiData(payload);
  const el = $('tempGaugeDisplay');
  if (!el) return;
  const temp = data.temperature ?? '-';
  const items = [
    ['市场温度', temp === '-' ? '-' : temp + '°'],
    ['上涨占比', data.advance_pct == null ? '-' : data.advance_pct + '%'],
    ['上涨/下跌', data.advance != null || data.decline != null ? `${data.advance ?? '-'}/${data.decline ?? '-'}` : '-'],
    ['成交额', data.total_amount_yi == null ? '-' : `${formatMacroNumber(data.total_amount_yi, 0)} 亿`],
  ];
  el.innerHTML = `<div class="macro-grid">${items.map(([l,v]) => `<div class="macro-card"><div class="name">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join('')}</div>`;
}

function renderPulseMargin(pulsePayload, marginPayload) {
  const pulse = unwrapApiData(pulsePayload);
  const margin = unwrapApiData(marginPayload);
  const el = $('pulseMarginDisplay');
  if (!el) return;
  // market_pulse API: { index: 62.0, level: '贪婪 📈', advice: '市场活跃', components: {...} }
  const fearGreed = pulse.index ?? pulse.fear_greed_index ?? pulse.fgi;
  const pulseLabel = pulse.level ?? pulse.market_pulse ?? pulse.pulse ?? pulse.status ?? '-';
  // margin API: { total_margin_balance, ss_margin_balance, sz_margin_balance }
  const sh = margin.ss_margin_balance ?? margin.sh_margin ?? margin.sh;
  const sz = margin.sz_margin_balance ?? margin.sz_margin ?? margin.sz;
  const total = margin.total_margin_balance ?? margin.total_margin ?? margin.total;
  const items = [
    ['恐贪指数', fearGreed == null ? '-' : `${formatMacroNumber(fearGreed, 1)}°`],
    ['市场脉搏', pulseLabel],
    ['两融余额', total == null ? '-' : `${formatMacroNumber(total, 0)} 亿`],
    ['沪/深两融', sh == null && sz == null ? '-' : `${formatMacroNumber(sh, 0)} / ${formatMacroNumber(sz, 0)} 亿`],
  ];
  el.innerHTML = `<div class="macro-grid">${items.map(([l,v]) => `<div class="macro-card"><div class="name">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join('')}</div>`;
}

function renderDashboardSentiment(payload) {
  const data = unwrapApiData(payload);
  const el = $('northFlowDisplay');
  if (!el) return;
  const score = data.score ?? '-';
  const total = data.total ?? '-';
  const up = data.hot_up ?? '-';
  const down = data.hot_down ?? '-';
  const note = data.note || Object.values(data.sources || {}).map(item => item.note).filter(Boolean).join('；') || '暂无舆情摘要';
  const status = data.status || (data.total != null || data.score != null ? 'partial' : 'missing');
  el.innerHTML = `<div class="macro-grid">
    <div class="macro-card"><div class="name">舆情指数</div><div class="value">${esc(score)}</div></div>
    <div class="macro-card"><div class="name">热度样本</div><div class="value">${esc(total)}</div></div>
    <div class="macro-card"><div class="name">上涨 / 下跌</div><div class="value">${esc(up)} / ${esc(down)}</div></div>
    <div class="macro-card"><div class="name">拥挤状态</div><div class="value">${data.crowded == null ? '未知' : data.crowded ? '偏热' : '正常'}</div></div>
    <div class="macro-card"><div class="name">数据状态</div><div class="value">${esc(status)}</div></div>
  </div><div class="empty" style="margin-top:8px">${esc(note)}</div>`;
}

function renderNorthBoundSummary(payload) {
  const el = $('northBoundDisplay') || $('northFlowDisplay');
  if (!el) return;
  const raw = unwrapApiData(payload), summary = raw.summary || raw, sourceRaw = raw.raw || raw;
  const disclosed = sourceRaw.net_buy_disclosed !== false && summary.net_buy_disclosed !== false;
  const hold = raw.holdings || {};
  el.innerHTML = `<div class="macro-grid"><div class="macro-card"><div class="name">资金流</div><div class="value">${disclosed ? esc(summary['当日净买入(亿)'] ?? summary.total_net_yi ?? '-') : '已停止披露'}</div></div><div class="macro-card"><div class="name">沪深通状态</div><div class="value">${esc(raw.trading_status || '口径确认')}</div></div><div class="macro-card"><div class="name">持股数据</div><div class="value">${hold.count == null ? '独立查询' : esc(hold.count) + '条'}</div></div></div><div class="empty">${esc(disclosed ? (raw.note || '资金流与持股结构分层展示') : '2024-08-19 起北向盘中/日频净买入停止披露；持股数量、市值、占流通股比例仍可独立查询。')} <button class="btn-sm" type="button" onclick="window.quickViewIntegrated('market','m-north')">查看持股结构 →</button></div>`;
}

function renderNorthFlow(payload) {
  const raw = unwrapApiData(payload);
  const data = raw.summary || raw;
  const el = $('northFlowDisplay');
  if (!el) return;
  // 北向API字段：中文键名
  const status = raw.trading_status || data['交易状态'] || (raw.trading ? '交易中' : '非交易时段');
  const total = data['当日净买入(亿)'] ?? data.total_net_yi ?? data.total ?? data['合计净流入'];
  const sh = data['沪股通净买入(亿)'] ?? data.sh_net_yi ?? data.sh ?? data['沪股通净流入'];
  const sz = data['深股通净买入(亿)'] ?? data.sz_net_yi ?? data.sz ?? data['深股通净流入'];
  const shLimit = data['沪股通额度余额(亿)'] ?? data.sh_limit_yi;
  const szLimit = data['深股通额度余额(亿)'] ?? data.sz_limit_yi;
  const isTrading = raw.trading === true || (Number(total) !== 0);
  const totalClass = (Number(total) || 0) >= 0 ? 'up' : 'down';
  const items = [
    ['交易状态', status],
  ];
  if (isTrading) {
    items.push(['沪股通', sh == null ? '-' : `${formatMacroNumber(sh, 2)} 亿`]);
    items.push(['深股通', sz == null ? '-' : `${formatMacroNumber(sz, 2)} 亿`]);
  } else {
    items.push(['沪股通', '- (休市)']);
    items.push(['深股通', '- (休市)']);
    if (shLimit != null) items.push(['沪额度余额', `${formatMacroNumber(shLimit, 0)} 亿`]);
    if (szLimit != null) items.push(['深额度余额', `${formatMacroNumber(szLimit, 0)} 亿`]);
  }
  el.innerHTML = `<div class="macro-grid">${items.map(([l,v]) => `<div class="macro-card"><div class="name">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join('')}</div>`;
}

// Core 统一调度宏观、盘面复盘和系统状态；避免重复轮询覆盖同一 DOM。

// ── 策略库 ──
async function loadSkills(query) {
  try {
    const d = await api('/api/skills?mode=list');
    if (!d.ok) throw new Error(d.error || 'skills error');
    const skills = d.skills || d.data || [];
    const el = $('skillsList');
    const count = $('skillsCount');
    if (count) count.textContent = `共 ${d.count || skills.length} 个策略`;
    if (!el) return;

    // Group by category
    const groups = {};
    for (const s of skills) {
      const cat = s.category || 'other';
      if (!groups[cat]) groups[cat] = [];
      groups[cat].push(s);
    }

    let html = '';
    for (const [cat, items] of Object.entries(groups).sort()) {
      html += `<div style="font-weight:600;color:#c8cdd3;margin:8px 0 4px;">${cat} (${items.length})</div>`;
      html += '<div style="display:flex;flex-wrap:wrap;gap:6px;">';
      for (const s of items) {
        html += `<div class="skill-chip" onclick="showSkillDetail('${s.name}')" style="background:#24272d;border:1px solid #3a3d45;border-radius:6px;padding:6px 10px;cursor:pointer;font-size:12px;">${s.name}</div>`;
      }
      html += '</div>';
    }
    el.innerHTML = html;
  } catch(e) {
    console.error('loadSkills error:', e);
  }
}

async function showSkillDetail(name) {
  const el = $('skillsDetail');
  if (!el) return;
  try {
    const d = await api('/api/skills?mode=get&name=' + encodeURIComponent(name));
    if (!d.ok) { el.innerHTML = `<div class="error">${d.error}</div>`; el.style.display='block'; return; }
    const content = (d.data?.content || d.skill?.content || '无内容');
    el.innerHTML = `<div style="font-weight:600;margin-bottom:6px;">${name}</div><pre style="white-space:pre-wrap;font-size:12px;color:#b0b5c0;max-height:400px;overflow:auto;">${content}</pre>`;
    el.style.display = 'block';
  } catch(e) {
    el.innerHTML = `<div class="error">${e.message}</div>`;
    el.style.display = 'block';
  }
}

// ── : 持仓管理 ──
async function loadPortfolio() {
  setBusy('pfRefreshBtn', true); setStatus('加载持仓...');
  try {
    const d = await api('/api/portfolio');
    if (!d.ok) throw new Error(d.error || 'portfolio error');
    const data = d.data || d;
    const positions = Array.isArray(data.positions) ? data.positions : [];
    const totalValue = data.total_value ?? (positions.length ? positions.reduce((s,p)=>s + (Number(p.total_value ?? p.value) || 0), 0) : null);
    const pnl = data.total_pnl ?? data.pnl ?? (positions.length ? positions.reduce((s,p)=>s + (Number(p.pnl) || 0), 0) : null);
    $('pfMetrics').innerHTML = [
      ['纸面可用资金', fmt(data.available_cash)], ['总市值', fmt(totalValue)], ['总盈亏', fmt(pnl)], ['收益率', pct(data.total_pnl_pct ?? data.pnl_pct ?? data.return_pct)], ['持仓数', positions.length],
    ].map(([l,v])=>`<div class="metric"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join('');
    $('pfResult').innerHTML = `<div style="color:#657282;">账户：纸面交易账户 · ${esc(data.cash_basis || '以已记录订单为准')} · 持仓修改必须经订单预检。</div>`;
    const table = $('pfTable');
    table.innerHTML = `<thead><tr><th>代码</th><th>名称</th><th>股数</th><th>成本</th><th>现价</th><th>市值</th><th>盈亏</th><th>盈亏%</th><th>动作</th></tr></thead><tbody>${positions.length ? positions.map(p=>`<tr><td>${esc(p.symbol)}</td><td>${esc(p.name || p.symbol || '-')}</td><td>${esc(p.shares)}</td><td>${esc(fmt(p.cost_price))}</td><td>${esc(fmt(p.current_price))}</td><td>${esc(fmt(p.total_value ?? p.value))}</td><td>${esc(fmt(p.pnl))}</td><td class="${p.pnl_pct == null ? '' : Number(p.pnl_pct)>=0?'up':'down'}">${esc(pct(p.pnl_pct))}</td><td><button class="btn-sm" onclick="draftSell('${esc(p.symbol)}',${Number(p.shares)||0},${Number(p.current_price)||Number(p.cost_price)||0})">卖出草稿</button></td></tr>`).join('') : '<tr><td colspan="9" class="empty">暂无纸面持仓</td></tr>'}</tbody>`;
    setStatus(`持仓已更新 (${positions.length}只)`);
  } catch(e) {
    console.error(e);
    $('pfResult').innerHTML = `<div class="empty">${esc(e.message)}</div>`;
    setStatus('持仓加载失败: '+e.message);
  } finally { setBusy('pfRefreshBtn', false); }
}

function draftSell(symbol, shares, price) {
  const params = new URLSearchParams({ sell_symbol: symbol, sell_shares: String(shares), sell_price: String(price) });
  location.href = '/paper_trading.html?' + params.toString();
}
window.draftSell = draftSell;

async function addPosition() {
  const symbol = ($('pfSymbol')?.value || '').trim();
  const shares = Number($('pfShares')?.value || 0);
  const cost = Number($('pfCost')?.value || 0);
  if (!symbol || shares <= 0 || cost <= 0) { setStatus('请填写代码、股数和价格后生成订单草稿'); return; }
  const params = new URLSearchParams({ symbol, shares: String(shares), price: String(cost), name: ($('pfName')?.value || '').trim() });
  setStatus('已生成纸面买入草稿，跳转订单预检');
  location.href = '/paper_trading.html?' + params.toString();
}

// ── 报告历史中心：按日期分组、分页加载、HTML/Markdown 同源打开 ──
let reportCatalog = { offset: 0, total: 0, rows: [] };
async function loadReports(reset = true) {
  if (reset) reportCatalog = { offset: 0, total: 0, rows: [] };
  setBusy('reportsRefreshBtn', true); setStatus('加载报告历史...');
  try {
    const d = await api(`/api/reports?limit=30&offset=${reportCatalog.offset}`, undefined, { cache_ttl: 0 });
    if (!d.ok) throw new Error(d.error || 'reports error');
    const reports = d.reports || d.data?.reports || d.data || [];
    reportCatalog.rows = reset ? reports : reportCatalog.rows.concat(reports);
    reportCatalog.offset = reportCatalog.rows.length;
    reportCatalog.total = Number(d.total ?? reportCatalog.rows.length);
    renderReportsList(reportCatalog.rows, reportCatalog.total);
    setStatus(`报告历史已更新 (${reportCatalog.rows.length}/${reportCatalog.total}份)`);
  } catch(e) {
    console.error(e);
    $('reportsList').innerHTML = `<div class="empty">${esc(e.message)}</div>`;
    if ($('reportContent')) { $('reportContent').style.display = 'none'; $('reportContent').innerHTML = ''; }
    setStatus('报告加载失败: '+e.message);
  } finally { setBusy('reportsRefreshBtn', false); }
}

function renderReportsList(reports, total = reports.length) {
  const el = $('reportsList');
  const content = $('reportContent');
  if (content) content.style.display = 'none';
  if (!el) return;
  if (!reports.length) { el.innerHTML = '<div class="empty">暂无报告</div>'; return; }
  const groups = {};
  for (const r of reports) {
    const date = String(r.date || r.report_date || r.created_at || r.mtime || r.name || r.path || '未分类').slice(0, 10);
    if (!groups[date]) groups[date] = [];
    groups[date].push(r);
  }
  const grouped = Object.entries(groups).sort((a,b)=>b[0].localeCompare(a[0])).map(([date, items])=>`
    <div style="font-weight:650;color:#c8cdd3;margin:12px 0 6px;">${esc(date)} <span style="color:#8b949e;font-size:12px;">${items.length} 份</span></div>
    ${items.map(r=>{
      const path = r.path || r.file || r.name || '';
      const title = r.title || r.name || path.split('/').pop() || '未命名报告';
      const isHtml = /\.html?$/i.test(path) || /html/i.test(String(r.type || ''));
      const openAction = isHtml ? `openHtmlReport('${encodeURIComponent(path)}')` : `readReport(decodeURIComponent('${encodeURIComponent(path)}'))`;
      const delivery = r.delivery || {};
      const deliveryText = delivery.status === 'acknowledged' ? '飞书已确认' : delivery.status === 'failed' ? '飞书失败' : delivery.status === 'blocked' ? '交付阻断' : '未记录投递';
      const deliveryColor = delivery.status === 'acknowledged' ? '#3fb950' : delivery.status === 'failed' || delivery.status === 'blocked' ? '#f85149' : '#8b949e';
      return `<button class="btn-sm" style="display:flex;width:100%;justify-content:space-between;gap:10px;text-align:left;margin:4px 0;background:#24272d;" onclick="${openAction}"><span>${isHtml ? '📄 ' : '📝 '}${esc(title)}</span><span style="color:${deliveryColor};font-size:11px;">${esc(r.type || '-')} · ${esc(r.generated_at || '-')} · ${deliveryText}</span></button>`;
    }).join('')}`).join('');
  const more = reports.length < total ? `<button class="btn-sm" style="margin-top:12px;" onclick="loadReports(false)">加载更多（已显示 ${reports.length}/${total}）</button>` : `<div class="empty" style="padding-top:12px;">已显示全部 ${total} 份报告</div>`;
  el.innerHTML = `<div style="color:#8b949e;font-size:12px;margin-bottom:8px;">报告按实际报告日归档；HTML 打开同日期研报，Markdown 在此预览。</div>${grouped}${more}`;
}

async function readReport(path) {
  const el = $('reportContent');
  if (!path || !el) return;
  setStatus('读取报告...');
  try {
    const d = await api('/api/reports?mode=read&path=' + encodeURIComponent(path));
    if (!d.ok) throw new Error(d.error || 'read report error');
    const content = d.content || d.data?.content || d.data || '';
    el.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center;font-weight:600;margin-bottom:8px;"><span>${esc(path.split('/').pop())}</span><button class="btn-sm" type="button" onclick="closeReport()" title="退出报告阅读">✕ 关闭</button></div><pre style="white-space:pre-wrap;max-height:640px;overflow:auto;color:#c8cdd3;font-size:13px;">${esc(typeof content === 'string' ? content : JSON.stringify(content, null, 2))}</pre>`;
    el.style.display = 'block';
    setStatus('报告已打开');
  } catch(e) {
    console.error(e);
    el.innerHTML = `<div class="empty">${esc(e.message)}</div>`;
    el.style.display = 'block';
    setStatus('读取报告失败: '+e.message);
  }
}
window.readReport = readReport;
window.openHtmlReport = function openHtmlReport(date) {
  const raw = decodeURIComponent(String(date || ''));
  const isPath = raw.includes('/') || raw.includes('\\');
  const value = raw.match(/^20\d{2}-\d{2}-\d{2}$/) ? raw : '';
  window.open('/report.html?' + (isPath ? 'path=' + encodeURIComponent(raw) : 'date=' + encodeURIComponent(value)), '_blank', 'noopener');
};
window.closeReport = function closeReport() {
  const el = $('reportContent');
  if (el) { el.style.display = 'none'; el.innerHTML = ''; }
  setStatus('已退出报告阅读');
};

// ── : 市场状态 ──
async function loadMarketRegime() {
  setBusy('regimeRefreshBtn', true); setStatus('加载市场状态...');
  try {
    const d = await api('/api/market/regime?refresh=true', undefined, { cache_ttl: 0, refresh: true });
    if (!d.ok) throw new Error(d.error || 'market regime error');
    const data = d.data || d;
    const signals = data.signals || data.factors || data.items || [];
    const marketData = data.market_data || {};
    const latestMarketDate = marketData.trade_date || marketData.date || data.date || data.updated_at || '-';
    const riskLevel = data.risk_level || data.risk || marketData.risk_level || marketData.risk || '-';
    const updateTime = latestMarketDate;
    const stateDateRaw = String(latestMarketDate || '').replace(/-/g, '').slice(0, 8);
    const currentDataDate = new Date().toISOString().slice(0, 10).replace(/-/g, '');
    const dateNote = stateDateRaw && stateDateRaw < currentDataDate ? '（数据源滞后）' : '';
    $('regimeMetrics').innerHTML = [
      ['状态', data.regime || data.stage || data.status || '-'], ['置信度', fmt(data.confidence ?? data.score)],
      ['风险级别', riskLevel], ['更新时间', `${latestMarketDate}${dateNote}`],
    ].map(([l,v])=>`<div class="metric"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join('');
    $('regimeResult').innerHTML = `<h2 style="margin-top:0;">${esc(data.regime || data.stage || data.status || '市场状态')}</h2><div style="color:#c8cdd3;line-height:1.7;">${esc(data.summary || data.description || data.advice || '暂无文字说明')}</div>`;
    const rows = Array.isArray(signals) ? signals.map(s => typeof s === 'string' ? {signal:s} : s) : Object.entries(signals).map(([signal,value])=>({signal,value}));
    renderTable('regimeTable', rows, rows[0] ? Object.keys(rows[0]) : ['signal','value']);
    setStatus('市场状态已更新');
  } catch(e) {
    console.error(e);
    $('regimeResult').innerHTML = `<div class="empty">${esc(e.message)}</div>`;
    setStatus('市场状态失败: '+e.message);
  } finally { setBusy('regimeRefreshBtn', false); }
}

function escAttr(v) {
  return String(v).replace(/[&<>'"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
}

$('pfRefreshBtn')?.addEventListener('click', loadPortfolio);
$('pfAddBtn')?.addEventListener('click', addPosition);
$('reportsRefreshBtn')?.addEventListener('click', () => loadReports(true));
$('regimeRefreshBtn')?.addEventListener('click', loadMarketRegime);

// ── : 回测看板 ──
// 演示端点角标（/api/backtest_dashboard、/api/strategy_compare、/api/factor_analysis 返回占位数据）
function demoBadgeHtml(note) {
  const tip = note || '该端点当前返回占位/演示数据，未接入真实计算';
  return '<span class="badge badge-warn" style="margin-right:10px;cursor:help;" title="' + tip + '">🧪 演示数据</span>';
}
async function loadBacktestDashboard() {
  setStatus('加载回测看板...');
  try {
    const d = await api('/api/backtest_dashboard');
    if (!d.ok) throw new Error(d.error);
    const data = d.data || {};
    const strategies = data.strategies || [];
    const factorBt = data.factor_backtest || {};
    const multi = data.multi_asset || {};
    const iteration = factorBt.iteration || {};
    const summary = factorBt.summary || {};
    const el = $('btDashboardMetrics');
    if (el) {
      const avg = strategies.reduce((a,s)=>a+(s.total_return||0),0)/(strategies.length||1);
      const status = factorBt.validation?.status || (data.demo ? 'demo' : 'unavailable');
      const version = factorBt.strategy_version || '—';
      const maStatus = multi.ok ? '已接入' : '缺失';
      el.innerHTML = (data.demo ? demoBadgeHtml(data.demo_note) : '') +
        `<div class="metric"><div class="label">策略候选</div><div class="value">${Object.keys(iteration.candidate_strategies||{}).length || (iteration.candidate_strategies||[]).length || strategies.length}</div></div>` +
        `<div class="metric"><div class="label">滚动窗口</div><div class="value">${summary.window_count ?? '—'}</div></div>` +
        `<div class="metric"><div class="label">冠军版本</div><div class="value" title="${esc(version)}">${esc(String(version).slice(0,12))}</div></div>` +
        `<div class="metric"><div class="label">多市场组合</div><div class="value">${esc(maStatus)}</div></div>` +
        `<div class="metric"><div class="label">状态</div><div class="value">${esc(status)}</div></div>`;
    }
    const candidateRows = (iteration.candidate_strategies || []).map(s => ({
      name: s.name, status: s.status, total_return: s.metrics?.annual_return_pct,
      max_dd: s.metrics?.max_drawdown_pct, sharpe: s.metrics?.sharpe,
      win_rate: s.metrics?.factor_count_used, trades: s.metrics?.window_count,
    }));
    if (multi.ok && multi.metrics) {
      const blend = multi.metrics.blend || {};
      const bench = multi.metrics.benchmarks || {};
      candidateRows.push({
        name: '多市场组合(商品趋势+现金)', status: multi.validation?.status || 'research_only',
        total_return: Number(blend.annual_return || 0) * 100, max_dd: Number(blend.max_drawdown || 0) * 100,
        sharpe: blend.sharpe, win_rate: `沪深300ETF ${fmtNum(Number(bench.csi300_etf?.annual_return || 0) * 100, 1)}%/yr`,
        trades: `标普500 ${fmtNum(Number(bench.sp500?.annual_return || 0) * 100, 1)}%/yr`,
      });
    }
    renderTable('btDashboardTable', candidateRows.length ? candidateRows : strategies, ['name','status','total_return','max_dd','sharpe','win_rate','trades']);
    setStatus(data.demo ? '回测看板已更新（演示数据）' : `回测看板已更新 · ${iteration.champion || '无冠军'} · ${summary.window_count || 0} 个滚动窗口`);
  } catch(e) { setStatus('回测看板失败: '+e.message); }
}

// ── : 因子分析 ──
async function loadFactorAnalysis() {
  const esc = v => String(v ?? '').replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c] || c);
  setBusy('factorModelRefreshBtn', true); setStatus('加载因子模型...');
  try {
    const res = await api('/api/factor_report?date=' + new Date().toISOString().slice(0,10), undefined, { timeout: 60000, silent: true });
    let data = res?.data || {};
    let isDemo = false;
    if (data.error || !res.ok) {
      // factor_report 不可用时回退到演示端点，并透明标注
      const d2 = await api('/api/factor_analysis', undefined, { silent: true, cache_ttl: 300 }).catch(() => ({ ok: false }));
      if (d2 && d2.ok && d2.data) {
        data = d2.data;
        data.demo = true;
        isDemo = true;
      } else {
        throw new Error(data.error || '因子模型不可用');
      }
    }

    const stats = data.factor_stats || {};
    const ic = data.factor_ic || {};
    const groups = data.group_stats || {};

    // 演示回退渲染（factor_analysis 结构：factors[ic/return/vol] + stocks[score]）
    if (isDemo) {
      const factors = data.factors || [];
      const stocks = data.stocks || [];
      $('factorMetrics').innerHTML = demoBadgeHtml('因子 IC 为占位值（factor_report 计算不可用，回退演示端点）') +
        (factors.length ? factors.map(x => `<div class="metric"><div class="label">${esc(x.name)}</div><div class="value" style="color:${(x.ic||0)>=0?'#b42318':'#087443'}">IC ${x.ic}</div></div>`).join('') : '');
      renderTable('factorTable', stocks.map(s => ({ symbol: s.symbol, name: s.name, score: s.score })), ['symbol','name','score']);
      setStatus('因子分析已更新（⚠️ 演示数据：factor_report 计算不可用）');
      return;
    }

    // 因子IC ECharts
    const dom = $('factorChart');
    if (dom && typeof echarts !== 'undefined') {
      dom.innerHTML = '';
      const names = Object.keys(ic);
      const values = names.map(n => ic[n] ?? null);
      const colors = values.map(v => v >= 0 ? '#1f7a5c' : '#b42318');
      const c = echarts.init(dom);
      c.setOption({
        title:{text:'因子IC (信息系数)',left:'center',textStyle:{color:'#e5e7eb',fontSize:13}},
        tooltip:{trigger:'axis',axisPointer:{type:'shadow'}},
        grid:{left:'3%',right:'4%',bottom:'20%',containLabel:true},
        xAxis:{type:'category',data:names,axisLabel:{color:'#9ca3af',rotate:45,fontSize:9}},
        yAxis:{type:'value',axisLabel:{color:'#9ca3af'}},
        series:[{type:'bar',data:values.map((v,i)=>({value:v,itemStyle:{color:colors[i]}}))}]
      });
      setTimeout(() => c.resize(), 100);
    }

    // 因子统计表
    const rows = Object.entries(stats).map(([name, s]) => ({
      name, ...s,
      ic: ic[name] == null ? null : Number(ic[name]).toFixed(4)
    }));
    renderTable('factorTable', rows, ['name','mean','std','skew','pct_pos','n_valid','ic']);

    // 分组IC显示在表头上方
    const groupsHtml = Object.entries(groups).filter(([_,v]) => v && v.n_factors).map(([g, s]) =>
      `<div class="metric"><div class="label">${esc(g)}</div><div class="value" style="color:${s.mean_ic >= 0 ? '#b42318' : '#087443'}">${(s.mean_ic * 100).toFixed(2)}%</div><div style="font-size:11px;color:#8b8fa3;">${s.n_factors} 个因子</div></div>`
    ).join('');
    $('factorMetrics').innerHTML = groupsHtml || '<div class="empty">暂无分组数据</div>';

    setStatus('因子模型已更新 ()');
  } catch(e) {
    console.error('factorAnalysis error:', e);
    setStatus('因子加载失败: '+e.message);
  } finally { setBusy('factorModelRefreshBtn', false); }
}

// ── 个股新闻情绪 ──
async function loadNewsSentiment() {
  const symbol = ($('newsSymbol')?.value || '').trim() || '002714';
  try {
    const d = await api(`/api/news_sentiment?symbol=${encodeURIComponent(symbol)}`);
    if (!d.ok) throw new Error(d.error);
    const data = d.data || d || {};
    const el = $('newsSentimentDisplay');
    const headlines = Array.isArray(data.headlines) ? data.headlines : (Array.isArray(data.items) ? data.items : []);
    const total = data.total ?? (headlines.length || null);
    if (el) {
      const status = data.status || (headlines.length ? 'available' : 'missing');
      const color = data.sentiment==='positive'?'#087443':data.sentiment==='negative'?'#b42318':'#d97706';
      const stance = data.sentiment==='positive'?'📈积极':data.sentiment==='negative'?'📉消极':data.sentiment == null ? '— 未知' : '⚖️中性';
      const hasEvidence = (status === 'available' || status === 'partial') && headlines.length > 0;
      el.innerHTML = hasEvidence ? `<div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap;"><div class="metric"><div class="label">情感倾向</div><div class="value" style="color:${color}">${stance}</div></div><div class="metric"><div class="label">情感评分</div><div class="value">${data.score ?? '-'}</div></div><div class="metric"><div class="label">新闻条数</div><div class="value">${total ?? '-'}</div></div></div>` : `<div class="data-state ${status === 'error' ? 'error' : 'empty'}">${esc(data.error || (status === 'partial' ? '新闻字段不完整，未形成可展示的代码级证据。' : '未取得该代码的新闻标题，不使用市场聚合新闻替代。'))}</div>`;
    }
    renderTable('newsTable', headlines.slice(0,10), ['title','date','source','sentiment']);
  } catch(e) {
    const message = `本次 ${symbol} 新闻请求失败：${e.message}`;
    if ($('newsSentimentDisplay')) $('newsSentimentDisplay').innerHTML = `<div class="data-state error">${esc(message)}</div>`;
    renderTable('newsTable', [], ['title','date','source','sentiment']);
    setStatus(message);
  }
}

// ── : 策略对比 ──
async function loadStrategyCompare() {
  try {
    const d = await api('/api/strategy_compare');
    if (!d.ok) throw new Error(d.error);
    const strategies = (d.data?.strategies || []);
    if (d.data && d.data.demo) {
      const radarEl = $('strategyRadar');
      if (radarEl) radarEl.insertAdjacentHTML('beforebegin', '<div style="margin-bottom:8px;">' + demoBadgeHtml('策略指标为占位值，未接真实回测对比') + '</div>');
      setStatus('策略对比已更新（⚠️ 演示数据）');
    }
    renderTable('strategyTable', strategies, ['name','metrics.收益','metrics.回撤','metrics.夏普','metrics.胜率','metrics.交易次数']);
    const dom = $('strategyRadar');
    if (dom && strategies.length) {
      const metrics = Object.keys(strategies[0].metrics||{});
      const c = echarts.init(dom);
      c.setOption({
        title:{text:'策略雷达对比',left:'center',textStyle:{color:'#e5e7eb',fontSize:14}},
        legend:{data:strategies.map(s=>s.name),textStyle:{color:'#9ca3af'},bottom:0},
        radar:{indicator:metrics.map(m=>({name:m,max:100})),radius:'55%'},
        series:[{type:'radar',data:strategies.map(s=>({value:Object.values(s.metrics||{}),name:s.name}))}]
      });
    }
  } catch(e) { console.error(e); }
}

// ════════════════════════════════════════════════
// 组合优化面板
// ════════════════════════════════════════════════
async function loadPortfolioOptimizer() {
  setBusy('optRefreshBtn', true); setStatus('运行组合优化...');
  try {
    const d = await api('/api/portfolio_optimizer?mode=mv');
    if (!d.ok) throw new Error(d.error);
    const data = d.data || d;
    renderOptimizerResult(data, 'optMVResult', '均值-方差');
  } catch(e) { setStatus('组合优化失败: '+e.message); }
  finally { setBusy('optRefreshBtn', false); }
}

async function loadEfficientFrontier() {
  setBusy('optEfficientFrontierBtn', true);
  try {
    const d = await api('/api/portfolio_optimizer?mode=frontier');
    if (!d.ok) throw new Error(d.error);
    const data = d.data || d;
    renderEfficientFrontier(data);
    setStatus('有效前沿已更新');
  } catch(e) { setStatus('前沿计算失败: '+e.message); }
  finally { setBusy('optEfficientFrontierBtn', false); }
}

async function loadERC() {
  try {
    const d = await api('/api/portfolio_optimizer?mode=erc');
    if (!d.ok) throw new Error(d.error);
    const data = d.data || d;
    renderOptimizerResult(data, 'optERCResult', 'ERC');
  } catch(e) { /* silent */ }
}

function renderOptimizerResult(data, elId, title) {
  const el = $(elId);
  if (!el) return;
  if (data.note) { el.innerHTML = `<div class="empty">${esc(data.note)}</div>`; return; }
  const weights = data.weights || [];
  const metrics = data.metrics || {};
  const items = [
    ['预期年化收益', fmtPct(metrics.expected_return)],
    ['年化波动率', fmtPct(metrics.volatility)],
    ['夏普比率', fmt(metrics.sharpe_ratio)],
    ['分散度', fmt(metrics.diversification_ratio ?? '-')],
  ];
  let html = `<div class="metrics">${items.map(([l,v])=>`<div class="metric"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join('')}</div>`;
  // Pie chart for weights (top 10)
  if (weights.length) {
    const domId = elId + 'Pie';
    html += `<div id="${domId}" style="height:200px;"></div>`;
    el.innerHTML = html;
    const chart = echarts.init($(domId));
    const topW = weights.slice(0, 10);
    chart.setOption({
      tooltip: {trigger:'item', formatter:'{b}: {c}%'},
      series: [{
        type: 'pie', radius: ['30%','60%'],
        data: topW.map(w => ({name: w.symbol || w.name || w[0], value: parseFloat((w.weight || w[1] || 0) * 100).toFixed(1)})),
        label: {fontSize:10},
      }]
    });
    window.addEventListener('resize', () => chart.resize());
  } else {
    el.innerHTML = html;
  }
}

function renderEfficientFrontier(data) {
  const dom = $('optFrontierChart');
  if (!dom) return;
  const points = data.points || [];
  if (!points.length) { dom.innerHTML = '<div class="empty">无有效前沿数据</div>'; return; }
  const chart = echarts.init(dom);
  chart.setOption({
    tooltip: {trigger:'axis', formatter: function(p) {
      return `风险: ${p[0].value[0].toFixed(2)}%<br/>收益: ${p[0].value[1].toFixed(2)}%`;
    }},
    xAxis: {type:'value', name:'风险(波动率%)', axisLabel:{color:'#9ca3af'}},
    yAxis: {type:'value', name:'预期收益%', axisLabel:{color:'#9ca3af'}},
    series: [{
      type:'line', data: points.map(p => [p.volatility, p.return]),
      smooth: true, lineStyle:{color:'#2563eb', width:2},
      areaStyle:{color:'rgba(37,99,235,0.1)'},
      symbol: 'none',
    }]
  });
  window.addEventListener('resize', () => chart.resize());
}

// ════════════════════════════════════════════════
// 驱动因子归因
// ════════════════════════════════════════════════
window.LOAD_MARKET_attribution = async function() {
  setBusy('attrRefreshBtn', true); setStatus('加载驱动因子归因...');
  try {
    const d = await api('/api/market_attribution');
    if (!d.ok) throw new Error(d.error);
    const data = d.data || d;
    // Metrics
    const factors = data.factors || [];
    const topDrivers = data.top_drivers || [];
    const summary = data.summary || '';
    $('attrMetrics').innerHTML = factors.map(f =>
      `<div class="metric"><div class="label">${esc(f.name)}</div><div class="value" style="color:${(f.contribution||0)>=0?'#b42318':'#087443'}">${f.contribution >= 0 ? '+' : ''}${(f.contribution||0).toFixed(1)}%</div><div style="font-size:11px;color:#8b8fa3;">${esc(f.description||'')}</div></div>`
    ).join('') || '<div class="empty">暂无因子数据</div>';
    $('attrSummary').innerHTML = `<div style="color:#c8cdd3;line-height:1.7;">${esc(summary || '暂无文字说明')}</div>` +
      (topDrivers.length ? `<h3 style="margin-top:12px;">主要驱动因子</h3><ul>${topDrivers.map(d=>`<li>${esc(d)}</li>`).join('')}</ul>` : '');
    renderTable('attrTable', factors, ['name','contribution','direction','description']);
    setStatus('驱动归因已更新');
  } catch(e) {
    $('attrSummary').innerHTML = `<div class="empty">${esc(e.message)}</div>`;
    setStatus('驱动归因失败: '+e.message);
  } finally { setBusy('attrRefreshBtn', false); }
}

// ════════════════════════════════════════════════
// ML 监控面板
// ════════════════════════════════════════════════

async function loadMLMonitor() {
  const esc = v => String(v ?? '').replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c] || c);
  setBusy('mlmRefreshBtn', true);
  setStatus('加载ML监控数据...');
  try {
    const res = await api('/api/ml_model_info', undefined, { timeout: 60000, silent: true });
    const data = res?.data || {};

    // ── 模型文件状态（macro-panel） ──
    const models = data.models || [];
    $('mlmModelStatus').innerHTML = models.length
      ? '<table class="risk-table"><tr><th>模型</th><th>大小</th><th>最后修改</th><th>已过(h)</th></tr>' +
        models.map(m => `<tr><td>${esc(m.name)}</td><td>${m.size_kb}KB</td><td>${esc(m.mtime)}</td><td>${m.age_h}</td></tr>`).join('') + '</table>'
      : '<div class="empty">暂无已训练的模型</div>';

    // ── 训练记录 ──
    const logs = Array.isArray(data.training_log) ? data.training_log : data.training_log ? [data.training_log] : [];
    $('mlmTrainingLog').innerHTML = logs.length
      ? '<table class="risk-table"><tr><th>时间</th><th>模型</th><th>样本</th><th>得分</th></tr>' +
        logs.slice(0,10).map(r => `<tr><td>${esc(r.time || r.timestamp || '-')}</td><td>${esc(r.model || r.type || '-')}</td><td>${r.samples ?? '-'}</td><td>${r.score ?? '-'}</td></tr>`).join('') + '</table>'
      : '<div class="empty">暂无训练记录</div>';

    // ── 特征重要性 — ECharts 条形图 ──
    const featImp = data.feature_importance || [];
    const featEl = $('mlmFeatChart');
    featEl.innerHTML = '';
    if (featImp.length) {
      if (typeof echarts !== 'undefined') {
        const names = featImp.map(f => f.feature).reverse();
        const vals = featImp.map(f => f.importance).reverse();
        const colors = vals.map(v => v >= 0 ? '#1f7a5c' : '#b42318');
        const chart = echarts.init(featEl);
        chart.setOption({
          tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
          grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
          xAxis: { type: 'value' },
          yAxis: { type: 'category', data: names, axisLabel: { fontSize: 10 } },
          series: [{ type: 'bar', data: vals.map((v,i) => ({ value: v, itemStyle: { color: colors[i] } })) }]
        });
        setTimeout(() => chart.resize(), 100);
      } else {
        featEl.innerHTML = '<ul>' + featImp.map(f => `<li>${esc(f.feature)}: ${f.importance}</li>`).join('') + '</ul>';
      }
    } else {
      featEl.innerHTML = '<div class="empty">暂无特征重要性数据，请先训练模型</div>';
    }

    // ── Ensemble 配置 ──
    const ev = data.ensemble_version || {};
    const keys = Object.keys(ev);
    $('mlmEnsembleBody').innerHTML = keys.length
      ? '<table class="risk-table">' + keys.map(k =>
        `<tr><td style="font-weight:600;width:160px;">${esc(k)}</td><td>${esc(typeof ev[k] === 'object' ? JSON.stringify(ev[k]) : ev[k])}</td></tr>`
      ).join('') + '</table>'
      : '<div class="empty">暂无 Ensemble 配置信息</div>';

    setStatus('ML监控已更新');
  } catch(e) {
    $('mlmModelStatus').innerHTML = '<div class="empty">加载失败</div>';
    $('mlmTrainingLog').innerHTML = '<div class="empty">加载失败</div>';
    $('mlmFeatChart').innerHTML = '<div class="empty">加载失败: '+esc(e.message)+'</div>';
    setStatus('ML监控失败: '+e.message);
  } finally { setBusy('mlmRefreshBtn', false); }
}

async function mlRetrain() {
  if (!confirm('确定要重新训练所有ML模型？此操作可能耗时较久。')) return;
  const btn = $('mlmRetrainBtn');
  setBusy(btn, true);
  setStatus('重新训练中，请等待...');
  try {
    const res = await api('/api/ml_retrain', {}, { timeout: 300000, cache_ttl: 0, silent: true });
    setStatus('训练完成: ' + (res.data?.message || res.message || '成功'));
    loadMLMonitor(); // 刷新面板
  } catch(e) {
    setStatus('训练失败: '+e.message);
  } finally { setBusy(btn, false); }
}

// ════════════════════════════════════════════════
// : 事件驱动回测
// ════════════════════════════════════════════════

async function runEventBacktest() {
  const esc = v => String(v ?? '').replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c] || c);
  setBusy('eventBtRunBtn', true); setStatus('运行事件回测...');
  try {
    const syms = ($('eventBtSymbols')?.value || '002714').trim();
    const start = ($('eventBtStart')?.value || '20260101').trim();
    const mode = ($('eventBtMode')?.value || 'ma').trim();
    const capital = ($('eventBtCapital')?.value || '1000000').trim();
    const slip = ($('eventBtSlippage')?.value || 'dynamic').trim();
    const url = `/api/v8_backtest?symbols=${encodeURIComponent(syms)}&start=${encodeURIComponent(start)}&mode=${encodeURIComponent(mode)}&capital=${encodeURIComponent(capital)}&slippage=${encodeURIComponent(slip)}`;

    const res = await api(url, undefined, { cache_ttl: 0, refresh: true, timeout: 120000 });
    const data = res.data || res;
    const perf = data.performance || {};
    const summary = data.summary || {};

    // Metrics
    $('eventBtMetrics').innerHTML = `
      <div class="metric"><div class="label">总收益</div><div class="value" style="color:${perf.total_return_pct >= 0 ? '#b42318' : '#087443'}">${perf.total_return_pct >= 0 ? '+' : ''}${perf.total_return_pct}%</div></div>
      <div class="metric"><div class="label">年化收益</div><div class="value">${perf.annualized_return_pct}%</div></div>
      <div class="metric"><div class="label">年化波动</div><div class="value">${perf.annualized_volatility_pct}%</div></div>
      <div class="metric"><div class="label">夏普比</div><div class="value" style="color:${perf.sharpe_ratio >= 1 ? '#1f7a5c' : perf.sharpe_ratio >= 0 ? '#b45309' : '#b42318'}">${perf.sharpe_ratio}</div></div>
      <div class="metric"><div class="label">最大回撤</div><div class="value" style="color:${perf.max_drawdown_pct < -10 ? '#b42318' : '#087443'}">${perf.max_drawdown_pct}%</div></div>
      <div class="metric"><div class="label">胜率</div><div class="value">${perf.win_rate_pct}%</div></div>
      <div class="metric"><div class="label">盈亏比</div><div class="value">${perf.profit_factor}</div></div>
      <div class="metric"><div class="label">交易次数</div><div class="value">${summary.total_trades ?? 0}</div></div>
    `;

    // Equity curve chart
    const ec = data.equity_curve || [];
    const chartEl = $('eventBtChart');
    chartEl.innerHTML = '';
    if (ec.length && typeof echarts !== 'undefined') {
      const chart = echarts.init(chartEl);
      chart.setOption({
        tooltip: { trigger: 'axis' },
        grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
        xAxis: { type: 'category', data: ec.map(p => p.date.slice(5)), axisLabel: { fontSize: 10, rotate: 45 } },
        yAxis: { type: 'value', axisLabel: { formatter: v => v.toLocaleString() } },
        series: [{
          name: '净值', type: 'line', data: ec.map(p => p.equity),
          smooth: true, showSymbol: false,
          lineStyle: { color: '#1f7a5c', width: 2 },
          areaStyle: { color: 'rgba(31,122,92,0.1)' }
        }]
      });
      setTimeout(() => chart.resize(), 100);
    } else {
      chartEl.innerHTML = '<div class="empty">无曲线数据</div>';
    }

    // Trades
    const trades = data.trades || [];
    $('eventBtTrades').innerHTML = trades.length
      ? '<table class="risk-table" style="min-width:auto;"><tr><th>日期</th><th>标的</th><th>方向</th><th>价格</th><th>数量</th><th>盈亏</th></tr>' +
        trades.slice(0,50).map(t => `<tr>
          <td>${esc(t.date)}</td>
          <td>${esc(t.symbol)}</td>
          <td style="color:${t.side === 'buy' ? '#b42318' : '#087443'}">${t.side === 'buy' ? '买入' : '卖出'}</td>
          <td>${t.price}</td>
          <td>${t.qty}</td>
          <td style="color:${(t.pnl||0) >= 0 ? '#b42318' : '#087443'}">${t.pnl >= 0 ? '+' : ''}${t.pnl}</td>
        </tr>`).join('') + '</table>'
      : '<div class="empty">暂无交易记录</div>';

    setStatus('事件回测完成');
  } catch(e) {
    $('eventBtMetrics').innerHTML = `<div class="empty">${esc(e.message)}</div>`;
    setStatus('事件回测失败: '+e.message);
  } finally { setBusy('eventBtRunBtn', false); }
}

// ════════════════════════════════════════════════
// : 数据缓存面板
// ════════════════════════════════════════════════

async function loadCacheStatus() {
  const esc = v => String(v ?? '').replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c] || c);
  setBusy('cacheRefreshBtn', true); setStatus('加载缓存状态...');
  try {
    const res = await api('/api/cache_status', undefined, { cache_ttl: 15, silent: true });
    const data = res?.data || {};
    if (data.error) { throw new Error(data.error); }
    const store = data.data_store || data;
    const generated = data.generated_a_share_data || {};

    // Metrics: accept the canonical nested DataStore response and legacy flat shape.
    $('cacheMetrics').innerHTML = `
      <div class="metric"><div class="label">缓存标的</div><div class="value">${store.total_symbols ?? 0}</div></div>
      <div class="metric"><div class="label">新鲜</div><div class="value" style="color:#1f7a5c;">${store.fresh_count ?? 0}</div></div>
      <div class="metric"><div class="label">陈旧</div><div class="value" style="color:#b42318;">${store.stale_count ?? 0}</div></div>
      <div class="metric"><div class="label">平均质量</div><div class="value">${store.avg_quality == null ? '-' : (store.avg_quality * 100).toFixed(0) + '%'}</div></div>
      <div class="metric"><div class="label">DB大小</div><div class="value">${store.db_size_kb == null ? '-' : store.db_size_kb + 'KB'}</div></div>
      <div class="metric"><div class="label">行情缓存</div><div class="value">${generated.total_size_mb == null ? '-' : generated.total_size_mb + 'MB'}</div></div>
    `;

    // Entry table
    const entries = store.entries || data.entries || [];
    $('cacheEntries').innerHTML = entries.length
      ? '<table class="risk-table" style="min-width:auto;"><tr><th>标的</th><th>行数</th><th>质量</th><th>已过(h)</th><th>状态</th><th>最新日期</th></tr>' +
        entries.map(e => `<tr>
          <td>${esc(e.symbol)}</td>
          <td>${e.row_count}</td>
          <td>${(e.quality_score * 100).toFixed(0)}%</td>
          <td>${e.age_h}</td>
          <td>${e.stale ? '🟡 过期' : '🟢 新鲜'}</td>
          <td>${esc(e.last_date || '-')}</td>
        </tr>`).join('') + '</table>'
      : '<div class="empty">暂无缓存数据，请先查询股票</div>';

    setStatus('缓存状态已更新');
  } catch(e) {
    $('cacheEntries').innerHTML = '<div class="empty">加载失败: '+esc(e.message)+'</div>';
    setStatus('缓存状态失败: '+e.message);
  } finally { setBusy('cacheRefreshBtn', false); }
}

async function refreshAllCache() {
  if (!confirm('确定要刷新所有缓存数据？此操作可能需要较长时间。')) return;
  setBusy('cacheRefreshAllBtn', true); setStatus('正在刷新全部缓存...');
  try {
    const res = await api('/api/cache_refresh', undefined, { cache_ttl: 0, timeout: 180000, silent: true });
    const data = res?.data || {};
    setStatus('刷新完成: '+data.refreshed+' 成功, '+data.failed+' 失败');
    loadCacheStatus();
  } catch(e) {
    setStatus('刷新失败: '+e.message);
  } finally { setBusy('cacheRefreshAllBtn', false); }
}

// ──  财务因子 ──

async function loadFinancialData() {
  const esc = v => String(v ?? '').replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c] || c);
  const sym = $('finDataSymbol')?.value || '002714';
  setBusy('finDataRefreshBtn', true); setStatus('查询财务因子...');
  try {
    const res = await api('/api/financial_data?symbol='+encodeURIComponent(sym), undefined, { timeout: 60000, silent: true });
    if (!res?.ok) throw new Error(res?.error || '财务数据不可用');
    const data = res.data || {};
    const ind = data.indicators || {};
    const entries = Object.entries(ind).filter(([k, v]) =>
      !k.startsWith('_') && v !== null && v !== '' &&
      (typeof v !== 'number' || Number.isFinite(v))
    );
    // 负增长、亏损和负现金流也是有效财务指标，不能按 v > 0 丢弃。
    const filled = entries.length;
    $('finDataMetrics').innerHTML = `
      <div class="metric"><div class="label">代码</div><div class="value">${esc(sym)}</div></div>
      <div class="metric"><div class="label">有效指标</div><div class="value">${filled}/${entries.length}</div></div>
    `;
    // Table
    $('finDataPanel').innerHTML = entries.length
      ? '<table class="risk-table" style="min-width:auto;"><tr><th>财务因子</th><th>数值</th><th>分组</th></tr>' +
        entries.map(([k, v]) => {
          let grp = '价值';
          if (['roe','roa','gross_margin','debt_ratio','ocf_to_sales'].includes(k)) grp = '质量';
          if (['sales_growth_yoy','profit_growth_yoy','surprise'].includes(k)) grp = '成长';
          return `<tr><td>${esc(k)}</td><td>${typeof v === 'number' ? v.toFixed(4) : esc(v)}</td><td>${grp}</td></tr>`;
        }).join('') + '</table>'
      : '<div class="empty">当前股票没有可用财务指标，系统会在下一次季度数据更新后自动补齐。</div>';
    setStatus(`财务因子已加载: ${filled}项`);
  } catch(e) {
    $('finDataMetrics').innerHTML = '';
    $('finDataPanel').innerHTML = '<div class="empty">财务数据不可用: '+esc(e.message)+'</div>';
    setStatus('财务数据读取失败');
  } finally { setBusy('finDataRefreshBtn', false); }
}

// ──  特征存储 ──

// ── V2 (2026-08-07): 数据基座视图（前端与数据基座融合）──
async function loadDataBase() {
  const esc = v => String(v ?? '').replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c] || c);
  setBusy('dbRefreshBtn', true);
  try {
    const res = await api('/api/v1/health', undefined, { timeout: 60000, silent: true });
    // 体检是异步深度计算：computing 态自动重试拿到真实数据，而不是显示"无数据"
    if (res && res.ok && res.status === 'computing') {
      $('dbTable').innerHTML = '<div class="empty skeleton">🩺 数据基座体检计算中（数据集/K线覆盖/新鲜度），2.5s 后自动获取结果…</div>';
      setStatus('数据基座体检计算中…');
      if (!(loadDataBase._retries > 0)) loadDataBase._retries = 0;
      if (loadDataBase._retries < 6) {
        loadDataBase._retries++;
        setTimeout(() => loadDataBase(), 2500);
        return;
      }
    }
    loadDataBase._retries = 0;
    const fresh = (res && res.freshness) || [];
    const okN = fresh.filter(r => r.status === 'ok').length;
    const staleN = fresh.filter(r => r.status !== 'ok').length;
    $('dbMetrics').innerHTML = `
      <div class="metric"><div class="label">数据集</div><div class="value">${res.datasets ? res.datasets.length : 0}</div></div>
      <div class="metric"><div class="label">✅ 正常</div><div class="value" style="color:#1f7a5c;">${okN}</div></div>
      <div class="metric"><div class="label">⚠️ 待关注</div><div class="value" style="color:${staleN ? '#b42318' : 'inherit'};">${staleN}</div></div>
      <div class="metric"><div class="label">K线覆盖</div><div class="value">${res.kline_files || 0} 只</div></div>
      <div class="metric"><div class="label">估值覆盖</div><div class="value">${res.valuation_files || 0} 只</div></div>
      <div class="metric"><div class="label">任务</div><div class="value">${res.tasks ? (res.tasks.tasks || []).filter(t => t.status === 'running').length : 0} 运行中</div></div>
    `;
    if (!fresh.length) {
      $('dbTable').innerHTML = '<div class="empty">体检无返回行：DataStore().freshness() 异常或无数据集注册，请检查服务端日志后重试</div>';
    } else {
      const stCls = { 'ok': '#1f7a5c', 'stale': '#b42318', 'partial': '#b45309', 'missing': '#b42318', 'n/a': '#657282', 'error': '#b42318' };
      $('dbTable').innerHTML = `<table class="risk-table" style="min-width:auto;">
        <tr><th>数据集</th><th>状态</th><th>文件数</th><th>最新日期</th><th>预期</th><th>频率</th><th>说明</th></tr>` +
        fresh.map(r => `<tr>
          <td><b>${esc(r.dataset)}</b></td>
          <td style="color:${stCls[r.status] || 'inherit'};">${r.status === 'ok' ? '✅' : '⚠️'} ${esc(r.status)}</td>
          <td>${r.files}</td>
          <td>${esc(r.latest || '-')}</td>
          <td>${esc(r.expected || '-')}</td>
          <td>${esc(r.freq || '-')}</td>
          <td style="color:#657282;">${esc(r.note || '')}</td>
        </tr>`).join('') + '</table>';
    }
    setStatus('数据基座体检已更新');
    loadSegments();  // : 同步加载市场口径分层
  } catch(e) {
    $('dbTable').innerHTML = '<div class="empty">加载失败: '+esc(e.message)+'</div>';
  } finally { setBusy('dbRefreshBtn', false); }
}

// ──  (2026-08-07): 市场口径分层（主板/双创/北交所）──

async function loadSegments() {
  const esc = v => String(v ?? '').replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c] || c);
  setBusy('segRefreshBtn', true);
  try {
    const res = await api('/api/v1/segments', undefined, { timeout: 60000, silent: true });
    if (!res.ok || !res.data) throw new Error(res.error || '无数据');
    const d = res.data;
    const seg = d.segments || {};
    const style = d.style || {};
    let html = `<div style="color:#657282;font-size:12px;margin-bottom:8px;">快照: ${esc(d.snapshot || '')} · ${esc(d.time || '')}</div>`;
    // 口径分层表
    const segKeys = ['全A','主板','创业板','科创板','双创','北交所','剔除ST北交'];
    html += '<b>市场口径分层</b><table class="risk-table" style="min-width:auto;margin-top:6px;">' +
      '<tr><th>口径</th><th>股票数</th><th>等权涨跌幅%</th><th>上涨占比%</th><th>涨超5%</th><th>跌超5%</th><th>涨停</th><th>跌停</th></tr>' +
      segKeys.filter(k => seg[k]).map(k => {
        const s = seg[k];
        const ret = s['等权涨跌幅%'];
        const color = ret === null || ret === undefined ? '#657282' : (ret >= 0 ? '#1f7a5c' : '#b42318');
        return `<tr><td><b>${k}</b></td><td>${s['股票数'] ?? 0}</td>` +
          `<td style="color:${color};">${ret === null || ret === undefined ? '缺失' : ret}</td>` +
          `<td>${s['上涨占比%'] ?? '-'}</td><td>${s['涨超5%'] ?? '-'}</td>` +
          `<td>${s['跌超5%'] ?? '-'}</td><td>${s['近似涨停'] ?? '-'}</td><td>${s['近似跌停'] ?? '-'}</td></tr>`;
      }).join('') + '</table>';
    // 风格分层表
    const styleKeys = Object.keys(style).filter(k => k !== 'meta');
    if (styleKeys.length) {
      html += '<b style="display:block;margin-top:12px;">风格分层（成交额分位）</b><table class="risk-table" style="min-width:auto;margin-top:6px;">' +
        '<tr><th>风格</th><th>股票数</th><th>等权涨跌幅%</th><th>上涨占比%</th></tr>' +
        styleKeys.map(k => {
          const s = style[k];
          const ret = s['等权涨跌幅%'];
          const color = ret === null || ret === undefined ? '#657282' : (ret >= 0 ? '#1f7a5c' : '#b42318');
          return `<tr><td><b>${esc(k)}</b></td><td>${s['股票数'] ?? 0}</td>` +
            `<td style="color:${color};">${ret === null || ret === undefined ? '缺失' : ret}</td><td>${s['上涨占比%'] ?? '-'}</td></tr>`;
        }).join('') + '</table>';
    }
    // 短线热点题材（涨停股概念聚合）
    const sml = d.short_medium_long || {};
    const hot = sml['短线_热点题材(涨停概念)'];
    if (hot && hot.Top概念 && hot.Top概念.length) {
      html += '<b style="display:block;margin-top:12px;">短线热点题材（涨停股概念聚合）</b>' +
        `<div style="font-size:12px;color:#657282;margin:4px 0;">涨停股 ${esc(hot.涨停股数 ?? 0)} 只</div>` +
        '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:4px;">' +
        hot.Top概念.slice(0, 8).map(r =>
          `<span style="background:var(--bg2,#17202a);border:1px solid #2d3b4a;border-radius:10px;padding:2px 10px;font-size:12px;">${esc(r.concept)} <b style="color:#e8c15a;">${esc(r.涨停家数)}</b></span>`
        ).join('') + '</div>';
    }
    $('segTable').innerHTML = html;
    setStatus('市场分层已更新');
  } catch(e) {
    $('segTable').innerHTML = '<div class="empty">加载失败: '+esc(e.message)+'</div>';
  } finally { setBusy('segRefreshBtn', false); }
}

// ──  特征存储 ──

async function loadFeatureStore() {
  const esc = v => String(v ?? '').replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c] || c);
  setBusy('fsRefreshBtn', true);
  try {
    const res = await api('/api/feature_store', undefined, { timeout: 60000, silent: true });
    const d = res?.data || {};
    const st = d.stats || {};
    const dr = d.drift || {};
    $('fsMetrics').innerHTML = `
      <div class="metric"><div class="label">特征条目</div><div class="value">${st.n_features ?? 0}</div></div>
      <div class="metric"><div class="label">标的数</div><div class="value">${st.n_symbols ?? 0}</div></div>
      <div class="metric"><div class="label">日期数</div><div class="value">${st.n_dates ?? 0}</div></div>
      <div class="metric"><div class="label">预测记录</div><div class="value">${st.n_predictions ?? 0}</div></div>
      <div class="metric"><div class="label">已标注</div><div class="value">${st.n_labeled_predictions ?? 0}</div></div>
      <div class="metric"><div class="label">漂移告警</div><div class="value" style="color:${(st.drift_alerts || 0) > 0 ? '#b42318' : 'inherit'};">${st.drift_alerts ?? 0}</div></div>
    `;
    $('fsTable').innerHTML = `<table class="risk-table" style="min-width:auto;">
      <tr><th>维度</th><th>数值</th></tr>
      <tr><td>近期准确率</td><td>${((dr.recent_accuracy || 0)*100).toFixed(1)}%</td></tr>
      <tr><td>基线准确率</td><td>${((dr.baseline_accuracy || 0)*100).toFixed(1)}%</td></tr>
      <tr><td>漂移</td><td style="color:${dr.drift_detected ? '#b42318' : '#1f7a5c'};">${dr.drift_detected ? '⚠️ 是' : '✅ 否'}</td></tr>
      <tr><td>样本数</td><td>${dr.n_samples ?? 0}</td></tr>
    </table>`;
    setStatus('特征存储已更新');
  } catch(e) {
    $('fsTable').innerHTML = '<div class="empty">加载失败: '+esc(e.message)+'</div>';
  } finally { setBusy('fsRefreshBtn', false); }
}

// ──  漂移检测 ──

async function loadDriftCheck() {
  const esc = v => String(v ?? '').replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c] || c);
  setBusy('driftRefreshBtn', true); setStatus('检测模型漂移...');
  try {
    const res = await api('/api/drift_check', undefined, { timeout: 60000, silent: true });
    const d = res?.data || {};
    const drifted = d.drift_detected;
    $('driftMetrics').innerHTML = `
      <div class="metric"><div class="label">近期准确率</div><div class="value" style="color:${drifted ? '#b42318' : '#1f7a5c'};">${((d.recent_accuracy||0)*100).toFixed(1)}%</div></div>
      <div class="metric"><div class="label">基线准确率</div><div class="value">${((d.baseline_accuracy||0)*100).toFixed(1)}%</div></div>
      <div class="metric"><div class="label">偏移</div><div class="value" style="color:${drifted ? '#b42318' : '#657282'};">${(d.delta||0) > 0 ? '+' : ''}${((d.delta||0)*100).toFixed(1)}%</div></div>
      <div class="metric"><div class="label">样本数</div><div class="value">${d.n_samples ?? 0}</div></div>
    `;
    $('driftStatus').innerHTML = drifted
      ? `<div style="padding:20px;background:#2a1a1a;border-left:4px solid #b42318;border-radius:4px;">
           <span style="font-size:24px;">⚠️</span> <strong>模型漂移已检测到</strong><br>
           <span style="color:#b0b8c0;">近期准确率 ${((d.recent_accuracy||0)*100).toFixed(1)}% 低于基线 ${((d.baseline_accuracy||0)*100).toFixed(1)}%（偏移 ${((d.delta||0)*100).toFixed(1)}%）</span><br>
           <span style="color:#b0b8c0;">建议重新训练模型</span>
         </div>`
      : `<div style="padding:20px;background:#1a2a1a;border-left:4px solid #1f7a5c;border-radius:4px;">
           <span style="font-size:24px;">✅</span> <strong>模型正常</strong><br>
           <span style="color:#b0b8c0;">近期准确率 ${((d.recent_accuracy||0)*100).toFixed(1)}%（基线 ${((d.baseline_accuracy||0)*100).toFixed(1)}%，偏移 ${((d.delta||0) > 0 ? '+' : '')}${((d.delta||0)*100).toFixed(1)}%）</span>
         </div>`;
    setStatus('漂移检测完成');
  } catch(e) {
    $('driftStatus').innerHTML = '<div class="empty">检测失败: '+esc(e.message)+'</div>';
  } finally { setBusy('driftRefreshBtn', false); }
}

// ──  资产配置引擎 ──

async function loadAssetAllocation() {
  const esc = v => String(v ?? '').replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c] || c);
  const syms = $('allocSymbols')?.value || '600519,000858,002714,601899,002594,300750,600036';
  const obj = $('allocObjective')?.value || 'max_sharpe';
  setBusy('allocRunBtn', true); setStatus('运行资产配置...');
  try {
    const res = await api('/api/asset_allocation?symbols='+encodeURIComponent(syms)+'&objective='+obj, undefined, { timeout: 120000, silent: true });
    const data = res?.data || {};
    if (!data.weights || Object.keys(data.weights).length === 0) {
      $('allocMetrics').innerHTML = '<div class="empty">配置失败: '+(data.status||'未知')+'</div>';
      return;
    }
    const weights = data.weights;
    const metrics = data.metrics || {};
    const sectors = data.sector_exposure || {};
    const stress = data.stress_test || [];
    const risk_breakdown = data.risk_breakdown || {};

    // Metrics
    $('allocMetrics').innerHTML = `
      <div class="metric"><div class="label">预期年化</div><div class="value">${((metrics.expected_return||0)*242*100).toFixed(1)}%</div></div>
      <div class="metric"><div class="label">年化波动</div><div class="value">${((metrics.volatility||0)*Math.sqrt(242)*100).toFixed(1)}%</div></div>
      <div class="metric"><div class="label">夏普比</div><div class="value" style="color:${(metrics.sharpe_ratio||0) > 1 ? '#1f7a5c' : '#b42318'};">${(metrics.sharpe_ratio||0).toFixed(2)}</div></div>
      <div class="metric"><div class="label">有效个数</div><div class="value">${(data.concentration?.effective_n||1).toFixed(1)}</div></div>
      <div class="metric"><div class="label">标的数</div><div class="value">${data.n_symbols||0}</div></div>
      <div class="metric"><div class="label">耗时</div><div class="value">${data.elapsed||0}s</div></div>
    `;

    // Weight pie chart (ECharts)
    const wEntries = Object.entries(weights).sort((a,b) => b[1]-a[1]).slice(0, 10);
    if (typeof echarts !== 'undefined') {
      const wc = echarts.init($('allocWeights'));
      wc.setOption({
        tooltip: { trigger:'item', formatter: '{b}: {d}%' },
        series: [{
          type:'pie', radius:['30%','70%'],
          data: wEntries.map(([k,v]) => ({name: esc(k), value: (v*100).toFixed(2)})),
          label: { color:'#b0b8c0', formatter:'{b}\n{d}%' },
          itemStyle: { borderRadius: 4 },
        }]
      });

      // Sector pie
      const sEntries = Object.entries(sectors).sort((a,b) => b[1]-a[1]);
      const sc = echarts.init($('allocSectors'));
      sc.setOption({
        tooltip: { trigger:'item', formatter:'{b}: {d}%' },
        series: [{
          type:'pie', radius:['30%','70%'],
          data: sEntries.map(([k,v]) => ({name: esc(k), value: (v*100).toFixed(2)})),
          label: { color:'#b0b8c0', formatter:'{b}\n{d}%' },
          itemStyle: { borderRadius: 4 },
        }]
      });

      window.addEventListener('resize', () => { wc.resize(); sc.resize(); });
    }

    // Stress test
    $('allocStress').innerHTML = stress.length
      ? '<table class="risk-table" style="min-width:auto;"><tr><th>情景</th><th>影响</th><th>波动乘数</th></tr>' +
        stress.map(s => `<tr><td>${esc(s.scenario)}</td><td style="color:${s.impact_pct < 0 ? '#b42318' : '#1f7a5c'};">${(s.impact_pct > 0 ? '+' : '')}${s.impact_pct.toFixed(2)}%</td><td>${s.vol_multiplier}x</td></tr>`).join('') + '</table>'
      : '<div class="empty">暂无压力测试数据</div>';

    // Risk breakdown
    const rbEntries = Object.entries(risk_breakdown).sort((a,b) => b[1]-a[1]);
    $('allocRiskTable').innerHTML = rbEntries.length
      ? '<table class="risk-table" style="min-width:auto;"><tr><th>标的</th><th>风险贡献</th><th>权重</th></tr>' +
        rbEntries.map(([k,v]) => `<tr><td>${esc(k)}</td><td>${(v*100).toFixed(1)}%</td><td>${((weights[k]||0)*100).toFixed(1)}%</td></tr>`).join('') + '</table>'
      : '<tr><td>暂无风险分解数据</td></tr>';

    setStatus('资产配置完成');
  } catch(e) {
    $('allocMetrics').innerHTML = '<div class="empty">配置失败: '+esc(e.message)+'</div>';
  } finally { setBusy('allocRunBtn', false); }
}

// ── 2.0 Fama-MacBeth 截面回归 ──

async function loadFamaMacBeth() {
  const esc = v => String(v ?? '').replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c] || c);
  const nDates = parseInt($('famaDates')?.value || '40', 10);
  setBusy('famaRunBtn', true); setStatus('运行截面回归... (需约30秒)');
  try {
    const symbols = '600519,000858,002714,601899,002594,300750,600036,601318,000333,600276,000568,002415,000001,601166,600900,600887,601398,601939,601288,601988';
    const res = await api('/api/fama_macbeth?symbols='+encodeURIComponent(symbols)+'&step_days='+Math.max(5, Math.round(40/nDates)), undefined, { timeout: 180000, silent: true });
    const data = res?.data || {};
    if (data.status !== 'ok') {
      $('famaMetrics').innerHTML = '<div class="empty">截面回归失败: '+(data.status||'未知')+'</div>';
      return;
    }
    const fr = data.factor_results || {};
    const entries = Object.entries(fr);
    // Metrics
    const nSig = entries.filter(([,v]) => v.significant_5pct).length;
    $('famaMetrics').innerHTML = `
      <div class="metric"><div class="label">截面数</div><div class="value">${data.n_dates}</div></div>
      <div class="metric"><div class="label">范围</div><div class="value">${data.date_range}</div></div>
      <div class="metric"><div class="label">显著因子</div><div class="value" style="color:${nSig > 0 ? '#1f7a5c' : '#657282'};">${nSig}/${entries.length}</div></div>
      <div class="metric"><div class="label">Alpha均值</div><div class="value">${data.alpha_mean}</div></div>
      <div class="metric"><div class="label">R²均值</div><div class="value">${data.r_squared_mean}</div></div>
      <div class="metric"><div class="label">前向天数</div><div class="value">${data.forward_days}</div></div>
    `;
    // Factor table
    const sorted = entries.sort((a,b) => Math.abs(b[1].t_stat) - Math.abs(a[1].t_stat));
    $('famaFactorTable').innerHTML = '<table class="risk-table" style="min-width:auto;"><tr><th>因子</th><th>μ</th><th>t-stat</th><th>IR月</th><th>自相关</th><th>累计%</th><th>显著?</th></tr>' +
      sorted.map(([name, v]) => `<tr>
        <td>${esc(name)}</td>
        <td style="color:${v.mean > 0 ? '#087443' : '#b42318'};">${v.mean.toFixed(6)}</td>
        <td style="color:${Math.abs(v.t_stat) > 1.96 ? '#1f7a5c' : '#657282'};">${(v.t_stat > 0 ? '+' : '')}${v.t_stat.toFixed(2)}</td>
        <td>${v.sharpe_monthly.toFixed(3)}</td>
        <td>${(v.autocorr > 0 ? '+' : '')}${v.autocorr.toFixed(4)}</td>
        <td style="color:${v.cum_return_pct > 0 ? '#087443' : '#b42318'};">${(v.cum_return_pct > 0 ? '+' : '')}${v.cum_return_pct.toFixed(2)}%</td>
        <td>${v.significant_5pct ? '✅' : ' '}</td>
      </tr>`).join('') + '</table>';
    // Cumulative chart
    const daily = data.daily_results || [];
    if (daily.length > 1 && typeof echarts !== 'undefined') {
      const dates = daily.map(d => d.date.slice(5));
      const allFactors = Object.keys(daily[0].factor_returns);
      const top6 = sorted.slice(0, 6).map(([n]) => n);
      const cumSeries = [];
      for (const f of top6) {
        let cum = 0;
        cumSeries.push({
          name: esc(f),
          type: 'line',
          data: daily.map(d => { cum += d.factor_returns[f] || 0; return +(cum*100).toFixed(4); }),
          smooth: true,
          lineStyle: { width: 1.5 },
        });
      }
      const chart = echarts.init($('famaCumChart'));
      chart.setOption({
        tooltip: { trigger: 'axis' },
        legend: { textStyle: { color: '#b0b8c0' }, top: 0 },
        grid: { left: 60, right: 20, top: 40, bottom: 30 },
        xAxis: { type: 'category', data: dates, axisLabel: { color: '#657282', fontSize: 10 } },
        yAxis: { type: 'value', axisLabel: { color: '#657282', formatter: '{value}%' } },
        series: cumSeries,
      });
      window.addEventListener('resize', () => chart.resize());
    }
    // Daily table
    const last10 = daily.slice(-10).reverse();
    $('famaDailyTable').innerHTML = '<table class="risk-table" style="min-width:auto;"><tr><th>日期</th><th>n</th><th>Alpha</th><th>R²</th><th>Top因子</th></tr>' +
      last10.map(d => {
        const top = Object.entries(d.factor_returns).sort((a,b) => Math.abs(b[1])-Math.abs(a[1]))[0];
        return `<tr><td>${esc(d.date)}</td><td>${d.n}</td><td>${d.alpha.toFixed(6)}</td><td>${d.r_squared.toFixed(4)}</td><td>${esc(top[0])} ${(top[1]>0?'+':'')}${top[1].toFixed(6)}</td></tr>`;
      }).join('') + '</table>';
    setStatus('截面回归完成');
  } catch(e) {
    $('famaMetrics').innerHTML = '<div class="empty">错误: '+esc(e.message)+'</div>';
  } finally { setBusy('famaRunBtn', false); }
}

// 独立页面兼容入口改为主页面内嵌面板，避免丢失主页面已加载状态。
window.openIntegratedPanel = function(kind) {
  const url = kind === 'intraday' ? '/intraday.html' : kind === 'review' ? '/review.html' : '/research_dashboard.html';
  let host = document.getElementById('integratedPanelHost');
  if (!host) {
    host = document.createElement('div'); host.id = 'integratedPanelHost';
    host.style.cssText = 'position:fixed;inset:0;z-index:9999;background:#0a0e17;padding:12px;overflow:auto';
    host.innerHTML = '<button id="closeIntegratedPanel" style="position:fixed;right:18px;top:14px;z-index:2;background:#b42318;color:#fff;border:0;border-radius:6px;padding:8px 12px;cursor:pointer">× 返回主页面</button><iframe style="width:100%;height:calc(100vh - 24px);border:0;border-radius:8px;background:#0a0e17"></iframe>';
    document.body.appendChild(host); host.querySelector('#closeIntegratedPanel').onclick=()=>host.remove();
  }
  host.querySelector('iframe').src = url;
};

// ── 全局加载注册 ──
window.LOAD_RISK_optimizer = loadPortfolioOptimizer;
window.LOAD_OPS_mlmonitor = loadMLMonitor;
async function loadEtfFlow() {
  setBusy('etfRefreshBtn', true); setStatus('加载ETF资金与申赎状态...');
  try {
    const code = ($('etfSymbolSearch')?.value || '510300').trim();
    const days = ($('etfHistoryDays')?.value || '60');
    const d = await api('/api/v12/etf_flow?code='+encodeURIComponent(code)+'&days='+days);
    const rows = d.rows || [];
    const latest = rows[0] || {};
    const m = $('etfMetrics');
    const direction = flowDirectionLabel(d.signals?.flow_direction);
    const officialLabel = d.signals?.official_flow_available ? '真实逐日流水' : '暂无基金公司逐日申赎流水';
    if (m) m.innerHTML = [['数据日', d.as_of || '-'], ['历史范围', `${d.requested_days || days} / ${d.actual_days || d.history?.length || rows.length}日`], ['最新成交额', latest.amount_yi == null ? '-' : `${latest.amount_yi}亿`], ['最新份额差', d.signals?.latest_shares_delta_m == null ? '暂无' : `${d.signals.latest_shares_delta_m}百万份`], ['5日成交额变化', d.signals?.amount_change_5d == null ? '-' : `${d.signals.amount_change_5d}%`]].map(([l,v]) => `<div class="metric"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div></div>`).join('');
    if ($('etfStatus')) $('etfStatus').innerHTML = `<b>数据口径：</b>${esc(officialLabel)}<br><span class="data-meta">历史快照方向：${esc(direction)} · 观测区间 ${esc(d.signals?.share_observation_start || '未知')} 至 ${esc(d.signals?.share_observation_end || '未知')}</span><br><span class="data-meta">相邻观测份额差仅作历史方向参考，不代表单日净申购或净赎回。</span>`;
    const eventBox = $('etfEventList');
    if (eventBox) eventBox.innerHTML = (d.official_events || []).length
      ? d.official_events.slice(0, 6).map(event => `<article class="evidence-item"><div><strong>${safeHttpUrl(event.url) ? `<a href="${esc(safeHttpUrl(event.url))}" target="_blank" rel="noopener noreferrer">${esc(event.title || '官方事件')}</a>` : esc(event.title || '官方事件')}</strong><span>${esc(event.date || '日期缺失')} · ${esc(event.source || '')}</span></div><p>范围：市场级稳市/汇金信息，不作为该ETF每日申购赎回结论。</p></article>`).join('')
      : '<div class="data-state empty">暂无官方稳市/汇金事件参考。</div>';
    const recentHistory = (d.history?.length ? d.history : rows).slice().sort((a, b) => String(b.date || '').localeCompare(String(a.date || ''))).slice(0, 10);
     renderTable('etfTable', recentHistory, ['code','name','date','price','shares_m','shares_delta_m','share_change_notional_yi','scale_yi','amount_yi','main_net_yi']);
    if (d.history?.length && typeof echarts !== 'undefined') {
      const h=d.history, dates=h.map(x=>x.date), share= h.map(x=>x.shares_delta_m == null ? null : x.shares_delta_m), scale=h.map(x=>x.scale_yi == null ? null : x.scale_yi), amount=h.map(x=>x.amount_yi == null ? null : x.amount_yi);
      const a=echarts.init($('etfShareChart')), b=echarts.init($('etfScaleChart'));
      a.setOption({title:{text:'历史快照份额差（非逐日申赎流水）',textStyle:{color:'#9ca3af',fontSize:12}},tooltip:{trigger:'axis'},xAxis:{type:'category',data:dates},yAxis:{type:'value',name:'百万份'},dataZoom:[{type:'inside'},{type:'slider'}],series:[{type:'bar',name:'相邻观测份额差',data:share}]},true);
      b.setOption({tooltip:{trigger:'axis'},legend:{textStyle:{color:'#9ca3af'}},xAxis:{type:'category',data:dates},yAxis:{type:'value',name:'亿元'},dataZoom:[{type:'inside'},{type:'slider'}],series:[{type:'line',name:'总规模',data:scale},{type:'line',name:'成交额',data:amount}]},true);
      window.addEventListener('resize',()=>{a.resize();b.resize()});
    }
    setStatus('ETF资金已更新');
  } catch (e) { if ($('etfStatus')) $('etfStatus').textContent = `加载失败：${e.message}`; setStatus('ETF资金加载失败: '+e.message); }
  finally { setBusy('etfRefreshBtn', false); }
}

// loadMarketAttribution 已注册为 window.LOAD_MARKET_attribution

// ── 绑定事件 ──
document.addEventListener('DOMContentLoaded', () => {
  $('optRefreshBtn')?.addEventListener('click', () => { window.LOAD_RISK_optimizer(); loadERC(); });
  $('optEfficientFrontierBtn')?.addEventListener('click', loadEfficientFrontier);
  $('attrRefreshBtn')?.addEventListener('click', window.LOAD_MARKET_attribution);
  $('etfRefreshBtn')?.addEventListener('click', loadEtfFlow);
  document.querySelectorAll('[data-stock-flow-mode]').forEach(btn => btn.addEventListener('click', () => renderSelectedStockFlow(btn.dataset.stockFlowMode)));
  $('mhRefreshBtn')?.addEventListener('click', () => loadMarketHistoryCatalog(false));
  $('mhFetchBtn')?.addEventListener('click', () => loadMarketHistoryCatalog(true));
  $('mlmRefreshBtn')?.addEventListener('click', window.LOAD_OPS_mlmonitor);
  $('mlmRetrainBtn')?.addEventListener('click', mlRetrain);
  $('cacheRefreshBtn')?.addEventListener('click', loadCacheStatus);
  $('cacheRefreshAllBtn')?.addEventListener('click', refreshAllCache);
  $('eventBtRunBtn')?.addEventListener('click', runEventBacktest);
  //  财务因子
  $('finDataRefreshBtn')?.addEventListener('click', loadFinancialData);
  //  特征存储
  $('fsRefreshBtn')?.addEventListener('click', loadFeatureStore);
  //  漂移检测
  $('driftRefreshBtn')?.addEventListener('click', loadDriftCheck);
  $('driftRetrainBtn')?.addEventListener('click', async () => {
    if (!confirm('确定重新训练模型? 可能需要几分钟。')) return;
    setBusy('driftRetrainBtn', true); setStatus('开始重新训练...');
    try {
      const res = await api('/api/ml_retrain', {}, { timeout: 300000, cache_ttl: 0, silent: true });
      setStatus('重新训练: '+(res?.data?.status || res?.status || '完成'));
    } catch(e) { setStatus('训练失败: '+e.message); }
    finally { setBusy('driftRetrainBtn', false); loadDriftCheck(); }
  });
  //  资产配置
  $('allocRunBtn')?.addEventListener('click', loadAssetAllocation);
  // 2.0 Fama-MacBeth
  $('famaRunBtn')?.addEventListener('click', loadFamaMacBeth);
  // 2.0 滑点对比
  $('slipCompareBtn')?.addEventListener('click', loadSlipCompare);
});

// ── 2.0 滑点模型对比 ──

async function loadSlipCompare() {
  const esc = v => String(v ?? '').replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c] || c);
  const syms = $('slipSymbols')?.value || '002714,600519,000858';
  setBusy('slipCompareBtn', true); setStatus('运行滑点对比...');
  try {
    const models = ['flat','dynamic','almgren'];
    const results = await Promise.all(models.map(m =>
      api('/api/backtest_almgren?slippage='+m+'&symbols='+encodeURIComponent(syms), undefined, { timeout: 180000, silent: true })));
    const data = results.map((r, i) => ({
      model: models[i],
      metrics: r?.data?.metrics || {},
      trades: (r?.data?.trades || []).length,
    }));
    $('slipMetrics').innerHTML = data.map(d =>
      `<div class="metric"><div class="label">${d.model.toUpperCase()}</div><div class="value">回报=${((d.metrics.total_return||0)*100).toFixed(2)}% 夏普=${(d.metrics.sharpe||0).toFixed(2)} 交易=${d.trades}</div></div>`
    ).join('');
    // Chart
    if (typeof echarts !== 'undefined') {
      const ch = echarts.init($('slipChart'));
      ch.setOption({
        tooltip: { trigger:'axis' },
        legend: { textStyle:{color:'#b0b8c0'} },
        xAxis: { type:'category', data: ['回报%','夏普','交易数'], axisLabel:{color:'#657282'} },
        yAxis: [{ type:'value', axisLabel:{color:'#657282'} }],
        series: data.map(d => ({
          name: d.model.toUpperCase(),
          type: 'bar',
          data: [((d.metrics.total_return||0)*100).toFixed(2), (d.metrics.sharpe||0).toFixed(2), d.trades],
        })),
      });
    }
    $('slipTable').innerHTML = '<table class="risk-table"><tr><th>模型</th><th>总回报</th><th>夏普</th><th>回撤</th><th>交易</th><th>胜率</th></tr>' +
      data.map(d => `<tr><td>${d.model}</td><td>${((d.metrics.total_return||0)*100).toFixed(2)}%</td><td>${(d.metrics.sharpe||0).toFixed(2)}</td><td>${((d.metrics.max_drawdown||0)*100).toFixed(2)}%</td><td>${d.trades}</td><td>-</td></tr>`).join('') + '</table>';
    setStatus('滑点对比完成');
  } catch(e) {
    $('slipMetrics').innerHTML = '<div class="empty">对比失败: '+esc(e.message)+'</div>';
  } finally { setBusy('slipCompareBtn', false); }
}

/* ════════════════════════════════════════════════════════════
   统一前端增强集成（KPI、图表与交互）
   依赖：styles.css（P0 主题）、viz9.js（P1 图表库）、ui9.js（P2 交互库）
   全部防御式：组件缺失时静默跳过，不影响原有功能
   ════════════════════════════════════════════════════════════ */
window.INTEGRATED_VISUALS = (function () {
  const has = (lib) => typeof window[lib] !== 'undefined' && window[lib] !== null;
  const esc9 = v => String(v ?? '').replace(/[<>&"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'})[c] || c);

  /** 工具：在指定父容器后插入 chart-panel 容器 */
  function ensurePanel(parentEl, id, title) {
    if (!parentEl) return null;
    let el = document.getElementById(id);
    if (el) return el;
    el = document.createElement('div');
    el.id = id;
    el.className = 'chart-panel';
    el.innerHTML = '<div class="cp-title">' + esc9(title) + '</div>';
    el.style.minHeight = '280px';
    parentEl.appendChild(el);
    return el;
  }

  /** 工具：找到第一个可见的输入框（全局搜索跳转用） */
  function findVisibleSymbolInput() {
    const inputs = document.querySelectorAll('input[id*="ymbol"], input[id*="ode"]');
    for (let i = 0; i < inputs.length; i++) {
      if (inputs[i].offsetParent !== null) return inputs[i];
    }
    return inputs[0] || null;
  }

  /* ─────────── P0: 总览 KPI 仪表盘 ─────────── */
  function renderKpiGrid(market) {
    const grid = document.getElementById('kpiGrid');
    if (!grid) return;
    try {
      const unwrap = (p) => (p && p.data && typeof p.data === 'object' && !p.data.ashare && p.data.ok === undefined) ? p.data : (p && p.data ? p.data : p);
      const indices = unwrap(market.indices) || {};
      const temp = unwrap(market.temp) || {};
      const pulse = unwrap(market.pulse) || {};
      const margin = unwrap(market.margin) || {};

      const cards = [];
      // 主要指数
      const ash = indices.ashare || {};
      const idxNames = ['上证指数','深证成指','创业板指','沪深300'];
      const idxAlias = { '上证指数':'sh','深证成指':'sz','创业板指':'cy','沪深300':'hs300' };
      Object.entries(ash).forEach(([name, v]) => {
        if (!v) return;
        const pctV = v.change_pct ?? v.change ?? 0;
        cards.push({
          label: name,
          value: v.index != null ? Number(v.index).toFixed(2) : '-',
          delta: (pctV >= 0 ? '+' : '') + Number(pctV).toFixed(2) + '%',
          cls: pctV > 0 ? 'up' : (pctV < 0 ? 'down' : 'flat'),
          sub: '指数'
        });
      });
      // 市场温度
      if (temp.temperature != null) {
        cards.push({
          label: '市场温度',
          value: temp.temperature + '°',
          delta: '上涨 ' + (temp.advance ?? '-') + ' / 下跌 ' + (temp.decline ?? '-'),
          cls: temp.temperature >= 60 ? 'up' : (temp.temperature <= 40 ? 'down' : 'flat'),
          sub: 'MA300上方 ' + (temp.pct_above_ma300 != null ? temp.pct_above_ma300 + '%' : '-')
        });
      }
      // 恐贪指数
      const fg = pulse.index ?? pulse.fear_greed_index ?? pulse.fgi;
      if (fg != null) {
        cards.push({
          label: '恐贪指数',
          value: Number(fg).toFixed(1),
          delta: pulse.level ?? '市场脉搏',
          cls: fg >= 60 ? 'up' : (fg <= 40 ? 'down' : 'flat'),
          sub: '0-100'
        });
      }
      // 两融余额
      const totalM = margin.total_margin_balance ?? margin.total_margin ?? margin.total;
      if (totalM != null) {
        cards.push({
          label: '两融余额',
          value: formatMacroNumber(totalM, 0) + ' 亿',
          delta: '沪 ' + formatMacroNumber(margin.ss_margin_balance ?? margin.sh_margin ?? '-', 0) + ' / 深 ' + formatMacroNumber(margin.sz_margin_balance ?? margin.sz_margin ?? '-', 0),
          cls: 'flat',
          sub: '全市场融资融券'
        });
      }

      if (!cards.length) { grid.innerHTML = ''; return; }
      // 视觉层： 图标映射 + 涨跌箭头 + 胶囊 delta + 趋势条
      const KPI_ICONS = { '上证指数':'🏛️','深证成指':'🏙️','创业板指':'🚀','沪深300':'💎','市场温度':'🌡️','恐贪指数':'💓','两融余额':'🏦' };
      grid.innerHTML = cards.map(c => {
        const icon = KPI_ICONS[c.label] || '📈';
        const arrow = c.cls === 'up' ? '▲' : (c.cls === 'down' ? '▼' : '—');
        const deltaNum = parseFloat(String(c.delta).replace(/[^+\-\d.]/g, ''));
        const trendPct = isNaN(deltaNum) ? 0 : Math.max(6, Math.min(100, Math.abs(deltaNum) * 12 + 10));
        return `<div class="kpi-card kpi-${c.cls}">
          <div class="kpi-head"><div class="kpi-label">${esc9(c.label)}</div><div class="kpi-icon">${icon}</div></div>
          <div class="kpi-value">${esc9(c.value)}</div>
          <div class="kpi-delta ${c.cls}">${arrow} ${esc9(c.delta)}</div>
          <div class="kpi-trendbar"><i style="width:${trendPct}%"></i></div>
          <div class="kpi-sub">${esc9(c.sub || '')}</div>
        </div>`; }).join('');
    } catch (e) { console.error('KPI grid error:', e); }
  }

  /* ─────────── V13: 市场宽度 + Hero 条（/api/market） ─────────── */
  function renderMarketBreadth(payload) {
    const data = unwrapApiData(payload) || {};
    const el = $('breadthDisplay');
    const b = data.breadth || {};
    if (el && (b['上涨'] != null || b['下跌'] != null)) {
      const up = Number(b['上涨'] || 0), down = Number(b['下跌'] || 0), flat = Number(b['平盘'] || 0);
      const total = up + down + flat || 1;
      const pct = (n) => Math.round(n / total * 100);
      el.innerHTML = `
        <div class="breadth-bar">
          <div class="b-up" style="flex:${Math.max(up, 0.001)};">${pct(up)}% ↑</div>
          <div class="b-flat" style="flex:${Math.max(flat, 0.0001)};"></div>
          <div class="b-down" style="flex:${Math.max(down, 0.001)};">${pct(down)}% ↓</div>
        </div>
        <div class="breadth-stats">
          <span><span class="up-b">▲ 上涨</span> <b>${up}</b></span>
          <span><span class="down-b">▼ 下跌</span> <b>${down}</b></span>
          <span>平盘 <b>${flat}</b></span>
          <span>上涨占比 <b>${b['上涨占比%'] ?? '-'}%</b></span>
          <span>等权涨跌 <b>${b['全A等权涨跌幅%'] ?? '-'}%</b></span>
          <span>中位数 <b>${b['涨跌幅中位数%'] ?? '-'}%</b></span>
          <span>涨超5% <b>${b['涨超5%'] ?? '-'}</b></span>
          <span>跌超5% <b>${b['跌超5%'] ?? '-'}</b></span>
          <span>涨停 <b>${b['近似涨停'] ?? '-'}</b> / 跌停 <b>${b['近似跌停'] ?? '-'}</b></span>
        </div>`;
    } else if (el && !el.querySelector('.breadth-bar')) {
      el.innerHTML = '<div class="empty">暂无市场宽度数据</div>';
    }
    // Hero 条
    const fmtDate = (s) => { s = String(s || ''); return s.length === 8 ? `${s.slice(0,4)}-${s.slice(4,6)}-${s.slice(6,8)}` : s; };
    const set = (id, v) => { const e = $(id); if (e && v != null && v !== '') e.textContent = v; };
    set('heroTradeDate', fmtDate(data.trade_date));
    set('heroBreadthPct', (b['上涨占比%'] ?? '-') + '%');
    set('heroMedian', (b['涨跌幅中位数%'] ?? '-') + '%');
    set('heroLimit', (b['近似涨停'] ?? '-') + ' / ' + (b['近似跌停'] ?? '-'));
    const rp = $('heroRiskPill');
    if (rp && data.risk_level) {
      rp.style.display = '';
      rp.classList.toggle('hi', Number(data.risk) >= 4);
      const rt = $('heroRiskText'); if (rt) rt.textContent = data.risk_level + (data.risk != null ? ` · L${data.risk}` : '');
    }
  }

  function updateHeroMarketStatus() {
    const pill = $('heroMarketStatus');
    const txt = $('heroMarketStatusText');
    if (!pill || !txt) return;
    const now = new Date();
    const day = now.getDay();
    const mins = now.getHours() * 60 + now.getMinutes();
    const sessionOpen = day >= 1 && day <= 5 && ((mins >= 570 && mins <= 690) || (mins >= 780 && mins <= 900));
    const today = now.toISOString().slice(0, 10);
    const dataLive = window.__marketTradeDate === today;
    const trading = sessionOpen && dataLive;
    pill.classList.toggle('live', trading);
    pill.classList.toggle('closed', !trading);
    txt.textContent = trading ? '盘中 · 实时' : (sessionOpen ? `盘中 · 数据截至 ${window.__marketTradeDate || '未知'}` : '休市');
  }

  /* ─────────── P1: 回测净值曲线（追加到回测摘要后） ─────────── */
  function renderBacktestCurve(r) {
    if (!has('Viz9') || !r || !r.equity_curve || r.equity_curve.length < 2) return;
    const parent = document.getElementById('btSummary');
    if (!parent) return;
    const panel = ensurePanel(parent, 'btCurveIntegrated', '📈 净值曲线与回撤');
    const eq = r.equity_curve || [];
    const dates = eq.map(e => e.date || e.datetime || '');
    const init = eq[0]?.equity || r.initial_equity || 100000;
    const strategy = eq.map(e => init ? ((e.equity / init - 1) * 100) : 0);
    // 滚动回撤
    let peak = -Infinity;
    const drawdown = eq.map(e => {
      peak = Math.max(peak, e.equity);
      return peak ? ((e.equity / peak - 1) * 100) : 0;
    });
    window.Viz9.renderBacktestCurve(panel, {
      dates, strategy,
      benchmark: null,
      drawdown
    });
  }

  /* ─────────── P1: 板块轮动 Treemap（追加到轮动表后） ─────────── */
  function renderSectorTreemap(rows) {
    if (!has('Viz9') || !rows || !rows.length) return;
    const parent = document.getElementById('srHeatmapChart') || (document.getElementById('srChart') ? document.getElementById('srChart').parentElement : null);
    if (!parent) return;
    const panel = parent.id === 'srHeatmapChart' ? parent : ensurePanel(parent, 'srTreemapIntegrated', '🗺 板块涨跌热力图');
    window.Viz9.renderSectorTreemap(panel, rows.map(r => ({
      name: r.name || r.code || '',
      value: Math.abs(Number(r.change_pct) || 0) * 1000 + 100, // 面积权重：涨跌幅绝对值
      change: Number(r.change_pct) || 0
    })));
  }

  /* ─────────── P1: 因子 IC 热力图（因子分析面板） ─────────── */
  function renderIcHeatmapIntegrated(factorIc) {
    if (!has('Viz9') || !factorIc || !Object.keys(factorIc).length) return;
    const parent = document.getElementById('factorChart') ? document.getElementById('factorChart').parentElement : null;
    if (!parent) return;
    const panel = ensurePanel(parent, 'icHeatIntegrated', '🌡 因子 IC 排名');
    const entries = Object.entries(factorIc).sort((a, b) => b[1] - a[1]);
    // 用条形替代热力图（单期截面数据，无时序）
    window.Viz9.renderRiskWaterfall(panel, {
      breakdown: entries.map(([n, v]) => ({ name: n, value: Number(v) || 0 }))
    });
  }

  /* ─────────── P1: 市场温度 gauge（替换增强） ─────────── */
  function renderTempGaugeIntegrated(data) {
    if (!has('Viz9') || !data) return;
    const dom = document.getElementById('tempGaugeDisplay');
    if (!dom) return;
    const temp = data.temperature;
    if (temp == null) return;
    const panel = ensurePanel(dom, 'tempGaugeIntegrated', '🌡 市场温度');
    panel.style.minHeight = '200px';
    window.Viz9.renderGaugeMini(panel, { value: Number(temp), max: 100, name: '市场温度', unit: '°' });
  }

  /* ─────────── P1: 组合风险瀑布（组合风险面板） ─────────── */
  function renderRiskIntegrated(riskData) {
    if (!has('Viz9') || !riskData) return;
    const parent = document.getElementById('prMetrics');
    if (!parent) return;
    const panel = ensurePanel(parent, 'riskBarIntegrated', '📊 风险指标概览');
    const v = riskData.var || {};
    const items = [];
    if (v.var_95 != null) items.push({ name: 'VaR95', value: Number(v.var_95) });
    if (v.var_99 != null) items.push({ name: 'VaR99', value: Number(v.var_99) });
    if (v.cvar_95 != null) items.push({ name: 'CVaR95', value: Number(v.cvar_95) });
    if (v.volatility != null) items.push({ name: '年化波动', value: Number(v.volatility) });
    const c = riskData.concentration || {};
    Object.entries(c.industry_map || {}).slice(0, 8).forEach(([ind, pct]) => {
      items.push({ name: '行业:' + ind, value: Number(String(pct).replace('%', '')) });
    });
    if (!items.length) { panel.style.display = 'none'; return; }
    window.Viz9.renderRiskWaterfall(panel, { breakdown: items });
  }

  /* ─────────── P2: 全局搜索增强（注册 UI9.GlobalSearch） ─────────── */
  function initGlobalSearch() {
    if (!has('UI9') || !document.getElementById('globalSearchInput')) return;
    try { new window.UI9.GlobalSearch(); } catch (e) { console.error('GlobalSearch:', e); }
  }

  /* ─────────── P2: 自选股按钮增强（挂在个股标题旁） ─────────── */
  function initWatchlistButtons() {
    if (!has('UI9')) return;
    const inject = (btn, code, name) => {
      if (!btn || !code) return;
      if (btn.parentElement.querySelector('.wl-toggle-integrated')) return;
      const t = document.createElement('button');
      t.className = 'btn-sm wl-toggle-integrated';
      try {
        window.UI9.Watchlist.toggleButton(t, code, name || code);
        btn.parentElement.insertBefore(t, btn.nextSibling);
      } catch (e) { console.error('watchlist btn:', e); }
    };
    // 快速查询区
    const qa = document.getElementById('qaSymbol');
    if (qa) inject(qa, qa.value.trim(), '');
    // 个股页
    const sSym = document.getElementById('sSymbol');
    if (sSym) {
      const btn = document.getElementById('loadChartBtn');
      inject(btn || sSym, sSym.value.trim(), '');
      sSym.addEventListener('change', () => inject(document.getElementById('loadChartBtn'), sSym.value.trim(), ''));
    }
  }

  /* ─────────── P2: 表格增强（扫描/回测/实时表支持排序导出） ─────────── */
  function enhanceTables() {
    if (!has('UI9')) return;
    // 延迟到表格渲染后
    setTimeout(() => {
      const targets = [
        ['scanTable', ['symbol','action','close','composite_score','trend_score','risk_score','suggested_shares']],
        ['backtestTable', ['symbol','total_return_pct','max_drawdown_pct','sharpe','trade_count','win_rate_pct','final_equity']],
        ['sectorTable', ['sector_name','symbol']],
        ['srTable', ['name','change_pct','price','change']],
      ];
      targets.forEach(([id]) => {
        const el = document.getElementById(id);
        if (!el || el.dataset.integratedEnhanced) return;
        const wrap = el.closest('.tableWrap') || el.parentElement;
        if (!wrap) return;
        el.dataset.integratedEnhanced = '1';
      });
    }, 800);
  }

  /* ─────────── 初始化：hook 到现有渲染流程 ─────────── */
  function init() {
    initGlobalSearch();
    initWatchlistButtons();
    updateHeroMarketStatus();
    setInterval(updateHeroMarketStatus, 60000);
    // 监听自定义事件（由下方 renderMarket 等 hook 触发）
    document.addEventListener('visual:backtest', (e) => renderBacktestCurve(e.detail));
    document.addEventListener('visual:sector', (e) => renderSectorTreemap(e.detail));
    document.addEventListener('visual:ic', (e) => renderIcHeatmapIntegrated(e.detail));
    document.addEventListener('visual:temp', (e) => renderTempGaugeIntegrated(e.detail));
    document.addEventListener('visual:risk', (e) => renderRiskIntegrated(e.detail));
    setInterval(enhanceTables, 3000);
  }

  return { init, renderKpiGrid, renderBacktestCurve, renderSectorTreemap, renderIcHeatmapIntegrated, renderTempGaugeIntegrated, renderRiskIntegrated, initGlobalSearch, initWatchlistButtons };
})();

// 增强能力只注册一次；实际初始化由统一启动入口执行。

// ════════════════════════════════════════════════
// V10: 因子工坊（因子库 / 个股扫描 / IC / 合成）
// ════════════════════════════════════════════════

let _factorLib = null;  // 因子库缓存（内存）
let _factorOverview = null;

function factorNum(v, digits=3) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(digits) : '—';
}
function factorPct(v) {
  const n = Number(v);
  return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : '—';
}
function factorTierBadge(tier) {
  const labels = {core:['核心','#49c78a'],candidate:['候选','#f0b429'],observe:['观察','#8ea0b8'],blocked:['阻断','#f06a69']};
  const [label,color] = labels[tier] || [tier || '未知','#8ea0b8'];
  return `<span class="factor-tier" style="--factor-tier-color:${color}">${label}</span>`;
}
function factorStateBadge(f) {
  if (f.oos_verified && f.eligible) return '<span class="factor-state good">OOS已验证</span>';
  if (f.eligible) return '<span class="factor-state warn">待复核</span>';
  return '<span class="factor-state bad">未通过</span>';
}

async function loadFactorOverview(refresh=false) {
  const meta = $('foMeta');
  if (meta) meta.textContent = '正在读取质量账本、IC和真实回测...';
  try {
    const opts = { cache_ttl: refresh ? 0 : 300, refresh };
    const [quality, ic, bt] = await Promise.all([
      api('/api/factor_quality', undefined, opts),
      api('/api/factor_ic?start=20230101', undefined, opts),
      api('/api/backtest_dashboard', undefined, opts),
    ]);
    if (!quality.ok) throw new Error(quality.error || '质量账本不可用');
    const factors = Array.isArray(quality.factors) ? quality.factors : [];
    const icMap = Object.fromEntries((ic.factors || []).map(x => [x.name, x]));
    _factorOverview = { ...quality, factors: factors.map(f => ({...f, ...(icMap[f.factor] || {})})), backtest: bt.data || {} };
    renderFactorOverview();
    if (meta) meta.textContent = `质量账本 ${quality.generated_at || '未知'} · 来源 ${quality.source || '本地账本'} · ${factors.length} 个因子`;
  } catch (e) {
    if (meta) meta.innerHTML = `<span class="factor-error">加载失败：${escFactor(e.message)}</span>`;
    if ($('foGate')) $('foGate').innerHTML = '<span class="factor-error">无法确认因子是否可用，禁止将页面结果当作交易信号。</span>';
  }
}

function renderFactorOverview() {
  const d = _factorOverview || {}, c = d.counts || {};
  const bt = d.backtest?.factor_backtest || {}, validation = bt.validation || {}, iteration = bt.iteration || {}, summary = bt.summary || {};
  const gate = $('foGate');
  const eligible = Number(c.eligible || 0), total = Number(c.total || 0);
  const deployable = validation.deployable === true;
  if (gate) gate.innerHTML = `<span class="factor-gate-icon">${deployable ? '✓' : '!'}</span><div><b>${deployable ? '滚动回测门禁通过，可进入观察执行' : '当前仅研究级，禁止自动实盘'}</b><span>策略 ${escFactor(bt.strategy_version || '未生成')} · 冠军 ${escFactor(iteration.champion || '—')} · ${summary.window_count || 0} 个滚动窗口 · 注册表 ${eligible}/${total} 因子通过</span></div>`;
  const kpis = $('foKpis');
  if (kpis) kpis.innerHTML = [
    ['滚动入选', summary.factor_count_used ?? '—', `${iteration.lifecycle?.latest_selected?.length || 0} 个当前权重`],
    ['滚动窗口', summary.window_count ?? '—', `embargo ${validation.embargo_days ?? '—'} 日`],
    ['冠军策略', iteration.champion || '—', bt.strategy_version || '未生成版本'],
    ['验证状态', validation.status || '缺失', deployable ? '可进入观察执行' : '研究级，禁止自动实盘'],
  ].map(([label,value,note]) => `<div class="factor-kpi"><span>${label}</span><b>${escFactor(value)}</b><small>${escFactor(note)}</small></div>`).join('');
  renderFactorWeightChart();
  renderFactorBacktest();
  renderFactorOverviewTable();
}

function renderFactorWeightChart() {
  const el = $('foWeightChart');
  if (!el) return;
  const bt = _factorOverview?.backtest?.factor_backtest || {};
  const lifecycle = bt.iteration?.lifecycle || {};
  const rows = (lifecycle.factors || []).filter(f => Number(f.latest_weight || 0) > 0).sort((a,b) => Number(b.latest_weight||0)-Number(a.latest_weight||0));
  if (!rows.length) { el.innerHTML = '<div class="empty">当前滚动窗口没有通过深筛的因子</div>'; return; }
  const maxWeight = Math.max(...rows.map(f => Number(f.latest_weight || 0)), 0.0001);
  el.innerHTML = rows.map(f => {
    const strength = Number(f.latest_weight || 0) / maxWeight * 100;
    return `<div class="factor-weight-row"><span title="${escFactor(f.factor)} · ${escFactor(f.state)}">${escFactor(f.factor)}</span><div class="factor-track"><i style="width:${strength.toFixed(1)}%"></i></div><b>${factorPct(f.latest_weight)}</b></div>`;
  }).join('');
  const delta = bt.iteration?.since_previous_version || {};
  if ($('foWeightNote')) $('foWeightNote').textContent = `本轮晋级 ${delta.promoted?.length || 0} · 淘汰 ${delta.demoted?.length || 0} · 权重仅来自最近训练窗`;
}

function renderFactorBacktest() {
  const el = $('foBacktest'), bt = _factorOverview?.backtest?.factor_backtest || {};
  if (!el) return;
  if (!Object.keys(bt).length) { el.innerHTML = '<div class="factor-empty-state">真实因子回测产物缺失，禁止用占位收益。</div>'; return; }
  const ls = bt.long_short || {}, lg = bt.long_group1 || {}, bm = bt.benchmark || {};
  const validation = bt.validation || {}, summary = bt.summary || {}, iteration = bt.iteration || {};
  const risks = (validation.risk_flags || []).map(x => escFactor(x)).join('；');
  const warning = validation.deployable
    ? `滚动门禁通过；冠军 ${escFactor(iteration.champion || '—')}，仍需实盘前观察。`
    : `研究级结果，禁止自动实盘。${risks}`;
  el.innerHTML = `<div class="factor-backtest-warning">${warning}</div><div class="factor-backtest-grid">${[['多头净年化',factorPct((lg.annual_return_pct || 0)/100),`${summary.window_count || 0}个滚动窗口`],['多头回撤',factorPct((lg.max_drawdown_pct || 0)/100),'风险门禁'],['多头Sharpe',factorNum(lg.sharpe,2),validation.status || '研究级'],['双腿成本后多���年化',factorPct((ls.annual_return_pct || 0)/100),'多头+空头均扣成本'],['基准年化',factorPct((bm.annual_return_pct || 0)/100),'等权基准'],['策略版本',escFactor(bt.strategy_version || '—'),iteration.champion || '—']].map(x => `<div><span>${x[0]}</span><b>${x[1]}</b><small>${escFactor(x[2])}</small></div>`).join('')}</div>`;
}

function renderFactorOverviewTable() {
  const tbody = $('foTbody'); if (!tbody) return;
  const q = ($('foSearch')?.value || '').toLowerCase().trim(), tier = $('foTier')?.value || '', oos = $('foOos')?.value || '';
  const fs = (_factorOverview?.factors || []).filter(f => (!q || String(f.factor || f.name).toLowerCase().includes(q)) && (!tier || f.tier === tier) && (!oos || (oos === 'verified' ? f.oos_verified : !f.oos_verified))).sort((a,b) => ({core:0,candidate:1,observe:2,blocked:3}[a.tier] ?? 9) - ({core:0,candidate:1,observe:2,blocked:3}[b.tier] ?? 9) || Math.abs(Number(b.oos_icir || b.icir || 0)) - Math.abs(Number(a.oos_icir || a.icir || 0)));
  if (!fs.length) { tbody.innerHTML = '<tr><td colspan="10"><div class="empty">没有匹配因子</div></td></tr>'; return; }
  tbody.innerHTML = fs.slice(0,150).map(f => `<tr><td><b>${escFactor(f.factor || f.name)}</b><small>${escFactor(f.category || '')}</small></td><td>${factorTierBadge(f.tier)}</td><td>${f.eligible ? factorNum(Math.abs(Number(f.oos_icir || f.icir || 0)),2) : '—'}</td><td>${Number(f.direction) < 0 ? '反向' : '正向'}</td><td>${factorNum(f.ic_mean,4)}</td><td>${factorNum(f.icir,3)}</td><td>${factorNum(f.oos_ic,4)}</td><td>${factorNum(f.oos_icir,2)}</td><td>${factorPct(f.coverage)}</td><td>${factorStateBadge(f)} <small>${escFactor(f.reason || '')}</small></td></tr>`).join('');
}

async function getFactorLib(refresh) {
  if (_factorLib && !refresh) return _factorLib;
  const d = await api('/api/factor_library', undefined, { cache_ttl: 600, refresh: !!refresh });
  _factorLib = d;
  return d;
}

function escFactor(s) {
  if (s === null || s === undefined || s === '') return '—';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function factorPayload(response) {
  if (!response || typeof response !== 'object') return {};
  return response.scan || response.data || response;
}

function normalizeFactorItem(item) {
  const f = item || {};
  return {
    ...f,
    name: f.name ?? f.factor ?? f.factor_name ?? f.factorName ?? '未命名因子',
    value: f.value ?? f.score ?? f.z_score ?? f.zscore ?? f.exposure ?? null,
    category: f.category ?? f.family ?? f.group ?? '其他',
  };
}

function normalizeFactorScan(scan) {
  const s = scan || {};
  return {
    ...s,
    extremes: (s.extremes || s.extreme_factors || s.alerts || []).map(normalizeFactorItem),
    factors: (s.factors || s.factor_values || s.items || []).map(normalizeFactorItem),
  };
}

async function renderFactorLibrary() {
  const tbody = $('flTbody');
  if (!tbody) return;
  try {
    const lib = await getFactorLib();
    const q = ($('flSearch')?.value || '').trim().toLowerCase();
    const cat = $('flCat')?.value || '';
    const usage = $('flUsage')?.value || '';
    const list = (lib.factors || []).filter(f => {
      if (cat && f.category !== cat) return false;
      if (usage && f.usage !== usage) return false;
      if (q && !(f.name.toLowerCase().includes(q) || (f.desc||'').toLowerCase().includes(q) || f.category.toLowerCase().includes(q))) return false;
      return true;
    });
    const sumEl = $('flSummary');
    if (sumEl) sumEl.textContent = `共 ${lib.total} 因子 · 量化候选 ${lib.summary?.quant||0} · 分析用 ${lib.summary?.analysis||0} · 当前显示 ${list.length}`;
    if (!list.length) { tbody.innerHTML = '<tr><td colspan="9"><div class="empty">无匹配因子</div></td></tr>'; return; }
    tbody.innerHTML = list.map(f => {
      const hi = f.extreme_hi != null ? escFactor(f.extreme_hi) : (f.pct_hi ? (f.pct_hi*100)+'%' : '—');
      const lo = f.extreme_lo != null ? escFactor(f.extreme_lo) : (f.pct_lo ? (f.pct_lo*100)+'%' : '—');
      const usageBadge = f.usage === 'quant'
        ? '<span style="background:#1a5c3a;color:#6fe3a1;padding:2px 8px;border-radius:10px;font-size:12px;">量化候选</span>'
        : '<span style="background:#7c3aed;color:#d8b4fe;padding:2px 8px;border-radius:10px;font-size:12px;">分析用</span>';
      return `<tr>
        <td><b>${escFactor(f.name)}</b></td>
        <td>${escFactor(f.category)}</td>
        <td>${escFactor(f.weight)}</td>
        <td>${escFactor(f.direction_desc)}</td>
        <td>${hi} / ${lo}</td>
        <td>${escFactor((f.pct_hi*100)+'% / '+(f.pct_lo*100)+'%')}</td>
        <td>${escFactor((f.z_hi??'—')+' / '+(f.z_lo??'—'))}</td>
        <td>${usageBadge}</td>
        <td style="color:#9aa4b2;font-size:12px;">${escFactor(f.desc)}</td>
      </tr>`;
    }).join('');
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="9"><div class="empty">因子库加载失败: ${escFactor(e.message)}</div></td></tr>`;
  }
}

// 个股扫描（因子工坊页复用）
async function loadFactorScanFrom(inputId, resultId) {
  const sym = ($(inputId)?.value || '').trim();
  if (!sym) return;
  const box = $(resultId);
  if (box) box.innerHTML = '<div class="empty">扫描中…（44个定义因子，显示可用/缺失状态）</div>';
  try {
    const d = await api('/api/factor_scan?symbol=' + encodeURIComponent(sym), undefined, { cache_ttl: 300 });
    renderFactorScanResult(box, normalizeFactorScan(factorPayload(d)));
  } catch (e) {
    if (box) box.innerHTML = `<div class="empty">扫描失败: ${escFactor(e.message)}</div>`;
  }
}

// 个股分析页的多因子扫描也复用同一渲染
async function loadFactorScan() {
  const sym = $('factorSymbol')?.value || state.lastKlineSymbol || '002714';
  const box = $('factorScanResult');
  if (box) box.innerHTML = '<div class="empty">扫描中…（44个定义因子，显示可用/缺失状态）</div>';
  try {
    const d = await api('/api/factor_scan?symbol=' + encodeURIComponent(sym), undefined, { cache_ttl: 300 });
    renderFactorScanResult(box, normalizeFactorScan(factorPayload(d)));
  } catch (e) {
    if (box) box.innerHTML = `<div class="empty">扫描失败: ${escFactor(e.message)}</div>`;
  }
}

function renderFactorScanResult(box, scan) {
  if (!box) return;
  scan = normalizeFactorScan(scan);
  if (!scan) { box.innerHTML = '<div class="empty">无数据</div>'; return; }
  const lvl = scan.level || '正常';
  const lvlColor = lvl === '危险' ? '#fca5a5' : (lvl === '警惕' ? '#fcd34d' : '#6fe3a1');
  const lvlBg = lvl === '危险' ? '#7f1d1d' : (lvl === '警惕' ? '#78350f' : '#14532d');
  const ex = scan.extremes || [];
  const facs = scan.factors || [];
  const definedCount = Number(scan.n_factor_definitions ?? scan.n_factors ?? facs.length);
  const availableCount = Number(scan.n_available_factors ?? facs.filter(f => f.available !== false && f.value != null).length);
  const missingCount = Number(scan.n_missing_factors ?? Math.max(0, definedCount - availableCount));
  const exHtml = ex.length
    ? `<div class="factor-extreme-list">${ex.map(x => `
      <div class="factor-extreme-row">
        <div class="factor-extreme-head">
          <span class="factor-extreme-name">${escFactor(x.name)}</span>
          <span class="factor-extreme-value">${escFactor(x.value)}</span>
        </div>
        <div class="factor-extreme-meta">
          <span>${escFactor(x.category)}</span>
          <span>权重 ${escFactor(x.weight)}</span>
          ${x.direction != null ? `<span>${Number(x.direction) >= 0 ? '正向风险' : '反向风险'}</span>` : ''}
        </div>
        ${x.message ? `<div class="factor-extreme-message">${escFactor(x.message)}</div>` : ''}
      </div>`).join('')}</div>`
    : '<div class="ok-note">无极端因子</div>';
  let html = `
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px;">
      <div style="background:${lvlBg};color:${lvlColor};padding:10px 18px;border-radius:10px;font-size:18px;font-weight:700;">${escFactor(lvl)}</div>
      <div style="padding:8px 0;"><b>综合评分</b> <span style="font-size:20px;color:#e5e7eb;">${escFactor(scan.score)}</span> / 100</div>
      <div style="padding:8px 0;"><b>因子口径</b> <span style="font-size:18px;color:#e5e7eb;">定义 ${definedCount} / 可用 ${availableCount} / 缺失 ${missingCount}</span></div>
      <div style="padding:8px 0;"><b>极端因子</b> <span style="font-size:20px;color:#fca5a5;">${ex.length}</span> 个</div>
    </div>
    <div style="margin-bottom:12px;"><b>⚠️ 极端因子明细（权重小也提醒）：</b><br/>${exHtml}</div>`;
  if (facs.length) {
    html += `<details style="margin-top:10px;"><summary style="cursor:pointer;color:#9aa4b2;">查看全部 ${definedCount} 个定义因子（可用 ${availableCount}）</summary><div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;">` +
      facs.map(f => {
        const bad = (f.extreme === true) || (f.level === '极端');
        const missing = f.available === false || f.status === 'missing' || f.value == null;
        const style = bad ? 'border-color:#7f1d1d;color:#fca5a5;' : (missing ? 'opacity:.6;border-style:dashed;' : '');
        const value = missing ? `缺失${f.missing_reason ? ' · '+f.missing_reason : ''}` : `<b>${escFactor(f.value)}</b>`;
        return `<span class="factor-all" style="${style}" title="${missing ? escFactor(f.missing_reason || '数据缺失') : ''}">${escFactor(f.name)}: ${value}</span>`;
      }).join('') + '</div></details>';
  }
  box.innerHTML = html;
}

// IC 分析（因子与未来收益相关性）
async function loadFactorIC() {
  const metrics = $('ficMetrics');
  const start = $('ficStart')?.value || '20230101';
  const end = $('ficEnd')?.value || '';
  const tbody = $('ficTbody'), chart = $('ficChart');
  if (metrics) metrics.innerHTML = '<div class="empty">计算中…</div>';
  try {
    const d = factorPayload(await api(`/api/factor_ic?start=${start}&end=${end}`, undefined, { cache_ttl: 1800, timeout: 60000 }));
    if (metrics) metrics.innerHTML = `<span style="color:#6fe3a1;">✅ IC 计算完成：${d.factors?.length || 0} 因子，区间 ${escFactor(d.range||'')}</span>`;
    if (tbody) {
      const rows = d.factors || [];
      if (!rows.length) tbody.innerHTML = '<tr><td colspan="5"><div class="empty">无数据（数据量不足）</div></td></tr>';
      else tbody.innerHTML = rows.map(f => `<tr>
        <td><b>${escFactor(f.name)}</b></td>
        <td style="color:${f.ic_mean>=0?'#fca5a5':'#6fe3a1'};">${escFactor(f.ic_mean)}</td>
        <td>${escFactor(f.icir)}</td>
        <td>${escFactor(f.win_rate)}%</td>
        <td>${escFactor(f.rating)}</td></tr>`).join('');
    }
    if (chart && d.chart) {
      if (state.factorIcChart) state.factorIcChart.dispose();
      state.factorIcChart = echarts.init(chart);
      state.factorIcChart.setOption(d.chart);
    }
  } catch (e) {
    if (metrics) metrics.innerHTML = `<span style="color:#fca5a5;">IC 计算失败: ${escFactor(e.message)}</span>`;
  }
}

// 因子合成评估
async function loadFactorCombine() {
  const metrics = $('fcMetrics');
  const method = $('fcMethod')?.value || 'equal';
  const chart = $('fcChart');
  if (metrics) metrics.innerHTML = '<div class="empty">合成评估中…</div>';
  try {
    const d = factorPayload(await api('/api/factor_combine?method=' + encodeURIComponent(method), undefined, { cache_ttl: 1800, timeout: 60000 }));
    if (metrics) metrics.innerHTML = `<span style="color:#6fe3a1;">✅ 合成完成：方法=${escFactor(d.method||'')}，IC=${escFactor(d.ic_mean)}，ICIR=${escFactor(d.icir)}</span>`;
    if (chart && d.chart) {
      if (state.factorCombineChart) state.factorCombineChart.dispose();
      state.factorCombineChart = (echarts.getInstanceByDom && echarts.getInstanceByDom(chart)) || echarts.init(chart);
      state.factorCombineChart.setOption(d.chart); setTimeout(() => state.factorCombineChart && state.factorCombineChart.resize(), 120);
    }
  } catch (e) {
    if (metrics) metrics.innerHTML = `<span style="color:#fca5a5;">合成评估失败: ${escFactor(e.message)}</span>`;
  }
}

// 因子工坊初始化：第一屏加载真实质量/OOS/回测决策总览。
window.V10_FACTORS_INIT = function() {
  if ($('foTbody')) loadFactorOverview();
};


// ════════════════════════════════════════════════
// V10: AI 多模型分析（选模型 → OpenClaw → 指定模型分析）
// ════════════════════════════════════════════════

let _aiAnalyzing = false;
async function loadStockAiAnalysis() {
  if (_aiAnalyzing) return;
  const sym = $('sSymbol')?.value || state.lastKlineSymbol || '002714';
  const model = $('aiModel')?.value || 'deepseek/deepseek-v4-flash';
  const atype = $('aiType')?.value || 'panorama';
  const box = $('aiAnalyzeResult');
  if (!box) return;
  _aiAnalyzing = true;
  const btn = $('aiAnalyzeBtn');
  if (btn) btn.disabled = true;
  const t0 = Date.now();
  box.innerHTML = `<div class="empty">⏳ ${escFactor(model)} 正在分析（上下文：全景+因子+行情）… 约1-3分钟</div>`;
  try {
    const d = await api(`/api/ai_analyze?symbol=${encodeURIComponent(sym)}&model=${encodeURIComponent(model)}&type=${encodeURIComponent(atype)}`,
      undefined, { timeout: 300000, cache_ttl: 0 });
    const secs = Math.round((Date.now() - t0) / 1000);
    const reply = d.reply || {};
    const text = reply.text || JSON.stringify(reply, null, 2);
    const stderr = reply.stderr ? `<div style="color:#fca5a5;font-size:12px;margin-top:8px;">stderr: ${escFactor(reply.stderr.slice(0, 500))}</div>` : '';
    box.innerHTML = `
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;">
        <span style="background:#1f2937;color:#e8c547;padding:4px 12px;border-radius:14px;font-size:12px;">${escFactor(model)}</span>
        <span style="background:#1f2937;color:#9aa4b2;padding:4px 12px;border-radius:14px;font-size:12px;">${escFactor(sym)} · ${escFactor(atype)}</span>
        <span style="background:#14532d;color:#6fe3a1;padding:4px 12px;border-radius:14px;font-size:12px;">✅ ${secs}s</span>
      </div>
      <div class="panel" style="white-space:pre-wrap;line-height:1.7;font-size:14px;">${escFactor(text)}</div>${stderr}`;
  } catch (e) {
    box.innerHTML = `<div class="empty">分析失败: ${escFactor(e.message)}</div>`;
  } finally {
    _aiAnalyzing = false;
    if (btn) btn.disabled = false;
  }
}

/* ════════════════════════════════════════════════════════════════
   统一实现 监控新页面：告警中心 / 数据健康 / 定时任务状态
   端点：/api/alerts_status、/api/data_health、/api/tasks_status
   约定：异常也是 200 + {ok:false,error}，页面渲染错误提示而非崩溃。
   自动刷新：三个页面 60s 定时刷新（仅当对应 sub-view 激活时）。
   ════════════════════════════════════════════════════════════════ */

// ── 告警中心 ──
async function loadAlertsStatus() {
  const box = $('alertsRecent'), guard = $('alertsGuardian'), dedup = $('alertsDedup'), meta = $('alertsMeta');
  if (!box) return;
  try {
    const d = await api('/api/alerts_status', undefined, { cache_ttl: 0, silent: true });
    if (!d.ok) throw new Error(d.error || '告警状态失败');
    const data = d.data || {};
    if (meta) meta.textContent = `最近告警 ${(data.recent_alerts || []).length} 条`;
    // 最近告警表
    const alerts = data.recent_alerts || [];
    box.innerHTML = alerts.length
      ? '<table class="risk-table"><thead><tr><th>代码</th><th>动作</th><th>价格</th><th>最后发送</th></tr></thead><tbody>' +
        alerts.map(a => `<tr><td><b>${esc(a.symbol || '-')}</b></td><td>${esc(a.action || '-')}</td><td>${esc(a.price || '-')}</td><td style="font-size:12px;color:#8b949e;">${esc(a.time || '-')}</td></tr>`).join('') +
        '</tbody></table>'
      : '<div class="empty">暂无告警记录</div>';
    // 守护进程检查
    const g = data.guardian || {};
    const tx = g.tx || {};
    const txHtml = Object.keys(tx).length
      ? Object.entries(tx).map(([k, v]) =>
          `<div style="display:flex;justify-content:space-between;gap:12px;padding:5px 0;border-bottom:1px solid #21262d;font-size:13px;"><span>${esc(k)}</span><span style="color:#8b949e;">${esc(v)}</span></div>`).join('')
      : '<div class="empty">无检查记录</div>';
    const mem = g.last_mem || {};
    const memHtml = Object.keys(mem).length
      ? '<div style="margin-top:10px;font-size:12px;color:#8b949e;">内存快照（前8）:</div>' +
        Object.entries(mem).slice(0, 8).map(([k, v]) =>
          `<div style="display:flex;justify-content:space-between;gap:12px;padding:3px 0;font-size:12px;"><span style="color:#8b949e;">${esc(k)}</span><span>${v && v.rss_mb != null ? v.rss_mb + ' MB' : esc(JSON.stringify(v))}</span></div>`).join('')
      : '';
    guard.innerHTML = txHtml + memHtml +
      (g.mem_over_count != null ? `<div style="margin-top:8px;">内存超限次数: <b>${g.mem_over_count}</b></div>` : '');
    // 去重状态（channel 最后发送）
    const dd = data.dedup_state || {};
    const ddEntries = Object.entries(dd).filter(([k]) => !String(k).startsWith('_'));
    dedup.innerHTML = ddEntries.length
      ? '<table class="risk-table"><thead><tr><th>签名</th><th>最后时间戳</th></tr></thead><tbody>' +
        ddEntries.slice(0, 20).map(([k, v]) => `<tr><td style="font-size:12px;">${esc(k)}</td><td>${esc(v)}</td></tr>`).join('') + '</tbody></table>'
      : '<div class="empty">暂无去重记录</div>';
  } catch (e) {
    box.innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
}

// ── 数据健康 ──
let dataHealthEntries = [];
function renderDataHealthEntries() {
  const box = $('dhTable'); if (!box) return;
  const query = ($('dhSearch')?.value || '').trim().toLowerCase();
  const staleOnly = Boolean($('dhStaleOnly')?.checked);
  const entries = dataHealthEntries.filter(e => (!staleOnly || e.stale) && (!query || String(e.file || '').toLowerCase().includes(query)));
  box.innerHTML = entries.length
    ? '<table class="risk-table"><thead><tr><th>数据集</th><th>数据日</th><th>滞后天数</th><th>更新阈值</th><th>状态</th><th>说明</th></tr></thead><tbody>' + entries.slice(0, 500).map(e => { const lag = Number(e.stale_days), threshold = Number(e.threshold_days); const missing = !e.as_of; const expired = !!e.stale && Number.isFinite(lag) && Number.isFinite(threshold) && lag > threshold * 3; const stale = !!e.stale && !expired; const state = missing ? ['❌ 缺失', '#f85149'] : expired ? ['⛔ 过期，不可用于当前判断', '#f85149'] : stale ? ['⚠️ 陈旧，仅作历史参考', '#d29922'] : ['✅ 新鲜', '#3fb950']; const note = e.note || ''; return `<tr${(stale || expired || missing) ? ` style="background:${expired || missing ? 'rgba(248,81,73,.10)' : 'rgba(210,153,34,.12)'};"` : ''}><td title="${esc(note)}">${esc(e.file || '-')}</td><td>${esc(e.as_of || '-')}</td><td>${e.stale_days ?? '-'}</td><td>${e.threshold_days ?? '-'}</td><td style="color:${state[1]};">${state[0]}</td><td style="font-size:12px;color:#8b949e;">${esc(note)}</td></tr>`; }).join('') + '</tbody></table>' + (entries.length > 500 ? `<div class="data-meta">当前筛选命中 ${entries.length} 条，仅展示前 500 条，请继续筛选。</div>` : '')
    : '<div class="empty">当前筛选没有匹配的数据集</div>';
}
async function loadDataHealth() {
  const box = $('dhTable'), rep = $('dhReport'), meta = $('dhMeta');
  if (!box) return;
  try {
    const d = await api('/api/data_health', undefined, { cache_ttl: 0, silent: true });
    if (!d.ok) throw new Error(d.error || '数据健康失败');
    const data = d.data || {};
    dataHealthEntries = data.entries || [];
    const s = data.summary || {};
    if (meta) meta.textContent = `数据集 ${s.total || 0} · 新鲜 ${s.fresh || 0} · 异常 ${s.stale || 0} · 按风险优先展示`;
    renderDataHealthEntries();
    // markdown 报告（简单文本渲染）
    const report = data.report;
    rep.innerHTML = report
      ? (report.content
          ? `<div style="color:#8b949e;font-size:12px;margin-bottom:6px;">📄 ${esc(report.file)}</div><pre style="margin:0;white-space:pre-wrap;font-size:13px;line-height:1.7;">${esc(report.content)}</pre>`
          : `<div class="empty">${esc(report.error || '无内容')}</div>`)
      : '<div class="empty">暂无数据健康报告</div>';
  } catch (e) {
    box.innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
}

// ── 定时任务状态 ──
async function loadCronTasks() {
  const box = $('cronTable'), errBox = $('cronError'), meta = $('cronMeta');
  if (!box) return;
  try {
    const d = await api('/api/tasks_status', undefined, { cache_ttl: 0, silent: true });
    if (!d.ok) throw new Error(d.error || '任务状态失败');
    const data = d.data || {};
    const tasks = data.tasks || [];
    const failed = tasks.filter(t => t.last_status === 'error' || Number(t.consecutive_errors || 0) > 0).length;
    if (meta) meta.textContent = `enabled ${tasks.length} · 异常 ${failed} · 数据来自 OpenClaw 状态库`;
    box.innerHTML = tasks.length
      ? '<table class="risk-table"><thead><tr><th>名称</th><th>计划</th><th>最近运行</th><th>结果</th><th>连续错误</th><th>下次运行</th><th>失败原因</th></tr></thead><tbody>' +
        tasks.map(t => { const bad = t.last_status === 'error' || Number(t.consecutive_errors || 0) > 0; return `<tr${bad ? ' style="background:rgba(248,81,73,.08);"' : ''}><td><b>${esc(t.name)}</b></td><td style="font-family:var(--font-num);font-size:12px;">${esc(t.expr || t.kind || '-')}</td><td>${esc(t.last_run || '-')}</td><td style="color:${bad ? '#f85149' : '#3fb950'};">${esc(t.last_status || 'idle')}</td><td>${esc(t.consecutive_errors || 0)}</td><td>${esc(t.next_run || '-')}</td><td style="font-size:12px;color:#8b949e;max-width:320px;">${esc(t.last_error || '-')}</td></tr>`; }).join('') +
        '</tbody></table>'
      : '<div class="empty">没有读取到 enabled 定时任务；请查看网关状态库错误。</div>';
    if (errBox) errBox.innerHTML = data.error
      ? `<div class="empty" style="color:#f85149;">⚠️ ${esc(data.error)}</div>` : '';
  } catch (e) {
    box.innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
}

// ── 60s 定时刷新已并入 AUTO_REFRESH_MAP（见 startAutoRefresh 统一调度器）──
// 这里仅保留刷新按钮绑定（脚本在 body 末尾执行，元素已存在）
(function () {
  if ($('alertsRefreshBtn')) $('alertsRefreshBtn').addEventListener('click', loadAlertsStatus);
  if ($('dhRefreshBtn')) $('dhRefreshBtn').addEventListener('click', loadDataHealth);
  if ($('dhStaleOnly')) $('dhStaleOnly').addEventListener('change', renderDataHealthEntries);
  if ($('dhSearch')) $('dhSearch').addEventListener('input', renderDataHealthEntries);
  if ($('cronRefreshBtn')) $('cronRefreshBtn').addEventListener('click', loadCronTasks);
  if ($('chainRefreshBtn')) $('chainRefreshBtn').addEventListener('click', loadDecisionChain);
  if ($('chainMode')) $('chainMode').addEventListener('change', loadDecisionChain);
  if ($('ragBtn')) $('ragBtn').addEventListener('click', doRagSearch);
  if ($('ragQuery')) $('ragQuery').addEventListener('keydown', e => { if (e.key === 'Enter') doRagSearch(); });
  if ($('lensRefreshBtn')) $('lensRefreshBtn').addEventListener('click', loadStockLens);
  if ($('lensCode')) $('lensCode').addEventListener('keydown', e => { if (e.key === 'Enter') loadStockLens(); });
  if ($('lensWatchlistBtn')) $('lensWatchlistBtn').addEventListener('click', loadWatchlistLens);
  // 进入个股全景页自动加载默认代码
  if ($('lensCode') && $('lensContent') && !$('lensContent').dataset.loaded) {
    setTimeout(loadStockLens, 300);
  }
})();

// ════════════════════════════════════════════
// 集成层： 决策链 + RAG 知识库检索（🧭 决策链+RAG 页）
// ════════════════════════════════════════════

// 决策链渲染：mode=intraday(盘中) | after_close(盘后龙虎榜+低估池) | battle_map(作战地图)
// 2026-08-29 隔离：/api/decision_chain 已转 deprecated shim，返回统一工作台投影。
async function loadDecisionChain() {
  const box = $('chainContent'), errBox = $('chainError'), meta = $('chainMeta');
  if (!box) return;
  const mode = ($('chainMode') && $('chainMode').value) || 'intraday';
  try {
    const d = await api('/api/decision_chain?mode=' + encodeURIComponent(mode), undefined, { cache_ttl: 0, silent: true });
    if (!d.ok) throw new Error(d.error || '决策链失败');
    const dd = d.data || {};
    if (dd.content_source === 'unified_decision_snapshot' || d.deprecated) {
      // 旧入口已隔离：渲染统一工作台投影的摘要，并提示迁移到 /api/workbench。
      const c = dd.content || {};
      const mk = c.market || {};
      const lines = [];
      lines.push(`<div class="macro-card" style="margin-bottom:10px;"><div class="macro-header">🪧 旧决策入口已隔离 · 统一决策快照</div>`);
      lines.push(`<div style="font-size:12px;color:#8b949e;line-height:1.6;">本页原决策链已合并进统一决策工作台。状态：<b>${esc(c.state_label || c.state || '-')}</b> · 数据截至 <b>${esc(c.as_of || '-')}</b> · 情绪 ${esc(mk.emotion_stage || '-')} · 温度 ${esc(mk.temperature ?? '-')}</div>`);
      if ((c.mainlines || []).length) {
        lines.push(`<div style="margin-top:6px;font-size:13px;">主线：${(c.mainlines||[]).slice(0,6).map(m=>esc(m.name||m.label||m)).join(' · ')}</div>`);
      }
      lines.push(`<div style="margin-top:8px;font-size:12px;color:#d29922;">→ 完整决策快照已并入首屏「统一决策工作台」，同参数查看</div></div>`);
      box.innerHTML = lines.join('');
      if (meta) meta.textContent = `已隔离 · ${mode} · ${c.as_of || '-'}`;
      if (errBox) errBox.innerHTML = '';
      return;
    }
    const data = dd.content || {};
    const date = dd.date || '-';
    if (meta) meta.textContent = `📅 ${date} · ${mode}`;
    if (dd.error || data.error) { box.innerHTML = `<div class="empty">⚠️ ${esc(dd.error || data.error)}</div>`; return; }
    if (mode === 'intraday') {
      const b = data.bias || {};
      const toneIcon = { '防御': '🛡️', '谨慎': '⚖️', '进攻': '🚀', '休市': '🌙' }[b.tone] || '❓';
      const rows = data.rows || [];
      box.innerHTML =
        `<div class="macro-card" style="margin-bottom:10px;">
          <div style="font-size:16px;font-weight:700;">${toneIcon} 大盘基调: <span style="color:var(--accent)">${esc(b.tone || '?')}</span>
          <span style="color:#8b949e;font-size:12px;margin-left:8px;">上证 ${b.index_pct!=null ? (b.index_pct>0?'+':'')+b.index_pct+'%' : '-'} · 涨跌 ${b.advance ?? '-'}/${b.decline ?? '-'} · 广度 ${b.breadth ?? '-'}</span></div>
          ${b.summary ? `<div style="color:#8b949e;font-size:12px;margin-top:4px;">${esc(b.summary)}</div>` : ''}
        </div>` +
        (rows.length
          ? `<table class="risk-table"><thead><tr><th>代码</th><th>名称</th><th>现价</th><th>涨跌</th><th>主导视角</th><th>侧重结论</th></tr></thead><tbody>` +
            rows.map(r => `<tr>
              <td style="font-family:var(--font-num)">${esc(r.symbol)}</td>
              <td><b>${esc(r.name || '')}</b></td>
              <td style="font-family:var(--font-num)">${r.price ?? '-'}</td>
              <td class="${(r.change_pct||0)>=0 ? 'acell-buy' : 'acell-sell'}">${(r.change_pct||0)>0?'+':''}${r.change_pct ?? '-'}%</td>
              <td><span class="badge" style="background:rgba(88,166,255,0.15);color:#58a6ff;">${esc(r.dominant_label || '')}</span></td>
              <td style="font-size:12px;max-width:340px;">${esc(r.summary || '')}</td>
            </tr>`).join('') + '</tbody></table>'
          : `<div class="empty">${esc(data.skipped || '当前无显著机会')}</div>`);
    } else if (mode === 'after_close') {
      const lhb = data.lhb || {}, vp = data.value_picks || [];
      let html = '';
      if ((lhb.stock_top || []).length) {
        html += `<div class="macro-card" style="margin-bottom:10px;"><div class="macro-header">💰 龙虎榜 · 个股净买 Top (${esc(lhb.stock_top[0].day || '')})</div>` +
          lhb.stock_top.map(s => `<div style="padding:4px 0;font-size:13px;">💵 <b>${esc(s.name)}</b>(${esc(s.code)}) 净买 <span class="acell-buy" style="font-family:var(--font-num)">${s.net}亿</span> ${(s.pct||0)>0?'+':''}${s.pct}% <span style="color:#8b949e;font-size:12px;">${esc(s.reason || '')}</span>${s.fwd1!=null?` <span style="color:#d29922;font-size:11px;">→后1日 ${s.fwd1>0?'+':''}${s.fwd1}%</span>`:''}</div>`).join('') + '</div>';
      }
      if ((lhb.broker_top || []).length) {
        html += `<div class="macro-card" style="margin-bottom:10px;"><div class="macro-header">🏦 龙虎榜 · 游资活跃席位</div>` +
          lhb.broker_top.map(b => `<div style="padding:4px 0;font-size:13px;">🏦 <b>${esc(b.name)}</b> 买入 ${b.buy}亿 ${b.net>=0?`净买 <span class="acell-buy">${b.net}亿</span>`:`净卖 <span class="acell-sell">${b.net}亿</span>`} <span style="color:#8b949e;font-size:12px;">${esc(b.stocks || '')}</span></div>`).join('') + '</div>';
      }
      if (vp.length) {
        html += `<div class="macro-card"><div class="macro-header">💎 中长线低估池 (${vp.length}只)</div>` +
          vp.map(v => `<div style="padding:4px 0;font-size:13px;">💎 <b>${esc(v.name)}</b>(${esc(v.code)}) PE ${v.pe} · PB ${v.pb} ${v.roe!=null?`· ROE ${v.roe}%`:''} · 距52周低+${v.dist52w}% <span class="badge" style="background:rgba(210,153,34,0.15);color:#d29922;">打分 ${v.score}</span></div>`).join('') + '</div>';
      }
      box.innerHTML = html || '<div class="empty">盘后产物缺失（等待 18:50 盘后任务生成）</div>';
    } else {
      // battle_map: 原样 JSON 摘要
      box.innerHTML = `<div class="macro-card" style="margin-bottom:10px;"><div class="macro-header">🗺️ 作战地图 · ${esc(data.date || date)}</div>` +
        `<div style="font-size:12px;color:#8b949e;">主线/支线/板块梯队等结构化数据见 JSON:</div></div>` +
        `<pre style="white-space:pre-wrap;font-size:11px;color:#c9d1d9;max-height:520px;overflow:auto;">${esc(JSON.stringify(data, null, 1).slice(0, 4000))}</pre>`;
    }
    if (errBox) errBox.innerHTML = '';
  } catch (e) {
    box.innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
}

// RAG 知识库检索：返回技能/书籍方法论片段作为决策依据
async function doRagSearch() {
  const box = $('ragResults'), input = $('ragQuery');
  if (!box || !input) return;
  const q = input.value.trim();
  if (!q) { box.innerHTML = '<div class="empty">请输入检索关键词（如: 低吸/情绪周期/因子IC/仓位管理）</div>'; return; }
  box.innerHTML = '<div class="empty skeleton">📚 检索知识库 (230+ skills / 33 本书籍方法论)...</div>';
  try {
    const d = await api('/api/rag_search?q=' + encodeURIComponent(q), undefined, { cache_ttl: 60, silent: true });
    if (!d.ok) throw new Error(d.error || 'RAG 检索失败');
    const hits = d.hits || [];
    box.innerHTML = hits.length
      ? hits.map(h => `<div style="padding:8px 10px;border-bottom:1px solid #21262d;">
          <div style="font-size:12px;"><span class="badge" style="background:rgba(88,166,255,0.15);color:#58a6ff;">${esc(h.cat || 'skill')}</span>
          <span style="color:#8b949e;">${esc(h.file || '')}</span>
          <span style="color:#d29922;float:right;font-family:var(--font-num);">${h.score != null ? Number(h.score).toFixed(3) : ''}</span></div>
          <div style="font-size:12px;color:#c9d1d9;margin-top:4px;line-height:1.6;">${esc(h.text || '').slice(0, 260)}</div>
        </div>`).join('')
      : '<div class="empty">无命中，换关键词试试（如: 情绪周期/涨停复盘/低估价值）</div>';
  } catch (e) {
    box.innerHTML = `<div class="empty">检索失败: ${esc(e.message)}</div>`;
  }
}

// ════════════════════════════════════════════
// 集成层： 个股全景透视 (stock_lens) — 短线+中线双视角
// ════════════════════════════════════════════

function lensBar(label, score, max, color) {
  const pct = Math.min(100, Math.round((score / max) * 100));
  return `<div style="margin:6px 0;">
    <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;">
      <span>${label}</span><span style="font-family:var(--font-num);color:${color};font-weight:700;">${score}/${max}</span>
    </div>
    <div style="height:8px;background:#21262d;border-radius:4px;overflow:hidden;">
      <div style="height:100%;width:${pct}%;background:linear-gradient(90deg,${color}66,${color});border-radius:4px;transition:width .4s;"></div>
    </div>
  </div>`;
}

function lensPartCard(p) {
  const colors = { '涨停基因': '#f85149', '龙虎榜': '#d29922', '概念共振': '#58a6ff', '资金流': '#3fb950', '热度人气': '#ff7b72', '融资盘': '#bc8cff', '筹码(股东户数)': '#79c0ff', '基金持仓': '#56d364', '估值': '#e3b341', '产业链联动': '#39c5cf', '事件信号': '#f778ba', '宏观驱动': '#7ee787', '深度价值': '#e3b341', '质量成长': '#56d364', '市场水位': '#79c0ff', '事件底仓': '#f778ba' };
  const c = colors[p.key] || '#8b949e';
  const d = p.data || {};
  let extra = '';
  if (p.key === '龙虎榜' && d.times_60d) extra = `上榜${d.times_60d}次${d.fwd1_winrate!=null?`·胜率${(d.fwd1_winrate*100).toFixed(0)}%`:''}`;
  else if (p.key === '涨停基因' && d.zt_30d) extra = `${d.zt_30d}次${d.max_board>=2?`·${d.max_board}连板`:''}`;
  else if (p.key === '融资盘' && d.chg_40d != null) extra = `余额40日${d.chg_40d>0?'+':''}${d.chg_40d}%`;
  else if (p.key === '筹码(股东户数)' && d.change_pct != null) extra = `户数${d.change_pct>0?'+':''}${d.change_pct}%`;
  else if (p.key === '基金持仓' && d.fund_count != null) extra = `${d.fund_count}家${d.change==='增仓'?'·增仓':''}`;
  else if (p.key === '估值' && d.pe != null) extra = `PE${d.pe}/PB${d.pb??'-'}${d.roe!=null?`/ROE${d.roe}%`:''}`;
  else if (p.key === '概念共振' && d.zt_counts) extra = `概念内${d.zt_counts.zt_stocks??0}家涨停${d.stage?`·${d.stage}`:''}`;
  else if (p.key === '热度人气' && d.rank) extra = `第${d.rank}名${d.rank_change!=null?`(${d.rank_change>0?'↑':d.rank_change<0?'↓':'='}${Math.abs(d.rank_change)})`:''}`;
  else if (p.key === '产业链联动' && d.links) extra = `${d.links.length}个关联板块`;
  else if (p.key === '事件信号' && d.announce && d.announce.records) extra = `公告${d.announce.records}条${d.block&&d.block.institution_buyer?`·大宗机构${d.block.institution_buyer}笔`:''}`;
  else if (p.key === '宏观驱动' && d.pmi != null) extra = `PMI${d.pmi}${d.cpi_yoy!=null?`·CPI${d.cpi_yoy}%`:''}`;
  else if (p.key === '深度价值' && d.pe_pct != null) extra = `PE历史${(d.pe_pct*100).toFixed(0)}%分位`;
  else if (p.key === '质量成长' && d.roe_4q != null) extra = `ROE${d.roe_4q}%${d.net_margin!=null?`·净利率${d.net_margin}%`:''}`;
  else if (p.key === '市场水位' && d.qvix_pct != null) extra = `恐慌${(d.qvix_pct*100).toFixed(0)}%分位${d.below_net_ratio!=null?`·破净${(d.below_net_ratio*100).toFixed(1)}%`:''}`;
  else if (p.key === '事件底仓' && d.bullish) extra = `利好${d.bullish.length}条${d.bearish&&d.bearish.length?`·利空${d.bearish.length}条`:''}`;
  return `<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:8px 10px;">
    <div style="font-size:12px;color:#8b949e;">${p.key}</div>
    <div style="font-size:20px;font-weight:800;color:${c};font-family:var(--font-num);margin:2px 0;">${p.score}<span style="font-size:11px;color:#484f58;">/${p.max}</span></div>
    <div style="font-size:11px;color:#8b949e;min-height:14px;">${extra||'无数据'}</div>
  </div>`;
}

async function loadStockLens() {
  const box = $('lensContent'), meta = $('lensMeta');
  if (!box) return;
  const code = ($('lensCode') && $('lensCode').value.trim()) || sharedSymbol();
  box.innerHTML = '<div class="empty skeleton">🔭 加载个股全景透视...</div>';
  try {
    const d = await api('/api/stock_lens?code=' + encodeURIComponent(code), undefined, { cache_ttl: 60, silent: true });
    if (!d.ok) throw new Error(d.error || '全景分析失败');
    const r = d.data || {}, s = r.synth || {};
    if (meta) meta.textContent = `${r.code} ${r.name||''} · ${r.as_of||''}`;
    const toneColor = { '进攻': '#f85149', '进攻偏谨慎': '#d29922', '防守反击': '#3fb950', '观察': '#58a6ff', '回避': '#8b949e', '逢低布局': '#3fb950', '谨慎进攻': '#d29922', '左侧': '#3fb950' }[s.tone] || '#8b949e';
    const ss = s.short_score||0, ms = s.mid_score||0, ls = s.long_score||0;
    const shortParts = (r.short && r.short.parts) || [];
    const midParts = (r.mid && r.mid.parts) || [];
    const longParts = (r.long && r.long.parts) || [];
    const shortReasons = (r.short && r.short.reasons) || [];
    const midReasons = (r.mid && r.mid.reasons) || [];
    const longReasons = (r.long && r.long.reasons) || [];
    box.innerHTML = `
      <div class="macro-card" style="margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
          <div style="font-size:16px;font-weight:700;">${esc(r.name||r.code)} <span style="color:#657282;font-size:12px;">${esc(r.code)}</span></div>
          <span class="badge" style="background:${toneColor}22;color:${toneColor};font-size:13px;padding:4px 10px;border-radius:12px;">${esc(s.verdict||'')}</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:10px;">
          <div>${lensBar('📈 短线(游资/情绪)', ss, 100, '#f85149')}</div>
          <div>${lensBar('📊 中线(筹码/事件)', ms, 100, '#39c5cf')}</div>
          <div>${lensBar('🏛 长线(宏观/价值)', ls, 86, '#3fb950')}</div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin-bottom:10px;">
        ${[...shortParts, ...midParts, ...longParts].map(lensPartCard).join('')}
      </div>
      <div class="macro-card" style="margin-bottom:10px;">
        <div class="macro-header">📋 分析依据</div>
        <div style="font-size:12px;color:#c9d1d9;line-height:1.9;">
          ${shortReasons.map(r2=>`<span class="badge" style="background:rgba(248,81,73,0.12);color:#f85149;margin:2px;">${esc(r2)}</span>`).join('')}
          ${midReasons.map(r2=>`<span class="badge" style="background:rgba(57,197,207,0.12);color:#39c5cf;margin:2px;">${esc(r2)}</span>`).join('')}
          ${longReasons.map(r2=>`<span class="badge" style="background:rgba(63,185,80,0.12);color:#3fb950;margin:2px;">${esc(r2)}</span>`).join('')}
          ${(shortReasons.length+midReasons.length+longReasons.length)===0?'<span style="color:#657282;">无显著信号</span>':''}
        </div>
      </div>
      <div class="macro-card">
        <div class="macro-header">💡 决策建议</div>
        <div style="font-size:13px;color:#e6edf3;line-height:1.8;">${esc(s.advice||'')}</div>
      </div>`;
  } catch (e) {
    box.innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
}

// 自选+持仓 全池三视角评分表
async function loadWatchlistLens() {
  const box = $('lensContent'), meta = $('lensMeta');
  if (!box) return;
  box.innerHTML = '<div class="empty skeleton">📋 加载自选全景(短线/中线/长线三视角评分)...</div>';
  try {
    const d = await api('/api/stock_lens?mode=watchlist', undefined, { cache_ttl: 120, silent: true });
    if (!d.ok) throw new Error(d.error || '自选全景失败');
    const items = d.data || [];
    if (meta) meta.textContent = `自选+持仓 ${items.length} 只`;
    const toneColor = { '进攻': '#f85149', '进攻偏谨慎': '#d29922', '防守反击': '#3fb950', '观察': '#58a6ff', '回避': '#8b949e', '逢低布局': '#3fb950', '谨慎进攻': '#d29922' };
    box.innerHTML = `
      <div style="overflow-x:auto;">
      <table class="risk-table"><thead><tr>
        <th>代码</th><th>名称</th><th>短线</th><th>中线</th><th>长线</th><th>结论</th><th>强轴</th>
      </tr></thead><tbody>
      ${items.map(it => {
        const tc = toneColor[it.tone] || '#8b949e';
        const strongMap = {'短线':'📈','中线':'📊','长线':'🏛'};
        return `<tr onclick="document.getElementById('lensCode').value='${esc(it.code)}';loadStockLens();" style="cursor:pointer;">
          <td style="font-family:var(--font-num)">${esc(it.code)}</td>
          <td><b>${esc(it.name||'')}</b></td>
          <td style="font-family:var(--font-num);color:${it.short>=60?'#f85149':it.short>=40?'#d29922':'#8b949e'};font-weight:700;">${it.short}</td>
          <td style="font-family:var(--font-num);color:${it.mid>=60?'#39c5cf':it.mid>=40?'#58a6ff':'#8b949e'};font-weight:700;">${it.mid}</td>
          <td style="font-family:var(--font-num);color:${it.long>=60?'#3fb950':it.long>=40?'#7ee787':'#8b949e'};font-weight:700;">${it.long}</td>
          <td><span class="badge" style="background:${tc}22;color:${tc};">${esc(it.verdict||'')}</span></td>
          <td>${(it.strong||[]).map(s=>strongMap[s]||'').join(' ')}</td>
        </tr>`;
      }).join('')}
      </tbody></table>
      <div style="font-size:11px;color:#657282;margin-top:6px;">点击任意行 → 查看该股全景透视</div>
      </div>`;
  } catch (e) {
    box.innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
}

// 唯一启动入口：所有能力完成注册后再启动，避免重复初始化。
window.QUANT_TERMINAL_CORE.boot = async function bootQuantTerminal() {
  if (this.booted) return;
  this.booted = true;
  restoreNavigationState();
  if (window.INTEGRATED_VISUALS && window.INTEGRATED_VISUALS.init) window.INTEGRATED_VISUALS.init();
  startCoreHealth();
  startAutoRefresh();
  try {
    const dashboardActive = (readNavigationState().view || 'dashboard') === 'dashboard';
    // 首屏行情优先：配置与自选不应阻塞指数、宽度和盘后快照。
    const marketPromise = dashboardActive ? loadMarket() : Promise.resolve();
    await Promise.all([loadConfig(), loadWatchlist(), marketPromise]);
  } catch (error) {
    console.error('Quant Terminal boot failed:', error);
    setStatus(`工作台启动失败：${error.message}`);
  }
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => window.QUANT_TERMINAL_CORE.boot());
} else {
  window.QUANT_TERMINAL_CORE.boot();
}
