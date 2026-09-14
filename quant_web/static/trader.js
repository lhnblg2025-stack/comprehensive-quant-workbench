/* trader.js — 行情交易员（K线 / 热力图 / 资金流瀑布 / 交互式复盘卡）
   数据源: /api/v12/kline /api/v12/heatmap /api/v12/fund_flow /api/v12/stock_card
   无数据区块统一降级为「数据未覆盖」。 */
'use strict';
const $ = (s) => document.querySelector(s);
const UP = '#f6465d';
const DOWN = '#2ebd85';
const FLAT = '#8b949e';

async function api(url) {
  const r = await fetch(url);
  return r.json();
}

function fmt(v, nd = 2) {
  if (v === null || v === undefined || v === '') return '—';
  const f = Number(v);
  return Number.isFinite(f) ? f.toFixed(nd) : '—';
}

function fmtPct(v) {
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return '—';
  const f = Number(v);
  return (f > 0 ? '+' : '') + f.toFixed(2) + '%';
}

function noData(el, msg) {
  if (!el) return;
  el.innerHTML = '<div class="no-data">' + (msg || '数据未覆盖') + '</div>';
}

function reasonOf(d) {
  return d && d.reason ? d.reason : '';
}

/* ────────────────────────── K线 ────────────────────────── */
let klineChart = null;
let klineCurrent = '600519';
const klineEl = () => $('#klineChart');

function initKlineChart() {
  const el = $('#klineChart');
  if (!el || typeof echarts === 'undefined') return;
  klineChart = echarts.init(el);
  klineChart.on('click', (p) => {
    if (p && p.seriesType === 'candlestick' && klineCurrent) loadStockCard(klineCurrent);
  });
  window.addEventListener('resize', () => klineChart && klineChart.resize());
}

function renderKline(d) {
  const el = klineEl();
  if (!d || d.ok === false || !d.dates || !d.dates.length) {
    noData(el, '数据未覆盖：' + (reasonOf(d) || '无K线数据'));
    $('#klineTitle').textContent = '';
    $('#klineMeta').textContent = reasonOf(d) || '';
    return;
  }
  $('#klineTitle').textContent = d.name + ' · 截止 ' + d.as_of + ' · ATR14 ' + fmt(d.atr14);
  const chip = d.chip || {};
  let chipHtml = '';
  if (chip.poc !== null && chip.poc !== undefined) {
    chipHtml = '筹码 POC <b>' + fmt(chip.poc) + '</b> ｜ 密集区 <b>' + fmt(chip.lower) +
               ' ~ ' + fmt(chip.upper) + '</b>';
  } else {
    chipHtml = '筹码 <span class="flat-text">' + (chip.note || '样本不足') + '</span>';
  }
  $('#klineMeta').innerHTML = chipHtml + '<br>MA20/60/144 与筹码上下沿虚线已叠加 · 点击K线可联动复盘卡';

  const markLine = {
    silent: true, symbol: 'none', lineStyle: { type: 'dashed', width: 1 },
    label: { show: true, position: 'insideEndTop', fontSize: 10 },
    data: [],
  };
  if (chip.upper !== null && chip.upper !== undefined) {
    markLine.data.push({ yAxis: chip.upper, name: '筹码上沿', lineStyle: { color: UP }, label: { color: UP } });
    markLine.data.push({ yAxis: chip.lower, name: '筹码下沿', lineStyle: { color: DOWN }, label: { color: DOWN } });
  }
  const maSeries = [
    { name: 'MA20', data: d.ma20, color: '#f0b429' },
    { name: 'MA60', data: d.ma60, color: '#58a6ff' },
    { name: 'MA144', data: d.ma144, color: '#d2a8ff' },
  ].map((m) => ({
    name: m.name, type: 'line', data: m.data, smooth: true, showSymbol: false,
    lineStyle: { width: 1, color: m.color }, itemStyle: { color: m.color },
    symbol: 'none',
  }));
  klineChart.setOption({
    animation: false,
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'cross' },
      backgroundColor: '#161b22', borderColor: '#30363d',
      textStyle: { color: '#e6edf3', fontSize: 12 },
    },
    legend: { data: ['MA20', 'MA60', 'MA144'], top: 0, textStyle: { color: FLAT }, itemWidth: 14 },
    axisPointer: { link: [{ xAxisIndex: 'all' }] },
    grid: [
      { left: 62, right: 16, top: 26, height: '56%' },
      { left: 62, right: 16, top: '74%', height: '16%' },
    ],
    xAxis: [
      { type: 'category', data: d.dates, boundaryGap: true,
        axisLine: { lineStyle: { color: '#30363d' } }, axisLabel: { color: FLAT, hideOverlap: true, interval: Math.max(1, Math.floor(d.dates.length / 8)) },
        axisPointer: { label: { backgroundColor: '#30363d' } } },
      { type: 'category', gridIndex: 1, data: d.dates, boundaryGap: true,
        axisLine: { lineStyle: { color: '#30363d' } }, axisLabel: { show: false } },
    ],
    yAxis: [
      { scale: true, splitLine: { lineStyle: { color: '#21262d' } }, axisLabel: { color: FLAT } },
      { gridIndex: 1, scale: true, splitLine: { show: false }, axisLabel: { color: FLAT } },
    ],
    dataZoom: [
      { type: 'inside', xAxisIndex: [0, 1], start: 60, end: 100 },
      { type: 'slider', xAxisIndex: [0, 1], bottom: 4, height: 14,
        borderColor: '#30363d', textStyle: { color: FLAT } },
    ],
    series: [
      {
        name: 'K线', type: 'candlestick', data: d.ohlc,
        itemStyle: { color: UP, color0: DOWN, borderColor: UP, borderColor0: DOWN },
        markLine,
      },
      ...maSeries,
      {
        name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1,
        data: d.volumes.map((v, i) => {
          const c = d.ohlc[i] || [];
          return { value: v, itemStyle: { color: c[1] >= c[0] ? UP : DOWN, opacity: 0.75 } };
        }),
      },
    ],
  }, true);
}

/* ────────────────────────── 热力图 ────────────────────────── */
let heatmapChart = null;
let heatmapType = 'sw1';
let marketTreemap = null;

function initHeatmapChart() {
  const el = $('#heatmapChart');
  if (!el) return;
  if (typeof Plotly !== 'undefined') return;
  if (typeof echarts === 'undefined') return;
  heatmapChart = echarts.init(el);
  heatmapChart.on('click', (p) => {
    const item = p && p.data;
    if (!item || !item.name) return;
    if (item.leader_code) {
      loadKline(item.leader_code);
      loadStockCard(item.leader_code);
      $('#heatmapMeta').innerHTML = '已联动：<b>' + item.name + '</b> → 龙头 <b>' +
        (item.leader_name || item.leader_code) + '</b>';
    } else if (item.constituents && item.constituents.length) {
      const first = item.constituents[0];
      loadKline(first.code);
      loadStockCard(first.code);
      const chips = item.constituents.map((x) =>
        '<span class="chip ' + (first.code === x.code ? 'up' : 'flat') + '">' +
        x.name + ' ' + x.code + '</span>').join('');
      $('#heatmapMeta').innerHTML = '成分（前5大权重）：' + chips + '<br>已联动：<b>' +
        first.name + '</b>';
    }
  });
  window.addEventListener('resize', () => heatmapChart && heatmapChart.resize());
}

function colorScale(v, maxAbs) {
  if (v > 0) return 'rgba(246,70,93,' + (0.25 + 0.75 * Math.min(1, v / maxAbs)).toFixed(3) + ')';
  if (v < 0) return 'rgba(46,189,133,' + (0.25 + 0.75 * Math.min(1, -v / maxAbs)).toFixed(3) + ')';
  return 'rgba(139,148,158,0.4)';
}

function renderFundTreemap(d) {
  const el = $('#heatmapChart');
  if (!el || typeof Plotly === 'undefined') return false;
  if (!d || d.ok === false || !d.items || !d.items.length) {
    noData(el, (d && d.reason) || '资金流数据未覆盖');
    $('#heatmapMeta').textContent = (d && d.reason) || '';
    return true;
  }
  Plotly.newPlot(el, [{type:'treemap', ids:d.items.map(x=>x.id), labels:d.items.map(x=>x.name), parents:d.items.map(x=>x.parent), values:d.items.map(x=>x.value), customdata:d.items, branchvalues:'total', textinfo:'label', hovertemplate:'%{customdata.name}<br>净流入 %{customdata.net_yi:.3f} 亿<extra></extra>', marker:{colors:d.items.map(x=>x.net_yi), colorscale:[[0,'#2ebd85'],[0.5,'#252d38'],[1,'#f6465d']], cmid:0, colorbar:{title:'净流入(亿)'}}}], {paper_bgcolor:'transparent',plot_bgcolor:'transparent',margin:{l:0,r:0,t:8,b:0},font:{color:'#e6edf3'},uniformtext:{minsize:9,mode:'hide'}}, {responsive:true,displaylogo:false,modeBarButtonsToAdd:['toImage']});
  el.on('plotly_click', event => { const item=event.points&&event.points[0]&&event.points[0].customdata; if(item&&item.kind==='stock'){loadKline(item.code);loadStockCard(item.code);$('#heatmapMeta').textContent=`已联动 ${item.name}（${item.code}） · 净流入 ${item.net_yi} 亿`;}});
  $('#heatmapMeta').textContent=`资金日 ${d.as_of} · 行业 ${d.coverage.industry_count} 个 · 个股 ${d.coverage.stock_count} 只 · 面积=净额绝对值 · ${d.coverage.status}`;
  return true;
}

function renderMarketTreemap(d) {
  const el = $('#heatmapChart');
  if (!el || typeof Plotly === 'undefined') return false;
  if (!d || d.ok === false || !d.items || !d.items.length) {
    noData(el, (d && d.reason) || '全市场热力数据未覆盖');
    $('#heatmapMeta').textContent = (d && d.reason) || '';
    return true;
  }
  const labels=d.items.map((x) => x.name);
  Plotly.newPlot(el, [{type:'treemap', ids:d.items.map((x) => x.id), labels, parents:d.items.map((x) => x.parent), values:d.items.map((x) => x.value), customdata:d.items, branchvalues:'total', textinfo:'label', hovertemplate:'%{customdata.name}<br>涨跌 %{customdata.pct}%<extra></extra>', marker:{colors:d.items.map((x) => x.pct), colorscale:[[0,'#2ebd85'],[0.5,'#252d38'],[1,'#f6465d']], cmid:0, colorbar:{title:'涨跌%'}}}], {paper_bgcolor:'transparent',plot_bgcolor:'transparent',margin:{l:0,r:0,t:8,b:0},font:{color:'#e6edf3'},uniformtext:{minsize:9,mode:'hide'}}, {responsive:true,displaylogo:false,modeBarButtonsToAdd:['toImage']});
  el.on('plotly_click', (event) => { const item=event.points && event.points[0] && event.points[0].customdata; if(item && item.kind==='stock'){loadKline(item.id);loadStockCard(item.id);$('#heatmapMeta').textContent=`已联动 ${item.name}（${item.id}） · 涨跌 ${item.pct}%`;}});
  $('#heatmapMeta').textContent=`数据日 ${d.as_of} · 行业 ${d.coverage.industry_count} 个 · 个股 ${d.coverage.stock_count} 只 · 面积=市值；市值缺失时按成交额代理 · ${d.coverage.status || '覆盖状态待确认'}`;
  return true;
}

function renderHeatmap(d) {
  const el = $('#heatmapChart');
  if (!heatmapChart && typeof echarts !== 'undefined' && typeof Plotly === 'undefined') initHeatmapChart();
  if (!d || d.ok === false || !d.items || !d.items.length) {
    noData(el, '数据未覆盖：' + (reasonOf(d) || '无热力数据'));
    $('#heatmapMeta').textContent = reasonOf(d) || '';
    return;
  }
  const maxAbs = Math.max(0.01, ...d.items.map((x) => Math.abs(x.value)));
  const data = d.items.map((x) => ({
    name: x.name,
    value: Math.abs(x.value) + 0.01, // treemap 面积需正值；涨跌幅存 pct
    pct: x.value,
    up: x.up, down: x.down,
    leader_code: x.leader_code || '',
    leader_name: x.leader_name || '',
    constituents: x.constituents || [],
    itemStyle: { color: colorScale(x.value, maxAbs) },
  }));
  heatmapChart.setOption({
    animation: false,
    backgroundColor: 'transparent',
    tooltip: {
      backgroundColor: '#161b22', borderColor: '#30363d',
      textStyle: { color: '#e6edf3', fontSize: 12 },
      formatter: (p) => {
        const it = p.data || {};
        return it.name + '<br>涨跌幅 <b style="color:' +
          (it.pct >= 0 ? UP : DOWN) + '">' + fmtPct(it.pct) + '</b>';
      },
    },
    series: [{
      type: 'treemap', roam: false, nodeClick: false, breadcrumb: { show: false },
      label: {
        show: true, fontSize: 11, color: '#e6edf3', overflow: 'truncate',
        formatter: (p) => (p.data.name || '').slice(0, 10) + '\n' + fmtPct(p.data.pct),
      },
      upperLabel: { show: true, height: 20, color: '#e6edf3', fontSize: 11 },
      itemStyle: { borderColor: '#0d1117', borderWidth: 1, gapWidth: 1 },
      data,
    }],
  }, true);
  $('#heatmapMeta').textContent = '数据日期 ' + d.as_of + '（' + d.items.length +
    ' 个板块）｜点击板块联动 K线与复盘卡';
}

/* ────────────────────────── 瀑布图 ────────────────────────── */
let flowChart = null;

function initFlowChart() {
  const el = $('#flowChart');
  if (!el || typeof echarts === 'undefined') return;
  flowChart = echarts.init(el);
  window.addEventListener('resize', () => flowChart && flowChart.resize());
}

function renderFlow(d) {
  const el = $('#flowChart');
  if (!d || d.ok === false || !d.items || !d.items.length) {
    noData(el, '数据未覆盖：' + (reasonOf(d) || '无资金流数据'));
    return;
  }
  const top = d.items.slice(0, 15);
  flowChart.setOption({
    animation: false,
    backgroundColor: 'transparent',
    grid: { left: 96, right: 46, top: 26, bottom: 64 },
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'shadow' },
      backgroundColor: '#161b22', borderColor: '#30363d',
      textStyle: { color: '#e6edf3', fontSize: 12 },
      formatter: (ps) => {
        const p = ps[0];
        const v = p.value;
        return p.name + '<br>主力净流入 <b style="color:' +
          (v >= 0 ? UP : DOWN) + '">' + fmt(v, 2) + '</b> 亿';
      },
    },
    xAxis: {
      type: 'category', data: top.map((x) => x.name),
      axisLabel: { color: FLAT, fontSize: 10, rotate: 35 },
      axisLine: { lineStyle: { color: '#30363d' } },
    },
    yAxis: {
      type: 'value', name: '亿元', nameTextStyle: { color: FLAT },
      splitLine: { lineStyle: { color: '#21262d' } }, axisLabel: { color: FLAT },
    },
    series: [{
      type: 'bar', barMaxWidth: 18,
      data: top.map((x) => ({
        value: x.main_net_inflow,
        itemStyle: { color: x.main_net_inflow >= 0 ? UP : DOWN, borderRadius: [3, 3, 0, 0] },
      })),
      label: { show: true, position: 'top', fontSize: 9, color: FLAT,
               formatter: (p) => fmt(p.value, 1) },
    }],
  }, true);
}

/* ────────────────────────── 复盘卡 ────────────────────────── */
function renderStockCard(d) {
  const el = $('#stockCard');
  if (!d || d.ok === false || !d.card) {
    el.innerHTML = '<div class="no-data">数据未覆盖：' +
      (reasonOf(d) || (d && d.card && d.card.error) || '无复盘数据') + '</div>';
    $('#cardTitle').textContent = '';
    return;
  }
  const c = d.card;
  if (c.missing) {
    el.innerHTML = '<div class="no-data">数据未覆盖：' + (c.error || '无复盘数据') + '</div>';
    $('#cardTitle').textContent = d.name || '';
    return;
  }
  $('#cardTitle').textContent = d.name + ' · ' + String(c.date || d.date || '').slice(0, 10);
  const nh = c.new_highs || {};
  const vp = c.volume_profile || {};
  const ff = c.fund_flow || {};
  const rs = c.rs || {};
  const pctCls = (c.pct_chg || 0) >= 0 ? 'up-text' : 'down-text';
  let ffTxt;
  if (ff.main_net === null || ff.main_net === undefined) {
    ffTxt = '<span class="flat-text">n/a（' + (ff.note || '接口失败') + '）</span>';
  } else {
    ffTxt = '<b>' + fmt(ff.main_net / 1e8, 2) + '</b> 亿（' + (ff.signal || '') + '）';
  }
  const nhFmt = (v) => (v === null || v === undefined ? '—' : v ? '是' : '否');
  const vpTxt = (vp.poc !== null && vp.poc !== undefined)
    ? fmt(vp.poc) + '（区间 ' + fmt(vp.lower) + ' ~ ' + fmt(vp.upper) + '）'
    : '<span class="flat-text">' + (vp.note || '样本不足') + '</span>';
  const rows = [
    ['收盘 / 涨跌幅', fmt(c.close) + '（<span class="' + pctCls + '">' + fmtPct(c.pct_chg) + '</span>）'],
    ['量比（量/20日均量）', fmt(c.volume_ratio)],
    ['影线标记', c.wicks && c.wicks.length ? c.wicks.join('、') : '—'],
    ['RS 评级', rs.rating || '—'],
    ['RS / RS_MA20', fmt(rs.rs, 3) + ' / ' + fmt(rs.rs_ma20, 3)],
    ['RS 斜率（%/日）', fmt(rs.rs_slope, 3)],
    ['ATR14 止损位', fmt(c.stop_loss) + '（支撑 ' + fmt(c.support) + ' / 阻力 ' + fmt(c.resistance) + '）'],
    ['筹码 POC', vpTxt],
    ['主力资金流', ffTxt],
    ['60/120/250日新高', nhFmt(nh[60]) + ' / ' + nhFmt(nh[120]) + ' / ' + nhFmt(nh[250])],
    ['操作提示', c.hint || '—'],
  ];
  el.innerHTML = '<table class="stock-table">' +
    rows.map((r) => '<tr><th>' + r[0] + '</th><td>' + r[1] + '</td></tr>').join('') +
    '</table>';
}

/* ────────────────────────── 加载器 ────────────────────────── */
async function loadKline(symbol) {
  klineCurrent = symbol;
  try {
    const d = await api('/api/v12/kline?symbol=' + encodeURIComponent(symbol) +
                        '&days=120&adjust=qfq');
    renderKline(d);
  } catch (e) {
    noData(klineEl(), '数据未覆盖：K线请求失败');
  }
}

async function loadHeatmap(type) {
  heatmapType = type;
  if (type === 'market' || type === 'fund') {
    try {
      const d = await api(type === 'fund' ? '/api/v12/fund_treemap' : '/api/v12/market_treemap');
      type === 'fund' ? renderFundTreemap(d) : renderMarketTreemap(d);
    } catch (e) {
      noData($('#heatmapChart'), '数据未覆盖：热力图请求失败');
    }
    return;
  }
  try {
    const d = await api('/api/v12/heatmap?type=' + encodeURIComponent(type));
    renderHeatmap(d);
  } catch (e) {
    noData($('#heatmapChart'), '数据未覆盖：热力图请求失败');
  }
}

async function loadFlow(type) {
  try {
    const d = await api('/api/v12/fund_flow?type=' + encodeURIComponent(type));
    renderFlow(d);
  } catch (e) {
    noData($('#flowChart'), '数据未覆盖：资金流请求失败');
  }
}

async function loadStockCard(symbol) {
  try {
    const d = await api('/api/v12/stock_card?symbol=' + encodeURIComponent(symbol));
    renderStockCard(d);
  } catch (e) {
    $('#stockCard').innerHTML = '<div class="no-data">数据未覆盖：复盘卡请求失败</div>';
  }
}

/* ────────────────────────── 交互绑定 ────────────────────────── */
let suggestTimer = null;

function wireSearch() {
  const input = $('#klineSearch');
  const sug = $('#suggest');
  if (!input || !sug) return;
  input.addEventListener('input', () => {
    clearTimeout(suggestTimer);
    const q = input.value.trim();
    if (!q) { sug.style.display = 'none'; return; }
    suggestTimer = setTimeout(async () => {
      try {
        const d = await api('/api/search_stock?q=' + encodeURIComponent(q));
        const list = (d.results || []).slice(0, 8);
        if (!list.length) { sug.style.display = 'none'; return; }
        sug.innerHTML = list.map((x) =>
          '<div data-code="' + x.code + '" data-name="' + x.name + '">' +
          x.name + ' · ' + x.code + '</div>').join('');
        sug.style.display = 'block';
        sug.querySelectorAll('div').forEach((el) => el.addEventListener('click', () => {
          sug.style.display = 'none';
          input.value = el.dataset.name;
          loadKline(el.dataset.code);
          loadStockCard(el.dataset.code);
        }));
      } catch (e) {
        sug.style.display = 'none';
      }
    }, 200);
  });
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.kline-search')) sug.style.display = 'none';
  });
}

function goSearch() {
  const q = $('#klineSearch').value.trim();
  if (/^\d{6}$/.test(q)) { loadKline(q); loadStockCard(q); }
}

function wireTabs() {
  document.querySelectorAll('.tabs button[data-hm]').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('.tabs button[data-hm]').forEach((x) => x.classList.remove('active'));
    b.classList.add('active');
    loadHeatmap(b.dataset.hm);
  }));
  document.querySelectorAll('.tabs button[data-ff]').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('.tabs button[data-ff]').forEach((x) => x.classList.remove('active'));
    b.classList.add('active');
    loadFlow(b.dataset.ff);
  }));
  document.querySelectorAll('.idx-shortcuts button').forEach((b) => b.addEventListener('click', () => {
    loadKline(b.dataset.symbol);
  }));
  $('#klineGo').addEventListener('click', goSearch);
  $('#klineSearch').addEventListener('keydown', (e) => { if (e.key === 'Enter') goSearch(); });
}

document.addEventListener('DOMContentLoaded', () => {
  initKlineChart();
  initHeatmapChart();
  initFlowChart();
  wireSearch();
  wireTabs();
  loadKline('600519');
  loadHeatmap('market');
  loadFlow('concept');
  loadStockCard('600519');
});
