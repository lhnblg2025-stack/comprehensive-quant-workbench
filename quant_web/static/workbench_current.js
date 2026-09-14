/* Current workbench market overview: no northbound dependency. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const unwrap = x => x && x.data ? x.data : (x || {});
  const esc = x => String(x == null ? '-' : x).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  async function read(url) { const r = await fetch(url, {cache: 'no-store'}); if (!r.ok) throw Error(url + ' HTTP ' + r.status); return r.json(); }
  function hideNorth() {
    ['macroNorthBound','northBoundDisplay','northboundState','northboundMetrics','northboundTable','northboundHoldingsState','northboundHoldingsMetrics','northboundHoldingsTable'].forEach(id => { const n=$(id); if(n){ n.closest('.macro-panel,.northbound-columns,section')?.setAttribute('hidden','hidden'); n.setAttribute('hidden','hidden'); } });
  }
  function render(data) {
    const pano=unwrap(data.pano), temp=unwrap(data.temp), pulse=unwrap(data.pulse);
    const breadth=pano.breadth || {}; const indices=pano.indices || pano.index || {};
    const cards=[];
    const indexNames=['上证指数','深证成指','沪深300','创业板指','科创50'];
    indexNames.forEach(name=>{const v=indices[name]; if(v&&v.index!=null) cards.push([name,Number(v.index).toFixed(2),`${Number(v.change_pct??v.change??0)>=0?'▲':'▼'} ${Number(v.change_pct??v.change??0).toFixed(2)}%`]);});
    const up=Number(breadth['上涨']||temp.advance||0), down=Number(breadth['下跌']||temp.decline||0);
    if(up+down) cards.push(['市场宽度',`${up} / ${down}`,`上涨 / 下跌 · 数据日 ${pano.trade_date||pano.date||temp.date||'-'}`]);
    const t=temp.temperature ?? (breadth['上涨占比%']); if(t!=null) cards.push(['市场温度',`${Number(t).toFixed(1)}°`,`上涨 ${up} / 下跌 ${down}`]);
    const fg=pulse.index ?? pulse.fear_greed_index ?? pulse.fgi; if(fg!=null) cards.push(['恐贪指数',Number(fg).toFixed(1),pulse.level||'市场脉搏']);
    const grid=$('kpiGrid'); if(grid&&cards.length) grid.innerHTML=cards.map(c=>`<div class="kpi-card kpi-flat"><div class="kpi-head"><div class="kpi-label">${esc(c[0])}</div><div class="kpi-icon">📈</div></div><div class="kpi-value">${esc(c[1])}</div><div class="kpi-delta flat">— ${esc(c[2])}</div><div class="kpi-sub">最近可用交易日</div></div>`).join('');
    const status=$('systemStatus'); if(status) status.innerHTML=`<div class="data-state">行情总览正常 · 最近可用交易日：<b>${esc(pano.trade_date||pano.date||'-')}</b> · 不使用北向资金</div>`;
    hideNorth();
  }
  async function loadCurrentOverview(){
    try { const [pano,temp,pulse,legacy]=await Promise.all([read('/api/market_pano'),read('/api/market_temp'),read('/api/market_pulse'),read('/api/market')]); if(!pano.indices || (Array.isArray(pano.indices) && !pano.indices.length) || (!Array.isArray(pano.indices) && !Object.keys(pano.indices).length)) pano.indices=unwrap(legacy).indices||{}; if(!pano.breadth) pano.breadth=unwrap(legacy).breadth||{}; pano.trade_date=pano.trade_date||unwrap(legacy).trade_date; render({pano,temp,pulse}); }
    catch(e) { const s=$('systemStatus'); if(s) s.innerHTML=`<div class="data-state">市场总览读取失败：${esc(e.message)}</div>`; hideNorth(); }
  }
  // Compatibility exports only. This module no longer overrides the canonical
  // dashboard renderer in app.js or hides existing market data modules.
  window.refreshCurrentMarketOverview = loadCurrentOverview;
})();
