/* ============================================================
   viz9.js — 牧云天枢统一可视化图表库
   ------------------------------------------------------------
   纯函数图表渲染库：所有函数挂 window.Viz9 命名空间。
   统一签名：Viz9.renderXxx(elOrId, data) —— 只渲染，不请求数据。
   依赖：全局 echarts（echarts.min.js）
   配色：暗色金融终端主题（红涨 #f6465d / 绿跌 #2ebd85）
   ============================================================ */
(function () {
  'use strict';

  var CH = {
    text: '#c9d1d9',
    muted: '#8b949e',
    faint: '#6e7681',
    up: '#f6465d',
    down: '#2ebd85',
    flat: '#8b949e',
    accent: '#2563eb',
    grid: 'rgba(255,255,255,0.06)',
    split: 'rgba(255,255,255,0.12)',
    bg: 'transparent'
  };

  var _instances = {};

  /** 获取 echarts 实例（复用），并注册 ResizeObserver 自适应 */
  function _get(el) {
    var dom = typeof el === 'string' ? document.getElementById(el) : el;
    if (!dom) return null;
    if (typeof echarts === 'undefined') {
      dom.innerHTML = '<div class="empty">图表库未加载（echarts 缺失）</div>';
      return null;
    }
    var key = dom.id || ('el_' + Math.random().toString(36).slice(2, 8));
    dom.id = dom.id || key;
    var inst = _instances[key];
    if (!inst) {
      inst = echarts.init(dom);
      _instances[key] = inst;
      if (typeof ResizeObserver !== 'undefined') {
        var ro = new ResizeObserver(function () { inst.resize(); });
        ro.observe(dom);
        inst.__ro = ro;
      }
    }
    // Reuse the existing ECharts instance; clearing the container would remove
    // the canvas that the instance owns and break subsequent renders.
    return inst;
  }

  /** 防御：容器内显示中文提示 */
  function _empty(el, msg) {
    var dom = typeof el === 'string' ? document.getElementById(el) : el;
    if (dom) dom.innerHTML = '<div class="empty">' + (msg || '数据不足') + '</div>';
  }

  /** 数值数组判断是否全空 */
  function _hasData(arr) {
    if (!Array.isArray(arr) || !arr.length) return false;
    return arr.some(function (v) { return v !== null && v !== undefined && !isNaN(v); });
  }

  var Viz9 = {
    /* ═══════════════════════ 1. 因子 IC 热力图 ═══════════════════════
       data: { dates:[], factors:[], matrix:[[行=因子, 列=日期]] }
    */
    renderIcHeatmap: function (el, data) {
      var inst = _get(el);
      if (!inst) return;
      var dates = (data && data.dates) || [];
      var factors = (data && data.factors) || [];
      var matrix = (data && data.matrix) || [];
      if (!dates.length || !factors.length || !matrix.length) { _empty(el, '因子 IC 数据不足'); return; }

      var values = [];
      for (var i = 0; i < factors.length; i++) {
        for (var j = 0; j < dates.length; j++) {
          var v = matrix[i] ? matrix[i][j] : null;
          if (v === null || v === undefined || isNaN(v)) v = '-';
          values.push([j, i, v]);
        }
      }
      inst.setOption({
        title: { text: '因子 IC 热力图', left: 'center', textStyle: { color: CH.text, fontSize: 13 } },
        tooltip: {
          position: 'top',
          formatter: function (p) {
            return '<b>' + factors[p.value[1]] + '</b> @ ' + dates[p.value[0]] + '<br/>IC: <b>' + p.value[2] + '</b>';
          }
        },
        grid: { left: 90, right: 20, top: 50, bottom: 60 },
        xAxis: {
          type: 'category', data: dates,
          axisLabel: { color: CH.muted, fontSize: 10, rotate: 45 },
          splitArea: { show: true }
        },
        yAxis: {
          type: 'category', data: factors,
          axisLabel: { color: CH.muted, fontSize: 10 }
        },
        visualMap: {
          min: -0.3, max: 0.3, calculable: true, orient: 'horizontal', left: 'center', bottom: 0,
          inRange: { color: ['#1a5c3a', '#0d1117', '#5c1a2e'] },
          textStyle: { color: CH.muted }
        },
        series: [{
          type: 'heatmap', data: values,
          label: { show: false },
          emphasis: { itemStyle: { shadowBlur: 8, shadowColor: 'rgba(0,0,0,0.5)' } }
        }]
      });
    },

    /* ═══════════════════════ 2. 回测净值曲线 + 回撤 ═══════════════════════
       data: { dates:[], strategy:[], benchmark:[], drawdown:[] }
    */
    renderBacktestCurve: function (el, data) {
      var inst = _get(el);
      if (!inst) return;
      var dates = (data && data.dates) || [];
      if (!dates.length) { _empty(el, '回测净值数据不足'); return; }
      var strat = (data.strategy || []).map(function (v, i) { return [dates[i], v === null ? '-' : v]; });
      var bench = (data.benchmark || []).map(function (v, i) { return [dates[i], v === null ? '-' : v]; });
      var dd = (data.drawdown || []).map(function (v, i) { return [dates[i], v === null ? '-' : v]; });

      inst.setOption({
        title: { text: '回测净值与回撤', left: 'center', textStyle: { color: CH.text, fontSize: 13 } },
        tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
        legend: { top: 24, textStyle: { color: CH.muted }, data: ['策略', '基准', '回撤'] },
        grid: [{ left: 60, right: 20, top: 55, height: '55%' }, { left: 60, right: 20, top: '72%', height: '18%' }],
        xAxis: [
          { type: 'time', axisLabel: { color: CH.muted }, splitLine: { lineStyle: { color: CH.grid } } },
          { type: 'time', axisLabel: { color: CH.muted }, splitLine: { show: false } }
        ],
        yAxis: [
          { type: 'value', scale: true, axisLabel: { color: CH.muted, formatter: '{value}%' }, splitLine: { lineStyle: { color: CH.grid } } },
          { type: 'value', axisLabel: { color: CH.muted, formatter: '{value}%' }, splitLine: { show: false } }
        ],
        dataZoom: [{ type: 'inside', xAxisIndex: [0, 1] }, { type: 'slider', xAxisIndex: [0, 1], bottom: 0, height: 18 }],
        series: [
          { name: '策略', type: 'line', data: strat, showSymbol: false, lineStyle: { width: 2, color: CH.accent }, itemStyle: { color: CH.accent } },
          { name: '基准', type: 'line', data: bench, showSymbol: false, lineStyle: { width: 1.5, color: CH.flat, type: 'dashed' }, itemStyle: { color: CH.flat } },
          { name: '回撤', type: 'line', data: dd, showSymbol: false, xAxisIndex: 1, yAxisIndex: 1, lineStyle: { width: 1, color: CH.up }, areaStyle: { color: 'rgba(246,70,93,0.25)' } }
        ]
      });
    },

    /* ═══════════════════════ 3. 组合风险瀑布/暴露条形 ═══════════════════════
       data: { total: 数字, breakdown:[{name, value}], specific: 数字 }
       若 total 缺失则退化为横向条形图
    */
    renderRiskWaterfall: function (el, data) {
      var inst = _get(el);
      if (!inst) return;
      var bd = (data && data.breakdown) || [];
      if (!bd.length) { _empty(el, '风险分解数据不足'); return; }

      var names = bd.map(function (b) { return b.name; });
      var vals = bd.map(function (b) { return b.value; });

      if (data.total === undefined || data.total === null) {
        // 退化：横向条形图
        inst.setOption({
          title: { text: '因子风险暴露', left: 'center', textStyle: { color: CH.text, fontSize: 13 } },
          tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
          grid: { left: 90, right: 40, top: 40, bottom: 30 },
          xAxis: { type: 'value', axisLabel: { color: CH.muted }, splitLine: { lineStyle: { color: CH.grid } } },
          yAxis: { type: 'category', data: names, axisLabel: { color: CH.muted } },
          series: [{
            type: 'bar', data: vals,
            itemStyle: { color: function (p) { return p.value >= 0 ? CH.up : CH.down; }, borderRadius: [0, 3, 3, 0] },
            label: { show: true, position: 'right', color: CH.text, formatter: '{c}' }
          }]
        });
        return;
      }

      // 瀑布图：总风险 → 因子贡献(+) → 因子贡献(-) → 特异性 → 最终
      var total = Number(data.total) || 0;
      var specific = Number(data.specific) || 0;
      var items = [{ name: '总风险', value: total, itemStyle: { color: CH.accent } }];
      bd.forEach(function (b) {
        items.push({
          name: b.name,
          value: b.value,
          itemStyle: { color: b.value >= 0 ? CH.up : CH.down }
        });
      });
      var running = total;
      var calc = [running];
      var bars = [];
      var num = 0;
      bd.forEach(function (b) { num += b.value; bars.push(null); });
      items.forEach(function (it, idx) {
        if (idx === 0) { bars.push(it.value); return; }
        running -= it.value;
        bars.push(it.value);
        calc.push(running);
      });
      // 特异性
      bars.push(specific);
      calc.push(specific);
      items.push({ name: '特异性', value: specific, itemStyle: { color: CH.flat } });
      // 最终 = running - specific? 用剩余值
      var remain = running - specific;
      bars.push(remain);
      calc.push(remain);
      items.push({ name: '剩余风险', value: remain, itemStyle: { color: CH.muted } });

      inst.setOption({
        title: { text: '组合风险瀑布', left: 'center', textStyle: { color: CH.text, fontSize: 13 } },
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: function (ps) { return ps[0].name + ': <b>' + Number(ps[0].value).toFixed(4) + '</b>'; } },
        grid: { left: 60, right: 30, top: 45, bottom: 60 },
        xAxis: { type: 'category', data: items.map(function (i) { return i.name; }), axisLabel: { color: CH.muted, rotate: 30, fontSize: 10 } },
        yAxis: { type: 'value', axisLabel: { color: CH.muted }, splitLine: { lineStyle: { color: CH.grid } } },
        series: [{
          type: 'waterfall',
          data: bars,
          itemStyle: { color: function (p) { return p.data >= 0 ? CH.up : CH.down; } },
          label: { show: true, position: 'top', color: CH.text, fontSize: 10, formatter: function (p) { return Number(p.data).toFixed(3); } }
        }]
      });
    },

    /* ═══════════════════════ 4. 资产配置：权重饼图 + 行业条形 ═══════════════════════
       data: { weights:[{name,value}], industries:[{name,value}] } 或 {allocation:[{name,value}]}
       双图左右排布（两个容器需分别调用，el 传入数组 [饼图容器, 条形容器]）
    */
    renderAllocation: function (els, data) {
      var inst1 = _get(els[0]);
      var inst2 = _get(els[1]);
      if (!inst1 || !inst2) return;
      var w = (data && (data.weights || data.allocation)) || [];
      var ind = (data && data.industries) || [];
      if (!w.length && !ind.length) { _empty(els[0], '配置数据不足'); return; }

      var palette = ['#2563eb', '#8b5cf6', '#f6465d', '#2ebd85', '#e3b341', '#38bdf8', '#fb7185', '#34d399', '#a78bfa', '#f97316'];

      inst1.setOption({
        title: { text: '资产权重', left: 'center', textStyle: { color: CH.text, fontSize: 13 } },
        tooltip: { trigger: 'item', formatter: '{b}: {d}%' },
        legend: { bottom: 0, textStyle: { color: CH.muted, fontSize: 10 }, type: 'scroll' },
        series: [{
          type: 'pie', radius: ['38%', '68%'], center: ['50%', '48%'],
          itemStyle: { borderColor: '#0d1117', borderWidth: 2 },
          label: { color: CH.text, fontSize: 10, formatter: '{b} {d}%' },
          data: w.map(function (x, i) { return { name: x.name, value: x.value, itemStyle: { color: palette[i % palette.length] } }; })
        }]
      });

      inst2.setOption({
        title: { text: '行业分布', left: 'center', textStyle: { color: CH.text, fontSize: 13 } },
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: '{b}: {c}%' },
        grid: { left: 80, right: 30, top: 40, bottom: 30 },
        xAxis: { type: 'value', axisLabel: { color: CH.muted, formatter: '{value}%' }, splitLine: { lineStyle: { color: CH.grid } } },
        yAxis: { type: 'category', data: ind.map(function (x) { return x.name; }), axisLabel: { color: CH.muted, fontSize: 10 } },
        series: [{
          type: 'bar', data: ind.map(function (x) { return x.value; }),
          itemStyle: { color: CH.accent, borderRadius: [0, 3, 3, 0] },
          label: { show: true, position: 'right', color: CH.text, formatter: '{c}%' }
        }]
      });
    },

    /* ═══════════════════════ 5. 市场情绪仪表盘 gauge ═══════════════════════
       data: { value: 0-100, name: '情绪', zones 可选 }
    */
    renderSentimentGauge: function (el, data) {
      var inst = _get(el);
      if (!inst) return;
      var val = Number(data && data.value);
      if (isNaN(val)) { _empty(el, '情绪数据不足'); return; }
      var name = (data && data.name) || '市场情绪';
      inst.setOption({
        title: { text: name, left: 'center', textStyle: { color: CH.text, fontSize: 13 } },
        series: [{
          type: 'gauge', min: 0, max: 100, radius: '85%',
          progress: { show: true, width: 14, itemStyle: { color: val < 40 ? CH.down : (val < 60 ? CH.warn : CH.up) } },
          axisLine: { lineStyle: { width: 14, color: [[0.4, 'rgba(46,189,133,0.35)'], [0.6, 'rgba(210,153,34,0.35)'], [1, 'rgba(246,70,93,0.35)']] } },
          axisTick: { show: false },
          splitLine: { length: 8, lineStyle: { color: CH.split } },
          axisLabel: { color: CH.muted, fontSize: 10, distance: 18 },
          pointer: { itemStyle: { color: val < 40 ? CH.down : (val < 60 ? CH.warn : CH.up) } },
          anchor: { show: true, size: 8, itemStyle: { color: CH.text } },
          detail: {
            valueAnimation: true, formatter: '{value}',
            color: val < 40 ? CH.down : (val < 60 ? CH.warn : CH.up),
            fontSize: 28, offsetCenter: [0, '65%'], fontFamily: 'Consolas, monospace'
          },
          data: [{ value: Math.round(val * 10) / 10, name: name }],
          title: { color: CH.muted, fontSize: 11, offsetCenter: [0, '85%'] }
        }]
      });
    },

    /* ═══════════════════════ 6. 板块涨跌 Treemap ═══════════════════════
       data: [{name, value(成交额或显式面积权重), change(涨跌幅%)}]
    */
    renderSectorTreemap: function (el, data) {
      var inst = _get(el);
      if (!inst) return;
      var rows = (data || []).filter(function (r) { return r && r.name; });
      if (!rows.length) { _empty(el, '板块数据不足'); return; }
      var tree = rows.map(function (r) {
        var c = Number(r.change_pct != null ? r.change_pct : r.change) || 0;
        var amount = Number(r.amount != null ? r.amount : r.value) || 1;
        return {
          name: r.name + '\n' + (c >= 0 ? '+' : '') + c.toFixed(2) + '%',
          value: amount,
          itemStyle: { color: c >= 0 ? 'rgba(246,70,93,' + Math.min(0.85, 0.3 + Math.abs(c) / 20) + ')' : 'rgba(46,189,133,' + Math.min(0.85, 0.3 + Math.abs(c) / 20) + ')' }
        };
      });
      inst.setOption({
        title: { text: '板块涨跌热力图（面积=传入权重）', left: 'center', textStyle: { color: CH.text, fontSize: 13 } },
        tooltip: {
          formatter: function (info) {
            var s = info.data.name.split('\n');
            return '<b>' + s[0] + '</b><br/>涨跌: ' + s[1] + '<br/>面积权重: ' + (info.data.value || 0).toLocaleString();
          }
        },
        series: [{
          type: 'treemap', roam: false, nodeClick: false,
          breadcrumb: { show: false },
          label: { show: true, color: '#fff', fontSize: 11, formatter: '{b}' },
          itemStyle: { borderColor: '#0d1117', borderWidth: 2, gapWidth: 2 },
          data: tree
        }]
      });
    },

    /* ═══════════════════════ 7. 增强 K 线（MA/BOLL + 成交量副图） ═══════════════════════
       data: { dates:[], k:[[o,c,l,h]...], volume:[], ma:{m5,m10,m20}, boll:{up,mid,low} }
    */
    renderKlineEnhanced: function (el, data) {
      var inst = _get(el);
      if (!inst) return;
      var dates = (data && data.dates) || [];
      var k = (data && data.k) || [];
      if (!dates.length || !k.length) { _empty(el, 'K 线数据不足'); return; }

      var candles = k.map(function (row, i) { return [i, row[0], row[1], row[2], row[3]]; });
      var vols = (data.volume || []).map(function (v, i) {
        var o = k[i] ? k[i][0] : 0, c = k[i] ? k[i][1] : 0;
        return { value: v, itemStyle: { color: c >= o ? 'rgba(246,70,93,0.6)' : 'rgba(46,189,133,0.6)' } };
      });

      function maLine(name, arr, color, width) {
        return {
          name: name, type: 'line', data: (arr || []).map(function (v) { return v === null || v === undefined ? '-' : v; }),
          smooth: true, showSymbol: false, lineStyle: { width: width || 1, color: color }, itemStyle: { color: color }
        };
      }
      var series = [{
        name: 'K线', type: 'candlestick', data: candles,
        itemStyle: { color: CH.up, color0: CH.down, borderColor: CH.up, borderColor0: CH.down }
      }];
      if (data.ma) {
        if (data.ma.m5) series.push(maLine('MA5', data.ma.m5, '#e3b341'));
        if (data.ma.m10) series.push(maLine('MA10', data.ma.m10, '#38bdf8'));
        if (data.ma.m20) series.push(maLine('MA20', data.ma.m20, '#a78bfa'));
      }
      if (data.boll) {
        series.push(maLine('BOLL上', data.boll.up, 'rgba(139,148,158,0.8)', 0.8));
        series.push(maLine('BOLL中', data.boll.mid, 'rgba(139,148,158,0.6)', 0.8));
        series.push(maLine('BOLL下', data.boll.low, 'rgba(139,148,158,0.8)', 0.8));
      }
      series.push({
        name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: vols,
        barWidth: '60%'
      });

      var legendData = ['K线'];
      if (data.ma) { if (data.ma.m5) legendData.push('MA5'); if (data.ma.m10) legendData.push('MA10'); if (data.ma.m20) legendData.push('MA20'); }
      if (data.boll) { legendData.push('BOLL上', 'BOLL中', 'BOLL下'); }
      legendData.push('成交量');

      inst.setOption({
        animation: false,
        legend: { top: 4, textStyle: { color: CH.muted, fontSize: 10 }, data: legendData, type: 'scroll' },
        tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
        axisPointer: { link: [{ xAxisIndex: 'all' }], label: { backgroundColor: '#21262d' } },
        grid: [
          { left: 60, right: 20, top: 30, height: '58%' },
          { left: 60, right: 20, top: '74%', height: '16%' }
        ],
        xAxis: [
          { type: 'category', data: dates, axisLabel: { color: CH.muted, fontSize: 10 }, splitLine: { show: false } },
          { type: 'category', gridIndex: 1, data: dates, axisLabel: { show: false }, splitLine: { show: false } }
        ],
        yAxis: [
          { scale: true, axisLabel: { color: CH.muted, fontSize: 10 }, splitLine: { lineStyle: { color: CH.grid } } },
          { gridIndex: 1, axisLabel: { show: false }, splitLine: { show: false } }
        ],
        dataZoom: [
          { type: 'inside', xAxisIndex: [0, 1], start: 60, end: 100 },
          { type: 'slider', xAxisIndex: [0, 1], bottom: 0, height: 18, start: 60, end: 100 }
        ],
        series: series
      });
    },

    /* ═══════════════════════ 8. 多指标雷达图 ═══════════════════════
       data: { indicators:['a','b','c'], series:[{name, values:[...]}] }
    */
    renderRadarMulti: function (el, data) {
      var inst = _get(el);
      if (!inst) return;
      var inds = (data && data.indicators) || [];
      var series = (data && data.series) || [];
      if (!inds.length || !series.length) { _empty(el, '雷达数据不足'); return; }
      var palette = ['#2563eb', '#f6465d', '#2ebd85', '#e3b341', '#8b5cf6'];
      inst.setOption({
        title: { text: data.title || '多指标对比', left: 'center', textStyle: { color: CH.text, fontSize: 13 } },
        tooltip: {},
        legend: { bottom: 0, textStyle: { color: CH.muted, fontSize: 10 } },
        radar: {
          indicator: inds.map(function (i) { return { name: i, max: 100 }; }),
          radius: '62%',
          axisName: { color: CH.muted, fontSize: 10 },
          splitLine: { lineStyle: { color: CH.grid } },
          splitArea: { areaStyle: { color: ['rgba(37,99,235,0.03)', 'rgba(37,99,235,0.06)'] } },
          axisLine: { lineStyle: { color: CH.grid } }
        },
        series: [{
          type: 'radar',
          data: series.map(function (s, i) {
            return { name: s.name, value: s.values, areaStyle: { color: palette[i % palette.length], opacity: 0.15 }, lineStyle: { color: palette[i % palette.length] }, itemStyle: { color: palette[i % palette.length] } };
          })
        }]
      });
    },

    /* ═══════════════════════ 9. 因子相关性矩阵 ═══════════════════════
       data: { labels:[], matrix:[[...]] }
    */
    renderCorrMatrix: function (el, data) {
      var inst = _get(el);
      if (!inst) return;
      var labels = (data && data.labels) || [];
      var matrix = (data && data.matrix) || [];
      if (!labels.length || !matrix.length) { _empty(el, '相关性数据不足'); return; }
      var values = [];
      for (var i = 0; i < labels.length; i++) {
        for (var j = 0; j < labels.length; j++) {
          var v = matrix[i] ? matrix[i][j] : null;
          if (v === null || v === undefined || isNaN(v)) v = '-';
          values.push([j, i, v]);
        }
      }
      inst.setOption({
        title: { text: '因子相关性矩阵', left: 'center', textStyle: { color: CH.text, fontSize: 13 } },
        tooltip: {
          position: 'top',
          formatter: function (p) { return labels[p.value[1]] + ' × ' + labels[p.value[0]] + '<br/>ρ = <b>' + p.value[2] + '</b>'; }
        },
        grid: { left: 90, right: 30, top: 45, bottom: 70 },
        xAxis: { type: 'category', data: labels, axisLabel: { color: CH.muted, fontSize: 9, rotate: 45 } },
        yAxis: { type: 'category', data: labels, axisLabel: { color: CH.muted, fontSize: 9 } },
        visualMap: {
          min: -1, max: 1, calculable: true, orient: 'horizontal', left: 'center', bottom: 0,
          inRange: { color: ['#1a5c3a', '#0d1117', '#5c1a2e'] },
          textStyle: { color: CH.muted }
        },
        series: [{
          type: 'heatmap', data: values,
          label: { show: false },
          emphasis: { itemStyle: { shadowBlur: 8, shadowColor: 'rgba(0,0,0,0.5)' } }
        }]
      });
    },

    /* ═══════════════════════ 10. 小尺寸 gauge（温度计等单指标） ═══════════════════════
       data: { value, max, name, unit }
    */
    renderGaugeMini: function (el, data) {
      var inst = _get(el);
      if (!inst) return;
      var val = Number(data && data.value);
      var max = Number((data && data.max) || 100);
      if (isNaN(val)) { _empty(el, '数据不足'); return; }
      var name = (data && data.name) || '';
      var unit = (data && data.unit) || '';
      var ratio = val / max;
      var color = ratio < 0.4 ? CH.down : (ratio < 0.7 ? CH.warn : CH.up);
      inst.setOption({
        series: [{
          type: 'gauge', min: 0, max: max, radius: '90%',
          startAngle: 210, endAngle: -30,
          progress: { show: true, width: 10, itemStyle: { color: color } },
          axisLine: { lineStyle: { width: 10, color: [[1, 'rgba(255,255,255,0.08)']] } },
          axisTick: { show: false },
          splitLine: { show: false },
          axisLabel: { show: false },
          pointer: { show: false },
          anchor: { show: false },
          detail: {
            valueAnimation: true,
            formatter: function (v) { return v + (unit || ''); },
            color: color, fontSize: 20, offsetCenter: [0, '45%'], fontFamily: 'Consolas, monospace'
          },
          title: { color: CH.muted, fontSize: 11, offsetCenter: [0, '78%'] },
          data: [{ value: val, name: name }]
        }]
      });
    },

    /** 销毁某容器实例（可选） */
    dispose: function (el) {
      var dom = typeof el === 'string' ? document.getElementById(el) : el;
      if (!dom) return;
      var key = dom.id;
      var inst = _instances[key];
      if (inst) {
        if (inst.__ro) inst.__ro.disconnect();
        inst.dispose();
        delete _instances[key];
      }
    }
  };

  window.Viz9 = Viz9;
})();
