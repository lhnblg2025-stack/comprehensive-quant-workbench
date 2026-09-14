/* Data-health presentation follows /api/data_health, including unknown states. */
(function (root) {
  'use strict';
  function healthState(response) {
    const unavailable = (label, detail) => ({label, detail, className: 'bad', alerts: [detail]});
    if (!response || response.ok !== true) return unavailable('读取失败', response?.error || '健康接口未成功返回');
    const data = response.data;
    if (!data || data.freshness_error) return unavailable('格式异常', data?.freshness_error || '缺少健康数据');
    const summary = data.summary;
    const entries = data.entries;
    if (!summary || !Array.isArray(entries) || !['total', 'stale', 'fresh'].every(k => Number.isInteger(summary[k]) && summary[k] >= 0)
        || summary.total !== entries.length || summary.fresh + summary.stale !== summary.total
        || entries.some(e => !e || typeof e.stale !== 'boolean')
        || entries.filter(e => e.stale).length !== summary.stale) {
      return unavailable('格式异常', '健康统计与明细不一致');
    }
    if (summary.total === 0) return unavailable('暂无数据', '尚无数据新鲜度检查结果，无法判断健康');
    const unknown = entries.filter(e => !e.as_of || !e.checked_at || e.stale_days == null);
    if (unknown.length) return unavailable('状态未知', `${unknown.length} 项缺少数据日期或检查时间`);
    const day = value => {
      const s = String(value || '').slice(0, 10);
      const t = Date.parse(s + 'T00:00:00Z');
      return /^\d{4}-\d{2}-\d{2}$/.test(s) && Number.isFinite(t) && new Date(t).toISOString().slice(0,10) === s ? s : null;
    };
    const invalid = entries.filter(e => !day(e.as_of) || !day(e.checked_at)
      || typeof e.stale_days !== 'number' || !Number.isFinite(e.stale_days)
      || e.stale_days < 0 || day(e.as_of) > day(e.checked_at)
      || ['future','invalid','unknown','error','n/a'].includes(String(e.status).toLowerCase()));
    if (invalid.length) return unavailable('日期或状态异常', `${invalid.length} 项日期无效、晚于检查日期或状态不可用`);
    const alerts = entries.filter(e => e.stale).map(e => `${e.file || e.source || '数据集'}：已过期，数据日 ${e.as_of}`);
    return {
      label: summary.stale === 0 ? '正常' : summary.stale === summary.total ? '全部过期' : '部分过期',
      detail: `数据集 ${summary.total} · 新鲜 ${summary.fresh} · 过期 ${summary.stale}`,
      className: summary.stale ? 'warn' : 'ok', alerts,
    };
  }
  root.quantOpsHealthState = healthState;
  if (typeof module !== 'undefined') module.exports = healthState;
})(typeof window === 'undefined' ? globalThis : window);
