/**
 * ════════════════════════════════════════════════════════════════
 * ui9.js — 牧云天枢 quant_web 前端 P2 交互组件库
 * ════════════════════════════════════════════════════════════════
 *
 * 全部组件挂载到 window.UI9 命名空间，纯原生 JS（无 ES module、无 jQuery），
 * 不依赖 app.js 内部状态。仅在使用个别 window 全局函数时做 typeof 防御检查。
 *
 * 组件清单：
 *   UI9.DataTable         通用表格（排序 / 过滤 / 分页 / CSV 导出）
 *   UI9.toast             轻提示通知（右上角滑入，2.5s 自动消失）
 *   UI9.confirmDialog     确认弹窗（模态，返回 Promise<boolean>）
 *   UI9.GlobalSearch      全局股票搜索（绑定 #globalSearchInput / #globalSearchDropdown）
 *   UI9.Watchlist         自选股（localStorage 持久化 + 服务端同步）
 *   UI9.sseClient         SSE 增强封装（指数退避重连 / 事件订阅 / 页面隐藏暂停）
 *   UI9.format            格式化工具（pct / num / cn / color）
 *
 * 用法示例见文件末尾注释块。
 * ════════════════════════════════════════════════════════════════
 */
(function (global) {
  'use strict';

  /* ───────────────────────── 内部小工具 ───────────────────────── */

  // HTML 转义（文本与属性通用：& < > "）
  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"]/g, function (ch) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch];
    });
  }

  // 判断是否为函数
  function isFn(f) { return typeof f === 'function'; }

  // 判断字符串是否为纯数字（用于排序时自动识别数字列）
  function isNumericStr(s) {
    return /^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(String(s).trim());
  }

  // 数值比较：两个数字按数值比，否则按字符串比（中文 locale）
  function cmpValues(a, b) {
    if (a == null && b == null) return 0;
    if (a == null) return 1;   // 空值排最后
    if (b == null) return -1;
    var an = typeof a === 'number' || isNumericStr(a);
    var bn = typeof b === 'number' || isNumericStr(b);
    if (an && bn) return parseFloat(a) - parseFloat(b);
    return String(a).localeCompare(String(b), 'zh-Hans-CN');
  }

  // 等待 body 存在后执行（脚本可能在 head 中提前执行）
  function ensureBody(fn) {
    if (!isFn(fn)) return;
    if (document.body) { fn(); return; }
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', fn, { once: true });
    } else {
      setTimeout(fn, 50);
    }
  }

  // 注入组件自带的基础样式（仅一次；DataTable 的 .table-toolbar/.tableWrap 等
  // 按约定由主题 CSS 提供，不在此注入）
  var STYLE_ID = 'ui9-default-style';
  function injectDefaultStyles() {
    if (document.getElementById(STYLE_ID)) return;
    var css = [
      /* ── toast 轻提示 ── */
      '.ui9-toast-wrap{position:fixed;top:16px;right:16px;z-index:10001;display:flex;flex-direction:column;gap:8px;pointer-events:none;}',
      '.ui9-toast{color:#fff;font-size:13px;line-height:1.5;padding:10px 14px;border-radius:8px;box-shadow:0 4px 14px rgba(0,0,0,.25);min-width:200px;max-width:340px;transform:translateX(120%);opacity:0;transition:transform .28s ease,opacity .28s ease;pointer-events:auto;}',
      '.ui9-toast.ui9-toast-in{transform:translateX(0);opacity:1;}',
      '.ui9-toast.ui9-toast-out{transform:translateX(120%);opacity:0;}',
      '.ui9-toast-success{background:#087443;}',
      '.ui9-toast-error{background:#b42318;}',
      '.ui9-toast-info{background:#1f2937;}',
      /* ── 确认弹窗 ── */
      '.ui9-dialog-mask{position:fixed;inset:0;background:rgba(15,23,42,.45);z-index:9999;display:flex;align-items:center;justify-content:center;}',
      '.ui9-dialog{background:#fff;border-radius:10px;padding:20px 22px;min-width:300px;max-width:420px;box-shadow:0 10px 40px rgba(0,0,0,.25);}',
      '.ui9-dialog-title{font-size:15px;font-weight:700;color:#e6edf3;margin-bottom:10px;}',
      '.ui9-dialog-text{font-size:13px;color:#4b5563;margin-bottom:18px;line-height:1.6;white-space:pre-wrap;word-break:break-word;}',
      '.ui9-dialog-btns{display:flex;justify-content:flex-end;gap:10px;}',
      '.ui9-dialog-btns button{border:none;border-radius:6px;padding:8px 18px;font-size:13px;cursor:pointer;}',
      '.ui9-btn-ok{background:#2563eb;color:#fff;}',
      '.ui9-btn-ok:hover{background:#1d4ed8;}',
      '.ui9-btn-cancel{background:#e5e7eb;color:#374151;}',
      '.ui9-btn-cancel:hover{background:#d1d5db;}',
      /* ── 全局搜索下拉项 ── */
      '.ui9-search-item{display:flex;justify-content:space-between;gap:16px;padding:8px 12px;cursor:pointer;font-size:13px;border-bottom:1px solid #30363d;}',
      '.ui9-search-item:hover,.ui9-search-item.active{background:#21262d;}',
      '.ui9-search-name{font-weight:500;color:#e6edf3;}',
      '.ui9-search-code{color:#657282;font-family:monospace;font-size:12px;}',
      '.ui9-search-empty,.ui9-search-error{padding:10px 12px;font-size:13px;color:#657282;}',
      /* ── 自选股按钮状态 ── */
      '.ui9-watched{color:#e8c547;}',
      '.ui9-unwatched{color:#9ca3af;}',
    ].join('');
    var style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = css;
    (document.head || document.documentElement).appendChild(style);
  }

  /* ═══════════════════════ 1. 格式化工具 ═══════════════════════ */

  var format = {
    /**
     * 百分比：12.34%
     * 传入数字或可解析字符串均可；空值返回 '-'
     */
    pct: function (v) {
      if (v == null || v === '') return '-';
      if (typeof v === 'number') return v.toFixed(2) + '%';
      var n = parseFloat(v);
      return isNaN(n) ? String(v) : n.toFixed(2) + '%';
    },

    /**
     * 数字：千分位 + 两位小数，如 12,345.68；空值返回 '-'
     */
    num: function (v) {
      if (v == null || v === '') return '-';
      var n = typeof v === 'number' ? v : parseFloat(v);
      if (isNaN(n)) return String(v);
      return n.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    },

    /**
     * 中文金额：>=1亿 → x.xx亿；>=1万 → x.xx万；否则原样两位小数；空值返回 '-'
     */
    cn: function (v) {
      if (v == null || v === '') return '-';
      var n = typeof v === 'number' ? v : parseFloat(v);
      if (isNaN(n)) return String(v);
      var abs = Math.abs(n);
      if (abs >= 1e8) return (n / 1e8).toFixed(2) + '亿';
      if (abs >= 1e4) return (n / 1e4).toFixed(2) + '万';
      return n.toFixed(2);
    },

    /**
     * 涨跌色 class 名：>0 → 'rt-up'（红涨），<0 → 'rt-down'（绿跌），=0 → 'rt-flat'
     * 与主题 styles.css 中已有的 .rt-up / .rt-down / .rt-flat 对齐
     */
    color: function (v) {
      var n = parseFloat(v);
      if (isNaN(n)) return '';
      return n > 0 ? 'rt-up' : n < 0 ? 'rt-down' : 'rt-flat';
    },
  };

  /* ═══════════════════════ 2. toast 轻提示 ═══════════════════════ */

  var TOAST_DURATION = 2500; // 显示时长
  var TOAST_MAX = 5;         // 同屏最多条数

  function toast(msg, type) {
    msg = msg == null ? '' : String(msg);
    if (!msg) return;
    if (['success', 'error', 'info'].indexOf(type) === -1) type = 'info';
    ensureBody(function () {
      // 容器懒创建
      var wrap = document.getElementById('ui9-toast-wrap');
      if (!wrap) {
        wrap = document.createElement('div');
        wrap.id = 'ui9-toast-wrap';
        wrap.className = 'ui9-toast-wrap';
        document.body.appendChild(wrap);
      }
      // 超出上限时移除最旧的一条
      while (wrap.children.length >= TOAST_MAX) {
        var first = wrap.firstElementChild;
        if (first && first.parentNode) first.parentNode.removeChild(first);
      }
      var el = document.createElement('div');
      el.className = 'ui9-toast ui9-toast-' + type;
      el.textContent = msg;
      wrap.appendChild(el);
      // 滑入动画
      requestAnimationFrame(function () {
        el.classList.add('ui9-toast-in');
      });
      // 自动消失
      setTimeout(function () {
        el.classList.remove('ui9-toast-in');
        el.classList.add('ui9-toast-out');
        setTimeout(function () {
          if (el.parentNode) el.parentNode.removeChild(el);
        }, 300);
      }, TOAST_DURATION);
    });
  }

  /* ═══════════════════════ 3. confirmDialog 确认弹窗 ═══════════════════════ */

  var _dialogRef = null; // 当前打开的弹窗引用 { el, close }（防止叠加/监听器泄漏）

  /**
   * 确认弹窗：返回 Promise<boolean>；确定 → true，取消 / Esc / 点击遮罩 → false。
   * 可选 onOk(boolean) 回调在关闭后触发（兼容旧式用法）。
   */
  function confirmDialog(title, text, onOk) {
    return new Promise(function (resolve) {
      ensureBody(function () {
        // 关闭上一个未关闭的弹窗（清理监听器并 resolve 其 Promise）
        if (_dialogRef) { try { _dialogRef.close(false); } catch (e) { /* 忽略 */ } _dialogRef = null; }
        var mask = document.createElement('div');
        mask.className = 'ui9-dialog-mask';
        var dialog = document.createElement('div');
        dialog.className = 'ui9-dialog';
        var titleEl = document.createElement('div');
        titleEl.className = 'ui9-dialog-title';
        titleEl.textContent = title == null ? '提示' : String(title);
        var textEl = document.createElement('div');
        textEl.className = 'ui9-dialog-text';
        textEl.textContent = text == null ? '' : String(text);
        var btns = document.createElement('div');
        btns.className = 'ui9-dialog-btns';
        var btnCancel = document.createElement('button');
        btnCancel.type = 'button';
        btnCancel.className = 'ui9-btn-cancel';
        btnCancel.textContent = '取消';
        var btnOk = document.createElement('button');
        btnOk.type = 'button';
        btnOk.className = 'ui9-btn-ok';
        btnOk.textContent = '确定';
        btns.appendChild(btnCancel);
        btns.appendChild(btnOk);
        dialog.appendChild(titleEl);
        dialog.appendChild(textEl);
        dialog.appendChild(btns);
        mask.appendChild(dialog);
        document.body.appendChild(mask);
        _dialogMask = mask;

        var done = false;
        function close(result) {
          if (done) return;
          done = true;
          document.removeEventListener('keydown', onKey);
          if (mask.parentNode) mask.parentNode.removeChild(mask);
          if (_dialogRef && _dialogRef.el === mask) _dialogRef = null;
          resolve(result);
          if (isFn(onOk)) { try { onOk(result); } catch (e) { console.warn('[UI9] confirmDialog onOk 异常', e); } }
        }
        // Esc 取消 / Enter 确定
        function onKey(e) {
          if (e.key === 'Escape') close(false);
          else if (e.key === 'Enter') close(true);
        }
        document.addEventListener('keydown', onKey);
        // 点击遮罩（非弹窗本体）取消
        mask.addEventListener('click', function (e) {
          if (e.target === mask) close(false);
        });
        btnOk.addEventListener('click', function () { close(true); });
        btnCancel.addEventListener('click', function () { close(false); });
        // 登记当前弹窗（供后续弹窗清理）
        _dialogRef = { el: mask, close: close };
        // 自动聚焦确定按钮
        try { btnOk.focus(); } catch (e) { /* 忽略 */ }
      });
    });
  }

  /* ═══════════════════════ 4. DataTable 通用表格 ═══════════════════════ */

  /**
   * 用法：
   *   new UI9.DataTable(containerEl, {
   *     columns: [
   *       { key: 'symbol', title: '代码', sortable: true },
   *       { key: 'name',   title: '名称' },
   *       { key: 'change_pct', title: '涨跌幅', render: (v) => `<span class="${UI9.format.color(v)}">${UI9.format.pct(v)}</span>` },
   *     ],
   *     data: [{ symbol: '002714', name: '牧原股份', change_pct: 3.21 }],
   *     pageSize: 10,        // 默认 10
   *     filterable: true,    // 默认 true
   *     exportable: true,    // 默认 true
   *   });
   *   table.setData(newRows);   // 更新数据
   *   table.destroy();          // 销毁
   *
   * 约定：
   *   - 样式使用主题 CSS 提供的 class（.table-toolbar / .tableWrap / .ui9-table 等）
   *   - render(v, row) 返回的字符串视为可信 HTML，不做转义（请自行保证安全）；
   *     未提供 render 的列，原始值会自动转义后输出
   *   - CSV 导出的是当前过滤后数据的原始值（非 render 结果）
   *   - 点击新列默认降序排列，再次点击同列切换升降序
   */
  function DataTable(container, opts) {
    if (!container) return;
    if (typeof container === 'string') container = document.getElementById(container);
    if (!container || container._ui9Table) return; // 元素不存在或已实例化，防御式跳过

    opts = opts || {};
    this.container = container;
    this.columns = (opts.columns || []).map(function (c) {
      return typeof c === 'string' ? { key: c, title: c } : (c || {});
    });
    this.rawData = Array.isArray(opts.data) ? opts.data : [];
    this.pageSize = opts.pageSize || 10;
    if (this.pageSize < 1) this.pageSize = 10;
    this.filterable = opts.filterable !== false;
    this.exportable = opts.exportable !== false;
    this.filterText = '';
    this.sortKey = null;
    this.sortAsc = true;
    this.page = 1;

    var self = this;
    // 容器级事件委托：排序 / 分页 / 导出（避免重复绑定）
    this._onClick = function (e) {
      var th = e.target.closest ? e.target.closest('th[data-key]') : null;
      if (th) { self.sortBy(th.getAttribute('data-key')); return; }
      var btn = e.target.closest ? e.target.closest('[data-act]') : null;
      if (!btn || btn.disabled) return;
      var act = btn.getAttribute('data-act');
      if (act === 'export') { self.exportCsv(); }
      else if (act === 'prev') { self.page = Math.max(1, self.page - 1); self.render(); }
      else if (act === 'next') { self.page = Math.min(self.totalPages(), self.page + 1); self.render(); }
      else if (act === 'goto') { self.page = parseInt(btn.getAttribute('data-page'), 10) || 1; self.render(); }
    };
    container.addEventListener('click', this._onClick);
    container._ui9Table = this;
    this.render();
  }

  DataTable.prototype = {
    constructor: DataTable,

    /** 更新数据（保留当前过滤 / 排序 / 页码，自动收敛越界页码） */
    setData: function (data) {
      this.rawData = Array.isArray(data) ? data : [];
      this.render();
    },

    /** 更新列定义 */
    setColumns: function (cols) {
      this.columns = (cols || []).map(function (c) {
        return typeof c === 'string' ? { key: c, title: c } : (c || {});
      });
      // 排序列若已被移除则重置
      if (this.sortKey && !this.columns.some(function (c) { return c.key === this.sortKey; }, this)) {
        this.sortKey = null;
      }
      this.render();
    },

    /** 当前过滤后的数据（原始行对象） */
    getFiltered: function () {
      var kw = this.filterText.trim().toLowerCase();
      var filtered = kw
        ? this.rawData.filter(function (r) { return this._matchRow(r, kw); }, this)
        : this.rawData.slice();
      if (this.sortKey) {
        var key = this.sortKey, asc = this.sortAsc;
        filtered.sort(function (a, b) { return cmpValues(a[key], b[key]) * (asc ? 1 : -1); });
      }
      return filtered;
    },

    /** 总页数 */
    totalPages: function () {
      return Math.max(1, Math.ceil(this.getFiltered().length / this.pageSize));
    },

    /** 关键字匹配某行：任一列原始值包含关键字即命中 */
    _matchRow: function (row, kw) {
      return this.columns.some(function (c) {
        var v = row[c.key];
        if (v == null) return false;
        return String(v).toLowerCase().indexOf(kw) !== -1;
      });
    },

    /** 点击表头排序 */
    sortBy: function (key) {
      if (this.sortKey === key) {
        this.sortAsc = !this.sortAsc;   // 同列切换升降序
      } else {
        this.sortKey = key;
        this.sortAsc = false;           // 新列默认降序（与旧版 renderTable 一致）
      }
      this.page = 1;
      this.render();
    },

    /** 一次性 innerHTML 拼接渲染（不做逐行 DOM 操作） */
    render: function () {
      if (!this.container) return;
      var self = this;
      var filtered = this.getFiltered();
      var total = filtered.length;
      var totalPages = Math.max(1, Math.ceil(total / this.pageSize));
      if (this.page > totalPages) this.page = totalPages;
      var start = (this.page - 1) * this.pageSize;
      var pageRows = filtered.slice(start, start + this.pageSize);

      var html = '';

      // ── 工具栏：过滤框 + 导出按钮 ──
      if (this.filterable || this.exportable) {
        html += '<div class="table-toolbar">';
        if (this.filterable) {
          html += '<input type="text" class="table-filter-input" placeholder="输入关键字过滤..." value="' + esc(this.filterText) + '">';
        }
        if (this.exportable) {
          html += '<button type="button" class="table-export-btn" data-act="export">导出 CSV</button>';
        }
        html += '</div>';
      }

      // ── 表头 ──
      html += '<div class="tableWrap"><table class="ui9-table"><thead><tr>';
      html += this.columns.map(function (c) {
        var sortable = c.sortable !== false;
        var arrow = self.sortKey === c.key ? (self.sortAsc ? ' ▲' : ' ▼') : '';
        var th = '<th';
        if (sortable) th += ' class="sortable" data-key="' + esc(c.key) + '"';
        th += '>' + esc(c.title || c.key) + arrow + '</th>';
        return th;
      }).join('');
      html += '</tr></thead><tbody>';

      // ── 表体 ──
      if (!pageRows.length) {
        html += '<tr class="ui9-empty"><td colspan="' + this.columns.length + '">暂无数据</td></tr>';
      } else {
        html += pageRows.map(function (r) {
          return '<tr>' + self.columns.map(function (c) {
            var v = r[c.key];
            if (isFn(c.render)) {
              try { return '<td>' + c.render(v, r) + '</td>'; }
              catch (e) { console.warn('[UI9] 列 render 异常:', c.key, e); return '<td>' + esc(v) + '</td>'; }
            }
            return '<td>' + esc(v) + '</td>';
          }).join('') + '</tr>';
        }).join('');
      }
      html += '</tbody></table></div>';

      // ── 分页条 ──
      html += '<div class="ui9-pagination">';
      html += '<button type="button" class="ui9-page-btn" data-act="prev"' + (this.page <= 1 ? ' disabled' : '') + '>上一页</button>';
      var pages = this._pageWindow(totalPages);
      for (var i = 0; i < pages.length; i++) {
        var p = pages[i];
        if (p === '…') {
          html += '<span class="ui9-page-ellipsis">…</span>';
        } else {
          html += '<button type="button" class="ui9-page-btn' + (p === this.page ? ' active' : '') + '" data-act="goto" data-page="' + p + '">' + p + '</button>';
        }
      }
      html += '<button type="button" class="ui9-page-btn" data-act="next"' + (this.page >= totalPages ? ' disabled' : '') + '>下一页</button>';
      html += '<span class="ui9-page-info">共 ' + total + ' 条 · 第 ' + this.page + '/' + totalPages + ' 页</span>';
      html += '</div>';

      this.container.innerHTML = html;

      // ── 过滤框是每次重建的新元素，需重新绑定（并恢复焦点与光标位置） ──
      var input = this.container.querySelector('.table-filter-input');
      if (input) {
        input.addEventListener('input', function () {
          var hadFocus = document.activeElement === input;
          var pos = input.selectionStart;
          self.filterText = input.value;
          self.page = 1;
          self.render();
          if (hadFocus) {
            var ni = self.container.querySelector('.table-filter-input');
            if (ni) {
              ni.focus();
              try { ni.setSelectionRange(pos, pos); } catch (e) { /* 忽略 */ }
            }
          }
        });
      }
    },

    /** 分页页码窗口（最多 7 个，超长用省略号） */
    _pageWindow: function (totalPages) {
      var p = this.page, out = [];
      if (totalPages <= 7) {
        for (var i = 1; i <= totalPages; i++) out.push(i);
        return out;
      }
      out.push(1);
      if (p > 3) out.push('…');
      for (var j = Math.max(2, p - 1); j <= Math.min(totalPages - 1, p + 1); j++) out.push(j);
      if (p < totalPages - 2) out.push('…');
      out.push(totalPages);
      return out;
    },

    /** CSV 导出当前过滤后数据（原始值，含 BOM，Excel 可直接打开） */
    exportCsv: function () {
      var rows = this.getFiltered();
      if (!rows.length) { toast('没有数据可导出', 'info'); return; }
      var cols = this.columns.filter(function (c) { return c.key; });
      function quote(s) {
        return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
      }
      var header = cols.map(function (c) { return quote(c.title || c.key); }).join(',');
      var body = rows.map(function (r) {
        return cols.map(function (c) {
          var v = r[c.key];
          return quote(v == null ? '' : String(v));
        }).join(',');
      }).join('\n');
      var blob = new Blob(['\ufeff' + header + '\n' + body], { type: 'text/csv;charset=utf-8' });
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      var d = new Date(), pad = function (n) { return n < 10 ? '0' + n : String(n); };
      a.download = 'export_' + d.getFullYear() + pad(d.getMonth() + 1) + pad(d.getDate()) + '_' + pad(d.getHours()) + pad(d.getMinutes()) + '.csv';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(function () { URL.revokeObjectURL(a.href); }, 100);
      toast('CSV 已导出', 'success');
    },

    /** 销毁：解绑事件并清空容器 */
    destroy: function () {
      if (this.container && this._onClick) {
        this.container.removeEventListener('click', this._onClick);
        delete this.container._ui9Table;
      }
      if (this.container) this.container.innerHTML = '';
    },
  };

  /* ═══════════════════════ 5. GlobalSearch 全局股票搜索 ═══════════════════════ */

  /**
   * 绑定 #globalSearchInput / #globalSearchDropdown（并行任务可能尚未加入 DOM，
   * 元素不存在时静默跳过，并会在后续短暂重试）。
   * 交互：输入 300ms 防抖 → GET /api/search_stock?q= → 下拉展示 代码+名称；
   * 点击结果 → switchView('stock')（typeof 检查）+ 填充所有含 symbol/code 的输入框
   * + 触发 loadChart（存在时）；Esc / 点击外部关闭；↑↓ 移动、Enter 选中。
   */
  var GlobalSearch = {
    _bound: false,
    input: null,
    drop: null,
    timer: null,        // 防抖计时器
    activeIndex: -1,    // 键盘高亮索引
    results: [],        // 当前结果 [{name, code}]

    /** 初始化绑定；元素缺失返回 null（不抛异常） */
    init: function () {
      if (this._bound) return this;
      var input = document.getElementById('globalSearchInput');
      var drop = document.getElementById('globalSearchDropdown');
      if (!input || !drop) return null;
      this._bound = true;
      this.input = input;
      this.drop = drop;
      this.drop.className = (this.drop.className ? this.drop.className + ' ' : '') + 'ui9-search-dropdown';
      var self = this;

      input.addEventListener('input', function () { self._onInput(); });
      input.addEventListener('keydown', function (e) { self._onKey(e); });
      input.addEventListener('focus', function () {
        if (self.results.length) self.show();
      });
      // 点击外部关闭
      document.addEventListener('click', function (e) {
        if (self.drop && self.drop.contains(e.target)) return;
        if (e.target !== self.input) self.hide();
      });
      // 下拉内点击选中（mousedown 阻止默认，避免输入框失焦）
      drop.addEventListener('mousedown', function (e) { e.preventDefault(); });
      drop.addEventListener('click', function (e) {
        var item = e.target.closest ? e.target.closest('[data-code]') : null;
        if (item) self.pick(item.getAttribute('data-code'), item.getAttribute('data-name'));
      });
      return this;
    },

    /** 自动初始化（含短时重试，兼容并行任务稍后注入 DOM 的情况） */
    autoInit: function () {
      if (this.init()) return;
      var tries = 0;
      var t = setInterval(function () {
        tries++;
        if (GlobalSearch._bound || GlobalSearch.init()) { clearInterval(t); }
        else if (tries >= 8) { clearInterval(t); } // 约 6s 后放弃
      }, 750);
    },

    _onInput: function () {
      clearTimeout(this.timer);
      var q = (this.input.value || '').trim();
      if (!q) { this.results = []; this.hide(); return; }
      var self = this;
      this.timer = setTimeout(function () { self._search(q); }, 300);
    },

    /** 调 /api/search_stock 搜索 */
    _search: function (q) {
      var self = this;
      window.quantApiFetch('/api/search_stock?q=' + encodeURIComponent(q))
        .then(function (r) { return r.json(); })
        .then(function (data) {
          // 响应过期（输入已变化）则丢弃
          if ((self.input.value || '').trim() !== q) return;
          self.results = (data && data.ok && Array.isArray(data.results)) ? data.results : [];
          self.render();
        })
        .catch(function (err) {
          console.warn('[UI9] 搜索失败:', err);
          if ((self.input.value || '').trim() !== q) return;
          self.results = [];
          self.drop.innerHTML = '<div class="ui9-search-error">搜索失败，请稍后重试</div>';
          self.show();
        });
    },

    /** 渲染下拉（一次性 innerHTML） */
    render: function () {
      if (!this.drop) return;
      var self = this;
      if (!this.results.length) {
        this.drop.innerHTML = '<div class="ui9-search-empty">无结果</div>';
        this.activeIndex = -1;
        this.show();
        return;
      }
      this.activeIndex = 0;
      this.drop.innerHTML = this.results.map(function (r, i) {
        return '<div class="ui9-search-item' + (i === self.activeIndex ? ' active' : '') + '" data-code="' + esc(r.code) + '" data-name="' + esc(r.name) + '">'
          + '<span class="ui9-search-name">' + esc(r.name) + '</span>'
          + '<span class="ui9-search-code">' + esc(r.code) + '</span>'
          + '</div>';
      }).join('');
      this.show();
    },

    /** 显示下拉（fixed 定位在输入框正下方） */
    show: function () {
      if (!this.drop || !this.input) return;
      var rect = this.input.getBoundingClientRect();
      this.drop.style.position = 'fixed';
      this.drop.style.top = (rect.bottom + 4) + 'px';
      this.drop.style.left = rect.left + 'px';
      this.drop.style.width = Math.max(rect.width, 220) + 'px';
      this.drop.style.display = 'block';
      this.drop.style.zIndex = '9998';
      this.drop.style.background = '#161b22';
      this.drop.style.border = '1px solid #3b4552';
      this.drop.style.borderRadius = '6px';
      this.drop.style.boxShadow = '0 8px 24px rgba(0,0,0,.38)';
      this.drop.style.maxHeight = '320px';
      this.drop.style.overflowY = 'auto';
    },

    /** 隐藏下拉 */
    hide: function () {
      if (this.drop) this.drop.style.display = 'none';
      this.activeIndex = -1;
    },

    /** 键盘：Esc 关闭 / ↑↓ 移动 / Enter 选中 */
    _onKey: function (e) {
      if (e.key === 'Escape') { this.hide(); this.input.blur(); return; }
      if (!this.results.length) return;
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        this.activeIndex = Math.min(this.results.length - 1, this.activeIndex + 1);
        this._highlight();
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        this.activeIndex = Math.max(0, this.activeIndex - 1);
        this._highlight();
      } else if (e.key === 'Enter') {
        var r = this.results[this.activeIndex >= 0 ? this.activeIndex : 0];
        if (r) { e.preventDefault(); this.pick(r.code, r.name); }
      }
    },

    _highlight: function () {
      if (!this.drop) return;
      var items = this.drop.querySelectorAll('.ui9-search-item');
      items.forEach(function (el, i) {
        el.classList.toggle('active', i === GlobalSearch.activeIndex);
      });
      // 高亮项滚入可视区
      var cur = items[this.activeIndex];
      if (cur && cur.scrollIntoView) { try { cur.scrollIntoView({ block: 'nearest' }); } catch (e) { /* 忽略 */ } }
    },

    /**
     * 选中一只股票：切到个股视图 + 填充所有股票输入框 + 触发 loadChart
     */
    pick: function (code, name) {
      code = String(code || '').trim();
      if (!code) return;
      this.hide();
      this.input.value = code;

      // 1) 切到个股视图（若 app.js 已暴露 switchView）
      if (isFn(window.switchView)) {
        try { window.switchView('stock'); } catch (e) { console.warn('[UI9] switchView 异常', e); }
      }
      // 2) 填充页面中所有 id 含 symbol / code 的输入框（大小写不敏感）
      var filled = false;
      var inputs = document.querySelectorAll('input[id*="symbol" i], input[id*="code" i]');
      inputs.forEach(function (inp) {
        if (inp === GlobalSearch.input) return; // 跳过搜索框自身
        inp.value = code;
        filled = true;
      });
      // 触发 sSymbol 的 input 事件，让 app.js 的输入联动逻辑（lastKlineSymbol 等）同步
      var sSymbol = document.getElementById('sSymbol');
      if (sSymbol && sSymbol.value === code) {
        try { sSymbol.dispatchEvent(new Event('input', { bubbles: true })); } catch (e) { /* 忽略 */ }
      }
      // 3) 加载 K 线（若 app.js 已暴露 loadChart）
      if (isFn(window.loadChart)) {
        try { window.loadChart(); } catch (e) { console.warn('[UI9] loadChart 异常', e); }
      }
      toast('已定位 ' + (name || code), 'info');
      if (this.input) this.input.blur();
    },
  };

  /* ═══════════════════════ 6. Watchlist 自选股 ═══════════════════════ */

  var WATCHLIST_KEY = 'quant_watchlist_v9';

  /**
   * localStorage 持久化（本地优先），并尽力同步到服务端 /api/watchlist（失败静默）。
   * 数据项为 { symbol, name, ts }；list() 返回副本数组。
   */
  var Watchlist = {
    _cache: null,

    _load: function () {
      if (this._cache) return this._cache;
      var arr = [];
      try {
        var raw = localStorage.getItem(WATCHLIST_KEY);
        arr = raw ? JSON.parse(raw) : [];
      } catch (e) { arr = []; }
      if (!Array.isArray(arr)) arr = [];
      // 归一化为对象项
      this._cache = arr.map(function (it) {
        return typeof it === 'string' ? { symbol: it, name: '', ts: 0 } : it;
      }).filter(function (it) { return it && it.symbol; });
      return this._cache;
    },

    _persist: function () {
      try { localStorage.setItem(WATCHLIST_KEY, JSON.stringify(this._cache)); } catch (e) { /* 存储不可用则静默 */ }
    },

    _sync: function () {
      // 尽力同步到服务端，失败静默忽略（本地优先）
      try {
        window.quantApiFetch('/api/watchlist', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbols: this._cache.map(function (it) { return it.symbol; }) }),
        }).then(function (r) { return r.json(); }).catch(function () { /* 忽略 */ });
      } catch (e) { /* 忽略 */ }
    },

    /** 初始化：本地为空时尝试从服务端拉取种子数据 */
    _seedFromServer: function () {
      if (!this._cache || this._cache.length) return;
      var self = this;
      try {
        window.quantApiFetch('/api/watchlist')
          .then(function (r) { return r.json(); })
          .then(function (d) {
            if (self._cache.length) return; // 期间本地已有数据则放弃
            if (d && d.ok && Array.isArray(d.symbols) && d.symbols.length) {
              self._cache = d.symbols.map(function (s) {
                return typeof s === 'string' ? { symbol: s, name: '', ts: 0 } : s;
              });
              self._persist();
            }
          })
          .catch(function () { /* 忽略 */ });
      } catch (e) { /* 忽略 */ }
    },

    /** 全部自选（对象数组 [{symbol,name,ts}]） */
    list: function () {
      return this._load().slice();
    },

    /** 全部自选代码（字符串数组） */
    listSymbols: function () {
      return this._load().map(function (it) { return it.symbol; });
    },

    /** 添加自选；已存在则仅更新名称。返回 true 表示本次新增 */
    add: function (symbol, name) {
      symbol = String(symbol == null ? '' : symbol).trim();
      if (!symbol) return false;
      var list = this._load();
      var hit = null;
      for (var i = 0; i < list.length; i++) {
        if (list[i].symbol === symbol) { hit = list[i]; break; }
      }
      if (hit) {
        if (name && hit.name !== name) { hit.name = String(name); this._persist(); this._sync(); }
        return false;
      }
      list.push({ symbol: symbol, name: name ? String(name) : '', ts: Date.now() });
      this._persist();
      this._sync();
      return true;
    },

    /** 移除自选。返回 true 表示确实删除了 */
    remove: function (symbol) {
      symbol = String(symbol == null ? '' : symbol).trim();
      var list = this._load();
      var before = list.length;
      this._cache = list.filter(function (it) { return it.symbol !== symbol; });
      if (this._cache.length !== before) {
        this._persist();
        this._sync();
        return true;
      }
      return false;
    },

    /** 是否已自选 */
    isWatched: function (symbol) {
      symbol = String(symbol == null ? '' : symbol).trim();
      return this._load().some(function (it) { return it.symbol === symbol; });
    },

    /** 切换自选状态，返回切换后是否已自选 */
    toggle: function (symbol, name) {
      if (this.isWatched(symbol)) { this.remove(symbol); return false; }
      this.add(symbol, name);
      return true;
    },

    /**
     * 按钮辅助：点击切换自选状态并 toast 提示；返回按钮元素。
     * 状态通过读取 Watchlist 实时判断，多按钮间保持同步。
     */
    toggleButton: function (btn, symbol, name) {
      if (!btn) return btn;
      var self = this;
      btn.addEventListener('click', function (e) {
        if (e && e.preventDefault) e.preventDefault();
        if (e && e.stopPropagation) e.stopPropagation();
        var now = self.toggle(symbol, name);
        self.syncButton(btn, symbol, name);
        toast(now ? '已添加 ' + (name || symbol) + ' 到自选' : '已从自选移除 ' + (name || symbol), now ? 'success' : 'info');
      });
      this.syncButton(btn, symbol, name);
      return btn;
    },

    /** 刷新单个按钮的外观（☆ 加自选 / ★ 已自选） */
    syncButton: function (btn, symbol) {
      if (!btn) return;
      var watched = this.isWatched(symbol);
      btn.textContent = watched ? '★ 已自选' : '☆ 加自选';
      btn.classList.toggle('ui9-watched', watched);
      btn.classList.toggle('ui9-unwatched', !watched);
      btn.setAttribute('aria-pressed', watched ? 'true' : 'false');
      btn.dataset.watched = watched ? '1' : '0';
    },
  };

  /* ═══════════════════════ 7. sseClient SSE 增强封装 ═══════════════════════ */

  var SSE_URL = '/api/events';

  function sseSessionKey() {
    try {
      if (typeof global.quantApiKey === 'function') return global.quantApiKey() || '';
      return global.sessionStorage.getItem('quant_web_api_key') || '';
    } catch (e) { return ''; }
  }

  /**
   * 基于现有 /api/events 的增强封装：
   *   - 自动重连，指数退避 1s/2s/4s/8s/16s，封顶 30s；连接成功后退避重置
   *   - on(event, cb) 订阅；事件名取数据内 type（如 macro_overview），兼容命名事件
   *   - 收到数据事件默认 toast 提示：事件名含 alert/warn → error 样式，其余 info
   *     （heartbeat / connected / disconnected 不打扰用户，可通过 toastSkip 调整）
   *   - 页面隐藏时暂停（断开），重新可见时自动恢复
   * 注意：与 static/sse-client.js 的 window.sseClient 互不干扰，这里是独立实现。
   */
  var sseClient = {
    _es: null,
    _abort: null,
    _connected: false,
    _listeners: {},
    _retry: 0,          // 当前退避指数（0 → 1s）
    _timer: null,       // 重连计时器
    _paused: false,     // 页面隐藏暂停标志
    _toastSkip: ['heartbeat', 'connected', 'disconnected'], // 不打 toast 的事件

    /** 启动连接（幂等） */
    connect: async function () {
      if (this._abort || this._paused) return;
      var self = this;
      this._abort = new AbortController();
      var abort = this._abort;
      var headers = { Accept: 'text/event-stream' };
      var key = sseSessionKey();
      if (key) headers['X-API-Key'] = key;
      try {
        var response = await fetch(SSE_URL, { headers: headers, cache: 'no-store', signal: abort.signal });
        if (!response.ok) throw new Error('HTTP ' + response.status);
        if (!response.body) throw new Error('ReadableStream unavailable');
        self._connected = true;
        self._retry = 0;
        self._emit('connected', {});
        var reader = response.body.getReader();
        var decoder = new TextDecoder(), buffer = '', event = '', data = [];
        function consume(line) {
          if (line === '') {
            if (data.length) self._dispatch({ type: event || 'message', data: data.join('\n') });
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
        if (!abort.signal.aborted) console.warn('[UI9.SSE] 连接失败', e.message || e);
      } finally {
        if (self._abort === abort) self._abort = null;
        self._es = null;
        if (self._connected) { self._connected = false; self._emit('disconnected', {}); }
        if (!self._paused && !abort.signal.aborted) self._scheduleReconnect();
      }
    },

    /** 断开连接（同时取消待执行的重连） */
    disconnect: function () {
      if (this._timer) { clearTimeout(this._timer); this._timer = null; }
      if (this._abort) { this._abort.abort(); this._abort = null; }
      this._es = null;
      if (this._connected) { this._connected = false; this._emit('disconnected', {}); }
    },

    /** 订阅事件；返回 this 支持链式 */
    on: function (event, cb) {
      if (!event || !isFn(cb)) return this;
      if (!this._listeners[event]) this._listeners[event] = [];
      if (this._listeners[event].indexOf(cb) === -1) this._listeners[event].push(cb);
      return this;
    },

    /** 取消订阅 */
    off: function (event, cb) {
      var list = this._listeners[event];
      if (!list) return this;
      this._listeners[event] = list.filter(function (f) { return f !== cb; });
      return this;
    },

    /** 是否已连接 */
    isConnected: function () {
      return this._connected;
    },

    /** 立即重连（重置退避） */
    reconnectNow: function () {
      this._retry = 0;
      if (this._timer) { clearTimeout(this._timer); this._timer = null; }
      this.disconnect();
      this.connect();
    },

    /** 消息分发：解析 JSON → 按 type 派发 → toast 提示 */
    _dispatch: function (e) {
      var data, type, payload;
      try { data = JSON.parse(e.data); } catch (err) { data = null; }
      if (data && typeof data === 'object') {
        type = data.type || e.type || 'message';
        payload = data.data !== undefined ? data.data : data;
      } else {
        type = e.type || 'message';
        payload = data;
      }
      this._emit(type, payload);
      // 命名事件与 data.type 不一致时也补发一次命名事件
      if (e.type && e.type !== type) this._emit(e.type, payload);
      // 默认 toast 提示（心跳等事件跳过）
      if (this._toastSkip.indexOf(type) === -1) {
        var isWarn = /alert|warn/i.test(type);
        toast(this._brief(payload) || ('收到事件: ' + type), isWarn ? 'error' : 'info');
      }
    },

    /** 把事件负载压缩成适合 toast 的短文本 */
    _brief: function (payload) {
      if (payload == null) return '';
      if (typeof payload === 'string') return payload.slice(0, 120);
      if (typeof payload === 'object') {
        var msg = payload.message || payload.text || payload.title || payload.event || payload.type || '';
        if (msg) return String(msg).slice(0, 120);
        try { var s = JSON.stringify(payload); return s ? s.slice(0, 120) : ''; } catch (e) { return ''; }
      }
      return String(payload).slice(0, 120);
    },

    _emit: function (event, data) {
      var list = this._listeners[event];
      if (!list) return;
      list.slice().forEach(function (fn) {
        try { fn(data); } catch (e) { console.error('[UI9.SSE] 监听器异常:', e); }
      });
    },

    /** 指数退避重连：1s/2s/4s/8s/16s/30s… 封顶 30s */
    _scheduleReconnect: function () {
      if (this._timer || this._paused) return;
      var delay = Math.min(1000 * Math.pow(2, this._retry), 30000);
      this._retry++;
      var self = this;
      this._timer = setTimeout(function () {
        self._timer = null;
        self.connect();
      }, delay);
    },

    /** 页面隐藏 → 暂停；显示 → 恢复 */
    _onVisibility: function () {
      if (document.hidden) {
        this._paused = true;
        if (this._timer) { clearTimeout(this._timer); this._timer = null; }
        this.disconnect();
      } else {
        this._paused = false;
        this._retry = 0;
        this.connect();
      }
    },
  };

  /* ═══════════════════════ 导出 + 自动初始化 ═══════════════════════ */

  var UI9 = {
    VERSION: '1.0.0',
    DataTable: DataTable,
    toast: toast,
    confirmDialog: confirmDialog,
    GlobalSearch: GlobalSearch,
    Watchlist: Watchlist,
    sseClient: global.sseClient || sseClient,
    format: format,
  };
  global.UI9 = UI9;

  // 页面加载完成后自动初始化（全部防御式，缺元素不影响其他组件）
  function boot() {
    injectDefaultStyles();
    GlobalSearch.autoInit();   // 元素缺失时静默跳过
    Watchlist._seedFromServer();
    // static/sse-client.js owns the authenticated connection when loaded first.
    // Do not open a second /api/events stream from UI9.
    if (!global.sseClient) sseClient.connect();
    document.addEventListener('visibilitychange', function () {
      sseClient._onVisibility();
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

  console.log('[UI9] 交互组件库已加载 v' + UI9.VERSION);

  /* ═══════════════════════ 用法示例 ═══════════════════════
   *
   * ── DataTable 表格 ──
   * const table = new UI9.DataTable($('myTable'), {
   *   columns: [
   *     { key: 'symbol', title: '代码' },
   *     { key: 'name',   title: '名称' },
   *     { key: 'price',  title: '价格' },
   *     { key: 'change_pct', title: '涨跌幅',
   *       render: (v) => `<span class="${UI9.format.color(v)}">${UI9.format.pct(v)}</span>` },
   *   ],
   *   data: rows,
   *   pageSize: 15,
   * });
   * // 数据更新（保留排序/过滤状态）：
   * table.setData(newRows);
   *
   * ── toast 轻提示 ──
   * UI9.toast('保存成功', 'success');
   * UI9.toast('请求超时', 'error');
   * UI9.toast('普通消息', 'info');
   *
   * ── 确认弹窗 ──
   * const ok = await UI9.confirmDialog('删除确认', '确定要删除该记录吗？');
   * if (ok) { ... }
   * // 或回调式：UI9.confirmDialog('标题', '文案', (res) => { if (res) ... });
   *
   * ── 全局搜索（自动绑定 #globalSearchInput / #globalSearchDropdown）──
   * // 无需调用，页面加载后自动初始化；如需手动重试：UI9.GlobalSearch.init()
   *
   * ── 自选股 ──
   * UI9.Watchlist.add('002714', '牧原股份');
   * UI9.Watchlist.remove('002714');
   * UI9.Watchlist.list();        // [{symbol, name, ts}]
   * UI9.Watchlist.isWatched('002714');
   * UI9.Watchlist.toggleButton(document.querySelector('#starBtn'), '002714', '牧原股份');
   *
   * ── SSE 增强封装 ──
   * UI9.sseClient.on('macro_overview', (data) => renderMacro(data));
   * UI9.sseClient.on('connected', () => {});
   * UI9.sseClient.connect();     // 页面加载后自动连接，一般无需手动调用
   *
   * ── 格式化 ──
   * UI9.format.pct(0.1234);            // "0.12%"（注意：传小数请自行换算）
   * UI9.format.num(12345.678);         // "12,345.68"
   * UI9.format.cn(250000000);          // "2.50亿"
   * UI9.format.cn(88000);              // "8.80万"
   * UI9.format.color(2.5);             // "rt-up"（红涨）
   * UI9.format.color(-1.2);            // "rt-down"（绿跌）
   *
   * ═══════════════════════════════════════════════════════ */
})(window);
