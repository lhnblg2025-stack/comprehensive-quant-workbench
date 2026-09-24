"""融资融券数据 — 上交所 + 深交所两融明细。

海外可用性：东方财富 akshare 源在海外可能超时，但 SSE/SZSE 官方的数据
通过 `stock_margin_detail_sse()` 和 `stock_margin_detail_szse()` 走开放接口，
海外可以访问（已验证）。

Q4-fix（2026-08-02，V6 P0 修复 B 组）：
  #1  stock_margin_sse 显式传 end_date（akshare 1.18.64 默认 end_date 硬编码
      为 "20230922"，不传会导致恒取 2023-09 陈旧数据），并对返回数据做日期
      新鲜度校验，陈旧时置 ok=False + warning。
  #2  stock_margin_detail_sse/szse 显式传最近交易日日期，并校验数据日期。
  #6  stock_margin_szse 日期格式 "%Y-%m-%d" → "%Y%m%d"。
  #7  SSE 明细无"融券余额"列（仅"融券余量"=股数），新增 rqye_unit 字段
      （SSE="亿股"、SZSE="亿元"），禁止股数当金额。
  #16 深市失败不再静默用"沪市×0.67"混入总量：显式 sz_estimated + note。
  #17 date 标签改取数据源实际最新日期（data_date），与今天不一致时告警。
  #18 删除 TMT=总量×20% 拍脑袋估算；删除 TMT_SECTOR_CODES 死代码。
  #29 SSE 明细 code_col 回退时校验列内容为 6 位代码，否则 ok=False。
  #30 SZSE 个股明细补充数据日期。
  #31 清理 fetch_margin_summary 开头 result 覆盖死代码。
"""

from __future__ import annotations
import logging

from datetime import date, timedelta

import pandas as pd

# 北向/两融披露规则变更日：2024-08-19 起沪深交易所停止盘中/日度北向净买入披露。
# 两融无此变更，此处仅保留说明性注释。


def _today_compact() -> str:
    """今日日期 YYYYMMDD。"""
    return date.today().strftime("%Y%m%d")


def _latest_trading_day(max_back: int = 12) -> str:
    """尝试找到最近一个有数据的交易日（YYYYMMDD）。

    周末/节假日 akshare 的 SSE/SZSE 明细接口返回空，这里从今天往回逐日尝试。
    返回的日期不保证一定是交易日，但会优先落在接口有数据的日期上。
    """
    import akshare as ak

    for back in range(max_back):
        d = (date.today() - timedelta(days=back)).strftime("%Y%m%d")
        try:
            df = ak.stock_margin_detail_sse(date=d)
            if df is not None and not df.empty:
                return d
        except Exception as e:
            logging.getLogger(__name__).error(f"[margin] 操作失败: {e}", exc_info=True)
            continue
    return _today_compact()


def _check_data_freshness(rows: pd.DataFrame, date_col: str,
                          max_stale_days: int = 10) -> tuple[str, bool]:
    """校验数据最新日期。

    SSE 的 stock_margin_sse 返回按日期**降序**（最新在前），因此取第一行
    非空日期为最新。

    Returns
    -------
    (data_date, is_fresh)
      data_date: 数据源实际最新日期（"YYYY-MM-DD"）
      is_fresh:  数据是否在 max_stale_days 天内
    """
    try:
        vals = rows[date_col].dropna().astype(str).str.strip()
        if vals.empty:
            return "", False
        latest = vals.iloc[0]  # 降序：第一行最新
        # 兼容 "YYYYMMDD" 与 "YYYY-MM-DD"
        if len(latest) == 8 and latest.isdigit():
            latest_d = date(int(latest[:4]), int(latest[4:6]), int(latest[6:]))
        else:
            latest_d = pd.to_datetime(latest, errors="coerce")
            latest_d = latest_d.date() if hasattr(latest_d, "date") else latest_d
        if latest_d is None or pd.isna(latest_d):
            return latest, False
        fresh = (date.today() - latest_d).days <= max_stale_days
        return latest, fresh
    except Exception:
        return "", False


def _normalize_amount_series(vals) -> list[float]:
    """数值列统一换算为"元"。

    SSE stock_margin_sse 返回元；SZSE stock_margin_szse 返回亿元（量级 1e4）。
    按量级探测：列内最大值 < 1e9 视为亿元 → ×1e8。
    """
    out: list[float] = []
    for v in vals:
        try:
            out.append(float(v))
        except (ValueError, TypeError):
            continue
    if not out:
        return []
    # V11 审计修复（Medium）: 量级启发式（<1e9 视为亿元）在接口单位变化时
    # 静默误换算。修正: 换算后校验量级合理性（融资余额应在 1e7~1e13 元），
    # 超范围说明单位假设错误，返回空（由调用方可见降级，不静默污染）。
    if max(abs(x) for x in out) < 1e9:
        out = [x * 1e8 for x in out]  # 亿元 → 元
    if out and max(abs(x) for x in out) > 1e13:
        return []  # 量级异常：单位假设错误，拒绝使用
    return out


def _normalize_date_str(s) -> str:
    """将 YYYYMMDD 或 YYYY-MM-DD 统一为 YYYY-MM-DD（供 date/data_date 标签使用）。"""
    s = str(s or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def fetch_margin_summary() -> dict:
    """Fetch margin trading summary via akshare SSE+SZSE API.

    Returns
    -------
    dict with keys:
      - total_margin_balance      全市场融资余额（亿元，深市缺失时含估算并标注）
      - total_margin_prev         前一日全市场融资余额
      - total_margin_change       较前日变动（亿元）
      - tmt_margin_balance        TMT 行业融资余额（亿元）—— 已下线估算口径，恒 0
      - tmt_margin_prev           TMT 前一日（恒 0）
      - tmt_margin_change         TMT 较前日变动（恒 0）
      - ss_margin_balance         沪市融资余额
      - sz_margin_balance         深市融资余额
      - date / data_date          数据日期（数据源实际最新日期）
      - ok
    """
    # P2-Q4-fix(L424): 删除下方被无条件覆盖的 result 死初始化（Q4#31），
    # 失败路径统一由末尾的 result 组装与 error 标注兜底。
    # ── 用线程超时包装 AkShare 调用，防止网络超时阻塞 ──
    import threading as _th

    _sh_result = {"balance": 0.0, "prev": 0.0, "data_date": "", "fresh": False}
    _sz_result = {"balance": 0.0, "prev": 0.0, "data_date": "", "fresh": False}

    def _fetch_sh():
        try:
            import akshare as ak
            # Q4#1: 必须显式传 end_date，否则 akshare 默认 20230922（3 年前陈旧数据）
            end = _today_compact()
            start = (date.today() - timedelta(days=45)).strftime("%Y%m%d")
            try:
                df_sh = ak.stock_margin_sse(start_date=start, end_date=end)
            except Exception:
                # 显式传参失败则不再无参回退（无参回退=20230922 陈旧数据），
                # 直接返回失败，由上层标 ok=False
                df_sh = None
            if df_sh is not None and not df_sh.empty and len(df_sh.columns) > 1:
                # 数据日期校验（SSE 列为"信用交易日期"，格式 YYYYMMDD）
                date_col = next((c for c in df_sh.columns if "日期" in str(c)), None)
                if date_col is not None:
                    _sh_result["data_date"], _sh_result["fresh"] = _check_data_freshness(
                        df_sh, date_col)
                rzye_col = next((c for c in df_sh.columns
                                 if "融资余额" in str(c) and "融券" not in str(c)), None)
                if rzye_col:
                    # Q4-fix: SSE 返回按日期**降序**（最新在前），vals[0] 才是最新
                    vals = _normalize_amount_series(df_sh[rzye_col].dropna().values)
                    if len(vals) >= 2:
                        _sh_result["balance"] = float(vals[0])
                        _sh_result["prev"] = float(vals[1])
                    elif len(vals) == 1:
                        _sh_result["balance"] = float(vals[0])
        except Exception as e:
            logging.getLogger(__name__).error(f"[margin] 操作失败: {e}", exc_info=True)

    def _fetch_sz():
        try:
            import akshare as ak
            # Q4#6: szse 需要 "YYYYMMDD"（内部拼 txtDate），原代码传 "%Y-%m-%d"
            # 会拼出垃圾日期导致深市永远取不到。
            # 返回值为**亿元**（_normalize_amount_series 统一换算为元）；
            # 单日 1 行，prev 需取上一交易日。
            def _sz_on(d: str):
                try:
                    df = ak.stock_margin_szse(date=d)
                    if df is None or df.empty or len(df.columns) < 2:
                        return None, []
                    rzye_col = next((c for c in df.columns
                                     if "融资余额" in str(c) and "融券" not in str(c)), None)
                    if rzye_col is None:
                        return None, []
                    return d, _normalize_amount_series(df[rzye_col].dropna().values)
                except Exception:
                    return None, []

            d0, vals = _sz_on(_today_compact())
            if d0 is None:
                # 回退找最近有数据的交易日（周末/节假日接口返回空）
                for back in range(1, 8):
                    dd = (date.today() - timedelta(days=back)).strftime("%Y%m%d")
                    d0, vals = _sz_on(dd)
                    if d0 is not None:
                        break
            if d0 is not None and vals:
                _sz_result["balance"] = float(vals[-1]) if vals else 0.0
                _sz_result["data_date"] = d0
                _sz_result["fresh"] = (date.today() - pd.to_datetime(d0).date()).days <= 10
                # 上一交易日 prev（向前最多 8 天）
                for back in range(1, 9):
                    dd = (pd.to_datetime(d0).date() - timedelta(days=back)).strftime("%Y%m%d")
                    _, pvals = _sz_on(dd)
                    if pvals:
                        _sz_result["prev"] = float(pvals[-1])
                        break
        except Exception as e:
            logging.getLogger(__name__).error(f"[margin] 操作失败: {e}", exc_info=True)

    t_sh = _th.Thread(target=_fetch_sh, daemon=True)
    t_sz = _th.Thread(target=_fetch_sz, daemon=True)
    t_sh.start()
    t_sz.start()
    t_sh.join(timeout=10)  # 沪市最多等10s
    t_sz.join(timeout=10)  # 深市最多等10s

    sh_balance = _sh_result["balance"]
    sh_prev_balance = _sh_result["prev"]
    sz_balance = _sz_result["balance"]
    sz_prev_balance = _sz_result["prev"]

    # Q4#16/P2-Q4-fix(M409): 深市数据获取失败时不再静默混入"沪市×0.67"估算。
    # - 估算值仍保留在 total_margin_balance 供展示，但通过 sz_estimated /
    #   total_estimated / margin_scope 三重标注"仅沪市+估算"，禁止下游当精确值。
    # - ok 在含估算时置 False（深市失败=数据不完整），ok-gated 调用方不再消费估算总量。
    sz_estimated = False
    note = ""
    if sz_balance < 1e6 and sh_balance > 1e6:
        sz_balance = sh_balance * 0.67
        sz_prev_balance = sh_prev_balance * 0.67
        sz_estimated = True
        note = "深交所两融数据获取失败，总量含估算（沪市×0.67），非精确值"

    total = sh_balance + sz_balance
    total_prev = sh_prev_balance + sz_prev_balance

    # 数据日期：优先取真实数据源日期（Q4#17/P2-Q4-fix(M410)），统一为 YYYY-MM-DD
    data_date = _normalize_date_str(
        _sh_result["data_date"] or _sz_result["data_date"] or str(date.today()))
    warnings: list[str] = []
    if not _sh_result["fresh"] and sh_balance > 0:
        warnings.append(f"沪市两融数据可能陈旧（最新数据日期 {data_date}）")
    if sz_estimated:
        warnings.append(note)
    # P2-Q4-fix(M410): 数据日期与今天不一致时告警（避免前视/陈旧混淆）
    if data_date and data_date != str(date.today()):
        warnings.append(f"数据日期({data_date})与今天({str(date.today())})不一致，非当日数据")

    # ok 判定：必须有真实数据（沪市或深市）且数据新鲜（Q4#1/#17）；
    # P2-Q4-fix(M409): 含估算时 ok 置 False，禁止把估算总量当真实值喂给下游。
    sh_real = sh_balance > 1e6
    sz_real = sz_balance > 1e6 and not sz_estimated
    has_real = sh_real or sz_real
    ok = has_real and (_sh_result["fresh"] or _sz_result["fresh"]) and not sz_estimated

    # P2-Q4-fix(M409): 明确数据覆盖口径，便于下游判断
    if sz_estimated:
        margin_scope = "仅沪市真实+深市估算"
    elif sh_real and sz_real:
        margin_scope = "沪深全量真实"
    elif sh_real:
        margin_scope = "仅沪市真实(深市无数据)"
    elif sz_real:
        margin_scope = "仅深市真实(沪市无数据)"
    else:
        margin_scope = "无有效数据"

    result = {
        "ok": ok,
        # P2-Q4-fix(M410): date 标签取数据源实际最新日期，不再用今天伪标签
        "date": str(data_date),
        "data_date": str(data_date),
        "total_margin_balance": round(total / 1e8, 2) if total > 1e6 else round(total, 2),
        "total_margin_prev": round(total_prev / 1e8, 2) if total_prev > 1e6 else round(total_prev, 2),
        "total_margin_change": round((total - total_prev) / 1e8, 2) if total > 1e6 else round(total - total_prev, 2),
        "ss_margin_balance": round(sh_balance / 1e8, 2) if sh_balance > 1e6 else round(sh_balance, 2),
        "sz_margin_balance": round(sz_balance / 1e8, 2) if sz_balance > 1e6 else round(sz_balance, 2),
        "sz_estimated": sz_estimated,
        "total_estimated": sz_estimated,   # P2-Q4-fix(M409)
        "margin_scope": margin_scope,       # P2-Q4-fix(M409)
        # Q4#18: TMT 行业两融无现成数据源，删除 20% 拍脑袋估算口径
        "tmt_margin_balance": 0,
        "tmt_margin_prev": 0,
        "tmt_margin_change": 0,
        "tmt_estimated": False,
        "tmt_note": "TMT行业两融无公开数据源，已下线20%比例估算口径",
        "warnings": warnings,
    }
    if not ok:
        if sz_estimated:
            result["error"] = "深市两融数据获取失败，总量为估算值（仅沪市真实），ok=False"
        else:
            result["error"] = "两融数据获取失败或数据陈旧（沪/深接口均无有效数据）"
    if warnings:
        result["warning"] = "; ".join(warnings)

    return result


def fetch_margin_individual(symbol: str) -> dict:
    """Fetch individual stock margin detail from SSE or SZSE."""
    import akshare as ak

    # Normalize symbol: pad to 6 digits for matching
    sym = str(symbol).strip().zfill(6)

    try:
        # Q4#2: 显式传最近交易日，避免 akshare 默认值 20230922/20230925（3 年前陈旧）
        query_date = _latest_trading_day()

        # Try SSE first (code column is 6 digits, may have leading zeros)
        try:
            df = ak.stock_margin_detail_sse(date=query_date)
        except Exception:
            df = None
        if df is not None and not df.empty:
            code_col = "标的证券代码" if "标的证券代码" in df.columns else df.columns[1]
            # Q4#29: 回退列必须校验内容为 6 位代码，否则匹配静默失败会返回假成功
            code_vals = df[code_col].astype(str).str.strip()
            is_code_col = code_vals.str.fullmatch(r"\d{6}").mean() > 0.5
            if not is_code_col:
                return {"ok": False, "symbol": symbol,
                        "error": "SSE 明细代码列识别失败（列名与内容均不匹配）"}
            match = df[code_vals.str.zfill(6) == sym]
            if not match.empty:
                row = match.iloc[-1]
                rzye_col = next((c for c in df.columns if "融资余额" in c), None)
                # Q4#7: SSE 明细只有"融券余量"(股)，没有"融券余额"金额列
                rqye_col = next((c for c in df.columns if "融券余量" in c), None)
                date_col = next((c for c in df.columns if "日期" in str(c)), None)
                data_date = str(row.get(date_col, "")) if date_col else query_date
                return {
                    "ok": True,
                    "symbol": symbol,
                    "rzye": round(_f(row.get(rzye_col, 0)) / 1e8, 2) if rzye_col else 0,
                    "rqye": round(_f(row.get(rqye_col, 0)) / 1e8, 2) if rqye_col else 0,
                    "rqye_unit": "亿股",  # SSE 融券为股数口径
                    "date": str(data_date),
                    "data_date": str(data_date),
                    "source": "sse",
                }

        # Try SZSE（明细发布可能滞后于 SSE：先按查询日，失败则回退找最近
        # 有 SZSE 明细的日期）
        szse_df = None
        szse_date = query_date
        for back in range(0, 7):
            dd = (date.today() - timedelta(days=back)).strftime("%Y%m%d")
            try:
                df = ak.stock_margin_detail_szse(date=dd)
            except Exception:
                df = None
            if df is not None and not df.empty:
                szse_df = df
                szse_date = dd
                break
        if szse_df is not None and not szse_df.empty:
            df = szse_df
            code_col = "证券代码" if "证券代码" in df.columns else df.columns[0]
            code_vals = df[code_col].astype(str).str.strip()
            is_code_col = code_vals.str.fullmatch(r"\d{6}").mean() > 0.5
            if not is_code_col:
                return {"ok": False, "symbol": symbol,
                        "error": "SZSE 明细代码列识别失败（列名与内容均不匹配）"}
            match = df[code_vals.str.zfill(6) == sym]
            if not match.empty:
                row = match.iloc[-1]
                rzye_col = next((c for c in df.columns if "融资余额" in c), None)
                rqye_col = next((c for c in df.columns if "融券余额" in c), None)
                # Q4#30: SZSE 明细接口返回当日明细，日期即查询日
                return {
                    "ok": True,
                    "symbol": symbol,
                    "rzye": round(_f(row.get(rzye_col, 0)) / 1e8, 2) if rzye_col else 0,
                    "rqye": round(_f(row.get(rqye_col, 0)) / 1e8, 2) if rqye_col else 0,
                    "rqye_unit": "亿元",  # SZSE 融券为金额口径
                    "date": szse_date,
                    "data_date": szse_date,
                    "source": "szse",
                }

        return {"ok": False, "symbol": symbol,
                "error": f"未找到 {sym} 的两融明细（查询日 {query_date}）"}
    except Exception as exc:
        return {"ok": False, "symbol": symbol, "error": repr(exc)[:80]}


def _f(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0
