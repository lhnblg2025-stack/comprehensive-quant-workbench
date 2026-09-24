"""
市场脉搏（轻量版）— A股实时情绪+恐慌贪婪+热点

轻量版：只用涨跌家数/涨跌停/成交额三个轻量API + 北向资金缓存
避免：全量行情/行业遍历/指数历史等重操作
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CST = timezone(timedelta(hours=8))

# Q11-fix: ak.stock_adv_dec_stats 在 akshare 1.18.64 不存在（实测 AttributeError），
# 涨跌家数维度（恐慌贪婪 40% 权重）因此永久失效。改用 stock_zh_a_spot_em()
# 全市场快照自行统计涨跌家数（同 scripts/a_share_daily_report.py 口径）。
_ADV_DEC_CACHE: dict[str, tuple[float, dict]] = {}
_ADV_DEC_TTL = 120  # 秒，盘中 2 分钟刷新


def _fetch_adv_dec() -> dict:
    """统计沪深A股实时涨跌家数（上涨/下跌/平盘）+ 全市场成交额（亿元）。

    基于 ak.stock_zh_a_spot_em() 全市场快照（同 scripts/a_share_daily_report.py）。
    列名变化时在返回的 note 字段中标注，供上层可见降级，不静默跳过。

    Returns:
        {up, down, flat, total, up_ratio, amount_yi, note?}；
        网络/列名异常时返回仅含 note 的最小 dict（不含统计项）。
    """
    import time as _t
    now = _t.time()
    hit = _ADV_DEC_CACHE.get("data")
    if hit and now - hit[0] < _ADV_DEC_TTL:
        return hit[1]
    try:
        import akshare as ak
        import pandas as pd
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return {"note": "全市场快照为空(stock_zh_a_spot_em)"}
        result: dict = {}
        notes: list[str] = []
        # 涨跌家数
        if "涨跌幅" in df.columns:
            pct = pd.to_numeric(df["涨跌幅"], errors="coerce").dropna()
            if len(pct) > 0:
                result.update({
                    "up": int((pct > 0).sum()),
                    "down": int((pct < 0).sum()),
                    "flat": int((pct == 0).sum()),
                    "total": int(len(pct)),
                    "up_ratio": float((pct > 0).mean()),
                })
            else:
                notes.append("涨跌幅列无有效数值")
        else:
            notes.append("列名变更: stock_zh_a_spot_em 缺少'涨跌幅'列")
        # 全市场成交额（亿元）— P1-Q11-fix(H01): 原 stock_sse_summary 无'成交额'
        # 列导致 20% 权重静默失效，这里用同一快照汇总，命中 TTL 缓存不再重复取数。
        if "成交额" in df.columns:
            amt = pd.to_numeric(df["成交额"], errors="coerce").dropna()
            if len(amt) > 0:
                result["amount_yi"] = float(amt.sum() / 1e8)
            else:
                notes.append("成交额列无有效数值")
        else:
            notes.append("列名变更: stock_zh_a_spot_em 缺少'成交额'列")
        if notes:
            result["note"] = "; ".join(notes)
        _ADV_DEC_CACHE["data"] = (now, result)
        return result
    except Exception as e:
        return {"note": f"全市场快照获取失败: {type(e).__name__}: {e}"}


def get_fear_greed_index() -> dict:
    """A股恐慌贪婪指数（轻量版）

    基于: 涨跌家数(40%) + 涨跌停比(30%) + 成交量(20%) + 北向(10%)
    0-100分
    """
    score = 50
    components: dict = {}
    missing: list[str] = []
    component_status: dict[str, str] = {}

    try:
        import akshare as ak
        today = datetime.now(CST).strftime('%Y%m%d')

        # 1. 涨跌家数 (40%)
        try:
            adv = _fetch_adv_dec()
            up = adv.get('up', 0)
            down = adv.get('down', 0)
            if up + down > 0:
                adv_score = up / (up + down) * 100
                components['涨跌比'] = round(adv_score, 1)
                component_status['涨跌比'] = 'ok'
                score += (adv_score - 50) * 0.4
            else:
                note = adv.get('note', '无有效涨跌数据')
                component_status['涨跌比'] = f"missing: {note}"
                missing.append(f"涨跌比({note})")
        except Exception as e:
            component_status['涨跌比'] = f"error: {type(e).__name__}"
            missing.append(f"涨跌比({type(e).__name__})")

        # 2. 涨跌停 (30%)
        try:
            zt = ak.stock_zt_pool_em(date=today)
            dt = ak.stock_zt_pool_dtgc_em(date=today)
            up_l = len(zt) if zt is not None else 0
            down_l = len(dt) if dt is not None else 0
            if up_l + down_l > 0:
                limit_score = up_l / (up_l + down_l) * 100
                components['涨跌停比'] = round(limit_score, 1)
                component_status['涨跌停比'] = 'ok'
                score += (limit_score - 50) * 0.3
            else:
                component_status['涨跌停比'] = 'missing: 涨停/跌停均为空'
                missing.append("涨跌停比(涨停/跌停均为空)")
        except Exception as e:
            component_status['涨跌停比'] = f"error: {type(e).__name__}"
            missing.append(f"涨跌停比({type(e).__name__})")

        # 3. 成交额因子 (20%)
        # P1-Q11-fix(H01): 原 ak.stock_sse_summary() 实测列名为
        # ['项目','股票','主板','科创板']，无'成交额'列 → '成交额' in dat 恒 False
        # → 20% 权重静默失效。改用 _fetch_adv_dec 内 stock_zh_a_spot_em 汇总
        # 全市场成交额（亿元，命中 TTL 缓存与第1步共用一次 API），列名变化经
        # note 字段可见降级（运行时校验，不静默跳过）。
        try:
            adv = _fetch_adv_dec()
            amount_yi = adv.get('amount_yi', 0)
            if amount_yi > 0:
                # 5000亿以下→恐慌, 1.5万亿以上→贪婪
                amount_score = min(100, max(0, amount_yi / 15000 * 100))
                components['成交额'] = round(amount_score, 1)
                component_status['成交额'] = 'ok'
                score += (amount_score - 50) * 0.2
            else:
                note = adv.get('note', '无成交额数据')
                component_status['成交额'] = f"missing: {note}"
                missing.append(f"成交额({note})")
        except Exception as e:
            component_status['成交额'] = f"error: {type(e).__name__}"
            missing.append(f"成交额({type(e).__name__})")

        # 4. 北向资金 (10%) - 使用缓存
        # P1-Q23-fix(H03): 原代码 from quant_system.north_flow import
        # get_north_flow_summary — 该函数不存在 → ImportError 被吞 → 北向
        # 10% 权重因子恒缺失。改调真实存在的 fetch_north_summary 并适配其
        # 返回键（net_buy_disclosed/total_net_yi）；2024-08-19 起北向净买入
        # 停止披露，该分量按中性处理并明确标注。
        try:
            # P1-Q11-fix(H02): fetch_north_summary 存在于 north_flow.py。
            # 直接以脚本运行时 (python3 market_pulse.py) 父目录不在 sys.path，
            # quant_system 包不可导入，回退到同目录 north_flow 模块。
            try:
                from quant_system.north_flow import fetch_north_summary
            except ImportError:
                if str(ROOT) not in sys.path:
                    sys.path.insert(0, str(ROOT))
                from north_flow import fetch_north_summary
            north = fetch_north_summary()
            if north and "error" not in north:
                if north.get("net_buy_disclosed") is False:
                    components['北向资金'] = 50.0
                    components['北向资金_说明'] = '2024-08-19起北向净买入停止披露,该分量按中性处理'
                    component_status['北向资金'] = 'neutral: 盘中净买入已停止披露'
                else:
                    nf = north.get('total_net_yi', 0) * 1e8  # 亿元 → 元
                    nf_score = 50 + nf / 1e9 * 5
                    nf_score = max(0, min(100, nf_score))
                    components['北向资金'] = round(nf_score, 1)
                    component_status['北向资金'] = 'ok'
                    score += (nf_score - 50) * 0.1
            else:
                err = north.get('error', '空响应') if north else '空响应'
                component_status['北向资金'] = f"missing: {err}"
                missing.append(f"北向资金({err})")
        except ImportError as e:
            component_status['北向资金'] = f"error: ImportError: {e}"
            missing.append(f"北向资金(ImportError: {e})")
        except Exception as e:
            component_status['北向资金'] = f"error: {type(e).__name__}"
            missing.append(f"北向资金({type(e).__name__})")

    except ImportError:
        component_status['akshare'] = 'error: ImportError'
        missing.append("akshare(ImportError)")

    # P1-Q11-fix(H02): 输出 components 缺失清单供调用方感知数据完整性，避免静默降级
    if missing:
        components['_缺失'] = missing

    # P2-Q11-fix(M049): 全部组件失败时不能静默返回 50/中性；暴露组件状态和
    # data_available，调用方可区分真实中性与节假日/接口故障导致的无数据。
    expected_components = {'涨跌比', '涨跌停比', '成交额', '北向资金'}
    data_available = any(component_status.get(k, '').startswith(('ok', 'neutral')) for k in expected_components)

    score = max(0, min(100, score))

    if score <= 25:
        level, advice = "极度恐慌 🚨", "寻底中，关注超跌"
    elif score <= 45:
        level, advice = "恐慌 ⚠️", "谨慎观望"
    elif score <= 55:
        level, advice = "中性 ➖", "震荡格局"
    elif score <= 75:
        level, advice = "贪婪 📈", "市场活跃"
    else:
        level, advice = "极度贪婪 🚀", "警惕回调"

    return {
        'index': round(score, 1),
        'level': level,
        'advice': advice,
        'components': components,
        'data_available': data_available,
        'component_status': component_status,
        'warning': '全部市场脉搏组件不可用，指数按中性50占位' if not data_available else None,
    }


def format_pulse() -> str:
    now = datetime.now(CST)
    fg = get_fear_greed_index()

    lines = [f"# 🌡️ 市场脉搏 ({now.strftime('%Y-%m-%d %H:%M')})"]
    lines.append(f"{'=' * 40}")

    # 恐慌贪婪
    lines.append(f"\n🎯 恐慌贪婪: {fg.get('index', 'N/A')}/100 → {fg.get('level', 'N/A')}")
    lines.append(f"    建议: {fg.get('advice', '')}")
    for k, v in fg.get('components', {}).items():
        # P1-Q11-fix(H02): 跳过缺失清单(_缺失 list)与说明类字符串等元信息，
        # 只对数值分量做条形图
        if not isinstance(v, (int, float)):
            continue
        bars = "█" * max(0, int(v / 10))  # P2-Q11-fix(L054): 0 分组件不显示 1 格进度条
        lines.append(f"    {k}: {v:.0f}/100 {bars}")

    missing = fg.get('components', {}).get('_缺失')
    if missing:
        lines.append(f"    ⚠️ 缺失分量: {'; '.join(missing)}")

    return "\n".join(lines)


def main():
    if "--sentiment" in sys.argv:
        fg = get_fear_greed_index()
        import json
        print(json.dumps(fg, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_pulse())


if __name__ == "__main__":
    main()
