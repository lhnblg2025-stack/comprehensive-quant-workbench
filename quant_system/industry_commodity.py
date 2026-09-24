"""Auditable SW industry/company -> commodity futures exposure model.

This is an economic exposure model, not a price-correlation shortcut:
product prices are usually positive, input/fuel prices are usually negative, and
upstream proxies are explicitly marked. Missing or stale contracts never become
zero returns; only fresh observed weights participate and coverage is reported.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MARKET_DIR = ROOT / "data_warehouse" / "market"
SW_MAP_PATH = MARKET_DIR / "sw_industry_map.parquet"
SW_SECOND_CONS_PATH = ROOT / "data_warehouse" / "industry" / "sw_second_cons.parquet"
MAPPING_VERSION = "sw-2026.08-economic-v2"
COMMODITY_FRESHNESS_LIMIT_DAYS = 5
# 采集器只请求已验证可由 AkShare 主力连续接口提供的品种；其余
# COMMODITY_SPECS 仍可作为公司暴露登记，但会在证据层标记 not_observable/not_collected。
COLLECTOR_TAGS = ("lh", "meal", "soybean", "corn", "gold", "copper", "alu", "crude", "coal")


def _spec(tag: str, name: str, role: str, direction: int, weight: float,
          *, confidence: str = "medium", proxy: bool = False,
          basis: str = "industry_prior", lag_days: int = 0) -> dict[str, Any]:
    return {
        "tag": tag, "name": name, "role": role, "direction": int(direction),
        "weight": float(weight), "weight_gross": float(weight),
        "weight_basis": basis, "confidence": confidence, "proxy_flag": bool(proxy),
        "lag_days": int(lag_days), "missing_policy": "exclude_and_report",
    }


# Local file tag -> contract metadata. The collector uses the same tags.
COMMODITY_SPECS: dict[str, dict[str, Any]] = {
    "lh": {"name": "生猪期货", "contract": "LH0", "exchange": "DCE", "aliases": ("生猪", "猪肉", "LH")},
    "meal": {"name": "豆粕期货", "contract": "M0", "exchange": "DCE", "aliases": ("豆粕", "M")},
    "soybean": {"name": "黄大豆一号期货", "contract": "A0", "exchange": "DCE", "aliases": ("大豆", "黄大豆", "A")},
    "corn": {"name": "玉米期货", "contract": "C0", "exchange": "DCE", "aliases": ("玉米", "C")},
    "gold": {"name": "黄金期货", "contract": "AU0", "exchange": "SHFE", "aliases": ("黄金", "AU")},
    "silver": {"name": "白银期货", "contract": "AG0", "exchange": "SHFE", "aliases": ("白银", "AG")},
    "copper": {"name": "铜期货", "contract": "CU0", "exchange": "SHFE", "aliases": ("铜", "CU")},
    "alu": {"name": "铝期货", "contract": "AL0", "exchange": "SHFE", "aliases": ("铝", "AL")},
    "zinc": {"name": "锌期货", "contract": "ZN0", "exchange": "SHFE", "aliases": ("锌", "ZN")},
    "lead": {"name": "铅期货", "contract": "PB0", "exchange": "SHFE", "aliases": ("铅", "PB")},
    "tin": {"name": "锡期货", "contract": "SN0", "exchange": "SHFE", "aliases": ("锡", "SN")},
    "nickel": {"name": "镍期货", "contract": "NI0", "exchange": "SHFE", "aliases": ("镍", "NI")},
    "lithium": {"name": "碳酸锂期货", "contract": "LC0", "exchange": "GFEX", "aliases": ("碳酸锂", "锂", "LC")},
    "coal": {"name": "动力煤期货", "contract": "ZC0", "exchange": "CZCE", "aliases": ("动力煤", "煤炭", "ZC")},
    "coking_coal": {"name": "焦煤期货", "contract": "JM0", "exchange": "DCE", "aliases": ("焦煤", "JM")},
    "coke": {"name": "焦炭期货", "contract": "J0", "exchange": "DCE", "aliases": ("焦炭", "J")},
    "iron_ore": {"name": "铁矿石期货", "contract": "I0", "exchange": "DCE", "aliases": ("铁矿石", "铁矿", "I")},
    "rebar": {"name": "螺纹钢期货", "contract": "RB0", "exchange": "SHFE", "aliases": ("螺纹钢", "RB")},
    "hot_roll": {"name": "热轧卷板期货", "contract": "HC0", "exchange": "SHFE", "aliases": ("热轧", "热卷", "HC")},
    "crude": {"name": "原油期货", "contract": "SC0", "exchange": "INE", "aliases": ("原油", "SC")},
    "fuel_oil": {"name": "燃料油期货", "contract": "FU0", "exchange": "SHFE", "aliases": ("燃料油", "FU")},
    "methanol": {"name": "甲醇期货", "contract": "MA0", "exchange": "CZCE", "aliases": ("甲醇", "MA")},
    "urea": {"name": "尿素期货", "contract": "UR0", "exchange": "CZCE", "aliases": ("尿素", "UR")},
    "soda_ash": {"name": "纯碱期货", "contract": "SA0", "exchange": "CZCE", "aliases": ("纯碱", "SA")},
    "polypropylene": {"name": "聚丙烯期货", "contract": "PP0", "exchange": "DCE", "aliases": ("聚丙烯", "PP")},
    "polyethylene": {"name": "聚乙烯期货", "contract": "L0", "exchange": "DCE", "aliases": ("聚乙烯", "塑料", "L")},
    "rubber": {"name": "天然橡胶期货", "contract": "RU0", "exchange": "SHFE", "aliases": ("天然橡胶", "橡胶", "RU")},
    "pta": {"name": "PTA期货", "contract": "TA0", "exchange": "CZCE", "aliases": ("PTA", "TA")},
    "glass": {"name": "玻璃期货", "contract": "FG0", "exchange": "CZCE", "aliases": ("玻璃", "FG")},
}


# Relative economic exposure within each profile. These are priors, not
# fabricated observed elasticities; later calibration may replace them.
INDUSTRY_COMMODITY_MAP: dict[str, list[dict[str, Any]]] = {
    "农林牧渔": [
        _spec("lh", "生猪", "养殖产品价格", 1, 1.00, confidence="medium"),
        _spec("meal", "豆粕", "饲料成本", -1, 0.65, confidence="high"),
        _spec("corn", "玉米", "饲料成本", -1, 0.45, confidence="high"),
        _spec("soybean", "大豆", "豆粕上游成本代理", -1, 0.20, confidence="low", proxy=True, basis="upstream_proxy_not_additive"),
    ],
    "养殖业": [
        _spec("lh", "生猪", "养殖产品价格", 1, 1.20, confidence="high", basis="company_product_and_industry"),
        _spec("meal", "豆粕", "饲料蛋白成本", -1, 0.75, confidence="high", basis="feed_cost_prior"),
        _spec("corn", "玉米", "饲料能量成本", -1, 0.55, confidence="high", basis="feed_cost_prior"),
        _spec("soybean", "大豆", "豆粕上游成本代理", -1, 0.30, confidence="low", proxy=True, basis="upstream_proxy_not_additive"),
    ],
    "饲料": [
        _spec("meal", "豆粕", "饲料原料成本", -1, 1.00, confidence="high"),
        _spec("corn", "玉米", "饲料原料成本", -1, 0.80, confidence="high"),
        _spec("soybean", "大豆", "蛋白原料上游代理", -1, 0.25, confidence="low", proxy=True, basis="upstream_proxy_not_additive"),
    ],
    "农产品加工": [
        _spec("soybean", "大豆", "压榨原料成本", -1, 0.80, confidence="high"),
        _spec("meal", "豆粕", "副产品销售价格", 1, 0.55, confidence="medium"),
        _spec("corn", "玉米", "加工原料成本", -1, 0.35, confidence="medium"),
    ],
    "肉制品": [
        _spec("lh", "生猪", "肉制品原料成本代理", -1, 0.70, confidence="medium", basis="company_raw_material_prior"),
    ],
    "种植业": [
        _spec("corn", "玉米", "农产品价格", 1, 0.70, confidence="medium"),
        _spec("soybean", "大豆", "农产品价格", 1, 0.55, confidence="medium"),
    ],
    "贵金属": [_spec("gold", "黄金", "产品价格", 1, 1.00, confidence="high", basis="company_product")],
    "工业金属": [
        _spec("copper", "铜", "产品价格", 1, 1.00, confidence="high"),
        _spec("alu", "铝", "产品价格", 1, 0.85, confidence="high"),
    ],
    "有色金属": [
        _spec("copper", "铜", "产品价格/行业代理", 1, 0.95, confidence="medium", proxy=True),
        _spec("gold", "黄金", "产品价格/行业代理", 1, 0.80, confidence="medium", proxy=True),
        _spec("alu", "铝", "产品价格/行业代理", 1, 0.60, confidence="medium", proxy=True),
    ],
    "金属新材料": [
        _spec("copper", "铜", "材料产品价格代理", 1, 0.65, confidence="low", proxy=True),
        _spec("alu", "铝", "材料产品价格代理", 1, 0.55, confidence="low", proxy=True),
    ],
    "小金属": [
        _spec("tin", "锡", "产品价格", 1, 0.60, confidence="medium"),
        _spec("lead", "铅", "产品价格", 1, 0.40, confidence="medium"),
        _spec("nickel", "镍", "产品价格代理", 1, 0.30, confidence="low", proxy=True),
    ],
    "能源金属": [
        _spec("lithium", "碳酸锂", "产品价格", 1, 1.00, confidence="high", basis="company_product"),
    ],
    "金铜资源": [
        _spec("gold", "黄金", "产品价格", 1, 0.80, confidence="high", basis="company_product"),
        _spec("copper", "铜", "产品价格", 1, 0.70, confidence="high", basis="company_product"),
    ],
    "铜钼钴资源": [
        _spec("copper", "铜", "产品价格", 1, 0.70, confidence="high", basis="company_product"),
        _spec("molybdenum", "钼", "产品价格（无直接期货）", 1, 0.45, confidence="medium", basis="company_product", proxy=False),
        _spec("cobalt", "钴", "产品价格（无直接期货）", 1, 0.25, confidence="medium", basis="company_product", proxy=False),
    ],
    "钢铁": [
        _spec("rebar", "螺纹钢", "钢材产品价格", 1, 0.35, confidence="high"),
        _spec("hot_roll", "热轧卷板", "钢材产品价格", 1, 0.20, confidence="medium"),
        _spec("iron_ore", "铁矿石", "炉料成本", -1, 0.30, confidence="high"),
        _spec("coking_coal", "焦煤", "炉料成本", -1, 0.10, confidence="medium"),
        _spec("coke", "焦炭", "炉料成本", -1, 0.05, confidence="medium"),
    ],
    "普钢": [
        _spec("rebar", "螺纹钢", "钢材产品价格", 1, 0.35, confidence="high"),
        _spec("hot_roll", "热轧卷板", "钢材产品价格", 1, 0.20, confidence="medium"),
        _spec("iron_ore", "铁矿石", "炉料成本", -1, 0.30, confidence="high"),
        _spec("coking_coal", "焦煤", "炉料成本", -1, 0.10, confidence="medium"),
        _spec("coke", "焦炭", "炉料成本", -1, 0.05, confidence="medium"),
    ],
    "特钢": [
        _spec("hot_roll", "热轧卷板", "钢材产品价格", 1, 0.30, confidence="medium"),
        _spec("rebar", "螺纹钢", "钢材产品价格代理", 1, 0.15, confidence="low", proxy=True),
        _spec("iron_ore", "铁矿石", "炉料成本", -1, 0.25, confidence="high"),
        _spec("coking_coal", "焦煤", "炉料成本", -1, 0.15, confidence="medium"),
    ],
    "冶钢原料": [
        _spec("iron_ore", "铁矿石", "原料产品价格", 1, 0.70, confidence="high"),
        _spec("coking_coal", "焦煤", "原料产品价格", 1, 0.30, confidence="medium"),
    ],
    "煤炭": [_spec("coal", "动力煤", "产品价格", 1, 1.00, confidence="high")],
    "煤炭开采": [_spec("coal", "动力煤", "产品价格", 1, 1.00, confidence="high")],
    "焦炭": [
        _spec("coke", "焦炭", "产品价格", 1, 0.70, confidence="high"),
        _spec("coking_coal", "焦煤", "原料成本", -1, 0.30, confidence="high"),
    ],
    "石油石化": [_spec("crude", "原油", "产品/成本基准", 1, 1.00, confidence="medium", proxy=True)],
    "油气开采": [_spec("crude", "原油", "产品价格", 1, 1.00, confidence="high")],
    "油服工程": [_spec("crude", "原油", "资本开支周期代理", 1, 0.75, confidence="low", proxy=True)],
    "炼化及贸易": [
        _spec("crude", "原油", "炼化原料成本", -1, 0.55, confidence="high"),
        _spec("fuel_oil", "燃料油", "油品产品价格代理", 1, 0.20, confidence="low", proxy=True),
        _spec("polypropylene", "聚丙烯", "化工产品价格代理", 1, 0.15, confidence="low", proxy=True),
        _spec("pta", "PTA", "化工产品价格代理", 1, 0.10, confidence="low", proxy=True),
    ],
    "基础化工": [_spec("crude", "原油", "原料成本代理", -1, 0.70, confidence="medium", proxy=True)],
    "化学原料": [
        _spec("crude", "原油", "原料成本代理", -1, 0.50, confidence="medium", proxy=True),
        _spec("methanol", "甲醇", "产品/原料代理", 1, 0.30, confidence="low", proxy=True),
        _spec("coal", "动力煤", "能源成本代理", -1, 0.20, confidence="low", proxy=True),
    ],
    "化学制品": [_spec("crude", "原油", "原料成本代理", -1, 0.75, confidence="low", proxy=True)],
    "化学纤维": [
        _spec("crude", "原油", "原料成本代理", -1, 0.55, confidence="medium", proxy=True),
        _spec("pta", "PTA", "化纤产品/原料代理", 1, 0.45, confidence="low", proxy=True),
    ],
    "塑料": [
        _spec("polypropylene", "聚丙烯", "产品价格", 1, 0.45, confidence="high"),
        _spec("polyethylene", "聚乙烯", "产品价格", 1, 0.35, confidence="high"),
        _spec("crude", "原油", "原料成本代理", -1, 0.20, confidence="medium", proxy=True),
    ],
    "橡胶": [
        _spec("rubber", "天然橡胶", "原料成本", -1, 0.75, confidence="high"),
        _spec("crude", "原油", "合成胶成本代理", -1, 0.25, confidence="low", proxy=True),
    ],
    "农化制品": [
        _spec("urea", "尿素", "产品价格", 1, 0.65, confidence="high"),
        _spec("coal", "动力煤", "能源成本代理", -1, 0.35, confidence="low", proxy=True),
    ],
    "交通运输": [
        _spec("fuel_oil", "燃料油", "燃料成本代理", -1, 0.60, confidence="low", proxy=True),
        _spec("crude", "原油", "燃料成本代理", -1, 0.40, confidence="low", proxy=True),
    ],
    "航空机场": [
        _spec("fuel_oil", "燃料油", "航空煤油成本代理", -1, 0.60, confidence="low", proxy=True),
        _spec("crude", "原油", "航空燃油成本代理", -1, 0.40, confidence="low", proxy=True),
    ],
    "航运港口": [_spec("fuel_oil", "燃料油", "燃料成本代理", -1, 0.70, confidence="low", proxy=True)],
    "物流": [
        _spec("fuel_oil", "燃料油", "运输燃料成本代理", -1, 0.60, confidence="low", proxy=True),
        _spec("crude", "原油", "运输燃料成本代理", -1, 0.40, confidence="low", proxy=True),
    ],
}


# Company overrides are used only where the business mix is substantially
# clearer than a broad SW-I bucket. They retain the SW taxonomy for audit.
STOCK_PROFILE_OVERRIDES: dict[str, dict[str, str]] = {
    "002714": {"profile": "养殖业", "reason": "牧原股份主营生猪养殖；生猪为产品端，豆粕/玉米为饲料成本，大豆为上游代理"},
    "000895": {"profile": "肉制品", "reason": "双汇发展主营肉制品加工；生猪作为原料成本代理，缺少现货加工价差时不直接推断利润"},
    "002460": {"profile": "能源金属", "reason": "赣锋锂业核心暴露为碳酸锂产品价格"},
    "002466": {"profile": "能源金属", "reason": "天齐锂业核心暴露为碳酸锂产品价格"},
    "600547": {"profile": "贵金属", "reason": "山东黄金主营黄金采选"},
    "600489": {"profile": "贵金属", "reason": "中金黄金主营黄金采选"},
    "600988": {"profile": "贵金属", "reason": "赤峰黄金主营黄金资源"},
    "601899": {"profile": "金铜资源", "reason": "紫金矿业为金铜复合资源，仅采用黄金/铜直接产品价格"},
    "600362": {"profile": "工业金属", "reason": "江西铜业核心产品为铜"},
    "603993": {"profile": "铜钼钴资源", "reason": "洛阳钼业为铜钼钴多金属资源；仅铜有直接期货，钼/钴显式标记不可观测，不用铝代理"},
    "601600": {"profile": "工业金属", "reason": "中国铝业核心产品为氧化铝/电解铝"},
    "000807": {"profile": "工业金属", "reason": "云铝股份核心产品为电解铝"},
    "600111": {"profile": "小金属", "reason": "北方稀土属于稀有金属，使用锡/铅/镍仅作低置信代理，缺失不补零"},
    "601225": {"profile": "煤炭", "reason": "陕西煤业主营煤炭"},
    "601088": {"profile": "煤炭", "reason": "中国神华主营煤炭与电力"},
    "600019": {"profile": "普钢", "reason": "宝钢股份主营普钢，区分钢材产品与铁矿/焦煤炉料成本"},
    "000708": {"profile": "特钢", "reason": "中信特钢主营特钢"},
    "601857": {"profile": "油气开采", "reason": "中国石油上游油气暴露"},
    "600028": {"profile": "炼化及贸易", "reason": "中国石化炼化一体化，原油为主要成本基准"},
}


_PROFILE_ALIASES = {
    "农林牧渔": "农林牧渔", "养殖业": "养殖业", "饲料": "饲料", "农产品加工": "农产品加工",
    "工业金属": "工业金属", "贵金属": "贵金属", "小金属": "小金属", "能源金属": "能源金属",
    "煤炭开采": "煤炭开采", "焦炭Ⅱ": "焦炭", "油气开采Ⅱ": "油气开采",
    "油服工程": "油服工程", "炼化及贸易": "炼化及贸易", "航空机场": "航空机场",
    "航运港口": "航运港口", "物流": "物流", "化学原料": "化学原料", "化学制品": "化学制品",
    "化学纤维": "化学纤维", "塑料": "塑料", "橡胶": "橡胶", "农化制品": "农化制品",
    "冶钢原料": "冶钢原料", "普钢": "普钢", "特钢Ⅱ": "特钢", "种植业": "种植业",
    "肉制品": "肉制品", "能源金属": "能源金属", "金铜资源": "金铜资源", "铜钼钴资源": "铜钼钴资源", "农产品加工": "农产品加工",
}


def _norm_code(value: object) -> str:
    text = str(value or "").strip().upper()
    text = re.sub(r"^(SH|SZ|BJ)", "", text)
    text = re.sub(r"\.(SH|SZ|BJ)$", "", text)
    return text.zfill(6) if text.isdigit() else text


def _clean_text(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def _load_stock_taxonomy() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    try:
        frame = pd.read_parquet(SW_MAP_PATH)
        required = {"code", "industry"}
        if not required.issubset(frame.columns):
            return {}
        for row in frame.to_dict("records"):
            code = _norm_code(row.get("code"))
            if re.fullmatch(r"\d{6}", code):
                result[code] = {
                    "code": code, "name": _clean_text(row.get("name")),
                    "sw1": _clean_text(row.get("industry")),
                    "sw1_code": _clean_text(row.get("industry_code")),
                    "sw2": None, "sw2_code": None,
                    "taxonomy_source": "data_warehouse/market/sw_industry_map.parquet",
                    "taxonomy_version": "SW-I warehouse snapshot",
                }
    except Exception:
        return {}

    # The updater may later produce a second-level constituent file. Accept
    # several provider schemas so one API column rename does not erase coverage.
    if SW_SECOND_CONS_PATH.exists():
        try:
            second = pd.read_parquet(SW_SECOND_CONS_PATH)
            code_col = next((c for c in ("code", "证券代码", "股票代码") if c in second.columns), None)
            name_col = next((c for c in ("industry2", "二级行业", "行业名称", "申万二级行业") if c in second.columns), None)
            code2_col = next((c for c in ("industry2_code", "二级行业代码", "行业代码") if c in second.columns), None)
            if code_col and name_col:
                for row in second.to_dict("records"):
                    code = _norm_code(row.get(code_col))
                    if code not in result:
                        continue
                    sw2 = _clean_text(row.get(name_col))
                    if sw2:
                        result[code]["sw2"] = sw2
                        result[code]["sw2_code"] = _clean_text(row.get(code2_col)) if code2_col else None
                        result[code]["taxonomy_source"] = "data_warehouse/industry/sw_second_cons.parquet"
                        result[code]["taxonomy_version"] = "SW-II warehouse snapshot"
        except Exception:
            pass
    return result


def resolve_stock_taxonomy(code: str) -> dict[str, Any]:
    code6 = _norm_code(code)
    row = _load_stock_taxonomy().get(code6, {
        "code": code6, "name": "", "sw1": "", "sw1_code": "", "sw2": None,
        "sw2_code": None, "taxonomy_source": "unavailable", "taxonomy_version": None,
    })
    override = STOCK_PROFILE_OVERRIDES.get(code6)
    if override:
        return {**row, "profile": override["profile"], "profile_reason": override["reason"], "profile_source": "listed_company_override"}
    raw_profile = row.get("sw2") or row.get("sw1")
    profile = _PROFILE_ALIASES.get(raw_profile, raw_profile)
    return {
        **row, "profile": profile, "profile_reason": "申万二级行业映射" if row.get("sw2") else "申万一级行业映射",
        "profile_source": "sw2" if row.get("sw2") else "sw1",
    }


def _load_series(tag: str) -> pd.DataFrame | None:
    path = MARKET_DIR / f"commodity__{tag}.parquet"
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
        date_col = next((c for c in ("date", "日期") if c in frame.columns), None)
        close_col = next((c for c in ("close", "收盘价", "动态结算价") if c in frame.columns), None)
        if not date_col or not close_col:
            return None
        out = pd.DataFrame({
            "date": pd.to_datetime(frame[date_col], errors="coerce").dt.normalize(),
            "price": pd.to_numeric(frame[close_col], errors="coerce"),
        })
        out = out.dropna().query("price > 0").drop_duplicates("date").sort_values("date")
        return out.reset_index(drop=True) if not out.empty else None
    except Exception:
        return None


def _score_return(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(50.0 + 50.0 * math.tanh(value / 0.10), 1)


def build_industry_commodity_evidence(code: str, lookback: int = 120,
                                      reference_date: date | str | None = None) -> dict[str, Any]:
    taxonomy = resolve_stock_taxonomy(code)
    profile = taxonomy.get("profile") or taxonomy.get("sw1")
    specs = INDUSTRY_COMMODITY_MAP.get(profile, [])
    reference = pd.Timestamp(reference_date).date() if reference_date else datetime.now().date()
    base = {
        "mapping_version": MAPPING_VERSION, "code": taxonomy.get("code"), "name": taxonomy.get("name"),
        "sw1": taxonomy.get("sw1"), "sw1_code": taxonomy.get("sw1_code"),
        "sw2": taxonomy.get("sw2"), "sw2_code": taxonomy.get("sw2_code"), "profile": profile,
        "profile_source": taxonomy.get("profile_source"), "profile_reason": taxonomy.get("profile_reason"),
        "taxonomy_source": taxonomy.get("taxonomy_source"), "taxonomy_version": taxonomy.get("taxonomy_version"),
        "source": "data_warehouse/market/commodity__<tag>.parquet",
        "reference_date": reference.isoformat(), "freshness_limit_days": COMMODITY_FRESHNESS_LIMIT_DAYS,
    }
    if not specs:
        return {**base, "status": "unmapped", "items": [], "coverage": 0.0,
                "available_weight": 0.0, "total_weight": 0.0, "score": None,
                "positive_exposure": 0.0, "negative_exposure": 0.0, "net_exposure": 0.0,
                "missing_items": [], "methodology": "无对应行业商品模板，不进入商品评分"}

    # 豆粕是养殖业的直接饲料成本，大豆只作为豆粕缺失时的上游替代代理。
    # 先判断豆粕是否可用，再构造有效权重，避免两个高度相关的成本腿重复计量。
    series_by_tag = {spec["tag"]: _load_series(spec["tag"]) for spec in specs}

    def _usable(series: pd.DataFrame | None) -> bool:
        if series is None or len(series) < 21:
            return False
        latest = series["date"].iloc[-1].date()
        return max(0, (reference - latest).days) <= COMMODITY_FRESHNESS_LIMIT_DAYS

    meal_is_usable = _usable(series_by_tag.get("meal"))
    effective_specs = [
        spec for spec in specs
        if not (spec["tag"] == "soybean" and meal_is_usable)
    ]
    total_weight = sum(float(item["weight_gross"]) for item in effective_specs)
    items: list[dict[str, Any]] = []
    signed_returns: list[float] = []
    available_weight = 0.0
    as_of_values: list[str] = []
    for spec in specs:
        tag = spec["tag"]
        meta = COMMODITY_SPECS.get(tag, {})
        item = {
            **spec, "commodity_id": tag, "contract_code": meta.get("contract"),
            "exchange": meta.get("exchange"), "source_file": f"data_warehouse/market/commodity__{tag}.parquet",
            "source_locator": f"data_warehouse/market/commodity__{tag}.parquet",
            "included_in_score": spec in effective_specs,
        }
        if tag == "soybean" and meal_is_usable:
            item.update({"status": "substitute_only", "as_of": None, "freshness_days": None,
                         "return_20d_pct": None, "latest_price": None, "aligned_days": 0,
                         "score": None, "contribution": None,
                         "reason": "豆粕直接成本可用，大豆仅作上游替代展示，不进入主评分"})
            items.append(item)
            continue
        if not meta.get("contract"):
            item.update({"status": "not_observable", "as_of": None, "freshness_days": None,
                         "return_20d_pct": None, "latest_price": None, "aligned_days": 0,
                         "score": None, "contribution": None, "reason_code": "no_direct_futures",
                         "reason": "该品种没有统一可观测的直接期货合约，不使用行业代理替代"})
            items.append(item)
            continue
        series = series_by_tag.get(tag)
        if series is None:
            item.update({"status": "missing", "as_of": None, "freshness_days": None,
                         "return_20d_pct": None, "latest_price": None, "aligned_days": 0,
                         "score": None, "contribution": None, "reason_code": "not_collected",
                         "reason": "统一采集器尚未落盘该品种，或文件字段不兼容"})
            items.append(item)
            continue
        latest = series["date"].iloc[-1]
        as_of = latest.date().isoformat()
        freshness_days = max(0, (reference - latest.date()).days)
        as_of_values.append(as_of)
        if len(series) < 21:
            item.update({"status": "insufficient", "as_of": as_of, "freshness_days": freshness_days,
                         "return_20d_pct": None, "latest_price": round(float(series["price"].iloc[-1]), 4),
                         "aligned_days": int(len(series)), "score": None, "contribution": None,
                         "reason": "有效历史少于21个交易日"})
            items.append(item)
            continue
        recent = series.tail(max(21, int(lookback)))
        ret20 = float(recent["price"].pct_change(20).iloc[-1])
        status = "stale" if freshness_days > COMMODITY_FRESHNESS_LIMIT_DAYS else "available"
        item.update({
            "status": status, "as_of": as_of, "freshness_days": freshness_days,
            "return_20d_pct": round(ret20 * 100, 2), "latest_price": round(float(series["price"].iloc[-1]), 4),
            "aligned_days": int(len(recent)), "score": _score_return(ret20),
            "contribution": round(float(spec["direction"]) * float(spec["weight_gross"]) * ret20, 6),
            "reason": "数据超过新鲜度阈值" if status == "stale" else None,
        })
        if status == "available" and item["included_in_score"]:
            available_weight += float(spec["weight_gross"])
            signed_returns.append(float(spec["direction"]) * float(spec["weight_gross"]) * ret20)
        items.append(item)

    score = None
    signed_return = None
    if available_weight > 0:
        signed_return = sum(signed_returns) / available_weight
        score = _score_return(signed_return)
    positive = sum(float(x["weight_gross"]) for x in items if x["direction"] > 0 and x["status"] == "available")
    negative = sum(float(x["weight_gross"]) for x in items if x["direction"] < 0 and x["status"] == "available")
    product_returns = [float(x["weight_gross"]) * float(x["return_20d_pct"]) / 100 for x in items
                      if x["direction"] > 0 and x["status"] == "available" and x.get("return_20d_pct") is not None]
    cost_returns = [float(x["weight_gross"]) * float(x["return_20d_pct"]) / 100 for x in items
                    if x["direction"] < 0 and x["status"] == "available" and x.get("return_20d_pct") is not None]
    missing_items = [x["name"] for x in items if x["status"] in {"missing", "stale", "insufficient", "not_observable"}]
    coverage = available_weight / total_weight if total_weight else 0.0
    status = "available" if coverage >= 0.999 else "partial" if available_weight > 0 else "missing"
    product_signal = sum(product_returns) / available_weight if available_weight and product_returns else None
    cost_signal = sum(cost_returns) / available_weight if available_weight and cost_returns else None
    # This is an economic margin proxy, not a company earnings forecast: product
    # price support minus feed/input cost pressure, with stale/missing legs excluded.
    return {
        **base, "status": status, "as_of": max(as_of_values) if as_of_values else None,
        "items": items, "coverage": round(coverage, 3),
        "available_weight": round(available_weight, 4), "total_weight": round(total_weight, 4),
        "score": score, "signed_return_20d": round(signed_return, 6) if signed_return is not None else None,
        "positive_exposure": round(positive, 4), "negative_exposure": round(negative, 4),
        "net_exposure": round(positive - negative, 4), "missing_items": missing_items,
        "product_signal_20d": round(product_signal * 100, 2) if product_signal is not None else None,
        "cost_pressure_20d": round(cost_signal * 100, 2) if cost_signal is not None else None,
        "profit_proxy_20d_pct": round(signed_return * 100, 2) if signed_return is not None else None,
        "profit_proxy_status": "proxy_only" if signed_return is not None else "unavailable",
        "profit_proxy_method": "产品端价格收益×产品暴露 - 饲料/原料价格收益×成本暴露；仅为期货价格代理，不等同公司利润或现货养殖利润",
        "methodology": "商品20日收益×业务方向×经济暴露权重；缺失/陈旧品种不补零，按可用权重重归一；大豆仅在豆粕直接成本不可用时作为替代代理，避免重复计量",
    }
