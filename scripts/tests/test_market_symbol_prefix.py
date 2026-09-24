"""V12.3 审计 P2-1 回归: 北交所 920 前缀 + 统一交易所前缀规范。

覆盖 sources_all.market_prefix 与各脚本 symbol 转换(tencent/baostock),
确保 4/8/920→bj, 5/6/900→sh, 0/1/2/3→sz, 不再出现北交所标的错归深市。
"""
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
ROOT = SCRIPTS.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_system.sources_all import market_prefix  # noqa: E402


def test_market_prefix_bj():
    # 北交所: 4/8 开头, 或 920xxx 新段
    for code in ("430047", "830799", "832000", "920001", "920099"):
        assert market_prefix(code) == "bj", code


def test_market_prefix_sh():
    # 沪: 5(基金/ETF)/6(主板)/9(沪B/沪债) — 900xxx 沪B 归 sh
    for code in ("600519", "688001", "900901", "511880"):
        assert market_prefix(code) == "sh", code


def test_market_prefix_sz():
    # 深: 0/1/2/3
    for code in ("000001", "002594", "300750", "000858"):
        assert market_prefix(code) == "sz", code


def test_market_prefix_920_reboot():
    # 920 是"新段号"——4/8/920 三线都属北交所, 前缀规则须先匹配 920
    assert market_prefix("920007") == "bj"
    assert market_prefix("432001") == "bj"


@pytest.mark.parametrize("code,expect", [
    ("600519", "sh600519"), ("688001", "sh688001"),
    ("000001", "sz000001"), ("300750", "sz300750"),
    ("430047", "bj430047"), ("920001", "bj920001"),
])
def test_tencent_symbol(code, expect):
    from update_kline_tencent import tencent_symbol
    assert tencent_symbol(code) == expect, code


@pytest.mark.parametrize("code,expect", [
    ("600519", "sh.600519"), ("688001", "sh.688001"),
    ("000001", "sz.000001"), ("300750", "sz.300750"),
    ("430047", "bj.430047"), ("920001", "bj.920001"),
    ("900901", "sh.900901"),  # 沪B 原错归 sz
])
def test_bs_code(code, expect):
    from update_valuation_baostock import bs_code
    assert bs_code(code) == expect, code
