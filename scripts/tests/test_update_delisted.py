"""update_delisted.py 单元测试：全部 monkeypatch akshare，禁止真实网络。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import update_delisted as ud


class OfflineAk:
    """默认所有接口都失败，确保任何遗漏都会暴露而不是触发真实网络。"""

    def stock_info_sh_delist(self):
        raise ConnectionError("offline: stock_info_sh_delist")

    def stock_info_sz_delist(self):
        raise ConnectionError("offline: stock_info_sz_delist")

    def stock_zh_a_stop_em(self):
        raise ConnectionError("offline: stock_zh_a_stop_em")

    def stock_staq_net_stop(self):
        raise ConnectionError("offline: stock_staq_net_stop")


@pytest.fixture(autouse=True)
def offline_akshare(monkeypatch):
    monkeypatch.setattr(ud, "akshare", OfflineAk())


def _ak(**overrides):
    ak = OfflineAk()
    for name, func in overrides.items():
        setattr(ak, name, func)
    return ak


def test_sh_source_parsing_normalizes_fields(monkeypatch):
    monkeypatch.setattr(
        ud,
        "akshare",
        _ak(
            stock_info_sh_delist=lambda: pd.DataFrame(
                {
                    "公司代码": [600001],
                    "公司简称": ["PT水仙"],
                    "上市日期": ["1993-01-06"],
                    "暂停上市日期": ["2001-04-23"],
                }
            )
        ),
    )

    frame = ud._fetch_sh()

    assert frame.to_dict("records") == [
        {
            "code": "600001",
            "name": "PT水仙",
            "delist_date": "2001-04-23",
            "reason": "沪退市",
        }
    ]


def test_sz_failure_degrades_but_sh_still_succeeds(monkeypatch):
    sz_calls = {"count": 0}

    class FakeAk(OfflineAk):
        def stock_info_sh_delist(self):
            return pd.DataFrame(
                {
                    "公司代码": ["600002"],
                    "公司简称": ["退市样本"],
                    "暂停上市日期": ["2002-05-10"],
                }
            )

        def stock_info_sz_delist(self):
            sz_calls["count"] += 1
            raise ConnectionResetError("sz transient")

    monkeypatch.setattr(ud, "akshare", FakeAk())

    fresh, counts, errors = ud.fetch_all_sources()

    assert counts[ud.SOURCE_SH] == 1
    assert ud.SOURCE_SZ in errors
    assert len(fresh) == 1
    assert fresh.loc[0, "code"] == "600002"
    assert sz_calls["count"] == ud.SZ_RETRIES + 1


def test_merge_idempotent_dedupe_keeps_latest_record(monkeypatch, tmp_path):
    output = tmp_path / "delisted.parquet"
    existing = pd.DataFrame(
        {
            "code": ["000001", "000002"],
            "name": ["旧名称一", "旧名称二"],
            "delist_date": ["2010-01-01", "2011-01-01"],
            "reason": ["沪退市", "深退市"],
        }
    )
    existing.to_parquet(output, index=False)

    class FakeAk(OfflineAk):
        def stock_info_sh_delist(self):
            return pd.DataFrame(
                {
                    "公司代码": ["000001", "000003"],
                    "公司简称": ["新名称一", "新名称三"],
                    "暂停上市日期": ["2022-01-01", "2012-01-01"],
                }
            )

    monkeypatch.setattr(ud, "akshare", FakeAk())

    assert ud.run_update(output_path=output) == 0
    first = pd.read_parquet(output)
    assert len(first) == 3
    assert first["code"].is_unique
    assert first.loc[first["code"] == "000001", "name"].iloc[0] == "新名称一"
    assert first.loc[first["code"] == "000001", "delist_date"].iloc[0] == "2022-01-01"

    assert ud.run_update(output_path=output) == 0
    second = pd.read_parquet(output)
    assert len(second) == 3
    assert second["code"].is_unique


def test_missing_delist_date_column_is_tolerated(monkeypatch):
    monkeypatch.setattr(
        ud,
        "akshare",
        _ak(
            stock_info_sh_delist=lambda: pd.DataFrame(
                {"公司代码": ["000004"], "公司简称": ["无日期样本"]}
            )
        ),
    )

    frame = ud._fetch_sh()

    assert len(frame) == 1
    assert frame.loc[0, "delist_date"] is None
    assert frame.loc[0, "reason"] == "沪退市"


def test_dry_run_does_not_write_file(monkeypatch, tmp_path):
    output = tmp_path / "delisted.parquet"
    monkeypatch.setattr(
        ud,
        "akshare",
        _ak(
            stock_info_sh_delist=lambda: pd.DataFrame(
                {"公司代码": ["000005"], "公司简称": ["干跑样本"], "暂停上市日期": ["2003-01-01"]}
            )
        ),
    )

    code = ud.run_update(dry_run=True, output_path=output)

    assert code == 0
    assert not output.exists()


def test_all_sources_fail_returns_exit_code_2(monkeypatch, tmp_path):
    monkeypatch.setattr(ud, "DELISTED_FILE", tmp_path / "delisted.parquet")

    code = ud.main(["--dry-run"])

    assert code == 2
    assert not (tmp_path / "delisted.parquet").exists()
