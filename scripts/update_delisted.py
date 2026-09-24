#!/usr/bin/env python3
"""退市股数据抓取与增量合并（幸存者偏差数据层）。

数据仓库中的 ``data_warehouse/delisted.parquet`` 长期为空，导致回测只覆盖
存活至今的股票。本脚本从 akshare 抓取沪深退市/暂停上市清单，按 ``code``
去重后与既有数据合并写回，幂等可重复执行。

数据源：
  - 沪市退市：``stock_info_sh_delist``
  - 深市退市：``stock_info_sz_delist``（失败重试 3 次）
  - 暂停上市：``stock_zh_a_stop_em``
  - 东财退市：``stock_staq_net_stop``（可选补充源，失败跳过）
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Callable, Any

import pandas as pd

try:
    import akshare
except Exception:  # pragma: no cover - 环境缺少 akshare 时由抓取函数降级
    akshare = None

LOG = logging.getLogger("update_delisted")

ROOT = Path(__file__).resolve().parent.parent
DELISTED_FILE = ROOT / "data_warehouse" / "delisted.parquet"
COLUMNS = ["code", "name", "delist_date", "reason"]

SOURCE_SH = "sh"
SOURCE_SZ = "sz"
SOURCE_STOP = "stop"
SOURCE_EM = "em"

SOURCE_LABELS = {
    SOURCE_SH: "沪退市",
    SOURCE_SZ: "深退市",
    SOURCE_STOP: "暂停上市",
    SOURCE_EM: "东财",
}

CODE_KEYS = ["公司代码", "证券代码", "代码", "股票代码"]
NAME_KEYS = ["公司简称", "证券简称", "名称", "股票简称"]
DATE_KEYS = ["终止上市日期", "退市日期", "摘牌日期", "暂停上市日期", "日期"]

SZ_RETRIES = 3


def _empty_table() -> pd.DataFrame:
    """返回统一 schema 的空表。"""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in COLUMNS})


def _pick_column(df: pd.DataFrame, keys: list[str]) -> str | None:
    """返回首个存在的候选列名，无匹配返回 None。"""
    for key in keys:
        if key in df.columns:
            return key
    return None


def _normalize_code(value: Any) -> str:
    """将 akshare/交易所代码归一化为 6 位数字字符串。"""
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"(?i)^(sh|sz|bj)[._-]?", "", text)
    if "." in text:
        text = text.split(".", 1)[0]
    digits = re.sub(r"\D", "", text)
    if not digits:
        return ""
    if len(digits) > 6:
        digits = digits[-6:]
    return digits.zfill(6)


def _text(value: Any) -> str:
    """将单元格安全转为去空格文本，空值返回空字符串。"""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _format_dates(values: Any) -> list[str | None]:
    """统一转成 YYYY-MM-DD 字符串，无法解析/缺失返回 None。"""
    if values is None:
        return []
    parsed = pd.to_datetime(pd.Series(values), errors="coerce")
    return [None if pd.isna(item) else item.strftime("%Y-%m-%d") for item in parsed]


def _coerce_table(df: pd.DataFrame) -> pd.DataFrame:
    """补齐/规范 parquet 与抓取结果的 4 个标准字段。"""
    if df is None or df.empty:
        return _empty_table()

    out = pd.DataFrame(index=df.index)
    for column in COLUMNS:
        out[column] = df[column] if column in df.columns else None

    out["code"] = out["code"].map(_normalize_code)
    out["name"] = out["name"].map(_text)
    out["delist_date"] = _format_dates(out["delist_date"])
    out["reason"] = out["reason"].map(_text)
    out = out[out["code"].str.len() == 6].reset_index(drop=True)
    return out[COLUMNS]


def _normalize_frame(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    """按来源字段名归一化原始 DataFrame。"""
    if raw is None or raw.empty:
        return _empty_table()

    df = raw.copy()
    code_col = _pick_column(df, CODE_KEYS)
    if code_col is None:
        LOG.warning("[%s] 未找到代码列，跳过该来源；实际列: %s", source, list(df.columns))
        return _empty_table()

    name_col = _pick_column(df, NAME_KEYS)
    date_col = _pick_column(df, DATE_KEYS)

    normalized = pd.DataFrame(
        {
            "code": df[code_col],
            "name": df[name_col] if name_col else "",
            "delist_date": df[date_col] if date_col else None,
            "reason": SOURCE_LABELS.get(source, source),
        }
    )
    return _coerce_table(normalized)


def _akshare() -> Any:
    if akshare is None:
        raise RuntimeError("akshare 未安装，无法抓取退市股数据")
    return akshare


def _retry_call(func: Callable[[], pd.DataFrame], retries: int) -> pd.DataFrame:
    """执行接口调用；失败时按指定次数重试。"""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - 网络/解析错误统一降级
            last_error = exc
            LOG.warning("接口调用失败（第 %s/%s 次）: %s: %s", attempt + 1, retries + 1, type(exc).__name__, exc)
    assert last_error is not None
    raise last_error


def _fetch_sh() -> pd.DataFrame:
    """抓取沪市退市清单。"""
    raw = _akshare().stock_info_sh_delist()
    return _normalize_frame(raw, SOURCE_SH)


def _fetch_sz() -> pd.DataFrame:
    """抓取深市退市清单；偶发 ConnectionReset 时重试。"""
    raw = _retry_call(_akshare().stock_info_sz_delist, retries=SZ_RETRIES)
    return _normalize_frame(raw, SOURCE_SZ)


def _fetch_stop() -> pd.DataFrame:
    """抓取东方财富暂停上市/两网及退市清单。"""
    raw = _akshare().stock_zh_a_stop_em()
    return _normalize_frame(raw, SOURCE_STOP)


def _fetch_eastmoney() -> pd.DataFrame:
    """抓取东方财富退市板块；接口缺失/失败由外层降级。"""
    ak = _akshare()
    func = getattr(ak, "stock_staq_net_stop", None)
    if func is None:
        raise RuntimeError("akshare 缺少 stock_staq_net_stop 接口")
    return _normalize_frame(func(), SOURCE_EM)


SOURCE_FUNCTIONS: list[tuple[str, str, Callable[[], pd.DataFrame]]] = [
    (SOURCE_SH, SOURCE_LABELS[SOURCE_SH], _fetch_sh),
    (SOURCE_SZ, SOURCE_LABELS[SOURCE_SZ], _fetch_sz),
    (SOURCE_STOP, SOURCE_LABELS[SOURCE_STOP], _fetch_stop),
    (SOURCE_EM, SOURCE_LABELS[SOURCE_EM], _fetch_eastmoney),
]


def fetch_all_sources() -> tuple[pd.DataFrame, dict[str, int], dict[str, str]]:
    """抓取全部来源并合并，任一来源失败不影响其他来源。"""
    frames: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    errors: dict[str, str] = {}

    for source, label, func in SOURCE_FUNCTIONS:
        try:
            frame = func()
            count = len(frame)
            counts[source] = count
            LOG.info("[%s] %s: %s 条", source, label, count)
            if count:
                frames.append(frame)
        except Exception as exc:  # noqa: BLE001 - 单源失败降级不中断
            errors[source] = f"{type(exc).__name__}: {exc}"
            LOG.warning("[%s] %s 抓取失败，已跳过: %s", source, label, errors[source])

    if not frames:
        fresh = _empty_table()
    else:
        fresh = pd.concat(frames, ignore_index=True)
    return fresh, counts, errors


def _load_existing(path: Path) -> pd.DataFrame:
    """读取既有退市表；缺失/损坏按空表处理。"""
    if not path.exists():
        return _empty_table()
    try:
        raw = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("读取既有退市表失败，按空表处理: %s: %s", type(exc).__name__, exc)
        return _empty_table()
    return _coerce_table(raw)


def _merge_delisted(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """与既有表合并；按 code 去重并保留日期更新/抓取更新的记录。"""
    combined = pd.concat([_coerce_table(existing), _coerce_table(fresh)], ignore_index=True)
    if combined.empty:
        return _empty_table()

    date_rank = pd.to_datetime(combined["delist_date"], errors="coerce")
    combined["_date_rank"] = date_rank
    combined = combined.sort_values("_date_rank", na_position="first", kind="stable")
    combined = combined.drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)
    return combined[COLUMNS]


def _print_summary(total: int, new_count: int, counts: dict[str, int], errors: dict[str, str]) -> None:
    print(f"新增: {new_count}")
    print(f"总数: {total}")
    print("各来源数量:")
    for source, label in SOURCE_LABELS.items():
        if source in errors:
            print(f"  - {label}: 失败（{errors[source]}）")
        else:
            print(f"  - {label}: {counts.get(source, 0)}")


def run_update(dry_run: bool = False, output_path: Path | None = None) -> int:
    """执行抓取/合并/写回；返回 0 成功、2 全部数据源失败。"""
    path = Path(output_path) if output_path is not None else DELISTED_FILE
    existing = _load_existing(path)
    fresh, counts, errors = fetch_all_sources()

    all_sources_failed = len(errors) == len(SOURCE_FUNCTIONS)
    if all_sources_failed:
        print("退市股数据更新失败：全部数据源均抓取失败，未写文件（外部网络/接口限制）。")
        _print_summary(len(existing), 0, counts, errors)
        return 2

    if fresh.empty:
        final = existing
        new_count = 0
    else:
        final = _merge_delisted(existing, fresh)
        existing_codes = set(existing["code"]) if not existing.empty else set()
        new_count = len(set(final["code"]) - existing_codes)

    _print_summary(len(final), new_count, counts, errors)
    if dry_run:
        print(f"dry-run：未写文件 {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        final.to_parquet(path, index=False)
        print(f"已写文件: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="抓取并合并沪深退市股数据")
    parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    args = parser.parse_args(argv)
    return run_update(dry_run=args.dry_run)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
