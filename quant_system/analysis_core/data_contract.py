"""analysis_core.data_contract — 单位统一层（V12.1 阶段5）。

物理量契约：
  data_warehouse 统一存储小数（0.01 表示 1%）。
  展示层出口格式化 %（format_pct）。
  本模块只提供转换工具 + 只读审计，不修改任何现有存储。

用法：
  # 工具
  to_decimal(50)  -> 0.5
  to_pct(0.5)     -> 50.0
  format_pct(0.5) -> "50.00%"

  # 审计 + 输出
  python3 -m quant_system.analysis_core.data_contract --scan
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # workspace 根
WAREHOUSE_DIR = ROOT / "data_warehouse"
GENERATED_DIR = ROOT / "generated" / "data_contract"
CST = timezone(timedelta(hours=8))

# 小数契约边界：>1.5 或 < -1.5 时，1% 的涨跌幅用小数不可能出现。
UNIT_LIMIT = 1.5
DEFAULT_KLINE_SAMPLE = 200

# 单位审计扫描的关键仓库数据集（相对 data_warehouse）。
KEY_DATASETS = (
    "market/zt_pool_history.parquet",
    "market/zt_pool_em_daily.parquet",
    "market/zt_daily_stats.parquet",
    "market/theme_cycle.parquet",
    "market/fund_forces.parquet",
    "market/fusion.parquet",
)

# 涨跌幅/收益率类候选列关键词。只审计这些列，避免把价格/成交额误判为百分数。
CONTRACT_COLUMN_TOKENS = (
    "pct_chg",
    "pct",
    "chg",
    "change",
    "return",
    "yield",
    "ret",
    "涨跌幅",
    "涨幅",
    "涨跌",
    "收益率",
    "回报",
    "收益",
)


def today() -> str:
    """当前日期（Asia/Shanghai）→ YYYY-MM-DD。"""
    return datetime.now(CST).date().isoformat()


def _float(value) -> Optional[float]:
    """安全转 float；None/非数值/非有限值返回 None。"""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def to_decimal(pct_value) -> Optional[float]:
    """百分数 → 存储小数。50 → 0.5；0 → 0.0；100 → 1.0；-100 → -1.0。

    无效输入返回 None，不抛异常，便于数据契约层显式标记问题值。
    """
    value = _float(pct_value)
    return None if value is None else value / 100.0


def to_pct(decimal_value) -> Optional[float]:
    """存储小数 → 百分数。0.5 → 50.0；0 → 0.0；1 → 100.0；-1 → -100.0。"""
    value = _float(decimal_value)
    return None if value is None else value * 100.0


def format_pct(decimal_value, decimals: int = 2) -> str:
    """展示层出口格式化：0.5 → '50.00%'。"""
    pct = to_pct(decimal_value)
    if pct is None:
        return "—"
    decimals = max(0, int(decimals))
    return f"{pct:.{decimals}f}%"


def format_pct_value(value, digits: int = 2, unit: Optional[str] = None) -> str:
    """按列级单位格式化百分比展示字符串。

    unit 参数：
      - 'pct'：输入已是百分数（30.0 → '30.00%'），直接格式化。
      - 'decimal'：输入为存储小数（0.5 → '50.00%'），先 ×100 再格式化。
      - None：无列级标注时回退启发式（|v|>UNIT_LIMIT 视为 pct，否则视为 decimal）。

    边界：None/NaN/非数值/非有限值 → 'n/a'；0 → '0.00%'；±100 按普通数值
    格式化，不截断/不溢出。"""
    number = _float(value)
    if number is None:
        return "n/a"
    digits = max(0, int(digits))

    normalized_unit = (str(unit).strip().lower() if unit is not None else "")
    if normalized_unit == "pct":
        pct = number
    elif normalized_unit == "decimal":
        pct = number * 100.0
    else:
        pct = number if abs(number) > UNIT_LIMIT else number * 100.0

    return f"{pct:.{digits}f}%"


def infer_column_unit(series: pd.Series, limit: float = UNIT_LIMIT) -> dict:
    """按值域分布众数判断列单位：'decimal' 或 'pct'。

    每个有效数值按启发式投票（|v|>limit → pct，否则 → decimal），取多数。
    平票时，若存在越界值则按 pct 处理（审计宁可标记疑似百分数存储），
    否则为 decimal。"""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"unit": "empty", "decimal_votes": 0, "pct_votes": 0, "value_count": 0}

    exceeded = numeric.abs() > limit
    decimal_votes = int((~exceeded).sum())
    pct_votes = int(exceeded.sum())
    if pct_votes > decimal_votes:
        unit = "pct"
    elif decimal_votes > pct_votes:
        unit = "decimal"
    else:
        unit = "pct" if pct_votes > 0 else "decimal"

    return {
        "unit": unit,
        "decimal_votes": decimal_votes,
        "pct_votes": pct_votes,
        "value_count": int(numeric.size),
    }


def is_contract_column(column: str) -> bool:
    """是否为涨跌幅/收益率契约候选列。"""
    if not isinstance(column, str):
        return False
    lowered = column.strip().lower()
    return any(token in lowered for token in CONTRACT_COLUMN_TOKENS)


def audit_column(series: pd.Series, column: str, dataset: str,
                 limit: float = UNIT_LIMIT) -> dict:
    """审计单个数值列：输出值域、疑似单位与建议。"""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "dataset": dataset,
            "column": column,
            "value_count": 0,
            "min": None,
            "max": None,
            "range": None,
            "unit": "empty",
            "suspected_unit": "empty",
            "suggestion": "无可用数值，跳过",
        }

    min_value = float(numeric.min())
    max_value = float(numeric.max())
    unit_info = infer_column_unit(numeric, limit=limit)
    unit = unit_info["unit"]
    exceeded = unit == "pct"
    suspected_unit = "percent" if unit == "pct" else "decimal"
    if unit == "pct":
        suggestion = "疑似百分数存储；建议 to_decimal(...) 统一为小数（0.01=1%）"
    else:
        suggestion = "符合存储小数契约（0.01=1%），展示层用 format_pct"

    return {
        "dataset": dataset,
        "column": column,
        "value_count": int(numeric.size),
        "min": min_value,
        "max": max_value,
        "range": [min_value, max_value],
        "unit": unit,
        "decimal_votes": unit_info["decimal_votes"],
        "pct_votes": unit_info["pct_votes"],
        "suspected_unit": suspected_unit,
        "suggestion": suggestion,
    }


def scan_file(path: Path, warehouse_root: Optional[Path] = None,
              limit: float = UNIT_LIMIT) -> list[dict]:
    """审计单个 parquet 文件中的契约候选列。"""
    warehouse_root = warehouse_root or WAREHOUSE_DIR
    try:
        root = warehouse_root.resolve()
        dataset = str(path.resolve().relative_to(root)) if root in path.resolve().parents else path.name
    except ValueError:
        dataset = path.name

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return [{
            "dataset": dataset,
            "column": None,
            "value_count": 0,
            "min": None,
            "max": None,
            "range": None,
            "unit": "read_error",
            "suspected_unit": "read_error",
            "suggestion": f"读取失败: {str(exc)[:120]}",
        }]

    records: list[dict] = []
    for column in df.columns:
        if is_contract_column(str(column)):
            records.append(audit_column(df[column], str(column), dataset, limit=limit))
    return records


def discover_parquet_files(warehouse_root: Optional[Path] = None,
                           include_kline_sample: bool = True,
                           max_kline_files: int = DEFAULT_KLINE_SAMPLE) -> list[Path]:
    """发现待审计 parquet：关键数据集 +（可选）K 线确定性等距抽样。"""
    root = warehouse_root or WAREHOUSE_DIR
    files: list[Path] = []
    seen: set[Path] = set()

    for rel in KEY_DATASETS:
        path = root / rel
        if path.exists():
            resolved = path.resolve()
            if resolved not in seen:
                files.append(path)
                seen.add(resolved)

    if include_kline_sample:
        kline_dir = root / "kline"
        if kline_dir.exists():
            klines = sorted(
                p for p in kline_dir.glob("*.parquet")
                if p.stem.isdigit() and len(p.stem) == 6
            )
            if max_kline_files > 0 and len(klines) > max_kline_files:
                step = max(1, len(klines) // max_kline_files)
                klines = klines[::step][:max_kline_files]
            for path in klines:
                resolved = path.resolve()
                if resolved not in seen:
                    files.append(path)
                    seen.add(resolved)

    return files


def scan_warehouse(warehouse_root: Optional[Path] = None,
                   paths: Optional[Iterable[Path | str]] = None,
                   include_kline_sample: bool = True,
                   max_kline_files: int = DEFAULT_KLINE_SAMPLE,
                   limit: float = UNIT_LIMIT,
                   date: Optional[str] = None) -> dict:
    """扫描 data_warehouse 关键 parquet 的涨跌幅/收益率列。

    无数据时不抛异常，返回 status='no_data' 的降级结果。
    """
    root = warehouse_root or WAREHOUSE_DIR
    scan_date = date or today()

    if paths is None:
        file_paths = discover_parquet_files(
            warehouse_root=root,
            include_kline_sample=include_kline_sample,
            max_kline_files=max_kline_files,
        )
    else:
        file_paths = [Path(p) for p in paths if Path(p).exists()]

    records: list[dict] = []
    read_errors: list[dict] = []
    for path in file_paths:
        if path.suffix.lower() != ".parquet":
            continue
        for record in scan_file(path, warehouse_root=root, limit=limit):
            if record.get("suspected_unit") == "read_error":
                read_errors.append(record)
            else:
                records.append(record)

    suspected = [record for record in records if record.get("suspected_unit") == "percent"]
    scanned_columns = len(records)
    suspicious_columns = len(suspected)
    decimal_columns = len([r for r in records if r.get("unit") == "decimal"])
    pct_columns = len([r for r in records if r.get("unit") == "pct"])

    status = "ok"
    if not file_paths or not records:
        status = "no_data"
    elif suspicious_columns:
        status = "mixed_or_suspect"

    datasets: dict[str, list[dict]] = {}
    for record in records:
        datasets.setdefault(record["dataset"], []).append(record)
    dataset_views = [
        {
            "dataset": dataset,
            "columns": rows,
            "suspicious_columns": [row for row in rows if row.get("suspected_unit") == "percent"],
        }
        for dataset, rows in datasets.items()
    ]

    return {
        "date": scan_date,
        "status": status,
        "unit_limit": limit,
        "storage_contract": "decimal",
        "summary": {
            "scanned_files": len(file_paths),
            "scanned_columns": scanned_columns,
            "suspicious_columns": suspicious_columns,
            "decimal_columns": decimal_columns,
            "pct_columns": pct_columns,
            "suspicious_datasets": len([d for d in dataset_views if d["suspicious_columns"]]),
            "read_errors": len(read_errors),
        },
        "datasets": dataset_views,
        "suspicious": suspected,
        "read_errors": read_errors,
    }


def _render_markdown(result: dict) -> str:
    lines: list[str] = [
        f"# Data Contract Audit {result['date']}",
        "",
        f"- status: {result['status']}",
        f"- storage_contract: {result['storage_contract']}",
        f"- unit_limit: {result['unit_limit']}",
        f"- scanned_files: {result['summary']['scanned_files']}",
        f"- scanned_columns: {result['summary']['scanned_columns']}",
        f"- decimal_columns: {result['summary'].get('decimal_columns', 0)}",
        f"- pct_columns: {result['summary'].get('pct_columns', 0)}",
        f"- suspicious_columns: {result['summary']['suspicious_columns']}",
        f"- suspicious_datasets: {result['summary']['suspicious_datasets']}",
        f"- read_errors: {result['summary']['read_errors']}",
        "",
        "## Suspected",
        "",
    ]
    if result["suspicious"]:
        lines.append("| dataset | column | unit | min | max | suggestion |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in result["suspicious"]:
            lines.append(
                f"| {row['dataset']} | {row['column']} | {row.get('unit', '')} | "
                f"{row['min']} | {row['max']} | {row['suggestion']} |"
            )
    else:
        lines.append("无疑似百分数存储。")
    lines += ["", "## Datasets", ""]

    for dataset_view in result["datasets"]:
        lines.append(f"### {dataset_view['dataset']}")
        lines.append("")
        lines.append("| column | value_count | min | max | unit | suspected_unit | suggestion |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in dataset_view["columns"]:
            lines.append(
                f"| {row['column']} | {row['value_count']} | {row['min']} | {row['max']} | "
                f"{row.get('unit', '')} | {row['suspected_unit']} | {row['suggestion']} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_report(result: dict, output_dir: Optional[Path] = None) -> dict:
    """将审计结果写入 generated/data_contract/YYYYMMDD/{json,md}。"""
    out_dir = output_dir or GENERATED_DIR / result["date"]
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"data_contract_{result['date']}.json"
    md_path = out_dir / f"data_contract_{result['date']}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}


def run_scan(warehouse_root: Optional[Path] = None,
             paths: Optional[Iterable[Path | str]] = None,
             include_kline_sample: bool = True,
             max_kline_files: int = DEFAULT_KLINE_SAMPLE,
             output_dir: Optional[Path] = None,
             date: Optional[str] = None) -> dict:
    """执行扫描并落盘 JSON/MD。"""
    result = scan_warehouse(
        warehouse_root=warehouse_root,
        paths=paths,
        include_kline_sample=include_kline_sample,
        max_kline_files=max_kline_files,
        date=date,
    )
    result["output"] = write_report(result, output_dir=output_dir)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="data_warehouse 单位契约审计")
    parser.add_argument("--scan", action="store_true", help="扫描关键 parquet 并输出 JSON/MD")
    parser.add_argument("--warehouse", type=Path, default=WAREHOUSE_DIR,
                        help="data_warehouse 根目录")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="输出目录（默认 generated/data_contract/YYYYMMDD）")
    parser.add_argument("--no-kline-sample", action="store_true",
                        help="跳过 K 线抽样，只扫描关键数据集")
    parser.add_argument("--max-kline-files", type=int, default=DEFAULT_KLINE_SAMPLE,
                        help="K 线等距抽样上限")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if not args.scan:
        _build_parser().print_help()
        raise SystemExit(0)

    result = run_scan(
        warehouse_root=args.warehouse,
        include_kline_sample=not args.no_kline_sample,
        max_kline_files=args.max_kline_files,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
