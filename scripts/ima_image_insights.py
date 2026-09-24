#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""研报共享图片内容解析（共享分析模块）—— 2026-08-25

扫描 data_warehouse/ima_export/media/ 下的原始研报图片（png/jpg/jpeg/webp/bmp），
调用现有 image-capable gpt-5.5（复用 vision_ocr_ima.py 的凭证与图片压缩逻辑）
完整 OCR 并识别“图片中实际展示的分析内容”，输出结构化 JSON 持久化到
data_warehouse/ima_export/media/vision_analysis.jsonl。

设计约束:
- 只依据图片实际内容，不编造数据；没有明确数据就留空。
- 每日只处理“未成功/文件已变更”的图片（process_pending 增量）。
- 模型不可用/失败只记录 error 状态，绝不阻断报告生成（本模块不在 import 时联网）。
- 不使用本地 OCR 冒充模型结果。

用法:
  python3 scripts/ima_image_insights.py [--date YYYY-MM-DD] [--limit N] [--scan]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CST = timezone(timedelta(hours=8))

IMG_SUFFIX = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_SKIP_DIRS = {"extracted"}


def default_media_root() -> Path:
    env = os.environ.get("IMA_MEDIA_ROOT")
    return Path(env) if env else ROOT / "data_warehouse" / "ima_export" / "media"


def default_analysis_path() -> Path:
    env = os.environ.get("IMA_ANALYSIS_PATH")
    if env:
        return Path(env)
    return default_media_root() / "vision_analysis.jsonl"


def _vision_ocr_module():
    """复用 vision_ocr_ima.py 的凭证/图片压缩逻辑（不硬编码密钥）。"""
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import vision_ocr_ima  # noqa: PLC0415
    return vision_ocr_ima


_MODEL_PROMPT = """你是研报图表分析助手。这张图片来自投研共享文件夹（可能是研报/新闻/会议纪要/概念股汇总等截图）。
请完成两件事：
1) 完整 OCR：逐字输出图片中出现的全部文字（标题、日期、正文、表格、图例、坐标轴标签、单位都保留；中文原样，英文保留；不要总结、不要省略）。
2) 识别“图片中实际展示的分析内容”并给出结构化 JSON。

输出必须是一个 JSON 对象（不要输出 JSON 之外的解释文字），字段如下：
{
  "title": "图片标题（取自图内标题，没有则空字符串）",
  "source_name": "来源机构/媒体名称，没有则空字符串",
  "source_url": "图片中可见的来源 URL，没有则空字符串",
  "source_date": "来源/日期（图内可见的机构名或日期，没有则空字符串）",
  "as_of_date": "数据截至日期（YYYY-MM-DD 或空）",
  "unit": "主要数值单位，没有则空字符串",
  "currency": "币种，没有则空字符串",
  "frequency": "数据频率（日/周/月/季/年/一次性或空）",
  "metric_basis": "指标口径/计算口径，没有则空字符串",
  "module": "所属板块/栏目（如 题材概念产业库/会议纪要，来自文件夹名或图内栏目名）",
  "visual_type": "资金流向/持仓/份额/成交额/趋势/行业/ETF/估值/事件/概念股/其他（只选一个最贴切的）",
  "ocr_text": "第1步的完整 OCR 全文",
  "summary": "两三句话描述图片实际展示的分析结果（只描述图中内容，不要写买卖建议）",
  "key_points": ["图中呈现的要点1", "要点2"],
  "metrics": [{"label": "指标名", "date": "日期或空", "value": 数值, "unit": "单位或空", "direction": "up/down/neutral 或空"}],
  "series": [{"name": "序列名", "points": [{"date": "2026-08-01", "value": 数值}]}],
  "flow": {"inflow": 数值或空, "outflow": 数值或空, "net": 数值或空, "source": "资金流向来源说明或空", "amount": 数值或空},
  "entities": ["图中涉及的标的/ETF/行业/机构名称"],
  "trend": "图中呈现的趋势描述（一句话，或空）",
  "risks": ["图中体现的风险点"],
  "confidence": 0到100的整数（你对识别结果的把握度，无法判断则空）,
  "uncertainties": ["图中无法看清/无法确认的部分"]
}

硬性规则：
- 只依据图片中实际可见的内容；图上没有的数字/日期/结论一律留空、空列表或 null，严禁编造。
- 图表类图片必须把每个可见的数据点放进 metrics 或 series，保留原始单位与涨跌方向。
- OCR 文本放在 ocr_text，不得省略长正文。
- 不要输出买卖建议、不要输出“建议买入/卖出/目标价”等推断。
"""


def scan_images(media_root: Path | None = None) -> list[str]:
    """扫描原始图片，跳过 extracted/ 与隐藏目录/文件，按相对路径去重。

    返回相对路径列表（正斜杠，如 "题材概念产业库/xxx.png"）。
    """
    root = Path(media_root) if media_root is not None else default_media_root()
    out: list[str] = []
    seen: set[str] = set()
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        parts = rel.parts
        if any(part in _SKIP_DIRS or part.startswith(".") for part in parts):
            continue
        if p.suffix.lower() not in IMG_SUFFIX:
            continue
        key = rel.as_posix()
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def load_analysis(path: Path | None = None) -> dict[str, dict]:
    """读 vision_analysis.jsonl → {相对路径: record}；损坏行跳过。"""
    p = Path(path) if path is not None else default_analysis_path()
    out: dict[str, dict] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(rec, dict) and rec.get("relative_path"):
                out[str(rec["relative_path"])] = rec
    return out


def write_analysis(records: dict[str, dict], path: Path | None = None) -> None:
    """整文件重写 jsonl（last-wins，保证错误状态也落盘且不重复堆积）。"""
    p = Path(path) if path is not None else default_analysis_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rel in sorted(records):
            f.write(json.dumps(records[rel], ensure_ascii=False) + "\n")
    tmp.replace(p)


def _extract_json(text: str) -> dict | None:
    """从模型输出里稳健提取 JSON 对象（容忍 ```json 包裹/前后杂文）。"""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:  # noqa: BLE001
            pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:  # noqa: BLE001
        return None


def call_vision(img_path: Path, timeout: int | None = None, retries: int = 3,
                prompt: str | None = None, chunk_index: int | None = None,
                image_data: tuple[str, bytes] | None = None) -> str:
    """调用共享视觉请求层；保留旧的 ``call_vision(path)`` 调用方式。"""
    vo = _vision_ocr_module()
    url, key = vo.get_credentials()
    if timeout is None:
        timeout = int(os.environ.get("IMA_VISION_TIMEOUT", "600"))
    mime, data = image_data or vo._image_b64(img_path)
    return vo._request_vision(url, key, mime, data, prompt or _MODEL_PROMPT,
                              timeout=timeout, retries=retries, chunk_index=chunk_index)


def _merge_unique(items: list, key_fn) -> list:
    out, seen = [], set()
    for item in items:
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _merge_chunk_records(records: list[dict], failures: list[dict], chunks: list) -> dict:
    """按块顺序合并结构化结果，重复 overlap 内容和数据点只保留一次。"""
    base = dict(records[0]) if records else {}
    base["ocr_text"] = "\n".join(r.get("ocr_text", "") for r in records if r.get("ocr_text"))
    base["key_points"] = _merge_unique([x for r in records for x in r.get("key_points", [])], lambda x: str(x).strip())
    base["entities"] = _merge_unique([x for r in records for x in r.get("entities", [])], lambda x: str(x).strip())
    base["risks"] = _merge_unique([x for r in records for x in r.get("risks", [])], lambda x: str(x).strip())
    base["uncertainties"] = _merge_unique([x for r in records for x in r.get("uncertainties", [])], lambda x: str(x).strip())
    base["metrics"] = _merge_unique([x for r in records for x in r.get("metrics", [])],
                                     lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
    series = {}
    for r in records:
        for item in r.get("series", []):
            name = item.get("name", "")
            target = series.setdefault(name, {"name": name, "points": []})
            target["points"].extend(item.get("points", []))
    base["series"] = []
    for item in series.values():
        item["points"] = _merge_unique(item["points"], lambda x: (x.get("date"), x.get("value")))
        base["series"].append(item)
    flows = [r.get("flow", {}) for r in records]
    base["flow"] = dict(next((f for f in flows if any(v is not None for v in f.values())), {}),)
    base["chunk_count"] = len(chunks)
    base["chunks"] = [{"index": c.index, "total": c.total, "y_start": c.y_start,
                       "y_end": c.y_end, "payload_bytes": c.payload_bytes,
                       "status": "ok" if i < len(records) else "error"}
                      for i, c in enumerate(chunks)]
    if failures:
        base["status"] = "partial"
        base["renderable"] = False
        base.setdefault("quality_issues", []).append(
            "未完成分块: " + ", ".join(str(f["index"]) for f in failures))
        base["chunk_errors"] = failures
    return base


def _chunk_prompt(index: int, total: int) -> str:
    return _MODEL_PROMPT + f"\n这是长图第 {index + 1}/{total} 个纵向分块。只识别本分块实际可见内容，严格输出上述 JSON；不要补全相邻分块不可见的内容。"


def _number_or_none(value):
    """接受 JSON 数值（排除 bool/非有限值），其余统一为空。"""
    if isinstance(value, bool) or value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        try:
            return value if __import__("math").isfinite(value) else None
        except (TypeError, ValueError):
            return None
    return None


def _number_field(value, field, issues):
    if value not in (None, "") and _number_or_none(value) is None:
        issues.append(f"{field} 不是数值")
    return _number_or_none(value)


def _text_or_empty(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalize_and_validate(parsed: dict) -> dict:
    """规范化模型输出，并把不可重绘原因显式写入记录。"""
    if not isinstance(parsed, dict):
        raise ValueError("模型输出不是 JSON 对象")
    text_fields = ("title", "source_name", "source_url", "source_date", "as_of_date",
                   "unit", "currency", "frequency", "metric_basis", "module", "visual_type",
                   "ocr_text", "summary", "trend")
    rec = {field: _text_or_empty(parsed.get(field)) for field in text_fields}
    list_fields = ("key_points", "entities", "risks", "uncertainties")
    for field in list_fields:
        value = parsed.get(field, [])
        rec[field] = [x.strip() if isinstance(x, str) else str(x) for x in value] if isinstance(value, list) else []

    issues = []
    metrics = parsed.get("metrics", [])
    if not isinstance(metrics, list):
        issues.append("metrics 不是 list")
        metrics = []
    normalized_metrics = []
    for item in metrics:
        if not isinstance(item, dict):
            issues.append("metrics 含非对象项")
            continue
        normalized_metrics.append({
            "label": _text_or_empty(item.get("label")), "date": _text_or_empty(item.get("date")),
            "value": _number_field(item.get("value"), "metrics.value", issues), "unit": _text_or_empty(item.get("unit")),
            "direction": _text_or_empty(item.get("direction")),
        })
    rec["metrics"] = normalized_metrics

    series = parsed.get("series", [])
    if not isinstance(series, list):
        issues.append("series 不是 list")
        series = []
    normalized_series = []
    for item in series:
        if not isinstance(item, dict):
            issues.append("series 含非对象项")
            continue
        points = item.get("points")
        if not isinstance(points, list):
            issues.append("series.points 不是 list")
            continue
        clean_points = []
        for point in points:
            if not isinstance(point, dict):
                issues.append("series.points 含非对象项")
                continue
            date = _text_or_empty(point.get("date"))
            value = _number_field(point.get("value"), "series.points.value", issues)
            if not date or value is None:
                issues.append("series.points 的 date/value 不成对")
                continue
            clean_points.append({"date": date, "value": value})
        normalized_series.append({"name": _text_or_empty(item.get("name")), "points": clean_points})
    rec["series"] = normalized_series

    flow = parsed.get("flow")
    if flow is None:
        flow = {}
    if not isinstance(flow, dict):
        issues.append("flow 不是对象")
        flow = {}
    rec["flow"] = {key: _number_field(flow.get(key), f"flow.{key}", issues) for key in ("inflow", "outflow", "net", "amount")}
    for key in ("source", "unit", "currency", "basis"):
        rec["flow"][key] = _text_or_empty(flow.get(key))

    confidence = _number_field(parsed.get("confidence"), "confidence", issues)
    if confidence is not None and not 0 <= confidence <= 100:
        issues.append("confidence 超出 0-100")
        confidence = None
    rec["confidence"] = confidence

    if not rec["title"] or not rec["source_date"]:
        rec["status"] = "partial"
    else:
        rec["status"] = "ok"
    if not rec["as_of_date"]:
        issues.append("缺 as_of_date/日期")
    has_unit = bool(rec["unit"] or rec["currency"] or any(m["unit"] for m in rec["metrics"]) or rec["flow"].get("unit") or rec["flow"].get("currency"))
    if not has_unit:
        issues.append("缺单位或币种")
    if not rec["series"] or any(len(s["points"]) < 2 for s in rec["series"]):
        issues.append("序列点少于 2")
    if confidence is None or confidence < 60:
        issues.append("置信度低于 60 或缺失")
    rec["quality_issues"] = list(dict.fromkeys(issues))
    rec["renderable"] = not rec["quality_issues"]
    return rec


def analyze_image(rel_path: str, media_root: Path | None = None,
                  analyzer=None, date: str | None = None) -> dict:
    """识别单张图片 → 结构化 record。模型失败也返回 error 状态 record（不抛异常）。"""
    root = Path(media_root) if media_root is not None else default_media_root()
    img = root / rel_path
    now = datetime.now(CST)
    rec: dict = {
        "relative_path": rel_path,
        "status": "error",
        "error": "",
        "analyzed_at": now.isoformat(timespec="seconds"),
        "report_date": date or now.strftime("%Y-%m-%d"),
    }
    try:
        st = img.stat()
        rec["file_size"] = st.st_size
        rec["file_mtime"] = round(st.st_mtime, 3)
    except OSError as e:
        rec["error"] = f"图片不可读: {e}"
        return rec
    try:
        if analyzer is not None:
            # 测试注入 analyzer 保持旧契约：单函数、可接收任意测试输入。
            parsed = _extract_json(analyzer(img))
            normalized = normalize_and_validate(parsed)
            rec.update(normalized)
        else:
            vo = _vision_ocr_module()
            chunks = vo.image_chunks(img)
            parsed_chunks, failures = [], []
            for chunk in chunks:
                try:
                    content = call_vision(img, timeout=int(os.environ.get("IMA_VISION_TIMEOUT", "600")),
                                          retries=int(os.environ.get("IMA_VISION_RETRIES", "3")),
                                          prompt=_chunk_prompt(chunk.index, chunk.total),
                                          chunk_index=chunk.index,
                                          image_data=(chunk.mime, chunk.data))
                    parsed = _extract_json(content)
                    parsed_chunks.append(normalize_and_validate(parsed))
                except Exception as exc:  # 保留失败块，不把它当成功
                    failures.append({"index": chunk.index, "payload_bytes": chunk.payload_bytes,
                                     "error": str(exc)[:500]})
            if not parsed_chunks:
                raise RuntimeError("所有分块失败: " + json.dumps(failures, ensure_ascii=False))
            rec.update(_merge_chunk_records(parsed_chunks, failures, chunks))
            if not failures and not rec.get("source_date"):
                rec["status"] = "partial"
    except Exception as e:  # noqa: BLE001 —— 单张失败只留错误状态
        rec["error"] = str(e)[:1000]
        rec["status"] = "error"
    return rec


def process_pending(date: str | None = None, limit: int = 0,
                    media_root: Path | None = None, analysis_path: Path | None = None,
                    analyzer=None, verbose: bool = True) -> dict:
    """增量处理：只处理未成功 / 文件已变更的图片。

    date: 记录本次分析所属的报告日期（写入 record.report_date）。
    limit: >0 时最多处理前 N 张（便于分批/测试）。
    analyzer: 可注入的模型调用函数（测试用）；None 用真实 gpt-5.5。
    任何模型/网络异常都只会落到 error 状态，不向调用方抛异常。
    """
    root = Path(media_root) if media_root is not None else default_media_root()
    out_path = Path(analysis_path) if analysis_path is not None else default_analysis_path()
    records = load_analysis(out_path)
    rels = scan_images(root)

    pending: list[str] = []
    for rel in rels:
        rec = records.get(rel)
        img = root / rel
        try:
            st = img.stat()
        except OSError:
            continue
        if rec and rec.get("status") == "ok" \
                and rec.get("file_size") == st.st_size \
                and abs((rec.get("file_mtime") or 0) - st.st_mtime) < 0.01:
            continue  # 已成功且未变更
        pending.append(rel)
    if limit and limit > 0:
        pending = pending[:limit]

    ok = 0
    for rel in pending:
        t0 = time.time()
        rec = analyze_image(rel, media_root=root, analyzer=analyzer, date=date)
        records[rel] = rec
        if rec["status"] == "ok":
            ok += 1
        if verbose:
            print(f"[ima_image_insights] {rec['status']:>5} {rel} "
                  f"({time.time() - t0:.0f}s)"
                  + (f" error={rec['error'][:120]}" if rec["error"] else ""), flush=True)
    if pending:
        write_analysis(records, out_path)
    return {
        "scanned": len(rels),
        "pending": len(pending),
        "ok": ok,
        "failed": len(pending) - ok,
        "records": len(records),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="研报共享图片内容解析（gpt-5.5 看图）")
    ap.add_argument("--date", default=None, help="报告日期 YYYY-MM-DD（写入 record.report_date）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理前 N 张待识别图片")
    ap.add_argument("--scan", action="store_true", help="只打印待识别清单，不调用模型")
    args = ap.parse_args()

    if args.scan:
        root = default_media_root()
        records = load_analysis()
        rels = scan_images(root)
        pending = []
        for rel in rels:
            rec = records.get(rel)
            try:
                st = (root / rel).stat()
            except OSError:
                continue
            if rec and rec.get("status") == "ok" \
                    and rec.get("file_size") == st.st_size \
                    and abs((rec.get("file_mtime") or 0) - st.st_mtime) < 0.01:
                continue
            pending.append(rel)
        print(f"共 {len(rels)} 张原始图片 · 待处理 {len(pending)} 张")
        for rel in pending:
            print("  PENDING", rel)
        return 0

    res = process_pending(date=args.date, limit=args.limit)
    print(f"[ima_image_insights] 完成: 扫描 {res['scanned']} · 待处理 {res['pending']} · "
          f"成功 {res['ok']} · 失败 {res['failed']} · 累计记录 {res['records']}")
    # 模型不可用/失败不阻断报告生成 → 恒返回 0（管线侧另行记 warning）。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
