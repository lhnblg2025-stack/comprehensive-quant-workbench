#!/usr/bin/env python3
"""随机 UI 输出测试：枚举 quant_web 全部页面/API 路由，随机采样请求并断言有输出。

- 自动解析 quant_web/server.py 的路由表，避免手工维护测试用例与代码脱节。
- 页面必须 200 且返回 HTML；API 只要 <500 且有非空响应体即视为“能输出”。
- 每个用例的请求、状态码、响应体长度、前 120 字符片段都会落盘保存。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "quant_web" / "server.py"
STATIC = ROOT / "quant_web" / "static"
BASE_URL = "http://127.0.0.1:8600"
TIMEOUT = 30.0

PUBLIC_PREFIXES = (
    "/api/health", "/api/v1/health", "/api/v4/health", "/api/v11/health",
    "/api/livez", "/api/readyz", "/api/version", "/api/auth_status",
)

QUERY_POOL = [
    "", "?code=600519", "?symbol=600519", "?date=2026-08-21", "?name=茅台",
    "?limit=5", "?days=10",
]


def _extract_routes() -> list[str]:
    text = SERVER.read_text(encoding="utf-8")
    exact = re.findall(r'parsed\.path == "([^"]+)"', text)
    prefix = re.findall(r'parsed\.path\.startswith\("([^"]+)"', text)
    routes = sorted(set(exact + prefix))
    return routes


def _static_pages() -> list[str]:
    pages = ["/"]
    if STATIC.exists():
        for p in sorted(STATIC.glob("*.html")):
            pages.append("/" + p.name)
    # server.py 里显式派发的别名页
    for alias in ("/base_panorama.html", "/v4.html", "/review.html", "/report.html"):
        if alias not in pages:
            pages.append(alias)
    return pages


def _classify(routes: list[str], pages: list[str]) -> dict:
    public = [r for r in routes if r in PUBLIC_PREFIXES]
    protected = [r for r in routes if r.startswith("/api/") and r not in PUBLIC_PREFIXES]
    return {"pages": pages, "public_api": public, "protected_api": protected}


def _request(path: str, timeout: float = TIMEOUT) -> dict:
    url = BASE_URL + path
    headers = {"User-Agent": "quant-ui-random-test/1.0"}
    api_key = os.environ.get("QUANT_WEB_API_KEY", "")
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(url, headers=headers)
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            ctype = resp.headers.get("Content-Type", "")
            if "text/event-stream" in ctype:
                # SSE 是持续流：读取前几行即视为“有输出”，不等待流结束。
                lines = []
                for _ in range(4):
                    line = resp.readline()
                    if not line:
                        break
                    lines.append(line)
                body = b"".join(lines)
            else:
                body = resp.read(4096 * 4)
    except urllib.error.HTTPError as e:
        body = e.read(4096 * 4) or b""
        status = e.code
        ctype = e.headers.get("Content-Type", "") if e.headers else ""
    except Exception as e:  # noqa: BLE001
        return {
            "path": path, "status": None, "content_type": "",
            "body_len": 0, "ok": False, "output_present": False,
            "error": f"{type(e).__name__}: {str(e)[:160]}", "latency_ms": int((time.time() - started) * 1000),
        }
    text = body.decode("utf-8", errors="replace")
    return {
        "path": path, "status": status, "content_type": ctype,
        "body_len": len(body), "ok": status is not None and status < 500 and len(body) > 0,
        "output_present": len(body) > 0,
        "snippet": text[:120].replace("\n", " "),
        "latency_ms": int((time.time() - started) * 1000),
    }


def build_cases(sample_seed: int, protected_sample: int | None) -> tuple[dict, list[dict]]:
    routes = _extract_routes()
    pages = _static_pages()
    classified = _classify(routes, pages)

    # 页面 + 公开 API 全量；受保护 API 随机采样以控制总时长。
    rng = random.Random(sample_seed)
    chosen = list(classified["pages"]) + list(classified["public_api"])
    protected = list(classified["protected_api"])
    if protected_sample is None or protected_sample >= len(protected):
        chosen += protected
    else:
        chosen += rng.sample(protected, protected_sample)

    cases = []
    for path in chosen:
        # 页面不注入随机参数，API 注入随机查询串模拟随机输入。
        is_page = not path.startswith("/api/")
        if is_page:
            final_path = path
        else:
            final_path = path + urllib.parse.quote(rng.choice(QUERY_POOL), safe="?&=")
        cases.append({"path": final_path, "kind": "page" if is_page else ("public_api" if path in PUBLIC_PREFIXES else "protected_api")})

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": BASE_URL,
        "seed": sample_seed,
        "total_routes_discovered": len(routes),
        "pages_discovered": len(pages),
        "public_api_count": len(classified["public_api"]),
        "protected_api_count": len(classified["protected_api"]),
        "protected_api_sampled": len([c for c in cases if c["kind"] == "protected_api"]),
        "sampled_total": len(cases),
    }
    return meta, cases


def run(sample_seed: int = 20260824, protected_sample: int | None = None, save: bool = True,
        workers: int = 8, timeout: float = TIMEOUT) -> dict:
    meta, cases = build_cases(sample_seed, protected_sample)
    paths = [c["path"] for c in cases]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        raw = list(ex.map(lambda p: _request(p, timeout=timeout), paths))
    results = []
    failures = []
    for case, r in zip(cases, raw):
        r["kind"] = case["kind"]
        results.append(r)
        is_page = case["kind"] == "page"
        if (is_page and r["status"] != 200) or (not r["ok"]):
            failures.append(r)
    results.sort(key=lambda r: r["path"])

    summary = {
        "generated_at": meta["generated_at"],
        "seed": sample_seed,
        "sampled_total": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "page_failures": [r for r in failures if r["kind"] == "page"],
        "api_failures": [r for r in failures if r["kind"] != "page"],
        "route_discovery": meta,
    }
    payload = {"summary": summary, "cases": results}

    if save:
        out_dir = ROOT / "generated" / "ui_test_cases"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        (out_dir / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / f"cases_{stamp}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"saved": str(out_dir / f"cases_{stamp}.json"), "summary": summary}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="随机 UI 输出测试")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--protected-sample", type=int, default=None, help="受保护 API 采样数（缺省=全部）")
    ap.add_argument("--workers", type=int, default=8, help="并发请求数")
    ap.add_argument("--timeout", type=float, default=TIMEOUT, help="单请求超时秒数")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    payload = run(sample_seed=args.seed, protected_sample=args.protected_sample,
                  save=not args.no_save, workers=args.workers, timeout=args.timeout)
    return 0 if payload["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
