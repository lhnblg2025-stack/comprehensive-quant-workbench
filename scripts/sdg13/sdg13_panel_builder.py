#!/usr/bin/env python3
"""Build SDG 13 country-year panel datasets (reproducible).

Produces 5 CSVs directly from authoritative APIs plus the existing
NDC probe evidence file.  No large intermediate JSON blobs saved to
disk, so it runs in under 2 minutes with ~300 MB RAM.

Outputs (under OUTPUT_DIR):
  13_2_2_ghg_excluding_lulucf.csv      – WDI EN.GHG.ALL.MT.CE.AR5
  ndc_first_submission_dummy.csv       – 0/1 panel derived from probe evidence
  13_a_1_climate_finance_unsd.csv     – UNSD 13.a.1 long table
  13_3_1_green_education_unsd.csv     – UNSD 13.3.1 + SE_SGE_* series
  13_1_3_local_drr_unsd.csv           – UNSD 13.1.3 (proportion)
  sdg13_country_year_panel.csv        – Merged panel: GHG + NDC dummy
  sdg13_build_manifest.json           – Per-step status record
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# ── Configuration ──────────────────────────────────────────────────────────
UNSD_API = "https://unstats.un.org/sdgapi/v1/sdg"
WDI_API = "https://api.worldbank.org/v2"
GHG_CODE = "EN.GHG.ALL.MT.CE.AR5"
YEARS = list(range(2010, 2026))

OUTPUT_DIR = Path("/mnt/hgfs/share/SDG")
RAW_DIR = OUTPUT_DIR / "raw_jsons"
CLEAN_DIR = OUTPUT_DIR  # CSVs go directly in the sdg folder

# NDC probe sources (try in order)
NDC_CANDIDATES = [
    Path("${PROJECT_ROOT}/sdg13_ndc_probe_results_final.csv"),
    Path("/mnt/hgfs/share/SDG/最终合并_SDG13_SDG16_SDG17_20260726/01_核心总表与联合数据/SDG13_NDC逐国0_1_带置信度来源.csv"),
]

# ── Helpers ────────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "SDG13PanelBuilder/1.0 (research)"


def req_json(url: str, params: dict[str, str] | None = None, retries: int = 3) -> Any:
    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(url, params=params or {}, timeout=90)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == retries:
                raise RuntimeError(f"Failed after {retries} retries: {e}") from e
            time.sleep(2 * attempt)


def save_json_mini(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def scalar(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return "|".join(str(x) for x in v if x is not None)
    return str(v)


# ── 1. Country metadata (WB) ───────────────────────────────────────────────
def fetch_country_meta() -> pd.DataFrame:
    print("  Fetching World Bank country metadata…")
    payload = req_json(f"{WDI_API}/country", {"format": "json", "per_page": 400})
    rows = []
    for item in payload[1]:
        region = (item.get("region") or {}).get("value", "")
        income = (item.get("incomeLevel") or {}).get("value", "")
        iso3 = item.get("id", "")
        if region == "Aggregates" or income == "Aggregates" or not iso3:
            continue
        rows.append({
            "iso3": iso3,
            "country": item.get("name", ""),
            "region": region,
            "income_level": income,
        })
    df = pd.DataFrame(rows).sort_values(["country", "iso3"])
    save_json_mini(RAW_DIR / "wb_country_meta.json", {"count": len(df)})
    return df


# ── 2. 13.2.2 – WDI GHG excluding LULUCF ───────────────────────────────────
def fetch_ghg_excluding_lulucf(country_meta: pd.DataFrame) -> pd.DataFrame:
    print("  Fetching 13.2.2 (GHG excl. LULUCF) from World Bank…")
    payload = req_json(f"{WDI_API}/country/all/indicator/{GHG_CODE}", {"format": "json", "per_page": 20000})
    meta, data_block = payload[0], payload[1] if len(payload) > 1 else []
    rows = []
    valid_iso3 = set(country_meta["iso3"])
    for item in data_block:
        iso3 = item.get("countryiso3code")
        if iso3 not in valid_iso3:
            continue
        year = pd.to_numeric(item.get("date"), errors="coerce")
        val = pd.to_numeric(item.get("value"), errors="coerce")
        if pd.isna(year) or pd.isna(val):
            continue
        rows.append({
            "iso3": iso3,
            "country": (item.get("country") or {}).get("value", ""),
            "year": int(year),
            "ghg_excluding_lulucf_mtco2e": float(val),
            "indicator": GHG_CODE,
            "source": "World Bank WDI",
        })
    df = pd.DataFrame(rows).sort_values(["iso3", "year"])
    save_json_mini(RAW_DIR / "ghg_meta.json", {"last_updated": meta.get("lastupdated", ""), "count": len(df)})
    return df


# ── 3. NDC first-submission 0/1 dummy (from probe CSV) ─────────────────────
def _norm_name(s: str) -> str:
    s = s.casefold()
    s = s.replace("&", "and")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def build_ndc_dummy() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use existing NDC probe CSV to construct the 0/1 panel."""
    source = None
    for p in NDC_CANDIDATES:
        if p.exists():
            source = p
            break
    if source is None:
        print("  ⚠️  No NDC probe CSV found – dummy panel will be empty.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"  Building NDC panel from {source.name}…")
    ndc = pd.read_csv(source)
    if "country" not in ndc.columns or "ndc_0_1" not in ndc.columns:
        print(f"  ⚠️  Unexpected columns: {list(ndc.columns)}")
        return pd.DataFrame(), pd.DataFrame()

    for col in ["confidence", "evidence_type", "evidence", "url"]:
        if col not in ndc.columns:
            ndc[col] = ""

    # Extract a conservative first-submission year from evidence
    extracted = ndc.apply(_extract_ndc_year, axis=1, result_type="expand")
    extracted.columns = ["first_submission_year", "year_quality", "year_evidence"]
    ndc = pd.concat([ndc, extracted], axis=1)
    ndc["first_submission_year"] = pd.to_numeric(ndc["first_submission_year"], errors="coerce").astype("Int64")

    # Build summary
    summary = ndc[["country", "ndc_0_1", "confidence", "evidence_type",
                   "first_submission_year", "year_quality", "year_evidence",
                   "url", "evidence"]].copy()

    # Expand to country-year panel
    panel_rows = []
    for _, row in summary.iterrows():
        fy = row["first_submission_year"]
        for yr in YEARS:
            if pd.isna(fy):
                dummy = pd.NA
                qual = "manual_verification_needed"
            else:
                dummy = 1 if yr >= int(fy) else 0
                qual = row["year_quality"]
            panel_rows.append({
                "country": row["country"],
                "country_normalized": _norm_name(row["country"]),
                "year": yr,
                "ndc_first_submission_dummy": dummy,
                "first_submission_year": fy,
                "year_quality": qual,
                "ndc_presence_0_1_current": row["ndc_0_1"],
                "presence_confidence": row["confidence"],
            })
    panel_df = pd.DataFrame(panel_rows)
    return panel_df, summary


MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _extract_ndc_year(row: pd.Series) -> tuple:
    evidence = str(row.get("evidence", "") or "")
    url = str(row.get("url", "") or "")
    conf = str(row.get("confidence", "") or "").lower()
    haystack = f"{evidence} {url.split('/')[-1]}"

    candidates = []
    for match in re.finditer(r"(20[0-9]{2})", haystack):
        y = int(match.group(1))
        snip = haystack[max(0, match.start() - 60):match.end() + 60]
        candidates.append((y, snip, match.start()))

    # Score: prefer early NDC years (2015–2018), penalise non-NDC context
    def score(c):
        y, snip, pos = c
        s = snip.casefold()
        pts = 0
        if y in {2015, 2016, 2017, 2018}:
            pts += 5
        if re.search(r"indc|first.*ndc|submitted.*first|hereby communicates", s):
            pts += 6
        if re.search(r"biennial|national communication|progress|btr", s):
            pts -= 5
        if "officially submitted" in s or "submitted its first" in s:
            pts += 5
        return pts

    plausible = [c for c in candidates if 2015 <= c[0] <= 2021]
    if not plausible:
        return pd.NA, "manual_verification_needed", "No year in 2015–2021 found in evidence"

    ranked = sorted(plausible, key=lambda c: (score(c), -abs(c[0] - 2015)), reverse=True)
    best = ranked[0]
    if score(best) < 4:
        return pd.NA, "manual_verification_needed", best[1][:250]
    qual = "automatic_high" if score(best) >= 8 and conf == "high" else "automatic_review"
    return int(best[0]), qual, best[1][:250]


# ── 4–6. UNSD indicator fetch (generic, small pageSize) ────────────────────
def fetch_unsd(indicator: str, name: str) -> pd.DataFrame:
    print(f"  Fetching {indicator} ({name})…")
    url = f"{UNSD_API}/Indicator/Data"
    page_size = 500  # safe for UNSD API, avoids OOM
    all_records: list[dict] = []

    first = req_json(url, {"indicator": indicator, "page": "1", "pageSize": str(page_size)})
    total_pages = int(first.get("totalPages") or 0) or 1
    payloads = [first]

    for p in range(2, total_pages + 1):
        payloads.append(req_json(url, {"indicator": indicator, "page": str(p), "pageSize": str(page_size)}))
        time.sleep(0.3)

    for payload in payloads:
        for item in payload.get("data") or []:
            attrs = item.get("attributes") or {}
            dims = item.get("dimensions") or {}
            all_records.append({
                "goal": scalar(item.get("goal")),
                "target": scalar(item.get("target")),
                "indicator": scalar(item.get("indicator")),
                "series_code": scalar(item.get("series")),
                "series_description": scalar(item.get("seriesDescription")),
                "geo_area_code": scalar(item.get("geoAreaCode")),
                "geo_area_name": scalar(item.get("geoAreaName")),
                "year": int(float(item.get("timePeriodStart", 0))) if item.get("timePeriodStart") else pd.NA,
                "value": item.get("value"),
                "value_numeric": pd.to_numeric(item.get("value"), errors="coerce"),
                "nature": scalar(attrs.get("Nature")),
                "units": scalar(attrs.get("Units")),
                "reporting_type": scalar(dims.get("Reporting Type")),
                "type_of_support": scalar(dims.get("Type of support")),
            })

    df = pd.DataFrame(all_records)
    if not df.empty:
        df["year"] = df["year"].astype("Int64")
        df = df.sort_values(["series_code", "geo_area_name", "year"])
    save_json_mini(RAW_DIR / f"{name}_meta.json", {"total_pages": total_pages, "records": len(df)})
    return df


# ── 7. Series catalog for green education ──────────────────────────────────
def fetch_education_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Official 13.3.1 series + keyword screen across all UNSD series."""
    print("  Fetching education series catalog…")
    indicator_series = req_json(f"{UNSD_API}/Indicator/13.3.1/Series/List")
    all_series = req_json(f"{UNSD_API}/Series/List")

    official = []
    for block in indicator_series:
        for item in block.get("series") or []:
            code = scalar(item.get("code"))
            cat = "GCED/ESD official" if code.startswith("SE_GCEDESD_") else "Greening score"
            official.append({
                "indicator": block.get("code", "13.3.1"),
                "series_code": code,
                "series_description": scalar(item.get("description")),
                "category": cat,
            })
    official_df = pd.DataFrame(official)

    # Keyword screen: all series mentioning education / green / climate / SD
    pat = re.compile(r"education|sustainable development|greening|climate change|biodiversity|environment", re.I)
    related = []
    for item in all_series:
        text = f"{scalar(item.get('code'))} {scalar(item.get('description'))}"
        goals = scalar(item.get("goal"))
        targets = scalar(item.get("target"))
        if pat.search(text) and any(g in goals for g in ["4", "12", "13"] if g):
            related.append({
                "goal": goals,
                "target": targets,
                "indicator": scalar(item.get("indicator")),
                "series_code": scalar(item.get("code")),
                "series_description": scalar(item.get("description")),
            })
    related_df = pd.DataFrame(related).sort_values(["indicator", "series_code"])
    save_json_mini(RAW_DIR / "education_series_meta.json", {
        "official_series": len(official_df), "keyword_series": len(related_df)
    })
    return official_df, related_df


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {}

    # 1. Country metadata
    try:
        country_meta = fetch_country_meta()
        print(f"  → {len(country_meta)} countries")
        manifest["country_metadata"] = {"status": "ok", "count": len(country_meta)}
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        manifest["country_metadata"] = {"status": "failed", "error": str(e)}
        return 1

    # 2. GHG excluding LULUCF
    try:
        ghg = fetch_ghg_excluding_lulucf(country_meta)
        ghg.to_csv(CLEAN_DIR / "13_2_2_ghg_excluding_lulucf.csv", index=False, encoding="utf-8-sig")
        print(f"  → {len(ghg)} rows")
        manifest["13.2.2_ghg"] = {"status": "ok", "rows": len(ghg)}
    except Exception as e:
        ghg = pd.DataFrame()
        print(f"  ✗ Failed: {e}")
        manifest["13.2.2_ghg"] = {"status": "failed", "error": str(e)}

    # 3. NDC dummy
    try:
        ndc_panel, ndc_summary = build_ndc_dummy()
        ndc_panel.to_csv(CLEAN_DIR / "ndc_first_submission_dummy.csv", index=False, encoding="utf-8-sig")
        ndc_summary.to_csv(CLEAN_DIR / "ndc_first_submission_evidence.csv", index=False, encoding="utf-8-sig")
        print(f"  → {len(ndc_panel)} panel rows, {len(ndc_summary)} countries")
        manifest["ndc_dummy"] = {"status": "ok", "panel_rows": len(ndc_panel), "countries": len(ndc_summary)}
    except Exception as e:
        ndc_panel = pd.DataFrame()
        print(f"  ✗ Failed: {e}")
        manifest["ndc_dummy"] = {"status": "failed", "error": str(e)}

    # 4. 13.a.1 climate finance
    try:
        finance = fetch_unsd("13.a.1", "13_a_1_climate_finance")
        finance.to_csv(CLEAN_DIR / "13_a_1_climate_finance_unsd.csv", index=False, encoding="utf-8-sig")
        print(f"  → {len(finance)} records")
        manifest["13.a.1_finance"] = {"status": "ok", "records": len(finance)}
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        manifest["13.a.1_finance"] = {"status": "failed", "error": str(e)}

    # 5. 13.3.1 green education
    try:
        education = fetch_unsd("13.3.1", "13_3_1_green_education")
        education.to_csv(CLEAN_DIR / "13_3_1_green_education_unsd.csv", index=False, encoding="utf-8-sig")
        print(f"  → {len(education)} records")
        manifest["13.3.1_education"] = {"status": "ok", "records": len(education)}
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        manifest["13.3.1_education"] = {"status": "failed", "error": str(e)}

    # 6. 13.1.3 DRR (lower priority)
    try:
        drr = fetch_unsd("13.1.3", "13_1_3_local_drr")
        drr.to_csv(CLEAN_DIR / "13_1_3_local_drr_unsd.csv", index=False, encoding="utf-8-sig")
        print(f"  → {len(drr)} records")
        manifest["13.1.3_drr"] = {"status": "ok", "records": len(drr)}
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        manifest["13.1.3_drr"] = {"status": "failed", "error": str(e)}

    # 7. Education series catalog
    try:
        ed_off, ed_rel = fetch_education_series()
        ed_off.to_csv(CLEAN_DIR / "13_3_1_education_series_catalog.csv", index=False, encoding="utf-8-sig")
        ed_rel.to_csv(CLEAN_DIR / "unsd_education_green_series_keyword.csv", index=False, encoding="utf-8-sig")
        print(f"  → {len(ed_off)} official + {len(ed_rel)} keyword-matched series")
        manifest["education_series"] = {"status": "ok", "official": len(ed_off), "keyword": len(ed_rel)}
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        manifest["education_series"] = {"status": "failed", "error": str(e)}

    # 8. Merged panel (GHG + NDC)
    if not country_meta.empty:
        try:
            base = country_meta.assign(_k=1).merge(pd.DataFrame({"year": YEARS, "_k": 1}), on="_k").drop(columns="_k")
            base["country_normalized"] = base["country"].map(_norm_name)

            if not ghg.empty:
                base = base.merge(
                    ghg[["iso3", "year", "ghg_excluding_lulucf_mtco2e"]], on=["iso3", "year"], how="left"
                )
            if not ndc_panel.empty:
                ndc_sub = ndc_panel[["country_normalized", "year", "ndc_first_submission_dummy",
                                    "first_submission_year", "year_quality"]]
                base = base.merge(ndc_sub, on=["country_normalized", "year"], how="left")

            base.to_csv(CLEAN_DIR / "sdg13_country_year_panel.csv", index=False, encoding="utf-8-sig")
            manifest["merged_panel"] = {"status": "ok", "rows": len(base)}
            print(f"  → Merged panel: {len(base)} rows")
        except Exception as e:
            print(f"  ✗ Panel merge failed: {e}")
            manifest["merged_panel"] = {"status": "failed", "error": str(e)}

    # Write manifest
    manifest["output_dir"] = str(CLEAN_DIR)
    with open(OUTPUT_DIR / "build_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n✅ All done. Output → {CLEAN_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
