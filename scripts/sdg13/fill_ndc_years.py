#!/usr/bin/env python3
"""Fill NDC first-submission-year gaps using Climate Watch + UNFCCC evidence.

Strategy:
1. Where probe evidence already extracted a year → keep it (65 countries)
2. For remaining 131 countries, use Climate Watch indicator data:
   - If `indc_submission = "INDC Submitted"` → first year likely 2015
   - If `ndce_revised = "No"` and has ndce_date → ndce_date year = first submission
   - If only submission_date available → use that year
3. Export filled CSV for manual review
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ── Config ─────────────────────────────────────────────────────────────────
PROBE_CSV = Path("./sdg13_ndc_probe_results_final.csv")
OUTPUT_DIR = Path("/mnt/hgfs/share/SDG")
CW_JSON = Path("/tmp/cw_ndcs.json")

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "NDCFillScript/1.0"


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return " ".join(s.split())


def load_cw_data() -> dict:
    """Extract all useful NDC indicators from Climate Watch JSON."""
    with open(CW_JSON) as f:
        d = json.load(f)

    indicators = {}
    for item in d["indicators"]:
        indicators[item["id"]] = item["locations"]

    # Fetch WB country mapping for ISO-to-name
    try:
        r = SESSION.get(
            "https://api.worldbank.org/v2/country",
            params={"format": "json", "per_page": 400},
            timeout=30,
        )
        payload = r.json()
    except Exception:
        payload = [[], []]

    wb = {}
    for item in payload[1] if len(payload) > 1 else []:
        region = (item.get("region") or {}).get("value", "")
        income = (item.get("incomeLevel") or {}).get("value", "")
        iso3 = item.get("id", "")
        if region == "Aggregates" or income == "Aggregates" or not iso3:
            continue
        wb[iso3] = item.get("name", "")

    return indicators, wb


def match_country_to_iso(
    country: str, wb_map: dict, wb_by_name: dict
) -> str | None:
    """Match a country name to ISO3 code."""
    n = normalize(country)
    # Direct match
    if n in wb_by_name:
        return wb_by_name[n]

    # Partial match
    for name, iso in wb_by_name.items():
        if n in name or name in n:
            return iso

    # Handle special cases
    # EU member states that submitted NDCs as part of EU bloc
    eu_members = ["austria", "belgium", "bulgaria", "croatia", "cyprus", "czechia",
                  "czech republic", "denmark", "estonia", "finland", "france", "germany",
                  "greece", "hungary", "ireland", "italy", "latvia", "lithuania",
                  "luxembourg", "malta", "netherlands", "poland", "portugal",
                  "romania", "slovakia", "slovak republic", "slovenia", "spain", "sweden"]

    special = {
        "brunei darussalam": "BRN",
        "congo": "COG",
        "democratic republic of the congo": "COD",
        "dominican republic": "DOM",
        "iran": "IRN",
        "laos": "LAO",
        "lao people s democratic republic": "LAO",
        "lao peoples democratic republic": "LAO",
        "libya": "LBY",
        "macedonia": "MKD",
        "north macedonia": "MKD",
        "micronesia": "FSM",
        "moldova": "MDA",
        "republic of korea": "KOR",
        "russian federation": "RUS",
        "syria": "SYR",
        "swaziland": "SWZ",
        "tanzania": "TZA",
        "tanzania united republic of": "TZA",
        "turkiye": "TUR",
        "trkiye": "TUR",  # Türkiye stripped of diacritic
        "turkey": "TUR",
        "turkish republic": "TUR",
        "united kingdom": "GBR",
        "united kingdom of great britain and northern ireland": "GBR",
        "united states": "USA",
        "united states of america": "USA",
        "viet nam": "VNM",
        "vietnam": "VNM",
        "kyrgyzstan": "KGZ",
        "kyrgyz republic": "KGZ",
        "slovakia": "SVK",
        "slovak republic": "SVK",
        "state of palestine": "PSE",
        "palestine": "PSE",
        "saint lucia": "LCA",
        "st lucia": "LCA",
        "saint kitts and nevis": "KNA",
        "st kitts and nevis": "KNA",
        "saint vincent and the grenadines": "VCT",
        "st vincent and the grenadines": "VCT",
        "sao tome and principe": "STP",
        "cabo verde": "CPV",
        "cote d ivoire": "CIV",
        "cote d'ivoire": "CIV",
        "niue": "NIU",
        "european union": "EUU",
    }

    if n in special:
        return special[n]

    # EU member states: match with any EU country ISO
    if n in eu_members:
        wb_name_map = {
            "austria": "AUT", "belgium": "BEL", "bulgaria": "BGR",
            "croatia": "HRV", "cyprus": "CYP", "czechia": "CZE",
            "czech republic": "CZE", "denmark": "DNK", "estonia": "EST",
            "finland": "FIN", "france": "FRA", "germany": "DEU",
            "greece": "GRC", "hungary": "HUN", "ireland": "IRL",
            "italy": "ITA", "latvia": "LVA", "lithuania": "LTU",
            "luxembourg": "LUX", "malta": "MLT", "netherlands": "NLD",
            "poland": "POL", "portugal": "PRT", "romania": "ROU",
            "slovakia": "SVK", "slovak republic": "SVK",
            "slovenia": "SVN", "spain": "ESP", "sweden": "SWE"
        }
        return wb_name_map.get(n)
    if n in special:
        return special[n]

    return None


def extract_year_from_date_str(date_str: str) -> int | None:
    """Try to extract year from date string like '11/23/2016' or '10/28/2021'."""
    for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"]:
        try:
            return datetime.strptime(date_str, fmt).year
        except ValueError:
            continue
    return None


def estimate_first_ndc_year(
    iso3: str, indicators: dict
) -> tuple[int | None, str, str]:
    """Estimate first NDC submission year for a country using CW indicators.

    Returns (year, quality, reason).
    """
    ndce_date_val = None
    ndce_date_year = None
    sub_date_val = None
    sub_date_year = None
    indc_status = None
    is_revised = None

    # Extract CW data
    locs = indicators.get(632868, {})
    if iso3 in locs and locs[iso3]:
        ndce_date_val = locs[iso3][0].get("value")
        ndce_date_year = extract_year_from_date_str(ndce_date_val) if ndce_date_val else None

    locs = indicators.get(632799, {})
    if iso3 in locs and locs[iso3]:
        sub_date_val = locs[iso3][0].get("value")
        sub_date_year = extract_year_from_date_str(sub_date_val) if sub_date_val else None

    locs = indicators.get(632800, {})
    if iso3 in locs and locs[iso3]:
        indc_status = locs[iso3][0].get("value", "")

    locs = indicators.get(632770, {})
    if iso3 in locs and locs[iso3]:
        is_revised = locs[iso3][0].get("value", "")

    # Strategy 1: INDC was submitted → first year likely 2015
    if indc_status and "INDC Submitted" in indc_status:
        # Most INDCs were submitted in 2015-2016
        # If ndce_date is available and country hasn't revised, ndce_date IS the first date
        if is_revised and "No" in is_revised and ndce_date_year:
            return ndce_date_year, "automatic_cw", f"CW ndce_date={ndce_date_val}, not revised, no INDC date needed"
        # If revised, the INDC was the first submission - use 2015
        return 2015, "estimated_inndc", f"CW indc_submitted, ndce_revised={is_revised}, INDC assumed 2015"

    # Strategy 2: No INDC, use ndce_date or submission_date as first
    if ndce_date_year:
        return ndce_date_year, "estimated_ndcedate", f"CW ndce_date={ndce_date_val}"
    if sub_date_year:
        return sub_date_year, "estimated_subdate", f"CW submission_date={sub_date_val}"

    return None, "manual_needed", "No CW data found"


def main():
    # Load data
    indicators, wb_map = load_cw_data()

    # Build reverse map
    wb_by_name = {normalize(name): iso for iso, name in wb_map.items()}

    # Read probe CSV
    ndc = pd.read_csv(PROBE_CSV)
    for col in ["confidence", "evidence_type", "evidence", "url"]:
        if col not in ndc.columns:
            ndc[col] = ""

    # Apply existing extraction (from builder script)
    # Reuse the same extraction logic
    ndc["first_submission_year"] = pd.NA
    ndc["year_quality"] = ""
    ndc["year_source"] = ""

    # First pass: try to extract from evidence
    MONTH_MAP = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }

    for idx, row in ndc.iterrows():
        evidence = str(row.get("evidence", "") or "")
        url = str(row.get("url", "") or "")
        conf = str(row.get("confidence", "") or "").lower()
        haystack = f"{evidence} {url.split('/')[-1]}"

        candidates = []
        for match in re.finditer(r"(20[0-9]{2})", haystack):
            y = int(match.group(1))
            snip = haystack[max(0, match.start() - 60):match.end() + 60]
            candidates.append((y, snip, match.start()))

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
            if conf == "high":
                pts += 1
            return pts

        plausible = [c for c in candidates if 2015 <= c[0] <= 2021]
        if plausible:
            ranked = sorted(plausible, key=lambda c: (score(c), -abs(c[0] - 2015)), reverse=True)
            best = ranked[0]
            if score(best) >= 4:
                ndc.at[idx, "first_submission_year"] = int(best[0])
                qual = "automatic_high" if score(best) >= 8 and conf == "high" else "automatic_review"
                ndc.at[idx, "year_quality"] = qual
                ndc.at[idx, "year_source"] = "evidence_text"
                continue

        # Fall back to CW data
        iso3 = match_country_to_iso(row["country"], wb_map, wb_by_name)
        if iso3:
            year, quality, reason = estimate_first_ndc_year(iso3, indicators)
            if year:
                ndc.at[idx, "first_submission_year"] = year
                ndc.at[idx, "year_quality"] = quality
                ndc.at[idx, "year_source"] = reason

    # Summary
    has_year = ndc["first_submission_year"].notna()
    print(f"Total countries: {len(ndc)}")
    print(f"With first_submission_year: {has_year.sum()}")
    print(f"  - From evidence text: {((ndc['year_source'] == 'evidence_text') & has_year).sum()}")
    print(f"  - From CW indicators: {((ndc['year_source'] != 'evidence_text') & has_year).sum()}")
    print(f"Still missing: {(~has_year).sum()}")

    # Show samples
    print("\n=== Samples from CW-filled ===")
    cw_filled = ndc[(ndc['year_source'] != 'evidence_text') & has_year]
    for _, r in cw_filled.head(15).iterrows():
        print(f"  {r['country']}: {r['first_submission_year']} ({r['year_quality']}) - {r['year_source'][:60]}")

    print("\n=== Still missing ===")
    missing = ndc[~has_year]
    for _, r in missing.iterrows():
        print(f"  {r['country']}")

    # Save
    out_path = OUTPUT_DIR / "ndc_first_submission_filled.csv"
    ndc.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {out_path}")

    # Also generate the panel
    years = list(range(2010, 2026))
    panel_rows = []
    for _, row in ndc.iterrows():
        fy = row["first_submission_year"]
        for yr in years:
            if pd.isna(fy):
                dummy = pd.NA
            else:
                dummy = 1 if yr >= int(float(fy)) else 0
            panel_rows.append({
                "country": row["country"],
                "year": yr,
                "ndc_first_submission_dummy": dummy,
                "first_submission_year": fy,
                "year_quality": row["year_quality"],
                "year_source": row["year_source"],
                "ndc_presence_0_1_current": row["ndc_0_1"],
                "presence_confidence": row["confidence"],
            })
    panel = pd.DataFrame(panel_rows)
    panel_path = OUTPUT_DIR / "ndc_first_submission_dummy_filled.csv"
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    print(f"Panel saved to {panel_path}")
    print(f"Panel rows: {len(panel)}")

    # Dummy distribution
    has_dummy = panel["ndc_first_submission_dummy"].notna()
    print(f"Panel with dummy value: {has_dummy.sum()}/{len(panel)}")
    if has_dummy.any():
        d1 = (panel["ndc_first_submission_dummy"] == 1).sum()
        d0 = (panel["ndc_first_submission_dummy"] == 0).sum()
        print(f"  dummy=1: {d1}, dummy=0: {d0}")


if __name__ == "__main__":
    main()
