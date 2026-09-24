"""Sector rotation analysis."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SectorRotation:
    leaders: list[str] = field(default_factory=list)
    laggards: list[str] = field(default_factory=list)
    diffusion: float = 0.0
    concentration: float = 0.0
    style: str = "balanced"
    detail: dict = field(default_factory=dict)


def analyze_sector_rotation(sector_df: pd.DataFrame | None, top_n: int = 5) -> SectorRotation:
    if sector_df is None or len(sector_df) == 0:
        return SectorRotation()
    df = sector_df.copy()
    name_col = "name" if "name" in df.columns else "板块" if "板块" in df.columns else df.columns[0]
    pct_col = "pct_chg" if "pct_chg" in df.columns else "涨跌幅" if "涨跌幅" in df.columns else None
    if pct_col is None:
        return SectorRotation()
    df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce")
    df = df.dropna(subset=[pct_col]).sort_values(pct_col, ascending=False)
    leaders = [str(x) for x in df.head(top_n)[name_col].tolist()]
    laggards = [str(x) for x in df.tail(top_n)[name_col].tolist()]
    diffusion = float((df[pct_col] > 0).mean()) if len(df) else 0.0
    total_abs = df[pct_col].abs().sum()
    concentration = float(df[pct_col].head(top_n).abs().sum() / total_abs) if total_abs else 0.0
    if diffusion > 0.65 and concentration < 0.35:
        style = "broad_rise"
    elif diffusion < 0.35:
        style = "broad_fall"
    elif concentration >= 0.45:
        style = "concentrated"
    else:
        style = "balanced"
    return SectorRotation(leaders, laggards, round(diffusion, 3), round(concentration, 3), style)
