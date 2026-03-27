from __future__ import annotations

from typing import Iterable

import pandas as pd


PRICE_EXOG_COLUMNS = ("min_price", "max_price", "change")


def clean_price_series_frame(
    df: pd.DataFrame,
    *,
    exog_cols: Iterable[str] = PRICE_EXOG_COLUMNS,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["date", "modal_price"])
    if "date" not in df.columns or "modal_price" not in df.columns:
        return pd.DataFrame(columns=["date", "modal_price"])

    keep_cols = ["date", "modal_price", *[str(col) for col in exog_cols if str(col) in df.columns]]
    work = df[keep_cols].copy()
    work["date"] = pd.to_datetime(work["date"], dayfirst=True, errors="coerce")
    work["modal_price"] = pd.to_numeric(work["modal_price"], errors="coerce")
    for col in keep_cols:
        if col in {"date", "modal_price"}:
            continue
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["date", "modal_price"]).sort_values("date")
    if work.empty:
        return pd.DataFrame(columns=keep_cols)

    out = work.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["modal_price"] = pd.to_numeric(out["modal_price"], errors="coerce")
    for col in keep_cols:
        if col in {"date", "modal_price"} or col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "modal_price"]).sort_values("date").reset_index(drop=True)
    return out


def clean_yield_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    crop_col = "crop_name" if "crop_name" in df.columns else "crop"
    year_col = "year" if "year" in df.columns else None
    target_col = "yield_quintal_per_hectare" if "yield_quintal_per_hectare" in df.columns else "yield"
    if crop_col not in df.columns or year_col is None or target_col not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work[crop_col] = work[crop_col].astype(str).str.strip()
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[crop_col, year_col, target_col]).copy()
    if work.empty:
        return pd.DataFrame(columns=list(df.columns))
    work = work[work[crop_col] != ""].copy()
    if work.empty:
        return pd.DataFrame(columns=list(df.columns))
    work[year_col] = work[year_col].astype(int)
    return work.drop_duplicates(subset=[crop_col, year_col], keep="last").sort_values([crop_col, year_col]).reset_index(drop=True)


def clean_cost_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    crop_col = "crop_name" if "crop_name" in df.columns else "cost_name"
    year_col = "year_start" if "year_start" in df.columns else "year"
    target_col = "india_cost_wavg_sample" if "india_cost_wavg_sample" in df.columns else "cost"
    if crop_col not in df.columns or year_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work[crop_col] = work[crop_col].astype(str).str.strip()
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[crop_col, year_col, target_col]).copy()
    if work.empty:
        return pd.DataFrame(columns=list(df.columns))
    work = work[work[crop_col] != ""].copy()
    if work.empty:
        return pd.DataFrame(columns=list(df.columns))
    work[year_col] = work[year_col].astype(int)
    return work.drop_duplicates(subset=[crop_col, year_col], keep="last").sort_values([crop_col, year_col]).reset_index(drop=True)
