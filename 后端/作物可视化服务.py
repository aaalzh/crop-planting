from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from 后端.数据清洗 import clean_price_series_frame
from 后端.时间对齐 import AlignmentPolicy, build_aligned_visual_payload
from 后端.数据加载 import canonicalize_price_file, load_cost_data, load_price_series, load_yield_history
from 后端.价格汇总 import summarize_forecast_tail


def _safe_float(v: Any) -> float | None:
    try:
        val = float(v)
    except Exception:
        return None
    if pd.isna(val):
        return None
    return val


def _safe_int(v: Any) -> int | None:
    try:
        val = int(v)
    except Exception:
        return None
    return val


def _aligned_to_date_rows(time_index: List[str], values: List[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(time_index, list) or not isinstance(values, list):
        return rows
    for year_text, value in zip(time_index, values):
        year = _safe_int(year_text)
        val = _safe_float(value)
        if year is None or val is None:
            continue
        rows.append({"date": f"{int(year):04d}-01-01", "value": val})
    return rows


def _yearly_mean_from_date_rows(rows: List[Dict[str, Any]]) -> Dict[int, float]:
    buckets: Dict[int, List[float]] = {}
    if not isinstance(rows, list):
        return {}
    for row in rows:
        date_text = str(row.get("date", "")).strip()
        value = _safe_float(row.get("value"))
        if not date_text or value is None:
            continue
        try:
            year = int(pd.Timestamp(date_text).year)
        except Exception:
            continue
        buckets.setdefault(year, []).append(float(value))

    out: Dict[int, float] = {}
    for year, values in buckets.items():
        if not values:
            continue
        out[int(year)] = float(np.mean(values))
    return out


def _aligned_ma_from_date_rows(time_index: List[str], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    yearly_map = _yearly_mean_from_date_rows(rows)
    out: List[Dict[str, Any]] = []
    if not isinstance(time_index, list) or not yearly_map:
        return out

    for year_text in time_index:
        year = _safe_int(year_text)
        if year is None:
            continue
        value = _safe_float(yearly_map.get(int(year)))
        if value is None:
            continue
        out.append({"date": f"{int(year):04d}-01-01", "value": value})
    return out


def _aligned_to_year_rows(time_index: List[str], values: List[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(time_index, list) or not isinstance(values, list):
        return rows
    for year_text, value in zip(time_index, values):
        year = _safe_int(year_text)
        val = _safe_float(value)
        if year is None or val is None:
            continue
        rows.append({"year": int(year), "value": val})
    return rows


def _aligned_forecast_point(time_index: List[str], values: List[Any]) -> Dict[str, Any]:
    if not isinstance(time_index, list) or not isinstance(values, list):
        return {}
    for year_text, value in zip(time_index, values):
        year = _safe_int(year_text)
        val = _safe_float(value)
        if year is None or val is None:
            continue
        return {"year": int(year), "value": val}
    return {}


def _trend_from_year_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        return []
    try:
        df = pd.DataFrame(rows)
    except Exception:
        return []
    if "year" not in df.columns or "value" not in df.columns:
        return []
    return _build_yearly_trend(df, "year", "value")


def _build_yearly_trend(df: pd.DataFrame, x_col: str, y_col: str) -> List[Dict[str, Any]]:
    if x_col not in df.columns or y_col not in df.columns:
        return []
    tmp = df[[x_col, y_col]].copy()
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.dropna(subset=[x_col, y_col]).sort_values(x_col)
    if len(tmp) < 2:
        return []

    x = tmp[x_col].to_numpy(dtype=float)
    y = tmp[y_col].to_numpy(dtype=float)
    if len(set(x.tolist())) < 2:
        return []

    try:
        coeff = np.polyfit(x, y, deg=1)
        y_fit = coeff[0] * x + coeff[1]
    except Exception:
        return []

    points: List[Dict[str, Any]] = []
    for x_val, y_val in zip(x, y_fit):
        x_int = _safe_int(round(float(x_val)))
        y_float = _safe_float(y_val)
        if x_int is None or y_float is None:
            continue
        points.append({"year": x_int, "value": y_float})
    return points


def _calc_cagr_pct(df: pd.DataFrame, year_col: str, value_col: str) -> float | None:
    if year_col not in df.columns or value_col not in df.columns:
        return None
    tmp = df[[year_col, value_col]].copy()
    tmp[year_col] = pd.to_numeric(tmp[year_col], errors="coerce")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[year_col, value_col]).sort_values(year_col)
    if len(tmp) < 2:
        return None

    start_year = _safe_int(tmp.iloc[0][year_col])
    end_year = _safe_int(tmp.iloc[-1][year_col])
    start_value = _safe_float(tmp.iloc[0][value_col])
    end_value = _safe_float(tmp.iloc[-1][value_col])
    if start_year is None or end_year is None or start_value is None or end_value is None:
        return None
    year_gap = end_year - start_year
    if year_gap <= 0 or start_value <= 0 or end_value <= 0:
        return None
    try:
        cagr = ((end_value / start_value) ** (1.0 / year_gap) - 1.0) * 100.0
    except Exception:
        return None
    return _safe_float(cagr)


def _exp_interpolated_forecast(
    *,
    latest_date: pd.Timestamp,
    latest_price: float,
    target_price: float,
    horizon_days: int,
) -> List[Dict[str, Any]]:
    safe_horizon = max(1, int(horizon_days))
    p0 = max(1e-6, float(latest_price))
    p1 = max(1e-6, float(target_price))
    log0 = float(np.log(p0))
    log1 = float(np.log(p1))

    out: List[Dict[str, Any]] = []
    for step in range(1, safe_horizon + 1):
        progress = float(step) / float(safe_horizon)
        value = float(np.exp(log0 + (log1 - log0) * progress))
        out.append({"date": (latest_date + pd.Timedelta(days=step)).strftime("%Y-%m-%d"), "value": value})
    return out


def _build_price_forecast_curve(
    *,
    view: pd.DataFrame,
    latest_date: pd.Timestamp,
    latest_price: float,
    target_price: float,
    horizon_days: int,
) -> List[Dict[str, Any]]:
    """
    Build a daily price path anchored by historical dynamics and the model target.
    This replaces the old constant-line forecast and keeps a smooth transition.
    """
    safe_horizon = max(1, int(horizon_days))
    safe_latest = max(1e-6, float(latest_price))
    safe_target = max(1e-6, float(target_price))

    if view.empty or "date" not in view.columns or "modal_price" not in view.columns:
        return _exp_interpolated_forecast(
            latest_date=latest_date,
            latest_price=safe_latest,
            target_price=safe_target,
            horizon_days=safe_horizon,
        )

    work = view[["date", "modal_price"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["modal_price"] = pd.to_numeric(work["modal_price"], errors="coerce")
    work = work.dropna(subset=["date", "modal_price"]).sort_values("date")
    if work.empty:
        return _exp_interpolated_forecast(
            latest_date=latest_date,
            latest_price=safe_latest,
            target_price=safe_target,
            horizon_days=safe_horizon,
        )

    cleaned = clean_price_series_frame(work)
    if cleaned.empty:
        return _exp_interpolated_forecast(
            latest_date=latest_date,
            latest_price=safe_latest,
            target_price=safe_target,
            horizon_days=safe_horizon,
        )

    daily = pd.Series(cleaned["modal_price"].to_numpy(dtype=float), index=cleaned["date"])
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    if daily.empty:
        return _exp_interpolated_forecast(
            latest_date=latest_date,
            latest_price=safe_latest,
            target_price=safe_target,
            horizon_days=safe_horizon,
        )

    recent = daily.tail(min(730, len(daily)))
    returns = recent.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    vol = _safe_float(returns.std()) if len(returns) >= 2 else None
    if vol is None:
        vol = 0.008
    vol = float(np.clip(vol, 0.001, 0.05))

    seasonal_map = {day: 1.0 for day in range(1, 367)}
    if len(recent) >= 120:
        trend = recent.rolling(30, min_periods=7).mean()
        ratio = (recent / trend).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio) >= 60:
            grouped = ratio.groupby(ratio.index.dayofyear).median()
            full_index = pd.Index(range(1, 367), dtype=int)
            expanded = grouped.reindex(full_index).interpolate(limit_direction="both").fillna(1.0)
            values = expanded.to_numpy(dtype=float)
            pad = np.concatenate([values[-3:], values, values[:3]])
            smooth = pd.Series(pad).rolling(7, center=True, min_periods=1).mean().to_numpy()[3:-3]
            seasonal_map = {int(day): float(val) for day, val in zip(full_index, smooth)}

    season_ref = float(seasonal_map.get(int(latest_date.dayofyear), 1.0))
    if abs(season_ref) < 1e-6:
        season_ref = 1.0

    required_log_step = abs(float(np.log(safe_target / safe_latest))) / float(safe_horizon)
    max_daily_move = float(np.clip(max(vol * 2.2 + 0.002, required_log_step * 2.4), 0.004, 0.08))

    progress_power = 1.18
    path: List[float] = []
    prev = safe_latest
    for step in range(1, safe_horizon + 1):
        day = latest_date + pd.Timedelta(days=step)
        progress = float(step) / float(safe_horizon)
        anchor_log = np.log(safe_latest) + (np.log(safe_target) - np.log(safe_latest)) * (progress**progress_power)
        anchor = float(np.exp(anchor_log))

        season = float(seasonal_map.get(int(day.dayofyear), 1.0) / season_ref)
        season = float(np.clip(season, 0.92, 1.08))
        desired = anchor * season

        blend = 0.20 + 0.65 * progress
        raw_next = prev + (desired - prev) * blend
        log_move = float(np.log(max(raw_next, 1e-6) / max(prev, 1e-6)))
        log_move = float(np.clip(log_move, -max_daily_move, max_daily_move))
        next_val = float(prev * np.exp(log_move))
        path.append(next_val)
        prev = next_val

    if path:
        # Gentle global correction so the endpoint matches the model's point prediction.
        ratio = safe_target / max(path[-1], 1e-6)
        path = [float(v * (ratio ** (idx / safe_horizon))) for idx, v in enumerate(path, start=1)]

        lower_bound = float(max(1e-6, recent.quantile(0.02) * 0.4))
        upper_bound = float(
            max(
                lower_bound * 1.2,
                float(recent.quantile(0.98) * 2.5),
                safe_latest * 1.4,
                safe_target * 1.4,
            )
        )
        path = [float(np.clip(v, lower_bound, upper_bound)) for v in path]

        # Re-anchor once after clipping.
        ratio2 = safe_target / max(path[-1], 1e-6)
        path = [float(v * (ratio2 ** (idx / safe_horizon))) for idx, v in enumerate(path, start=1)]

    return [
        {"date": (latest_date + pd.Timedelta(days=idx)).strftime("%Y-%m-%d"), "value": float(value)}
        for idx, value in enumerate(path, start=1)
    ]


def _normalize_external_price_forecast(
    rows: Optional[List[Dict[str, Any]]],
    *,
    latest_date: pd.Timestamp,
    horizon_days: int,
) -> List[Dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        return []

    parsed: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        dt_text = str(row.get("date", "")).strip()
        p50 = _safe_float(row.get("p50"))
        val = _safe_float(row.get("value"))
        p10 = _safe_float(row.get("p10"))
        p90 = _safe_float(row.get("p90"))
        blend_weight = _safe_float(row.get("blend_weight"))
        clip_applied = row.get("clip_applied")
        clip_amount = _safe_float(row.get("clip_amount"))
        if p50 is None:
            p50 = val
        if not dt_text or p50 is None:
            continue
        try:
            ts = pd.Timestamp(dt_text)
        except Exception:
            continue
        if pd.isna(ts):
            continue
        ts = ts.normalize()
        if ts <= latest_date:
            continue
        has_interval = p10 is not None or p90 is not None or ("p50" in row)
        item: Dict[str, Any] = {"date": ts.strftime("%Y-%m-%d"), "value": float(p50)}
        if has_interval:
            item["p50"] = float(p50)
        if p10 is not None:
            item["p10"] = float(p10)
        if p90 is not None:
            item["p90"] = float(p90)
        if blend_weight is not None:
            item["blend_weight"] = float(blend_weight)
        if clip_applied is not None:
            item["clip_applied"] = bool(clip_applied)
        if clip_amount is not None:
            item["clip_amount"] = float(clip_amount)
        parsed.append(item)

    if not parsed:
        return []

    parsed.sort(key=lambda x: x["date"])
    dedup: List[Dict[str, Any]] = []
    seen = set()
    for row in parsed:
        d = row["date"]
        if d in seen:
            dedup[-1] = row
            continue
        seen.add(d)
        dedup.append(row)
    return dedup[: max(1, int(horizon_days))]


def _build_price_visuals(
    price_dir: Path,
    price_file: str,
    price_pred: float | None,
    price_forecast: Optional[List[Dict[str, Any]]],
    horizon_days: int,
    history_years: int,
    as_of_date: Optional[pd.Timestamp] = None,
    actual_start_date: Optional[pd.Timestamp] = None,
    actual_end_date: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "history": [],
        "ma30": [],
        "ma90": [],
        "forecast": [],
        "actual": [],
        "stats": {
            "latest": None,
            "mean_90d": None,
            "volatility_90d_pct": None,
            "yoy_change_pct": None,
        },
        "warnings": [],
    }

    clean_price_file = str(price_file or "").strip()
    if not clean_price_file:
        out["warnings"].append("price_file_missing")
        return out

    try:
        df = load_price_series(str(price_dir), clean_price_file)
    except Exception:
        out["warnings"].append("price_history_unavailable")
        return out

    if df.empty or "date" not in df.columns or "modal_price" not in df.columns:
        out["warnings"].append("price_history_empty")
        return out

    full_view = df.sort_values("date").copy()
    if isinstance(actual_start_date, pd.Timestamp) and isinstance(actual_end_date, pd.Timestamp):
        actual_view = full_view[
            (pd.to_datetime(full_view["date"], errors="coerce") >= actual_start_date)
            & (pd.to_datetime(full_view["date"], errors="coerce") <= actual_end_date)
        ].copy()
        out["actual"] = [
            {"date": d.strftime("%Y-%m-%d"), "value": _safe_float(v)}
            for d, v in zip(actual_view["date"], actual_view["modal_price"])
            if _safe_float(v) is not None
        ]

    if isinstance(as_of_date, pd.Timestamp):
        df = df[pd.to_datetime(df["date"], errors="coerce") <= as_of_date].copy()
        if df.empty:
            out["warnings"].append("price_history_before_window_empty")
            return out

    safe_history_years = max(1, min(30, int(history_years or 8)))
    lookback_days = max(540, safe_history_years * 366)
    view = df.sort_values("date").tail(lookback_days).copy()
    view["ma30"] = view["modal_price"].rolling(30, min_periods=1).mean()
    view["ma90"] = view["modal_price"].rolling(90, min_periods=1).mean()

    out["history"] = [
        {"date": d.strftime("%Y-%m-%d"), "value": _safe_float(v)}
        for d, v in zip(view["date"], view["modal_price"])
        if _safe_float(v) is not None
    ]
    out["ma30"] = [
        {"date": d.strftime("%Y-%m-%d"), "value": _safe_float(v)}
        for d, v in zip(view["date"], view["ma30"])
        if _safe_float(v) is not None
    ]
    out["ma90"] = [
        {"date": d.strftime("%Y-%m-%d"), "value": _safe_float(v)}
        for d, v in zip(view["date"], view["ma90"])
        if _safe_float(v) is not None
    ]

    latest = _safe_float(view["modal_price"].iloc[-1])
    out["stats"]["latest"] = latest
    out["stats"]["mean_90d"] = _safe_float(view["modal_price"].tail(90).mean())

    pct = view["modal_price"].pct_change().tail(90).dropna()
    out["stats"]["volatility_90d_pct"] = _safe_float(float(pct.std()) * 100.0 if len(pct) else None)

    latest_date = view["date"].iloc[-1]
    prev = view[view["date"] <= latest_date - pd.Timedelta(days=365)]
    if not prev.empty:
        prev_val = _safe_float(prev["modal_price"].iloc[-1])
        if prev_val and prev_val > 0 and latest is not None:
            out["stats"]["yoy_change_pct"] = _safe_float((latest / prev_val - 1.0) * 100.0)

    external_forecast = _normalize_external_price_forecast(
        price_forecast,
        latest_date=latest_date,
        horizon_days=int(horizon_days),
    )
    if external_forecast:
        out["forecast"] = external_forecast
        price_pred_val = _safe_float(price_pred)
        if price_pred_val is None:
            price_pred_val = _safe_float(external_forecast[-1].get("value"))
    else:
        price_pred_val = _safe_float(price_pred)
        if price_pred_val is not None:
            anchor_latest = latest if latest is not None else price_pred_val
            out["forecast"] = _build_price_forecast_curve(
                view=view,
                latest_date=latest_date,
                latest_price=anchor_latest,
                target_price=price_pred_val,
                horizon_days=int(horizon_days),
            )

    return out


def _build_yield_visuals(
    yield_history_file: Path,
    crop: str,
    yield_pred: float | None,
    target_year: int,
    cutoff_year: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "history": [],
        "trend": [],
        "actual": [],
        "forecast": {"year": int(target_year), "value": _safe_float(yield_pred)},
        "stats": {
            "latest": None,
            "mean": None,
            "cagr_pct": None,
        },
        "warnings": [],
    }

    if not yield_history_file.exists():
        out["warnings"].append("yield_history_missing")
        return out

    try:
        df = load_yield_history(str(yield_history_file))
    except Exception:
        out["warnings"].append("yield_history_unreadable")
        return out

    if df.empty or "crop_name" not in df.columns:
        out["warnings"].append("yield_history_empty")
        return out

    view = df[df["crop_name"].astype(str).str.strip().str.lower() == crop].copy()
    if view.empty:
        out["warnings"].append("yield_crop_empty")
        return out

    view["year"] = pd.to_numeric(view.get("year"), errors="coerce")
    view["yield_quintal_per_hectare"] = pd.to_numeric(view.get("yield_quintal_per_hectare"), errors="coerce")
    actual_view = view[view["year"] == float(target_year)].copy()
    actual_view = actual_view.dropna(subset=["year", "yield_quintal_per_hectare"]).sort_values("year")
    out["actual"] = [
        {"year": int(year), "value": float(val)}
        for year, val in zip(actual_view["year"], actual_view["yield_quintal_per_hectare"])
    ]
    view = view[view["year"] <= float(cutoff_year)].copy()
    view = view.dropna(subset=["year", "yield_quintal_per_hectare"]).sort_values("year")
    if view.empty:
        out["warnings"].append("yield_crop_empty")
        return out

    out["history"] = [
        {"year": int(year), "value": float(val)}
        for year, val in zip(view["year"], view["yield_quintal_per_hectare"])
    ]
    out["trend"] = _build_yearly_trend(view, "year", "yield_quintal_per_hectare")
    out["stats"]["latest"] = _safe_float(view["yield_quintal_per_hectare"].iloc[-1])
    out["stats"]["mean"] = _safe_float(view["yield_quintal_per_hectare"].mean())
    out["stats"]["cagr_pct"] = _calc_cagr_pct(view, "year", "yield_quintal_per_hectare")
    return out


def _build_cost_visuals(
    cost_file: Path,
    crop: str,
    cost_name: str,
    cost_pred: float | None,
    target_year: int,
    cutoff_year: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "history": [],
        "trend": [],
        "actual": [],
        "forecast": {"year": int(target_year), "value": _safe_float(cost_pred)},
        "stats": {
            "latest": None,
            "mean": None,
            "cagr_pct": None,
        },
        "warnings": [],
    }

    if not cost_file.exists():
        out["warnings"].append("cost_history_missing")
        return out

    try:
        cost_df = load_cost_data(str(cost_file))
    except Exception:
        out["warnings"].append("cost_history_unreadable")
        return out

    if cost_df.empty or "crop_name" not in cost_df.columns:
        out["warnings"].append("cost_history_empty")
        return out

    synonyms = {"lentil": "masur", "rice": "paddy", "chickpea": "gram"}
    candidates: List[str] = []
    if cost_name:
        candidates.append(cost_name)
    candidates.extend([crop, crop.title()])
    if crop in synonyms:
        candidates.append(synonyms[crop])
    candidates = [str(x).strip().lower() for x in candidates if str(x).strip()]

    view = pd.DataFrame()
    for cand in candidates:
        subset = cost_df[cost_df["crop_name"].astype(str).str.strip().str.lower() == cand].copy()
        if not subset.empty:
            view = subset
            break

    if view.empty:
        out["warnings"].append("cost_crop_empty")
        return out

    view["year_start"] = pd.to_numeric(view.get("year_start"), errors="coerce")
    view["india_cost_wavg_sample"] = pd.to_numeric(view.get("india_cost_wavg_sample"), errors="coerce")
    actual_view = view[view["year_start"] == float(target_year)].copy()
    actual_view = actual_view.dropna(subset=["year_start", "india_cost_wavg_sample"]).sort_values("year_start")
    out["actual"] = [
        {"year": int(year), "value": float(val)}
        for year, val in zip(actual_view["year_start"], actual_view["india_cost_wavg_sample"])
    ]
    view = view[view["year_start"] <= float(cutoff_year)].copy()
    view = view.dropna(subset=["year_start", "india_cost_wavg_sample"]).sort_values("year_start")
    if view.empty:
        out["warnings"].append("cost_crop_empty")
        return out

    out["history"] = [
        {"year": int(year), "value": float(val)}
        for year, val in zip(view["year_start"], view["india_cost_wavg_sample"])
    ]
    out["trend"] = _build_yearly_trend(view, "year_start", "india_cost_wavg_sample")
    out["stats"]["latest"] = _safe_float(view["india_cost_wavg_sample"].iloc[-1])
    out["stats"]["mean"] = _safe_float(view["india_cost_wavg_sample"].mean())
    out["stats"]["cagr_pct"] = _calc_cagr_pct(view, "year_start", "india_cost_wavg_sample")
    return out


def build_crop_visual_payload(
    *,
    crop: str,
    price_file: str,
    cost_name: str,
    price_pred: float | None,
    price_forecast: Optional[List[Dict[str, Any]]],
    yield_pred: float | None,
    cost_pred: float | None,
    cost_pred_raw: float | None,
    profit_pred: float | None,
    env_prob: float | None,
    prob_best: float | None,
    risk: float | None,
    score: float | None,
    target_year: int,
    history_years: int,
    prediction_start_date: pd.Timestamp,
    prediction_end_date: pd.Timestamp,
    horizon_days: int,
    price_summary_window_days: int,
    price_dir_path: Path,
    cost_file_path: Path,
    yield_history_path: Path,
    alignment_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    canonical_price_file = canonicalize_price_file(price_file)
    cutoff_year = int(prediction_start_date.year)
    price_panel = _build_price_visuals(
        price_dir_path,
        canonical_price_file,
        price_pred,
        price_forecast,
        horizon_days,
        history_years,
        as_of_date=prediction_start_date,
        actual_start_date=prediction_start_date,
        actual_end_date=prediction_end_date,
    )
    price_pred_effective = _safe_float(price_pred)
    price_pred_p10 = None
    price_pred_p50 = _safe_float(price_pred)
    price_pred_p90 = None
    forecast_rows = price_panel.get("forecast", [])
    if isinstance(forecast_rows, list) and forecast_rows:
        price_summary = summarize_forecast_tail(
            forecast_rows,
            window_days=price_summary_window_days,
            fallback_price=price_pred_effective,
        )
        price_pred_effective = _safe_float(price_summary.get("price_pred"))
        price_pred_p10 = _safe_float(price_summary.get("price_p10"))
        price_pred_p50 = _safe_float(price_summary.get("price_p50")) or price_pred_effective
        price_pred_p90 = _safe_float(price_summary.get("price_p90"))
    if price_pred_p50 is None:
        price_pred_p50 = price_pred_effective
    yield_panel = _build_yield_visuals(
        yield_history_path,
        crop,
        yield_pred,
        target_year=int(target_year),
        cutoff_year=cutoff_year,
    )
    cost_panel = _build_cost_visuals(
        cost_file_path,
        crop,
        cost_name,
        cost_pred,
        target_year=int(target_year),
        cutoff_year=cutoff_year,
    )

    aligned_payload = build_aligned_visual_payload(
        price_history_rows=price_panel.get("history", []),
        yield_history_rows=yield_panel.get("history", []),
        cost_history_rows=cost_panel.get("history", []),
        target_year=target_year,
        price_pred=price_pred_effective,
        yield_pred=yield_pred,
        cost_pred=cost_pred,
        policy=AlignmentPolicy(
            frequency="year",
            history_years=history_years,
            strategy=str(alignment_cfg.get("strategy", "trend_extrapolate_with_uncertainty")),
            min_trend_points=max(2, _safe_int(alignment_cfg.get("min_trend_points")) or 3),
        ),
    )
    aligned_time_index = aligned_payload.get("time_index", [])
    aligned_series = aligned_payload.get("series", {}) if isinstance(aligned_payload, dict) else {}
    aligned_price = aligned_series.get("price", {}) if isinstance(aligned_series, dict) else {}
    aligned_yield = aligned_series.get("yield", {}) if isinstance(aligned_series, dict) else {}
    aligned_cost = aligned_series.get("cost", {}) if isinstance(aligned_series, dict) else {}

    price_history_aligned = _aligned_to_date_rows(aligned_time_index, aligned_price.get("history", []))
    price_forecast_aligned = _aligned_to_date_rows(aligned_time_index, aligned_price.get("forecast", []))
    price_ma30_aligned = _aligned_ma_from_date_rows(aligned_time_index, price_panel.get("ma30", []))
    price_ma90_aligned = _aligned_ma_from_date_rows(aligned_time_index, price_panel.get("ma90", []))
    yield_history_aligned = _aligned_to_year_rows(aligned_time_index, aligned_yield.get("history", []))
    yield_forecast_aligned = _aligned_forecast_point(aligned_time_index, aligned_yield.get("forecast", []))
    cost_history_aligned = _aligned_to_year_rows(aligned_time_index, aligned_cost.get("history", []))
    cost_forecast_aligned = _aligned_forecast_point(aligned_time_index, aligned_cost.get("forecast", []))
    yield_trend_aligned = _trend_from_year_rows(yield_history_aligned)
    cost_trend_aligned = _trend_from_year_rows(cost_history_aligned)

    if profit_pred is None and price_pred_effective is not None and yield_pred is not None and cost_pred is not None:
        profit_pred = _safe_float(price_pred_effective * yield_pred - cost_pred)

    revenue_pred = (
        _safe_float(price_pred_effective * yield_pred) if price_pred_effective is not None and yield_pred is not None else None
    )
    margin_pred = None
    if revenue_pred is not None and revenue_pred > 0 and profit_pred is not None:
        margin_pred = _safe_float((profit_pred / revenue_pred) * 100.0)

    warnings = (
        price_panel.get("warnings", [])
        + yield_panel.get("warnings", [])
        + cost_panel.get("warnings", [])
        + aligned_payload.get("warnings", [])
    )

    return {
        "crop": crop,
        "resolved": {
            "price_file": canonical_price_file or None,
            "cost_name": cost_name or None,
        },
        "profile": {
            "env_prob": env_prob,
            "prob_best": prob_best,
            "risk": risk,
            "score": score,
        },
        "prediction": {
            "year": int(target_year),
            "price_pred": price_pred_effective,
            "price_p10": price_pred_p10,
            "price_p50": price_pred_p50,
            "price_p90": price_pred_p90,
            "yield_pred": yield_pred,
            "cost_pred": cost_pred,
            "cost_pred_raw": cost_pred_raw,
            "revenue_pred": revenue_pred,
            "profit_pred": profit_pred,
            "margin_pct": margin_pred,
            "horizon_days": horizon_days,
        },
        "time_index": aligned_payload.get("time_index", []),
        "time_meta": aligned_payload.get("time_meta", {}),
        "prediction_window": {
            "start_date": prediction_start_date.strftime("%Y-%m-%d"),
            "end_date": prediction_end_date.strftime("%Y-%m-%d"),
            "price_horizon_days": int(horizon_days),
            "price_summary_window_days": int(max(1, price_summary_window_days)),
        },
        "alignment_notes": aligned_payload.get("notes", []),
        "price": {
            "history": price_history_aligned or price_panel.get("history", []),
            "ma30": price_ma30_aligned or price_panel.get("ma30", []),
            "ma90": price_ma90_aligned or price_panel.get("ma90", []),
            "forecast": price_forecast_aligned or price_panel.get("forecast", []),
            "actual": price_panel.get("actual", []),
            "raw": {
                "history": price_panel.get("history", []),
                "ma30": price_panel.get("ma30", []),
                "ma90": price_panel.get("ma90", []),
                "forecast": price_panel.get("forecast", []),
                "actual": price_panel.get("actual", []),
            },
            "stats": price_panel.get("stats", {}),
            "aligned": (aligned_payload.get("series", {}) or {}).get("price", {}),
        },
        "yield": {
            "history": yield_history_aligned or yield_panel.get("history", []),
            "trend": yield_trend_aligned or yield_panel.get("trend", []),
            "actual": yield_panel.get("actual", []),
            "forecast": yield_forecast_aligned or yield_panel.get("forecast", {}),
            "stats": yield_panel.get("stats", {}),
            "aligned": (aligned_payload.get("series", {}) or {}).get("yield", {}),
        },
        "cost": {
            "history": cost_history_aligned or cost_panel.get("history", []),
            "trend": cost_trend_aligned or cost_panel.get("trend", []),
            "actual": cost_panel.get("actual", []),
            "forecast": cost_forecast_aligned or cost_panel.get("forecast", {}),
            "stats": cost_panel.get("stats", {}),
            "aligned": (aligned_payload.get("series", {}) or {}).get("cost", {}),
        },
        "warnings": list(dict.fromkeys([w for w in warnings if w])),
    }
