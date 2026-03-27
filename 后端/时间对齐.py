from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_UNCERTAINTY_BY_SOURCE: Dict[str, float] = {
    "observed": 0.10,
    "interpolated": 0.30,
    "extrapolated": 0.65,
    "forward_fill": 0.80,
    "model_prediction": 0.40,
    "unavailable": 1.00,
}


@dataclass
class AlignmentPolicy:
    frequency: str = "year"
    history_years: int = 8
    strategy: str = "trend_extrapolate_with_uncertainty"
    min_trend_points: int = 3
    uncertainty_by_source: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_UNCERTAINTY_BY_SOURCE)
    )


def _safe_float(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except Exception:
        return None
    if pd.isna(val):
        return None
    return float(val)


def _safe_year(value: Any) -> Optional[int]:
    try:
        year = int(value)
    except Exception:
        return None
    if year < 1900 or year > 2200:
        return None
    return year


def _clip_non_negative(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(max(0.0, value))


def _fit_line(xs: List[int], ys: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if len(xs) < 2 or len(ys) < 2:
        return None, None
    if len(set(xs)) < 2:
        return None, None
    try:
        slope, intercept = np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), deg=1)
    except Exception:
        return None, None
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None, None
    return float(slope), float(intercept)


def _interpolate_between(points: Mapping[int, float], year: int) -> Optional[float]:
    years = sorted(points.keys())
    if len(years) < 2:
        return None
    prev_year = None
    next_year = None
    for y in years:
        if y < year:
            prev_year = y
        if y > year:
            next_year = y
            break
    if prev_year is None or next_year is None or next_year == prev_year:
        return None
    y0 = points[prev_year]
    y1 = points[next_year]
    ratio = float(year - prev_year) / float(next_year - prev_year)
    return y0 + (y1 - y0) * ratio


def yearly_map_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    year_key: str = "year",
    value_key: str = "value",
) -> Dict[int, float]:
    points: Dict[int, float] = {}
    for row in rows or []:
        year = _safe_year(row.get(year_key))
        value = _safe_float(row.get(value_key))
        if year is None or value is None:
            continue
        points[year] = float(value)
    return points


def price_yearly_map_from_history(rows: Iterable[Mapping[str, Any]]) -> Dict[int, float]:
    # Price history is daily in server response; align by annual mean.
    buckets: Dict[int, List[float]] = {}
    for row in rows or []:
        date_text = str(row.get("date", "")).strip()
        value = _safe_float(row.get("value"))
        if not date_text or value is None:
            continue
        try:
            year = pd.Timestamp(date_text).year
        except Exception:
            continue
        buckets.setdefault(int(year), []).append(float(value))

    out: Dict[int, float] = {}
    for year, values in buckets.items():
        if not values:
            continue
        out[int(year)] = float(np.mean(values))
    return out


def build_year_index(
    series_maps: Iterable[Mapping[int, float]],
    target_year: int,
    history_years: int,
) -> List[int]:
    target = _safe_year(target_year) or int(datetime.now().year)
    hist_years = max(2, int(history_years))
    lower_bound = target - hist_years + 1

    starts: List[int] = []
    for series in series_maps:
        if series:
            starts.append(min(int(y) for y in series.keys()))

    start_year = max(lower_bound, min(starts)) if starts else lower_bound
    if start_year > target:
        start_year = target
    return list(range(int(start_year), int(target) + 1))


def align_yearly_series(
    observed_points: Mapping[int, float],
    years: List[int],
    *,
    target_year: int,
    model_prediction: Optional[float],
    policy: AlignmentPolicy,
) -> Dict[str, Any]:
    obs = {int(k): float(v) for k, v in observed_points.items()}
    target = int(target_year)

    history: List[Optional[float]] = []
    forecast: List[Optional[float]] = []
    source: List[str] = []
    uncertainty: List[float] = []

    if obs:
        sorted_years = sorted(obs.keys())
        min_year = sorted_years[0]
        max_year = sorted_years[-1]

        tail_n = max(2, int(policy.min_trend_points))
        tail_years = sorted_years[-tail_n:]
        tail_vals = [obs[y] for y in tail_years]
        trend_slope, trend_intercept = _fit_line(tail_years, tail_vals)

        head_years = sorted_years[:tail_n]
        head_vals = [obs[y] for y in head_years]
        back_slope, back_intercept = _fit_line(head_years, head_vals)
    else:
        min_year = None
        max_year = None
        trend_slope = None
        trend_intercept = None
        back_slope = None
        back_intercept = None

    def estimate_value(year: int) -> Tuple[Optional[float], str]:
        if year in obs:
            return _clip_non_negative(obs[year]), "observed"
        if not obs:
            return None, "unavailable"

        if min_year is not None and max_year is not None and min_year < year < max_year:
            val = _interpolate_between(obs, year)
            if val is not None:
                return _clip_non_negative(val), "interpolated"
            return None, "unavailable"

        if max_year is not None and year > max_year:
            if trend_slope is not None and trend_intercept is not None:
                val = trend_slope * float(year) + trend_intercept
                return _clip_non_negative(val), "extrapolated"
            return _clip_non_negative(obs.get(max_year)), "forward_fill"

        if min_year is not None and year < min_year:
            if back_slope is not None and back_intercept is not None:
                val = back_slope * float(year) + back_intercept
                return _clip_non_negative(val), "extrapolated"
            return _clip_non_negative(obs.get(min_year)), "forward_fill"

        return None, "unavailable"

    for year in years:
        est_value, est_source = estimate_value(int(year))

        if int(year) < target:
            history.append(est_value)
            forecast.append(None)
            point_source = est_source
        elif int(year) == target:
            pred = _clip_non_negative(_safe_float(model_prediction))
            if pred is not None:
                history.append(None)
                forecast.append(pred)
                point_source = "model_prediction"
            else:
                history.append(None)
                forecast.append(est_value)
                point_source = est_source
        else:
            history.append(None)
            forecast.append(None)
            point_source = "unavailable"

        source.append(point_source)
        uncertainty.append(float(policy.uncertainty_by_source.get(point_source, 1.0)))

    counts: Dict[str, int] = {}
    for item in source:
        counts[item] = counts.get(item, 0) + 1

    last_observed = int(max(obs.keys())) if obs else None
    first_observed = int(min(obs.keys())) if obs else None
    gap_to_target = max(0, int(target) - int(last_observed)) if last_observed is not None else None

    return {
        "history": history,
        "forecast": forecast,
        "source": source,
        "uncertainty": uncertainty,
        "coverage": {
            "first_observed_year": first_observed,
            "last_observed_year": last_observed,
            "gap_to_target_years": gap_to_target,
            "source_counts": counts,
            "mean_uncertainty": float(np.mean(uncertainty)) if uncertainty else None,
        },
    }


def build_aligned_visual_payload(
    *,
    price_history_rows: Iterable[Mapping[str, Any]],
    yield_history_rows: Iterable[Mapping[str, Any]],
    cost_history_rows: Iterable[Mapping[str, Any]],
    target_year: Optional[int],
    price_pred: Optional[float],
    yield_pred: Optional[float],
    cost_pred: Optional[float],
    policy: Optional[AlignmentPolicy] = None,
) -> Dict[str, Any]:
    cfg = policy or AlignmentPolicy()
    if str(cfg.frequency).lower() != "year":
        raise ValueError("only yearly alignment is supported in current implementation")

    resolved_target_year = _safe_year(target_year) or int(datetime.now().year)
    price_map = price_yearly_map_from_history(price_history_rows)
    yield_map = yearly_map_from_rows(yield_history_rows)
    cost_map = yearly_map_from_rows(cost_history_rows)

    years = build_year_index(
        series_maps=[price_map, yield_map, cost_map],
        target_year=resolved_target_year,
        history_years=cfg.history_years,
    )
    time_index = [str(y) for y in years]

    aligned_price = align_yearly_series(
        price_map,
        years,
        target_year=resolved_target_year,
        model_prediction=price_pred,
        policy=cfg,
    )
    aligned_yield = align_yearly_series(
        yield_map,
        years,
        target_year=resolved_target_year,
        model_prediction=yield_pred,
        policy=cfg,
    )
    aligned_cost = align_yearly_series(
        cost_map,
        years,
        target_year=resolved_target_year,
        model_prediction=cost_pred,
        policy=cfg,
    )

    notes: List[str] = []
    warnings: List[str] = []
    for name, aligned in (
        ("price", aligned_price),
        ("yield", aligned_yield),
        ("cost", aligned_cost),
    ):
        coverage = aligned.get("coverage", {}) if isinstance(aligned, dict) else {}
        gap = coverage.get("gap_to_target_years")
        if isinstance(gap, int) and gap > 0:
            notes.append(f"{name} history does not reach target year; gap={gap} years")
            warnings.append(f"{name}_history_gap_to_target")
        if coverage.get("last_observed_year") is None:
            notes.append(f"{name} history is unavailable")
            warnings.append(f"{name}_history_missing")

    return {
        "time_index": time_index,
        "time_meta": {
            "frequency": "year",
            "target_year": int(resolved_target_year),
            "strategy": cfg.strategy,
            "history_years": int(cfg.history_years),
            "note": "Aligned on yearly index in backend; interpolation/extrapolation uncertainty is provided.",
        },
        "series": {
            "price": aligned_price,
            "yield": aligned_yield,
            "cost": aligned_cost,
        },
        "notes": notes,
        "warnings": list(dict.fromkeys([w for w in warnings if w])),
    }

