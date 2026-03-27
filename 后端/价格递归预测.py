from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from 后端.数据清洗 import clean_price_series_frame


PRICE_FLOOR = 1e-6
REQUIRED_EXOG_FEATURES = ("min_price", "max_price", "change")
_ONEHOT_ZERO_PREFIXES = ("dow_", "month_", "week_")


def _safe_float(v: Any) -> Optional[float]:
    try:
        out = float(v)
    except Exception:
        return None
    if pd.isna(out):
        return None
    return out


def _prepare_daily_panel(history_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_price_series_frame(history_df, exog_cols=REQUIRED_EXOG_FEATURES)
    if cleaned.empty:
        return pd.DataFrame()
    panel = cleaned.set_index("date")
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    if panel.empty:
        return pd.DataFrame()
    return panel


def _prepare_daily_history(history_df: pd.DataFrame) -> pd.Series:
    panel = _prepare_daily_panel(history_df)
    if panel.empty or "modal_price" not in panel.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(panel["modal_price"], errors="coerce")
    s = s.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return s.astype(float)


def _safe_log_price(values: np.ndarray, floor: float = PRICE_FLOOR) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.log(np.clip(arr, max(float(floor), 1e-9), None))


def _build_exogenous_state(history_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    panel = _prepare_daily_panel(history_df)
    if panel.empty or "modal_price" not in panel.columns:
        return None
    modal = pd.to_numeric(panel["modal_price"], errors="coerce").dropna()
    if modal.empty:
        return None

    last_modal = float(modal.iloc[-1])
    prev_modal = float(modal.iloc[-2]) if len(modal) >= 2 else float(last_modal)
    ratio_min = 0.93
    ratio_max = 1.07

    source_flags = {"min_price": "derived", "max_price": "derived", "change": "derived"}
    if "min_price" in panel.columns:
        r = pd.to_numeric(panel["min_price"], errors="coerce") / modal
        r = r.replace([np.inf, -np.inf], np.nan).dropna()
        r = r[(r > 0.2) & (r <= 1.0)]
        if not r.empty:
            ratio_min = float(np.clip(np.median(r.to_numpy(dtype=float)), 0.5, 1.0))
            source_flags["min_price"] = "observed"
    if "max_price" in panel.columns:
        r = pd.to_numeric(panel["max_price"], errors="coerce") / modal
        r = r.replace([np.inf, -np.inf], np.nan).dropna()
        r = r[(r >= 1.0) & (r < 3.0)]
        if not r.empty:
            ratio_max = float(np.clip(np.median(r.to_numpy(dtype=float)), 1.0, 2.0))
            source_flags["max_price"] = "observed"

    last_min_obs = _safe_float(panel["min_price"].iloc[-1]) if "min_price" in panel.columns else None
    last_max_obs = _safe_float(panel["max_price"].iloc[-1]) if "max_price" in panel.columns else None
    last_change_obs = _safe_float(panel["change"].iloc[-1]) if "change" in panel.columns else None
    if last_change_obs is not None:
        source_flags["change"] = "observed"

    last_min = float(last_min_obs) if last_min_obs is not None else float(max(PRICE_FLOOR, last_modal * ratio_min))
    last_max = float(last_max_obs) if last_max_obs is not None else float(max(last_min, last_modal * ratio_max))
    if last_min > last_max:
        last_min, last_max = last_max, last_min

    if last_change_obs is None:
        last_change = float(last_modal - prev_modal)
    else:
        last_change = float(last_change_obs)

    return {
        "min_ratio": float(np.clip(ratio_min, 0.5, 1.0)),
        "max_ratio": float(np.clip(ratio_max, 1.0, 2.0)),
        "current_min": float(max(PRICE_FLOOR, last_min)),
        "current_max": float(max(max(last_min, last_max), PRICE_FLOOR)),
        "current_change": float(last_change),
        "source_flags": source_flags,
    }


def _exogenous_row_from_state(state: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not isinstance(state, dict):
        return {}
    out: Dict[str, float] = {}
    min_v = _safe_float(state.get("current_min"))
    max_v = _safe_float(state.get("current_max"))
    chg_v = _safe_float(state.get("current_change"))
    if min_v is not None:
        out["min_price"] = float(max(PRICE_FLOOR, min_v))
    if max_v is not None:
        out["max_price"] = float(max(PRICE_FLOOR, max_v))
    if chg_v is not None:
        out["change"] = float(chg_v)
    if "min_price" in out and "max_price" in out and out["min_price"] > out["max_price"]:
        out["min_price"], out["max_price"] = out["max_price"], out["min_price"]
    return out


def _update_exogenous_state(state: Optional[Dict[str, Any]], *, prev_price: float, next_price: float) -> None:
    if not isinstance(state, dict):
        return
    prev_p = max(float(prev_price), PRICE_FLOOR)
    next_p = max(float(next_price), PRICE_FLOOR)

    ratio_min = float(np.clip(_safe_float(state.get("min_ratio")) or 0.93, 0.5, 1.0))
    ratio_max = float(np.clip(_safe_float(state.get("max_ratio")) or 1.07, 1.0, 2.0))

    next_min = max(PRICE_FLOOR, next_p * ratio_min)
    next_max = max(next_min, next_p * ratio_max)
    state["current_min"] = float(next_min)
    state["current_max"] = float(next_max)
    state["current_change"] = float(next_p - prev_p)


def _prepare_model_input(
    feature_row: Dict[str, float],
    feature_cols: Optional[List[str]],
    *,
    required_features: Optional[List[str]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if not isinstance(feature_row, dict) or not feature_row:
        raise ValueError("feature_row_empty")

    parsed: Dict[str, float] = {}
    for k, v in feature_row.items():
        fv = _safe_float(v)
        if fv is None:
            continue
        parsed[str(k)] = float(fv)

    if not feature_cols:
        frame = pd.DataFrame([parsed]).apply(pd.to_numeric, errors="coerce")
        if frame.isna().any().any():
            raise ValueError("feature_row_contains_nan")
        return frame

    cols = [str(c) for c in feature_cols]
    required = {str(c) for c in (required_features or []) if str(c).strip()}
    missing = [c for c in cols if c not in parsed]
    missing_required = [c for c in missing if c in required]
    if missing_required:
        raise ValueError(f"missing_required_features:{','.join(missing_required)}")

    filled_zero_cols: List[str] = []
    missing_nonoptional: List[str] = []
    for col in missing:
        if col.startswith(_ONEHOT_ZERO_PREFIXES):
            parsed[col] = 0.0
            filled_zero_cols.append(col)
            continue
        missing_nonoptional.append(col)
    if missing_nonoptional:
        raise ValueError(f"missing_nonoptional_features:{','.join(missing_nonoptional)}")

    frame = pd.DataFrame([{c: parsed.get(c, 0.0) for c in cols}], columns=cols)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any():
        bad = [str(c) for c in frame.columns[frame.isna().any()].tolist()]
        raise ValueError(f"feature_nan_after_cast:{','.join(bad)}")

    if isinstance(diagnostics, dict):
        schema = diagnostics.setdefault("feature_schema", {})
        schema["feature_cols_count"] = int(len(cols))
        schema["required_features"] = sorted(required)
        schema["zero_filled_optional_count"] = int(len(filled_zero_cols))
        if filled_zero_cols:
            schema["zero_filled_optional_cols"] = filled_zero_cols[:20]

    return frame


def _augment_lags_windows_from_feature_cols(
    lags: List[int],
    windows: List[int],
    feature_cols: Optional[List[str]],
) -> tuple[List[int], List[int]]:
    lag_set = {int(x) for x in (lags or []) if int(x) > 0}
    window_set = {int(x) for x in (windows or []) if int(x) > 0}
    for col in feature_cols or []:
        name = str(col)
        if name.startswith("lag_"):
            tail = name[4:]
            if tail.isdigit():
                lag_set.add(int(tail))
            continue
        for pref in ("roll_mean_", "roll_std_", "roll_min_", "roll_max_"):
            if name.startswith(pref):
                tail = name[len(pref) :]
                if tail.isdigit():
                    window_set.add(int(tail))
                break
    return sorted(lag_set), sorted(window_set)


def estimate_daily_move_limit(history_df: pd.DataFrame) -> float:
    """Legacy price-space clip estimator for v1 path."""
    s = _prepare_daily_history(history_df)
    if s.empty:
        return 0.08
    returns = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 10:
        return 0.08
    vol = _safe_float(returns.std())
    if vol is None:
        return 0.08
    return float(np.clip(vol * 3.0 + 0.01, 0.03, 0.25))


def estimate_return_clip_limit(
    history_df: pd.DataFrame,
    *,
    quantile: float = 0.98,
    safety_factor: float = 1.2,
    floor: float = PRICE_FLOOR,
) -> float:
    s = _prepare_daily_history(history_df)
    if s.empty or len(s) < 15:
        return 0.06
    lr = np.diff(_safe_log_price(s.to_numpy(dtype=float), floor=floor))
    lr = lr[np.isfinite(lr)]
    if lr.size < 10:
        return 0.06
    q = float(np.clip(_safe_float(quantile) or 0.98, 0.80, 0.999))
    sf = float(np.clip(_safe_float(safety_factor) or 1.2, 1.0, 3.0))
    base = float(np.quantile(np.abs(lr), q))
    return float(np.clip(base * sf, 0.003, 0.35))


def _normalize_time_raw_mode(include_raw_time_features: bool, time_raw_mode: str | None) -> str:
    if time_raw_mode is None:
        return "raw" if include_raw_time_features else "none"
    mode = str(time_raw_mode).strip().lower()
    if mode not in {"none", "raw", "onehot"}:
        return "none"
    return mode


def _time_feature_row(next_date: pd.Timestamp, *, include_raw_time_features: bool, time_raw_mode: str | None) -> Dict[str, float]:
    dow = int(next_date.dayofweek)
    month = int(next_date.month)
    iso = next_date.isocalendar()
    weekofyear = int(getattr(iso, "week", iso[1]))
    dayofyear = int(next_date.dayofyear)

    row: Dict[str, float] = {
        "doy_sin": float(np.sin(2.0 * np.pi * dayofyear / 365.25)),
        "doy_cos": float(np.cos(2.0 * np.pi * dayofyear / 365.25)),
        "dow_sin": float(np.sin(2.0 * np.pi * dow / 7.0)),
        "dow_cos": float(np.cos(2.0 * np.pi * dow / 7.0)),
        "week_sin": float(np.sin(2.0 * np.pi * weekofyear / 52.1775)),
        "week_cos": float(np.cos(2.0 * np.pi * weekofyear / 52.1775)),
        "month_sin": float(np.sin(2.0 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2.0 * np.pi * month / 12.0)),
        "is_weekend": float(1 if dow >= 5 else 0),
    }

    raw_mode = _normalize_time_raw_mode(include_raw_time_features, time_raw_mode)
    if raw_mode == "raw":
        row["dow"] = float(dow)
        row["month"] = float(month)
        row["weekofyear"] = float(weekofyear)
        row["dayofyear"] = float(dayofyear)
        row["day"] = float(int(next_date.day))
    elif raw_mode == "onehot":
        row[f"dow_{dow}"] = 1.0
        row[f"month_{month}"] = 1.0
        row[f"week_{weekofyear}"] = 1.0

    return row


def _build_feature_row(
    *,
    history_dates: List[pd.Timestamp],
    history_values: List[float],
    next_date: pd.Timestamp,
    lags: List[int],
    windows: List[int],
    include_raw_time_features: bool,
    time_raw_mode: str | None,
    exogenous_row: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, float]]:
    if not history_dates or not history_values:
        return None
    max_lag = max([int(x) for x in (lags or [1])] + [1])
    max_window = max([int(x) for x in (windows or [7])] + [1])
    required = max(max_lag, max_window)
    if len(history_values) < required:
        return None

    arr = np.asarray(history_values, dtype=float)
    if arr.size < required or not np.all(np.isfinite(arr[-required:])):
        return None

    row: Dict[str, float] = _time_feature_row(
        next_date,
        include_raw_time_features=include_raw_time_features,
        time_raw_mode=time_raw_mode,
    )
    if isinstance(exogenous_row, dict):
        for k, v in exogenous_row.items():
            fv = _safe_float(v)
            if fv is None:
                continue
            row[str(k)] = float(fv)

    for lag in lags or []:
        lag_i = int(lag)
        if lag_i <= 0 or lag_i > len(arr):
            return None
        row[f"lag_{lag_i}"] = float(arr[-lag_i])

    for window in windows or []:
        w = int(window)
        if w <= 0 or w > len(arr):
            return None
        seg = arr[-w:]
        row[f"roll_mean_{w}"] = float(np.mean(seg))
        row[f"roll_std_{w}"] = float(np.std(seg, ddof=1)) if len(seg) > 1 else 0.0
        row[f"roll_min_{w}"] = float(np.min(seg))
        row[f"roll_max_{w}"] = float(np.max(seg))

    return row


def _coerce_seasonal_map(seasonal_y_by_doy: Optional[Dict[str, Any]]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if not isinstance(seasonal_y_by_doy, dict):
        return out
    for k, v in seasonal_y_by_doy.items():
        try:
            key = int(k)
            val = float(v)
        except Exception:
            continue
        if key < 1 or key > 366 or not np.isfinite(val):
            continue
        out[key] = val
    return out


def _coerce_horizon_scale_points(points: Any) -> List[tuple[int, float]]:
    parsed: List[tuple[int, float]] = []
    if isinstance(points, dict):
        iterable = [{"horizon_days": k, "scale": v} for k, v in points.items()]
    elif isinstance(points, list):
        iterable = points
    else:
        iterable = []

    for item in iterable:
        h = None
        s = None
        if isinstance(item, dict):
            h = _safe_float(item.get("horizon_days"))
            if h is None:
                h = _safe_float(item.get("h"))
            if h is None:
                h = _safe_float(item.get("step"))
            s = _safe_float(item.get("scale"))
            if s is None:
                s = _safe_float(item.get("q_scale"))
            if s is None:
                s = _safe_float(item.get("multiplier"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            h = _safe_float(item[0])
            s = _safe_float(item[1])

        if h is None or s is None:
            continue
        h_i = int(h)
        s_f = float(s)
        if h_i <= 0 or not np.isfinite(s_f):
            continue
        parsed.append((h_i, max(1.0, s_f)))

    if not parsed:
        return []
    parsed = sorted(parsed, key=lambda x: x[0])

    dedup: List[tuple[int, float]] = []
    for h, s in parsed:
        if dedup and h == dedup[-1][0]:
            dedup[-1] = (h, max(dedup[-1][1], s))
        else:
            dedup.append((h, s))

    mono: List[tuple[int, float]] = []
    running = 1.0
    for h, s in dedup:
        running = max(running, float(s))
        mono.append((h, running))
    return mono


def _resolve_interval_growth_scale(
    *,
    step_idx: int,
    horizon_scale_points: List[tuple[int, float]],
    use_horizon_scale: bool,
    interval_power: float,
    interval_scale: float,
    interval_cap: float,
) -> tuple[float, str]:
    fallback = float(min(interval_cap, max(1.0, (float(step_idx) ** interval_power) * interval_scale)))
    if not use_horizon_scale or not horizon_scale_points:
        return fallback, "power_rule"

    target = int(max(1, step_idx))
    if target <= horizon_scale_points[0][0]:
        return float(min(interval_cap, max(1.0, horizon_scale_points[0][1]))), "horizon_scale_points"
    if target >= horizon_scale_points[-1][0]:
        return float(min(interval_cap, max(1.0, horizon_scale_points[-1][1]))), "horizon_scale_points"

    for idx in range(1, len(horizon_scale_points)):
        h1, s1 = horizon_scale_points[idx - 1]
        h2, s2 = horizon_scale_points[idx]
        if target > h2:
            continue
        if h2 <= h1:
            return float(min(interval_cap, max(1.0, s2))), "horizon_scale_points"
        w = float(target - h1) / float(h2 - h1)
        out = (1.0 - w) * float(s1) + w * float(s2)
        return float(min(interval_cap, max(1.0, out))), "horizon_scale_points"

    return fallback, "power_rule"


def _robust_return_std_from_log_path(history_log: List[float], *, window: int) -> float:
    w = max(5, int(window))
    if len(history_log) < 3:
        return 0.0
    arr = np.asarray(history_log[-(w + 1) :], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return 0.0
    rets = np.diff(arr)
    rets = rets[np.isfinite(rets)]
    if rets.size < 2:
        return 0.0
    med = float(np.median(rets))
    mad = float(np.median(np.abs(rets - med)))
    if np.isfinite(mad) and mad > 0:
        return float(1.4826 * mad)
    if rets.size > 2:
        return float(np.std(rets, ddof=1))
    return float(np.std(rets))


def _seasonal_value_with_fallback(seasonal_map: Dict[int, float], doy: int) -> Optional[float]:
    if not seasonal_map:
        return None
    day = int(np.clip(int(doy), 1, 366))
    for offset in (0, -1, 1, -2, 2, -3, 3, -7, 7, -14, 14):
        key = int(((day - 1 + offset) % 366) + 1)
        val = seasonal_map.get(key)
        if val is None:
            continue
        fv = _safe_float(val)
        if fv is None:
            continue
        return float(fv)
    return None


def _seasonal_return_map_from_levels(seasonal_map: Dict[int, float]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if not seasonal_map:
        return out
    for day in range(1, 367):
        cur = _seasonal_value_with_fallback(seasonal_map, day)
        prev = _seasonal_value_with_fallback(seasonal_map, 366 if day == 1 else day - 1)
        if cur is None or prev is None:
            continue
        out[day] = float(cur - prev)
    return out


def _slope_sign_from_log_path(history_log: List[float], *, window: int) -> int:
    w = max(5, int(window))
    if len(history_log) < w:
        w = len(history_log)
    if w < 5:
        return 0
    arr = np.asarray(history_log[-w:], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 5:
        return 0
    x = np.arange(arr.size, dtype=float)
    try:
        slope = float(np.polyfit(x, arr, deg=1)[0])
    except Exception:
        return 0
    dif = np.diff(arr)
    eps = 1e-6
    if dif.size:
        eps = max(eps, float(np.std(dif)) * 0.05)
    if abs(slope) <= eps:
        return 0
    return 1 if slope > 0 else -1


def _legacy_recursive_price_v1(
    *,
    model,
    history_df: pd.DataFrame,
    horizon_days: int,
    lags: List[int],
    windows: List[int],
    feature_cols: Optional[List[str]],
    max_daily_move_pct: Optional[float],
    anchor_end_value: Optional[float],
    anchor_max_blend: float,
    mean_reversion_strength: float,
    seasonal_strength: float,
    corridor_low_quantile: float,
    corridor_high_quantile: float,
    corridor_low_multiplier: float,
    corridor_high_multiplier: float,
    terminal_low_ratio_vs_latest: float,
    terminal_high_ratio_vs_latest: float,
) -> List[Dict[str, Any]]:
    safe_horizon = max(1, int(horizon_days))
    s = _prepare_daily_history(history_df)
    if s.empty:
        return []

    history_dates = [pd.Timestamp(x) for x in s.index.to_list()]
    history_prices = [float(x) for x in s.to_numpy(dtype=float).tolist()]
    if not history_dates or not history_prices:
        return []
    required_exog_features = [f for f in REQUIRED_EXOG_FEATURES if feature_cols and f in feature_cols]
    exog_state = _build_exogenous_state(history_df) if required_exog_features else None
    if required_exog_features and exog_state is None:
        return []
    effective_lags, effective_windows = _augment_lags_windows_from_feature_cols(lags, windows, feature_cols)

    clip_pct = _safe_float(max_daily_move_pct)
    if clip_pct is not None:
        clip_pct = float(np.clip(clip_pct, 0.0, 0.5))
    anchor_val = _safe_float(anchor_end_value)
    if anchor_val is not None:
        anchor_val = max(1e-6, float(anchor_val))
    anchor_blend = float(np.clip(_safe_float(anchor_max_blend) or 0.65, 0.0, 0.95))
    mr_strength = float(np.clip(_safe_float(mean_reversion_strength) or 0.18, 0.0, 0.9))
    season_strength = float(np.clip(_safe_float(seasonal_strength) or 0.22, 0.0, 0.9))
    q_low = float(np.clip(_safe_float(corridor_low_quantile) or 0.05, 0.0, 0.45))
    q_high = float(np.clip(_safe_float(corridor_high_quantile) or 0.95, 0.55, 1.0))
    if q_high <= q_low:
        q_low, q_high = 0.05, 0.95
    low_mul = float(np.clip(_safe_float(corridor_low_multiplier) or 0.75, 0.3, 1.0))
    high_mul = float(np.clip(_safe_float(corridor_high_multiplier) or 1.35, 1.0, 3.0))
    latest_origin = max(1e-6, float(history_prices[-1]))
    lo_terminal_ratio = float(np.clip(_safe_float(terminal_low_ratio_vs_latest) or 0.6, 0.2, 1.0))
    hi_terminal_ratio = float(np.clip(_safe_float(terminal_high_ratio_vs_latest) or 1.8, 1.0, 5.0))
    if lo_terminal_ratio > hi_terminal_ratio:
        lo_terminal_ratio, hi_terminal_ratio = hi_terminal_ratio, lo_terminal_ratio
    lo_terminal = latest_origin * lo_terminal_ratio
    hi_terminal = latest_origin * hi_terminal_ratio
    hist_series = pd.Series(np.asarray(history_prices, dtype=float), index=pd.to_datetime(history_dates))
    seasonal_map = {
        int(day): float(vals.median())
        for day, vals in hist_series.groupby(hist_series.index.dayofyear)
        if len(vals) >= 2
    }

    out: List[Dict[str, Any]] = []
    for step_idx in range(1, safe_horizon + 1):
        prev_date = history_dates[-1]
        prev_price = max(1e-6, float(history_prices[-1]))
        next_date = prev_date + pd.Timedelta(days=1)
        progress = float(step_idx) / float(safe_horizon)

        feat_row = _build_feature_row(
            history_dates=history_dates,
            history_values=history_prices,
            next_date=next_date,
            lags=effective_lags,
            windows=effective_windows,
            include_raw_time_features=True,
            time_raw_mode="raw",
            exogenous_row=_exogenous_row_from_state(exog_state) if required_exog_features else None,
        )
        if not feat_row:
            break

        try:
            X = _prepare_model_input(
                feat_row,
                feature_cols,
                required_features=required_exog_features,
                diagnostics=None,
            )
        except Exception:
            break

        try:
            pred = _safe_float(model.predict(X)[0])
        except Exception:
            pred = None
        if pred is None:
            break

        next_price = max(0.0, float(pred))
        tail = np.asarray(history_prices[-180:], dtype=float)
        tail = tail[np.isfinite(tail)]
        if tail.size >= 20:
            lo = max(1e-6, float(np.quantile(tail, q_low) * low_mul))
            hi = max(lo * 1.05, float(np.quantile(tail, q_high) * high_mul))
        else:
            lo = max(1e-6, prev_price * 0.55)
            hi = max(lo * 1.05, prev_price * 1.8)

        if tail.size >= 10:
            recent_mean = float(np.mean(tail[-min(30, tail.size) :]))
            mr_w = mr_strength * (0.3 + 0.7 * progress)
            next_price = (1.0 - mr_w) * next_price + mr_w * recent_mean

        seasonal_val = seasonal_map.get(int(next_date.dayofyear))
        if seasonal_val is not None and seasonal_val > 0:
            s_w = season_strength * (0.4 + 0.6 * progress)
            next_price = (1.0 - s_w) * next_price + s_w * float(seasonal_val)

        if anchor_val is not None and anchor_val > 0:
            anchor_log = np.log(prev_price) + (np.log(anchor_val) - np.log(prev_price)) * progress
            anchor_step = float(np.exp(anchor_log))
            a_w = anchor_blend * (progress**1.2)
            next_price = (1.0 - a_w) * next_price + a_w * anchor_step

        next_price = float(np.clip(next_price, max(lo, lo_terminal), min(hi, hi_terminal)))
        clip_applied = False
        clip_amount = 0.0
        if clip_pct is not None and prev_price > 0:
            low = prev_price * (1.0 - clip_pct)
            high = prev_price * (1.0 + clip_pct)
            clipped = float(np.clip(next_price, low, high))
            clip_amount = float(abs(clipped - next_price))
            clip_applied = clip_amount > 0.0
            next_price = clipped

        out.append(
            {
                "date": next_date.strftime("%Y-%m-%d"),
                "value": next_price,
                "p50": next_price,
                "clip_applied": bool(clip_applied),
                "clip_amount": float(clip_amount),
            }
        )
        history_dates.append(next_date)
        history_prices.append(next_price)
        _update_exogenous_state(exog_state, prev_price=prev_price, next_price=next_price)

    return out


def _recursive_return_v2(
    *,
    model,
    history_df: pd.DataFrame,
    horizon_days: int,
    lags: List[int],
    windows: List[int],
    feature_cols: Optional[List[str]],
    max_daily_move_pct: Optional[float],
    clip_r_max: Optional[float],
    return_clip_quantile: float,
    return_clip_safety_factor: float,
    return_bias_mean: Optional[float],
    enable_bias_correction: bool,
    conformal_abs_q: Optional[float],
    enable_conformal_interval: bool,
    enable_seasonal_anchor: bool,
    seasonal_y_by_doy: Optional[Dict[str, Any]],
    seasonal_tau_days: float,
    seasonal_min_ml_weight: float,
    clip_rate_warn_threshold: float,
    anchor_end_value: Optional[float],
    endpoint_direction_guard: bool,
    endpoint_opposite_slack: float,
    endpoint_same_dir_cap: float,
    terminal_anchor_enable: bool,
    terminal_anchor_weight: float,
    interval_growth_power: float,
    interval_growth_scale: float,
    interval_growth_cap: float,
    conformal_horizon_scale_points: Optional[List[Dict[str, Any]]],
    conformal_use_horizon_scale: bool,
    conformal_local_vol_window: int,
    conformal_local_vol_ratio_low: float,
    conformal_local_vol_ratio_high: float,
    conformal_local_vol_reference: Optional[float],
    return_smooth_alpha: float,
    return_jerk_clip_sigma: float,
    return_smooth_warmup_steps: int,
    include_raw_time_features: bool,
    time_raw_mode: str | None,
    diagnostics: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    safe_horizon = max(1, int(horizon_days))
    s = _prepare_daily_history(history_df)
    if s.empty:
        return []

    history_dates = [pd.Timestamp(x) for x in s.index.to_list()]
    history_price = np.asarray(s.to_numpy(dtype=float), dtype=float)
    history_y = _safe_log_price(history_price, floor=PRICE_FLOOR)
    history_log = [float(x) for x in history_y.tolist()]

    if not history_dates or not history_log:
        return []
    required_exog_features = [f for f in REQUIRED_EXOG_FEATURES if feature_cols and f in feature_cols]
    exog_state = _build_exogenous_state(history_df) if required_exog_features else None
    if required_exog_features and exog_state is None:
        if isinstance(diagnostics, dict):
            diagnostics["fallback_reason"] = "required_exogenous_features_unavailable"
        return []
    effective_lags, effective_windows = _augment_lags_windows_from_feature_cols(lags, windows, feature_cols)

    r_max = _safe_float(clip_r_max)
    if r_max is None:
        legacy_clip = _safe_float(max_daily_move_pct)
        if legacy_clip is not None:
            r_max = abs(float(legacy_clip))
    if r_max is None:
        r_max = estimate_return_clip_limit(
            history_df,
            quantile=return_clip_quantile,
            safety_factor=return_clip_safety_factor,
            floor=PRICE_FLOOR,
        )
    r_max = float(np.clip(abs(float(r_max)), 0.001, 0.35))

    b = _safe_float(return_bias_mean)
    if b is None:
        b = 0.0
    if not bool(enable_bias_correction):
        b = 0.0

    q_abs = _safe_float(conformal_abs_q)
    if q_abs is None:
        q_abs = 0.0
    if not bool(enable_conformal_interval):
        q_abs = 0.0
    q_abs = max(0.0, float(q_abs))

    seasonal_map = _coerce_seasonal_map(seasonal_y_by_doy)
    if not seasonal_map:
        hist_series = pd.Series(np.asarray(history_log, dtype=float), index=pd.to_datetime(history_dates))
        seasonal_map = {
            int(day): float(vals.median())
            for day, vals in hist_series.groupby(hist_series.index.dayofyear)
            if len(vals) >= 2
        }

    tau = float(np.clip(_safe_float(seasonal_tau_days) or 55.0, 5.0, 500.0))
    min_ml_weight = float(np.clip(_safe_float(seasonal_min_ml_weight) or 0.15, 0.0, 1.0))
    clip_warn = float(np.clip(_safe_float(clip_rate_warn_threshold) or 0.10, 0.0, 1.0))
    direction_guard_enabled = bool(endpoint_direction_guard)
    opposite_slack = float(np.clip(_safe_float(endpoint_opposite_slack) or 1.5, 0.2, 6.0))
    same_dir_cap = float(np.clip(_safe_float(endpoint_same_dir_cap) or 4.0, 1.0, 12.0))
    use_terminal_anchor = bool(terminal_anchor_enable)
    terminal_anchor_base_weight = float(np.clip(_safe_float(terminal_anchor_weight) or 0.35, 0.0, 1.0))
    interval_power = float(np.clip(_safe_float(interval_growth_power) or 0.5, 0.0, 1.5))
    interval_scale = float(np.clip(_safe_float(interval_growth_scale) or 1.0, 0.0, 20.0))
    interval_cap = float(np.clip(_safe_float(interval_growth_cap) or 200.0, 1.0, 500.0))
    horizon_scale_points = _coerce_horizon_scale_points(conformal_horizon_scale_points)
    use_horizon_scale = bool(conformal_use_horizon_scale)
    local_vol_window = int(np.clip(_safe_float(conformal_local_vol_window) or 45, 5, 730))
    local_vol_ratio_low = float(np.clip(_safe_float(conformal_local_vol_ratio_low) or 0.7, 0.2, 1.0))
    local_vol_ratio_high = float(np.clip(_safe_float(conformal_local_vol_ratio_high) or 1.4, 1.0, 5.0))
    if local_vol_ratio_low > local_vol_ratio_high:
        local_vol_ratio_low, local_vol_ratio_high = local_vol_ratio_high, local_vol_ratio_low
    local_vol_reference = _safe_float(conformal_local_vol_reference)
    if local_vol_reference is None or local_vol_reference <= 0:
        local_vol_reference = _robust_return_std_from_log_path(history_log, window=max(45, local_vol_window))
    if local_vol_reference is None or local_vol_reference <= 0:
        local_vol_reference = 0.01
    smooth_alpha = float(np.clip(_safe_float(return_smooth_alpha) or 0.15, 0.0, 0.95))
    jerk_clip_sigma = float(np.clip(_safe_float(return_jerk_clip_sigma) or 2.5, 0.0, 12.0))
    smooth_warmup_steps = max(1, int(_safe_float(return_smooth_warmup_steps) or 5))

    anchor_y = None
    if _safe_float(anchor_end_value) is not None and _safe_float(anchor_end_value) > 0:
        anchor_y = float(np.log(max(_safe_float(anchor_end_value), PRICE_FLOOR)))

    out: List[Dict[str, Any]] = []
    clip_count = 0
    clip_amounts: List[float] = []
    blend_weights: List[float] = []
    direction_guard_count = 0
    direction_guard_adjustments: List[float] = []
    interval_q_steps: List[float] = []
    interval_growth_steps: List[float] = []
    local_vol_ratios: List[float] = []
    interval_sources: List[str] = []
    smooth_adjustments: List[float] = []
    jerk_adjustments: List[float] = []
    predicted_returns: List[float] = []
    prev_q_step = 0.0

    for step_idx in range(1, safe_horizon + 1):
        prev_date = history_dates[-1]
        prev_y = float(history_log[-1])
        next_date = prev_date + pd.Timedelta(days=1)

        feat_row = _build_feature_row(
            history_dates=history_dates,
            history_values=history_log,
            next_date=next_date,
            lags=effective_lags,
            windows=effective_windows,
            include_raw_time_features=include_raw_time_features,
            time_raw_mode=time_raw_mode,
            exogenous_row=_exogenous_row_from_state(exog_state) if required_exog_features else None,
        )
        if not feat_row:
            break

        try:
            X = _prepare_model_input(
                feat_row,
                feature_cols,
                required_features=required_exog_features,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            if isinstance(diagnostics, dict):
                diagnostics.setdefault("feature_schema_errors", []).append(str(exc))
            break

        try:
            pred_r = _safe_float(model.predict(X)[0])
        except Exception:
            pred_r = None
        if pred_r is None:
            break

        r_raw = float(pred_r)
        r_adj = float(r_raw - b)
        r_limited = float(np.clip(r_adj, -r_max, r_max))
        clip_amount = float(abs(r_adj - r_limited))
        clip_applied = clip_amount > 1e-12

        if clip_applied:
            clip_count += 1
            clip_amounts.append(clip_amount)

        r_effective = float(r_limited)
        if direction_guard_enabled and anchor_y is not None:
            remain = max(1, int(safe_horizon - step_idx + 1))
            required_step = float((anchor_y - prev_y) / float(remain))
            required_abs = abs(required_step)
            if required_abs > 1e-10:
                sign_required = 1.0 if required_step > 0 else -1.0
                candidate = float(r_effective)
                guided = candidate
                if candidate * sign_required < 0.0:
                    cap = max(required_abs * opposite_slack, 0.10 * r_max)
                    guided = sign_required * min(cap, r_max)
                else:
                    cap = max(required_abs * same_dir_cap, 0.25 * r_max)
                    if abs(candidate) > cap:
                        guided = sign_required * min(cap, r_max)
                if abs(guided - candidate) > 1e-12:
                    direction_guard_count += 1
                    direction_guard_adjustments.append(float(abs(guided - candidate)))
                    r_effective = float(guided)

        smooth_adjust = 0.0
        jerk_adjust = 0.0
        if step_idx > 1 and (smooth_alpha > 1e-9 or jerk_clip_sigma > 1e-9):
            center_window = min(max(7, local_vol_window), max(7, len(history_log) - 1))
            arr_recent = np.asarray(history_log[-(center_window + 1) :], dtype=float)
            arr_recent = arr_recent[np.isfinite(arr_recent)]
            ret_center = 0.0
            if arr_recent.size >= 3:
                recent_ret = np.diff(arr_recent)
                recent_ret = recent_ret[np.isfinite(recent_ret)]
                if recent_ret.size:
                    ret_center = float(np.median(recent_ret[-min(14, recent_ret.size) :]))

            if smooth_alpha > 1e-9:
                smooth_ramp = min(1.0, float(step_idx - 1) / float(smooth_warmup_steps))
                smooth_w = float(np.clip(smooth_alpha * smooth_ramp, 0.0, smooth_alpha))
                before = float(r_effective)
                r_effective = float((1.0 - smooth_w) * r_effective + smooth_w * ret_center)
                smooth_adjust = float(abs(r_effective - before))

            if jerk_clip_sigma > 1e-9 and predicted_returns:
                before = float(r_effective)
                prev_r = float(predicted_returns[-1])
                ref_std = _robust_return_std_from_log_path(history_log, window=max(9, local_vol_window))
                if ref_std <= 0:
                    ref_std = abs(prev_r) * 0.35
                jerk_cap = max(0.001, float(ref_std) * jerk_clip_sigma)
                r_effective = float(prev_r + np.clip(r_effective - prev_r, -jerk_cap, jerk_cap))
                jerk_adjust = float(abs(r_effective - before))

        r_effective = float(np.clip(r_effective, -r_max, r_max))
        y_ml = prev_y + r_effective

        step0 = float(step_idx - 1)
        ml_weight = float(np.exp(-step0 / tau))
        ml_weight = float(np.clip(ml_weight, min_ml_weight, 1.0))
        if not bool(enable_seasonal_anchor):
            ml_weight = 1.0

        seasonal_y = seasonal_map.get(int(next_date.dayofyear)) if enable_seasonal_anchor else None
        if seasonal_y is None:
            y_blend = y_ml
            ml_weight_effective = 1.0
        else:
            y_blend = ml_weight * y_ml + (1.0 - ml_weight) * float(seasonal_y)
            ml_weight_effective = ml_weight

        if use_terminal_anchor and anchor_y is not None:
            # Keep optional terminal anchor as a weak late-horizon stabilizer.
            if bool(enable_seasonal_anchor):
                anchor_driver = max(0.0, 1.0 - ml_weight_effective)
            else:
                progress = float(step_idx) / float(max(1, safe_horizon))
                anchor_driver = progress**1.2
            anchor_w = float(np.clip(anchor_driver * terminal_anchor_base_weight, 0.0, terminal_anchor_base_weight))
            y_blend = (1.0 - anchor_w) * y_blend + anchor_w * anchor_y

        y_p50 = float(y_blend)
        interval_growth, interval_source = _resolve_interval_growth_scale(
            step_idx=step_idx,
            horizon_scale_points=horizon_scale_points,
            use_horizon_scale=use_horizon_scale,
            interval_power=interval_power,
            interval_scale=interval_scale,
            interval_cap=interval_cap,
        )
        local_vol = _robust_return_std_from_log_path(history_log, window=local_vol_window)
        if local_vol <= 0:
            local_vol = float(local_vol_reference)
        vol_ratio = float(np.clip(local_vol / max(local_vol_reference, 1e-9), local_vol_ratio_low, local_vol_ratio_high))
        q_step_raw = float(q_abs * interval_growth * vol_ratio)
        q_step = float(max(prev_q_step, q_step_raw))
        prev_q_step = q_step
        y_p10 = float(y_p50 - q_step)
        y_p90 = float(y_p50 + q_step)

        p50 = float(np.exp(y_p50))
        p10 = float(np.exp(y_p10))
        p90 = float(np.exp(y_p90))

        out.append(
            {
                "date": next_date.strftime("%Y-%m-%d"),
                "value": p50,
                "p10": p10,
                "p50": p50,
                "p90": p90,
                "blend_weight": float(ml_weight_effective),
                "interval_growth": float(interval_growth),
                "interval_source": interval_source,
                "interval_q_step": float(q_step),
                "local_vol_ratio": float(vol_ratio),
                "clip_applied": bool(clip_applied),
                "clip_amount": float(clip_amount),
                "smooth_adjustment": float(smooth_adjust),
                "jerk_adjustment": float(jerk_adjust),
            }
        )

        blend_weights.append(float(ml_weight_effective))
        interval_q_steps.append(float(q_step))
        interval_growth_steps.append(float(interval_growth))
        local_vol_ratios.append(float(vol_ratio))
        interval_sources.append(str(interval_source))
        smooth_adjustments.append(float(smooth_adjust))
        jerk_adjustments.append(float(jerk_adjust))
        predicted_returns.append(float(r_effective))
        history_dates.append(next_date)
        history_log.append(y_p50)
        _update_exogenous_state(exog_state, prev_price=float(np.exp(prev_y)), next_price=p50)

    total = max(1, len(out))
    clip_rate = float(clip_count) / float(total)
    clip_amount_mean = float(np.mean(clip_amounts)) if clip_amounts else 0.0
    blend_weight_mean = float(np.mean(blend_weights)) if blend_weights else 1.0
    quality_flag = "CLIP_TOO_OFTEN" if clip_rate > clip_warn else "OK"

    if isinstance(diagnostics, dict):
        diagnostics.update(
            {
                "prediction_mode": "return_recursive_v2",
                "clip_count": int(clip_count),
                "clip_rate": float(clip_rate),
                "clip_amount_mean": float(clip_amount_mean),
                "r_max": float(r_max),
                "quality_flag": quality_flag,
                "blend_weight_mean": float(blend_weight_mean),
                "return_bias_mean": float(b),
                "conformal_abs_q": float(q_abs),
                "conformal_interval_growth_power": float(interval_power),
                "conformal_interval_growth_scale": float(interval_scale),
                "conformal_interval_growth_cap": float(interval_cap),
                "conformal_interval_source": (
                    "horizon_scale_points"
                    if any(str(src) == "horizon_scale_points" for src in interval_sources)
                    else "power_rule"
                ),
                "conformal_horizon_scale_points_count": int(len(horizon_scale_points)),
                "conformal_interval_q_step_mean": float(np.mean(interval_q_steps)) if interval_q_steps else 0.0,
                "conformal_interval_growth_mean": float(np.mean(interval_growth_steps)) if interval_growth_steps else 1.0,
                "conformal_local_vol_reference": float(local_vol_reference),
                "conformal_local_vol_ratio_mean": float(np.mean(local_vol_ratios)) if local_vol_ratios else 1.0,
                "conformal_local_vol_ratio_bounds": [float(local_vol_ratio_low), float(local_vol_ratio_high)],
                "terminal_anchor_enable": bool(use_terminal_anchor),
                "terminal_anchor_weight": float(terminal_anchor_base_weight),
                "required_exogenous_features": required_exog_features,
                "effective_lags": effective_lags,
                "effective_windows": effective_windows,
                "exogenous_source_flags": (exog_state or {}).get("source_flags", {}),
                "endpoint_direction_guard": bool(direction_guard_enabled),
                "direction_guard_count": int(direction_guard_count),
                "direction_guard_rate": float(direction_guard_count) / float(total),
                "direction_guard_adjust_mean": (
                    float(np.mean(direction_guard_adjustments)) if direction_guard_adjustments else 0.0
                ),
                "return_smooth_alpha": float(smooth_alpha),
                "return_jerk_clip_sigma": float(jerk_clip_sigma),
                "smooth_adjustment_mean": float(np.mean(smooth_adjustments)) if smooth_adjustments else 0.0,
                "jerk_adjustment_mean": float(np.mean(jerk_adjustments)) if jerk_adjustments else 0.0,
                "horizon_steps": int(len(out)),
            }
        )

    return out


def _coerce_direct_reference_points(
    points: Optional[List[Dict[str, Any]]],
    *,
    horizon_days: int,
    floor: float = PRICE_FLOOR,
) -> List[Dict[str, float]]:
    if not points:
        return []

    point_map: Dict[int, float] = {}
    for item in points:
        h = None
        value = None
        if isinstance(item, dict):
            h = _safe_float(item.get("horizon", item.get("day", item.get("step"))))
            value = _safe_float(item.get("value", item.get("price", item.get("pred"))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            h = _safe_float(item[0])
            value = _safe_float(item[1])
        if h is None or value is None or not np.isfinite(value):
            continue
        step = int(h)
        if step <= 0 or step > int(horizon_days):
            continue
        point_map[step] = float(max(floor, value))

    return [{"horizon": int(step), "value": float(point_map[step])} for step in sorted(point_map)]


def _build_direct_reference_path(
    *,
    points: List[Dict[str, float]],
    latest_log: float,
    horizon_days: int,
    anchor_end_value: Optional[float] = None,
    floor: float = PRICE_FLOOR,
) -> tuple[Dict[int, float], List[int], List[Dict[str, float]]]:
    point_map: Dict[int, float] = {}
    for item in points:
        h = int(item.get("horizon", 0))
        value = _safe_float(item.get("value"))
        if h <= 0 or h > int(horizon_days) or value is None or value <= 0:
            continue
        point_map[h] = float(max(floor, value))

    anchor_end = _safe_float(anchor_end_value)
    if anchor_end is not None and anchor_end > 0:
        point_map[int(horizon_days)] = float(max(floor, anchor_end))

    future_anchor_days = sorted(point_map)
    if len(future_anchor_days) < 2:
        normalized = [{"horizon": int(step), "value": float(point_map[step])} for step in future_anchor_days]
        return {}, future_anchor_days, normalized

    path: Dict[int, float] = {}
    anchors: List[tuple[int, float]] = [(0, float(latest_log))]
    anchors.extend((int(step), float(np.log(max(point_map[step], floor)))) for step in future_anchor_days)

    for idx in range(1, len(anchors)):
        left_step, left_y = anchors[idx - 1]
        right_step, right_y = anchors[idx]
        if right_step <= left_step:
            continue
        span = float(right_step - left_step)
        for step in range(max(1, left_step + 1), right_step + 1):
            alpha = float(step - left_step) / span
            path[int(step)] = float((1.0 - alpha) * left_y + alpha * right_y)

    normalized = [{"horizon": int(step), "value": float(point_map[step])} for step in future_anchor_days]
    return path, future_anchor_days, normalized


def _recursive_return_v3(
    *,
    model,
    history_df: pd.DataFrame,
    horizon_days: int,
    lags: List[int],
    windows: List[int],
    feature_cols: Optional[List[str]],
    max_daily_move_pct: Optional[float],
    clip_r_max: Optional[float],
    return_clip_quantile: float,
    return_clip_safety_factor: float,
    return_bias_mean: Optional[float],
    enable_bias_correction: bool,
    conformal_abs_q: Optional[float],
    enable_conformal_interval: bool,
    enable_seasonal_anchor: bool,
    seasonal_y_by_doy: Optional[Dict[str, Any]],
    seasonal_tau_days: float,
    seasonal_min_ml_weight: float,
    seasonal_strength: float,
    seasonal_drift_max_alpha: float,
    seasonal_drift_growth_power: float,
    clip_rate_warn_threshold: float,
    anchor_end_value: Optional[float],
    endpoint_direction_guard: bool,
    terminal_anchor_enable: bool,
    terminal_anchor_weight: float,
    terminal_anchor_tail_days: int,
    terminal_anchor_tail_power: float,
    trend_guard_early_drop_floor: float,
    trend_guard_early_window_days: int,
    trend_guard_end_ratio_low: float,
    trend_guard_end_ratio_high: float,
    trend_guard_hist_window_days: int,
    trend_guard_future_window_days: int,
    trend_guard_t0_jump_max: float,
    trend_guard_t0_jump_steps: int,
    trend_guard_end_tail_days: int,
    trend_guard_end_max_weight: float,
    trend_guard_end_pressure_scale: float,
    trend_guard_end_backstop: bool,
    interval_growth_power: float,
    interval_growth_scale: float,
    interval_growth_cap: float,
    conformal_horizon_scale_points: Optional[List[Dict[str, Any]]],
    conformal_use_horizon_scale: bool,
    conformal_local_vol_window: int,
    conformal_local_vol_ratio_low: float,
    conformal_local_vol_ratio_high: float,
    conformal_local_vol_reference: Optional[float],
    return_smooth_alpha: float,
    return_jerk_clip_sigma: float,
    return_smooth_warmup_steps: int,
    direct_reference_points: Optional[List[Dict[str, Any]]],
    direct_reference_enable: bool,
    direct_reference_weight: float,
    direct_reference_max_weight: float,
    direct_reference_progress_power: float,
    direct_reference_anchor_radius_days: int,
    direct_reference_local_boost: float,
    direct_reference_min_future_points: int,
    include_raw_time_features: bool,
    time_raw_mode: str | None,
    diagnostics: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    safe_horizon = max(1, int(horizon_days))
    s = _prepare_daily_history(history_df)
    if s.empty:
        return []

    history_dates = [pd.Timestamp(x) for x in s.index.to_list()]
    history_price = np.asarray(s.to_numpy(dtype=float), dtype=float)
    history_y = _safe_log_price(history_price, floor=PRICE_FLOOR)
    history_log = [float(x) for x in history_y.tolist()]

    if not history_dates or not history_log:
        return []
    required_exog_features = [f for f in REQUIRED_EXOG_FEATURES if feature_cols and f in feature_cols]
    exog_state = _build_exogenous_state(history_df) if required_exog_features else None
    if required_exog_features and exog_state is None:
        if isinstance(diagnostics, dict):
            diagnostics["fallback_reason"] = "required_exogenous_features_unavailable"
        return []
    effective_lags, effective_windows = _augment_lags_windows_from_feature_cols(lags, windows, feature_cols)

    r_max = _safe_float(clip_r_max)
    if r_max is None:
        legacy_clip = _safe_float(max_daily_move_pct)
        if legacy_clip is not None:
            r_max = abs(float(legacy_clip))
    if r_max is None:
        r_max = estimate_return_clip_limit(
            history_df,
            quantile=return_clip_quantile,
            safety_factor=return_clip_safety_factor,
            floor=PRICE_FLOOR,
        )
    r_max = float(np.clip(abs(float(r_max)), 0.001, 0.35))

    b = _safe_float(return_bias_mean)
    if b is None:
        b = 0.0
    if not bool(enable_bias_correction):
        b = 0.0

    q_abs = _safe_float(conformal_abs_q)
    if q_abs is None:
        q_abs = 0.0
    if not bool(enable_conformal_interval):
        q_abs = 0.0
    q_abs = max(0.0, float(q_abs))

    seasonal_map = _coerce_seasonal_map(seasonal_y_by_doy)
    if not seasonal_map:
        hist_series = pd.Series(np.asarray(history_log, dtype=float), index=pd.to_datetime(history_dates))
        seasonal_map = {
            int(day): float(vals.median())
            for day, vals in hist_series.groupby(hist_series.index.dayofyear)
            if len(vals) >= 2
        }
    seasonal_return_map = _seasonal_return_map_from_levels(seasonal_map)

    tau = float(np.clip(_safe_float(seasonal_tau_days) or 110.0, 5.0, 500.0))
    min_ml_weight = float(np.clip(_safe_float(seasonal_min_ml_weight) or 0.3, 0.0, 1.0))
    clip_warn = float(np.clip(_safe_float(clip_rate_warn_threshold) or 0.10, 0.0, 1.0))
    direction_guard_enabled = bool(endpoint_direction_guard)
    use_terminal_anchor = bool(terminal_anchor_enable)
    terminal_anchor_base_weight = float(np.clip(_safe_float(terminal_anchor_weight) or 0.35, 0.0, 1.0))
    tail_days = int(np.clip(_safe_float(terminal_anchor_tail_days) or 45, 5, safe_horizon))
    tail_power = float(np.clip(_safe_float(terminal_anchor_tail_power) or 1.6, 0.5, 4.0))
    interval_power = float(np.clip(_safe_float(interval_growth_power) or 0.5, 0.0, 1.5))
    interval_scale = float(np.clip(_safe_float(interval_growth_scale) or 1.0, 0.0, 20.0))
    interval_cap = float(np.clip(_safe_float(interval_growth_cap) or 200.0, 1.0, 500.0))
    horizon_scale_points = _coerce_horizon_scale_points(conformal_horizon_scale_points)
    use_horizon_scale = bool(conformal_use_horizon_scale)
    local_vol_window = int(np.clip(_safe_float(conformal_local_vol_window) or 45, 5, 730))
    local_vol_ratio_low = float(np.clip(_safe_float(conformal_local_vol_ratio_low) or 0.7, 0.2, 1.0))
    local_vol_ratio_high = float(np.clip(_safe_float(conformal_local_vol_ratio_high) or 1.4, 1.0, 5.0))
    if local_vol_ratio_low > local_vol_ratio_high:
        local_vol_ratio_low, local_vol_ratio_high = local_vol_ratio_high, local_vol_ratio_low
    local_vol_reference = _safe_float(conformal_local_vol_reference)
    if local_vol_reference is None or local_vol_reference <= 0:
        local_vol_reference = _robust_return_std_from_log_path(history_log, window=max(45, local_vol_window))
    if local_vol_reference is None or local_vol_reference <= 0:
        local_vol_reference = 0.01
    smooth_alpha = float(np.clip(_safe_float(return_smooth_alpha) or 0.15, 0.0, 0.95))
    jerk_clip_sigma = float(np.clip(_safe_float(return_jerk_clip_sigma) or 2.5, 0.0, 12.0))
    smooth_warmup_steps = max(1, int(_safe_float(return_smooth_warmup_steps) or 5))
    drift_alpha_cap = _safe_float(seasonal_drift_max_alpha)
    if drift_alpha_cap is None:
        drift_alpha_cap = _safe_float(seasonal_strength)
    drift_alpha_cap = float(np.clip(drift_alpha_cap or 0.22, 0.0, 0.6))
    drift_growth = float(np.clip(_safe_float(seasonal_drift_growth_power) or 1.6, 0.5, 5.0))
    early_floor = float(np.clip(_safe_float(trend_guard_early_drop_floor) or -0.08, -0.5, 0.0))
    early_window = int(np.clip(_safe_float(trend_guard_early_window_days) or 28, 5, safe_horizon))
    end_ratio_low = float(np.clip(_safe_float(trend_guard_end_ratio_low) or 0.85, 0.4, 1.5))
    end_ratio_high = float(np.clip(_safe_float(trend_guard_end_ratio_high) or 1.15, 0.6, 2.5))
    if end_ratio_low > end_ratio_high:
        end_ratio_low, end_ratio_high = end_ratio_high, end_ratio_low
    hist_window = int(np.clip(_safe_float(trend_guard_hist_window_days) or 90, 10, 730))
    future_window = int(np.clip(_safe_float(trend_guard_future_window_days) or 60, 5, safe_horizon))
    t0_jump_max = float(np.clip(_safe_float(trend_guard_t0_jump_max) or 0.03, 0.0, 0.25))
    t0_jump_steps = int(np.clip(_safe_float(trend_guard_t0_jump_steps) or 1, 1, min(7, safe_horizon)))
    end_tail_days = int(np.clip(_safe_float(trend_guard_end_tail_days) or 30, 5, safe_horizon))
    end_max_weight = float(np.clip(_safe_float(trend_guard_end_max_weight) or 0.90, 0.2, 0.98))
    end_pressure_scale = float(np.clip(_safe_float(trend_guard_end_pressure_scale) or 6.0, 0.1, 20.0))
    end_backstop = bool(trend_guard_end_backstop)
    direct_path_enabled = bool(direct_reference_enable)
    direct_path_weight = float(np.clip(_safe_float(direct_reference_weight) or 0.10, 0.0, 0.9))
    direct_path_max_weight = float(np.clip(_safe_float(direct_reference_max_weight) or 0.35, 0.0, 0.95))
    direct_path_progress_power = float(np.clip(_safe_float(direct_reference_progress_power) or 1.25, 0.1, 5.0))
    direct_path_anchor_radius = int(np.clip(_safe_float(direct_reference_anchor_radius_days) or 14, 1, max(1, safe_horizon)))
    direct_path_local_boost = float(np.clip(_safe_float(direct_reference_local_boost) or 0.12, 0.0, 0.9))
    direct_path_min_future_points = max(2, int(_safe_float(direct_reference_min_future_points) or 2))

    anchor_y = None
    if _safe_float(anchor_end_value) is not None and _safe_float(anchor_end_value) > 0:
        anchor_y = float(np.log(max(_safe_float(anchor_end_value), PRICE_FLOOR)))
    latest_origin_price = float(np.exp(history_log[-1]))
    ma30_ref = float(np.mean(np.exp(np.asarray(history_log[-min(30, len(history_log)) :], dtype=float))))
    hist_trend_sign = _slope_sign_from_log_path(history_log, window=hist_window)
    direct_path_future_points = _coerce_direct_reference_points(
        direct_reference_points,
        horizon_days=safe_horizon,
        floor=PRICE_FLOOR,
    )
    direct_path, direct_path_anchor_days, direct_path_future_points = _build_direct_reference_path(
        points=direct_path_future_points,
        latest_log=float(history_log[-1]),
        horizon_days=safe_horizon,
        anchor_end_value=anchor_end_value,
        floor=PRICE_FLOOR,
    )
    if not direct_path_enabled or len(direct_path_anchor_days) < direct_path_min_future_points:
        direct_path = {}
        direct_path_anchor_days = []

    out: List[Dict[str, Any]] = []
    clip_count = 0
    clip_amounts: List[float] = []
    blend_weights: List[float] = []
    interval_q_steps: List[float] = []
    interval_growth_steps: List[float] = []
    local_vol_ratios: List[float] = []
    interval_sources: List[str] = []
    smooth_adjustments: List[float] = []
    jerk_adjustments: List[float] = []
    seasonal_adjustments: List[float] = []
    terminal_adjustments: List[float] = []
    direct_path_adjustments: List[float] = []
    direct_path_weights: List[float] = []
    trend_sign_adjustments: List[float] = []
    early_floor_adjustments: List[float] = []
    end_ratio_adjustments: List[float] = []
    t0_jump_adjustments: List[float] = []
    predicted_returns: List[float] = []
    trend_sign_guard_count = 0
    terminal_anchor_count = 0
    early_floor_guard_count = 0
    end_ratio_guard_count = 0
    t0_jump_guard_count = 0
    end_ratio_backstop_count = 0
    prev_q_step = 0.0
    tail_start = max(1, safe_horizon - tail_days + 1)

    for step_idx in range(1, safe_horizon + 1):
        prev_date = history_dates[-1]
        prev_y = float(history_log[-1])
        next_date = prev_date + pd.Timedelta(days=1)

        feat_row = _build_feature_row(
            history_dates=history_dates,
            history_values=history_log,
            next_date=next_date,
            lags=effective_lags,
            windows=effective_windows,
            include_raw_time_features=include_raw_time_features,
            time_raw_mode=time_raw_mode,
            exogenous_row=_exogenous_row_from_state(exog_state) if required_exog_features else None,
        )
        if not feat_row:
            break

        try:
            X = _prepare_model_input(
                feat_row,
                feature_cols,
                required_features=required_exog_features,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            if isinstance(diagnostics, dict):
                diagnostics.setdefault("feature_schema_errors", []).append(str(exc))
            break

        try:
            pred_r = _safe_float(model.predict(X)[0])
        except Exception:
            pred_r = None
        if pred_r is None:
            break

        r_raw = float(pred_r)
        r_adj = float(r_raw - b)
        r_limited = float(np.clip(r_adj, -r_max, r_max))
        clip_amount = float(abs(r_adj - r_limited))
        clip_applied = clip_amount > 1e-12
        if clip_applied:
            clip_count += 1
            clip_amounts.append(clip_amount)

        r_effective = float(r_limited)
        smooth_adjust = 0.0
        jerk_adjust = 0.0
        if step_idx > 1 and (smooth_alpha > 1e-9 or jerk_clip_sigma > 1e-9):
            center_window = min(max(7, local_vol_window), max(7, len(history_log) - 1))
            arr_recent = np.asarray(history_log[-(center_window + 1) :], dtype=float)
            arr_recent = arr_recent[np.isfinite(arr_recent)]
            ret_center = 0.0
            if arr_recent.size >= 3:
                recent_ret = np.diff(arr_recent)
                recent_ret = recent_ret[np.isfinite(recent_ret)]
                if recent_ret.size:
                    ret_center = float(np.median(recent_ret[-min(14, recent_ret.size) :]))

            if smooth_alpha > 1e-9:
                smooth_ramp = min(1.0, float(step_idx - 1) / float(smooth_warmup_steps))
                smooth_w = float(np.clip(smooth_alpha * smooth_ramp, 0.0, smooth_alpha))
                before = float(r_effective)
                r_effective = float((1.0 - smooth_w) * r_effective + smooth_w * ret_center)
                smooth_adjust = float(abs(r_effective - before))

            if jerk_clip_sigma > 1e-9 and predicted_returns:
                before = float(r_effective)
                prev_r = float(predicted_returns[-1])
                ref_std = _robust_return_std_from_log_path(history_log, window=max(9, local_vol_window))
                if ref_std <= 0:
                    ref_std = abs(prev_r) * 0.35
                jerk_cap = max(0.001, float(ref_std) * jerk_clip_sigma)
                r_effective = float(prev_r + np.clip(r_effective - prev_r, -jerk_cap, jerk_cap))
                jerk_adjust = float(abs(r_effective - before))

        seasonal_alpha = 0.0
        if bool(enable_seasonal_anchor) and seasonal_return_map:
            seasonal_r = _safe_float(seasonal_return_map.get(int(next_date.dayofyear)))
            if seasonal_r is not None and np.isfinite(seasonal_r):
                progress = float(step_idx) / float(max(1, safe_horizon))
                seasonal_alpha = float(np.clip(drift_alpha_cap * (progress**drift_growth), 0.0, drift_alpha_cap))
                seasonal_alpha = min(seasonal_alpha, 1.0 - min_ml_weight)
                if seasonal_alpha > 1e-12:
                    before = float(r_effective)
                    r_effective = float((1.0 - seasonal_alpha) * r_effective + seasonal_alpha * float(seasonal_r))
                    seasonal_adjustments.append(float(abs(r_effective - before)))

        direct_reference_y = _safe_float(direct_path.get(int(step_idx))) if direct_path else None
        direct_reference_weight_effective = 0.0
        direct_reference_adjust = 0.0
        if direct_reference_y is not None:
            progress = float(step_idx) / float(max(1, safe_horizon))
            progress_w = float(direct_path_weight * (progress**direct_path_progress_power))
            locality = 0.0
            if direct_path_anchor_days:
                nearest_anchor = min(abs(int(step_idx) - int(h)) for h in direct_path_anchor_days)
                locality = max(0.0, 1.0 - float(nearest_anchor) / float(max(1, direct_path_anchor_radius)))
            direct_reference_weight_effective = float(
                np.clip(progress_w + direct_path_local_boost * locality, 0.0, direct_path_max_weight)
            )
            if direct_reference_weight_effective > 1e-12:
                target_step = float(np.clip(float(direct_reference_y) - prev_y, -r_max, r_max))
                before = float(r_effective)
                r_effective = float(
                    (1.0 - direct_reference_weight_effective) * r_effective
                    + direct_reference_weight_effective * target_step
                )
                direct_reference_adjust = float(abs(r_effective - before))
                if direct_reference_adjust > 1e-12:
                    direct_path_adjustments.append(float(direct_reference_adjust))
            direct_path_weights.append(float(direct_reference_weight_effective))

        if direction_guard_enabled and hist_trend_sign != 0 and step_idx <= future_window:
            recent_window = max(3, min(len(predicted_returns), future_window - 1))
            recent_pred = predicted_returns[-recent_window:] if recent_window > 0 else []
            probe = np.asarray([*recent_pred, float(r_effective)], dtype=float)
            probe = probe[np.isfinite(probe)]
            if probe.size >= 3:
                probe_mean = float(np.mean(probe))
                probe_sign = 1 if probe_mean > 1e-9 else (-1 if probe_mean < -1e-9 else 0)
                if probe_sign != 0 and probe_sign != hist_trend_sign:
                    ref_std = _robust_return_std_from_log_path(history_log, window=hist_window)
                    cap = max(0.05 * r_max, 0.35 * ref_std)
                    guided = float(np.sign(hist_trend_sign) * min(abs(r_effective), cap))
                    if abs(guided - r_effective) > 1e-12:
                        trend_sign_guard_count += 1
                        trend_sign_adjustments.append(float(abs(guided - r_effective)))
                        r_effective = guided

        if use_terminal_anchor and anchor_y is not None and step_idx >= tail_start:
            remain = max(1, int(safe_horizon - step_idx + 1))
            target_step = float((anchor_y - prev_y) / float(remain))
            target_step = float(np.clip(target_step, -r_max, r_max))
            tail_progress = float(step_idx - tail_start + 1) / float(max(1, tail_days))
            anchor_w = float(np.clip(terminal_anchor_base_weight * (tail_progress**tail_power), 0.0, 0.9))
            before = float(r_effective)
            r_effective = float((1.0 - anchor_w) * r_effective + anchor_w * target_step)
            if abs(r_effective - before) > 1e-12:
                terminal_anchor_count += 1
                terminal_adjustments.append(float(abs(r_effective - before)))

        r_effective = float(np.clip(r_effective, -r_max, r_max))
        y_p50 = float(prev_y + r_effective)
        p50 = float(np.exp(y_p50))

        if step_idx <= t0_jump_steps:
            low_t0 = float(max(PRICE_FLOOR, latest_origin_price * (1.0 - t0_jump_max)))
            high_t0 = float(max(low_t0 * 1.01, latest_origin_price * (1.0 + t0_jump_max)))
            p50_clamped = float(np.clip(p50, low_t0, high_t0))
            if abs(p50_clamped - p50) > 1e-12:
                y_t0 = float(np.log(p50_clamped))
                t0_jump_guard_count += 1
                t0_jump_adjustments.append(float(abs(y_t0 - y_p50)))
                y_p50 = y_t0
                r_effective = float(np.clip(y_p50 - prev_y, -r_max, r_max))
                y_p50 = float(prev_y + r_effective)
                p50 = float(np.exp(y_p50))

        if step_idx <= early_window:
            floor_price = float(max(PRICE_FLOOR, latest_origin_price * (1.0 + early_floor)))
            if p50 < floor_price:
                y_floor = float(np.log(floor_price))
                early_floor_guard_count += 1
                early_floor_adjustments.append(float(abs(y_floor - y_p50)))
                y_p50 = y_floor
                r_effective = float(np.clip(y_p50 - prev_y, -r_max, r_max))
                y_p50 = float(prev_y + r_effective)
                p50 = float(np.exp(y_p50))

        remain = max(1, int(safe_horizon - step_idx + 1))
        low_end = float(max(PRICE_FLOOR, ma30_ref * end_ratio_low))
        high_end = float(max(low_end * 1.01, ma30_ref * end_ratio_high))
        projected_end = float(np.exp(prev_y + float(r_effective) * float(remain)))
        if projected_end < low_end or projected_end > high_end:
            target_end = float(np.clip(projected_end, low_end, high_end))
            need_step = float((np.log(target_end) - prev_y) / float(remain))
            need_step = float(np.clip(need_step, -r_max, r_max))
            progress = float(step_idx) / float(max(1, safe_horizon))
            pressure = abs(float(np.log(max(projected_end, PRICE_FLOOR) / max(target_end, PRICE_FLOOR))))
            guard_w = float(np.clip(0.18 + 0.42 * (progress**1.3) + min(0.35, pressure * end_pressure_scale), 0.0, end_max_weight))
            if step_idx <= future_window:
                guard_w *= 0.65
            if remain <= end_tail_days:
                tail_progress = 1.0 - float(remain - 1) / float(max(1, end_tail_days))
                guard_w = max(guard_w, min(end_max_weight, 0.55 + 0.35 * (tail_progress**1.2)))
            before = float(r_effective)
            r_effective = float((1.0 - guard_w) * r_effective + guard_w * need_step)
            if abs(r_effective - before) > 1e-12:
                end_ratio_guard_count += 1
                end_ratio_adjustments.append(float(abs(r_effective - before)))
            r_effective = float(np.clip(r_effective, -r_max, r_max))
            y_p50 = float(prev_y + r_effective)
            p50 = float(np.exp(y_p50))
        if end_backstop and remain <= 1:
            p50_clamped = float(np.clip(p50, low_end, high_end))
            if abs(p50_clamped - p50) > 1e-12:
                y_end = float(np.log(p50_clamped))
                end_ratio_backstop_count += 1
                end_ratio_adjustments.append(float(abs(y_end - y_p50)))
                y_p50 = y_end
                r_effective = float(np.clip(y_p50 - prev_y, -r_max, r_max))
                y_p50 = float(prev_y + r_effective)
                p50 = float(np.exp(y_p50))

        interval_growth, interval_source = _resolve_interval_growth_scale(
            step_idx=step_idx,
            horizon_scale_points=horizon_scale_points,
            use_horizon_scale=use_horizon_scale,
            interval_power=interval_power,
            interval_scale=interval_scale,
            interval_cap=interval_cap,
        )
        local_vol = _robust_return_std_from_log_path(history_log, window=local_vol_window)
        if local_vol <= 0:
            local_vol = float(local_vol_reference)
        vol_ratio = float(np.clip(local_vol / max(local_vol_reference, 1e-9), local_vol_ratio_low, local_vol_ratio_high))
        q_step_raw = float(q_abs * interval_growth * vol_ratio)
        q_step = float(max(prev_q_step, q_step_raw))
        prev_q_step = q_step
        y_p10 = float(y_p50 - q_step)
        y_p90 = float(y_p50 + q_step)
        p10 = float(np.exp(y_p10))
        p90 = float(np.exp(y_p90))

        ml_weight_effective = float(np.clip(1.0 - seasonal_alpha, min_ml_weight, 1.0))
        out.append(
            {
                "date": next_date.strftime("%Y-%m-%d"),
                "value": p50,
                "p10": p10,
                "p50": p50,
                "p90": p90,
                "blend_weight": float(ml_weight_effective),
                "interval_growth": float(interval_growth),
                "interval_source": interval_source,
                "interval_q_step": float(q_step),
                "local_vol_ratio": float(vol_ratio),
                "clip_applied": bool(clip_applied),
                "clip_amount": float(clip_amount),
                "smooth_adjustment": float(smooth_adjust),
                "jerk_adjustment": float(jerk_adjust),
                "direct_reference_value": (
                    float(np.exp(float(direct_reference_y))) if direct_reference_y is not None else None
                ),
                "direct_reference_weight": float(direct_reference_weight_effective),
                "direct_reference_adjustment": float(direct_reference_adjust),
            }
        )

        blend_weights.append(float(ml_weight_effective))
        interval_q_steps.append(float(q_step))
        interval_growth_steps.append(float(interval_growth))
        local_vol_ratios.append(float(vol_ratio))
        interval_sources.append(str(interval_source))
        smooth_adjustments.append(float(smooth_adjust))
        jerk_adjustments.append(float(jerk_adjust))
        predicted_returns.append(float(r_effective))
        history_dates.append(next_date)
        history_log.append(y_p50)
        _update_exogenous_state(exog_state, prev_price=float(np.exp(prev_y)), next_price=p50)

    total = max(1, len(out))
    clip_rate = float(clip_count) / float(total)
    clip_amount_mean = float(np.mean(clip_amounts)) if clip_amounts else 0.0
    blend_weight_mean = float(np.mean(blend_weights)) if blend_weights else 1.0
    quality_flag = "CLIP_TOO_OFTEN" if clip_rate > clip_warn else "OK"

    if isinstance(diagnostics, dict):
        diagnostics.update(
            {
                "prediction_mode": "return_recursive_v3",
                "clip_count": int(clip_count),
                "clip_rate": float(clip_rate),
                "clip_amount_mean": float(clip_amount_mean),
                "r_max": float(r_max),
                "quality_flag": quality_flag,
                "blend_weight_mean": float(blend_weight_mean),
                "return_bias_mean": float(b),
                "conformal_abs_q": float(q_abs),
                "conformal_interval_growth_power": float(interval_power),
                "conformal_interval_growth_scale": float(interval_scale),
                "conformal_interval_growth_cap": float(interval_cap),
                "conformal_interval_source": (
                    "horizon_scale_points"
                    if any(str(src) == "horizon_scale_points" for src in interval_sources)
                    else "power_rule"
                ),
                "conformal_horizon_scale_points_count": int(len(horizon_scale_points)),
                "conformal_interval_q_step_mean": float(np.mean(interval_q_steps)) if interval_q_steps else 0.0,
                "conformal_interval_growth_mean": float(np.mean(interval_growth_steps)) if interval_growth_steps else 1.0,
                "conformal_local_vol_reference": float(local_vol_reference),
                "conformal_local_vol_ratio_mean": float(np.mean(local_vol_ratios)) if local_vol_ratios else 1.0,
                "conformal_local_vol_ratio_bounds": [float(local_vol_ratio_low), float(local_vol_ratio_high)],
                "required_exogenous_features": required_exog_features,
                "effective_lags": effective_lags,
                "effective_windows": effective_windows,
                "exogenous_source_flags": (exog_state or {}).get("source_flags", {}),
                "endpoint_direction_guard": bool(direction_guard_enabled),
                "trend_guard_hist_sign": int(hist_trend_sign),
                "trend_sign_guard_count": int(trend_sign_guard_count),
                "trend_sign_guard_rate": float(trend_sign_guard_count) / float(total),
                "trend_sign_guard_adjust_mean": (
                    float(np.mean(trend_sign_adjustments)) if trend_sign_adjustments else 0.0
                ),
                "terminal_anchor_enable": bool(use_terminal_anchor),
                "terminal_anchor_weight": float(terminal_anchor_base_weight),
                "terminal_anchor_tail_days": int(tail_days),
                "terminal_anchor_count": int(terminal_anchor_count),
                "terminal_anchor_adjust_mean": float(np.mean(terminal_adjustments)) if terminal_adjustments else 0.0,
                "direct_reference_enable": bool(direct_path_enabled),
                "direct_reference_active": bool(direct_path),
                "direct_reference_weight": float(direct_path_weight),
                "direct_reference_max_weight": float(direct_path_max_weight),
                "direct_reference_progress_power": float(direct_path_progress_power),
                "direct_reference_anchor_radius_days": int(direct_path_anchor_radius),
                "direct_reference_local_boost": float(direct_path_local_boost),
                "direct_reference_min_future_points": int(direct_path_min_future_points),
                "direct_reference_points": list(direct_path_future_points),
                "direct_reference_points_count": int(len(direct_path_future_points)),
                "direct_reference_anchor_days": [int(x) for x in direct_path_anchor_days],
                "direct_reference_adjust_mean": (
                    float(np.mean(direct_path_adjustments)) if direct_path_adjustments else 0.0
                ),
                "direct_reference_weight_mean": float(np.mean(direct_path_weights)) if direct_path_weights else 0.0,
                "seasonal_drift_enable": bool(enable_seasonal_anchor),
                "seasonal_drift_alpha_cap": float(drift_alpha_cap),
                "seasonal_drift_growth_power": float(drift_growth),
                "seasonal_drift_adjust_mean": float(np.mean(seasonal_adjustments)) if seasonal_adjustments else 0.0,
                "trend_guard_early_drop_floor": float(early_floor),
                "trend_guard_early_window_days": int(early_window),
                "trend_guard_early_count": int(early_floor_guard_count),
                "trend_guard_early_adjust_mean": (
                    float(np.mean(early_floor_adjustments)) if early_floor_adjustments else 0.0
                ),
                "trend_guard_t0_jump_max": float(t0_jump_max),
                "trend_guard_t0_jump_steps": int(t0_jump_steps),
                "trend_guard_t0_jump_count": int(t0_jump_guard_count),
                "trend_guard_t0_jump_adjust_mean": (
                    float(np.mean(t0_jump_adjustments)) if t0_jump_adjustments else 0.0
                ),
                "trend_guard_end_ratio_bounds": [float(end_ratio_low), float(end_ratio_high)],
                "trend_guard_end_ratio_count": int(end_ratio_guard_count),
                "trend_guard_end_ratio_tail_days": int(end_tail_days),
                "trend_guard_end_ratio_max_weight": float(end_max_weight),
                "trend_guard_end_ratio_pressure_scale": float(end_pressure_scale),
                "trend_guard_end_ratio_backstop_count": int(end_ratio_backstop_count),
                "trend_guard_end_ratio_adjust_mean": (
                    float(np.mean(end_ratio_adjustments)) if end_ratio_adjustments else 0.0
                ),
                "return_smooth_alpha": float(smooth_alpha),
                "return_jerk_clip_sigma": float(jerk_clip_sigma),
                "smooth_adjustment_mean": float(np.mean(smooth_adjustments)) if smooth_adjustments else 0.0,
                "jerk_adjustment_mean": float(np.mean(jerk_adjustments)) if jerk_adjustments else 0.0,
                "horizon_steps": int(len(out)),
            }
        )

    return out


def recursive_multi_step_forecast(
    *,
    model,
    history_df: pd.DataFrame,
    horizon_days: int,
    lags: List[int],
    windows: List[int],
    feature_cols: Optional[List[str]] = None,
    max_daily_move_pct: Optional[float] = None,
    anchor_end_value: Optional[float] = None,
    anchor_max_blend: float = 0.65,
    mean_reversion_strength: float = 0.18,
    seasonal_strength: float = 0.22,
    corridor_low_quantile: float = 0.05,
    corridor_high_quantile: float = 0.95,
    corridor_low_multiplier: float = 0.75,
    corridor_high_multiplier: float = 1.35,
    terminal_low_ratio_vs_latest: float = 0.6,
    terminal_high_ratio_vs_latest: float = 1.8,
    prediction_mode: str = "price_recursive_v1",
    clip_r_max: Optional[float] = None,
    return_clip_quantile: float = 0.98,
    return_clip_safety_factor: float = 1.2,
    return_bias_mean: Optional[float] = None,
    enable_bias_correction: bool = True,
    conformal_abs_q: Optional[float] = None,
    enable_conformal_interval: bool = True,
    enable_seasonal_anchor: bool = True,
    seasonal_y_by_doy: Optional[Dict[str, Any]] = None,
    seasonal_tau_days: float = 55.0,
    seasonal_min_ml_weight: float = 0.15,
    seasonal_drift_max_alpha: float = 0.22,
    seasonal_drift_growth_power: float = 1.6,
    clip_rate_warn_threshold: float = 0.10,
    endpoint_direction_guard: bool = True,
    endpoint_opposite_slack: float = 1.5,
    endpoint_same_dir_cap: float = 4.0,
    terminal_anchor_enable: bool = True,
    terminal_anchor_weight: float = 0.35,
    terminal_anchor_tail_days: int = 45,
    terminal_anchor_tail_power: float = 1.6,
    trend_guard_early_drop_floor: float = -0.08,
    trend_guard_early_window_days: int = 28,
    trend_guard_end_ratio_low: float = 0.85,
    trend_guard_end_ratio_high: float = 1.15,
    trend_guard_hist_window_days: int = 90,
    trend_guard_future_window_days: int = 60,
    trend_guard_t0_jump_max: float = 0.03,
    trend_guard_t0_jump_steps: int = 1,
    trend_guard_end_tail_days: int = 30,
    trend_guard_end_max_weight: float = 0.90,
    trend_guard_end_pressure_scale: float = 6.0,
    trend_guard_end_backstop: bool = True,
    interval_growth_power: float = 0.5,
    interval_growth_scale: float = 1.0,
    interval_growth_cap: float = 200.0,
    conformal_horizon_scale_points: Optional[List[Dict[str, Any]]] = None,
    conformal_use_horizon_scale: bool = True,
    conformal_local_vol_window: int = 45,
    conformal_local_vol_ratio_low: float = 0.70,
    conformal_local_vol_ratio_high: float = 1.40,
    conformal_local_vol_reference: Optional[float] = None,
    return_smooth_alpha: float = 0.15,
    return_jerk_clip_sigma: float = 2.5,
    return_smooth_warmup_steps: int = 5,
    direct_reference_points: Optional[List[Dict[str, Any]]] = None,
    direct_reference_enable: bool = True,
    direct_reference_weight: float = 0.10,
    direct_reference_max_weight: float = 0.35,
    direct_reference_progress_power: float = 1.25,
    direct_reference_anchor_radius_days: int = 14,
    direct_reference_local_boost: float = 0.12,
    direct_reference_min_future_points: int = 2,
    include_raw_time_features: bool = False,
    time_raw_mode: str | None = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if model is None:
        return []

    mode = str(prediction_mode or "price_recursive_v1").strip().lower()
    if isinstance(diagnostics, dict) and mode in {"return_recursive_v2", "return_recursive_v2_legacy", "price_recursive_v1"}:
        diagnostics["legacy_mode_warning"] = (
            "legacy_recursive_mode_in_use; prefer return_recursive_v3 for production"
        )
    if mode in {"return_recursive_v2", "return_recursive_v2_legacy"}:
        return _recursive_return_v2(
            model=model,
            history_df=history_df,
            horizon_days=horizon_days,
            lags=lags,
            windows=windows,
            feature_cols=feature_cols,
            max_daily_move_pct=max_daily_move_pct,
            clip_r_max=clip_r_max,
            return_clip_quantile=return_clip_quantile,
            return_clip_safety_factor=return_clip_safety_factor,
            return_bias_mean=return_bias_mean,
            enable_bias_correction=enable_bias_correction,
            conformal_abs_q=conformal_abs_q,
            enable_conformal_interval=enable_conformal_interval,
            enable_seasonal_anchor=enable_seasonal_anchor,
            seasonal_y_by_doy=seasonal_y_by_doy,
            seasonal_tau_days=seasonal_tau_days,
            seasonal_min_ml_weight=seasonal_min_ml_weight,
            clip_rate_warn_threshold=clip_rate_warn_threshold,
            anchor_end_value=anchor_end_value,
            endpoint_direction_guard=endpoint_direction_guard,
            endpoint_opposite_slack=endpoint_opposite_slack,
            endpoint_same_dir_cap=endpoint_same_dir_cap,
            terminal_anchor_enable=terminal_anchor_enable,
            terminal_anchor_weight=terminal_anchor_weight,
            interval_growth_power=interval_growth_power,
            interval_growth_scale=interval_growth_scale,
            interval_growth_cap=interval_growth_cap,
            conformal_horizon_scale_points=conformal_horizon_scale_points,
            conformal_use_horizon_scale=conformal_use_horizon_scale,
            conformal_local_vol_window=conformal_local_vol_window,
            conformal_local_vol_ratio_low=conformal_local_vol_ratio_low,
            conformal_local_vol_ratio_high=conformal_local_vol_ratio_high,
            conformal_local_vol_reference=conformal_local_vol_reference,
            return_smooth_alpha=return_smooth_alpha,
            return_jerk_clip_sigma=return_jerk_clip_sigma,
            return_smooth_warmup_steps=return_smooth_warmup_steps,
            include_raw_time_features=include_raw_time_features,
            time_raw_mode=time_raw_mode,
            diagnostics=diagnostics,
        )
    if mode == "return_recursive_v3":
        return _recursive_return_v3(
            model=model,
            history_df=history_df,
            horizon_days=horizon_days,
            lags=lags,
            windows=windows,
            feature_cols=feature_cols,
            max_daily_move_pct=max_daily_move_pct,
            clip_r_max=clip_r_max,
            return_clip_quantile=return_clip_quantile,
            return_clip_safety_factor=return_clip_safety_factor,
            return_bias_mean=return_bias_mean,
            enable_bias_correction=enable_bias_correction,
            conformal_abs_q=conformal_abs_q,
            enable_conformal_interval=enable_conformal_interval,
            enable_seasonal_anchor=enable_seasonal_anchor,
            seasonal_y_by_doy=seasonal_y_by_doy,
            seasonal_tau_days=seasonal_tau_days,
            seasonal_min_ml_weight=seasonal_min_ml_weight,
            seasonal_strength=seasonal_strength,
            seasonal_drift_max_alpha=seasonal_drift_max_alpha,
            seasonal_drift_growth_power=seasonal_drift_growth_power,
            clip_rate_warn_threshold=clip_rate_warn_threshold,
            anchor_end_value=anchor_end_value,
            endpoint_direction_guard=endpoint_direction_guard,
            terminal_anchor_enable=terminal_anchor_enable,
            terminal_anchor_weight=terminal_anchor_weight,
            terminal_anchor_tail_days=terminal_anchor_tail_days,
            terminal_anchor_tail_power=terminal_anchor_tail_power,
            trend_guard_early_drop_floor=trend_guard_early_drop_floor,
            trend_guard_early_window_days=trend_guard_early_window_days,
            trend_guard_end_ratio_low=trend_guard_end_ratio_low,
            trend_guard_end_ratio_high=trend_guard_end_ratio_high,
            trend_guard_hist_window_days=trend_guard_hist_window_days,
            trend_guard_future_window_days=trend_guard_future_window_days,
            trend_guard_t0_jump_max=trend_guard_t0_jump_max,
            trend_guard_t0_jump_steps=trend_guard_t0_jump_steps,
            trend_guard_end_tail_days=trend_guard_end_tail_days,
            trend_guard_end_max_weight=trend_guard_end_max_weight,
            trend_guard_end_pressure_scale=trend_guard_end_pressure_scale,
            trend_guard_end_backstop=trend_guard_end_backstop,
            interval_growth_power=interval_growth_power,
            interval_growth_scale=interval_growth_scale,
            interval_growth_cap=interval_growth_cap,
            conformal_horizon_scale_points=conformal_horizon_scale_points,
            conformal_use_horizon_scale=conformal_use_horizon_scale,
            conformal_local_vol_window=conformal_local_vol_window,
            conformal_local_vol_ratio_low=conformal_local_vol_ratio_low,
            conformal_local_vol_ratio_high=conformal_local_vol_ratio_high,
            conformal_local_vol_reference=conformal_local_vol_reference,
            return_smooth_alpha=return_smooth_alpha,
            return_jerk_clip_sigma=return_jerk_clip_sigma,
            return_smooth_warmup_steps=return_smooth_warmup_steps,
            direct_reference_points=direct_reference_points,
            direct_reference_enable=direct_reference_enable,
            direct_reference_weight=direct_reference_weight,
            direct_reference_max_weight=direct_reference_max_weight,
            direct_reference_progress_power=direct_reference_progress_power,
            direct_reference_anchor_radius_days=direct_reference_anchor_radius_days,
            direct_reference_local_boost=direct_reference_local_boost,
            direct_reference_min_future_points=direct_reference_min_future_points,
            include_raw_time_features=include_raw_time_features,
            time_raw_mode=time_raw_mode,
            diagnostics=diagnostics,
        )

    effective_lags, effective_windows = _augment_lags_windows_from_feature_cols(lags, windows, feature_cols)
    out = _legacy_recursive_price_v1(
        model=model,
        history_df=history_df,
        horizon_days=horizon_days,
        lags=lags,
        windows=windows,
        feature_cols=feature_cols,
        max_daily_move_pct=max_daily_move_pct,
        anchor_end_value=anchor_end_value,
        anchor_max_blend=anchor_max_blend,
        mean_reversion_strength=mean_reversion_strength,
        seasonal_strength=seasonal_strength,
        corridor_low_quantile=corridor_low_quantile,
        corridor_high_quantile=corridor_high_quantile,
        corridor_low_multiplier=corridor_low_multiplier,
        corridor_high_multiplier=corridor_high_multiplier,
        terminal_low_ratio_vs_latest=terminal_low_ratio_vs_latest,
        terminal_high_ratio_vs_latest=terminal_high_ratio_vs_latest,
    )
    if isinstance(diagnostics, dict):
        clips = [float(x.get("clip_amount", 0.0)) for x in out if isinstance(x, dict)]
        clip_count = int(sum(1 for c in clips if c > 0.0))
        total = max(1, len(out))
        diagnostics.update(
            {
                "prediction_mode": "price_recursive_v1",
                "clip_count": int(clip_count),
                "clip_rate": float(clip_count / float(total)),
                "clip_amount_mean": float(np.mean([c for c in clips if c > 0.0])) if clip_count else 0.0,
                "quality_flag": "OK",
                "effective_lags": effective_lags,
                "effective_windows": effective_windows,
                "horizon_steps": int(len(out)),
            }
        )
    return out
