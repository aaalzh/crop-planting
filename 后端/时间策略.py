from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd


def _safe_int(value: Any) -> Optional[int]:
    try:
        out = int(float(value))
    except Exception:
        return None
    return out


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _safe_timestamp(value: Any) -> Optional[pd.Timestamp]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        ts = pd.Timestamp(text)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.normalize()


def _normalize_dates(values: Optional[Iterable[Any]]) -> pd.Series:
    if values is None:
        return pd.Series(dtype="datetime64[ns]")
    try:
        raw = pd.Series(list(values))
    except Exception:
        return pd.Series(dtype="datetime64[ns]")
    if raw.empty:
        return pd.Series(dtype="datetime64[ns]")
    out = pd.to_datetime(raw, errors="coerce")
    out = out.dropna()
    if out.empty:
        return pd.Series(dtype="datetime64[ns]")
    return out.dt.normalize()


def dynamic_in_sample_enabled(time_cfg: Dict[str, Any]) -> bool:
    return _safe_bool((time_cfg or {}).get("use_dynamic_in_sample_windows"), default=True)


def _format_date(ts: Optional[pd.Timestamp]) -> Optional[str]:
    if not isinstance(ts, pd.Timestamp):
        return None
    return ts.strftime("%Y-%m-%d")


def prediction_window_payload(policy: Dict[str, Any]) -> Dict[str, Any]:
    start_ts = policy.get("start_date")
    end_ts = policy.get("end_date")
    horizon = _safe_int(policy.get("price_horizon_days")) or 1
    return {
        "start_date": _format_date(start_ts),
        "end_date": _format_date(end_ts),
        "price_horizon_days": int(max(1, horizon)),
    }


def _resolve_price_window_with_bounds(
    *,
    min_date: Optional[pd.Timestamp],
    max_date: Optional[pd.Timestamp],
    time_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = time_cfg or {}
    dynamic = dynamic_in_sample_enabled(cfg)
    default_horizon = _safe_int(cfg.get("price_forecast_horizon_days")) or 181
    default_horizon = max(1, int(default_horizon))

    cutoff_ts = _safe_timestamp(cfg.get("train_validation_cutoff_date"))
    start_ts = _safe_timestamp(cfg.get("prediction_window_start_date"))
    end_ts = _safe_timestamp(cfg.get("prediction_window_end_date"))

    if dynamic and isinstance(max_date, pd.Timestamp):
        end_ts = max_date
        start_ts = end_ts - pd.Timedelta(days=default_horizon)
        if isinstance(min_date, pd.Timestamp) and start_ts < min_date:
            start_ts = min_date
        if end_ts <= start_ts:
            start_ts = end_ts - pd.Timedelta(days=1)
        horizon = max(1, int((end_ts - start_ts).days))
        cutoff_ts = start_ts - pd.Timedelta(days=horizon)
        if isinstance(min_date, pd.Timestamp) and cutoff_ts < min_date:
            cutoff_ts = min_date
        if cutoff_ts >= start_ts:
            cutoff_ts = start_ts - pd.Timedelta(days=1)
            if isinstance(min_date, pd.Timestamp) and cutoff_ts < min_date:
                cutoff_ts = min_date
    else:
        cutoff_base = cutoff_ts or pd.Timestamp("2020-12-31")
        start_ts = start_ts or cutoff_base
        end_ts = end_ts or (start_ts + pd.Timedelta(days=default_horizon))

        if isinstance(max_date, pd.Timestamp) and end_ts > max_date:
            end_ts = max_date
        if isinstance(min_date, pd.Timestamp) and start_ts < min_date:
            start_ts = min_date
        if end_ts <= start_ts:
            end_ts = start_ts + pd.Timedelta(days=1)
        horizon = max(1, int((end_ts - start_ts).days))

        if cutoff_ts is None:
            cutoff_ts = start_ts - pd.Timedelta(days=horizon)
        if isinstance(min_date, pd.Timestamp) and cutoff_ts < min_date:
            cutoff_ts = min_date

    horizon = max(1, int((end_ts - start_ts).days))
    if cutoff_ts >= end_ts:
        cutoff_ts = end_ts - pd.Timedelta(days=1)
        if isinstance(min_date, pd.Timestamp) and cutoff_ts < min_date:
            cutoff_ts = min_date

    return {
        "start_date": start_ts.normalize(),
        "end_date": end_ts.normalize(),
        "price_horizon_days": int(horizon),
        "train_validation_cutoff_date": _format_date(cutoff_ts.normalize()) or "2020-12-31",
        "data_start_date": _format_date(min_date),
        "data_end_date": _format_date(max_date),
        "dynamic_in_sample": bool(dynamic and isinstance(max_date, pd.Timestamp)),
    }


def resolve_price_window_from_dates(
    dates: Optional[Iterable[Any]],
    time_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    parsed = _normalize_dates(dates)
    if parsed.empty:
        return _resolve_price_window_with_bounds(min_date=None, max_date=None, time_cfg=time_cfg)
    return _resolve_price_window_with_bounds(
        min_date=parsed.min(),
        max_date=parsed.max(),
        time_cfg=time_cfg,
    )


def resolve_price_window_from_df(
    price_df: Optional[pd.DataFrame],
    time_cfg: Dict[str, Any],
    date_col: str = "date",
) -> Dict[str, Any]:
    if not isinstance(price_df, pd.DataFrame) or price_df.empty or date_col not in price_df.columns:
        return _resolve_price_window_with_bounds(min_date=None, max_date=None, time_cfg=time_cfg)
    parsed = pd.to_datetime(price_df[date_col], errors="coerce").dropna()
    if parsed.empty:
        return _resolve_price_window_with_bounds(min_date=None, max_date=None, time_cfg=time_cfg)
    return _resolve_price_window_with_bounds(
        min_date=parsed.min().normalize(),
        max_date=parsed.max().normalize(),
        time_cfg=time_cfg,
    )


def _price_dir_bounds(price_dir: Path) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    min_date: Optional[pd.Timestamp] = None
    max_date: Optional[pd.Timestamp] = None
    if not price_dir.exists() or not price_dir.is_dir():
        return min_date, max_date

    for path in price_dir.rglob("*.csv"):
        try:
            date_col = pd.read_csv(path, usecols=["date"])["date"]
        except Exception:
            continue
        parsed = pd.to_datetime(date_col, format="%d/%m/%Y", errors="coerce").dropna()
        if parsed.empty:
            continue
        local_min = parsed.min().normalize()
        local_max = parsed.max().normalize()
        if min_date is None or local_min < min_date:
            min_date = local_min
        if max_date is None or local_max > max_date:
            max_date = local_max
    return min_date, max_date


def resolve_price_window_from_price_dir(price_dir: str | Path, time_cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = Path(price_dir)
    if not p.is_absolute():
        p = Path.cwd() / p
    min_date, max_date = _price_dir_bounds(p)
    return _resolve_price_window_with_bounds(min_date=min_date, max_date=max_date, time_cfg=time_cfg)


def resolve_year_window_from_series(
    years: Optional[Iterable[Any]],
    *,
    time_cfg: Dict[str, Any],
    window_years_key: str,
    preferred_end_year: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = time_cfg or {}
    dynamic = dynamic_in_sample_enabled(cfg)
    window_years = _safe_int(cfg.get(window_years_key))
    if window_years is None:
        window_years = _safe_int(cfg.get("yearly_prediction_window_years"))
    window_years = max(1, int(window_years or 3))

    years_num = pd.to_numeric(pd.Series(list(years or [])), errors="coerce").dropna()
    data_min_year = int(years_num.min()) if not years_num.empty else None
    data_max_year = int(years_num.max()) if not years_num.empty else None

    explicit_cutoff_ts = _safe_timestamp(cfg.get("train_validation_cutoff_date"))
    explicit_cutoff_year = int(explicit_cutoff_ts.year) if isinstance(explicit_cutoff_ts, pd.Timestamp) else None

    if dynamic and data_max_year is not None:
        end_year = int(data_max_year)
        if preferred_end_year is not None:
            end_year = min(end_year, int(preferred_end_year))
        start_year = end_year - window_years + 1
        if data_min_year is not None and start_year < data_min_year:
            start_year = data_min_year
        cutoff_year = start_year - 1
    else:
        end_year = int(preferred_end_year) if preferred_end_year is not None else data_max_year
        if end_year is None:
            end_year = int(pd.Timestamp.today().year)
        if explicit_cutoff_year is not None:
            cutoff_year = int(explicit_cutoff_year)
            start_year = cutoff_year + 1
        else:
            start_year = end_year - window_years + 1
            cutoff_year = start_year - 1
        if data_min_year is not None and start_year < data_min_year:
            start_year = data_min_year
            cutoff_year = start_year - 1
        if data_max_year is not None and end_year > data_max_year:
            end_year = data_max_year

    if data_min_year is not None and end_year < data_min_year:
        end_year = data_min_year
    if data_max_year is not None and end_year > data_max_year:
        end_year = data_max_year
    if start_year > end_year:
        start_year = end_year
    cutoff_year = min(int(cutoff_year), int(start_year - 1))

    effective_window_years = max(1, int(end_year - start_year + 1))
    return {
        "start_year": int(start_year),
        "end_year": int(end_year),
        "window_years": int(effective_window_years),
        "requested_window_years": int(window_years),
        "train_validation_cutoff_year": int(cutoff_year),
        "train_validation_cutoff_date": f"{int(cutoff_year)}-12-31",
        "data_start_year": data_min_year,
        "data_end_year": data_max_year,
        "dynamic_in_sample": bool(dynamic and data_max_year is not None),
    }


def resolve_target_year(
    config: Dict[str, Any],
    *,
    cost_df: Optional[pd.DataFrame] = None,
    yield_df: Optional[pd.DataFrame] = None,
    fallback_year: Optional[int] = None,
) -> int:
    cfg = config or {}
    time_cfg = cfg.get("time", {})
    alignment_cfg = cfg.get("alignment", {})
    dynamic = dynamic_in_sample_enabled(time_cfg)

    explicit = _safe_int(alignment_cfg.get("target_year"))

    cap_candidates = []
    if isinstance(cost_df, pd.DataFrame) and "year_start" in cost_df.columns:
        years = pd.to_numeric(cost_df["year_start"], errors="coerce").dropna()
        if not years.empty:
            cap_candidates.append(int(years.max()))
    if isinstance(yield_df, pd.DataFrame) and "year" in yield_df.columns:
        years = pd.to_numeric(yield_df["year"], errors="coerce").dropna()
        if not years.empty:
            cap_candidates.append(int(years.max()))

    cap_year = min(cap_candidates) if cap_candidates else None
    today_year = int(pd.Timestamp.today().year)
    fallback = int(fallback_year) if fallback_year is not None else today_year

    if dynamic:
        if explicit is not None and cap_year is not None:
            return int(min(explicit, cap_year))
        if explicit is not None:
            return int(explicit)
        if cap_year is not None:
            return int(cap_year)
        return fallback

    if explicit is not None:
        return int(explicit)
    if cap_year is not None:
        return int(cap_year)
    return fallback
