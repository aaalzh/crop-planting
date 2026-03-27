import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


PRICE_FLOOR = 1e-6


def _safe_log_price(s: pd.Series, floor: float = PRICE_FLOOR) -> pd.Series:
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    arr = np.log(np.clip(arr, max(float(floor), 1e-9), None))
    return pd.Series(arr, index=s.index)


def _normalize_time_raw_mode(include_raw_time_features: bool, time_raw_mode: Optional[str]) -> str:
    if time_raw_mode is None:
        return "raw" if include_raw_time_features else "none"
    mode = str(time_raw_mode).strip().lower()
    if mode not in {"none", "raw", "onehot"}:
        return "none"
    return mode


def add_time_features(
    df: pd.DataFrame,
    *,
    include_raw_time_features: bool = False,
    time_raw_mode: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["date"].dt.dayofweek.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    out["dayofyear"] = out["date"].dt.dayofyear.astype(int)
    out["day"] = out["date"].dt.day.astype(int)

    # Continuous periodic encoding to avoid 12->1 and week/year boundary jumps.
    out["doy_sin"] = np.sin(2.0 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * out["dayofyear"] / 365.25)
    out["dow_sin"] = np.sin(2.0 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * out["dow"] / 7.0)
    out["week_sin"] = np.sin(2.0 * np.pi * out["weekofyear"] / 52.1775)
    out["week_cos"] = np.cos(2.0 * np.pi * out["weekofyear"] / 52.1775)
    out["month_sin"] = np.sin(2.0 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * out["month"] / 12.0)
    out["is_weekend"] = (out["dow"] >= 5).astype(float)

    raw_mode = _normalize_time_raw_mode(include_raw_time_features, time_raw_mode)
    if raw_mode == "none":
        return out.drop(columns=["dow", "month", "weekofyear", "dayofyear", "day"])

    if raw_mode == "onehot":
        # Keep periodic terms and use one-hot for discrete calendar buckets.
        out = pd.get_dummies(
            out,
            columns=["dow", "month", "weekofyear"],
            prefix=["dow", "month", "week"],
            prefix_sep="_",
            dtype=float,
        )
        return out.drop(columns=["dayofyear", "day"])

    return out


def add_lag_features(df: pd.DataFrame, target_col: str, lags: List[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        lag_i = int(lag)
        if lag_i <= 0:
            continue
        out[f"lag_{lag_i}"] = out[target_col].shift(lag_i)
    return out


def add_roll_features(df: pd.DataFrame, target_col: str, windows: List[int]) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        win_i = int(w)
        if win_i <= 0:
            continue
        out[f"roll_mean_{win_i}"] = out[target_col].rolling(window=win_i).mean()
        out[f"roll_std_{win_i}"] = out[target_col].rolling(window=win_i).std()
        out[f"roll_min_{win_i}"] = out[target_col].rolling(window=win_i).min()
        out[f"roll_max_{win_i}"] = out[target_col].rolling(window=win_i).max()
    return out


def make_supervised(
    df: pd.DataFrame,
    target_col: str,
    horizon: int,
    lags: List[int],
    windows: List[int],
    return_dates: bool = False,
    *,
    return_target_dates: bool = False,
    target_mode: str = "price",
    feature_space: str = "price",
    include_raw_time_features: bool = False,
    time_raw_mode: Optional[str] = None,
    price_floor: float = PRICE_FLOOR,
) -> Tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=["date", target_col]).sort_values("date").reset_index(drop=True)

    safe_horizon = max(1, int(horizon))
    work = add_time_features(
        work,
        include_raw_time_features=include_raw_time_features,
        time_raw_mode=time_raw_mode,
    )

    fspace = str(feature_space or "price").strip().lower()
    if fspace == "log_price":
        work["_feature_base"] = _safe_log_price(work[target_col], floor=price_floor)
    else:
        work["_feature_base"] = pd.to_numeric(work[target_col], errors="coerce")

    work = add_lag_features(work, "_feature_base", lags)
    work = add_roll_features(work, "_feature_base", windows)

    tmode = str(target_mode or "price").strip().lower()
    if tmode == "log_return":
        log_price = _safe_log_price(work[target_col], floor=price_floor)
        work["y"] = log_price.shift(-safe_horizon) - log_price
    else:
        work["y"] = pd.to_numeric(work[target_col], errors="coerce").shift(-safe_horizon)

    work = work.dropna().reset_index(drop=True)
    y = pd.to_numeric(work["y"], errors="coerce").astype(float)
    feature_dates = work["date"].copy()
    target_dates = feature_dates + pd.to_timedelta(safe_horizon, unit="D")
    X = work.drop(columns=["y", target_col, "date", "_feature_base"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")

    if return_dates and return_target_dates:
        return X, y, feature_dates, target_dates
    if return_dates:
        return X, y, feature_dates
    return X, y


def make_recent_features(
    df: pd.DataFrame,
    target_col: str,
    horizon: int,
    lags: List[int],
    windows: List[int],
    *,
    feature_space: str = "price",
    include_raw_time_features: bool = False,
    time_raw_mode: Optional[str] = None,
    price_floor: float = PRICE_FLOOR,
) -> pd.DataFrame:
    """Build the latest feature row for forecasting horizon days ahead."""
    _ = max(1, int(horizon))  # keep signature compatibility and explicit intent
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=["date", target_col]).sort_values("date").reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    work = add_time_features(
        work,
        include_raw_time_features=include_raw_time_features,
        time_raw_mode=time_raw_mode,
    )

    fspace = str(feature_space or "price").strip().lower()
    if fspace == "log_price":
        work["_feature_base"] = _safe_log_price(work[target_col], floor=price_floor)
    else:
        work["_feature_base"] = pd.to_numeric(work[target_col], errors="coerce")

    work = add_lag_features(work, "_feature_base", lags)
    work = add_roll_features(work, "_feature_base", windows)
    work = work.dropna().reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    X = work.drop(columns=[target_col, "date", "_feature_base"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.tail(1)

