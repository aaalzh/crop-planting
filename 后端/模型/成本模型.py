from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


DEFAULT_VALIDATION_CUTOFF_DATE = "2020-12-31"
DEFAULT_VALIDATION_CUTOFF_YEAR = 2020


@dataclass
class CostModelResult:
    model: object
    metrics: Dict[str, float]
    feature_cols: list
    feature_meta: Optional[dict] = None
    fill_values: Optional[dict] = None


class EnsembleModel:
    def __init__(
        self,
        models,
        weights,
        feature_cols: Optional[List[str]] = None,
        feature_meta: Optional[dict] = None,
        fill_values: Optional[dict] = None,
        huber_index: Optional[int] = None,
        huber_blend_weight: float = 0.0,
    ):
        self.models = models
        self.weights = weights
        self.feature_cols = list(feature_cols or [])
        self.cost_feature_meta = dict(feature_meta or {})
        self.cost_fill_values = dict(fill_values or {})
        try:
            idx = int(huber_index) if huber_index is not None else None
        except Exception:
            idx = None
        if idx is not None and (idx < 0 or idx >= len(self.models)):
            idx = None
        self.huber_index = idx
        self.huber_blend_weight = max(0.0, min(1.0, float(huber_blend_weight)))

    def predict(self, X: pd.DataFrame):
        preds = []
        for m in self.models:
            preds.append(m.predict(X))
        preds = np.vstack(preds)
        w = np.array(self.weights).reshape(-1, 1)
        out = (preds * w).sum(axis=0)
        huber_index = getattr(self, "huber_index", None)
        huber_blend_weight = max(0.0, min(1.0, float(getattr(self, "huber_blend_weight", 0.0))))
        if huber_index is not None and huber_blend_weight > 0.0:
            huber_pred = preds[int(huber_index)]
            out = (1.0 - huber_blend_weight) * out + huber_blend_weight * huber_pred
        return np.clip(out, 0.0, None)


class CostPanelServingModel:
    def __init__(
        self,
        model: object,
        crop: str,
        cost_name: str = "",
        crop_group: str = "",
        feature_cols: Optional[List[str]] = None,
        feature_meta: Optional[dict] = None,
        fill_values: Optional[dict] = None,
    ):
        self.model = model
        self.serving_crop = str(crop or "").strip().lower()
        self.serving_cost_name = str(cost_name or "").strip()
        self.serving_crop_group = str(crop_group or "").strip().lower()
        self.feature_cols = list(feature_cols or [])
        self.cost_feature_meta = dict(feature_meta or {})
        self.cost_fill_values = dict(fill_values or {})

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)


def _safe_year_int(value: object) -> Optional[int]:
    try:
        year = int(float(value))
    except Exception:
        return None
    if year < 1900 or year > 2200:
        return None
    return year


def _resolve_cutoff_year(value: Optional[str]) -> int:
    text = str(value or "").strip() or DEFAULT_VALIDATION_CUTOFF_DATE
    try:
        ts = pd.Timestamp(text)
        year = int(ts.year)
    except Exception:
        year = DEFAULT_VALIDATION_CUTOFF_YEAR
    return _safe_year_int(year) or DEFAULT_VALIDATION_CUTOFF_YEAR


def _year_at_index(years: pd.Series, idx: np.ndarray, pick: str) -> Optional[int]:
    if len(idx) == 0:
        return None
    vals = pd.to_numeric(years.iloc[idx], errors="coerce").dropna()
    if vals.empty:
        return None
    target = vals.min() if pick == "min" else vals.max()
    return _safe_year_int(target)


def _build_regressor(cfg: dict, reg: str):
    n_jobs = int(cfg.get("n_jobs", 1))
    if reg == "rf":
        return RandomForestRegressor(
            n_estimators=int(cfg.get("rf_n_estimators", 500)),
            min_samples_leaf=int(cfg.get("rf_min_samples_leaf", 2)),
            random_state=42,
            n_jobs=n_jobs,
        )
    if reg == "etr":
        return ExtraTreesRegressor(
            n_estimators=int(cfg.get("etr_n_estimators", 700)),
            min_samples_leaf=int(cfg.get("etr_min_samples_leaf", 2)),
            random_state=42,
            n_jobs=n_jobs,
        )
    if reg == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=int(cfg.get("hgb_max_iter", 600)),
            max_depth=cfg.get("hgb_max_depth", 6),
            learning_rate=float(cfg.get("hgb_learning_rate", 0.03)),
            l2_regularization=float(cfg.get("hgb_l2_regularization", 1e-3)),
            loss=str(cfg.get("hgb_loss", "squared_error")),
            random_state=42,
        )
    if reg == "huber":
        return HuberRegressor(
            alpha=float(cfg.get("huber_alpha", 1.0)),
            epsilon=float(cfg.get("huber_epsilon", 1.35)),
            max_iter=int(cfg.get("huber_max_iter", 200)),
        )
    if reg == "qgb":
        return GradientBoostingRegressor(
            loss="quantile",
            alpha=float(cfg.get("qgb_alpha", 0.5)),
            n_estimators=int(cfg.get("qgb_n_estimators", 500)),
            learning_rate=float(cfg.get("qgb_learning_rate", 0.03)),
            max_depth=int(cfg.get("qgb_max_depth", 2)),
            min_samples_leaf=int(cfg.get("qgb_min_samples_leaf", 2)),
            random_state=42,
        )
    if reg == "gbr_huber":
        return GradientBoostingRegressor(
            loss="huber",
            alpha=float(cfg.get("gbr_huber_alpha", 0.9)),
            n_estimators=int(cfg.get("gbr_n_estimators", 500)),
            learning_rate=float(cfg.get("gbr_learning_rate", 0.03)),
            max_depth=int(cfg.get("gbr_max_depth", 2)),
            min_samples_leaf=int(cfg.get("gbr_min_samples_leaf", 2)),
            random_state=42,
        )
    return Ridge(alpha=float(cfg.get("alpha", 1.0)))


def _wrap_target_transform(model, cfg: dict):
    t = str(cfg.get("target_transform", "none")).lower()
    if t == "log1p":
        return TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    return model


def _cost_feature_meta(df: pd.DataFrame, cfg: Optional[dict] = None) -> dict:
    cfg = cfg or {}
    feature_set = str(cfg.get("feature_set", "legacy")).strip().lower()
    if feature_set == "legacy":
        return {"feature_set": "legacy", "anchor": 1990.0, "year_min": 1990.0}

    year_raw = pd.to_numeric(df["year_start"], errors="coerce").astype(float)
    year_clean = year_raw[np.isfinite(year_raw)]
    if year_clean.empty:
        anchor = 1990.0
        year_min = 1990.0
    else:
        anchor = float(np.nanmedian(year_clean))
        year_min = float(np.nanmin(year_clean))
    return {"feature_set": "extended", "anchor": anchor, "year_min": year_min}


def _make_cost_features(df: pd.DataFrame, cfg: Optional[dict] = None, feature_meta: Optional[dict] = None) -> pd.DataFrame:
    cfg = cfg or {}
    meta = dict(feature_meta or _cost_feature_meta(df, cfg=cfg))
    feature_set = str(meta.get("feature_set", cfg.get("feature_set", "legacy"))).strip().lower()
    x = pd.DataFrame()
    year_raw = pd.to_numeric(df["year_start"], errors="coerce").astype(float)
    if feature_set == "legacy":
        x["year"] = year_raw - 1990.0
        x["year2"] = x["year"] ** 2
        return x

    # Extended features for robustness experiments.
    anchor = float(meta.get("anchor", 1990.0))
    x["year"] = year_raw - anchor
    x["year2"] = x["year"] ** 2
    x["year3"] = x["year"] ** 3
    year_min = float(meta.get("year_min", 1990.0))
    year_shift = np.maximum(year_raw - year_min + 1.0, 1.0)
    x["log_year"] = np.log1p(year_shift)
    x["sqrt_year"] = np.sqrt(year_shift)
    return x


def _norm_text(value: object) -> str:
    return str(value or "").strip().lower()


def crop_group_from_cost_name(cost_name: str) -> str:
    name = _norm_text(cost_name)
    if "proxy" in name:
        if "horticulture" in name:
            return "horticulture"
        if "plantation" in name:
            return "plantation"
        if "pulse" in name:
            return "pulses"
        return "proxy"
    if any(token in name for token in ["gram", "lentil", "moong", "urad", "tur", "bean"]):
        return "pulses"
    if any(token in name for token in ["paddy", "rice", "maize", "wheat", "bajra", "barley", "jowar", "ragi"]):
        return "cereal"
    if any(token in name for token in ["cotton", "jute", "sugarcane"]):
        return "industrial"
    if any(token in name for token in ["coffee", "tea", "coconut", "banana"]):
        return "plantation"
    if any(token in name for token in ["onion", "potato", "tapioca", "vegetable"]):
        return "vegetable"
    if any(token in name for token in ["apple", "orange", "mango", "papaya", "grape", "melon", "pomegranate"]):
        return "horticulture"
    return "other"


def _panel_lite_lags(cfg: dict) -> List[int]:
    raw = cfg.get("panel_lags", [1, 2, 3, 4])
    out = []
    for item in list(raw or []):
        try:
            lag = int(item)
        except Exception:
            continue
        if lag > 0 and lag not in out:
            out.append(lag)
    return out or [1, 2, 3, 4]


def _panel_lite_windows(cfg: dict) -> List[int]:
    raw = cfg.get("panel_windows", [2, 3, 5, 7])
    out = []
    for item in list(raw or []):
        try:
            win = int(item)
        except Exception:
            continue
        if win > 0 and win not in out:
            out.append(win)
    return out or [2, 3, 5, 7]


def _panel_lite_feature_meta(df: pd.DataFrame, cfg: Optional[dict] = None) -> dict:
    cfg = cfg or {}
    years = pd.to_numeric(df.get("year"), errors="coerce").dropna()
    if years.empty:
        anchor_year = 1990
    else:
        try:
            anchor_year = int(cfg.get("panel_anchor_year", int(years.min())))
        except Exception:
            anchor_year = int(years.min())
    return {
        "feature_set": "panel_lite",
        "anchor_year": int(anchor_year),
        "lags": _panel_lite_lags(cfg),
        "windows": _panel_lite_windows(cfg),
        "include_crop_hierarchy": bool(cfg.get("panel_include_crop_hierarchy", True)),
        "include_price_lags": bool(cfg.get("panel_include_price_lags", True)),
        "include_yield_lags": bool(cfg.get("panel_include_yield_lags", True)),
        "include_quality_lags": bool(cfg.get("panel_include_quality_lags", True)),
        "lag1_blend_weight": max(0.0, min(1.0, float(cfg.get("panel_lag1_blend_weight", 0.0)))),
    }


def _ensure_panel_lite_columns(
    df: pd.DataFrame,
    *,
    crop: Optional[str] = None,
    cost_name: Optional[str] = None,
    crop_group: Optional[str] = None,
) -> pd.DataFrame:
    out = pd.DataFrame(df).copy()

    if "crop" not in out.columns:
        out["crop"] = str(crop or "").strip().lower()
    out["crop"] = out["crop"].astype(str).str.strip().str.lower()

    if "cost_name" not in out.columns:
        out["cost_name"] = str(cost_name or "").strip()
    out["cost_name"] = out["cost_name"].astype(str).str.strip()

    if "crop_group" not in out.columns:
        out["crop_group"] = ""
    inferred_group = str(crop_group or "").strip().lower()
    out["crop_group"] = out["crop_group"].astype(str).str.strip().str.lower()
    if inferred_group:
        out.loc[out["crop_group"] == "", "crop_group"] = inferred_group
    out.loc[out["crop_group"] == "", "crop_group"] = out.loc[out["crop_group"] == "", "cost_name"].map(
        crop_group_from_cost_name
    )
    out["crop_group"] = out["crop_group"].replace("", "other")

    if "year" not in out.columns:
        out["year"] = out.get("year_start")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(float)

    if "cost" not in out.columns:
        out["cost"] = out.get("india_cost_wavg_sample")
    out["cost"] = pd.to_numeric(out["cost"], errors="coerce")

    for col in ["price", "yield", "n_states", "n_rows_cost", "sum_sample_weight", "price_year_std", "price_obs_days"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _panel_lite_year_features(df: pd.DataFrame, feature_meta: dict) -> pd.DataFrame:
    out = df.copy()
    anchor = float(feature_meta.get("anchor_year", 1990))
    out["year_idx"] = pd.to_numeric(out["year"], errors="coerce").astype(float) - anchor
    out["year_idx2"] = out["year_idx"] ** 2
    out["year_sin"] = np.sin(2.0 * np.pi * out["year_idx"] / 12.0)
    out["year_cos"] = np.cos(2.0 * np.pi * out["year_idx"] / 12.0)
    return out


def _panel_lite_add_lag_family(
    df: pd.DataFrame,
    value_col: str,
    prefix: str,
    lags: List[int],
    windows: List[int],
) -> pd.DataFrame:
    out = df.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    g = out.groupby("crop")[value_col]
    for lag in lags:
        out[f"{prefix}_lag{lag}"] = g.shift(lag)
    shifted = g.shift(1)
    for win in windows:
        out[f"{prefix}_roll_mean_{win}"] = shifted.groupby(out["crop"]).transform(
            lambda s: s.rolling(win, min_periods=1).mean()
        )
        out[f"{prefix}_roll_std_{win}"] = shifted.groupby(out["crop"]).transform(
            lambda s: s.rolling(win, min_periods=1).std()
        ).fillna(0.0)
    out[f"{prefix}_exp_mean"] = shifted.groupby(out["crop"]).transform(lambda s: s.expanding().mean())
    out[f"{prefix}_exp_std"] = shifted.groupby(out["crop"]).transform(lambda s: s.expanding().std()).fillna(0.0)

    if f"{prefix}_lag1" in out.columns and f"{prefix}_lag2" in out.columns:
        lag1 = pd.to_numeric(out[f"{prefix}_lag1"], errors="coerce")
        lag2 = pd.to_numeric(out[f"{prefix}_lag2"], errors="coerce")
        denom = np.maximum(np.abs(lag2.to_numpy(dtype=float)), 1e-6)
        out[f"{prefix}_diff_1_2"] = lag1 - lag2
        out[f"{prefix}_ratio_1_2"] = lag1.to_numpy(dtype=float) / denom
        out[f"{prefix}_yoy_1_2"] = out[f"{prefix}_ratio_1_2"] - 1.0

    if f"{prefix}_lag1" in out.columns and f"{prefix}_exp_mean" in out.columns and f"{prefix}_exp_std" in out.columns:
        lag1 = pd.to_numeric(out[f"{prefix}_lag1"], errors="coerce").to_numpy(dtype=float)
        exp_mean = pd.to_numeric(out[f"{prefix}_exp_mean"], errors="coerce").to_numpy(dtype=float)
        exp_std = pd.to_numeric(out[f"{prefix}_exp_std"], errors="coerce").to_numpy(dtype=float)
        z = (lag1 - exp_mean) / np.maximum(exp_std, 1e-6)
        out[f"{prefix}_zscore_lag1"] = np.clip(z, -8.0, 8.0)

    if windows:
        w_min = min(windows)
        w_max = max(windows)
        c_min = f"{prefix}_roll_mean_{w_min}"
        c_max = f"{prefix}_roll_mean_{w_max}"
        if c_min in out.columns and c_max in out.columns and w_min != w_max:
            out[f"{prefix}_roll_mean_gap_{w_min}_{w_max}"] = out[c_min] - out[c_max]
    return out


def _panel_lite_add_gap_features(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    observed_year = out["year"].where(out[value_col].notna())
    prev_year = observed_year.groupby(out["crop"]).ffill().groupby(out["crop"]).shift(1)
    gap = pd.to_numeric(out["year"], errors="coerce") - pd.to_numeric(prev_year, errors="coerce")
    out[f"{prefix}_gap_years"] = gap.fillna(99.0).clip(0.0, 99.0)
    seen = out[value_col].notna().astype(float)
    out[f"{prefix}_hist_count"] = seen.groupby(out["crop"]).cumsum().groupby(out["crop"]).shift(1).fillna(0.0)
    return out


def _panel_lite_add_flag_lags(df: pd.DataFrame, col: str, prefix: str, lags: List[int]) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    g = out.groupby("crop")[col]
    for lag in lags:
        out[f"{prefix}_lag{lag}"] = g.shift(lag)
    return out


def _panel_lite_with_crop_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["crop"] = out["crop"].astype(str)
    out["crop_group"] = out["crop_group"].astype(str)
    crop_d = pd.get_dummies(out["crop"], prefix="crop", dtype=float)
    group_d = pd.get_dummies(out["crop_group"], prefix="group", dtype=float)
    return pd.concat([out, crop_d, group_d], axis=1)


def _default_cost_fill_value(col: str) -> float:
    if col.startswith("crop_") or col.startswith("group_"):
        return 0.0
    if col.endswith("_hist_count"):
        return 0.0
    if col.endswith("_gap_years"):
        return 99.0
    return 0.0


def _derive_cost_fill_values(X: pd.DataFrame) -> dict:
    fill_values: Dict[str, float] = {}
    for col in list(X.columns):
        series = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            fill_values[str(col)] = _default_cost_fill_value(str(col))
            continue
        fill_values[str(col)] = float(series.median())
    return fill_values


def _apply_cost_fill_values(X: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
    out = pd.DataFrame(X).copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    for col, value in dict(fill_values or {}).items():
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(float(value))
    for col in out.columns:
        if col not in fill_values:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(_default_cost_fill_value(str(col)))
    return out


def _apply_panel_lag1_blend(pred: np.ndarray, X: pd.DataFrame, feature_meta: dict) -> np.ndarray:
    weight = max(0.0, min(1.0, float((feature_meta or {}).get("lag1_blend_weight", 0.0))))
    if weight <= 0.0 or "cost_lag1" not in X.columns:
        return np.asarray(pred, dtype=float)
    lag1 = pd.to_numeric(X["cost_lag1"], errors="coerce").to_numpy(dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    out = np.where(np.isfinite(lag1), (1.0 - weight) * pred_arr + weight * lag1, pred_arr)
    return np.clip(out, 0.0, None)


def _make_panel_lite_features(
    df: pd.DataFrame,
    cfg: Optional[dict] = None,
    feature_meta: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = cfg or {}
    work = _ensure_panel_lite_columns(df)
    work = work.dropna(subset=["crop", "year"]).sort_values(["crop", "year"]).reset_index(drop=True)

    meta = dict(feature_meta or _panel_lite_feature_meta(work, cfg=cfg))
    lags = list(meta.get("lags", _panel_lite_lags(cfg)))
    windows = list(meta.get("windows", _panel_lite_windows(cfg)))
    include_crop_hierarchy = bool(meta.get("include_crop_hierarchy", True))
    include_price_lags = bool(meta.get("include_price_lags", True))
    include_yield_lags = bool(meta.get("include_yield_lags", True))
    include_quality_lags = bool(meta.get("include_quality_lags", True))

    work = _panel_lite_year_features(work, meta)

    value_specs: List[Tuple[str, str]] = [("cost", "cost")]
    if include_price_lags:
        value_specs.append(("price", "price"))
    if include_yield_lags:
        value_specs.append(("yield", "yield"))
    if include_quality_lags:
        value_specs.extend(
            [
                ("n_states", "n_states"),
                ("n_rows_cost", "n_rows_cost"),
                ("sum_sample_weight", "sum_sample_weight"),
                ("price_year_std", "price_year_std"),
                ("price_obs_days", "price_obs_days"),
            ]
        )

    for value_col, prefix in value_specs:
        work = _panel_lite_add_lag_family(work, value_col=value_col, prefix=prefix, lags=lags, windows=windows)
        work = _panel_lite_add_gap_features(work, value_col=value_col, prefix=prefix)

    if include_crop_hierarchy:
        work = _panel_lite_with_crop_features(work)

    feature_cols: List[str] = ["year", "year_idx", "year_idx2", "year_sin", "year_cos", "cost_gap_years", "cost_hist_count"]
    if include_price_lags:
        feature_cols.extend(["price_gap_years", "price_hist_count"])
    if include_yield_lags:
        feature_cols.extend(["yield_gap_years", "yield_hist_count"])
    if include_quality_lags:
        feature_cols.extend(
            [
                "n_states_gap_years",
                "n_states_hist_count",
                "n_rows_cost_gap_years",
                "n_rows_cost_hist_count",
                "sum_sample_weight_gap_years",
                "sum_sample_weight_hist_count",
                "price_year_std_gap_years",
                "price_year_std_hist_count",
                "price_obs_days_gap_years",
                "price_obs_days_hist_count",
            ]
        )

    dynamic_prefixes = ["cost_"]
    if include_price_lags:
        dynamic_prefixes.extend(["price_"])
    if include_yield_lags:
        dynamic_prefixes.extend(["yield_"])
    if include_quality_lags:
        dynamic_prefixes.extend(
            ["n_states_", "n_rows_cost_", "sum_sample_weight_", "price_year_std_", "price_obs_days_"]
        )

    for col in work.columns:
        if any(col.startswith(prefix) for prefix in dynamic_prefixes):
            feature_cols.append(col)
        if include_crop_hierarchy and (col.startswith("crop_") or col.startswith("group_")):
            feature_cols.append(col)

    feature_cols = [str(col) for col in feature_cols if col in work.columns]
    feature_cols = list(dict.fromkeys(feature_cols))
    X = work[feature_cols].copy().apply(pd.to_numeric, errors="coerce")
    return X, work, meta


def _aggregate_price_yearly(price_history_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if not isinstance(price_history_df, pd.DataFrame) or price_history_df.empty:
        return pd.DataFrame(columns=["year", "price", "price_year_std", "price_obs_days"])
    if "date" not in price_history_df.columns:
        return pd.DataFrame(columns=["year", "price", "price_year_std", "price_obs_days"])
    price_col = "modal_price" if "modal_price" in price_history_df.columns else "price"
    work = price_history_df[["date", price_col]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work = work.dropna(subset=["date", price_col])
    if work.empty:
        return pd.DataFrame(columns=["year", "price", "price_year_std", "price_obs_days"])
    work["year"] = work["date"].dt.year.astype(int)
    out = work.groupby("year", as_index=False).agg(
        price=(price_col, "mean"),
        price_year_std=(price_col, "std"),
        price_obs_days=(price_col, "count"),
    )
    out["price_year_std"] = pd.to_numeric(out["price_year_std"], errors="coerce").fillna(0.0)
    out["price_obs_days"] = pd.to_numeric(out["price_obs_days"], errors="coerce").fillna(0.0)
    return out


def _aggregate_yield_yearly(yield_history_df: Optional[pd.DataFrame], crop: str) -> pd.DataFrame:
    if not isinstance(yield_history_df, pd.DataFrame) or yield_history_df.empty:
        return pd.DataFrame(columns=["year", "yield"])
    crop_col = "crop_name" if "crop_name" in yield_history_df.columns else "crop"
    year_col = "year" if "year" in yield_history_df.columns else None
    target_col = "yield_quintal_per_hectare" if "yield_quintal_per_hectare" in yield_history_df.columns else "yield"
    if crop_col not in yield_history_df.columns or year_col is None or target_col not in yield_history_df.columns:
        return pd.DataFrame(columns=["year", "yield"])
    work = yield_history_df[[crop_col, year_col, target_col]].copy()
    work[crop_col] = work[crop_col].astype(str).str.strip().str.lower()
    work = work[work[crop_col] == str(crop or "").strip().lower()].copy()
    if work.empty:
        return pd.DataFrame(columns=["year", "yield"])
    work["year"] = pd.to_numeric(work[year_col], errors="coerce")
    work["yield"] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=["year", "yield"])
    if work.empty:
        return pd.DataFrame(columns=["year", "yield"])
    out = work.groupby("year", as_index=False).agg(
        yield_value=("yield", "mean"),
    )
    out = out.rename(columns={"yield_value": "yield"})
    out["year"] = out["year"].astype(int)
    return out


def _aggregate_cost_history(cost_history_df: Optional[pd.DataFrame], cost_name: str = "") -> pd.DataFrame:
    if not isinstance(cost_history_df, pd.DataFrame) or cost_history_df.empty:
        return pd.DataFrame(
            columns=["year", "cost", "n_states", "n_rows_cost", "sum_sample_weight"]
        )
    year_col = "year" if "year" in cost_history_df.columns else "year_start"
    target_col = "cost" if "cost" in cost_history_df.columns else "india_cost_wavg_sample"
    if year_col not in cost_history_df.columns or target_col not in cost_history_df.columns:
        return pd.DataFrame(
            columns=["year", "cost", "n_states", "n_rows_cost", "sum_sample_weight"]
        )
    keep_cols = [year_col, target_col]
    for col in ["n_states", "n_rows_cost", "sum_sample_weight"]:
        if col in cost_history_df.columns:
            keep_cols.append(col)
    work = cost_history_df[keep_cols].copy()
    work["year"] = pd.to_numeric(work[year_col], errors="coerce")
    work["cost"] = pd.to_numeric(work[target_col], errors="coerce")
    for col in ["n_states", "n_rows_cost", "sum_sample_weight"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["year"])
    if work.empty:
        return pd.DataFrame(
            columns=["year", "cost", "n_states", "n_rows_cost", "sum_sample_weight"]
        )
    out = work.groupby("year", as_index=False).agg(
        cost=("cost", "mean"),
        n_states=("n_states", "mean"),
        n_rows_cost=("n_rows_cost", "mean"),
        sum_sample_weight=("sum_sample_weight", "mean"),
    )
    out["year"] = out["year"].astype(int)
    return out


def _panel_lite_prediction_row(
    year: int,
    *,
    feature_cols: List[str],
    feature_meta: dict,
    fill_values: dict,
    cost_history_df: Optional[pd.DataFrame],
    price_history_df: Optional[pd.DataFrame],
    yield_history_df: Optional[pd.DataFrame],
    crop: str,
    cost_name: str,
    crop_group: str,
) -> pd.DataFrame:
    target_year = int(year)
    cost_hist = _aggregate_cost_history(cost_history_df, cost_name=cost_name)
    price_hist = _aggregate_price_yearly(price_history_df)
    yield_hist = _aggregate_yield_yearly(yield_history_df, crop=crop)

    year_pool = set()
    for frame in (cost_hist, price_hist, yield_hist):
        if not frame.empty and "year" in frame.columns:
            year_pool.update(pd.to_numeric(frame["year"], errors="coerce").dropna().astype(int).tolist())
    year_pool = {y for y in year_pool if y <= target_year}
    year_pool.add(target_year)
    years = sorted(year_pool)

    base = pd.DataFrame(
        {
            "crop": [str(crop or "").strip().lower()] * len(years),
            "year": years,
            "cost_name": [str(cost_name or "").strip()] * len(years),
            "crop_group": [str(crop_group or "").strip().lower() or crop_group_from_cost_name(cost_name)] * len(years),
        }
    )
    base = base.merge(cost_hist, on="year", how="left")
    base = base.merge(price_hist, on="year", how="left")
    base = base.merge(yield_hist, on="year", how="left")

    mask_target = pd.to_numeric(base["year"], errors="coerce") == float(target_year)
    for col in ["cost", "price", "yield", "n_states", "n_rows_cost", "sum_sample_weight", "price_year_std", "price_obs_days"]:
        if col in base.columns:
            base.loc[mask_target, col] = np.nan
    X_all, work, _ = _make_panel_lite_features(base, feature_meta=feature_meta)
    row = X_all.loc[pd.to_numeric(work["year"], errors="coerce") == float(target_year)].tail(1)
    if row.empty:
        row = pd.DataFrame([{col: np.nan for col in feature_cols}])
    row = row.reindex(columns=feature_cols, fill_value=np.nan)
    return _apply_cost_fill_values(row, fill_values)


def _split_train_test(
    years: pd.Series,
    n_rows: int,
    test_ratio: Optional[float] = None,
    validation_cutoff: Optional[str] = None,
    strict_cutoff_split: bool = True,
):
    ratio = 0.2 if test_ratio is None else float(test_ratio)
    cutoff_year = _resolve_cutoff_year(validation_cutoff)

    years_num = pd.to_numeric(years, errors="coerce").astype(float)
    years_num = years_num.reindex(range(n_rows))
    train_mask = (years_num <= cutoff_year).fillna(False).to_numpy(dtype=bool)
    test_mask = (years_num > cutoff_year).fillna(False).to_numpy(dtype=bool)

    if train_mask.any() and test_mask.any():
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        split_info = {
            "split_mode": "fixed_cutoff",
            "split_cutoff_year": int(cutoff_year),
            "split_used_post_cutoff_validation": True,
            "split_post_cutoff_rows": int(test_mask.sum()),
            "split_train_rows": int(len(train_idx)),
            "split_test_rows": int(len(test_idx)),
            "split_train_start_year": _year_at_index(years_num, train_idx, "min"),
            "split_train_end_year": _year_at_index(years_num, train_idx, "max"),
            "split_test_start_year": _year_at_index(years_num, test_idx, "min"),
            "split_test_end_year": _year_at_index(years_num, test_idx, "max"),
        }
        return train_idx, test_idx, split_info

    if train_mask.any() and not test_mask.any() and strict_cutoff_split:
        train_idx = np.where(train_mask)[0]
        test_idx = np.asarray([], dtype=int)
        split_info = {
            "split_mode": "fixed_cutoff_no_validation",
            "split_cutoff_year": int(cutoff_year),
            "split_used_post_cutoff_validation": False,
            "split_post_cutoff_rows": 0,
            "split_train_rows": int(len(train_idx)),
            "split_test_rows": 0,
            "split_train_start_year": _year_at_index(years_num, train_idx, "min"),
            "split_train_end_year": _year_at_index(years_num, train_idx, "max"),
            "split_test_start_year": None,
            "split_test_end_year": None,
        }
        return train_idx, test_idx, split_info

    n_test = max(2, int(n_rows * ratio))
    if n_rows <= n_test:
        raise ValueError("not enough rows for train/test split")
    train_idx = np.arange(0, n_rows - n_test)
    test_idx = np.arange(n_rows - n_test, n_rows)
    split_info = {
        "split_mode": "ratio_fallback",
        "split_cutoff_year": int(cutoff_year),
        "split_used_post_cutoff_validation": False,
        "split_post_cutoff_rows": int(test_mask.sum()),
        "split_train_rows": int(len(train_idx)),
        "split_test_rows": int(len(test_idx)),
        "split_train_start_year": _year_at_index(years_num, train_idx, "min"),
        "split_train_end_year": _year_at_index(years_num, train_idx, "max"),
        "split_test_start_year": _year_at_index(years_num, test_idx, "min"),
        "split_test_end_year": _year_at_index(years_num, test_idx, "max"),
    }
    return train_idx, test_idx, split_info


def _recency_weights(X_train: pd.DataFrame, cfg: dict):
    if not bool(cfg.get("use_recency_weight", True)):
        return None
    if "year" not in X_train.columns:
        return None
    years = pd.to_numeric(X_train["year"], errors="coerce").astype(float)
    if years.empty:
        return None
    max_year = float(years.max())
    ages = np.maximum(max_year - years.values, 0.0)
    half_life = max(1.0, float(cfg.get("recency_halflife_years", 8.0)))
    w = 0.5 ** (ages / half_life)
    w = np.asarray(w, dtype=float)
    if not np.all(np.isfinite(w)) or np.sum(w) <= 0:
        return None
    return w / np.mean(w)


def _fit_model(model, X_train, y_train, sample_weight=None):
    if sample_weight is None:
        model.fit(X_train, y_train)
        return model
    try:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    except TypeError:
        model.fit(X_train, y_train)
    return model


def _fit_and_pred(X_train, y_train, X_test, cfg: dict, reg: str):
    model = _wrap_target_transform(_build_regressor(cfg, reg), cfg)
    sw = _recency_weights(X_train, cfg)
    model = _fit_model(model, X_train, y_train, sample_weight=sw)
    if X_test.empty:
        pred = np.asarray([], dtype=float)
    else:
        pred = np.clip(model.predict(X_test), 0.0, None)
    return model, pred


def _eval_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        return {"mae": None, "rmse": None, "mape": None}
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))))
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _inner_val_mae(X_train, y_train, cfg: dict, reg: str):
    n_val = max(2, int(len(X_train) * 0.25))
    if len(X_train) <= n_val + 2:
        return None
    X_fit, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    y_fit, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]
    model = _wrap_target_transform(_build_regressor(cfg, reg), cfg)
    sw = _recency_weights(X_fit, cfg)
    model = _fit_model(model, X_fit, y_fit, sample_weight=sw)
    pred = np.clip(model.predict(X_val), 0.0, None)
    return float(mean_absolute_error(y_val, pred))


def _select_auto_regressor(X_train, y_train, cfg: dict) -> str:
    candidates = cfg.get("auto_candidates", ["ridge", "huber", "hgb", "qgb", "etr", "rf"])
    n_val = max(2, int(len(X_train) * 0.25))
    if len(X_train) <= n_val + 2:
        return "ridge"

    X_fit, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    y_fit, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]
    best_reg = "ridge"
    best_mae = float("inf")

    for reg in candidates:
        try:
            model = _wrap_target_transform(_build_regressor(cfg, reg), cfg)
            sw = _recency_weights(X_fit, cfg)
            model = _fit_model(model, X_fit, y_fit, sample_weight=sw)
            pred = np.clip(model.predict(X_val), 0.0, None)
            mae = float(mean_absolute_error(y_val, pred))
            if mae < best_mae:
                best_mae = mae
                best_reg = reg
        except Exception:
            continue
    return best_reg


def train_one_crop(
    df: pd.DataFrame,
    cfg: dict,
    test_ratio: Optional[float] = None,
    validation_cutoff: Optional[str] = None,
    strict_cutoff_split: bool = True,
    verbose: bool = False,
    label: Optional[str] = None,
) -> CostModelResult:
    if str((cfg or {}).get("feature_set", "legacy")).strip().lower() == "panel_lite":
        raise ValueError("panel_lite feature_set requires train_panel_model")
    df = df.sort_values("year_start").reset_index(drop=True)
    if len(df) < 5:
        raise ValueError("not enough data for cost model")

    feature_meta = _cost_feature_meta(df, cfg=cfg)
    X = _make_cost_features(df, cfg=cfg, feature_meta=feature_meta)
    y = pd.to_numeric(df["india_cost_wavg_sample"], errors="coerce")
    valid = y.notna() & X.notna().all(axis=1)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    years = pd.to_numeric(df.loc[valid, "year_start"], errors="coerce").reset_index(drop=True)
    if len(X) < 5:
        raise ValueError("not enough valid rows for cost model")

    train_idx, test_idx, split_info = _split_train_test(
        years=years,
        n_rows=len(X),
        test_ratio=test_ratio,
        validation_cutoff=validation_cutoff,
        strict_cutoff_split=strict_cutoff_split,
    )
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    reg = str(cfg.get("regressor", "auto")).lower()
    if reg == "auto":
        reg = _select_auto_regressor(X_train, y_train, cfg)

    if reg == "ensemble":
        members = cfg.get("ensemble_members", ["ridge", "huber", "qgb", "gbr_huber", "hgb", "etr", "rf"])
        models = []
        weights = []
        member_rows = []
        for m in members:
            try:
                model, pred = _fit_and_pred(X_train, y_train, X_test, cfg, m)
                eval_row = _eval_metrics(y_test, pred)
                v_mae = _inner_val_mae(X_train, y_train, cfg, m)
                models.append(model)
                eval_mae = eval_row.get("mae")
                if v_mae is None:
                    weight_basis = 1.0 if eval_mae is None else eval_mae
                else:
                    weight_basis = v_mae
                weights.append(1.0 / max(weight_basis, 1e-6))
                member_rows.append({"name": m, "val_mae": v_mae, **eval_row})
            except Exception:
                continue
        if not models:
            raise ValueError("ensemble members all failed")
        weights = np.array(weights, dtype=float)
        weights = (weights / weights.sum()).tolist()
        huber_index = None
        for idx, row in enumerate(member_rows):
            if str(row.get("name", "")).strip().lower() == "huber":
                huber_index = idx
                break
        huber_blend_weight = max(0.0, min(1.0, float(cfg.get("ensemble_huber_blend_weight", 0.0))))
        model = EnsembleModel(
            models,
            weights,
            feature_cols=list(X.columns),
            feature_meta=feature_meta,
            huber_index=huber_index,
            huber_blend_weight=huber_blend_weight,
        )
        pred = model.predict(X_test) if not X_test.empty else np.asarray([], dtype=float)
        metrics = {
            **_eval_metrics(y_test, pred),
            "n_test": int(len(y_test)),
            "ensemble_members": member_rows,
            "ensemble_weights": weights,
        }
        if huber_index is not None and huber_blend_weight > 0.0:
            metrics["ensemble_huber_blend_weight"] = huber_blend_weight
            metrics["ensemble_huber_member_index"] = int(huber_index)
    else:
        model, pred = _fit_and_pred(X_train, y_train, X_test, cfg, reg)
        # Persist feature schema for serving-time robust inference.
        try:
            setattr(model, "feature_cols", list(X.columns))
            setattr(model, "cost_feature_meta", dict(feature_meta))
        except Exception:
            pass
        metrics = {
            **_eval_metrics(y_test, pred),
            "n_test": int(len(y_test)),
            "selected_model": reg,
        }

    metrics.update(split_info)

    if verbose:
        name = label or "crop"
        extra = metrics.get("selected_model", reg)
        mae_text = "NA" if metrics.get("mae") is None else f"{metrics['mae']:.4f}"
        print(f"[cost][{name}][{extra}] n_test={metrics['n_test']} mae={mae_text}")

    return CostModelResult(
        model=model,
        metrics=metrics,
        feature_cols=list(X.columns),
        feature_meta=dict(feature_meta),
    )


def train_panel_model(
    panel_df: pd.DataFrame,
    cfg: dict,
    test_ratio: Optional[float] = None,
    validation_cutoff: Optional[str] = None,
    strict_cutoff_split: bool = True,
    verbose: bool = False,
    label: Optional[str] = None,
) -> CostModelResult:
    if not isinstance(panel_df, pd.DataFrame) or panel_df.empty:
        raise ValueError("empty panel dataframe for cost panel model")

    X_raw, work, feature_meta = _make_panel_lite_features(panel_df, cfg=cfg)
    y = pd.to_numeric(work["cost"], errors="coerce")
    valid = y.notna()
    X_raw = X_raw.loc[valid].reset_index(drop=True)
    work = work.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    if len(X_raw) < 12:
        raise ValueError("not enough valid rows for cost panel model")

    order = np.lexsort((work["crop"].astype(str).to_numpy(), pd.to_numeric(work["year"], errors="coerce").to_numpy()))
    X_raw = X_raw.iloc[order].reset_index(drop=True)
    work = work.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    years = pd.to_numeric(work["year"], errors="coerce").reset_index(drop=True)

    train_idx, test_idx, split_info = _split_train_test(
        years=years,
        n_rows=len(X_raw),
        test_ratio=test_ratio,
        validation_cutoff=validation_cutoff,
        strict_cutoff_split=strict_cutoff_split,
    )

    X_train_raw = X_raw.iloc[train_idx].copy()
    X_test_raw = X_raw.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    fill_values = _derive_cost_fill_values(X_train_raw)
    X_train = _apply_cost_fill_values(X_train_raw, fill_values)
    X_test = _apply_cost_fill_values(X_test_raw, fill_values)

    reg = str(cfg.get("regressor", "auto")).lower()
    if reg == "auto":
        reg = _select_auto_regressor(X_train, y_train, cfg)

    if reg == "ensemble":
        members = cfg.get("ensemble_members", ["ridge", "huber", "qgb", "gbr_huber", "hgb", "etr", "rf"])
        models = []
        weights = []
        member_rows = []
        for member_name in members:
            try:
                model, pred = _fit_and_pred(X_train, y_train, X_test, cfg, member_name)
                eval_row = _eval_metrics(y_test, pred)
                v_mae = _inner_val_mae(X_train, y_train, cfg, member_name)
                models.append(model)
                eval_mae = eval_row.get("mae")
                weight_basis = v_mae if v_mae is not None else (1.0 if eval_mae is None else eval_mae)
                weights.append(1.0 / max(weight_basis, 1e-6))
                member_rows.append({"name": member_name, "val_mae": v_mae, **eval_row})
            except Exception:
                continue
        if not models:
            raise ValueError("panel-lite ensemble members all failed")
        weights_arr = np.asarray(weights, dtype=float)
        weights_arr = weights_arr / weights_arr.sum()
        weights = weights_arr.tolist()
        huber_index = None
        for idx, row in enumerate(member_rows):
            if str(row.get("name", "")).strip().lower() == "huber":
                huber_index = idx
                break
        huber_blend_weight = max(0.0, min(1.0, float(cfg.get("ensemble_huber_blend_weight", 0.0))))
        model = EnsembleModel(
            models,
            weights,
            feature_cols=list(X_train.columns),
            feature_meta=feature_meta,
            fill_values=fill_values,
            huber_index=huber_index,
            huber_blend_weight=huber_blend_weight,
        )
        pred = model.predict(X_test) if not X_test.empty else np.asarray([], dtype=float)
        pred = _apply_panel_lag1_blend(pred, X_test, feature_meta)
        metrics = {
            **_eval_metrics(y_test, pred),
            "n_test": int(len(y_test)),
            "selected_model": "ensemble",
            "ensemble_members": member_rows,
            "ensemble_weights": weights,
        }
        if huber_index is not None and huber_blend_weight > 0.0:
            metrics["ensemble_huber_blend_weight"] = huber_blend_weight
            metrics["ensemble_huber_member_index"] = int(huber_index)
    else:
        model, pred = _fit_and_pred(X_train, y_train, X_test, cfg, reg)
        pred = _apply_panel_lag1_blend(pred, X_test, feature_meta)
        try:
            setattr(model, "feature_cols", list(X_train.columns))
            setattr(model, "cost_feature_meta", dict(feature_meta))
            setattr(model, "cost_fill_values", dict(fill_values))
        except Exception:
            pass
        metrics = {
            **_eval_metrics(y_test, pred),
            "n_test": int(len(y_test)),
            "selected_model": reg,
        }

    metrics.update(split_info)
    metrics["shared_panel_model"] = True
    metrics["n_rows"] = int(len(X_raw))
    metrics["n_crops"] = int(work["crop"].nunique())

    if verbose:
        model_name = label or "panel_lite"
        mae_text = "NA" if metrics.get("mae") is None else f"{metrics['mae']:.4f}"
        print(f"[cost][{model_name}] n_test={metrics['n_test']} mae={mae_text}")

    return CostModelResult(
        model=model,
        metrics=metrics,
        feature_cols=list(X_train.columns),
        feature_meta=dict(feature_meta),
        fill_values=dict(fill_values),
    )


def _feature_cols_from_model(model) -> List[str]:
    cols = getattr(model, "feature_cols", None)
    if isinstance(cols, (list, tuple)) and cols:
        return [str(c) for c in cols]

    names = getattr(model, "feature_names_in_", None)
    if names is not None and len(names):
        return [str(c) for c in names]

    inner = getattr(model, "regressor_", None) or getattr(model, "regressor", None)
    if inner is not None:
        names = getattr(inner, "feature_names_in_", None)
        if names is not None and len(names):
            return [str(c) for c in names]

    members = getattr(model, "models", None)
    if isinstance(members, list):
        for member in members:
            cols = _feature_cols_from_model(member)
            if cols:
                return cols
    return []


def _feature_meta_from_model(model, feature_cols: List[str]) -> dict:
    meta = getattr(model, "cost_feature_meta", None)
    if isinstance(meta, dict) and meta:
        return dict(meta)

    if any(
        str(col).startswith(
            ("cost_", "price_", "yield_", "n_states_", "n_rows_cost_", "sum_sample_weight_", "crop_", "group_")
        )
        for col in feature_cols
    ):
        return {
            "feature_set": "panel_lite",
            "anchor_year": 1990,
            "lags": [1, 2, 3, 4],
            "windows": [2, 3, 5, 7],
            "include_crop_hierarchy": True,
            "include_price_lags": True,
            "include_yield_lags": True,
            "include_quality_lags": True,
        }

    feature_set = "extended" if any(x in feature_cols for x in ("year3", "log_year", "sqrt_year")) else "legacy"
    return {"feature_set": feature_set, "anchor": 1990.0, "year_min": 1990.0}


def _fill_values_from_model(model, feature_cols: List[str]) -> dict:
    fill_values = getattr(model, "cost_fill_values", None)
    if isinstance(fill_values, dict) and fill_values:
        return {str(k): float(v) for k, v in fill_values.items()}

    inner = getattr(model, "model", None) or getattr(model, "regressor_", None) or getattr(model, "regressor", None)
    if inner is not None:
        fill_values = getattr(inner, "cost_fill_values", None)
        if isinstance(fill_values, dict) and fill_values:
            return {str(k): float(v) for k, v in fill_values.items()}

    return {str(col): _default_cost_fill_value(str(col)) for col in feature_cols}


def make_panel_lite_serving_model(
    model: object,
    *,
    crop: str,
    cost_name: str = "",
    crop_group: Optional[str] = None,
) -> CostPanelServingModel:
    feature_cols = _feature_cols_from_model(model)
    feature_meta = _feature_meta_from_model(model, feature_cols)
    fill_values = _fill_values_from_model(model, feature_cols)
    return CostPanelServingModel(
        model=model,
        crop=crop,
        cost_name=cost_name,
        crop_group=str(crop_group or crop_group_from_cost_name(cost_name)),
        feature_cols=feature_cols,
        feature_meta=feature_meta,
        fill_values=fill_values,
    )


def predict_cost(
    model,
    year: int,
    cost_history_df: Optional[pd.DataFrame] = None,
    price_history_df: Optional[pd.DataFrame] = None,
    yield_history_df: Optional[pd.DataFrame] = None,
    crop: Optional[str] = None,
    cost_name: Optional[str] = None,
) -> float:
    feature_cols = _feature_cols_from_model(model)
    feature_meta = _feature_meta_from_model(model, feature_cols)
    fill_values = _fill_values_from_model(model, feature_cols)

    if str(feature_meta.get("feature_set", "")).strip().lower() == "panel_lite":
        serving_crop = str(crop or getattr(model, "serving_crop", "")).strip().lower()
        serving_cost_name = str(cost_name or getattr(model, "serving_cost_name", "")).strip()
        serving_crop_group = str(getattr(model, "serving_crop_group", "")).strip().lower()
        if not serving_crop_group:
            serving_crop_group = crop_group_from_cost_name(serving_cost_name)
        X = _panel_lite_prediction_row(
            year,
            feature_cols=feature_cols,
            feature_meta=feature_meta,
            fill_values=fill_values,
            cost_history_df=cost_history_df,
            price_history_df=price_history_df,
            yield_history_df=yield_history_df,
            crop=serving_crop,
            cost_name=serving_cost_name,
            crop_group=serving_crop_group,
        )
        pred = np.asarray(model.predict(X), dtype=float)
        pred = _apply_panel_lag1_blend(pred, X, feature_meta)
        return float(np.clip(pred[0], 0.0, None))

    base = pd.DataFrame({"year_start": [float(year)]})
    X = _make_cost_features(base, feature_meta=feature_meta)
    if feature_cols:
        X = X.reindex(columns=feature_cols, fill_value=0.0)
    elif "year" not in X.columns:
        # Conservative fallback for unknown legacy artifacts.
        X["year"] = float(year) - 1990.0
        X["year2"] = X["year"] ** 2

    pred = model.predict(X)[0]
    return float(np.clip(pred, 0.0, None))
