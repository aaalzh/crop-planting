from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


DEFAULT_VALIDATION_CUTOFF_DATE = "2020-12-31"
DEFAULT_VALIDATION_CUTOFF_YEAR = 2020
TARGET_COL = "yield_quintal_per_hectare"
EPSILON = 1e-6
PRIOR_COMPONENT_COLS = ["prior_p1", "prior_p2", "prior_p3", "prior_p4", "prior_p5"]
HISTORY_FEATURE_COLS = [
    "yield_lag1",
    "yield_lag2",
    "yield_lag3",
    "yield_delta1",
    "yield_delta2",
    "yield_growth1",
    "yield_ma2",
    "yield_ma3",
    "yield_med3",
    "yield_std3",
    "yield_range3",
    "yield_local_trend",
    "yield_long_median",
    "yield_rel_to_median",
    "yield_hist_count",
    "yield_gap_years",
    *PRIOR_COMPONENT_COLS,
]
LAMBDA_FEATURE_COLS = [
    "yield_hist_count",
    "yield_gap_years",
    "yield_std3",
    "model_member_std",
    "prior_component_std",
    "yield_abs_local_trend",
]
SEARCH_METRIC_CHOICES = {"mae", "rmse", "mape"}


@dataclass
class YieldTable:
    yields: Dict[str, float]
    missing: list


@dataclass
class YieldModelResult:
    model: object
    metrics: Dict[str, object]
    feature_cols: list


@dataclass
class YieldOOFResult:
    X: pd.DataFrame
    y: pd.Series
    years: List[int]
    member_preds: Dict[str, np.ndarray]
    member_metrics: List[Dict[str, object]]


class EnsembleModel:
    def __init__(self, models: List[object], weights: List[float], feature_cols: List[str]):
        self.models = models
        self.weights = weights
        self.feature_cols = feature_cols

    def predict_members(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for model in self.models:
            pred = np.asarray(model.predict(X), dtype=float)
            preds.append(np.clip(pred, 0.0, None))
        if not preds:
            return np.zeros((len(X), 0), dtype=float)
        return np.column_stack(preds)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        member_preds = self.predict_members(X)
        if member_preds.size == 0:
            return np.zeros(len(X), dtype=float)
        weights = np.asarray(self.weights, dtype=float).reshape(-1)
        out = member_preds @ weights
        return np.clip(out, 0.0, None)


class BlendedYieldModel:
    # Backward-compatible fixed-prior wrapper for already-trained artifacts.
    def __init__(
        self,
        model: object,
        crop_last: Dict[str, float],
        global_prior: float,
        blend_last_weight: float,
        feature_cols: List[str],
    ):
        self.model = model
        self.crop_last = crop_last
        self.global_prior = float(global_prior)
        self.blend_last_weight = float(blend_last_weight)
        self.feature_cols = feature_cols

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base_pred = np.clip(self.model.predict(X), 0.0, None)
        if self.blend_last_weight <= 0.0:
            return base_pred
        prior = _crop_prior_from_features(X, self.crop_last, self.global_prior)
        out = (1.0 - self.blend_last_weight) * base_pred + self.blend_last_weight * prior
        return np.clip(out, 0.0, None)


class AdaptiveLambdaModel:
    def __init__(
        self,
        model: Ridge,
        scaler: StandardScaler,
        feature_cols: List[str],
        fill_values: Dict[str, float],
    ):
        self.model = model
        self.scaler = scaler
        self.feature_cols = list(feature_cols)
        self.fill_values = {str(k): float(v) for k, v in fill_values.items()}

    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        work = pd.DataFrame(X).copy()
        for col in self.feature_cols:
            if col not in work.columns:
                work[col] = np.nan
        work = work[self.feature_cols].astype(float)
        for col, value in self.fill_values.items():
            work[col] = work[col].fillna(value)
        arr = work.to_numpy(dtype=float)
        return self.scaler.transform(arr)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        z = self._prepare(X)
        logits = np.asarray(self.model.predict(z), dtype=float)
        out = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        return np.clip(out, 0.0, 1.0)


class AdaptiveBlendYieldModel:
    def __init__(
        self,
        model: object,
        crop_last: Dict[str, float],
        global_prior: float,
        prior_weights: List[float],
        lambda_model: Optional[AdaptiveLambdaModel],
        default_lambda: float,
        feature_cols: List[str],
    ):
        self.model = model
        self.crop_last = crop_last
        self.global_prior = float(global_prior)
        self.prior_weights = [float(v) for v in prior_weights]
        self.lambda_model = lambda_model
        self.default_lambda = float(default_lambda)
        self.feature_cols = feature_cols

    def predict_details(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        member_preds = _predict_member_matrix(self.model, X)
        base_pred = _combine_member_predictions(member_preds, self.model)
        prior_pred = _blend_prior_from_features(
            X,
            prior_weights=self.prior_weights,
            crop_last=self.crop_last,
            global_prior=self.global_prior,
        )
        lambda_features = _build_lambda_feature_frame(X, member_preds)
        if self.lambda_model is not None:
            lambda_pred = self.lambda_model.predict(lambda_features)
        else:
            lambda_pred = np.full(len(X), self.default_lambda, dtype=float)
        final_pred = (1.0 - lambda_pred) * base_pred + lambda_pred * prior_pred
        return {
            "member_preds": member_preds,
            "base_pred": np.clip(base_pred, 0.0, None),
            "prior_pred": np.clip(prior_pred, 0.0, None),
            "lambda_pred": np.clip(lambda_pred, 0.0, 1.0),
            "final_pred": np.clip(final_pred, 0.0, None),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        details = self.predict_details(X)
        return details["final_pred"]


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


def _normalize_crop_token(value: object) -> str:
    return str(value or "").strip().lower()


def _coerce_optional_int(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_max_features(value: object):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"none", "null"}:
            return None
        if text in {"sqrt", "log2"}:
            return text
        try:
            numeric = float(text)
        except Exception:
            return value
        return numeric
    try:
        return float(value)
    except Exception:
        return value


def _value_equals(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, str) or isinstance(right, str):
        return str(left) == str(right)
    try:
        return abs(float(left) - float(right)) <= 1e-12
    except Exception:
        return left == right


def _unique_preserve(values: List[Any]) -> List[Any]:
    out: List[Any] = []
    for value in values:
        if any(_value_equals(value, existing) for existing in out):
            continue
        out.append(value)
    return out


def _resolve_member_params(cfg: dict, name: str) -> Dict[str, Any]:
    key = str(name or "").strip().lower()
    if key == "rf":
        return {
            "n_estimators": int(cfg.get("rf_n_estimators", 500)),
            "min_samples_leaf": int(cfg.get("rf_min_samples_leaf", 2)),
            "max_depth": _coerce_optional_int(cfg.get("rf_max_depth", None)),
            "max_features": _coerce_max_features(cfg.get("rf_max_features", 1.0)),
        }
    if key == "etr":
        return {
            "n_estimators": int(cfg.get("etr_n_estimators", 700)),
            "min_samples_leaf": int(cfg.get("etr_min_samples_leaf", 2)),
            "max_depth": _coerce_optional_int(cfg.get("etr_max_depth", None)),
            "max_features": _coerce_max_features(cfg.get("etr_max_features", 1.0)),
        }
    return {
        "max_iter": int(cfg.get("max_iter", 700)),
        "max_depth": _coerce_optional_int(cfg.get("max_depth", 8)),
        "learning_rate": float(cfg.get("learning_rate", 0.03)),
        "l2_regularization": float(cfg.get("l2_regularization", 1e-3)),
        "min_samples_leaf": int(cfg.get("hgb_min_samples_leaf", 20)),
    }


def _resolve_yield_search_cfg(cfg: dict, random_state: int = 42) -> Dict[str, Any]:
    raw = cfg.get("search", {})
    search_cfg = raw if isinstance(raw, dict) else {}
    metric = str(search_cfg.get("metric", cfg.get("ensemble_weight_metric", "rmse"))).strip().lower()
    if metric not in SEARCH_METRIC_CHOICES:
        metric = "rmse"
    hgb_baseline = _resolve_member_params(cfg, "hgb")
    rf_baseline = _resolve_member_params(cfg, "rf")
    etr_baseline = _resolve_member_params(cfg, "etr")
    return {
        "enabled": bool(search_cfg.get("enabled", False)),
        "trials": max(1, int(search_cfg.get("trials", 10))),
        "seed": int(search_cfg.get("seed", random_state + 211)),
        "metric": metric,
        "hgb_max_iter_choices": _unique_preserve(
            [int(x) for x in search_cfg.get("hgb_max_iter_choices", [200, 300, 450, 700, 900, 1200, hgb_baseline["max_iter"]])]
        ),
        "hgb_max_depth_choices": _unique_preserve(
            [_coerce_optional_int(x) for x in search_cfg.get("hgb_max_depth_choices", [3, 4, 5, 6, 8, None, hgb_baseline["max_depth"]])]
        ),
        "hgb_learning_rate_choices": _unique_preserve(
            [float(x) for x in search_cfg.get("hgb_learning_rate_choices", [0.015, 0.02, 0.03, 0.05, 0.08, 0.12, hgb_baseline["learning_rate"]])]
        ),
        "hgb_l2_regularization_choices": _unique_preserve(
            [float(x) for x in search_cfg.get("hgb_l2_regularization_choices", [0.0, 0.001, 0.01, 0.03, 0.1, hgb_baseline["l2_regularization"]])]
        ),
        "hgb_min_samples_leaf_choices": _unique_preserve(
            [max(1, int(x)) for x in search_cfg.get("hgb_min_samples_leaf_choices", [7, 12, 20, 30, 40, hgb_baseline["min_samples_leaf"]])]
        ),
        "rf_n_estimators_choices": _unique_preserve(
            [int(x) for x in search_cfg.get("rf_n_estimators_choices", [200, 300, 400, 500, 700, 900, rf_baseline["n_estimators"]])]
        ),
        "rf_min_samples_leaf_choices": _unique_preserve(
            [max(1, int(x)) for x in search_cfg.get("rf_min_samples_leaf_choices", [1, 2, 3, 4, 5, rf_baseline["min_samples_leaf"]])]
        ),
        "rf_max_depth_choices": _unique_preserve(
            [_coerce_optional_int(x) for x in search_cfg.get("rf_max_depth_choices", [None, 4, 6, 8, 10, 12, rf_baseline["max_depth"]])]
        ),
        "rf_max_features_choices": _unique_preserve(
            [_coerce_max_features(x) for x in search_cfg.get("rf_max_features_choices", [0.6, 0.8, 1.0, "sqrt", "log2", rf_baseline["max_features"]])]
        ),
        "etr_n_estimators_choices": _unique_preserve(
            [int(x) for x in search_cfg.get("etr_n_estimators_choices", [200, 300, 500, 700, 900, etr_baseline["n_estimators"]])]
        ),
        "etr_min_samples_leaf_choices": _unique_preserve(
            [max(1, int(x)) for x in search_cfg.get("etr_min_samples_leaf_choices", [1, 2, 3, 4, 5, etr_baseline["min_samples_leaf"]])]
        ),
        "etr_max_depth_choices": _unique_preserve(
            [_coerce_optional_int(x) for x in search_cfg.get("etr_max_depth_choices", [None, 4, 6, 8, 10, 12, etr_baseline["max_depth"]])]
        ),
        "etr_max_features_choices": _unique_preserve(
            [_coerce_max_features(x) for x in search_cfg.get("etr_max_features_choices", [0.6, 0.8, 1.0, "sqrt", "log2", etr_baseline["max_features"]])]
        ),
    }


def _member_param_key(name: str, params: Dict[str, Any]) -> Tuple[Any, ...]:
    key = str(name or "").strip().lower()
    if key == "rf":
        return ("rf", params.get("n_estimators"), params.get("min_samples_leaf"), params.get("max_depth"), params.get("max_features"))
    if key == "etr":
        return ("etr", params.get("n_estimators"), params.get("min_samples_leaf"), params.get("max_depth"), params.get("max_features"))
    return (
        "hgb",
        params.get("max_iter"),
        params.get("max_depth"),
        params.get("learning_rate"),
        params.get("l2_regularization"),
        params.get("min_samples_leaf"),
    )


def _sample_member_search_params(
    rng: np.random.RandomState,
    model_name: str,
    baseline_params: Dict[str, Any],
    search_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    name = str(model_name or "").strip().lower()
    if name == "rf":
        return {
            "n_estimators": int(rng.choice(search_cfg["rf_n_estimators_choices"])),
            "min_samples_leaf": int(rng.choice(search_cfg["rf_min_samples_leaf_choices"])),
            "max_depth": search_cfg["rf_max_depth_choices"][int(rng.randint(len(search_cfg["rf_max_depth_choices"])))],
            "max_features": search_cfg["rf_max_features_choices"][int(rng.randint(len(search_cfg["rf_max_features_choices"])))],
        }
    if name == "etr":
        return {
            "n_estimators": int(rng.choice(search_cfg["etr_n_estimators_choices"])),
            "min_samples_leaf": int(rng.choice(search_cfg["etr_min_samples_leaf_choices"])),
            "max_depth": search_cfg["etr_max_depth_choices"][int(rng.randint(len(search_cfg["etr_max_depth_choices"])))],
            "max_features": search_cfg["etr_max_features_choices"][int(rng.randint(len(search_cfg["etr_max_features_choices"])))],
        }
    return {
        "max_iter": int(rng.choice(search_cfg["hgb_max_iter_choices"])),
        "max_depth": search_cfg["hgb_max_depth_choices"][int(rng.randint(len(search_cfg["hgb_max_depth_choices"])))],
        "learning_rate": float(rng.choice(search_cfg["hgb_learning_rate_choices"])),
        "l2_regularization": float(rng.choice(search_cfg["hgb_l2_regularization_choices"])),
        "min_samples_leaf": int(rng.choice(search_cfg["hgb_min_samples_leaf_choices"])),
    }


def load_yield_table(df: pd.DataFrame) -> YieldTable:
    yields = {}
    missing = []
    for _, row in df.iterrows():
        crop = _normalize_crop_token(row.get("crop_name", ""))
        if not crop or crop.startswith("#"):
            continue
        val = row.get(TARGET_COL, None)
        try:
            val = float(val)
        except Exception:
            val = None
        if val is None:
            missing.append(crop)
            continue
        yields[crop] = val
    return YieldTable(yields=yields, missing=missing)


def _build_regressor(cfg: dict, name: str, params_override: Optional[Dict[str, Any]] = None):
    n_jobs = int(cfg.get("n_jobs", 1))
    params = dict(_resolve_member_params(cfg, name))
    if params_override:
        params.update(params_override)
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 500)),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            max_depth=_coerce_optional_int(params.get("max_depth")),
            max_features=_coerce_max_features(params.get("max_features", 1.0)),
            random_state=42,
            n_jobs=n_jobs,
        )
    if name == "etr":
        return ExtraTreesRegressor(
            n_estimators=int(params.get("n_estimators", 700)),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            max_depth=_coerce_optional_int(params.get("max_depth")),
            max_features=_coerce_max_features(params.get("max_features", 1.0)),
            random_state=42,
            n_jobs=n_jobs,
        )
    return HistGradientBoostingRegressor(
        max_iter=int(params.get("max_iter", 700)),
        max_depth=_coerce_optional_int(params.get("max_depth", 8)),
        learning_rate=float(params.get("learning_rate", 0.03)),
        l2_regularization=float(params.get("l2_regularization", 1e-3)),
        min_samples_leaf=int(params.get("min_samples_leaf", 20)),
        random_state=42,
    )


def _wrap_target_transform(model, cfg: dict):
    transform = str(cfg.get("target_transform", "log1p")).lower()
    if transform == "log1p":
        return TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    return model


def _prepare_history_source(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["crop_name", "year", target_col])
    work = df.copy()
    if "crop_name" not in work.columns or "year" not in work.columns:
        return pd.DataFrame(columns=["crop_name", "year", target_col])
    work["crop_name"] = work["crop_name"].map(_normalize_crop_token)
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    if target_col in work.columns:
        work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    else:
        work[target_col] = np.nan
    work = work.dropna(subset=["crop_name", "year", target_col])
    if work.empty:
        return pd.DataFrame(columns=["crop_name", "year", target_col])
    work["year"] = work["year"].astype(int)
    work = work.sort_values(["crop_name", "year"]).reset_index(drop=True)
    return work


def _build_history_lookup(history_df: Optional[pd.DataFrame], target_col: str = TARGET_COL) -> Tuple[Dict[str, Dict[str, np.ndarray]], float]:
    hist = _prepare_history_source(history_df if history_df is not None else pd.DataFrame(), target_col=target_col)
    lookup: Dict[str, Dict[str, np.ndarray]] = {}
    global_prior = 0.0
    if hist.empty:
        return lookup, global_prior
    global_prior = float(hist[target_col].median())
    for crop, grp in hist.groupby("crop_name", sort=False):
        lookup[str(crop)] = {
            "years": grp["year"].to_numpy(dtype=int),
            "values": grp[target_col].to_numpy(dtype=float),
        }
    return lookup, global_prior


def _local_trend(years: np.ndarray, values: np.ndarray, window: int) -> float:
    if len(values) < 2:
        return np.nan
    n = min(int(window), len(values))
    x = years[-n:].astype(float)
    y = values[-n:].astype(float)
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered))
    if denom <= EPSILON:
        return np.nan
    return float(np.dot(x_centered, y - y.mean()) / denom)


def _build_history_feature_row(
    crop: str,
    year: Optional[int],
    lookup: Dict[str, Dict[str, np.ndarray]],
    cfg: Optional[dict] = None,
) -> Dict[str, float]:
    cfg = cfg or {}
    trend_window = max(2, int(cfg.get("local_trend_window", 3)))
    features = {name: np.nan for name in HISTORY_FEATURE_COLS}
    if not crop or year is None:
        return features
    item = lookup.get(crop)
    if not item:
        features["yield_hist_count"] = 0.0
        return features
    years = item["years"]
    values = item["values"]
    cutoff = int(np.searchsorted(years, int(year), side="left"))
    past_years = years[:cutoff]
    past_values = values[:cutoff]
    n_hist = len(past_values)
    features["yield_hist_count"] = float(n_hist)
    if n_hist <= 0:
        return features

    lag1 = float(past_values[-1])
    lag2 = float(past_values[-2]) if n_hist >= 2 else np.nan
    lag3 = float(past_values[-3]) if n_hist >= 3 else np.nan
    features["yield_lag1"] = lag1
    features["yield_lag2"] = lag2
    features["yield_lag3"] = lag3
    features["yield_gap_years"] = float(int(year) - int(past_years[-1]))
    features["yield_long_median"] = float(np.median(past_values))
    long_scale = max(abs(features["yield_long_median"]), EPSILON)
    features["yield_rel_to_median"] = lag1 / long_scale
    if n_hist >= 2:
        features["yield_delta1"] = lag1 - lag2
        features["yield_growth1"] = (lag1 - lag2) / max(abs(lag2), EPSILON)
        features["yield_ma2"] = float(np.mean(past_values[-2:]))
    if n_hist >= 3:
        tail3 = past_values[-3:]
        features["yield_delta2"] = lag1 - lag3
        features["yield_ma3"] = float(np.mean(tail3))
        features["yield_med3"] = float(np.median(tail3))
        features["yield_std3"] = float(np.std(tail3, ddof=0))
        features["yield_range3"] = float(np.max(tail3) - np.min(tail3))
    features["yield_local_trend"] = _local_trend(past_years, past_values, trend_window)

    if n_hist >= 1:
        features["prior_p1"] = lag1
        features["prior_p5"] = features["yield_long_median"]
    if n_hist >= 2:
        features["prior_p2"] = float(np.mean(past_values[-2:]))
        if np.isfinite(features["yield_local_trend"]):
            features["prior_p4"] = lag1 + features["yield_local_trend"]
    if n_hist >= 3:
        features["prior_p3"] = float(np.median(past_values[-3:]))
    return features


def _attach_history_features(
    df: pd.DataFrame,
    history_df: Optional[pd.DataFrame],
    cfg: Optional[dict] = None,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    work = df.copy().reset_index(drop=True)
    if "crop_name" not in work.columns or "year" not in work.columns:
        for col in HISTORY_FEATURE_COLS:
            if col not in work.columns:
                work[col] = np.nan
        return work
    lookup, _ = _build_history_lookup(history_df, target_col=target_col)
    rows = []
    for _, row in work.iterrows():
        crop = _normalize_crop_token(row.get("crop_name", ""))
        year = _safe_year_int(row.get("year"))
        rows.append(_build_history_feature_row(crop, year, lookup, cfg=cfg))
    hist_feat = pd.DataFrame(rows)
    return pd.concat([work, hist_feat], axis=1)


def _base_feature_frame(df: pd.DataFrame, cfg: Optional[dict] = None, history_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    cfg = cfg or {}
    out = df.copy().reset_index(drop=True)
    if "crop_name" in out.columns:
        out["crop_name"] = out["crop_name"].map(_normalize_crop_token)
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce")

    if bool(cfg.get("enable_history_features", True)) and "crop_name" in out.columns and "year" in out.columns:
        source = history_df if history_df is not None else df
        out = out.drop(columns=[col for col in HISTORY_FEATURE_COLS if col in out.columns], errors="ignore")
        out = _attach_history_features(out, source, cfg=cfg, target_col=TARGET_COL)

    if "crop_name" in out.columns:
        out = pd.get_dummies(out, columns=["crop_name"], prefix="crop")

    if "year" in out.columns:
        poly_degree = int(cfg.get("year_poly_degree", 3))
        if poly_degree >= 2:
            for deg in range(2, poly_degree + 1):
                out[f"year{deg}"] = out["year"] ** deg
        if bool(cfg.get("use_crop_year_interaction", True)):
            crop_cols = [col for col in out.columns if col.startswith("crop_")]
            for col in crop_cols:
                out[f"{col}_x_year"] = out[col] * out["year"]

    out = out.select_dtypes(include=["number", "bool"]).copy()
    out = out.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _prepare_features(
    df: pd.DataFrame,
    target_col: str,
    cfg: Optional[dict] = None,
    history_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    work = df.copy().reset_index(drop=True)
    if target_col not in work.columns:
        raise ValueError(f"missing target column: {target_col}")
    y = pd.to_numeric(work[target_col], errors="coerce")
    feat_df = work.drop(columns=[target_col])
    X = _base_feature_frame(feat_df, cfg=cfg, history_df=history_df)
    valid = y.notna() & X.notna().all(axis=1)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    return X, y, list(X.columns)


def make_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    cfg: Optional[dict] = None,
    history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    X = _base_feature_frame(df, cfg=cfg, history_df=history_df)
    X = X.reindex(columns=feature_cols, fill_value=0.0)
    return X


def _split_train_test(
    df: pd.DataFrame,
    test_ratio: Optional[float] = None,
    validation_cutoff: Optional[str] = None,
    strict_cutoff_split: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    ratio = 0.2 if test_ratio is None else float(test_ratio)
    cutoff_year = _resolve_cutoff_year(validation_cutoff)
    if "year" in df.columns:
        work = df.copy()
        work["year"] = pd.to_numeric(work["year"], errors="coerce")
        work = work.dropna(subset=["year"]).sort_values("year")

        train_df = work[work["year"] <= cutoff_year].copy()
        test_df = work[work["year"] > cutoff_year].copy()
        if not train_df.empty and not test_df.empty:
            split_info = {
                "split_mode": "fixed_cutoff",
                "split_cutoff_year": int(cutoff_year),
                "split_used_post_cutoff_validation": True,
                "split_post_cutoff_rows": int(len(test_df)),
                "split_train_rows": int(len(train_df)),
                "split_test_rows": int(len(test_df)),
                "split_train_start_year": _safe_year_int(train_df["year"].min()),
                "split_train_end_year": _safe_year_int(train_df["year"].max()),
                "split_test_start_year": _safe_year_int(test_df["year"].min()),
                "split_test_end_year": _safe_year_int(test_df["year"].max()),
            }
            return train_df, test_df, split_info
        if not train_df.empty and test_df.empty and strict_cutoff_split:
            split_info = {
                "split_mode": "fixed_cutoff_no_validation",
                "split_cutoff_year": int(cutoff_year),
                "split_used_post_cutoff_validation": False,
                "split_post_cutoff_rows": 0,
                "split_train_rows": int(len(train_df)),
                "split_test_rows": 0,
                "split_train_start_year": _safe_year_int(train_df["year"].min()),
                "split_train_end_year": _safe_year_int(train_df["year"].max()),
                "split_test_start_year": None,
                "split_test_end_year": None,
            }
            return train_df, test_df, split_info

        uniq_years = sorted(work["year"].astype(int).unique().tolist())
        if len(uniq_years) >= 8:
            n_test_years = max(3, int(len(uniq_years) * ratio))
            test_years = set(uniq_years[-n_test_years:])
            train_df = work[~work["year"].astype(int).isin(test_years)].copy()
            test_df = work[work["year"].astype(int).isin(test_years)].copy()
            if len(train_df) >= 30 and len(test_df) >= 10:
                split_info = {
                    "split_mode": "year_tail_fallback",
                    "split_cutoff_year": int(cutoff_year),
                    "split_used_post_cutoff_validation": False,
                    "split_post_cutoff_rows": int(len(work[work["year"] > cutoff_year])),
                    "split_train_rows": int(len(train_df)),
                    "split_test_rows": int(len(test_df)),
                    "split_train_start_year": _safe_year_int(train_df["year"].min()),
                    "split_train_end_year": _safe_year_int(train_df["year"].max()),
                    "split_test_start_year": _safe_year_int(test_df["year"].min()),
                    "split_test_end_year": _safe_year_int(test_df["year"].max()),
                }
                return train_df, test_df, split_info
        df = work

    df = df.sort_values("year") if "year" in df.columns else df.copy()
    n_test = max(5, int(len(df) * ratio))
    train_df = df.iloc[:-n_test].copy()
    test_df = df.iloc[-n_test:].copy()
    split_info = {
        "split_mode": "ratio_fallback",
        "split_cutoff_year": int(cutoff_year),
        "split_used_post_cutoff_validation": False,
        "split_post_cutoff_rows": int(len(df[df["year"] > cutoff_year])) if "year" in df.columns else 0,
        "split_train_rows": int(len(train_df)),
        "split_test_rows": int(len(test_df)),
        "split_train_start_year": _safe_year_int(train_df["year"].min()) if "year" in train_df.columns else None,
        "split_train_end_year": _safe_year_int(train_df["year"].max()) if "year" in train_df.columns else None,
        "split_test_start_year": _safe_year_int(test_df["year"].min()) if "year" in test_df.columns else None,
        "split_test_end_year": _safe_year_int(test_df["year"].max()) if "year" in test_df.columns else None,
    }
    return train_df, test_df, split_info


def _sample_weight_for_relative_error(y: pd.Series, cfg: dict):
    if not cfg.get("use_relative_weight", True):
        return None
    floor = float(cfg.get("relative_error_floor", 5.0))
    w = 1.0 / np.maximum(np.abs(y.values.astype(float)), floor)
    p95 = float(np.quantile(w, 0.95))
    w = np.clip(w, None, p95)
    return w / np.mean(w)


def _eval_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        return {"mae": None, "rmse": None, "mape": None}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), EPSILON))))
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def _build_crop_last_prior(train_df: pd.DataFrame, target_col: str) -> Tuple[Dict[str, float], float]:
    if "crop_name" not in train_df.columns or target_col not in train_df.columns:
        return {}, 0.0
    work = train_df.copy()
    work["crop_name"] = work["crop_name"].map(_normalize_crop_token)
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=["crop_name", target_col])
    if work.empty:
        return {}, 0.0
    if "year" in work.columns:
        work["year"] = pd.to_numeric(work["year"], errors="coerce")
        work = work.sort_values(["crop_name", "year"])
    else:
        work = work.sort_values(["crop_name"])
    crop_last = work.groupby("crop_name")[target_col].last().to_dict()
    global_prior = float(work[target_col].median())
    return crop_last, global_prior


def _crop_prior_from_features(X: pd.DataFrame, crop_last: Dict[str, float], global_prior: float) -> np.ndarray:
    prior = np.full(len(X), float(global_prior), dtype=float)
    if not isinstance(X, pd.DataFrame) or not crop_last:
        return prior
    for col in X.columns:
        if not col.startswith("crop_") or col.endswith("_x_year"):
            continue
        crop = col[len("crop_") :]
        val = crop_last.get(crop)
        if val is None:
            continue
        mask = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.5
        if mask.any():
            prior[mask] = float(val)
    return prior


def _fit_regressor_model(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    model_name: str,
    params_override: Optional[Dict[str, Any]] = None,
):
    model = _wrap_target_transform(_build_regressor(cfg, model_name, params_override=params_override), cfg)
    sample_weight = _sample_weight_for_relative_error(y, cfg)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    try:
        model.fit(X, y, **fit_kwargs)
    except TypeError:
        model.fit(X, y)
    return model


def _train_one(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: dict,
    model_name: str,
    params_override: Optional[Dict[str, Any]] = None,
):
    model = _fit_regressor_model(X_train, y_train, cfg, model_name, params_override=params_override)
    if X_test.empty or len(y_test) == 0:
        pred = np.asarray([], dtype=float)
        metrics = {"mae": None, "rmse": None, "mape": None, "n_test": 0}
    else:
        pred = np.clip(np.asarray(model.predict(X_test), dtype=float), 0.0, None)
        metrics = _eval_metrics(y_test, pred)
        metrics["n_test"] = int(len(y_test))
    return model, pred, metrics


def _predict_member_matrix(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_members"):
        mat = np.asarray(model.predict_members(X), dtype=float)
    else:
        pred = np.asarray(model.predict(X), dtype=float)
        mat = pred.reshape(-1, 1)
    return np.clip(mat, 0.0, None)


def _combine_member_predictions(member_preds: np.ndarray, model: object) -> np.ndarray:
    if member_preds.size == 0:
        return np.zeros(member_preds.shape[0], dtype=float)
    if isinstance(model, EnsembleModel):
        weights = np.asarray(model.weights, dtype=float).reshape(-1)
        return np.clip(member_preds @ weights, 0.0, None)
    return np.clip(member_preds[:, 0], 0.0, None)


def _prior_availability_matrix(X: pd.DataFrame) -> np.ndarray:
    n_hist = pd.to_numeric(X.get("yield_hist_count", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.column_stack(
        [
            n_hist >= 1.0,
            n_hist >= 2.0,
            n_hist >= 3.0,
            n_hist >= 2.0,
            n_hist >= 1.0,
        ]
    ).astype(float)


def _prior_component_matrix(X: pd.DataFrame) -> np.ndarray:
    cols = []
    for col in PRIOR_COMPONENT_COLS:
        cols.append(pd.to_numeric(X.get(col, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float))
    return np.column_stack(cols)


def _masked_row_std(values: np.ndarray, availability: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    for i in range(len(values)):
        valid = availability[i] > 0.5
        if not valid.any():
            out[i] = 0.0
            continue
        out[i] = float(np.std(values[i, valid], ddof=0))
    return out


def _blend_prior_from_features(
    X: pd.DataFrame,
    prior_weights: List[float],
    crop_last: Dict[str, float],
    global_prior: float,
) -> np.ndarray:
    component_matrix = _prior_component_matrix(X)
    availability = _prior_availability_matrix(X)
    fallback = _crop_prior_from_features(X, crop_last, global_prior)
    weights = np.asarray(prior_weights, dtype=float).reshape(1, -1)
    weighted_availability = availability * weights
    denom = weighted_availability.sum(axis=1)
    numer = (component_matrix * weighted_availability).sum(axis=1)
    prior = np.array(fallback, dtype=float)
    valid = denom > EPSILON
    prior[valid] = numer[valid] / denom[valid]
    return np.clip(prior, 0.0, None)


def _loss_per_candidate(y_true: np.ndarray, pred_matrix: np.ndarray, metric: str) -> np.ndarray:
    diff = pred_matrix - y_true.reshape(1, -1)
    metric_name = str(metric or "mae").lower()
    if metric_name == "rmse":
        return np.sqrt(np.mean(diff ** 2, axis=1))
    if metric_name == "mape":
        denom = np.maximum(np.abs(y_true.reshape(1, -1)), EPSILON)
        return np.mean(np.abs(diff) / denom, axis=1)
    return np.mean(np.abs(diff), axis=1)


def _simplex_integer_grid(dim: int, total: int):
    if dim == 1:
        yield [total]
        return
    for value in range(total + 1):
        for rest in _simplex_integer_grid(dim - 1, total - value):
            yield [value] + rest


def _search_simplex_weights(
    component_matrix: np.ndarray,
    y_true: np.ndarray,
    metric: str = "mae",
    grid_steps: int = 20,
    availability: Optional[np.ndarray] = None,
    fallback: Optional[np.ndarray] = None,
) -> Tuple[List[float], float]:
    comp = np.asarray(component_matrix, dtype=float)
    y = np.asarray(y_true, dtype=float).reshape(-1)
    if comp.ndim != 2 or comp.shape[0] == 0 or comp.shape[1] == 0:
        return [], np.nan
    if availability is None:
        availability = np.isfinite(comp).astype(float)
    avail = np.asarray(availability, dtype=float)
    comp = np.where(np.isfinite(comp), comp, 0.0)
    if fallback is None:
        fallback_arr = np.full(len(y), float(np.median(y)) if len(y) else 0.0, dtype=float)
    else:
        fallback_arr = np.asarray(fallback, dtype=float).reshape(-1)
        if fallback_arr.size == 1:
            fallback_arr = np.full(len(y), float(fallback_arr[0]), dtype=float)
    grid_steps = max(1, int(grid_steps))
    combos = list(_simplex_integer_grid(comp.shape[1], grid_steps))
    if not combos:
        weights = np.full(comp.shape[1], 1.0 / comp.shape[1], dtype=float)
        return weights.tolist(), np.nan

    best_loss = np.inf
    best_weights = np.full(comp.shape[1], 1.0 / comp.shape[1], dtype=float)
    batch_size = 512
    for start in range(0, len(combos), batch_size):
        batch = np.asarray(combos[start : start + batch_size], dtype=float) / float(grid_steps)
        weighted = batch[:, None, :] * avail[None, :, :]
        denom = weighted.sum(axis=2)
        numer = (weighted * comp[None, :, :]).sum(axis=2)
        pred = np.broadcast_to(fallback_arr, (len(batch), len(y))).copy()
        valid = denom > EPSILON
        pred[valid] = numer[valid] / denom[valid]
        losses = _loss_per_candidate(y, pred, metric)
        idx = int(np.argmin(losses))
        loss = float(losses[idx])
        if loss < best_loss:
            best_loss = loss
            best_weights = batch[idx]
    return best_weights.tolist(), float(best_loss)


def _build_lambda_feature_frame(X: pd.DataFrame, member_preds: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    out["yield_hist_count"] = pd.to_numeric(X.get("yield_hist_count", 0.0), errors="coerce")
    out["yield_gap_years"] = pd.to_numeric(X.get("yield_gap_years", 0.0), errors="coerce")
    out["yield_std3"] = pd.to_numeric(X.get("yield_std3", 0.0), errors="coerce")
    out["yield_abs_local_trend"] = pd.to_numeric(X.get("yield_local_trend", 0.0), errors="coerce").abs()
    if member_preds.ndim == 1:
        member_preds = member_preds.reshape(-1, 1)
    out["model_member_std"] = np.std(member_preds, axis=1) if member_preds.size else 0.0
    prior_matrix = _prior_component_matrix(X)
    availability = _prior_availability_matrix(X)
    out["prior_component_std"] = _masked_row_std(prior_matrix, availability)
    return out


def _fit_lambda_model(
    lambda_features: pd.DataFrame,
    lambda_target: np.ndarray,
    cfg: dict,
) -> Tuple[Optional[AdaptiveLambdaModel], Dict[str, object]]:
    target = np.asarray(lambda_target, dtype=float)
    spread_floor = float(cfg.get("lambda_target_gap_floor", 1e-3))
    valid = np.isfinite(target)
    if valid.sum() < max(20, int(cfg.get("lambda_min_train_rows", 60))):
        return None, {"kind": "constant_fallback", "feature_cols": LAMBDA_FEATURE_COLS}

    work = lambda_features.loc[valid, LAMBDA_FEATURE_COLS].copy()
    target = target[valid]
    fill_values = {}
    for col in LAMBDA_FEATURE_COLS:
        med = work[col].median()
        fill_values[col] = 0.0 if pd.isna(med) else float(med)
        work[col] = work[col].fillna(fill_values[col])

    clipped = np.clip(target, spread_floor, 1.0 - spread_floor)
    logits = np.log(clipped / np.maximum(1.0 - clipped, spread_floor))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(work.to_numpy(dtype=float))
    model = Ridge(alpha=float(cfg.get("lambda_ridge_alpha", 1.0)))
    model.fit(X_scaled, logits)
    adaptive = AdaptiveLambdaModel(
        model=model,
        scaler=scaler,
        feature_cols=LAMBDA_FEATURE_COLS,
        fill_values=fill_values,
    )
    coef = {name: float(val) for name, val in zip(LAMBDA_FEATURE_COLS, np.asarray(model.coef_, dtype=float))}
    meta = {
        "kind": "ridge_sigmoid",
        "feature_cols": LAMBDA_FEATURE_COLS,
        "intercept": float(np.asarray(model.intercept_).reshape(-1)[0]),
        "coefficients": coef,
        "train_rows": int(valid.sum()),
    }
    return adaptive, meta


def _learn_default_lambda(y_true: np.ndarray, base_pred: np.ndarray, prior_pred: np.ndarray, cfg: dict) -> float:
    if len(y_true) == 0:
        return float(cfg.get("blend_last_weight", 0.0))
    lambdas = np.linspace(0.0, 1.0, int(cfg.get("lambda_grid_size", 51)))
    metric = str(cfg.get("lambda_constant_metric", "mae")).lower()
    best_lambda = float(cfg.get("blend_last_weight", 0.0))
    best_loss = np.inf
    y = np.asarray(y_true, dtype=float)
    base = np.asarray(base_pred, dtype=float)
    prior = np.asarray(prior_pred, dtype=float)
    for value in lambdas:
        pred = (1.0 - value) * base + value * prior
        loss = float(_loss_per_candidate(y, pred.reshape(1, -1), metric)[0])
        if loss < best_loss:
            best_loss = loss
            best_lambda = float(value)
    return float(np.clip(best_lambda, 0.0, 1.0))


def _build_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    row_df: pd.DataFrame,
    cfg: dict,
    members: List[str],
    member_param_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> YieldOOFResult:
    if X.empty or y.empty:
        return YieldOOFResult(
            X=X.iloc[0:0].copy(),
            y=y.iloc[0:0].copy(),
            years=[],
            member_preds={name: np.asarray([], dtype=float) for name in members},
            member_metrics=[],
        )

    years = pd.to_numeric(row_df.get("year"), errors="coerce").reset_index(drop=True)
    unique_years = sorted(int(v) for v in years.dropna().unique().tolist())
    min_train_years = max(3, int(cfg.get("oof_min_train_years", 5)))
    min_train_rows = max(24, int(cfg.get("oof_min_train_rows", 60)))
    max_validation_years = int(cfg.get("oof_max_validation_years", 8))
    pred_store = {name: np.full(len(X), np.nan, dtype=float) for name in members}
    used_years: List[int] = []

    candidate_years = unique_years[min_train_years:]
    if max_validation_years > 0 and len(candidate_years) > max_validation_years:
        candidate_years = candidate_years[-max_validation_years:]

    for val_year in candidate_years:
        train_mask = (years < val_year).to_numpy(dtype=bool)
        val_mask = (years == val_year).to_numpy(dtype=bool)
        if train_mask.sum() < min_train_rows or val_mask.sum() == 0:
            continue
        used_years.append(int(val_year))
        X_fit = X.loc[train_mask].reset_index(drop=True)
        y_fit = y.loc[train_mask].reset_index(drop=True)
        X_val = X.loc[val_mask].reset_index(drop=True)
        for name in members:
            params_override = None if member_param_overrides is None else member_param_overrides.get(name)
            model = _fit_regressor_model(X_fit, y_fit, cfg, name, params_override=params_override)
            pred_store[name][val_mask] = np.clip(np.asarray(model.predict(X_val), dtype=float), 0.0, None)

    valid_mask = np.ones(len(X), dtype=bool)
    for name in members:
        valid_mask &= np.isfinite(pred_store[name])

    X_oof = X.loc[valid_mask].reset_index(drop=True)
    y_oof = y.loc[valid_mask].reset_index(drop=True)
    member_preds = {name: pred_store[name][valid_mask] for name in members}
    metrics = []
    for name in members:
        item = _eval_metrics(y_oof, member_preds[name])
        item["name"] = name
        item["n_oof"] = int(len(y_oof))
        metrics.append(item)
    return YieldOOFResult(
        X=X_oof,
        y=y_oof,
        years=used_years,
        member_preds=member_preds,
        member_metrics=metrics,
    )


def _search_member_best_params(
    X: pd.DataFrame,
    y: pd.Series,
    row_df: pd.DataFrame,
    cfg: dict,
    model_name: str,
    search_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    baseline_params = dict(_resolve_member_params(cfg, model_name))
    if not search_cfg.get("enabled", False):
        return baseline_params, None, [], {"strategy": "fixed_config", "trials_requested": 0, "trials_run": 0}

    model_seed_offset = {"hgb": 11, "rf": 23, "etr": 37}.get(str(model_name or "").strip().lower(), 53)
    rng = np.random.RandomState(int(search_cfg.get("seed", 253)) + model_seed_offset)
    trials_requested = max(1, int(search_cfg.get("trials", 10)))
    history: List[Dict[str, Any]] = []
    seen = set()
    best_params = dict(baseline_params)
    best_record: Optional[Dict[str, Any]] = None
    trials_run = 0

    for trial in range(trials_requested):
        if trial == 0:
            params = dict(baseline_params)
            source = "baseline_config"
        else:
            params = _sample_member_search_params(rng, model_name, baseline_params, search_cfg)
            for _ in range(20):
                if _member_param_key(model_name, params) not in seen:
                    break
                params = _sample_member_search_params(rng, model_name, baseline_params, search_cfg)
            source = "random_search"
        key = _member_param_key(model_name, params)
        if key in seen:
            continue
        seen.add(key)
        oof = _build_oof_predictions(
            X,
            y,
            row_df,
            cfg,
            [model_name],
            member_param_overrides={str(model_name): dict(params)},
        )
        if len(oof.y) <= 0:
            history.append(
                {
                    "trial": int(trial),
                    "source": source,
                    "params": dict(params),
                    "metric": None,
                    "oof_rows": 0,
                    "oof_years": [],
                }
            )
            trials_run += 1
            continue
        pred = oof.member_preds[str(model_name)]
        loss = float(_loss_per_candidate(oof.y.to_numpy(dtype=float), pred.reshape(1, -1), str(search_cfg.get("metric", "rmse")))[0])
        metrics = _eval_metrics(oof.y, pred)
        record = {
            "trial": int(trial),
            "source": source,
            "params": dict(params),
            "metric": loss,
            "oof_rows": int(len(oof.y)),
            "oof_years": [int(v) for v in oof.years],
            **metrics,
        }
        history.append(record)
        trials_run += 1
        if best_record is None or float(record["metric"]) < float(best_record["metric"]):
            best_record = dict(record)
            best_params = dict(params)

    meta = {
        "strategy": "random_search_walk_forward_oof",
        "trials_requested": int(trials_requested),
        "trials_run": int(trials_run),
        "metric": str(search_cfg.get("metric", "rmse")),
    }
    return best_params, best_record, history, meta


def train_yield_model(
    df: pd.DataFrame,
    cfg: dict,
    test_ratio: Optional[float] = None,
    validation_cutoff: Optional[str] = None,
    strict_cutoff_split: bool = True,
    verbose: bool = False,
    label: Optional[str] = None,
) -> YieldModelResult:
    if df.empty:
        raise ValueError("empty yield history")

    target_col = TARGET_COL
    train_df, test_df, split_info = _split_train_test(
        df,
        test_ratio=test_ratio,
        validation_cutoff=validation_cutoff,
        strict_cutoff_split=strict_cutoff_split,
    )

    X_train, y_train, feat_cols = _prepare_features(train_df, target_col, cfg=cfg, history_df=train_df)
    X_test_raw, y_test, _ = _prepare_features(test_df, target_col, cfg=cfg, history_df=df)
    X_test = X_test_raw.reindex(columns=feat_cols, fill_value=0.0)
    crop_last, global_prior = _build_crop_last_prior(train_df, target_col)

    if X_train.empty:
        raise ValueError("not enough data for yield model")

    reg = str(cfg.get("regressor", "hgb")).lower()
    members = list(cfg.get("ensemble_members", ["hgb", "rf", "etr"])) if reg == "ensemble" else [reg]
    yield_search_cfg = _resolve_yield_search_cfg(cfg)
    member_best_params: Dict[str, Dict[str, Any]] = {}
    member_search_summary: Dict[str, Any] = {}
    member_search_history: Dict[str, List[Dict[str, Any]]] = {}

    for name in members:
        best_params, best_record, search_history, search_meta = _search_member_best_params(
            X_train,
            y_train,
            train_df.reset_index(drop=True),
            cfg,
            name,
            yield_search_cfg,
        )
        member_best_params[str(name)] = dict(best_params)
        member_search_summary[str(name)] = {
            **search_meta,
            "best_params": dict(best_params),
            "best_record": best_record,
        }
        member_search_history[str(name)] = list(search_history)

    oof = _build_oof_predictions(
        X_train,
        y_train,
        train_df.reset_index(drop=True),
        cfg,
        members,
        member_param_overrides=member_best_params,
    )

    if len(members) > 1 and len(oof.y) > 0:
        oof_member_matrix = np.column_stack([oof.member_preds[name] for name in members])
        ensemble_weights, ensemble_oof_loss = _search_simplex_weights(
            oof_member_matrix,
            oof.y.to_numpy(dtype=float),
            metric=str(cfg.get("ensemble_weight_metric", "rmse")),
            grid_steps=int(cfg.get("ensemble_weight_grid_steps", 20)),
        )
    else:
        oof_member_matrix = (
            np.column_stack([oof.member_preds[name] for name in members]) if len(oof.y) > 0 else np.zeros((0, len(members)))
        )
        ensemble_weights = [1.0]
        ensemble_oof_loss = np.nan

    if len(oof.y) > 0:
        prior_weights, prior_oof_loss = _search_simplex_weights(
            _prior_component_matrix(oof.X),
            oof.y.to_numpy(dtype=float),
            metric=str(cfg.get("prior_weight_metric", "mae")),
            grid_steps=int(cfg.get("prior_weight_grid_steps", 20)),
            availability=_prior_availability_matrix(oof.X),
            fallback=np.full(len(oof.y), global_prior, dtype=float),
        )
    else:
        prior_weights = [0.35, 0.20, 0.20, 0.10, 0.15]
        prior_oof_loss = np.nan

    prior_weights = np.asarray(prior_weights, dtype=float)
    if prior_weights.size != len(PRIOR_COMPONENT_COLS) or prior_weights.sum() <= EPSILON:
        prior_weights = np.full(len(PRIOR_COMPONENT_COLS), 1.0 / len(PRIOR_COMPONENT_COLS), dtype=float)
    else:
        prior_weights = prior_weights / prior_weights.sum()

    lambda_model = None
    lambda_meta: Dict[str, object] = {"kind": "constant_fallback", "feature_cols": LAMBDA_FEATURE_COLS}
    default_lambda = float(cfg.get("blend_last_weight", 0.0))
    lambda_stats = {"lambda_train_rows": 0, "lambda_oof_mean": None, "lambda_oof_min": None, "lambda_oof_max": None}

    if len(oof.y) > 0:
        base_oof = (
            oof_member_matrix @ np.asarray(ensemble_weights, dtype=float)
            if oof_member_matrix.size
            else np.zeros(len(oof.y), dtype=float)
        )
        prior_oof = _blend_prior_from_features(
            oof.X,
            prior_weights=prior_weights.tolist(),
            crop_last=crop_last,
            global_prior=global_prior,
        )
        lambda_star = np.clip(
            ((oof.y.to_numpy(dtype=float) - base_oof) * (prior_oof - base_oof))
            / ((prior_oof - base_oof) ** 2 + EPSILON),
            0.0,
            1.0,
        )
        default_lambda = _learn_default_lambda(oof.y.to_numpy(dtype=float), base_oof, prior_oof, cfg)
        lambda_features = _build_lambda_feature_frame(oof.X, oof_member_matrix)
        lambda_model, lambda_meta = _fit_lambda_model(lambda_features, lambda_star, cfg)
        lambda_stats = {
            "lambda_train_rows": int(len(lambda_star)),
            "lambda_oof_mean": float(np.mean(lambda_star)),
            "lambda_oof_min": float(np.min(lambda_star)),
            "lambda_oof_max": float(np.max(lambda_star)),
        }

    models = []
    metrics_list = []
    test_member_preds = []
    for name in members:
        params_override = member_best_params.get(str(name))
        model, pred, metrics = _train_one(X_train, y_train, X_test, y_test, cfg, name, params_override=params_override)
        models.append(model)
        metrics_list.append({"name": name, "params": dict(params_override or _resolve_member_params(cfg, name)), **metrics})
        if len(y_test) > 0:
            test_member_preds.append(pred)

    if reg == "ensemble":
        base_model: object = EnsembleModel(models=models, weights=ensemble_weights, feature_cols=feat_cols)
    else:
        base_model = models[0]

    final_model = AdaptiveBlendYieldModel(
        model=base_model,
        crop_last=crop_last,
        global_prior=global_prior,
        prior_weights=prior_weights.tolist(),
        lambda_model=lambda_model,
        default_lambda=default_lambda,
        feature_cols=feat_cols,
    )

    if len(y_test) == 0:
        metrics = {"mae": None, "rmse": None, "mape": None, "n_test": 0}
    else:
        member_matrix_test = np.column_stack(test_member_preds) if test_member_preds else np.zeros((len(y_test), 0), dtype=float)
        details = final_model.predict_details(X_test)
        metrics = _eval_metrics(y_test, details["final_pred"])
        metrics["n_test"] = int(len(y_test))
        metrics["lambda_mean_test"] = float(np.mean(details["lambda_pred"]))
        metrics["lambda_min_test"] = float(np.min(details["lambda_pred"]))
        metrics["lambda_max_test"] = float(np.max(details["lambda_pred"]))
        metrics["prior_mean_test"] = float(np.mean(details["prior_pred"]))
        metrics["base_mean_test"] = float(np.mean(details["base_pred"]))
        if member_matrix_test.size:
            metrics["member_std_mean_test"] = float(np.mean(np.std(member_matrix_test, axis=1)))

    metrics["ensemble_members"] = metrics_list
    metrics["ensemble_weights"] = [float(v) for v in ensemble_weights]
    metrics["ensemble_oof_loss"] = None if np.isnan(ensemble_oof_loss) else float(ensemble_oof_loss)
    metrics["oof_member_metrics"] = oof.member_metrics
    metrics["oof_rows"] = int(len(oof.y))
    metrics["oof_years"] = [int(v) for v in oof.years]
    metrics["search_enabled"] = bool(yield_search_cfg.get("enabled", False))
    metrics["search_metric"] = str(yield_search_cfg.get("metric", "rmse"))
    metrics["hpo_best_params"] = {str(name): dict(params) for name, params in member_best_params.items()}
    metrics["hpo_search_summary"] = member_search_summary
    metrics["hpo_search_history"] = member_search_history
    metrics["prior_weights"] = [float(v) for v in prior_weights.tolist()]
    metrics["prior_oof_loss"] = None if np.isnan(prior_oof_loss) else float(prior_oof_loss)
    metrics["prior_component_cols"] = PRIOR_COMPONENT_COLS
    metrics["lambda_default"] = float(default_lambda)
    metrics["lambda_model"] = lambda_meta
    metrics.update(lambda_stats)
    metrics.update(split_info)

    if verbose:
        name = label or "yield"
        mae_text = "NA" if metrics.get("mae") is None else f"{metrics['mae']:.4f}"
        rmse_text = "NA" if metrics.get("rmse") is None else f"{metrics['rmse']:.4f}"
        mape_text = "NA" if metrics.get("mape") is None else f"{metrics['mape']:.4f}"
        print(
            f"[yield][{name}][{reg}] n_test={metrics['n_test']} "
            f"mae={mae_text} rmse={rmse_text} mape={mape_text}"
        )
    return YieldModelResult(model=final_model, metrics=metrics, feature_cols=feat_cols)
