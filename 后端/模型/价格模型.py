from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import warnings
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - import checked at runtime before use
    LGBMRegressor = None

from 后端.特征工程 import PRICE_FLOOR, make_supervised


@dataclass
class PriceModelResult:
    model: object
    metrics: Dict[str, float]
    feature_cols: list
    artifacts: Optional[Dict[str, Any]] = None


@dataclass
class _PriceDataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_dates: pd.Series
    test_dates: pd.Series
    feature_cols: List[str]
    split_info: Dict[str, object]


@dataclass
class _HybridStepBundle:
    baseline: object
    trend: object
    residual: object
    trend_feature_cols: List[str]


DEFAULT_VALIDATION_CUTOFF_DATE = "2020-12-31"
DIRECT_PRICE_ARTIFACT_TAG = "直接趋势残差_v3"
LEGACY_DIRECT_PRICE_ARTIFACT_TAG = "混合直推_v3"


def _format_date(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def _resolve_cutoff_timestamp(value: Optional[str]) -> pd.Timestamp:
    text = str(value or "").strip() or DEFAULT_VALIDATION_CUTOFF_DATE
    try:
        ts = pd.Timestamp(text)
    except Exception:
        ts = pd.Timestamp(DEFAULT_VALIDATION_CUTOFF_DATE)
    if pd.isna(ts):
        ts = pd.Timestamp(DEFAULT_VALIDATION_CUTOFF_DATE)
    return ts.normalize()


def _resolve_target_mode(cfg: dict) -> str:
    mode = str(cfg.get("target_mode", "price")).strip().lower()
    if mode in {"return", "logreturn", "log_return", "ret", "delta"}:
        return "log_return"
    return "price"


def _resolve_feature_space(cfg: dict, target_mode: str) -> str:
    raw = str(cfg.get("feature_space", "")).strip().lower()
    if raw in {"price", "log_price"}:
        return raw
    return "log_price" if target_mode == "log_return" else "price"


def _resolve_time_raw_mode(cfg: dict, target_mode: str) -> Optional[str]:
    if "time_raw_mode" in cfg:
        val = str(cfg.get("time_raw_mode") or "").strip().lower()
        return val if val in {"none", "raw", "onehot"} else "none"
    include_raw = cfg.get("include_raw_time_features")
    if include_raw is None:
        return "none" if target_mode == "log_return" else "none"
    return "raw" if bool(include_raw) else "none"


def _use_direct_trend_residual_v3(cfg: dict) -> bool:
    if "enable_direct_trend_residual_v3" in cfg:
        return bool(cfg.get("enable_direct_trend_residual_v3"))
    return bool(cfg.get("enable_hybrid_direct_v3", True))


def _direct_trend_residual_fallback_to_legacy(cfg: dict) -> bool:
    if "direct_trend_residual_fallback_to_legacy" in cfg:
        return bool(cfg.get("direct_trend_residual_fallback_to_legacy"))
    return bool(cfg.get("hybrid_fallback_to_legacy", True))


def _fit_with_sample_weight(model, X, y, sample_weight: Optional[np.ndarray]):
    if sample_weight is None:
        model.fit(X, y)
        return
    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)


class EnsembleModel:
    def __init__(
        self,
        models: List[object],
        weights: List[float],
        feature_cols: List[str],
        prediction_space: str = "price",
    ):
        self.models = models
        self.weights = weights
        self.feature_cols = feature_cols
        self.prediction_space = str(prediction_space or "price").strip().lower()

    def predict(self, X: pd.DataFrame):
        preds = []
        for m in self.models:
            preds.append(m.predict(X))
        preds = np.vstack(preds)
        w = np.array(self.weights).reshape(-1, 1)
        out = (preds * w).sum(axis=0)
        prediction_space = str(getattr(self, "prediction_space", "price") or "price").strip().lower()
        if prediction_space == "price":
            return np.clip(out, 0.0, None)
        return out


class BaselineBlendModel:
    """Legacy rule-based baseline kept as one anchor candidate."""

    DEFAULT_FEATURE_WEIGHTS = {
        "lag_1": 0.28,
        "lag_3": 0.06,
        "lag_7": 0.22,
        "lag_14": 0.08,
        "lag_21": 0.05,
        "lag_30": 0.07,
        "lag_60": 0.05,
        "lag_90": 0.04,
        "lag_180": 0.03,
        "lag_365": 0.12,
        "roll_mean_7": 0.10,
        "roll_mean_14": 0.08,
        "roll_mean_30": 0.10,
        "roll_mean_90": 0.06,
    }

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.feature_weights = dict(self.DEFAULT_FEATURE_WEIGHTS)
        override = cfg.get("baseline_feature_weights")
        if isinstance(override, dict):
            for k, v in override.items():
                key = str(k).strip()
                try:
                    val = float(v)
                except Exception:
                    continue
                if not key:
                    continue
                self.feature_weights[key] = max(0.0, val)
        self.price_floor = max(float(cfg.get("price_floor", PRICE_FLOOR)), PRICE_FLOOR)
        self.fallback_value = 0.0
        self.feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_cols = [str(c) for c in list(X.columns)]
        y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
        y_arr = y_arr[np.isfinite(y_arr)]
        if y_arr.size:
            self.fallback_value = float(np.median(y_arr))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if X is None:
            return np.asarray([], dtype=float)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        out = np.zeros(len(X), dtype=float)
        denom = np.zeros(len(X), dtype=float)

        for col, weight in self.feature_weights.items():
            if weight <= 0.0 or col not in X.columns:
                continue
            vals = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(vals)
            if not np.any(valid):
                continue
            out[valid] += vals[valid] * float(weight)
            denom[valid] += float(weight)

        pred = np.where(denom > 0.0, out / np.maximum(denom, 1e-9), self.fallback_value)
        pred = np.where(np.isfinite(pred), pred, self.fallback_value)
        return np.clip(pred.astype(float), self.price_floor, None)


def _numeric_feature_array(X: pd.DataFrame, col: str) -> np.ndarray:
    if col not in X.columns:
        return np.full(len(X), np.nan, dtype=float)
    vals = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
    vals = vals.astype(float, copy=False)
    vals[~np.isfinite(vals)] = np.nan
    return vals


def _weighted_feature_average(X: pd.DataFrame, feature_weights: Dict[str, float], fallback: float) -> np.ndarray:
    out = np.zeros(len(X), dtype=float)
    denom = np.zeros(len(X), dtype=float)
    for col, weight in feature_weights.items():
        weight = float(weight)
        if weight <= 0.0:
            continue
        vals = _numeric_feature_array(X, str(col))
        valid = np.isfinite(vals)
        if not np.any(valid):
            continue
        out[valid] += vals[valid] * weight
        denom[valid] += weight
    pred = np.where(denom > 0.0, out / np.maximum(denom, 1e-9), float(fallback))
    pred = np.where(np.isfinite(pred), pred, float(fallback))
    return pred.astype(float)


def _first_valid_feature(X: pd.DataFrame, candidates: List[str], fallback: float) -> np.ndarray:
    pred = np.full(len(X), np.nan, dtype=float)
    for col in candidates:
        vals = _numeric_feature_array(X, str(col))
        fill_mask = ~np.isfinite(pred) & np.isfinite(vals)
        if np.any(fill_mask):
            pred[fill_mask] = vals[fill_mask]
    pred = np.where(np.isfinite(pred), pred, float(fallback))
    return pred.astype(float)


def _median_feature_value(X: pd.DataFrame, candidates: List[str], fallback: float) -> np.ndarray:
    arrays = [_numeric_feature_array(X, str(col)) for col in candidates]
    if not arrays:
        return np.full(len(X), float(fallback), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pred = np.nanmedian(np.vstack(arrays), axis=0)
    pred = np.where(np.isfinite(pred), pred, float(fallback))
    return pred.astype(float)


def _normalize_weight_vector(values: np.ndarray, size: int, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    raw = np.asarray(values, dtype=float).reshape(-1)
    if raw.size != int(size):
        if fallback is not None:
            raw = np.asarray(fallback, dtype=float).reshape(-1)
        else:
            raw = np.ones(int(size), dtype=float)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    raw = np.clip(raw, 0.0, None)
    total = float(raw.sum())
    if total <= 0.0:
        if fallback is not None:
            raw = np.asarray(fallback, dtype=float).reshape(-1)
            raw = np.where(np.isfinite(raw), raw, 0.0)
            raw = np.clip(raw, 0.0, None)
            total = float(raw.sum())
        if total <= 0.0:
            raw = np.ones(int(size), dtype=float)
            total = float(raw.sum())
    return (raw / max(total, 1e-12)).astype(float)


def _project_to_simplex(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=float).reshape(-1)
    if v.size == 1:
        return np.asarray([1.0], dtype=float)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, u.size + 1, dtype=float)
    cond = u - cssv / ind > 0.0
    if not np.any(cond):
        return np.full(v.size, 1.0 / float(v.size), dtype=float)
    rho = int(ind[cond][-1])
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0.0)
    total = float(w.sum())
    if total <= 0.0:
        return np.full(v.size, 1.0 / float(v.size), dtype=float)
    return (w / total).astype(float)


def _solve_convex_weights(
    pred_matrix: np.ndarray,
    y_true: np.ndarray,
    *,
    reference: np.ndarray,
    l2_reg: float,
    max_iter: int = 300,
    tol: float = 1e-8,
) -> np.ndarray:
    P = np.asarray(pred_matrix, dtype=float)
    y = np.asarray(y_true, dtype=float).reshape(-1)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    if P.size == 0 or y.size == 0:
        return _normalize_weight_vector(reference, P.shape[1] if P.ndim == 2 else 1)
    valid = np.all(np.isfinite(P), axis=1) & np.isfinite(y)
    P = P[valid]
    y = y[valid]
    n_rows = int(P.shape[0])
    n_cols = int(P.shape[1])
    ref = _normalize_weight_vector(reference, n_cols)
    if n_rows == 0 or n_cols == 0:
        return ref
    if n_cols == 1:
        return np.asarray([1.0], dtype=float)

    reg = max(0.0, float(l2_reg))
    gram_scale = float(np.linalg.norm(P, ord=2) ** 2) / float(max(n_rows, 1))
    lipschitz = max(1e-8, 2.0 * gram_scale + 2.0 * reg)
    step = 1.0 / lipschitz
    w = ref.copy()

    for _ in range(int(max_iter)):
        err = P @ w - y
        grad = (2.0 / float(max(n_rows, 1))) * (P.T @ err) + 2.0 * reg * (w - ref)
        w_next = _project_to_simplex(w - step * grad)
        if float(np.linalg.norm(w_next - w, ord=1)) <= tol:
            w = w_next
            break
        w = w_next
    return _normalize_weight_vector(w, n_cols, fallback=ref)


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    valid = np.isfinite(y) & np.isfinite(p)
    if not np.any(valid):
        return float("inf")
    return float(mean_absolute_error(y[valid], p[valid]))


def _relative_improvement(reference_value: float, candidate_value: float) -> float:
    ref = float(reference_value)
    cand = float(candidate_value)
    if not np.isfinite(ref) or ref <= 0.0 or not np.isfinite(cand):
        return 0.0
    return float((ref - cand) / max(ref, 1e-9))


def _blend_two_predictions(main_pred: np.ndarray, baseline_pred: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = _normalize_weight_vector(weights, 2)
    out = (float(w[0]) * np.asarray(main_pred, dtype=float).reshape(-1)) + (
        float(w[1]) * np.asarray(baseline_pred, dtype=float).reshape(-1)
    )
    return np.asarray(out, dtype=float).reshape(-1)


def _shrink_weights_towards_reference(weights: np.ndarray, reference: np.ndarray, trust: float) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    ref = _normalize_weight_vector(reference, w.size if w.size else 1)
    if w.size != ref.size:
        w = ref.copy()
    else:
        w = _normalize_weight_vector(w, ref.size, fallback=ref)
    alpha = min(max(float(trust), 0.0), 1.0)
    return _normalize_weight_vector(((1.0 - alpha) * ref) + (alpha * w), ref.size, fallback=ref)


def _shrink_weights_to_max(weights: np.ndarray, reference: np.ndarray, max_weight: float) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    ref = _normalize_weight_vector(reference, w.size if w.size else 1)
    if w.size != ref.size:
        w = ref.copy()
    else:
        w = _normalize_weight_vector(w, ref.size, fallback=ref)
    cap = min(max(float(max_weight), 1.0 / float(max(ref.size, 1))), 1.0)
    if ref.size <= 1 or float(np.max(w)) <= cap + 1e-9:
        return w
    if float(np.max(ref)) > cap + 1e-9:
        ref = np.full(ref.size, 1.0 / float(ref.size), dtype=float)
    lo = 0.0
    hi = 1.0
    best = ref.copy()
    for _ in range(40):
        mid = (lo + hi) / 2.0
        cand = _normalize_weight_vector(((1.0 - mid) * ref) + (mid * w), ref.size, fallback=ref)
        if float(np.max(cand)) <= cap + 1e-9:
            best = cand
            lo = mid
        else:
            hi = mid
    return _normalize_weight_vector(best, ref.size, fallback=ref)


def _make_two_way_weight(winner_idx: int, winner_weight: float) -> np.ndarray:
    idx = 0 if int(winner_idx) <= 0 else 1
    win = min(max(float(winner_weight), 0.0), 1.0)
    if idx == 0:
        return np.asarray([win, 1.0 - win], dtype=float)
    return np.asarray([1.0 - win, win], dtype=float)


def _guard_anchor_component_weights(
    pred_matrix: np.ndarray,
    y_true: np.ndarray,
    learned: np.ndarray,
    reference: np.ndarray,
    cfg: dict,
) -> Tuple[np.ndarray, str]:
    P = np.asarray(pred_matrix, dtype=float)
    y = np.asarray(y_true, dtype=float).reshape(-1)
    ref = _normalize_weight_vector(reference, P.shape[1] if P.ndim == 2 else 1)
    if P.ndim != 2 or P.size == 0 or y.size == 0:
        return ref, "reference_fallback"
    learned_raw = _normalize_weight_vector(learned, P.shape[1], fallback=ref)
    ref_pred = P @ ref
    raw_pred = P @ learned_raw
    ref_mae = _safe_mae(y, ref_pred)
    raw_mae = _safe_mae(y, raw_pred)

    min_gain = max(0.0, float(cfg.get("anchor_guard_min_relative_gain", 0.01)))
    full_trust_gain = max(min_gain, float(cfg.get("anchor_guard_full_trust_gain", 0.04)))
    selection_tol = max(0.0, float(cfg.get("anchor_guard_selection_tolerance", 0.003)))
    max_weight = min(max(float(cfg.get("anchor_component_max_weight", 0.65)), 1.0 / float(P.shape[1])), 1.0)

    gain = _relative_improvement(ref_mae, raw_mae)
    trust = 0.0 if gain <= 0.0 else min(1.0, gain / max(full_trust_gain, 1e-9))
    guarded = _shrink_weights_towards_reference(learned_raw, ref, trust)
    guarded = _shrink_weights_to_max(guarded, ref, max_weight=max_weight)

    candidates = [
        ("reference", ref),
        ("guarded", guarded),
    ]
    scored = []
    for name, weights in candidates:
        pred = P @ weights
        scored.append(
            {
                "name": name,
                "weights": _normalize_weight_vector(weights, P.shape[1], fallback=ref),
                "mae": _safe_mae(y, pred),
                "l1_to_ref": float(np.abs(weights - ref).sum()),
                "max_weight": float(np.max(weights)),
            }
        )
    best_mae = min(float(row["mae"]) for row in scored)
    best_gain = _relative_improvement(ref_mae, best_mae)
    if best_gain < min_gain:
        return ref, "tail_calibration_guard_reference"

    eligible = [row for row in scored if row["name"] != "reference" and float(row["mae"]) <= best_mae * (1.0 + selection_tol)]
    if not eligible:
        picked = min(scored, key=lambda row: (float(row["mae"]), float(row["l1_to_ref"]), float(row["max_weight"])))
        return np.asarray(picked["weights"], dtype=float), f"tail_calibration_guard_{picked['name']}"

    picked = min(eligible, key=lambda row: (float(row["l1_to_ref"]), float(row["max_weight"]), float(row["mae"])))
    return np.asarray(picked["weights"], dtype=float), f"tail_calibration_guard_{picked['name']}"


def _guard_blend_weights(
    *,
    main_pred: np.ndarray,
    baseline_pred: np.ndarray,
    y_true: np.ndarray,
    learned: np.ndarray,
    reference: np.ndarray,
    cfg: dict,
) -> Tuple[np.ndarray, str]:
    ref = _normalize_weight_vector(reference, 2)
    raw = _normalize_weight_vector(learned, 2, fallback=ref)
    y = np.asarray(y_true, dtype=float).reshape(-1)
    main_arr = np.asarray(main_pred, dtype=float).reshape(-1)
    base_arr = np.asarray(baseline_pred, dtype=float).reshape(-1)
    if y.size == 0 or main_arr.size == 0 or base_arr.size == 0:
        return ref, "reference_fallback"

    main_mae = _safe_mae(y, main_arr)
    base_mae = _safe_mae(y, base_arr)
    ref_mae = _safe_mae(y, _blend_two_predictions(main_arr, base_arr, ref))
    raw_mae = _safe_mae(y, _blend_two_predictions(main_arr, base_arr, raw))

    min_gain = max(0.0, float(cfg.get("blend_guard_min_relative_gain", 0.03)))
    full_trust_gain = max(min_gain, float(cfg.get("blend_guard_full_trust_gain", 0.12)))
    selection_tol = max(0.0, float(cfg.get("blend_guard_selection_tolerance", 0.005)))
    max_weight = min(max(float(cfg.get("blend_guard_max_weight", 0.85)), 0.5), 1.0)
    winner_soft_weight = min(max(float(cfg.get("blend_guard_winner_soft_weight", 0.65)), 0.5), max_weight)
    winner_floor = min(max(float(cfg.get("blend_guard_winner_min_weight", 0.55)), 0.5), max_weight)
    direction_margin = max(0.0, float(cfg.get("blend_guard_direction_margin", 0.08)))

    gain = _relative_improvement(ref_mae, raw_mae)
    trust = 0.0 if gain <= 0.0 else min(1.0, gain / max(full_trust_gain, 1e-9))
    guarded = _shrink_weights_towards_reference(raw, ref, trust)
    guarded = _shrink_weights_to_max(guarded, ref, max_weight=max_weight)

    winner_idx = 0 if main_mae <= base_mae else 1
    winner_mae = min(main_mae, base_mae)
    loser_mae = max(main_mae, base_mae)
    winner_gap = _relative_improvement(winner_mae, winner_mae)
    if np.isfinite(winner_mae) and winner_mae > 0.0 and np.isfinite(loser_mae):
        winner_gap = float((loser_mae - winner_mae) / max(winner_mae, 1e-9))
    if winner_gap >= direction_margin and float(guarded[winner_idx]) < winner_floor:
        guarded = _make_two_way_weight(winner_idx, winner_floor)

    winner_soft = _make_two_way_weight(winner_idx, max(float(ref[winner_idx]), winner_soft_weight))
    winner_strong = _make_two_way_weight(winner_idx, max(float(ref[winner_idx]), max_weight))
    candidates = [
        ("reference", ref),
        ("guarded", guarded),
        ("winner_soft", winner_soft),
        ("winner_strong", winner_strong),
    ]
    scored = []
    for name, weights in candidates:
        w = _normalize_weight_vector(weights, 2, fallback=ref)
        pred = _blend_two_predictions(main_arr, base_arr, w)
        scored.append(
            {
                "name": name,
                "weights": w,
                "mae": _safe_mae(y, pred),
                "l1_to_ref": float(np.abs(w - ref).sum()),
                "max_weight": float(np.max(w)),
            }
        )
    best_mae = min(float(row["mae"]) for row in scored)
    best_gain = _relative_improvement(ref_mae, best_mae)
    if best_gain < min_gain:
        return ref, "tail_calibration_guard_reference"

    eligible = [row for row in scored if row["name"] != "reference" and float(row["mae"]) <= best_mae * (1.0 + selection_tol)]
    if not eligible:
        picked = min(scored, key=lambda row: (float(row["mae"]), float(row["l1_to_ref"]), float(row["max_weight"])))
        return np.asarray(picked["weights"], dtype=float), f"tail_calibration_guard_{picked['name']}"

    picked = min(eligible, key=lambda row: (float(row["l1_to_ref"]), float(row["max_weight"]), float(row["mae"])))
    return np.asarray(picked["weights"], dtype=float), f"tail_calibration_guard_{picked['name']}"


class BenchmarkAnchorEnsemble:
    """Small benchmark family used as the direct-model anchor layer."""

    DEFAULT_COMPONENT_NAMES = [
        "legacy_blend",
        "last_price",
        "short_cycle",
        "medium_cycle",
        "yearly_cycle",
        "robust_level",
    ]

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.cfg = dict(cfg)
        raw_names = cfg.get("anchor_component_names")
        names = []
        if isinstance(raw_names, (list, tuple, set)):
            names = [str(x).strip() for x in raw_names if str(x).strip()]
        allowed = [name for name in names if name in self.DEFAULT_COMPONENT_NAMES]
        self.component_names = allowed or list(self.DEFAULT_COMPONENT_NAMES)
        self.component_weights = np.full(len(self.component_names), 1.0 / float(len(self.component_names)), dtype=float)
        self.price_floor = max(float(cfg.get("price_floor", PRICE_FLOOR)), PRICE_FLOOR)
        self.fallback_value = 0.0
        self.feature_cols: List[str] = []
        self.legacy_blend = BaselineBlendModel(cfg)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_cols = [str(c) for c in list(X.columns)]
        y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
        y_arr = y_arr[np.isfinite(y_arr)]
        if y_arr.size:
            self.fallback_value = float(np.median(y_arr))
        self.legacy_blend.fit(X, y)
        return self

    def _coerce_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if self.feature_cols:
            X = X.reindex(columns=self.feature_cols, fill_value=0.0)
        return X

    def set_component_weights(self, weights: np.ndarray):
        self.component_weights = _normalize_weight_vector(weights, len(self.component_names), fallback=self.component_weights)
        return self

    def get_component_weight_map(self) -> Dict[str, float]:
        return {name: float(weight) for name, weight in zip(self.component_names, self.component_weights)}

    def _predict_component(self, X: pd.DataFrame, name: str) -> np.ndarray:
        if name == "legacy_blend":
            pred = self.legacy_blend.predict(X)
        elif name == "last_price":
            pred = _first_valid_feature(
                X,
                ["lag_1", "lag_3", "lag_7", "roll_mean_7", "roll_mean_14", "roll_mean_30"],
                self.fallback_value,
            )
        elif name == "short_cycle":
            pred = _weighted_feature_average(
                X,
                {"lag_7": 0.35, "roll_mean_7": 0.30, "lag_14": 0.20, "roll_mean_14": 0.15},
                self.fallback_value,
            )
        elif name == "medium_cycle":
            pred = _weighted_feature_average(
                X,
                {"lag_21": 0.20, "lag_30": 0.40, "roll_mean_30": 0.25, "lag_60": 0.15},
                self.fallback_value,
            )
        elif name == "yearly_cycle":
            pred = _first_valid_feature(
                X,
                ["lag_365", "lag_180", "lag_90", "roll_mean_90", "lag_60"],
                self.fallback_value,
            )
        elif name == "robust_level":
            pred = _median_feature_value(
                X,
                ["lag_1", "lag_7", "lag_30", "lag_90", "lag_365", "roll_mean_7", "roll_mean_30", "roll_mean_90"],
                self.fallback_value,
            )
        else:
            pred = np.full(len(X), float(self.fallback_value), dtype=float)
        pred = np.where(np.isfinite(pred), pred, float(self.fallback_value))
        return np.clip(pred.astype(float), self.price_floor, None)

    def predict_components(self, X: pd.DataFrame) -> pd.DataFrame:
        X_in = self._coerce_X(X)
        data = {name: self._predict_component(X_in, name) for name in self.component_names}
        return pd.DataFrame(data, index=X_in.index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        comp = self.predict_components(X)
        if comp.empty:
            return np.full(len(X), float(self.fallback_value), dtype=float)
        weights = _normalize_weight_vector(self.component_weights, comp.shape[1])
        pred = comp.to_numpy(dtype=float) @ weights
        pred = np.where(np.isfinite(pred), pred, float(self.fallback_value))
        return np.clip(pred.astype(float), self.price_floor, None)


class HybridDirectPriceModel:
    """Baseline + trend + residual direct multi-horizon model."""

    def __init__(
        self,
        *,
        step_models: Dict[int, _HybridStepBundle],
        feature_cols: List[str],
        blend_weight_main: float,
        blend_weight_baseline: float,
        serve_horizon: int,
        price_floor: float,
    ):
        if not step_models:
            raise ValueError("empty step_models for HybridDirectPriceModel")
        self.step_models = {int(k): v for k, v in step_models.items()}
        self.horizons = sorted(self.step_models.keys())
        self.feature_cols = [str(c) for c in (feature_cols or [])]
        self.blend_weight_main = float(blend_weight_main)
        self.blend_weight_baseline = float(blend_weight_baseline)
        self.serve_horizon = int(serve_horizon)
        self.price_floor = max(float(price_floor), PRICE_FLOOR)

    def _coerce_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if self.feature_cols:
            X = X.reindex(columns=self.feature_cols, fill_value=0.0)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return X

    def _predict_exact(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        h = int(horizon)
        bundle = self.step_models[h]
        X_full = self._coerce_X(X)
        X_trend = X_full.reindex(columns=bundle.trend_feature_cols, fill_value=0.0)

        baseline_pred = np.asarray(bundle.baseline.predict(X_full), dtype=float).reshape(-1)
        trend_pred = np.asarray(bundle.trend.predict(X_trend), dtype=float).reshape(-1)
        residual_pred = np.asarray(bundle.residual.predict(X_full), dtype=float).reshape(-1)

        main_pred = trend_pred + residual_pred
        main_weight = float(getattr(bundle, "blend_weight_main", self.blend_weight_main))
        baseline_weight = float(getattr(bundle, "blend_weight_baseline", self.blend_weight_baseline))
        out = main_weight * main_pred + baseline_weight * baseline_pred
        out = np.where(np.isfinite(out), out, baseline_pred)
        return np.clip(out.astype(float), self.price_floor, None)

    def _bounded_horizon(self, horizon: int) -> int:
        h = int(horizon)
        if h in self.step_models:
            return h
        if h <= self.horizons[0]:
            return int(self.horizons[0])
        if h >= self.horizons[-1]:
            return int(self.horizons[-1])
        return h

    def predict_for_horizon(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        h = self._bounded_horizon(int(horizon))
        if h in self.step_models:
            return self._predict_exact(X, h)

        lower = max([x for x in self.horizons if x < h])
        upper = min([x for x in self.horizons if x > h])
        if lower == upper:
            return self._predict_exact(X, lower)

        left = self._predict_exact(X, lower)
        right = self._predict_exact(X, upper)
        alpha = (h - lower) / float(upper - lower)
        out = (1.0 - alpha) * left + alpha * right
        return np.clip(out.astype(float), self.price_floor, None)

    def predict_multi(self, X: pd.DataFrame) -> np.ndarray:
        X_full = self._coerce_X(X)
        preds = [self._predict_exact(X_full, h) for h in self.horizons]
        if not preds:
            return np.zeros((len(X_full), 0), dtype=float)
        return np.column_stack(preds)

    def predict(self, X: pd.DataFrame):
        return self.predict_for_horizon(X, self.serve_horizon)


def _weights_from_metric(metrics_list: List[dict], metric_key: str, power: float) -> List[float]:
    safe_power = max(1e-6, float(power))
    values = []
    for row in metrics_list:
        try:
            v = float(row.get(metric_key, row.get("rmse", 1.0)))
        except Exception:
            v = 1.0
        values.append(max(v, 1e-9))
    raw = 1.0 / np.power(np.asarray(values, dtype=float), safe_power)
    if not np.all(np.isfinite(raw)) or float(raw.sum()) <= 0.0:
        raw = np.ones(len(metrics_list), dtype=float)
    w = raw / raw.sum()
    return w.tolist()


def _select_ensemble_weights(metrics_list: List[dict], cfg: dict) -> Tuple[List[float], dict]:
    metric_key = str(cfg.get("ensemble_weight_metric", "rmse")).strip().lower() or "rmse"
    power = float(cfg.get("ensemble_weight_power", 1.0))
    if metric_key not in {"mae", "rmse", "mape"}:
        metric_key = "rmse"
    w = _weights_from_metric(metrics_list, metric_key=metric_key, power=power)

    detail = {
        "ensemble_mode": "weighted",
        "ensemble_weight_metric": metric_key,
        "ensemble_weight_power": max(1e-6, float(power)),
    }

    ratio_metric = str(cfg.get("ensemble_best_ratio_metric", "mape")).strip().lower() or "mape"
    ratio_threshold = float(cfg.get("ensemble_best_ratio_threshold", 0.0))
    gap_min = max(0.0, float(cfg.get("ensemble_best_gap_min", 0.0)))
    if ratio_metric not in {"mae", "rmse", "mape"}:
        ratio_metric = "mape"

    def _metric_value(row: dict, key: str) -> float:
        try:
            val = float(row.get(key))
            if np.isfinite(val):
                return val
        except Exception:
            pass
        for alt in ("rmse", "mae", "mape"):
            try:
                alt_val = float(row.get(alt))
                if np.isfinite(alt_val):
                    return alt_val
            except Exception:
                continue
        return float("inf")

    if ratio_threshold > 0.0 and len(metrics_list) >= 2:
        ranked = sorted(
            metrics_list,
            key=lambda row: _metric_value(row, ratio_metric),
        )
        best = ranked[0]
        second = ranked[1]
        best_val = _metric_value(best, ratio_metric)
        second_val = _metric_value(second, ratio_metric)
        if np.isfinite(best_val) and np.isfinite(second_val) and second_val > 0.0:
            best_ratio = best_val / max(second_val, 1e-9)
            best_gap = second_val - best_val
            if best_ratio <= ratio_threshold and best_gap >= gap_min:
                selected = str(best.get("name", "")).strip()
                one_hot = [1.0 if str(row.get("name", "")).strip() == selected else 0.0 for row in metrics_list]
                if sum(one_hot) > 0:
                    detail.update(
                        {
                            "ensemble_mode": "single_best",
                            "selected_member": selected,
                            "ensemble_best_ratio_metric": ratio_metric,
                            "ensemble_best_ratio": float(best_ratio),
                            "ensemble_best_gap": float(best_gap),
                            "ensemble_best_ratio_threshold": float(ratio_threshold),
                            "ensemble_best_gap_min": float(gap_min),
                        }
                    )
                    return one_hot, detail
    return w, detail


def build_regressor(cfg: dict, name: str):
    model_name = str(name or "hgb").strip().lower()
    n_jobs = int(cfg.get("n_jobs", 1))

    if model_name == "lgbm":
        if LGBMRegressor is None:
            raise ModuleNotFoundError(
                "lightgbm is required for price model residual training; install lightgbm instead of using an automatic fallback"
            )
        max_depth = int(cfg.get("lgbm_max_depth", -1))
        max_depth = max_depth if max_depth > 0 else -1
        return LGBMRegressor(
            n_estimators=int(cfg.get("lgbm_n_estimators", 520)),
            learning_rate=float(cfg.get("lgbm_learning_rate", 0.03)),
            num_leaves=int(cfg.get("lgbm_num_leaves", 63)),
            max_depth=max_depth,
            min_child_samples=int(cfg.get("lgbm_min_child_samples", 16)),
            subsample=float(cfg.get("lgbm_subsample", 0.9)),
            colsample_bytree=float(cfg.get("lgbm_colsample_bytree", 0.9)),
            reg_alpha=float(cfg.get("lgbm_reg_alpha", 0.0)),
            reg_lambda=float(cfg.get("lgbm_reg_lambda", 1.0)),
            random_state=42,
            n_jobs=n_jobs,
        )

    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=int(cfg.get("rf_n_estimators", 500)),
            min_samples_leaf=int(cfg.get("rf_min_samples_leaf", 2)),
            random_state=42,
            n_jobs=n_jobs,
        )
    if model_name == "etr":
        return ExtraTreesRegressor(
            n_estimators=int(cfg.get("etr_n_estimators", 700)),
            min_samples_leaf=int(cfg.get("etr_min_samples_leaf", 2)),
            random_state=42,
            n_jobs=n_jobs,
        )
    return HistGradientBoostingRegressor(
        max_iter=int(cfg.get("max_iter", 700)),
        max_depth=cfg.get("max_depth", 8),
        learning_rate=float(cfg.get("learning_rate", 0.03)),
        l2_regularization=float(cfg.get("l2_regularization", 1e-3)),
        random_state=42,
    )


def _wrap_target_transform(model, cfg: dict, target_mode: str):
    if target_mode == "log_return":
        return model
    t = str(cfg.get("target_transform", "log1p")).lower()
    if t == "log1p":
        return TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    return model


def _prepare_dataset(
    df: pd.DataFrame,
    cfg: dict,
    horizon: int,
    lags: list,
    windows: list,
    backtest_days: int,
    test_ratio: Optional[float] = None,
    validation_cutoff: Optional[str] = None,
    strict_cutoff_split: bool = True,
) -> _PriceDataset:
    target_mode = _resolve_target_mode(cfg)
    feature_space = _resolve_feature_space(cfg, target_mode)
    time_raw_mode = _resolve_time_raw_mode(cfg, target_mode)
    include_raw_time_features = time_raw_mode == "raw"

    X_all, y_all, dates = make_supervised(
        df,
        "modal_price",
        horizon,
        lags,
        windows,
        return_dates=True,
        target_mode=target_mode,
        feature_space=feature_space,
        include_raw_time_features=include_raw_time_features,
        time_raw_mode=time_raw_mode,
        price_floor=float(cfg.get("price_floor", PRICE_FLOOR)),
    )
    if X_all.empty:
        raise ValueError("not enough data for training price model")

    cutoff_ts = _resolve_cutoff_timestamp(validation_cutoff)
    train_idx = np.where(dates <= cutoff_ts)[0]
    test_idx = np.where(dates > cutoff_ts)[0]
    split_mode = "fixed_cutoff"
    post_cutoff_rows = int(len(test_idx))

    if len(train_idx) == 0:
        raise ValueError("no training rows at or before validation cutoff for price model")
    if len(test_idx) == 0:
        if strict_cutoff_split:
            split_mode = "fixed_cutoff_no_validation"
        elif test_ratio is not None and 0.0 < test_ratio < 1.0:
            n_test = max(20, int(len(X_all) * test_ratio))
            if len(X_all) <= n_test:
                raise ValueError("not enough supervised rows after split for price model")
            train_idx = np.arange(0, len(X_all) - n_test)
            test_idx = np.arange(len(X_all) - n_test, len(X_all))
            split_mode = "ratio_fallback"
        else:
            cutoff = dates.max() - pd.Timedelta(days=backtest_days)
            train_idx = np.where(dates <= cutoff)[0]
            test_idx = np.where(dates > cutoff)[0]
            split_mode = "backtest_window"
            if len(test_idx) < 20 or len(train_idx) < 50:
                n_test = max(20, int(len(X_all) * 0.2))
                train_idx = np.arange(0, len(X_all) - n_test)
                test_idx = np.arange(len(X_all) - n_test, len(X_all))
                split_mode = "ratio_fallback"

    X_train = X_all.iloc[train_idx].copy()
    y_train = y_all.iloc[train_idx].copy()
    X_test = X_all.iloc[test_idx].copy()
    y_test = y_all.iloc[test_idx].copy()
    d_train = dates.iloc[train_idx].copy()
    d_test = dates.iloc[test_idx].copy()
    feat_cols = list(X_train.columns)

    split_info = {
        "split_mode": split_mode,
        "split_cutoff_date": _format_date(cutoff_ts),
        "split_used_post_cutoff_validation": split_mode == "fixed_cutoff" and post_cutoff_rows > 0,
        "split_post_cutoff_rows": post_cutoff_rows,
        "split_train_rows": int(len(train_idx)),
        "split_test_rows": int(len(test_idx)),
        "split_train_start": _format_date(d_train.min() if len(d_train) else None),
        "split_train_end": _format_date(d_train.max() if len(d_train) else None),
        "split_test_start": _format_date(d_test.min() if len(d_test) else None),
        "split_test_end": _format_date(d_test.max() if len(d_test) else None),
        "target_mode": target_mode,
        "feature_space": feature_space,
        "time_raw_mode": time_raw_mode,
        "include_raw_time_features": bool(include_raw_time_features),
    }

    return _PriceDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_dates=d_train,
        test_dates=d_test,
        feature_cols=feat_cols,
        split_info=split_info,
    )


def _recency_weights(train_dates: pd.Series, cfg: dict):
    if not cfg.get("use_recency_weight", True):
        return None
    if train_dates.empty:
        return None
    halflife = float(cfg.get("recency_halflife_days", 365.0))
    halflife = max(1.0, halflife)
    age_days = (train_dates.max() - train_dates).dt.days.astype(float)
    w = np.power(0.5, age_days / halflife)
    w = np.clip(w, 1e-3, None)
    return (w / np.mean(w)).values


def _eval_metrics(y_true, y_pred, *, include_mape: bool) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        return {"mae": None, "rmse": None, "mape": None}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if include_mape:
        mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))))
    else:
        mape = None
    return {"mae": float(mae), "rmse": float(rmse), "mape": mape}


def _crop_key_from_label(label: Optional[str]) -> str:
    text = str(label or "").strip().lower()
    if ":" in text:
        text = text.split(":", 1)[0].strip()
    return text


def _to_int_list(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (int, float)):
        try:
            v = int(raw)
        except Exception:
            return []
        return [v] if v > 0 else []
    if not isinstance(raw, (list, tuple, set)):
        return []
    out: List[int] = []
    for item in raw:
        try:
            v = int(item)
        except Exception:
            continue
        if v > 0:
            out.append(v)
    return out


def _auto_horizon_grid(max_horizon: int) -> List[int]:
    h = max(1, int(max_horizon))
    anchors = [1, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 150, 180]
    out = [x for x in anchors if x <= h]
    if h not in out:
        out.append(h)
    return sorted(set(out))


def _resolve_direct_horizon_plan(cfg: dict, horizon: int, label: Optional[str]) -> Tuple[List[int], int, bool]:
    default_h = max(1, int(horizon))
    cap = max(default_h, int(cfg.get("direct_multi_horizon_cap", max(default_h, 365))))
    crop_key = _crop_key_from_label(label)
    crop_overrides = cfg.get("direct_multi_horizon_by_crop", {})
    crop_override_val = None
    if isinstance(crop_overrides, dict):
        for k, v in crop_overrides.items():
            if str(k).strip().lower() == crop_key and crop_key:
                crop_override_val = v
                break

    crop_override_applied = crop_override_val is not None
    if crop_override_val is None:
        raw_horizons = _to_int_list(cfg.get("direct_multi_horizons"))
    else:
        raw_horizons = _to_int_list(crop_override_val)
        if len(raw_horizons) == 1 and not isinstance(crop_override_val, (list, tuple, set)):
            raw_horizons = _auto_horizon_grid(raw_horizons[0])

    if not raw_horizons:
        raw_horizons = _auto_horizon_grid(default_h)

    horizon_list = [int(x) for x in raw_horizons if 1 <= int(x) <= cap]
    if not horizon_list:
        horizon_list = _auto_horizon_grid(default_h)

    force_include_default = bool(cfg.get("direct_multi_force_include_default_horizon", False))
    if force_include_default:
        horizon_list.append(default_h)

    serve_horizon = int(cfg.get("direct_serve_horizon", default_h))
    if crop_override_applied and bool(cfg.get("direct_use_crop_serve_horizon", True)):
        serve_horizon = int(max(horizon_list))

    if serve_horizon <= 0:
        serve_horizon = default_h
    horizon_list.append(serve_horizon)
    horizon_list = sorted(set(horizon_list))
    return horizon_list, serve_horizon, crop_override_applied


def _trend_feature_cols(all_feature_cols: List[str], cfg: dict) -> List[str]:
    if not all_feature_cols:
        return []
    prefixes = cfg.get(
        "trend_feature_prefixes",
        ["lag_", "roll_mean_", "roll_std_", "doy_", "dow_", "week_", "month_", "is_weekend"],
    )
    custom = [str(x).strip() for x in prefixes if str(x).strip()]
    if not custom:
        custom = ["lag_", "roll_mean_", "doy_", "week_", "month_", "is_weekend"]

    out = []
    for col in all_feature_cols:
        text = str(col)
        if any(text.startswith(prefix) for prefix in custom):
            out.append(text)
    if not out:
        out = [str(c) for c in all_feature_cols]
    return out


def _normalize_blend_weights(cfg: dict) -> Tuple[float, float]:
    w_main = float(cfg.get("blend_weight_main", 0.82))
    w_base = float(cfg.get("blend_weight_baseline", 0.18))
    w_main = max(0.0, min(1.0, w_main))
    w_base = max(0.0, min(1.0, w_base))
    if (w_main + w_base) <= 0.0:
        return 0.82, 0.18
    s = w_main + w_base
    return float(w_main / s), float(w_base / s)


def _resolve_blend_calibration_size(n_rows: int, cfg: dict) -> int:
    total = int(max(0, n_rows))
    if total <= 0:
        return 0
    ratio = float(cfg.get("blend_calibration_ratio", 0.16))
    min_rows = int(cfg.get("blend_calibration_min_rows", 45))
    max_rows = int(cfg.get("blend_calibration_max_rows", 160))
    min_train_rows = int(cfg.get("blend_calibration_min_train_rows", 240))
    ratio = min(max(ratio, 0.0), 0.49)
    min_rows = max(20, min_rows)
    max_rows = max(min_rows, max_rows)
    min_train_rows = max(80, min_train_rows)
    if total <= (min_train_rows + min_rows):
        return 0
    n_cal = max(min_rows, int(round(total * ratio)))
    n_cal = min(max_rows, n_cal)
    if total - n_cal < min_train_rows:
        n_cal = max(0, total - min_train_rows)
    if n_cal < min_rows:
        return 0
    return int(n_cal)


def _as_numeric_frame(X: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = X.reindex(columns=columns, fill_value=0.0).copy()
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out


def _train_hybrid_step(
    data: _PriceDataset,
    cfg: dict,
    *,
    blend_weight_main: float,
    blend_weight_baseline: float,
    horizon: int,
) -> Tuple[_HybridStepBundle, dict]:
    feature_cols = [str(c) for c in data.feature_cols]
    X_train = _as_numeric_frame(data.X_train, feature_cols)
    X_test = _as_numeric_frame(data.X_test, feature_cols)
    y_train = pd.to_numeric(data.y_train, errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(data.y_test, errors="coerce").to_numpy(dtype=float)
    trend_cols = _trend_feature_cols(feature_cols, cfg)
    X_train_trend = _as_numeric_frame(X_train, trend_cols)
    X_test_trend = _as_numeric_frame(X_test, trend_cols)

    n_cal = _resolve_blend_calibration_size(len(X_train), cfg)
    if n_cal > 0:
        split_at = int(len(X_train) - n_cal)
        X_fit = X_train.iloc[:split_at].copy()
        y_fit = y_train[:split_at].copy()
        d_fit = data.train_dates.iloc[:split_at].copy()
        X_fit_trend = X_train_trend.iloc[:split_at].copy()

        X_cal = X_train.iloc[split_at:].copy()
        y_cal = y_train[split_at:].copy()
        X_cal_trend = X_train_trend.iloc[split_at:].copy()
    else:
        X_fit = X_train
        y_fit = y_train
        d_fit = data.train_dates.copy()
        X_fit_trend = X_train_trend
        X_cal = pd.DataFrame(columns=X_train.columns)
        y_cal = np.asarray([], dtype=float)
        X_cal_trend = pd.DataFrame(columns=X_train_trend.columns)

    sw_fit = _recency_weights(d_fit, cfg)

    baseline_core = BenchmarkAnchorEnsemble(cfg).fit(X_fit, pd.Series(y_fit))
    trend_core = Ridge(alpha=float(cfg.get("trend_alpha", 1.0)))
    _fit_with_sample_weight(trend_core, X_fit_trend, y_fit, sw_fit)

    trend_fit = np.asarray(trend_core.predict(X_fit_trend), dtype=float).reshape(-1)
    residual_target_fit = y_fit - trend_fit

    residual_core = build_regressor(cfg, "lgbm")
    _fit_with_sample_weight(residual_core, X_fit, residual_target_fit, sw_fit)

    anchor_reference = np.full(len(baseline_core.component_names), 1.0 / float(len(baseline_core.component_names)), dtype=float)
    anchor_l2 = float(cfg.get("anchor_component_weight_l2", 0.03))
    blend_l2 = float(cfg.get("blend_weight_l2", 0.08))
    learned_main_weight = float(blend_weight_main)
    learned_baseline_weight = float(blend_weight_baseline)
    learned_anchor_weights = anchor_reference.copy()
    weight_learning_source = "reference_fallback"
    anchor_weight_learning_source = "reference_fallback"

    if len(y_cal) > 0 and not X_cal.empty:
        anchor_cal_components = baseline_core.predict_components(X_cal)
        raw_anchor_weights = _solve_convex_weights(
            anchor_cal_components.to_numpy(dtype=float),
            y_cal,
            reference=anchor_reference,
            l2_reg=anchor_l2,
        )
        learned_anchor_weights, anchor_weight_learning_source = _guard_anchor_component_weights(
            anchor_cal_components.to_numpy(dtype=float),
            y_cal,
            raw_anchor_weights,
            anchor_reference,
            cfg,
        )
        baseline_core.set_component_weights(learned_anchor_weights)
        anchor_cal_pred = np.asarray(baseline_core.predict(X_cal), dtype=float).reshape(-1)
        trend_cal_pred = np.asarray(trend_core.predict(X_cal_trend), dtype=float).reshape(-1)
        residual_cal_pred = np.asarray(residual_core.predict(X_cal), dtype=float).reshape(-1)
        main_cal_pred = trend_cal_pred + residual_cal_pred
        raw_blend = _solve_convex_weights(
            np.column_stack([main_cal_pred, anchor_cal_pred]),
            y_cal,
            reference=np.asarray([blend_weight_main, blend_weight_baseline], dtype=float),
            l2_reg=blend_l2,
        )
        learned_blend, weight_learning_source = _guard_blend_weights(
            main_pred=main_cal_pred,
            baseline_pred=anchor_cal_pred,
            y_true=y_cal,
            learned=raw_blend,
            reference=np.asarray([blend_weight_main, blend_weight_baseline], dtype=float),
            cfg=cfg,
        )
        learned_main_weight = float(learned_blend[0])
        learned_baseline_weight = float(learned_blend[1])

    sw_full = _recency_weights(data.train_dates, cfg)
    baseline = BenchmarkAnchorEnsemble(cfg).fit(X_train, data.y_train)
    baseline.set_component_weights(learned_anchor_weights)
    trend = Ridge(alpha=float(cfg.get("trend_alpha", 1.0)))
    _fit_with_sample_weight(trend, X_train_trend, y_train, sw_full)

    trend_train = np.asarray(trend.predict(X_train_trend), dtype=float).reshape(-1)
    residual_target = y_train - trend_train

    residual = build_regressor(cfg, "lgbm")
    _fit_with_sample_weight(residual, X_train, residual_target, sw_full)
    setattr(baseline, "calibration_rows", int(len(y_cal)))
    setattr(baseline, "weight_learning_source", weight_learning_source)
    setattr(baseline, "anchor_weight_learning_source", anchor_weight_learning_source)

    if len(y_test) == 0:
        step_metrics = {
            "horizon": int(horizon),
            "mae": None,
            "rmse": None,
            "mape": None,
            "n_test": 0,
            "baseline_mae": None,
            "baseline_rmse": None,
            "baseline_mape": None,
            "main_mae": None,
            "main_rmse": None,
            "main_mape": None,
            "blend_weight_main": float(learned_main_weight),
            "blend_weight_baseline": float(learned_baseline_weight),
            "blend_weight_reference_main": float(blend_weight_main),
            "blend_weight_reference_baseline": float(blend_weight_baseline),
            "anchor_component_names": list(baseline.component_names),
            "anchor_component_weights": baseline.get_component_weight_map(),
            "weight_learning_source": weight_learning_source,
            "anchor_weight_learning_source": anchor_weight_learning_source,
            "blend_calibration_rows": int(len(y_cal)),
        }
        bundle = _HybridStepBundle(baseline=baseline, trend=trend, residual=residual, trend_feature_cols=trend_cols)
        setattr(bundle, "blend_weight_main", float(learned_main_weight))
        setattr(bundle, "blend_weight_baseline", float(learned_baseline_weight))
        return bundle, step_metrics

    baseline_pred = np.asarray(baseline.predict(X_test), dtype=float).reshape(-1)
    trend_pred = np.asarray(trend.predict(X_test_trend), dtype=float).reshape(-1)
    residual_pred = np.asarray(residual.predict(X_test), dtype=float).reshape(-1)

    main_pred = trend_pred + residual_pred
    blended = learned_main_weight * main_pred + learned_baseline_weight * baseline_pred
    blended = np.clip(blended.astype(float), max(float(cfg.get("price_floor", PRICE_FLOOR)), PRICE_FLOOR), None)

    blended_metrics = _eval_metrics(y_test, blended, include_mape=True)
    baseline_metrics = _eval_metrics(y_test, baseline_pred, include_mape=True)
    main_metrics = _eval_metrics(y_test, main_pred, include_mape=True)
    step_metrics = {
        "horizon": int(horizon),
        "n_test": int(len(y_test)),
        "mae": blended_metrics.get("mae"),
        "rmse": blended_metrics.get("rmse"),
        "mape": blended_metrics.get("mape"),
        "baseline_mae": baseline_metrics.get("mae"),
        "baseline_rmse": baseline_metrics.get("rmse"),
        "baseline_mape": baseline_metrics.get("mape"),
        "main_mae": main_metrics.get("mae"),
        "main_rmse": main_metrics.get("rmse"),
        "main_mape": main_metrics.get("mape"),
        "blend_weight_main": float(learned_main_weight),
        "blend_weight_baseline": float(learned_baseline_weight),
        "blend_weight_reference_main": float(blend_weight_main),
        "blend_weight_reference_baseline": float(blend_weight_baseline),
        "anchor_component_names": list(baseline.component_names),
        "anchor_component_weights": baseline.get_component_weight_map(),
        "weight_learning_source": weight_learning_source,
        "anchor_weight_learning_source": anchor_weight_learning_source,
        "blend_calibration_rows": int(len(y_cal)),
    }
    bundle = _HybridStepBundle(baseline=baseline, trend=trend, residual=residual, trend_feature_cols=trend_cols)
    setattr(bundle, "blend_weight_main", float(learned_main_weight))
    setattr(bundle, "blend_weight_baseline", float(learned_baseline_weight))
    return bundle, step_metrics


def _train_direct_price_hybrid(
    df: pd.DataFrame,
    cfg: dict,
    lags: list,
    windows: list,
    horizon: int,
    backtest_days: int,
    test_ratio: Optional[float],
    validation_cutoff: Optional[str],
    strict_cutoff_split: bool,
    verbose: bool,
    label: Optional[str],
) -> PriceModelResult:
    horizon_plan, serve_horizon, crop_override_applied = _resolve_direct_horizon_plan(cfg, horizon, label)
    blend_weight_main, blend_weight_baseline = _normalize_blend_weights(cfg)

    step_models: Dict[int, _HybridStepBundle] = {}
    step_metrics: List[dict] = []
    step_data_map: Dict[int, _PriceDataset] = {}
    failed_steps: List[dict] = []

    for h in horizon_plan:
        try:
            data_h = _prepare_dataset(
                df,
                cfg,
                h,
                lags,
                windows,
                backtest_days,
                test_ratio=test_ratio,
                validation_cutoff=validation_cutoff,
                strict_cutoff_split=strict_cutoff_split,
            )
            bundle_h, metrics_h = _train_hybrid_step(
                data_h,
                cfg,
                blend_weight_main=blend_weight_main,
                blend_weight_baseline=blend_weight_baseline,
                horizon=h,
            )
            step_models[int(h)] = bundle_h
            step_metrics.append(metrics_h)
            step_data_map[int(h)] = data_h
        except Exception as exc:
            failed_steps.append({"horizon": int(h), "error": str(exc)})

    if not step_models:
        raise ValueError("no direct-horizon step model trained successfully")

    trained_horizons = sorted(step_models.keys())
    if serve_horizon not in step_models:
        serve_horizon = int(trained_horizons[-1])

    model = HybridDirectPriceModel(
        step_models=step_models,
        feature_cols=step_data_map[trained_horizons[0]].feature_cols,
        blend_weight_main=blend_weight_main,
        blend_weight_baseline=blend_weight_baseline,
        serve_horizon=serve_horizon,
        price_floor=float(cfg.get("price_floor", PRICE_FLOOR)),
    )

    summary_metric = None
    for row in step_metrics:
        if int(row.get("horizon", -1)) == int(serve_horizon):
            summary_metric = row
            break
    if summary_metric is None and step_metrics:
        summary_metric = step_metrics[-1]
    if summary_metric is None:
        summary_metric = {"mae": None, "rmse": None, "mape": None, "n_test": 0}

    metrics: Dict[str, Any] = {
        "mae": summary_metric.get("mae"),
        "rmse": summary_metric.get("rmse"),
        "mape": summary_metric.get("mape"),
        "n_test": int(summary_metric.get("n_test") or 0),
        "model_family": "anchor_ensemble_lgbm_trend_residual_v5",
        "blend_weight_main": float(summary_metric.get("blend_weight_main", blend_weight_main)),
        "blend_weight_baseline": float(summary_metric.get("blend_weight_baseline", blend_weight_baseline)),
        "blend_weight_reference_main": float(blend_weight_main),
        "blend_weight_reference_baseline": float(blend_weight_baseline),
        "trained_horizons": trained_horizons,
        "serve_horizon": int(serve_horizon),
        "horizon_metrics": step_metrics,
        "failed_horizons": failed_steps,
        "crop_horizon_override_applied": bool(crop_override_applied),
        "lightgbm_available": bool(LGBMRegressor is not None),
        "residual_model_family": "lgbm",
        "baseline_name": "benchmark_anchor_ensemble",
        "anchor_component_names": list(summary_metric.get("anchor_component_names") or []),
        "anchor_component_weights": dict(summary_metric.get("anchor_component_weights") or {}),
        "weight_learning_source": str(summary_metric.get("weight_learning_source") or "reference_fallback"),
        "anchor_weight_learning_source": str(summary_metric.get("anchor_weight_learning_source") or "reference_fallback"),
        "blend_calibration_rows": int(summary_metric.get("blend_calibration_rows") or 0),
        "trend_model_name": "ridge",
    }
    metrics.update(step_data_map[serve_horizon].split_info)

    artifacts = {
        "artifact_role": "price_direct_main",
        "artifact_name_tag": DIRECT_PRICE_ARTIFACT_TAG,
        "artifact_name_tag_legacy": LEGACY_DIRECT_PRICE_ARTIFACT_TAG,
        "model_architecture": "anchor_ensemble_lgbm_trend_residual_v5",
        "target_mode": "price",
        "feature_space": _resolve_feature_space(cfg, "price"),
        "time_raw_mode": _resolve_time_raw_mode(cfg, "price"),
        "include_raw_time_features": bool(_resolve_time_raw_mode(cfg, "price") == "raw"),
        "price_floor": float(cfg.get("price_floor", PRICE_FLOOR)),
        "model_family": "anchor_ensemble_lgbm_trend_residual_v5",
        "trained_horizons": trained_horizons,
        "serve_horizon": int(serve_horizon),
        "blend": {
            "main_weight": float(summary_metric.get("blend_weight_main", blend_weight_main)),
            "baseline_weight": float(summary_metric.get("blend_weight_baseline", blend_weight_baseline)),
            "reference_main_weight": float(blend_weight_main),
            "reference_baseline_weight": float(blend_weight_baseline),
        },
        "anchor_components": {
            "names": list(summary_metric.get("anchor_component_names") or []),
            "weights": dict(summary_metric.get("anchor_component_weights") or {}),
            "weight_learning_source": str(summary_metric.get("anchor_weight_learning_source") or "reference_fallback"),
            "calibration_rows": int(summary_metric.get("blend_calibration_rows") or 0),
        },
        "lightgbm_available": bool(LGBMRegressor is not None),
    }

    if verbose:
        name = label or "crop"
        mae_text = "NA" if metrics.get("mae") is None else f"{metrics['mae']:.4f}"
        rmse_text = "NA" if metrics.get("rmse") is None else f"{metrics['rmse']:.4f}"
        mape_text = "NA" if metrics.get("mape") is None else f"{metrics['mape']:.4f}"
        print(
            f"[price][{name}][anchor_ensemble_lgbm_trend_residual_v5] h={serve_horizon} n_test={metrics['n_test']} "
            f"mae={mae_text} rmse={rmse_text} mape={mape_text}"
        )

    return PriceModelResult(
        model=model,
        metrics=metrics,
        feature_cols=list(model.feature_cols),
        artifacts=artifacts,
    )


def _train_single_from_dataset(
    data: _PriceDataset,
    cfg: dict,
    model_name: str,
    *,
    target_mode: str,
):
    base = build_regressor(cfg, model_name)
    model = _wrap_target_transform(base, cfg, target_mode=target_mode)
    sw = _recency_weights(data.train_dates, cfg)

    _fit_with_sample_weight(model, data.X_train, data.y_train, sw)

    if data.X_test.empty or len(data.y_test) == 0:
        pred = np.asarray([], dtype=float)
        metrics = {"mae": None, "rmse": None, "mape": None, "n_test": 0}
    else:
        pred_raw = np.asarray(model.predict(data.X_test), dtype=float)
        pred = np.clip(pred_raw, 0.0, None) if target_mode == "price" else pred_raw
        metrics = _eval_metrics(data.y_test, pred, include_mape=target_mode == "price")
        metrics["n_test"] = int(len(data.y_test))
    return model, pred, metrics


def _seasonal_anchor_map(
    df: pd.DataFrame,
    *,
    train_end_date: Optional[pd.Timestamp],
    price_floor: float,
    min_samples_per_doy: int,
) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    work = df[["date", "modal_price"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["modal_price"] = pd.to_numeric(work["modal_price"], errors="coerce")
    work = work.dropna(subset=["date", "modal_price"]).sort_values("date")
    if work.empty:
        return {}
    if train_end_date is not None:
        work = work[work["date"] <= pd.Timestamp(train_end_date)]
    if work.empty:
        return {}

    work["log_price"] = np.log(np.clip(work["modal_price"].to_numpy(dtype=float), max(float(price_floor), 1e-9), None))
    work["doy"] = work["date"].dt.dayofyear.astype(int)

    out: Dict[str, float] = {}
    min_count = max(1, int(min_samples_per_doy))
    for doy, grp in work.groupby("doy"):
        vals = pd.to_numeric(grp["log_price"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size < min_count:
            continue
        out[str(int(doy))] = float(np.median(vals))
    return out


def _robust_std(arr: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if np.isfinite(mad) and mad > 0:
        return float(1.4826 * mad)
    if vals.size > 2:
        return float(np.std(vals, ddof=1))
    return float(np.std(vals))


def _conformal_horizon_scale_points(
    residual: np.ndarray,
    *,
    q: float,
    max_horizon: int,
) -> List[Dict[str, Any]]:
    resid = np.asarray(residual, dtype=float)
    resid = resid[np.isfinite(resid)]
    if resid.size < 8:
        return []

    q = float(np.clip(q, 0.5, 0.995))
    base = float(np.quantile(np.abs(resid), q))
    if not np.isfinite(base) or base <= 0:
        return []

    h_cap = max(1, int(max_horizon))
    anchors = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 150, 181, h_cap]
    horizons = sorted({int(h) for h in anchors if int(h) > 0 and int(h) <= h_cap})
    if 1 not in horizons:
        horizons.insert(0, 1)

    rows: List[Dict[str, Any]] = []
    running_scale = 1.0
    for h in horizons:
        if h <= 1:
            q_h = base
            src = "base"
        else:
            q_h = None
            if resid.size >= h:
                kernel = np.ones(int(h), dtype=float)
                sums = np.convolve(resid, kernel, mode="valid")
                sums = sums[np.isfinite(sums)]
                if sums.size >= 6:
                    q_h = float(np.quantile(np.abs(sums), q))
            if q_h is None:
                q_h = float(base * np.sqrt(float(h)))
                src = "sqrt_fallback"
            else:
                src = "rolling_sum_quantile"

        scale = float(max(1.0, q_h / max(base, 1e-12)))
        running_scale = max(running_scale, scale)
        rows.append(
            {
                "horizon_days": int(h),
                "abs_q": float(max(base, q_h)),
                "scale": float(running_scale),
                "source": src,
            }
        )
    return rows


def _return_artifacts(
    *,
    cfg: dict,
    df: pd.DataFrame,
    data: _PriceDataset,
    pred_val: np.ndarray,
) -> Dict[str, Any]:
    y_true = pd.to_numeric(data.y_test, errors="coerce").to_numpy(dtype=float)
    y_train = pd.to_numeric(data.y_train, errors="coerce").to_numpy(dtype=float)

    bias = pred_val - y_true if pred_val.size and y_true.size else np.asarray([], dtype=float)
    return_bias_mean = float(np.mean(bias)) if bias.size else 0.0
    return_bias_std = float(np.std(bias)) if bias.size else 0.0

    residual = y_true - pred_val if pred_val.size and y_true.size else np.asarray([], dtype=float)
    conformal_q = float(cfg.get("conformal_abs_quantile", 0.9))
    conformal_q = float(np.clip(conformal_q, 0.5, 0.995))
    conformal_abs = float(np.quantile(np.abs(residual), conformal_q)) if residual.size else 0.0
    conf_horizon_cap = max(1, int(cfg.get("conformal_horizon_scale_cap_days", 181)))
    horizon_scale_points = _conformal_horizon_scale_points(
        residual,
        q=conformal_q,
        max_horizon=conf_horizon_cap,
    )
    local_vol_reference = _robust_std(y_train[-min(180, y_train.size) :]) if y_train.size else 0.0
    if local_vol_reference <= 0:
        local_vol_reference = _robust_std(residual)
    if local_vol_reference <= 0:
        local_vol_reference = 0.01

    clip_q = float(cfg.get("return_clip_quantile", 0.98))
    clip_q = float(np.clip(clip_q, 0.8, 0.999))
    clip_sf = float(cfg.get("return_clip_safety_factor", 1.2))
    clip_sf = float(np.clip(clip_sf, 1.0, 3.0))
    abs_train = np.abs(y_train[np.isfinite(y_train)]) if y_train.size else np.asarray([], dtype=float)
    abs_ref = float(np.quantile(abs_train, clip_q)) if abs_train.size else 0.03
    r_max = float(np.clip(abs_ref * clip_sf, 0.003, 0.35))

    train_end = pd.Timestamp(data.train_dates.max()) if len(data.train_dates) else None
    price_floor = float(cfg.get("price_floor", PRICE_FLOOR))
    min_samples_per_doy = max(1, int(cfg.get("seasonal_anchor_min_samples_per_doy", 2)))
    seasonal_map = _seasonal_anchor_map(
        df,
        train_end_date=train_end,
        price_floor=price_floor,
        min_samples_per_doy=min_samples_per_doy,
    )

    return {
        "artifact_role": "price_recursive_step",
        "artifact_name_tag": "recursive_step_model",
        "target_mode": "log_return",
        "feature_space": "log_price",
        "time_raw_mode": "none",
        "include_raw_time_features": False,
        "price_floor": float(price_floor),
        "bias_correction": {
            "enabled": bool(cfg.get("enable_bias_correction", True)),
            "return_bias_mean": float(return_bias_mean),
            "return_bias_std": float(return_bias_std),
        },
        "diagnostics": {
            "return_bias_mean": float(return_bias_mean),
            "return_bias_std": float(return_bias_std),
        },
        "clip": {
            "r_max": float(r_max),
            "quantile": float(clip_q),
            "safety_factor": float(clip_sf),
        },
        "conformal": {
            "enabled": bool(cfg.get("enable_conformal_interval", True)),
            "abs_error_quantile": float(conformal_q),
            "abs_q": float(conformal_abs),
            "horizon_scale_points": horizon_scale_points,
            "horizon_scale_cap_days": int(conf_horizon_cap),
            "local_vol_reference": float(local_vol_reference),
            "horizon_scale_source": (
                "rolling_sum_quantile" if horizon_scale_points else "none"
            ),
        },
        "seasonal_anchor": {
            "enabled": bool(cfg.get("enable_seasonal_anchor", True)),
            "min_samples_per_doy": int(min_samples_per_doy),
            "dayofyear_log_price_median": seasonal_map,
        },
    }


def _legacy_train_one_crop(
    df: pd.DataFrame,
    cfg: dict,
    lags: list,
    windows: list,
    horizon: int,
    backtest_days: int,
    test_ratio: Optional[float] = None,
    validation_cutoff: Optional[str] = None,
    strict_cutoff_split: bool = True,
    verbose: bool = False,
    label: Optional[str] = None,
) -> PriceModelResult:
    target_mode = _resolve_target_mode(cfg)
    data = _prepare_dataset(
        df,
        cfg,
        horizon,
        lags,
        windows,
        backtest_days,
        test_ratio=test_ratio,
        validation_cutoff=validation_cutoff,
        strict_cutoff_split=strict_cutoff_split,
    )

    reg = cfg.get("regressor", "hgb")
    if reg != "ensemble":
        model, pred, metrics = _train_single_from_dataset(data, cfg, reg, target_mode=target_mode)
        metrics.update(data.split_info)
        artifacts = None
        if target_mode == "log_return":
            artifacts = _return_artifacts(cfg=cfg, df=df, data=data, pred_val=np.asarray(pred, dtype=float))
            metrics["return_bias_mean"] = float(artifacts["diagnostics"].get("return_bias_mean", 0.0))
            metrics["return_bias_std"] = float(artifacts["diagnostics"].get("return_bias_std", 0.0))
        if verbose:
            name = label or "crop"
            mae_text = "NA" if metrics.get("mae") is None else f"{metrics['mae']:.4f}"
            rmse_text = "NA" if metrics.get("rmse") is None else f"{metrics['rmse']:.4f}"
            mape_text = "NA" if metrics.get("mape") is None else f"{metrics['mape']:.4f}"
            print(
                f"[price][{name}][{reg}] n_test={metrics['n_test']} "
                f"mae={mae_text} rmse={rmse_text} mape={mape_text}"
            )
        return PriceModelResult(model=model, metrics=metrics, feature_cols=data.feature_cols, artifacts=artifacts)

    members = cfg.get("ensemble_members", ["hgb", "rf", "etr"])
    models = []
    metrics_list = []

    for name in members:
        model, pred, metrics = _train_single_from_dataset(data, cfg, name, target_mode=target_mode)
        models.append(model)
        metrics_list.append({"name": name, **metrics})
    w, ensemble_detail = _select_ensemble_weights(metrics_list, cfg)
    ens = EnsembleModel(
        models=models,
        weights=w,
        feature_cols=data.feature_cols,
        prediction_space="return" if target_mode == "log_return" else "price",
    )

    if data.X_test.empty or len(data.y_test) == 0:
        ens_pred = np.asarray([], dtype=float)
        metrics = {"mae": None, "rmse": None, "mape": None, "n_test": 0}
    else:
        ens_pred = np.asarray(ens.predict(data.X_test), dtype=float)
        metrics = _eval_metrics(data.y_test, ens_pred, include_mape=target_mode == "price")
        metrics["n_test"] = int(len(data.y_test))

    metrics["ensemble_members"] = metrics_list
    metrics["ensemble_weights"] = w
    metrics.update(ensemble_detail)
    metrics.update(data.split_info)

    artifacts = None
    if target_mode == "log_return":
        artifacts = _return_artifacts(cfg=cfg, df=df, data=data, pred_val=ens_pred)
        metrics["return_bias_mean"] = float(artifacts["diagnostics"].get("return_bias_mean", 0.0))
        metrics["return_bias_std"] = float(artifacts["diagnostics"].get("return_bias_std", 0.0))

    if verbose:
        name = label or "crop"
        mae_text = "NA" if metrics.get("mae") is None else f"{metrics['mae']:.4f}"
        rmse_text = "NA" if metrics.get("rmse") is None else f"{metrics['rmse']:.4f}"
        mape_text = "NA" if metrics.get("mape") is None else f"{metrics['mape']:.4f}"
        print(
            f"[price][{name}][ensemble] n_test={metrics['n_test']} "
            f"mae={mae_text} rmse={rmse_text} mape={mape_text}"
        )

    return PriceModelResult(model=ens, metrics=metrics, feature_cols=data.feature_cols, artifacts=artifacts)


def train_one_crop(
    df: pd.DataFrame,
    cfg: dict,
    lags: list,
    windows: list,
    horizon: int,
    backtest_days: int,
    test_ratio: Optional[float] = None,
    validation_cutoff: Optional[str] = None,
    strict_cutoff_split: bool = True,
    verbose: bool = False,
    label: Optional[str] = None,
) -> PriceModelResult:
    target_mode = _resolve_target_mode(cfg)
    use_hybrid_direct = _use_direct_trend_residual_v3(cfg)
    fallback_to_legacy = _direct_trend_residual_fallback_to_legacy(cfg)

    if target_mode == "price" and use_hybrid_direct:
        try:
            return _train_direct_price_hybrid(
                df=df,
                cfg=cfg,
                lags=lags,
                windows=windows,
                horizon=horizon,
                backtest_days=backtest_days,
                test_ratio=test_ratio,
                validation_cutoff=validation_cutoff,
                strict_cutoff_split=strict_cutoff_split,
                verbose=verbose,
                label=label,
            )
        except Exception:
            if not fallback_to_legacy:
                raise

    return _legacy_train_one_crop(
        df=df,
        cfg=cfg,
        lags=lags,
        windows=windows,
        horizon=horizon,
        backtest_days=backtest_days,
        test_ratio=test_ratio,
        validation_cutoff=validation_cutoff,
        strict_cutoff_split=strict_cutoff_split,
        verbose=verbose,
        label=label,
    )
