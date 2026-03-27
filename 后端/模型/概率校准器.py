import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
try:
    # sklearn >= 1.8
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover - executed on older sklearn versions
    FrozenEstimator = None

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False


@dataclass
class CalibratorResult:
    model: object
    metrics: Dict[str, float]
    feature_cols: List[str]
    n_train: int
    n_valid: int
    train_range: Tuple[str, str]
    valid_range: Tuple[str, str]
    diagnostics: Dict[str, Any]


def _infer_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_key_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {
        "date": _infer_col(df, ["date", "Date", "dt", "datetime"]),
        "env_id": _infer_col(df, ["env_id", "env", "environment_id", "site_id"]),
        "crop": _infer_col(df, ["crop_name", "crop", "label", "crop_id"]),
        "price_real": _infer_col(df, ["price_real", "price", "real_price", "modal_price_real"]),
        "yield_real": _infer_col(df, ["yield_real", "yield", "real_yield", "yield_quintal_per_hectare"]),
        "cost_real": _infer_col(df, ["cost_real", "cost", "real_cost", "india_cost_wavg_sample"]),
        "prob_env": _infer_col(df, ["prob_env", "env_prob"]),
        "profit_pred": _infer_col(df, ["profit_pred", "profit", "profit_est"]),
        "price_pred": _infer_col(df, ["price_pred", "pred_price"]),
        "yield_pred": _infer_col(df, ["yield_pred", "pred_yield"]),
        "cost_pred": _infer_col(df, ["cost_pred", "pred_cost"]),
        "risk_score": _infer_col(df, ["risk_score", "risk"]),
        "score_total": _infer_col(df, ["score_total", "score"]),
        "volatility": _infer_col(df, ["volatility", "price_volatility"]),
        "env_confidence": _infer_col(df, ["env_confidence", "confidence"]),
        "ood_flag": _infer_col(df, ["ood_flag", "is_ood", "ood"]),
    }


def compute_profit_real(df: pd.DataFrame, cols: Dict[str, str]) -> pd.Series:
    return (
        pd.to_numeric(df[cols["price_real"]], errors="coerce")
        * pd.to_numeric(df[cols["yield_real"]], errors="coerce")
        - pd.to_numeric(df[cols["cost_real"]], errors="coerce")
    )


def build_labels(df: pd.DataFrame, cols: Dict[str, str], topk: int = 1) -> pd.Series:
    key = [cols["env_id"], cols["date"]]
    df = df.copy()
    df["profit_real"] = compute_profit_real(df, cols)

    df["rank"] = df.groupby(key)["profit_real"].rank(ascending=False, method="first")
    if topk <= 1:
        return (df["rank"] == 1).astype(int)
    return (df["rank"] <= topk).astype(int)


def _encode_confidence(conf_val) -> float:
    s = str(conf_val).strip().lower()
    if s in ["high", "高"]:
        return 2.0
    if s in ["mid", "中"]:
        return 1.0
    if s in ["low", "低"]:
        return 0.0
    try:
        return float(conf_val)
    except Exception:
        return 0.0


def build_features(df: pd.DataFrame, cols: Dict[str, str], cfg: dict) -> Tuple[pd.DataFrame, List[str]]:
    feats = {}

    def add_if(name, col_key):
        col = cols.get(col_key)
        if col and col in df.columns:
            feats[name] = pd.to_numeric(df[col], errors="coerce")

    if cfg.get("use_prob_env", True):
        add_if("prob_env", "prob_env")
    if cfg.get("use_profit_pred", True):
        add_if("profit_pred", "profit_pred")
    if cfg.get("use_price_pred", True):
        add_if("price_pred", "price_pred")
    if cfg.get("use_yield_pred", True):
        add_if("yield_pred", "yield_pred")
    if cfg.get("use_cost_pred", True):
        add_if("cost_pred", "cost_pred")
    if cfg.get("use_risk_score", True):
        add_if("risk_score", "risk_score")
    if cfg.get("use_score_total", True):
        add_if("score_total", "score_total")
    if cfg.get("use_volatility", True):
        add_if("volatility", "volatility")
    if cfg.get("use_ood_flag", True):
        add_if("ood_flag", "ood_flag")
    if cfg.get("use_env_confidence", True):
        col = cols.get("env_confidence")
        if col and col in df.columns:
            feats["env_confidence"] = df[col].apply(_encode_confidence)

    X = pd.DataFrame(feats)
    X = X.fillna(0.0)
    return X, list(X.columns)


def walk_forward_splits(dates: pd.Series, initial_train_ratio: float, step_size: int, horizon: int):
    d = pd.to_datetime(dates)
    uniq_dates = np.array(sorted(d.dt.normalize().unique()))
    n = len(uniq_dates)
    if n <= 2:
        return
    start = max(1, int(n * initial_train_ratio))
    while start + horizon <= n:
        train_dates = set(uniq_dates[:start])
        test_dates = set(uniq_dates[start:start + horizon])
        train_idx = np.where(d.dt.normalize().isin(train_dates))[0]
        test_idx = np.where(d.dt.normalize().isin(test_dates))[0]
        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx, test_idx
        start += step_size


def _ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def _build_base_model(cfg: dict):
    class_weight = cfg.get("class_weight", "balanced")
    if HAS_LGB:
        return lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=cfg.get("seed", 42),
            class_weight=class_weight if class_weight != "balanced" else None,
        )

    return LogisticRegression(
        max_iter=2000,
        class_weight=class_weight if class_weight != "balanced" else "balanced",
        random_state=cfg.get("seed", 42),
    )


def _fit_base_model(base, X_fit, y_fit, sample_weight: Optional[np.ndarray] = None):
    if sample_weight is None:
        base.fit(X_fit, y_fit)
        return base
    try:
        base.fit(X_fit, y_fit, sample_weight=np.asarray(sample_weight, dtype=float))
    except TypeError:
        base.fit(X_fit, y_fit)
    return base


def _fit_calibrator_prefit(base, X_cal, y_cal, method: str, sample_weight: Optional[np.ndarray] = None):
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=float)

    # sklearn >=1.8 removed `cv='prefit'`; use FrozenEstimator when available.
    if FrozenEstimator is not None:
        try:
            calibrator = CalibratedClassifierCV(FrozenEstimator(base), method=method, cv=None)
            try:
                calibrator.fit(X_cal, y_cal, **fit_kwargs)
            except TypeError:
                calibrator.fit(X_cal, y_cal)
            return calibrator
        except Exception:
            pass

    # Backward compatibility for older sklearn where prefit is still supported.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r"The `cv='prefit'` option is deprecated.*",
        )
        calibrator = CalibratedClassifierCV(base, method=method, cv="prefit")
        try:
            calibrator.fit(X_cal, y_cal, **fit_kwargs)
        except TypeError:
            calibrator.fit(X_cal, y_cal)
        return calibrator


def _split_fit_cal(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    min_cal: int = 50,
):
    if len(X_train) < min_cal + 30:
        return None, None, None, None, None, None

    sw = None
    if sample_weight is not None:
        sw = pd.to_numeric(sample_weight, errors="coerce").fillna(1.0)
        sw = sw.reindex(X_train.index).fillna(1.0)

    candidates = [
        max(min_cal, int(len(X_train) * 0.2)),
        max(min_cal, int(len(X_train) * 0.25)),
        max(min_cal, int(len(X_train) * 0.3)),
    ]
    for n_cal in candidates:
        if len(X_train) <= n_cal + 10:
            continue
        X_fit, y_fit = X_train.iloc[:-n_cal], y_train.iloc[:-n_cal]
        X_cal, y_cal = X_train.iloc[-n_cal:], y_train.iloc[-n_cal:]
        w_fit = None if sw is None else sw.iloc[:-n_cal].to_numpy(dtype=float)
        w_cal = None if sw is None else sw.iloc[-n_cal:].to_numpy(dtype=float)
        if y_fit.nunique() >= 2 and y_cal.nunique() >= 2:
            return X_fit, y_fit, X_cal, y_cal, w_fit, w_cal
    return None, None, None, None, None, None


def _build_group_margin_frame(df: pd.DataFrame, cols: Dict[str, str]) -> pd.DataFrame:
    key = [cols["env_id"], cols["date"]]
    if any(k not in df.columns for k in key):
        return pd.DataFrame(columns=key + ["group_rel_margin"])

    work = df[key].copy()
    work["profit_real"] = compute_profit_real(df, cols)
    work = work.dropna(subset=["profit_real"])
    if work.empty:
        return pd.DataFrame(columns=key + ["group_rel_margin"])

    rows: List[Dict[str, Any]] = []
    for group_key, grp in work.groupby(key, dropna=False):
        vals = (
            pd.to_numeric(grp["profit_real"], errors="coerce")
            .dropna()
            .sort_values(ascending=False)
            .to_numpy(dtype=float)
        )
        rel_margin = 0.0
        if vals.size >= 2:
            top = float(vals[0])
            second = float(vals[1])
            rel_margin = float((top - second) / max(abs(top), 1e-6))
        if isinstance(group_key, tuple):
            row = {key[0]: group_key[0], key[1]: group_key[1]}
        else:
            row = {key[0]: group_key, key[1]: None}
        row["group_rel_margin"] = rel_margin
        rows.append(row)
    return pd.DataFrame(rows)


def _build_difficulty_sample_weight(
    df: pd.DataFrame,
    cols: Dict[str, str],
    cfg: dict,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    weights = np.ones(len(df), dtype=float)
    if not bool(cfg.get("downweight_easy_groups", True)):
        return weights, {"enabled": False, "reason": "disabled_by_config"}

    key = [cols["env_id"], cols["date"]]
    margin_df = _build_group_margin_frame(df, cols)
    if margin_df.empty:
        return weights, {"enabled": False, "reason": "no_margin_groups"}

    merged = df[key].copy().merge(margin_df, on=key, how="left")
    margin = pd.to_numeric(merged["group_rel_margin"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ref = max(float(cfg.get("easy_group_rel_margin_ref", 0.20)), 1e-6)
    floor = float(np.clip(float(cfg.get("easy_group_weight_floor", 0.35)), 0.05, 1.0))

    weights = 1.0 / (1.0 + margin / ref)
    weights = np.clip(weights, floor, 1.0)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
    weights = weights / np.mean(weights)

    return weights, {
        "enabled": True,
        "method": "inverse_rel_margin",
        "margin_reference": float(ref),
        "weight_floor": float(floor),
        "weight_mean": float(np.mean(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "group_rel_margin_median": float(np.median(margin)) if margin.size else None,
        "group_rel_margin_p90": float(np.quantile(margin, 0.9)) if margin.size else None,
    }


def _sanitize_feature_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    if X is None or X.empty:
        return X, [], {"applied": False, "reason": "empty_feature_frame"}

    auto_drop = bool(cfg.get("auto_drop_suspicious_features", True))
    auc_threshold = float(cfg.get("suspicious_feature_auc_threshold", 0.995))
    min_kept = max(1, int(cfg.get("min_feature_count_after_sanitization", 2)))
    manual_drop = {
        str(name).strip()
        for name in cfg.get("drop_feature_names", [])
        if str(name).strip()
    }

    drop_reasons: List[Dict[str, Any]] = []
    top_auc_features: List[Dict[str, Any]] = []
    drop_names: List[str] = []

    for feature in list(X.columns):
        auc = _max_directional_auc(y, X[feature])
        if auc is not None:
            top_auc_features.append({"feature": str(feature), "auc": float(auc)})
        reason = None
        if feature in manual_drop:
            reason = "manual_blocklist"
        elif auto_drop and auc is not None and auc >= auc_threshold:
            reason = "proxy_auc_too_high"
        if reason is not None:
            drop_names.append(str(feature))
            row = {"feature": str(feature), "reason": reason}
            if auc is not None:
                row["auc"] = float(auc)
                row["threshold"] = float(auc_threshold)
            drop_reasons.append(row)

    top_auc_features.sort(key=lambda row: row["auc"], reverse=True)
    kept = [str(c) for c in X.columns if str(c) not in set(drop_names)]
    applied = bool(drop_reasons) and len(kept) >= int(min_kept)
    if not applied:
        kept = [str(c) for c in X.columns]

    return X[kept].copy(), kept, {
        "applied": bool(applied),
        "auto_drop_enabled": bool(auto_drop),
        "auc_threshold": float(auc_threshold),
        "raw_feature_count": int(X.shape[1]),
        "selected_feature_count": int(len(kept)),
        "selected_feature_cols": kept,
        "dropped_features": drop_reasons if applied else [],
        "suppressed_drop_due_to_min_feature_count": bool(drop_reasons) and not applied,
        "top_auc_features": top_auc_features[:5],
    }


def _easy_label_profile(df: pd.DataFrame, cols: Dict[str, str]) -> Dict[str, Any]:
    margin_df = _build_group_margin_frame(df, cols)
    if margin_df.empty:
        return {"available": False}
    margins = pd.to_numeric(margin_df["group_rel_margin"], errors="coerce").dropna().to_numpy(dtype=float)
    if margins.size == 0:
        return {"available": False}
    return {
        "available": True,
        "group_rel_margin_median": float(np.median(margins)),
        "group_rel_margin_p90": float(np.quantile(margins, 0.9)),
        "group_rel_margin_ge_0_2_ratio": float((margins >= 0.2).mean()),
    }


def _resolve_calibration_method(
    cfg: dict,
    n_rows: int,
    easy_label_profile: Optional[Dict[str, Any]] = None,
    feature_sanitization: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    requested = str(cfg.get("calibration_method", "isotonic")).strip().lower()
    if requested != "auto":
        return requested, {"requested": requested, "resolved": requested, "reason": "explicit_config"}

    resolution = {
        "requested": "auto",
        "resolved": None,
        "reason": None,
        "min_samples_isotonic": int(cfg.get("min_samples_isotonic", 2000)),
    }

    easy = easy_label_profile or {}
    prefer_sigmoid = bool(cfg.get("prefer_sigmoid_on_easy_labels", True))
    median_threshold = float(cfg.get("sigmoid_easy_label_margin_median_threshold", 0.6))
    easy_ratio_threshold = float(cfg.get("sigmoid_easy_label_ratio_threshold", 0.95))
    rel_margin_median = _safe_float(easy.get("group_rel_margin_median"))
    easy_ratio = _safe_float(easy.get("group_rel_margin_ge_0_2_ratio"))
    suspicious_drop = bool((feature_sanitization or {}).get("applied"))

    if prefer_sigmoid and (
        suspicious_drop
        or (rel_margin_median is not None and rel_margin_median >= median_threshold)
        or (easy_ratio is not None and easy_ratio >= easy_ratio_threshold)
    ):
        resolution["resolved"] = "sigmoid"
        resolution["reason"] = "easy_label_guard"
        resolution["guard_inputs"] = {
            "suspicious_features_dropped": suspicious_drop,
            "group_rel_margin_median": rel_margin_median,
            "group_rel_margin_ge_0_2_ratio": easy_ratio,
        }
        return "sigmoid", resolution

    resolved = "isotonic" if int(n_rows) >= int(cfg.get("min_samples_isotonic", 2000)) else "sigmoid"
    resolution["resolved"] = resolved
    resolution["reason"] = "size_rule"
    return resolved, resolution


def _safe_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _date_range(values: pd.Series) -> Tuple[str, str]:
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return "", ""
    return str(dates.min().date()), str(dates.max().date())


def _group_count(df: pd.DataFrame, cols: Dict[str, str]) -> int:
    keys = [cols["env_id"], cols["date"]]
    if any(k not in df.columns for k in keys):
        return 0
    return int(df[keys].drop_duplicates().shape[0])


def _max_directional_auc(y_true: pd.Series, values: pd.Series) -> Optional[float]:
    y_arr = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    x_arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(x_arr)
    if int(mask.sum()) < 20:
        return None
    y_masked = y_arr[mask]
    if np.unique(y_masked).size < 2:
        return None
    try:
        auc = float(roc_auc_score(y_masked, x_arr[mask]))
    except Exception:
        return None
    if not np.isfinite(auc):
        return None
    return float(max(auc, 1.0 - auc))


def _pred_real_match_ratio(df: pd.DataFrame, pred_col: str, real_col: str) -> Optional[float]:
    if pred_col not in df.columns or real_col not in df.columns:
        return None
    pred = pd.to_numeric(df[pred_col], errors="coerce")
    real = pd.to_numeric(df[real_col], errors="coerce")
    mask = pred.notna() & real.notna()
    if int(mask.sum()) == 0:
        return None
    pred_arr = pred.loc[mask].to_numpy(dtype=float)
    real_arr = real.loc[mask].to_numpy(dtype=float)
    return float(np.isclose(pred_arr, real_arr, rtol=1e-9, atol=1e-9).mean())


def _dedupe_alerts(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for alert in alerts:
        key = (
            str(alert.get("code", "")),
            str(alert.get("feature", "")),
            str(alert.get("pair", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(alert)
    return out


def _build_diagnostics(
    df: pd.DataFrame,
    cols: Dict[str, str],
    y: pd.Series,
    X: pd.DataFrame,
    feature_cols: List[str],
    metrics_all: List[Dict[str, Any]],
    selected_feature_cols: Optional[List[str]] = None,
    feature_sanitization: Optional[Dict[str, Any]] = None,
    training_weight_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    key = [cols["env_id"], cols["date"]]
    group_sizes = df.groupby(key).size()
    positive_per_group = y.groupby([df[cols["env_id"]], df[cols["date"]]]).sum()

    label_profile: Dict[str, Any] = {
        "rows": int(len(df)),
        "groups": int(group_sizes.shape[0]),
        "positive_rate": float(y.mean()) if len(y) else None,
        "avg_candidates_per_group": float(group_sizes.mean()) if not group_sizes.empty else None,
        "min_candidates_per_group": int(group_sizes.min()) if not group_sizes.empty else None,
        "max_candidates_per_group": int(group_sizes.max()) if not group_sizes.empty else None,
        "avg_positives_per_group": float(positive_per_group.mean()) if not positive_per_group.empty else None,
    }

    work = df[key].copy()
    work["profit_real"] = compute_profit_real(df, cols)
    margin_abs: List[float] = []
    margin_rel: List[float] = []
    for _, grp in work.groupby(key):
        vals = pd.to_numeric(grp["profit_real"], errors="coerce").dropna().sort_values(ascending=False).to_numpy(dtype=float)
        if vals.size < 2:
            continue
        top = float(vals[0])
        second = float(vals[1])
        diff = float(top - second)
        margin_abs.append(diff)
        margin_rel.append(diff / max(abs(top), 1e-6))

    margin_abs_arr = np.asarray(margin_abs, dtype=float)
    margin_rel_arr = np.asarray(margin_rel, dtype=float)
    profit_margin: Dict[str, Any] = {
        "groups_with_margin": int(margin_abs_arr.size),
        "abs_median": float(np.median(margin_abs_arr)) if margin_abs_arr.size else None,
        "abs_p90": float(np.quantile(margin_abs_arr, 0.9)) if margin_abs_arr.size else None,
        "rel_median": float(np.median(margin_rel_arr)) if margin_rel_arr.size else None,
        "rel_p90": float(np.quantile(margin_rel_arr, 0.9)) if margin_rel_arr.size else None,
        "rel_ge_0_2_ratio": float((margin_rel_arr >= 0.2).mean()) if margin_rel_arr.size else None,
    }

    feature_proxy_auc: Dict[str, float] = {}
    strongest_features: List[Dict[str, Any]] = []
    for feature in feature_cols:
        if feature not in X.columns:
            continue
        auc = _max_directional_auc(y, X[feature])
        if auc is None:
            continue
        feature_proxy_auc[str(feature)] = float(auc)
        strongest_features.append({"feature": str(feature), "auc": float(auc)})
    strongest_features.sort(key=lambda row: row["auc"], reverse=True)

    real_pred_match_ratio: Dict[str, float] = {}
    for pred_key, real_key in [
        ("price_pred", "price_real"),
        ("yield_pred", "yield_real"),
        ("cost_pred", "cost_real"),
    ]:
        pred_col = cols.get(pred_key)
        real_col = cols.get(real_key)
        if not pred_col or not real_col:
            continue
        ratio = _pred_real_match_ratio(df, pred_col, real_col)
        if ratio is not None:
            real_pred_match_ratio[f"{pred_key}_vs_{real_key}"] = float(ratio)

    near_perfect_fold_count = 0
    extreme_prob_ratios: List[float] = []
    for row in metrics_all:
        ll = _safe_float(row.get("logloss"))
        brier = _safe_float(row.get("brier"))
        ece = _safe_float(row.get("ece"))
        extreme = _safe_float(row.get("prob_extreme_ratio"))
        if extreme is not None:
            extreme_prob_ratios.append(extreme)
        if ll is None or brier is None or ece is None:
            continue
        if ll <= 1e-4 and brier <= 1e-6 and ece <= 1e-4:
            near_perfect_fold_count += 1

    n_folds = int(len(metrics_all))
    logloss_vals = [float(row["logloss"]) for row in metrics_all if _safe_float(row.get("logloss")) is not None]
    fold_behavior: Dict[str, Any] = {
        "n_folds": n_folds,
        "near_perfect_fold_count": int(near_perfect_fold_count),
        "near_perfect_fold_ratio": float(near_perfect_fold_count / max(n_folds, 1)),
        "mean_prob_extreme_ratio": float(np.mean(extreme_prob_ratios)) if extreme_prob_ratios else None,
        "max_logloss": float(max(logloss_vals)) if logloss_vals else None,
        "min_logloss": float(min(logloss_vals)) if logloss_vals else None,
    }

    alerts: List[Dict[str, Any]] = []
    rel_margin_median = _safe_float(profit_margin.get("rel_median"))
    rel_margin_easy_ratio = _safe_float(profit_margin.get("rel_ge_0_2_ratio"))
    if rel_margin_median is not None and rel_margin_median >= 0.6:
        alerts.append(
            {
                "severity": "warning",
                "code": "top1_margin_too_large",
                "value": rel_margin_median,
                "threshold": 0.6,
            }
        )
    if rel_margin_easy_ratio is not None and rel_margin_easy_ratio >= 0.95:
        alerts.append(
            {
                "severity": "warning",
                "code": "labels_almost_always_easy",
                "value": rel_margin_easy_ratio,
                "threshold": 0.95,
            }
        )
    for row in strongest_features[:3]:
        auc = _safe_float(row.get("auc"))
        if auc is None or auc < 0.995:
            continue
        alerts.append(
            {
                "severity": "warning",
                "code": "feature_proxy_near_perfect",
                "feature": row["feature"],
                "value": auc,
                "threshold": 0.995,
            }
        )
    for pair, ratio in real_pred_match_ratio.items():
        if ratio < 0.15:
            continue
        alerts.append(
            {
                "severity": "warning",
                "code": "pred_matches_real_too_often",
                "pair": pair,
                "value": float(ratio),
                "threshold": 0.15,
            }
        )
    near_perfect_fold_ratio = _safe_float(fold_behavior.get("near_perfect_fold_ratio"))
    if near_perfect_fold_ratio is not None and near_perfect_fold_ratio >= 0.3:
        alerts.append(
            {
                "severity": "warning",
                "code": "many_near_perfect_folds",
                "value": near_perfect_fold_ratio,
                "threshold": 0.3,
            }
        )
    mean_prob_extreme_ratio = _safe_float(fold_behavior.get("mean_prob_extreme_ratio"))
    if mean_prob_extreme_ratio is not None and mean_prob_extreme_ratio >= 0.9:
        alerts.append(
            {
                "severity": "warning",
                "code": "probabilities_saturate_to_edges",
                "value": mean_prob_extreme_ratio,
                "threshold": 0.9,
            }
        )
    if feature_sanitization and feature_sanitization.get("applied") and feature_sanitization.get("dropped_features"):
        alerts.append(
            {
                "severity": "info",
                "code": "suspicious_features_dropped",
                "value": int(len(feature_sanitization.get("dropped_features") or [])),
            }
        )

    return {
        "label_profile": label_profile,
        "profit_margin": profit_margin,
        "feature_proxy_auc": feature_proxy_auc,
        "strongest_feature_proxy": strongest_features[:5],
        "real_pred_match_ratio": real_pred_match_ratio,
        "fold_behavior": fold_behavior,
        "selected_feature_cols": list(selected_feature_cols or feature_cols),
        "feature_sanitization": feature_sanitization or {},
        "training_weight_summary": training_weight_summary or {},
        "alerts": _dedupe_alerts(alerts),
    }


def train_calibrator(df: pd.DataFrame, cfg: dict) -> CalibratorResult:
    cols = infer_key_columns(df)
    required = ["date", "env_id", "crop", "price_real", "yield_real", "cost_real"]
    missing = [r for r in required if not cols.get(r)]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = df.copy()
    df[cols["date"]] = pd.to_datetime(df[cols["date"]])
    df = df.sort_values(cols["date"]).reset_index(drop=True)

    y = build_labels(df, cols, topk=int(cfg.get("topk", 1)))
    X_raw, raw_feature_cols = build_features(df, cols, cfg)
    X, feature_cols, feature_sanitization = _sanitize_feature_matrix(X_raw, y, cfg)
    sample_weight, training_weight_summary = _build_difficulty_sample_weight(df, cols, cfg)
    sample_weight_series = pd.Series(sample_weight, index=X.index, dtype=float)
    easy_label_profile = _easy_label_profile(df, cols)

    wf = cfg.get("walk_forward", {})
    if wf.get("enable", True):
        splits = list(walk_forward_splits(
            df[cols["date"]],
            wf.get("initial_train_ratio", 0.6),
            int(wf.get("step_size", 30)),
            int(wf.get("horizon", 30)),
        ))
    else:
        n = len(df)
        n_test = max(10, int(n * 0.2))
        splits = [(np.arange(0, n - n_test), np.arange(n - n_test, n))]

    metrics_all = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        base = _build_base_model(cfg)

        # calibration method selection
        method, _ = _resolve_calibration_method(
            cfg,
            n_rows=len(X_train),
            easy_label_profile=easy_label_profile,
            feature_sanitization=feature_sanitization,
        )

        # split tail for calibration
        X_fit, y_fit, X_cal, y_cal, w_fit, w_cal = _split_fit_cal(
            X_train,
            y_train,
            sample_weight=sample_weight_series.iloc[train_idx],
            min_cal=50,
        )
        if X_fit is None:
            continue

        _fit_base_model(base, X_fit, y_fit, sample_weight=w_fit)
        calibrator = _fit_calibrator_prefit(base, X_cal, y_cal, method, sample_weight=w_cal)

        y_prob = calibrator.predict_proba(X_test)[:, 1]

        ll = log_loss(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        ece = _ece_score(y_test.values, y_prob)
        prob_extreme_ratio = float(((y_prob <= 1e-6) | (y_prob >= 1.0 - 1e-6)).mean()) if len(y_prob) else 0.0
        train_frame = df.iloc[train_idx]
        test_frame = df.iloc[test_idx]

        metrics_all.append({
            "logloss": float(ll),
            "brier": float(brier),
            "ece": float(ece),
            "prob_extreme_ratio": float(prob_extreme_ratio),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "train_groups": _group_count(train_frame, cols),
            "test_groups": _group_count(test_frame, cols),
            "train_range": _date_range(train_frame[cols["date"]]),
            "test_range": _date_range(test_frame[cols["date"]]),
        })

    if not metrics_all:
        raise ValueError("no valid walk-forward splits for calibrator (class imbalance or too few samples)")

    # train final model on full data with tail calibration
    base = _build_base_model(cfg)
    method, method_resolution = _resolve_calibration_method(
        cfg,
        n_rows=len(X),
        easy_label_profile=easy_label_profile,
        feature_sanitization=feature_sanitization,
    )

    X_fit, y_fit, X_cal, y_cal, w_fit, w_cal = _split_fit_cal(
        X,
        y,
        sample_weight=sample_weight_series,
        min_cal=100,
    )
    if X_fit is None:
        raise ValueError("cannot build final fit/cal split for calibrator")

    _fit_base_model(base, X_fit, y_fit, sample_weight=w_fit)
    calibrator = _fit_calibrator_prefit(base, X_cal, y_cal, method, sample_weight=w_cal)

    metrics = {
        "cv": metrics_all,
        "cv_mean_logloss": float(np.mean([m["logloss"] for m in metrics_all])),
        "cv_mean_brier": float(np.mean([m["brier"] for m in metrics_all])),
        "cv_mean_ece": float(np.mean([m["ece"] for m in metrics_all])),
    }

    train_range = _date_range(df.loc[X_fit.index, cols["date"]])
    valid_range = _date_range(df.loc[X_cal.index, cols["date"]])
    diagnostics = _build_diagnostics(
        df=df,
        cols=cols,
        y=y,
        X=X_raw,
        feature_cols=raw_feature_cols,
        metrics_all=metrics_all,
        selected_feature_cols=feature_cols,
        feature_sanitization=feature_sanitization,
        training_weight_summary=training_weight_summary,
    )
    diagnostics["easy_label_profile"] = easy_label_profile
    diagnostics["calibration_method_resolution"] = method_resolution

    return CalibratorResult(
        model=calibrator,
        metrics=metrics,
        feature_cols=feature_cols,
        n_train=int(len(X_fit)),
        n_valid=int(len(X_cal)),
        train_range=train_range,
        valid_range=valid_range,
        diagnostics=diagnostics,
    )


def save_calibrator(out_dir: str, res: CalibratorResult):
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "概率校准器.pkl")
    meta_path = os.path.join(out_dir, "概率校准器指标.json")

    from joblib import dump
    dump(res.model, model_path)

    payload = {
        "metrics": res.metrics,
        "feature_cols": res.feature_cols,
        "n_train": res.n_train,
        "n_valid": res.n_valid,
        "train_range": res.train_range,
        "valid_range": res.valid_range,
        "diagnostics": res.diagnostics,
        "alerts": res.diagnostics.get("alerts", []) if isinstance(res.diagnostics, dict) else [],
        "model_type": str(type(res.model)),
        "has_lightgbm": HAS_LGB,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return model_path, meta_path
