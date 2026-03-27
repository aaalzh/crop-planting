import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load

from 后端.兼容层 import tune_loaded_model
from 后端.数据加载 import (
    load_cost_data,
    load_env_predictor,
    load_name_map,
    load_price_series,
    load_yield_data,
    load_yield_history,
    resolve_names,
)
from 后端.模型产物 import (
    cost_model_candidates,
    price_model_candidates,
    price_recursive_model_candidates,
    yield_model_candidates,
)
from 后端.价格作物覆盖 import resolve_price_cfg_for_crop
from 后端.模型.产量模型 import load_yield_table, make_features
from 后端.模型.成本模型 import predict_cost
from 后端.模型.概率校准器 import _encode_confidence
from 后端.时间策略 import (
    prediction_window_payload,
    resolve_price_window_from_dates,
    resolve_price_window_from_df,
    resolve_price_window_from_price_dir,
    resolve_target_year,
    resolve_year_window_from_series,
)
from 后端.特征工程 import make_recent_features
from 后端.风险评估 import price_volatility, risk_score
from 后端.价格汇总 import summarize_forecast_tail


from 后端.价格递归预测 import estimate_daily_move_limit, recursive_multi_step_forecast
from 后端.模型.价格模型 import train_one_crop as train_price_one_crop

logger = logging.getLogger(__name__)

DEFAULT_MODEL_VERSION = "v2"
_MODEL_CACHE: Dict[str, Tuple[int, object]] = {}
_META_CACHE: Dict[str, Tuple[int, dict]] = {}
_PRICE_CONTEXT_CACHE: Dict[str, Tuple[int, pd.DataFrame, float, Optional[int]]] = {}
_PRICE_HISTORY_CACHE: Dict[str, Tuple[int, pd.DataFrame, float, Optional[int]]] = {}
_PRICE_STEP_MODEL_CACHE: Dict[str, Tuple[int, object, dict]] = {}
_ENV_PREDICTOR_CACHE: Dict[str, object] = {}
_COST_DATA_CACHE: Dict[str, Tuple[int, pd.DataFrame]] = {}
_YIELD_TABLE_CACHE: Dict[str, Tuple[int, object]] = {}
_YIELD_HISTORY_CACHE: Dict[str, Tuple[int, pd.DataFrame]] = {}
_YIELD_HISTORY_LAST_YEAR_CACHE: Dict[str, Tuple[int, Dict[str, int]]] = {}
_CALIBRATOR_HEALTH_CACHE: Dict[str, Tuple[int, dict]] = {}


def _serving_cfg(config: dict) -> dict:
    return config.get("serving", {})


def _model_version(config: dict) -> str:
    return str(_serving_cfg(config).get("model_cache_version", DEFAULT_MODEL_VERSION))


def _strict_loading(config: dict) -> bool:
    return bool(_serving_cfg(config).get("strict_model_loading", False))


def _calibrator_fallback_mode(config: dict) -> str:
    mode = str(_serving_cfg(config).get("calibrator_degraded_fallback", "env_prob")).strip().lower()
    if mode not in {"env_prob", "none"}:
        return "env_prob"
    return mode


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


def _load_joblib_cached(path: Path) -> object:
    key = str(path.resolve())
    mtime = path.stat().st_mtime_ns
    item = _MODEL_CACHE.get(key)
    if item and item[0] == mtime:
        return item[1]
    obj = tune_loaded_model(load(path))
    _MODEL_CACHE[key] = (mtime, obj)
    return obj


def _load_json_cached(path: Path) -> Optional[dict]:
    key = str(path.resolve())
    mtime = path.stat().st_mtime_ns
    item = _META_CACHE.get(key)
    if item and item[0] == mtime:
        return item[1]
    meta = _read_json(path) or {}
    _META_CACHE[key] = (mtime, meta)
    return meta


def _resolve_meta_path(meta_path: Path) -> Path:
    candidates = [
        meta_path,
        meta_path.with_name(meta_path.stem + "_metrics.json") if not meta_path.name.endswith("_metrics.json") else meta_path,
    ]
    if meta_path.name.endswith("_指标.json"):
        legacy_name = meta_path.name.replace("_指标.json", "_metrics.json")
        candidates.append(meta_path.with_name(legacy_name))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return meta_path


def _load_model_and_meta(
    candidates: List[Tuple[Path, Path]],
    model_kind: str,
    strict: bool,
    missing_models: List[str],
) -> Tuple[Optional[object], Optional[dict]]:
    for model_path, meta_path in candidates:
        if not model_path.exists():
            continue
        t0 = time.perf_counter()
        try:
            model = _load_joblib_cached(model_path)
            resolved_meta_path = _resolve_meta_path(meta_path)
            meta = _load_json_cached(resolved_meta_path) if resolved_meta_path.exists() else {}
            logger.info(
                "loaded %s model from %s in %.2f ms",
                model_kind,
                model_path.as_posix(),
                (time.perf_counter() - t0) * 1000.0,
            )
            return model, meta
        except Exception:
            logger.exception("failed loading %s model: %s", model_kind, model_path.as_posix())

    missing = f"{model_kind}: " + ", ".join([p.as_posix() for p, _ in candidates[:2]])
    missing_models.append(missing)
    if strict:
        raise FileNotFoundError(f"missing required {model_kind} model; checked: {missing}")
    logger.warning("missing %s model; checked candidates: %s", model_kind, missing)
    return None, None


def _load_calibrator(config: dict) -> Tuple[Optional[object], Optional[dict]]:
    prob_cfg = config.get("probability", {})
    if not prob_cfg.get("enable_calibrator", False):
        return None, None

    out_dir = Path(config["output"]["out_dir"])
    model_path = out_dir / "概率校准器.pkl"
    meta_path = out_dir / "概率校准器指标.json"
    if not model_path.exists():
        logger.warning("calibrator model not found: %s", model_path.as_posix())
        return None, None

    try:
        calibrator = _load_joblib_cached(model_path)
        meta = _load_json_cached(meta_path) if meta_path.exists() else {}
        return calibrator, meta
    except Exception:
        logger.exception("failed loading probability calibrator")
    return None, None


def _history_unique_count(df: pd.DataFrame, col: str, round_digits: Optional[int] = None) -> int:
    if col not in df.columns:
        return 0
    s = pd.to_numeric(df[col], errors="coerce")
    s = s.dropna()
    if s.empty:
        return 0
    if round_digits is not None:
        s = s.round(round_digits)
    return int(s.nunique())


def _history_text_unique_count(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    s = df[col].astype(str).str.strip().str.lower()
    s = s[(s != "") & (s != "nan")]
    if s.empty:
        return 0
    return int(s.nunique())


def _release_score_weights(config: dict) -> Optional[Dict[str, float]]:
    candidates = [
        ((config.get("serving") or {}).get("active_release") or {}).get("score_weights"),
        (config.get("scoring") or {}).get("release_score_weights"),
    ]
    for item in candidates:
        if not isinstance(item, dict):
            continue
        keys = {"w_profit", "w_env", "w_risk", "w_uncertainty"}
        if not keys.issubset(set(item.keys())):
            continue
        try:
            return {key: float(item[key]) for key in keys}
        except Exception:
            continue
    return None


def _estimate_row_uncertainty(row: dict) -> float:
    base = _safe_float(row.get("uncertainty"))
    if base is not None:
        return max(0.0, float(base))

    p10 = _safe_float(row.get("price_p10"))
    p50 = _safe_float(row.get("price_p50"))
    p90 = _safe_float(row.get("price_p90"))
    price_width = None
    if p10 is not None and p50 is not None and p90 is not None:
        denom = max(abs(float(p50)), 1.0)
        price_width = max(0.0, float(p90) - float(p10)) / denom

    alignment = row.get("time_alignment") or {}
    gaps = (alignment.get("gaps") or {}) if isinstance(alignment, dict) else {}
    gap_values = [
        _safe_float(gaps.get("price_gap_years")),
        _safe_float(gaps.get("yield_gap_years")),
        _safe_float(gaps.get("cost_gap_years")),
    ]
    gap_penalty = sum(v for v in gap_values if v is not None) * 0.08
    risk = _safe_float(row.get("risk")) or 0.0
    base_value = price_width if price_width is not None else risk
    return max(0.0, float(base_value) + float(gap_penalty))


def _apply_release_score_fusion(results: List[dict], config: dict) -> Tuple[str, Optional[Dict[str, float]]]:
    weights = _release_score_weights(config)
    if not weights:
        for row in results:
            if isinstance(row, dict) and row.get("uncertainty") is None:
                row["uncertainty"] = _estimate_row_uncertainty(row)
        return "legacy_profit_risk_v1", None

    profit_values: List[float] = []
    uncertainty_values: List[float] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        profit = _safe_float(row.get("profit"))
        uncertainty = _estimate_row_uncertainty(row)
        row["uncertainty"] = uncertainty
        if profit is not None:
            profit_values.append(float(profit))
        uncertainty_values.append(float(uncertainty))

    if not profit_values:
        return "legacy_profit_risk_v1", None

    profit_arr = np.asarray(profit_values, dtype=float)
    profit_mean = float(np.nanmean(profit_arr))
    profit_std = float(np.nanstd(profit_arr))

    unc_arr = np.asarray(uncertainty_values, dtype=float) if uncertainty_values else np.asarray([], dtype=float)
    unc_min = float(np.nanmin(unc_arr)) if unc_arr.size else 0.0
    unc_max = float(np.nanmax(unc_arr)) if unc_arr.size else 0.0

    for row in results:
        if not isinstance(row, dict):
            continue
        profit = _safe_float(row.get("profit"))
        if profit is None:
            continue

        env_prob = _safe_float(row.get("env_prob")) or 0.0
        risk = _safe_float(row.get("risk")) or 0.0
        uncertainty = _safe_float(row.get("uncertainty")) or 0.0
        alignment = row.get("time_alignment") or {}
        score_weight = _safe_float((alignment.get("score_weight") if isinstance(alignment, dict) else None)) or 1.0

        if np.isfinite(profit_std) and profit_std > 1e-9:
            profit_z = (float(profit) - profit_mean) / profit_std
        else:
            profit_z = float(profit) - profit_mean

        if unc_max > unc_min:
            uncertainty_norm = (float(uncertainty) - unc_min) / max(unc_max - unc_min, 1e-9)
        else:
            uncertainty_norm = 0.0

        score_raw = (
            float(weights["w_profit"]) * float(profit_z)
            + float(weights["w_env"]) * float(env_prob)
            - float(weights["w_risk"]) * float(risk)
            - float(weights["w_uncertainty"]) * float(uncertainty_norm)
        )
        row["score_raw"] = float(score_raw)
        row["score"] = float(score_raw * score_weight)
        row["score_source"] = "release_score_fusion_v1"
        row["score_components"] = {
            "profit_z": float(profit_z),
            "env_prob": float(env_prob),
            "risk": float(risk),
            "uncertainty_norm": float(uncertainty_norm),
            "alignment_weight": float(score_weight),
        }

    return "release_score_fusion_v1", weights


def _calibrator_health(config: dict, calibrator_meta: Optional[dict]) -> dict:
    prob_cfg = config.get("probability", {})
    if not bool(prob_cfg.get("enable_calibrator", False)):
        return {"degraded": False, "reasons": ["disabled_by_config"], "warnings": [], "stats": {}}
    history_file = str(prob_cfg.get("history_file", "")).strip()
    if not history_file:
        return {"degraded": True, "reasons": ["history_file_empty"], "stats": {}}

    history_path = Path(history_file)
    meta_signature = ""
    if isinstance(calibrator_meta, dict):
        try:
            meta_signature = json.dumps(
                {
                    "metrics": calibrator_meta.get("metrics", {}),
                    "alerts": ((calibrator_meta.get("diagnostics") or {}).get("alerts") or []),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        except Exception:
            meta_signature = str(type(calibrator_meta))
    key = f"{history_path.resolve()}|{hash(meta_signature)}"
    mtime = history_path.stat().st_mtime_ns if history_path.exists() else -1
    cached = _CALIBRATOR_HEALTH_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]

    stats = {"history_file": history_path.as_posix()}
    reasons: List[str] = []
    warnings: List[str] = []

    if not history_path.exists():
        reasons.append("history_file_missing")
        health = {"degraded": True, "reasons": reasons, "stats": stats}
        _CALIBRATOR_HEALTH_CACHE[key] = (mtime, health)
        return health

    try:
        hdf = pd.read_csv(history_path, comment="#")
    except Exception:
        reasons.append("history_file_unreadable")
        health = {"degraded": True, "reasons": reasons, "stats": stats}
        _CALIBRATOR_HEALTH_CACHE[key] = (mtime, health)
        return health

    rows = int(len(hdf))
    stats["history_rows"] = rows
    if rows < 200:
        reasons.append("history_rows_too_small")

    prob_env_unique = _history_unique_count(hdf, "prob_env", round_digits=6)
    env_conf_unique = _history_text_unique_count(hdf, "env_confidence")
    ood_unique = _history_unique_count(hdf, "ood_flag")
    risk_unique = _history_unique_count(hdf, "risk_score", round_digits=4)

    stats["prob_env_unique"] = prob_env_unique
    stats["env_confidence_unique"] = env_conf_unique
    stats["ood_flag_unique"] = ood_unique
    stats["risk_score_unique"] = risk_unique

    # Core degradation signals: if environment probability collapses to one value,
    # calibrator cannot learn meaningful ranking confidence.
    if prob_env_unique <= 1:
        reasons.append("prob_env_constant")

    # Secondary signal: most uncertainty-related features have no variation.
    if env_conf_unique <= 1 and ood_unique <= 1 and risk_unique <= 1:
        reasons.append("uncertainty_features_constant")

    metrics = (calibrator_meta or {}).get("metrics", {}) if isinstance(calibrator_meta, dict) else {}
    try:
        ll = float(metrics.get("cv_mean_logloss", 0.0))
        brier = float(metrics.get("cv_mean_brier", 0.0))
        ece = float(metrics.get("cv_mean_ece", 0.0))
        stats["cv_mean_logloss"] = ll
        stats["cv_mean_brier"] = brier
        stats["cv_mean_ece"] = ece
        if ll < 1e-8 and brier < 1e-8 and ece < 1e-8:
            reasons.append("metrics_suspiciously_perfect")
    except Exception:
        pass

    diagnostics = (calibrator_meta or {}).get("diagnostics", {}) if isinstance(calibrator_meta, dict) else {}
    if isinstance(diagnostics, dict):
        label_profile = diagnostics.get("label_profile", {})
        profit_margin = diagnostics.get("profit_margin", {})
        fold_behavior = diagnostics.get("fold_behavior", {})
        alerts = diagnostics.get("alerts", [])

        if isinstance(label_profile, dict):
            positive_rate = label_profile.get("positive_rate")
            avg_candidates = label_profile.get("avg_candidates_per_group")
            if positive_rate is not None:
                stats["label_positive_rate"] = positive_rate
            if avg_candidates is not None:
                stats["avg_candidates_per_group"] = avg_candidates
        if isinstance(profit_margin, dict):
            rel_margin_median = profit_margin.get("rel_median")
            if rel_margin_median is not None:
                stats["profit_margin_rel_median"] = rel_margin_median
        if isinstance(fold_behavior, dict):
            near_perfect_fold_ratio = fold_behavior.get("near_perfect_fold_ratio")
            if near_perfect_fold_ratio is not None:
                stats["near_perfect_fold_ratio"] = near_perfect_fold_ratio
        if isinstance(alerts, list):
            for alert in alerts:
                if not isinstance(alert, dict):
                    continue
                code = str(alert.get("code") or "").strip()
                if code:
                    warnings.append(code)
            if warnings:
                stats["diagnostic_alert_count"] = int(len(warnings))
                stats["diagnostic_alerts"] = warnings

    health = {
        "degraded": len(reasons) > 0,
        "reasons": reasons,
        "warnings": warnings,
        "stats": stats,
    }
    _CALIBRATOR_HEALTH_CACHE[key] = (mtime, health)
    return health


def _price_context(
    price_dir: str,
    price_file: str,
    horizon: int,
    lags: list,
    windows: list,
    feature_space: str,
    include_raw_time_features: bool,
    time_raw_mode: Optional[str],
    as_of_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, float, Optional[int]]:
    csv_path = Path(price_dir) / f"{price_file}.csv"
    mtime = csv_path.stat().st_mtime_ns
    as_of_text = as_of_date.strftime("%Y-%m-%d") if isinstance(as_of_date, pd.Timestamp) else ""
    key = (
        f"{csv_path.resolve()}|h={horizon}|l={','.join(map(str, lags))}|w={','.join(map(str, windows))}"
        f"|fs={feature_space}|raw={int(bool(include_raw_time_features))}|tmode={str(time_raw_mode or '')}"
        f"|asof={as_of_text}"
    )
    item = _PRICE_CONTEXT_CACHE.get(key)
    if item and item[0] == mtime:
        return item[1], item[2], item[3]

    df = load_price_series(price_dir, price_file)
    if isinstance(as_of_date, pd.Timestamp):
        df = df[pd.to_datetime(df["date"], errors="coerce") <= as_of_date].copy()
    if df.empty:
        _PRICE_CONTEXT_CACHE[key] = (mtime, pd.DataFrame(), 0.0, None)
        return pd.DataFrame(), 0.0, None
    X_recent = make_recent_features(
        df,
        "modal_price",
        horizon,
        lags,
        windows,
        feature_space=feature_space,
        include_raw_time_features=include_raw_time_features,
        time_raw_mode=time_raw_mode,
    )
    vol = price_volatility(df, window_days=90)
    last_year = None
    if "date" in df.columns and not df.empty:
        try:
            last_year = int(pd.to_datetime(df["date"], errors="coerce").dropna().dt.year.max())
        except Exception:
            last_year = None
    _PRICE_CONTEXT_CACHE[key] = (mtime, X_recent, vol, last_year)
    return X_recent, vol, last_year


def _price_history_context(
    price_dir: str,
    price_file: str,
    as_of_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, float, Optional[int]]:
    csv_path = Path(price_dir) / f"{price_file}.csv"
    mtime = csv_path.stat().st_mtime_ns
    as_of_text = as_of_date.strftime("%Y-%m-%d") if isinstance(as_of_date, pd.Timestamp) else ""
    key = f"{csv_path.resolve()}|asof={as_of_text}"
    item = _PRICE_HISTORY_CACHE.get(key)
    if item and item[0] == mtime:
        return item[1], item[2], item[3]

    df = load_price_series(price_dir, price_file)
    if isinstance(as_of_date, pd.Timestamp):
        df = df[pd.to_datetime(df["date"], errors="coerce") <= as_of_date].copy()
    if df.empty:
        _PRICE_HISTORY_CACHE[key] = (mtime, pd.DataFrame(), 0.0, None)
        return pd.DataFrame(), 0.0, None

    df = df.sort_values("date").reset_index(drop=True)
    vol = price_volatility(df, window_days=90)
    last_year = None
    if "date" in df.columns and not df.empty:
        try:
            last_year = int(pd.to_datetime(df["date"], errors="coerce").dropna().dt.year.max())
        except Exception:
            last_year = None

    _PRICE_HISTORY_CACHE[key] = (mtime, df, vol, last_year)
    return df, vol, last_year


def _feature_cols_from_meta_or_model(meta: Optional[dict], model: Optional[object]) -> List[str]:
    cols: List[str] = []
    if isinstance(meta, dict):
        maybe = meta.get("feature_cols", [])
        if isinstance(maybe, list):
            cols = [str(x) for x in maybe if str(x).strip()]
    if not cols and model is not None:
        maybe = getattr(model, "feature_cols", None)
        if isinstance(maybe, list):
            cols = [str(x) for x in maybe if str(x).strip()]
    return cols


def _normalize_prediction_mode(price_cfg: dict) -> str:
    mode = str((price_cfg or {}).get("prediction_mode", "return_recursive_v3")).strip().lower()
    if mode not in {"return_recursive_v3", "return_recursive_v2", "price_recursive_v1", "direct_horizon_v1"}:
        return "return_recursive_v3"
    return mode


def _build_inline_step_cfg(base_cfg: dict, prediction_mode: str) -> dict:
    cfg = deepcopy(base_cfg or {})
    cfg["regressor"] = str(base_cfg.get("recursive_fallback_regressor", "hgb")).strip().lower() or "hgb"
    cfg["max_iter"] = int(base_cfg.get("recursive_fallback_max_iter", 260))
    cfg["max_depth"] = int(base_cfg.get("recursive_fallback_max_depth", base_cfg.get("max_depth", 8) or 8))
    cfg["learning_rate"] = float(base_cfg.get("recursive_fallback_learning_rate", base_cfg.get("learning_rate", 0.04)))
    cfg["l2_regularization"] = float(
        base_cfg.get("recursive_fallback_l2_regularization", base_cfg.get("l2_regularization", 1e-3))
    )
    cfg["use_recency_weight"] = bool(base_cfg.get("use_recency_weight", True))
    cfg["recency_halflife_days"] = float(base_cfg.get("recency_halflife_days", 365))
    cfg["n_jobs"] = int(base_cfg.get("n_jobs", 1))
    cfg["verbose"] = False
    mode = _normalize_prediction_mode({"prediction_mode": prediction_mode})
    if mode in {"return_recursive_v3", "return_recursive_v2"}:
        cfg["target_mode"] = "log_return"
        cfg["feature_space"] = "log_price"
        cfg["time_raw_mode"] = "none"
        cfg["include_raw_time_features"] = False
        cfg["target_transform"] = "none"
        cfg["enable_bias_correction"] = bool(base_cfg.get("enable_bias_correction", True))
        cfg["enable_seasonal_anchor"] = bool(base_cfg.get("enable_seasonal_anchor", True))
        cfg["enable_conformal_interval"] = bool(base_cfg.get("enable_conformal_interval", True))
    else:
        cfg["target_mode"] = "price"
        cfg["feature_space"] = "price"
        cfg["time_raw_mode"] = str(base_cfg.get("legacy_time_raw_mode", "raw"))
        cfg["include_raw_time_features"] = bool(base_cfg.get("legacy_include_raw_time_features", True))
        cfg["target_transform"] = str(base_cfg.get("target_transform", "log1p"))
    return cfg


def _inline_step_model(
    *,
    crop: str,
    price_history_df: pd.DataFrame,
    price_file: str,
    price_cfg: dict,
    prediction_mode: str,
    lags: list,
    windows: list,
    backtest_days: int,
    validation_cutoff: str,
    strict_cutoff_split: bool,
    as_of_date: Optional[pd.Timestamp],
) -> Tuple[Optional[object], Optional[dict]]:
    if price_history_df is None or price_history_df.empty:
        return None, None

    cache_key = (
        f"{crop}|{price_file}|asof={as_of_date.strftime('%Y-%m-%d') if isinstance(as_of_date, pd.Timestamp) else ''}|"
        f"l={','.join(map(str, lags))}|w={','.join(map(str, windows))}|cfg="
        f"{json.dumps(_build_inline_step_cfg(price_cfg, prediction_mode), ensure_ascii=False, sort_keys=True)}"
    )
    mtime = int(pd.to_datetime(price_history_df["date"], errors="coerce").max().value) if "date" in price_history_df.columns else -1
    cached = _PRICE_STEP_MODEL_CACHE.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]

    try:
        step_cfg = _build_inline_step_cfg(price_cfg, prediction_mode)
        res = train_price_one_crop(
            price_history_df,
            step_cfg,
            lags,
            windows,
            horizon=1,
            backtest_days=max(30, int(backtest_days)),
            test_ratio=step_cfg.get("test_ratio", price_cfg.get("test_ratio")),
            validation_cutoff=validation_cutoff,
            strict_cutoff_split=strict_cutoff_split,
            verbose=False,
            label=f"{crop}:inline_step1",
        )
        meta = {
            "feature_cols": list(res.feature_cols or []),
            "training_horizon_days": 1,
            "forecast_mode": "recursive_inline_step1",
            "metrics": res.metrics,
            "target_mode": step_cfg.get("target_mode"),
            "feature_space": step_cfg.get("feature_space"),
            "time_raw_mode": step_cfg.get("time_raw_mode"),
            "include_raw_time_features": bool(step_cfg.get("include_raw_time_features", False)),
            "artifacts": res.artifacts or {},
        }
        _PRICE_STEP_MODEL_CACHE[cache_key] = (mtime, res.model, meta)
        return res.model, meta
    except Exception:
        logger.exception("failed training inline step model for crop=%s price_file=%s", crop, price_file)
        return None, None


def _recursive_price_forecast(
    *,
    crop: str,
    model_dir: Path,
    version: str,
    strict: bool,
    missing_models: List[str],
    price_cfg: dict,
    price_history_df: pd.DataFrame,
    price_file: str,
    horizon_days: int,
    lags: list,
    windows: list,
    backtest_days: int,
    validation_cutoff: str,
    strict_cutoff_split: bool,
    as_of_date: Optional[pd.Timestamp],
    prediction_mode: str,
    anchor_end_price: Optional[float] = None,
    direct_reference_points: Optional[List[dict]] = None,
) -> Tuple[List[dict], Optional[str], Dict[str, object]]:
    requested_mode = _normalize_prediction_mode({"prediction_mode": prediction_mode})
    price_cfg, crop_override_keys, override_status = resolve_price_cfg_for_crop(
        price_cfg,
        crop,
        prediction_mode=requested_mode,
        as_of_date=as_of_date,
        include_status=True,
    )
    diag: Dict[str, object] = {
        "prediction_mode": requested_mode,
        "crop_override_keys": list(crop_override_keys),
        "override_status": dict(override_status or {}),
    }
    for notice in (override_status or {}).get("warnings", []):
        if not isinstance(notice, dict):
            continue
        logger.warning(
            "price override warning crop=%s code=%s message=%s",
            crop,
            str(notice.get("code") or ""),
            str(notice.get("message") or ""),
        )
    if price_history_df is None or price_history_df.empty:
        diag["fallback_reason"] = "price_history_empty"
        return [], None, diag

    model = None
    meta: Optional[dict] = None
    mode: Optional[str] = None
    expected_target = "log_return" if requested_mode in {"return_recursive_v3", "return_recursive_v2"} else "price"

    def _meta_target_mode(meta_obj: Optional[dict]) -> str:
        if not isinstance(meta_obj, dict):
            return ""
        for key in ("target_mode",):
            val = str(meta_obj.get(key, "")).strip().lower()
            if val:
                return "log_return" if val in {"log_return", "return", "ret", "delta"} else "price"
        artifacts = meta_obj.get("artifacts", {})
        if isinstance(artifacts, dict):
            val = str(artifacts.get("target_mode", "")).strip().lower()
            if val:
                return "log_return" if val in {"log_return", "return", "ret", "delta"} else "price"
        return ""

    step_candidates = price_recursive_model_candidates(model_dir, crop, price_cfg, version)
    if any(path.exists() for path, _ in step_candidates):
        model, meta = _load_model_and_meta(
            step_candidates,
            model_kind=f"price_step1[{crop}]",
            strict=False,
            missing_models=missing_models,
        )
        if model is None and missing_models and str(missing_models[-1]).startswith(f"price_step1[{crop}]"):
            missing_models.pop()
    if model is not None:
        horizon_meta = _safe_float((meta or {}).get("training_horizon_days"))
        meta_target = _meta_target_mode(meta)
        if horizon_meta is not None and int(horizon_meta) != 1:
            model = None
            meta = None
        elif requested_mode in {"return_recursive_v3", "return_recursive_v2"} and meta_target != "log_return":
            model = None
            meta = None
            diag["artifact_rejected"] = "target_mode_mismatch"
        elif requested_mode == "price_recursive_v1" and meta_target == "log_return":
            model = None
            meta = None
            diag["artifact_rejected"] = "target_mode_mismatch"
        else:
            mode = "recursive_step1_artifact"

    if model is None and bool(price_cfg.get("recursive_inline_train_on_missing", True)):
        model, meta = _inline_step_model(
            crop=crop,
            price_history_df=price_history_df,
            price_file=price_file,
            price_cfg=price_cfg,
            prediction_mode=requested_mode,
            lags=lags,
            windows=windows,
            backtest_days=backtest_days,
            validation_cutoff=validation_cutoff,
            strict_cutoff_split=strict_cutoff_split,
            as_of_date=as_of_date,
        )
        if model is not None:
            mode = "recursive_step1_inline"

    if model is None:
        diag["fallback_reason"] = "step_model_unavailable"
        return [], None, diag

    feature_cols = _feature_cols_from_meta_or_model(meta, model)
    artifacts = (meta or {}).get("artifacts", {}) if isinstance(meta, dict) else {}
    if not isinstance(artifacts, dict):
        artifacts = {}
    max_daily_move_pct = _safe_float(price_cfg.get("recursive_max_daily_move_pct"))
    if requested_mode == "price_recursive_v1" and max_daily_move_pct is None:
        max_daily_move_pct = estimate_daily_move_limit(price_history_df)

    clip_art = artifacts.get("clip", {}) if isinstance(artifacts.get("clip"), dict) else {}
    bias_art = (
        artifacts.get("bias_correction", {}) if isinstance(artifacts.get("bias_correction"), dict) else {}
    )
    conformal_art = artifacts.get("conformal", {}) if isinstance(artifacts.get("conformal"), dict) else {}
    seasonal_art = (
        artifacts.get("seasonal_anchor", {}) if isinstance(artifacts.get("seasonal_anchor"), dict) else {}
    )

    include_raw_time_features = bool(
        (meta or {}).get(
            "include_raw_time_features",
            False if requested_mode in {"return_recursive_v3", "return_recursive_v2"} else True,
        )
    )
    time_raw_mode = (meta or {}).get(
        "time_raw_mode",
        "none"
        if requested_mode in {"return_recursive_v3", "return_recursive_v2"}
        else str(price_cfg.get("legacy_time_raw_mode", "raw")),
    )

    anchor_effective = _safe_float(anchor_end_price)
    use_terminal_anchor = bool(price_cfg.get("recursive_use_terminal_anchor", True))
    if not use_terminal_anchor:
        anchor_effective = None
    if anchor_effective is not None and anchor_effective > 0:
        try:
            hist = pd.to_numeric(price_history_df.get("modal_price"), errors="coerce").dropna().tail(180)
            if len(hist) >= 20:
                med = _safe_float(hist.median())
                if med is not None and med > 0:
                    ratio = anchor_effective / med
                    lo = float(price_cfg.get("recursive_anchor_valid_ratio_low", 0.7))
                    hi = float(price_cfg.get("recursive_anchor_valid_ratio_high", 1.4))
                    if lo > hi:
                        lo, hi = hi, lo
                    if ratio < lo or ratio > hi:
                        anchor_effective = None
        except Exception:
            anchor_effective = None

    forecast = recursive_multi_step_forecast(
        model=model,
        history_df=price_history_df,
        horizon_days=horizon_days,
        lags=lags,
        windows=windows,
        feature_cols=feature_cols or None,
        max_daily_move_pct=max_daily_move_pct,
        anchor_end_value=anchor_effective,
        anchor_max_blend=float(price_cfg.get("recursive_anchor_max_blend", 0.65)),
        mean_reversion_strength=float(price_cfg.get("recursive_mean_reversion_strength", 0.18)),
        seasonal_strength=float(price_cfg.get("recursive_seasonal_strength", 0.22)),
        corridor_low_quantile=float(price_cfg.get("recursive_corridor_low_quantile", 0.05)),
        corridor_high_quantile=float(price_cfg.get("recursive_corridor_high_quantile", 0.95)),
        corridor_low_multiplier=float(price_cfg.get("recursive_corridor_low_multiplier", 0.75)),
        corridor_high_multiplier=float(price_cfg.get("recursive_corridor_high_multiplier", 1.35)),
        terminal_low_ratio_vs_latest=float(price_cfg.get("recursive_terminal_low_ratio_vs_latest", 0.6)),
        terminal_high_ratio_vs_latest=float(price_cfg.get("recursive_terminal_high_ratio_vs_latest", 1.8)),
        prediction_mode=requested_mode,
        clip_r_max=_safe_float(price_cfg.get("return_clip_r_max")) or _safe_float(clip_art.get("r_max")),
        return_clip_quantile=float(price_cfg.get("return_clip_quantile", clip_art.get("quantile", 0.98))),
        return_clip_safety_factor=float(
            price_cfg.get("return_clip_safety_factor", clip_art.get("safety_factor", 1.2))
        ),
        return_bias_mean=_safe_float(bias_art.get("return_bias_mean")),
        enable_bias_correction=bool(price_cfg.get("enable_bias_correction", True)),
        conformal_abs_q=_safe_float(conformal_art.get("abs_q")),
        enable_conformal_interval=bool(price_cfg.get("enable_conformal_interval", True)),
        enable_seasonal_anchor=bool(price_cfg.get("enable_seasonal_anchor", True)),
        seasonal_y_by_doy=seasonal_art.get("dayofyear_log_price_median"),
        seasonal_tau_days=float(price_cfg.get("seasonal_anchor_tau_days", 55.0)),
        seasonal_min_ml_weight=float(price_cfg.get("seasonal_anchor_min_ml_weight", 0.15)),
        seasonal_drift_max_alpha=float(price_cfg.get("seasonal_drift_max_alpha", price_cfg.get("recursive_seasonal_strength", 0.22))),
        seasonal_drift_growth_power=float(price_cfg.get("seasonal_drift_growth_power", 1.6)),
        clip_rate_warn_threshold=float(price_cfg.get("clip_rate_warn_threshold", 0.10)),
        endpoint_direction_guard=bool(price_cfg.get("endpoint_direction_guard", True)),
        endpoint_opposite_slack=float(price_cfg.get("endpoint_opposite_slack", 1.5)),
        endpoint_same_dir_cap=float(price_cfg.get("endpoint_same_dir_cap", 4.0)),
        terminal_anchor_enable=bool(price_cfg.get("recursive_use_terminal_anchor", True)),
        terminal_anchor_weight=float(price_cfg.get("recursive_terminal_anchor_weight", 0.35)),
        terminal_anchor_tail_days=int(price_cfg.get("recursive_terminal_anchor_tail_days", 45)),
        terminal_anchor_tail_power=float(price_cfg.get("recursive_terminal_anchor_tail_power", 1.6)),
        trend_guard_early_drop_floor=float(price_cfg.get("trend_guard_early_drop_floor", -0.08)),
        trend_guard_early_window_days=int(price_cfg.get("trend_guard_early_window_days", 28)),
        trend_guard_end_ratio_low=float(price_cfg.get("trend_guard_end_ratio_low", 0.85)),
        trend_guard_end_ratio_high=float(price_cfg.get("trend_guard_end_ratio_high", 1.15)),
        trend_guard_hist_window_days=int(price_cfg.get("trend_guard_hist_window_days", 90)),
        trend_guard_future_window_days=int(price_cfg.get("trend_guard_future_window_days", 60)),
        trend_guard_t0_jump_max=float(price_cfg.get("trend_guard_t0_jump_max", 0.03)),
        trend_guard_t0_jump_steps=int(price_cfg.get("trend_guard_t0_jump_steps", 1)),
        trend_guard_end_tail_days=int(price_cfg.get("trend_guard_end_tail_days", 30)),
        trend_guard_end_max_weight=float(price_cfg.get("trend_guard_end_max_weight", 0.90)),
        trend_guard_end_pressure_scale=float(price_cfg.get("trend_guard_end_pressure_scale", 6.0)),
        trend_guard_end_backstop=bool(price_cfg.get("trend_guard_end_backstop", True)),
        interval_growth_power=float(price_cfg.get("conformal_interval_growth_power", 0.5)),
        interval_growth_scale=float(price_cfg.get("conformal_interval_growth_scale", 1.0)),
        interval_growth_cap=float(price_cfg.get("conformal_interval_growth_cap", 200.0)),
        conformal_horizon_scale_points=conformal_art.get("horizon_scale_points"),
        conformal_use_horizon_scale=bool(price_cfg.get("conformal_use_horizon_scale", True)),
        conformal_local_vol_window=int(price_cfg.get("conformal_local_vol_window_days", 45)),
        conformal_local_vol_ratio_low=float(price_cfg.get("conformal_local_vol_ratio_low", 0.70)),
        conformal_local_vol_ratio_high=float(price_cfg.get("conformal_local_vol_ratio_high", 1.40)),
        conformal_local_vol_reference=_safe_float(conformal_art.get("local_vol_reference")),
        return_smooth_alpha=float(price_cfg.get("recursive_return_smooth_alpha", 0.15)),
        return_jerk_clip_sigma=float(price_cfg.get("recursive_return_jerk_clip_sigma", 2.5)),
        return_smooth_warmup_steps=int(price_cfg.get("recursive_return_smooth_warmup_steps", 5)),
        direct_reference_points=direct_reference_points,
        direct_reference_enable=bool(price_cfg.get("direct_reference_enable", True)),
        direct_reference_weight=float(price_cfg.get("direct_reference_weight", 0.10)),
        direct_reference_max_weight=float(price_cfg.get("direct_reference_max_weight", 0.35)),
        direct_reference_progress_power=float(price_cfg.get("direct_reference_progress_power", 1.25)),
        direct_reference_anchor_radius_days=int(price_cfg.get("direct_reference_anchor_radius_days", 14)),
        direct_reference_local_boost=float(price_cfg.get("direct_reference_local_boost", 0.12)),
        direct_reference_min_future_points=int(price_cfg.get("direct_reference_min_future_points", 2)),
        include_raw_time_features=include_raw_time_features,
        time_raw_mode=str(time_raw_mode) if time_raw_mode is not None else None,
        diagnostics=diag,
    )
    if not forecast:
        if "fallback_reason" not in diag:
            diag["fallback_reason"] = "recursive_forecast_empty"
        return [], None, diag

    if (
        requested_mode in {"return_recursive_v3", "return_recursive_v2"}
        and str(diag.get("quality_flag", "OK")).upper() == "CLIP_TOO_OFTEN"
        and bool(price_cfg.get("degrade_to_direct_on_clip_too_often", True))
    ):
        diag["fallback_reason"] = "clip_too_often"
        diag["degraded_to_direct"] = True
        return [], None, diag

    diag["target_mode"] = expected_target
    diag["online_price_diagnostics"] = _build_online_price_diagnostics(
        forecast=forecast,
        history_df=price_history_df,
        price_cfg=price_cfg,
        requested_mode=requested_mode,
        model_source=mode,
        recursive_diag=diag,
    )
    return forecast, mode, diag


def _cost_data_cached(cost_file: str) -> pd.DataFrame:
    p = Path(cost_file)
    key = str(p.resolve())
    mtime = p.stat().st_mtime_ns
    item = _COST_DATA_CACHE.get(key)
    if item and item[0] == mtime:
        return item[1]
    df = load_cost_data(cost_file)
    _COST_DATA_CACHE[key] = (mtime, df)
    return df


def _yield_table_cached(yield_file: str):
    p = Path(yield_file)
    key = str(p.resolve())
    mtime = p.stat().st_mtime_ns if p.exists() else -1
    item = _YIELD_TABLE_CACHE.get(key)
    if item and item[0] == mtime:
        return item[1]
    table = load_yield_table(load_yield_data(yield_file))
    _YIELD_TABLE_CACHE[key] = (mtime, table)
    return table


def _yield_history_cached(yield_history_file: str) -> pd.DataFrame:
    p = Path(yield_history_file)
    key = str(p.resolve())
    mtime = p.stat().st_mtime_ns if p.exists() else -1
    item = _YIELD_HISTORY_CACHE.get(key)
    if item and item[0] == mtime:
        return item[1]
    if not p.exists():
        empty = pd.DataFrame()
        _YIELD_HISTORY_CACHE[key] = (mtime, empty)
        return empty
    try:
        df = load_yield_history(str(p))
    except Exception:
        df = pd.DataFrame()
    _YIELD_HISTORY_CACHE[key] = (mtime, df)
    return df


def _yield_last_year_map_cached(yield_history_file: str) -> Dict[str, int]:
    p = Path(yield_history_file)
    key = str(p.resolve())
    mtime = p.stat().st_mtime_ns if p.exists() else -1
    item = _YIELD_HISTORY_LAST_YEAR_CACHE.get(key)
    if item and item[0] == mtime:
        return item[1]

    out: Dict[str, int] = {}
    if not p.exists():
        _YIELD_HISTORY_LAST_YEAR_CACHE[key] = (mtime, out)
        return out

    try:
        df = load_yield_history(str(p))
    except Exception:
        _YIELD_HISTORY_LAST_YEAR_CACHE[key] = (mtime, out)
        return out

    if df.empty or "crop_name" not in df.columns or "year" not in df.columns:
        _YIELD_HISTORY_LAST_YEAR_CACHE[key] = (mtime, out)
        return out

    work = df[["crop_name", "year"]].copy()
    work["crop_name"] = work["crop_name"].astype(str).str.strip().str.lower()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work = work.dropna(subset=["crop_name", "year"])
    if not work.empty:
        grouped = work.groupby("crop_name")["year"].max()
        out = {str(c): int(y) for c, y in grouped.items()}

    _YIELD_HISTORY_LAST_YEAR_CACHE[key] = (mtime, out)
    return out


def _predict_yield_from_model(
    *,
    crop: str,
    target_year: int,
    env_input: dict,
    yield_model: object,
    yield_meta: Optional[dict],
    yield_history_df: pd.DataFrame,
    yield_cfg: dict,
) -> Optional[float]:
    if yield_model is None or yield_meta is None:
        return None
    feature_cols = yield_meta.get("feature_cols", [])
    if not feature_cols:
        return None
    feat = {"crop_name": crop, "year": int(target_year)}
    for k, v in env_input.items():
        feat[k] = v
    X_feat = make_features(
        pd.DataFrame([feat]),
        feature_cols,
        cfg=yield_cfg,
        history_df=yield_history_df,
    )
    if X_feat.empty:
        return None
    pred = yield_model.predict(X_feat)
    if pred is None or len(pred) == 0:
        return None
    value = float(pred[0])
    if not np.isfinite(value):
        return None
    return max(0.0, value)


def _resolve_target_year(config: dict) -> int:
    return resolve_target_year(config, fallback_year=int(pd.Timestamp.today().year))


def _resolve_prediction_window(config: dict) -> Dict[str, object]:
    policy = resolve_price_window_from_dates(None, time_cfg=config.get("time", {}))
    return {
        "start_date": policy["start_date"],
        "end_date": policy["end_date"],
        "price_horizon_days": int(policy["price_horizon_days"]),
        "train_validation_cutoff_date": str(policy.get("train_validation_cutoff_date") or "2020-12-31"),
    }


def _cost_last_year(cost_df: pd.DataFrame) -> Optional[int]:
    if cost_df is None or cost_df.empty or "year_start" not in cost_df.columns:
        return None
    years = pd.to_numeric(cost_df["year_start"], errors="coerce").dropna()
    if years.empty:
        return None
    return int(years.max())


def _alignment_score_weight(
    *,
    gaps: List[Optional[int]],
    alignment_cfg: dict,
) -> float:
    per_year_penalty = float(alignment_cfg.get("score_penalty_per_year", 0.05))
    max_penalty = float(alignment_cfg.get("max_score_penalty", 0.35))
    known_gaps = [int(g) for g in gaps if g is not None and int(g) > 0]
    if not known_gaps:
        return 1.0
    max_gap = max(known_gaps)
    penalty = min(max_penalty, max(0.0, per_year_penalty) * float(max_gap))
    return float(max(0.0, 1.0 - penalty))


def _normalized_confidence(best_prob: float) -> str:
    if best_prob >= 0.80:
        return "high"
    if best_prob >= 0.55:
        return "mid"
    return "low"


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
    except Exception:
        return None
    if pd.isna(f):
        return None
    return f


def _extract_direct_prediction(model: object, X: pd.DataFrame, horizon: int) -> Optional[float]:
    pred_out = None
    if model is None:
        return None

    if hasattr(model, "predict_for_horizon"):
        try:
            pred_out = model.predict_for_horizon(X, horizon=horizon)
        except TypeError:
            try:
                pred_out = model.predict_for_horizon(X, horizon)
            except Exception:
                pred_out = None
        except Exception:
            pred_out = None

    if pred_out is None:
        try:
            pred_out = model.predict(X)
        except Exception:
            return None

    try:
        arr = np.asarray(pred_out, dtype=float)
    except Exception:
        return _safe_float(pred_out)

    if arr.size == 0:
        return None
    if arr.ndim >= 2:
        first_row = arr.reshape(arr.shape[0], -1)[0]
        return _safe_float(first_row[-1])
    return _safe_float(arr.reshape(-1)[0])


def _extract_direct_reference_points(
    model: object,
    X: pd.DataFrame,
    *,
    horizon: int,
    meta: Optional[dict] = None,
) -> List[dict]:
    if model is None or X is None or X.empty:
        return []

    target_h = max(1, int(horizon))
    candidate_horizons: List[int] = []
    if hasattr(model, "horizons"):
        try:
            candidate_horizons.extend(int(x) for x in list(getattr(model, "horizons") or []))
        except Exception:
            pass

    if isinstance(meta, dict):
        metrics = meta.get("metrics", {}) if isinstance(meta.get("metrics"), dict) else {}
        artifacts = meta.get("artifacts", {}) if isinstance(meta.get("artifacts"), dict) else {}
        for src in (meta.get("trained_horizons"), metrics.get("trained_horizons"), artifacts.get("trained_horizons")):
            if not isinstance(src, (list, tuple)):
                continue
            for item in src:
                try:
                    candidate_horizons.append(int(item))
                except Exception:
                    continue

    candidate_horizons = sorted({int(h) for h in candidate_horizons if int(h) > 0 and int(h) <= target_h})
    if target_h not in candidate_horizons:
        candidate_horizons.append(target_h)
    if not candidate_horizons:
        candidate_horizons = [target_h]

    exact_map: Dict[int, Optional[float]] = {}
    model_horizons: List[int] = []
    if hasattr(model, "horizons"):
        try:
            model_horizons = [int(x) for x in list(getattr(model, "horizons") or [])]
        except Exception:
            model_horizons = []
    if hasattr(model, "predict_multi") and model_horizons:
        try:
            pred_multi = np.asarray(model.predict_multi(X), dtype=float)
            if pred_multi.ndim >= 2 and pred_multi.shape[0] > 0:
                first_row = pred_multi.reshape(pred_multi.shape[0], -1)[0]
                if first_row.size == len(model_horizons):
                    for idx, h in enumerate(model_horizons):
                        exact_map[int(h)] = _safe_float(first_row[idx])
        except Exception:
            exact_map = {}

    out: List[dict] = []
    for h in candidate_horizons:
        value = exact_map.get(int(h))
        source = "exact" if int(h) in exact_map else "interpolated"
        if value is None:
            value = _extract_direct_prediction(model, X, horizon=int(h))
        if value is None or not np.isfinite(value) or float(value) <= 0.0:
            continue
        out.append(
            {
                "horizon": int(h),
                "value": float(value),
                "source": source,
            }
        )
    return out


def _clip_probability(v: Optional[float], epsilon: float = 0.0) -> Optional[float]:
    p = _safe_float(v)
    if p is None:
        return None
    eps = _safe_float(epsilon)
    eps = 0.0 if eps is None else max(0.0, min(0.1, eps))
    lo = eps
    hi = 1.0 - eps
    if p < lo:
        return lo
    if p > hi:
        return hi
    return p


def _slope_sign(values: np.ndarray) -> int:
    arr = np.asarray(values, dtype=float)
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


def _width_non_decreasing_ratio(p10: np.ndarray, p90: np.ndarray) -> Optional[float]:
    if p10.size == 0 or p90.size == 0 or p10.size != p90.size:
        return None
    width = p90 - p10
    width = width[np.isfinite(width)]
    if width.size <= 1:
        return 1.0
    dif = np.diff(width)
    return float(np.mean(dif >= -1e-10))


def _build_online_price_diagnostics(
    *,
    forecast: List[dict],
    history_df: pd.DataFrame,
    price_cfg: dict,
    requested_mode: str,
    model_source: Optional[str],
    recursive_diag: Dict[str, object],
) -> Dict[str, object]:
    hist = pd.to_numeric(history_df.get("modal_price"), errors="coerce").dropna().to_numpy(dtype=float)
    if hist.size == 0:
        return {
            "prediction_mode": str(requested_mode),
            "model_source": model_source,
            "error": "history_empty",
        }
    last = float(hist[-1])
    ma30 = float(np.mean(hist[-min(30, hist.size) :]))

    p50 = np.asarray([_safe_float(x.get("p50", x.get("value"))) for x in forecast], dtype=float)
    p10 = np.asarray([_safe_float(x.get("p10")) for x in forecast], dtype=float)
    p90 = np.asarray([_safe_float(x.get("p90")) for x in forecast], dtype=float)
    if p50.size == 0 or not np.all(np.isfinite(p50)):
        return {
            "prediction_mode": str(requested_mode),
            "model_source": model_source,
            "error": "forecast_invalid",
            "last": last,
            "ma30": ma30,
        }

    early_end_idx = min(28, int(p50.size - 1))
    early_drop = float(np.min(p50[: early_end_idx + 1]) / max(last, 1e-6) - 1.0)
    end_ratio = float(p50[-1] / max(ma30, 1e-6))
    hist_sign = _slope_sign(np.log(np.clip(hist[-min(90, hist.size) :], 1e-6, None)))
    fut_sign = _slope_sign(np.log(np.clip(p50[: min(60, p50.size)], 1e-6, None)))
    direction_mismatch = int(hist_sign != 0 and fut_sign != 0 and int(hist_sign) != int(fut_sign))

    def _checkpoint(day_offset: int) -> Dict[str, Optional[float]]:
        idx = int(np.clip(day_offset, 0, max(0, p50.size - 1)))
        row = forecast[idx] if idx < len(forecast) else {}
        return {
            "index": int(idx),
            "date": str(row.get("date") or ""),
            "p50": _safe_float(p50[idx]) if idx < p50.size else None,
            "p10": _safe_float(p10[idx]) if idx < p10.size else None,
            "p90": _safe_float(p90[idx]) if idx < p90.size else None,
        }

    interval_width_ratio = _width_non_decreasing_ratio(p10, p90)
    return {
        "prediction_mode": str(requested_mode),
        "model_source": model_source,
        "flags": {
            "enable_seasonal_anchor": bool(price_cfg.get("enable_seasonal_anchor", True)),
            "recursive_use_terminal_anchor": bool(price_cfg.get("recursive_use_terminal_anchor", True)),
            "enable_conformal_interval": bool(price_cfg.get("enable_conformal_interval", True)),
            "endpoint_direction_guard": bool(price_cfg.get("endpoint_direction_guard", True)),
            "direct_reference_enable": bool(price_cfg.get("direct_reference_enable", True)),
            "trend_guard_enabled": bool(requested_mode == "return_recursive_v3"),
        },
        "params": {
            "seasonal_drift_max_alpha": _safe_float(price_cfg.get("seasonal_drift_max_alpha")),
            "seasonal_drift_growth_power": _safe_float(price_cfg.get("seasonal_drift_growth_power")),
            "terminal_anchor_weight": _safe_float(price_cfg.get("recursive_terminal_anchor_weight")),
            "terminal_anchor_tail_days": _safe_float(price_cfg.get("recursive_terminal_anchor_tail_days")),
            "terminal_anchor_tail_power": _safe_float(price_cfg.get("recursive_terminal_anchor_tail_power")),
            "direct_reference_weight": _safe_float(price_cfg.get("direct_reference_weight")),
            "direct_reference_max_weight": _safe_float(price_cfg.get("direct_reference_max_weight")),
            "direct_reference_progress_power": _safe_float(price_cfg.get("direct_reference_progress_power")),
            "direct_reference_anchor_radius_days": _safe_float(price_cfg.get("direct_reference_anchor_radius_days")),
            "direct_reference_local_boost": _safe_float(price_cfg.get("direct_reference_local_boost")),
            "trend_guard_early_drop_floor": _safe_float(price_cfg.get("trend_guard_early_drop_floor")),
            "trend_guard_end_ratio_low": _safe_float(price_cfg.get("trend_guard_end_ratio_low")),
            "trend_guard_end_ratio_high": _safe_float(price_cfg.get("trend_guard_end_ratio_high")),
            "trend_guard_t0_jump_max": _safe_float(price_cfg.get("trend_guard_t0_jump_max")),
            "trend_guard_t0_jump_steps": _safe_float(price_cfg.get("trend_guard_t0_jump_steps")),
            "trend_guard_end_tail_days": _safe_float(price_cfg.get("trend_guard_end_tail_days")),
            "trend_guard_end_max_weight": _safe_float(price_cfg.get("trend_guard_end_max_weight")),
            "trend_guard_end_backstop": bool(price_cfg.get("trend_guard_end_backstop", True)),
        },
        "stats": {
            "last": last,
            "ma30": ma30,
            "t0_jump": float(abs(p50[0] - last) / max(last, 1e-6)),
            "early_drop_28d": early_drop,
            "end_ratio": end_ratio,
            "hist_slope_sign90": int(hist_sign),
            "fut_slope_sign60": int(fut_sign),
            "direction_mismatch": int(direction_mismatch),
            "interval_width_non_decreasing_ratio": _safe_float(interval_width_ratio),
        },
        "checkpoints": {
            "t0": _checkpoint(0),
            "t0_plus_7": _checkpoint(7),
            "t0_plus_30": _checkpoint(30),
            "t0_plus_90": _checkpoint(90),
            "t0_plus_h": _checkpoint(max(0, int(p50.size - 1))),
        },
        "recursive_diag": dict(recursive_diag or {}),
        "crop_override_keys": list((recursive_diag or {}).get("crop_override_keys") or []),
        "crop_override_status": dict((recursive_diag or {}).get("override_status") or {}),
        "crop_override_warnings": list((((recursive_diag or {}).get("override_status") or {}).get("warnings") or [])),
    }


def _cost_latest_and_trend(cost_df: pd.DataFrame, target_year: int) -> Tuple[Optional[float], Optional[float]]:
    if cost_df is None or cost_df.empty:
        return None, None
    if "year_start" not in cost_df.columns or "india_cost_wavg_sample" not in cost_df.columns:
        return None, None

    work = cost_df[["year_start", "india_cost_wavg_sample"]].copy()
    work["year_start"] = pd.to_numeric(work["year_start"], errors="coerce")
    work["india_cost_wavg_sample"] = pd.to_numeric(work["india_cost_wavg_sample"], errors="coerce")
    work = work.dropna(subset=["year_start", "india_cost_wavg_sample"]).sort_values("year_start")
    if work.empty:
        return None, None

    latest = _safe_float(work["india_cost_wavg_sample"].iloc[-1])
    if len(work) < 2:
        return latest, None

    xs = work["year_start"].astype(float).tolist()
    ys = work["india_cost_wavg_sample"].astype(float).tolist()
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-9:
        return latest, None

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    trend_pred = _safe_float(slope * float(target_year) + intercept)
    if trend_pred is not None and trend_pred < 0:
        trend_pred = 0.0
    return latest, trend_pred


def _stabilize_cost_prediction(
    raw_pred: Optional[float],
    cost_df: pd.DataFrame,
    target_year: int,
    cost_cfg: dict,
) -> Tuple[Optional[float], Optional[dict]]:
    pred = _safe_float(raw_pred)
    latest, trend = _cost_latest_and_trend(cost_df, target_year)

    if pred is None:
        if trend is None:
            return None, None
        return trend, {
            "raw": None,
            "latest": latest,
            "trend": trend,
            "deviation_pct": None,
            "applied": ["fallback_trend"],
        }

    cfg = cost_cfg or {}
    if not bool(cfg.get("postprocess_enable", True)):
        return pred, None

    min_ratio = float(cfg.get("postprocess_min_ratio_vs_latest", 0.55))
    max_ratio = float(cfg.get("postprocess_max_ratio_vs_latest", 1.80))
    if min_ratio > max_ratio:
        min_ratio, max_ratio = max_ratio, min_ratio
    blend_weight = float(cfg.get("postprocess_trend_blend_weight", 0.60))
    blend_weight = max(0.0, min(1.0, blend_weight))
    deviation_threshold = float(cfg.get("postprocess_trend_deviation_threshold", 0.35))
    applied: List[str] = []

    deviation_pct = None
    adjusted = pred
    if trend is not None and trend > 0:
        deviation_pct = abs(pred - trend) / trend
        if deviation_pct >= deviation_threshold:
            adjusted = (1.0 - blend_weight) * pred + blend_weight * trend
            applied.append("trend_blend")

    baseline = latest if (latest is not None and latest > 0) else trend
    if baseline is not None and baseline > 0:
        low = baseline * min_ratio
        high = baseline * max_ratio
        clipped = min(max(adjusted, low), high)
        if abs(clipped - adjusted) > 1e-9:
            adjusted = clipped
            applied.append("ratio_clip")

    adjusted = max(0.0, adjusted)
    if not applied:
        return adjusted, None
    return adjusted, {
        "raw": pred,
        "latest": latest,
        "trend": trend,
        "deviation_pct": deviation_pct,
        "applied": applied,
    }


def recommend(env_input: dict, config: dict) -> dict:
    t_all = time.perf_counter()
    strict = _strict_loading(config)
    version = _model_version(config)
    missing_models: List[str] = []

    paths = config["paths"]
    name_map = resolve_names(load_name_map(paths["name_map"]))

    env_key = str(Path(paths["env_predict_py"]).resolve())
    env_mod = _ENV_PREDICTOR_CACHE.get(env_key)
    if env_mod is None:
        env_mod = load_env_predictor(paths["env_predict_py"])
        _ENV_PREDICTOR_CACHE[env_key] = env_mod
    env_mod.MODEL_PATH = paths["env_model_bundle"]

    t_env = time.perf_counter()
    env_out = env_mod.predict_topk(env_input, k=config["scoring"]["max_candidates"])
    best_prob = float(env_out.get("best_prob", 0.0))
    conf_norm = _normalized_confidence(best_prob)
    topk = env_out.get("topk", [])
    logger.info("environment inference done in %.2f ms", (time.perf_counter() - t_env) * 1000.0)

    price_dir = paths["price_dir"]
    cost_df_all = _cost_data_cached(paths["cost_file"])
    yield_cfg = config.get("model", {}).get("yield", {})
    prefer_yield_model = bool(yield_cfg.get("prefer_model_prediction", True))
    yield_table = _yield_table_cached(paths["yield_file"])
    yield_history_df = _yield_history_cached(str(paths.get("yield_history", "")))
    yield_last_year_map = _yield_last_year_map_cached(str(paths.get("yield_history", "")))
    alignment_cfg = config.get("alignment", {})
    time_cfg = config.get("time", {})
    base_target_year = _resolve_target_year(config)
    base_prediction_window = resolve_price_window_from_price_dir(price_dir=price_dir, time_cfg=time_cfg)
    base_window_end = base_prediction_window["end_date"]
    base_price_horizon_days = int(base_prediction_window["price_horizon_days"])
    price_summary_window_days = max(1, int(time_cfg.get("price_summary_window_days", 30)))
    base_cost_cap = _cost_last_year(cost_df_all)
    base_yield_cap = max(yield_last_year_map.values()) if yield_last_year_map else None
    base_window_cap = int(base_window_end.year) if isinstance(base_window_end, pd.Timestamp) else None
    for cap in (base_cost_cap, base_yield_cap, base_window_cap):
        if cap is not None:
            base_target_year = min(int(base_target_year), int(cap))

    model_dir = Path(config["output"]["out_dir"]) / "模型"
    model_dir.mkdir(parents=True, exist_ok=True)

    yield_model, yield_meta = _load_model_and_meta(
        yield_model_candidates(model_dir, yield_cfg, version),
        model_kind="yield",
        strict=False,  # yield table can be used as fallback
        missing_models=missing_models,
    )

    calibrator, calibrator_meta = _load_calibrator(config)
    calibrator_health = _calibrator_health(config, calibrator_meta)
    fallback_mode = _calibrator_fallback_mode(config)
    calibrator_ready = calibrator is not None and calibrator_meta is not None and not calibrator_health["degraded"]
    prob_cfg = config.get("probability", {})
    prob_clip_eps = _safe_float(prob_cfg.get("predict_clip_epsilon"))
    if prob_clip_eps is None:
        prob_clip_eps = 1e-4
    prob_clip_eps = max(0.0, min(0.1, prob_clip_eps))
    min_prob = float(config["scoring"]["min_env_prob"])
    strict_cutoff_split = bool(time_cfg.get("strict_cutoff_split", True))

    price_cfg_base = config.get("model", {}).get("price", {})
    results = []
    for crop, prob in topk:
        if float(prob) < min_prob:
            results.append(
                {
                    "crop": crop,
                    "env_prob": float(prob),
                    "price_pred": None,
                    "price_p10": None,
                    "price_p50": None,
                    "price_p90": None,
                    "price_forecast": [],
                    "price_forecast_mode": None,
                    "price_diagnostics": {
                        "skipped_reason": "env_prob_below_min_threshold",
                        "min_env_prob": float(min_prob),
                    },
                    "price_clip_rate": None,
                    "price_quality_flag": None,
                    "price_blend_weight": None,
                    "cost_pred": None,
                    "cost_pred_raw": None,
                    "cost_adjustment": None,
                    "yield": None,
                    "yield_source": None,
                    "profit": None,
                    "volatility": None,
                    "risk": 0.0,
                    "score": None,
                    "score_raw": None,
                    "prob_best": _clip_probability(float(prob), prob_clip_eps) if fallback_mode == "env_prob" else None,
                    "prob_best_source": "fallback_env_prob" if fallback_mode == "env_prob" else "fallback_none",
                    "price_file": None,
                    "cost_name": None,
                    "target_year": int(base_target_year),
                    "prediction_window": prediction_window_payload(base_prediction_window),
                    "time_alignment": {
                        "target_year": int(base_target_year),
                        "frequency": "year",
                        "strategy": str(alignment_cfg.get("strategy", "trend_extrapolate_with_uncertainty")),
                        "coverage": {
                            "price_last_year": None,
                            "yield_last_year": None,
                            "cost_last_year": None,
                        },
                        "gaps": {
                            "price_gap_years": None,
                            "yield_gap_years": None,
                            "cost_gap_years": None,
                        },
                        "score_weight": 1.0,
                    },
                }
            )
            continue

        mapping = name_map.get(crop, {})
        price_file = mapping.get("price_file", "")
        cost_name = mapping.get("cost_name", "")
        cost_cfg = config.get("model", {}).get("cost", {})
        row_prediction_window = dict(base_prediction_window)
        if not isinstance(row_prediction_window.get("start_date"), pd.Timestamp) or not isinstance(
            row_prediction_window.get("end_date"), pd.Timestamp
        ):
            row_prediction_window = resolve_price_window_from_dates(None, time_cfg=time_cfg)
        row_target_year = int(base_target_year)
        if isinstance(row_prediction_window.get("end_date"), pd.Timestamp):
            row_target_year = min(row_target_year, int(row_prediction_window["end_date"].year))
        price_last_year: Optional[int] = None
        cost_last_year: Optional[int] = None
        yield_last_year: Optional[int] = yield_last_year_map.get(str(crop))

        price_pred = None
        price_p10 = None
        price_p50 = None
        price_p90 = None
        price_forecast: List[dict] = []
        price_forecast_mode: Optional[str] = None
        price_diagnostics: Optional[Dict[str, object]] = None
        price_clip_rate = None
        price_quality_flag = None
        price_blend_weight = None
        vol = None
        price_history_df = pd.DataFrame()
        if price_file:
            try:
                lags = config["time"]["price_lags"]
                windows = config["time"]["price_roll_windows"]
                backtest_days = int(time_cfg.get("price_backtest_days", 180))
                price_cfg = deepcopy(price_cfg_base)
                crop_override_keys: List[str] = []
                override_notices: List[Dict[str, object]] = []
                prediction_mode = _normalize_prediction_mode(price_cfg)
                direct_horizon_pred: Optional[float] = None
                direct_reference_points: List[dict] = []

                price_full_df, _, price_last_year_full = _price_history_context(
                    price_dir,
                    price_file,
                    as_of_date=None,
                )
                if not price_full_df.empty:
                    row_prediction_window = resolve_price_window_from_df(price_full_df, time_cfg=time_cfg)
                    if isinstance(row_prediction_window.get("end_date"), pd.Timestamp):
                        row_target_year = min(row_target_year, int(row_prediction_window["end_date"].year))
                    price_last_year = price_last_year_full

                horizon = int(row_prediction_window.get("price_horizon_days", base_price_horizon_days))
                validation_cutoff = str(
                    row_prediction_window.get("train_validation_cutoff_date")
                    or base_prediction_window.get("train_validation_cutoff_date")
                    or "2020-12-31"
                ).strip()
                as_of_ts = row_prediction_window.get("start_date") if isinstance(
                    row_prediction_window.get("start_date"), pd.Timestamp
                ) else None

                price_history_df, vol, price_last_year = _price_history_context(
                    price_dir,
                    price_file,
                    as_of_date=as_of_ts,
                )

                price_model, price_meta = _load_model_and_meta(
                    price_model_candidates(model_dir, crop, price_cfg, version),
                    model_kind=f"price[{crop}]",
                    strict=strict,
                    missing_models=missing_models,
                )
                direct_feature_space = str((price_meta or {}).get("feature_space", "price"))
                direct_time_raw_mode = (price_meta or {}).get("time_raw_mode")
                direct_include_raw = bool(
                    (price_meta or {}).get(
                        "include_raw_time_features",
                        False if direct_feature_space == "log_price" else bool(price_cfg.get("legacy_include_raw_time_features", True)),
                    )
                )
                if direct_time_raw_mode is None:
                    direct_time_raw_mode = "raw" if direct_include_raw else "none"
                X_recent, _, _ = _price_context(
                    price_dir,
                    price_file,
                    horizon,
                    lags,
                    windows,
                    feature_space=direct_feature_space,
                    include_raw_time_features=direct_include_raw,
                    time_raw_mode=str(direct_time_raw_mode),
                    as_of_date=as_of_ts,
                )
                if price_model is not None and not X_recent.empty:
                    direct_feature_cols = _feature_cols_from_meta_or_model(price_meta, price_model)
                    X_direct = X_recent.reindex(columns=direct_feature_cols, fill_value=0.0) if direct_feature_cols else X_recent
                    pred_raw = _extract_direct_prediction(price_model, X_direct, horizon=horizon)
                    if pred_raw is not None:
                        direct_horizon_pred = max(0.0, float(pred_raw))
                    direct_reference_points = _extract_direct_reference_points(
                        price_model,
                        X_direct,
                        horizon=horizon,
                        meta=price_meta,
                    )

                if (
                    prediction_mode != "direct_horizon_v1"
                    and not price_history_df.empty
                    and bool(price_cfg.get("recursive_enable", True))
                ):
                    rec_diag: Dict[str, object] = {}
                    price_forecast, price_forecast_mode, rec_diag = _recursive_price_forecast(
                        crop=crop,
                        model_dir=model_dir,
                        version=version,
                        strict=strict,
                        missing_models=missing_models,
                        price_cfg=price_cfg,
                        price_history_df=price_history_df,
                        price_file=price_file,
                        horizon_days=horizon,
                        lags=lags,
                        windows=windows,
                        backtest_days=backtest_days,
                        validation_cutoff=validation_cutoff,
                        strict_cutoff_split=strict_cutoff_split,
                        as_of_date=as_of_ts,
                        prediction_mode=prediction_mode,
                        anchor_end_price=direct_horizon_pred,
                        direct_reference_points=direct_reference_points,
                    )
                    if crop_override_keys:
                        rec_diag["crop_override_keys"] = list(crop_override_keys)
                    override_status = rec_diag.get("override_status", {}) if isinstance(rec_diag, dict) else {}
                    if isinstance(override_status, dict):
                        crop_override_keys = list(override_status.get("applied_keys") or crop_override_keys)
                        override_notices = list(override_status.get("warnings") or [])
                    if price_forecast:
                        price_summary = summarize_forecast_tail(
                            price_forecast,
                            window_days=price_summary_window_days,
                            fallback_price=price_pred,
                        )
                        price_p10 = _safe_float(price_summary.get("price_p10"))
                        price_p50 = _safe_float(price_summary.get("price_p50"))
                        price_p90 = _safe_float(price_summary.get("price_p90"))
                        price_pred_summary = _safe_float(price_summary.get("price_pred"))
                        if price_pred_summary is not None:
                            price_pred = max(0.0, float(price_pred_summary))
                        price_clip_rate = _safe_float(rec_diag.get("clip_rate"))
                        price_quality_flag = str(rec_diag.get("quality_flag") or "OK")
                        price_blend_weight = _safe_float(rec_diag.get("blend_weight_mean"))
                        price_diagnostics = rec_diag.get("online_price_diagnostics", rec_diag)
                        if isinstance(price_diagnostics, dict):
                            price_diagnostics["summary_window_days"] = int(price_summary_window_days)
                            price_diagnostics["summary_window_used_days"] = int(price_summary.get("used_days") or 0)
                        if isinstance(price_diagnostics, dict) and crop_override_keys:
                            price_diagnostics["crop_override_keys"] = list(crop_override_keys)
                        if isinstance(price_diagnostics, dict) and override_notices:
                            price_diagnostics["crop_override_warnings"] = override_notices
                    else:
                        fallback_reason = str(rec_diag.get("fallback_reason") or "")
                        if fallback_reason:
                            price_quality_flag = f"FALLBACK:{fallback_reason}"
                            price_diagnostics = rec_diag.get("online_price_diagnostics", rec_diag)
                            if isinstance(price_diagnostics, dict) and crop_override_keys:
                                price_diagnostics["crop_override_keys"] = list(crop_override_keys)
                            if isinstance(price_diagnostics, dict) and override_notices:
                                price_diagnostics["crop_override_warnings"] = override_notices

                if price_pred is None:
                    if direct_horizon_pred is not None:
                        price_pred = direct_horizon_pred
                        price_p50 = direct_horizon_pred
                        price_forecast_mode = price_forecast_mode or "direct_horizon_fallback"
                        if price_diagnostics is None:
                            price_diagnostics = {
                                "prediction_mode": prediction_mode,
                                "fallback_reason": "direct_horizon_only",
                            }
                        if isinstance(price_diagnostics, dict) and crop_override_keys:
                            price_diagnostics["crop_override_keys"] = list(crop_override_keys)
                        if isinstance(price_diagnostics, dict) and override_notices:
                            price_diagnostics["crop_override_warnings"] = override_notices
            except FileNotFoundError:
                logger.warning("price history file missing for crop=%s price_file=%s", crop, price_file)
            except Exception:
                logger.exception("failed computing price for crop=%s", crop)

        cost_pred = None
        cost_pred_raw = None
        cost_adjustment = None
        if cost_name:
            try:
                cost_df = cost_df_all[cost_df_all["crop_name"].str.lower() == cost_name.lower()].copy()
                if cost_df.empty:
                    synonyms = {"lentil": "Masur", "rice": "Paddy", "chickpea": "Gram"}
                    alt = synonyms.get(crop, None)
                    if alt:
                        cost_df = cost_df_all[cost_df_all["crop_name"].str.lower() == alt.lower()].copy()

                if not cost_df.empty:
                    preferred_end_year = int(row_target_year)
                    if isinstance(row_prediction_window.get("end_date"), pd.Timestamp):
                        preferred_end_year = min(preferred_end_year, int(row_prediction_window["end_date"].year))
                    cost_df["year_start"] = pd.to_numeric(cost_df["year_start"], errors="coerce")
                    cost_df = cost_df.dropna(subset=["year_start"])
                    cost_window = resolve_year_window_from_series(
                        cost_df["year_start"].tolist(),
                        time_cfg=time_cfg,
                        window_years_key="cost_prediction_window_years",
                        preferred_end_year=preferred_end_year,
                    )
                    cost_df = cost_df[cost_df["year_start"] <= float(cost_window["end_year"])].copy()
                    if cost_df.empty:
                        raise ValueError(f"no historical cost rows in year window for crop={crop}")
                    cost_last_year = _cost_last_year(cost_df)
                    if cost_last_year is not None:
                        row_target_year = min(int(row_target_year), int(cost_last_year))
                    cost_model, _ = _load_model_and_meta(
                        cost_model_candidates(model_dir, crop, config["model"]["cost"], version),
                        model_kind=f"cost[{crop}]",
                        strict=strict,
                        missing_models=missing_models,
                    )
                    if cost_model is not None:
                        cost_pred_raw = predict_cost(
                            cost_model,
                            row_target_year,
                            cost_history_df=cost_df,
                            price_history_df=price_history_df,
                            yield_history_df=yield_history_df,
                            crop=crop,
                            cost_name=cost_name,
                        )
                        cost_pred, cost_adjustment = _stabilize_cost_prediction(
                            cost_pred_raw,
                            cost_df,
                            row_target_year,
                            cost_cfg,
                        )
            except Exception:
                logger.exception("failed computing cost for crop=%s", crop)

        yld = None
        yld_source = None
        if yield_last_year is not None:
            row_target_year = min(int(row_target_year), int(yield_last_year))
        if prefer_yield_model:
            try:
                yld = _predict_yield_from_model(
                    crop=crop,
                    target_year=int(row_target_year),
                    env_input=env_input,
                    yield_model=yield_model,
                    yield_meta=yield_meta,
                    yield_history_df=yield_history_df,
                    yield_cfg=yield_cfg,
                )
                if yld is not None:
                    yld_source = "model"
            except Exception:
                logger.exception("failed computing yield for crop=%s", crop)
                yld = None
                yld_source = None
        if yld is None:
            yld = yield_table.yields.get(crop, None)
            if yld is not None:
                yld_source = "table"
        if yld is None and not prefer_yield_model:
            try:
                yld = _predict_yield_from_model(
                    crop=crop,
                    target_year=int(row_target_year),
                    env_input=env_input,
                    yield_model=yield_model,
                    yield_meta=yield_meta,
                    yield_history_df=yield_history_df,
                    yield_cfg=yield_cfg,
                )
                if yld is not None:
                    yld_source = "model"
            except Exception:
                logger.exception("failed computing yield for crop=%s", crop)
                yld = None
                yld_source = None

        profit = None
        if price_pred is not None and cost_pred is not None and yld is not None:
            profit = price_pred * yld - cost_pred

        risk = risk_score(vol if vol is not None else 0.0, conf_norm, env_out.get("warnings"))
        score = None
        score_raw = None
        price_gap = None if price_last_year is None else max(0, int(row_target_year) - int(price_last_year))
        yield_gap = None if yield_last_year is None else max(0, int(row_target_year) - int(yield_last_year))
        cost_gap = None if cost_last_year is None else max(0, int(row_target_year) - int(cost_last_year))
        alignment_weight = _alignment_score_weight(
            gaps=[price_gap, yield_gap, cost_gap],
            alignment_cfg=alignment_cfg,
        )

        if profit is not None:
            score_raw = profit * (1.0 - config["scoring"]["risk_penalty"] * risk) * (
                float(prob) ** config["scoring"]["env_prob_weight"]
            )
            score = score_raw * alignment_weight

        prob_best = None
        prob_best_source = "none"
        if calibrator_ready:
            feat_vals = {
                "prob_env": float(prob),
                "profit_pred": profit,
                "price_pred": price_pred,
                "yield_pred": yld,
                "cost_pred": cost_pred,
                "cost_pred_raw": cost_pred_raw,
                "cost_adjustment": cost_adjustment,
                "risk_score": risk,
                "score_total": score,
                "volatility": vol,
                "ood_flag": 1.0 if env_out.get("warnings") else 0.0,
                "env_confidence": _encode_confidence(conf_norm),
            }
            feature_cols = calibrator_meta.get("feature_cols", [])
            X_cal = (
                pd.DataFrame([{c: feat_vals.get(c, 0.0) for c in feature_cols}])
                if feature_cols
                else pd.DataFrame([feat_vals])
            )
            X_cal = X_cal.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            try:
                prob_best = _clip_probability(float(calibrator.predict_proba(X_cal)[0, 1]), prob_clip_eps)
                prob_best_source = "calibrator"
            except Exception:
                logger.exception("failed calibrator inference for crop=%s", crop)
                if fallback_mode == "env_prob":
                    prob_best = _clip_probability(float(prob), prob_clip_eps)
                    prob_best_source = "fallback_env_prob"
                else:
                    prob_best = None
                    prob_best_source = "fallback_none"
        else:
            if fallback_mode == "env_prob":
                prob_best = _clip_probability(float(prob), prob_clip_eps)
                prob_best_source = "fallback_env_prob"
            else:
                prob_best = None
                prob_best_source = "fallback_none"

        results.append(
            {
                "crop": crop,
                "env_prob": float(prob),
                "price_pred": price_pred,
                "price_p10": price_p10,
                "price_p50": price_p50 if price_p50 is not None else price_pred,
                "price_p90": price_p90,
                "price_forecast": price_forecast,
                "price_forecast_mode": price_forecast_mode,
                "price_diagnostics": price_diagnostics,
                "price_clip_rate": price_clip_rate,
                "price_quality_flag": price_quality_flag,
                "price_blend_weight": price_blend_weight,
                "cost_pred": cost_pred,
                "cost_pred_raw": cost_pred_raw,
                "cost_adjustment": cost_adjustment,
                "yield": yld,
                "yield_source": yld_source,
                "profit": profit,
                "volatility": vol,
                "risk": risk,
                "score": score,
                "score_raw": score_raw,
                "prob_best": prob_best,
                "prob_best_source": prob_best_source,
                "price_file": price_file,
                "cost_name": cost_name,
                "target_year": int(row_target_year),
                "prediction_window": prediction_window_payload(row_prediction_window),
                "time_alignment": {
                    "target_year": int(row_target_year),
                    "frequency": "year",
                    "strategy": str(alignment_cfg.get("strategy", "trend_extrapolate_with_uncertainty")),
                    "coverage": {
                        "price_last_year": price_last_year,
                        "yield_last_year": yield_last_year,
                        "cost_last_year": cost_last_year,
                    },
                    "gaps": {
                        "price_gap_years": price_gap,
                        "yield_gap_years": yield_gap,
                        "cost_gap_years": cost_gap,
                    },
                    "score_weight": alignment_weight,
                },
            }
        )

    score_source, score_weights = _apply_release_score_fusion(results, config)
    candidates_before_filter = len(results)
    results = [r for r in results if r["env_prob"] >= min_prob]
    candidates_after_filter = len(results)

    sort_by = prob_cfg.get("sort_by", "score_total")
    if sort_by == "prob_best":
        results.sort(key=lambda x: (x["prob_best"] is None, -(x["prob_best"] or -1e9)))
    else:
        results.sort(key=lambda x: (x["score"] is None, -(x["score"] or -1e9)))

    final_topk = [{"crop": r["crop"], "env_prob": float(r["env_prob"])} for r in results[:8]]
    runtime_target_year = max(
        (int(r.get("target_year")) for r in results if r.get("target_year") is not None),
        default=int(base_target_year),
    )
    runtime_prediction_window = (
        results[0].get("prediction_window")
        if results and isinstance(results[0].get("prediction_window"), dict)
        else prediction_window_payload(base_prediction_window)
    )

    elapsed_ms = (time.perf_counter() - t_all) * 1000.0
    logger.info(
        "recommendation done in %.2f ms; candidates=%d; results=%d; missing_models=%d",
        elapsed_ms,
        len(topk),
        len(results),
        len(missing_models),
    )

    return {
        "env": env_out,
        "env_confidence_norm": conf_norm,
        "final_topk": final_topk,
        "results": results,
        "yield_missing": yield_table.missing,
        "runtime": {
            "elapsed_ms": round(elapsed_ms, 3),
            "model_version": version,
            "missing_models": missing_models,
            "score_source": score_source,
            "score_weights": score_weights or {
                "risk_penalty": float(config["scoring"]["risk_penalty"]),
                "env_prob_weight": float(config["scoring"]["env_prob_weight"]),
            },
            "filters": {
                "min_env_prob": min_prob,
                "before": candidates_before_filter,
                "after": candidates_after_filter,
            },
            "calibrator": {
                "enabled": calibrator is not None and calibrator_meta is not None,
                "degraded": bool(calibrator_health.get("degraded", False)),
                "reasons": calibrator_health.get("reasons", []),
                "warnings": calibrator_health.get("warnings", []),
                "fallback_mode": fallback_mode,
                "history_stats": calibrator_health.get("stats", {}),
            },
            "time_alignment": {
                "target_year": int(runtime_target_year),
                "frequency": "year",
                "strategy": str(alignment_cfg.get("strategy", "trend_extrapolate_with_uncertainty")),
                "score_penalty_per_year": float(alignment_cfg.get("score_penalty_per_year", 0.05)),
                "max_score_penalty": float(alignment_cfg.get("max_score_penalty", 0.35)),
            },
            "prediction_window": runtime_prediction_window,
        },
    }
