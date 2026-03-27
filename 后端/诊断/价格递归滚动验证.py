from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from 后端.兼容层 import tune_loaded_model
from 后端.数据加载 import load_config, load_name_map, load_price_series, resolve_names
from 后端.模型产物 import price_model_candidates, price_recursive_model_candidates
from 后端.价格作物覆盖 import resolve_price_cfg_for_crop
from 后端.模型.价格模型 import train_one_crop as train_price_one_crop
from 后端.价格递归预测 import recursive_multi_step_forecast
from 后端.时间策略 import resolve_price_window_from_df
from 后端.特征工程 import make_recent_features


logger = logging.getLogger(__name__)

H_BUCKETS = (7, 30, 90, 180)
TREND_METRIC_DEFAULTS = {
    "a_t0_jump_median_max": 0.03,
    "b_early_drop_median_min": -0.08,
    "c_end_ratio_median_min": 0.85,
    "c_end_ratio_median_max": 1.15,
    "d_direction_mismatch_ratio_max": 0.50,
    "e_coverage_global_min": 0.75,
    "e_interval_non_decreasing_ratio_min": 0.90,
    "majority_crop_ratio_min": 0.80,
}


def _safe_float(v: Any) -> Optional[float]:
    try:
        out = float(v)
    except Exception:
        return None
    if pd.isna(out):
        return None
    return out


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


def _to_naive_day(values: Any) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(parsed, pd.Series):
        return parsed.dt.tz_convert(None).dt.normalize()
    return pd.Series(dtype="datetime64[ns]")


def _normalize_prediction_mode(price_cfg: dict) -> str:
    mode = str((price_cfg or {}).get("prediction_mode", "return_recursive_v3")).strip().lower()
    if mode not in {"return_recursive_v3", "return_recursive_v2", "price_recursive_v1", "direct_horizon_v1"}:
        return "return_recursive_v3"
    return mode


def _meta_target_mode(meta: Optional[dict]) -> str:
    if not isinstance(meta, dict):
        return ""
    val = str(meta.get("target_mode", "")).strip().lower()
    if not val and isinstance(meta.get("artifacts"), dict):
        val = str(meta["artifacts"].get("target_mode", "")).strip().lower()
    if val in {"log_return", "return", "ret", "delta"}:
        return "log_return"
    if val:
        return "price"
    return ""


def _feature_cols_from_meta_or_model(meta: Optional[dict], model: Optional[object]) -> List[str]:
    cols: List[str] = []
    if isinstance(meta, dict):
        raw = meta.get("feature_cols", [])
        if isinstance(raw, list):
            cols = [str(x) for x in raw if str(x).strip()]
    if not cols and model is not None:
        maybe = getattr(model, "feature_cols", None)
        if isinstance(maybe, list):
            cols = [str(x) for x in maybe if str(x).strip()]
    return cols


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return {}


def _resolve_meta_path(meta_path: Path) -> Path:
    candidates = [
        meta_path,
        meta_path.with_name(meta_path.name.replace("_指标.json", "_metrics.json")) if meta_path.name.endswith("_指标.json") else meta_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return meta_path


def _load_first_available_model(candidates: Iterable[Tuple[Path, Path]]) -> Tuple[Optional[object], Optional[dict]]:
    for model_path, meta_path in candidates:
        if not model_path.exists():
            continue
        try:
            model = tune_loaded_model(load(model_path))
        except Exception:
            continue
        return model, _load_json(_resolve_meta_path(meta_path))
    return None, None


def _build_step_cfg(price_cfg: dict) -> dict:
    cfg = dict(price_cfg or {})
    cfg["regressor"] = str(price_cfg.get("recursive_fallback_regressor", "hgb")).strip().lower() or "hgb"
    cfg["max_iter"] = int(price_cfg.get("recursive_fallback_max_iter", 260))
    cfg["max_depth"] = int(price_cfg.get("recursive_fallback_max_depth", price_cfg.get("max_depth", 8) or 8))
    cfg["learning_rate"] = float(price_cfg.get("recursive_fallback_learning_rate", price_cfg.get("learning_rate", 0.04)))
    cfg["l2_regularization"] = float(
        price_cfg.get("recursive_fallback_l2_regularization", price_cfg.get("l2_regularization", 1e-3))
    )
    cfg["use_recency_weight"] = bool(price_cfg.get("use_recency_weight", True))
    cfg["recency_halflife_days"] = float(price_cfg.get("recency_halflife_days", 365))
    cfg["n_jobs"] = int(price_cfg.get("n_jobs", 1))
    cfg["verbose"] = False
    mode = _normalize_prediction_mode(cfg)
    if mode in {"return_recursive_v3", "return_recursive_v2"}:
        cfg["target_mode"] = "log_return"
        cfg["feature_space"] = "log_price"
        cfg["time_raw_mode"] = "none"
        cfg["include_raw_time_features"] = False
        cfg["target_transform"] = "none"
    else:
        cfg["target_mode"] = "price"
        cfg["feature_space"] = "price"
        cfg["time_raw_mode"] = str(price_cfg.get("legacy_time_raw_mode", "raw"))
        cfg["include_raw_time_features"] = bool(price_cfg.get("legacy_include_raw_time_features", True))
    return cfg


def _load_step_model(
    *,
    model_dir: Path,
    crop: str,
    price_cfg: dict,
    version: str,
    history_df: pd.DataFrame,
    lags: List[int],
    windows: List[int],
    backtest_days: int,
    validation_cutoff: str,
    strict_cutoff_split: bool,
) -> Tuple[Optional[object], Optional[dict], str]:
    requested_mode = _normalize_prediction_mode(price_cfg)

    for model_path, meta_path in price_recursive_model_candidates(model_dir, crop, price_cfg, version):
        if not model_path.exists():
            continue
        try:
            model = tune_loaded_model(load(model_path))
        except Exception:
            continue
        meta = _load_json(_resolve_meta_path(meta_path))
        target_mode = _meta_target_mode(meta)
        if requested_mode in {"return_recursive_v3", "return_recursive_v2"} and target_mode != "log_return":
            continue
        if requested_mode == "price_recursive_v1" and target_mode == "log_return":
            continue
        return model, meta, "artifact"

    step_cfg = _build_step_cfg(price_cfg)
    try:
        res = train_price_one_crop(
            history_df,
            step_cfg,
            lags,
            windows,
            horizon=1,
            backtest_days=max(30, int(backtest_days)),
            test_ratio=step_cfg.get("test_ratio", price_cfg.get("test_ratio")),
            validation_cutoff=validation_cutoff,
            strict_cutoff_split=strict_cutoff_split,
            verbose=False,
            label=f"roll_eval:{crop}",
        )
    except Exception:
        return None, None, "missing"

    meta = {
        "feature_cols": list(res.feature_cols or []),
        "target_mode": step_cfg.get("target_mode"),
        "feature_space": step_cfg.get("feature_space"),
        "time_raw_mode": step_cfg.get("time_raw_mode"),
        "include_raw_time_features": bool(step_cfg.get("include_raw_time_features", False)),
        "artifacts": res.artifacts or {},
    }
    return res.model, meta, "inline"


def _daily_price_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "date" not in df.columns or "modal_price" not in df.columns:
        return pd.Series(dtype=float)

    work = df[["date", "modal_price"]].copy()
    work["date"] = _to_naive_day(work["date"])
    work["modal_price"] = pd.to_numeric(work["modal_price"], errors="coerce")
    work = work.dropna(subset=["date", "modal_price"]).sort_values("date")
    if work.empty:
        return pd.Series(dtype=float)

    s = pd.Series(work["modal_price"].to_numpy(dtype=float), index=work["date"])
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.empty:
        return pd.Series(dtype=float)
    return s.astype(float)


def _history_df_from_series(s: pd.Series, end_date: pd.Timestamp) -> pd.DataFrame:
    hist = s.loc[:end_date]
    return pd.DataFrame({"date": hist.index, "modal_price": hist.to_numpy(dtype=float)})


def _effective_anchor(anchor_end_price: Optional[float], history_df: pd.DataFrame, price_cfg: dict) -> Optional[float]:
    anchor = _safe_float(anchor_end_price)
    if anchor is None or anchor <= 0:
        return None
    try:
        hist = pd.to_numeric(history_df.get("modal_price"), errors="coerce").dropna().tail(180)
        if len(hist) < 20:
            return anchor
        med = _safe_float(hist.median())
        if med is None or med <= 0:
            return anchor
        ratio = anchor / med
        lo = float(price_cfg.get("recursive_anchor_valid_ratio_low", 0.7))
        hi = float(price_cfg.get("recursive_anchor_valid_ratio_high", 1.4))
        if lo > hi:
            lo, hi = hi, lo
        if ratio < lo or ratio > hi:
            return None
        return anchor
    except Exception:
        return None


def _direct_predict(
    *,
    model: Optional[object],
    meta: Optional[dict],
    history_df: pd.DataFrame,
    horizon_days: int,
    lags: List[int],
    windows: List[int],
    price_cfg: dict,
) -> Optional[float]:
    if model is None or history_df is None or history_df.empty:
        return None
    target_mode = _meta_target_mode(meta)
    feature_space = str((meta or {}).get("feature_space", "price")).strip().lower() or "price"
    include_raw = bool(
        (meta or {}).get(
            "include_raw_time_features",
            False if feature_space == "log_price" else bool(price_cfg.get("legacy_include_raw_time_features", True)),
        )
    )
    time_raw_mode = (meta or {}).get("time_raw_mode")
    if time_raw_mode is None:
        time_raw_mode = "raw" if include_raw else "none"

    X_recent = make_recent_features(
        history_df,
        "modal_price",
        horizon_days,
        lags,
        windows,
        feature_space=feature_space,
        include_raw_time_features=include_raw,
        time_raw_mode=str(time_raw_mode),
    )
    if X_recent.empty:
        return None

    feat_cols = _feature_cols_from_meta_or_model(meta, model)
    X_model = X_recent.reindex(columns=feat_cols, fill_value=0.0) if feat_cols else X_recent
    try:
        raw = _safe_float(model.predict(X_model)[0])
    except Exception:
        return None
    if raw is None:
        return None

    latest = _safe_float(pd.to_numeric(history_df["modal_price"], errors="coerce").iloc[-1])
    if target_mode == "log_return":
        if latest is None or latest <= 0:
            return None
        return float(np.exp(np.log(max(latest, 1e-6)) + float(raw)))
    return float(max(0.0, raw))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if y_true.size == 0 or y_pred.size == 0:
        return None
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs(y_pred - y_true) / denom))


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if y_true.size == 0 or y_pred.size == 0:
        return None
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-6)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def _coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Optional[float]:
    if y_true.size == 0 or lo.size == 0 or hi.size == 0:
        return None
    mask = np.isfinite(y_true) & np.isfinite(lo) & np.isfinite(hi)
    if int(mask.sum()) == 0:
        return None
    covered = (y_true[mask] >= lo[mask]) & (y_true[mask] <= hi[mask])
    return float(np.mean(covered))


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


def _non_decreasing_ratio(values: np.ndarray, *, tol: float = 1e-10) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 1.0
    dif = np.diff(arr)
    return float(np.mean(dif >= -abs(float(tol))))


def _interval_non_decreasing_ratio(p10: np.ndarray, p90: np.ndarray) -> float:
    if p10.size == 0 or p90.size == 0 or p10.size != p90.size:
        return 0.0
    width = p90 - p10
    return _non_decreasing_ratio(width)


def _bucket_metrics(actual: np.ndarray, p50: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for b in H_BUCKETS:
        if actual.size < b or p50.size < b:
            out[f"path_mape_h{b}"] = None
            out[f"path_smape_h{b}"] = None
            out[f"coverage_p10p90_h{b}"] = None
            continue
        a = actual[:b]
        p = p50[:b]
        lo = p10[:b] if p10.size >= b else np.asarray([], dtype=float)
        hi = p90[:b] if p90.size >= b else np.asarray([], dtype=float)
        out[f"path_mape_h{b}"] = _mape(a, p)
        out[f"path_smape_h{b}"] = _smape(a, p)
        out[f"coverage_p10p90_h{b}"] = _coverage(a, lo, hi)
    return out


def _quantile_pack(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    if values is None:
        raw_values: List[Any] = []
    elif isinstance(values, pd.Series):
        raw_values = values.tolist()
    elif isinstance(values, np.ndarray):
        raw_values = values.tolist()
    elif isinstance(values, (list, tuple)):
        raw_values = list(values)
    else:
        raw_values = [values]

    arr = pd.to_numeric(pd.Series(raw_values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def _summarize_rows(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "n_rolls": 0,
            "jump_1_abs": _quantile_pack([]),
            "jump_1_pos_ratio": None,
            "end_shift_last": _quantile_pack([]),
            "end_shift_last_abs": _quantile_pack([]),
            "end_bias": _quantile_pack([]),
            "end_bias_abs": _quantile_pack([]),
            "clip_rate": _quantile_pack([]),
            "gap_rd": _quantile_pack([]),
            "gap_rd_gt_30_ratio": None,
            "coverage_p10p90_per_origin": _quantile_pack([]),
            "coverage_p10p90_global": None,
            "horizon_buckets": {},
        }

    jump = pd.to_numeric(df["jump_1"], errors="coerce")
    end_shift = pd.to_numeric(df["end_shift_last"], errors="coerce")
    end_bias = pd.to_numeric(df["end_bias"], errors="coerce")
    clip_rate = pd.to_numeric(df["clip_rate"], errors="coerce")
    gap = pd.to_numeric(df["gap_rd"], errors="coerce")
    cov = pd.to_numeric(df["coverage_p10p90"], errors="coerce")

    cov_n = pd.to_numeric(df.get("coverage_n_points"), errors="coerce").fillna(0.0)
    valid_cov = cov.notna() & (cov_n > 0)
    if bool(valid_cov.any()):
        weighted_cov = float(np.average(cov[valid_cov].to_numpy(dtype=float), weights=cov_n[valid_cov].to_numpy(dtype=float)))
    else:
        weighted_cov = None

    out: Dict[str, Any] = {
        "n_rolls": int(len(df)),
        "jump_1_abs": _quantile_pack(np.abs(jump)),
        "jump_1_pos_ratio": float(np.mean((jump > 0).dropna())) if int(jump.notna().sum()) else None,
        "end_shift_last": _quantile_pack(end_shift),
        "end_shift_last_abs": _quantile_pack(np.abs(end_shift)),
        "end_bias": _quantile_pack(end_bias),
        "end_bias_abs": _quantile_pack(np.abs(end_bias)),
        "clip_rate": _quantile_pack(clip_rate),
        "gap_rd": _quantile_pack(gap),
        "gap_rd_gt_30_ratio": float(np.mean((gap > 0.30).dropna())) if int(gap.notna().sum()) else None,
        "coverage_p10p90_per_origin": _quantile_pack(cov),
        "coverage_p10p90_global": weighted_cov,
        "horizon_buckets": {},
    }

    for b in H_BUCKETS:
        out["horizon_buckets"][str(b)] = {
            "path_mape": _quantile_pack(pd.to_numeric(df.get(f"path_mape_h{b}"), errors="coerce")),
            "path_smape": _quantile_pack(pd.to_numeric(df.get(f"path_smape_h{b}"), errors="coerce")),
            "coverage_p10p90": _quantile_pack(pd.to_numeric(df.get(f"coverage_p10p90_h{b}"), errors="coerce")),
        }

    return out


def _trend_thresholds(price_cfg: Dict[str, Any]) -> Dict[str, float]:
    base = dict(TREND_METRIC_DEFAULTS)
    if not isinstance(price_cfg, dict):
        return base
    base["a_t0_jump_median_max"] = float(price_cfg.get("trend_a_t0_jump_median_max", base["a_t0_jump_median_max"]))
    base["b_early_drop_median_min"] = float(price_cfg.get("trend_b_early_drop_median_min", base["b_early_drop_median_min"]))
    base["c_end_ratio_median_min"] = float(price_cfg.get("trend_c_end_ratio_median_min", base["c_end_ratio_median_min"]))
    base["c_end_ratio_median_max"] = float(price_cfg.get("trend_c_end_ratio_median_max", base["c_end_ratio_median_max"]))
    if base["c_end_ratio_median_min"] > base["c_end_ratio_median_max"]:
        base["c_end_ratio_median_min"], base["c_end_ratio_median_max"] = (
            base["c_end_ratio_median_max"],
            base["c_end_ratio_median_min"],
        )
    base["d_direction_mismatch_ratio_max"] = float(
        price_cfg.get("trend_d_direction_mismatch_ratio_max", base["d_direction_mismatch_ratio_max"])
    )
    base["e_coverage_global_min"] = float(price_cfg.get("trend_e_coverage_global_min", base["e_coverage_global_min"]))
    base["e_interval_non_decreasing_ratio_min"] = float(
        price_cfg.get(
            "trend_e_interval_non_decreasing_ratio_min",
            base["e_interval_non_decreasing_ratio_min"],
        )
    )
    base["majority_crop_ratio_min"] = float(price_cfg.get("trend_majority_crop_ratio_min", base["majority_crop_ratio_min"]))
    return base


def _summarize_trend_rows(df: pd.DataFrame, thresholds: Dict[str, float]) -> Dict[str, Any]:
    if df.empty:
        return {
            "n_rolls": 0,
            "t0_jump_abs_median": None,
            "early_drop_median": None,
            "end_ratio_median": None,
            "direction_mismatch_ratio": None,
            "coverage_p10p90_global": None,
            "interval_non_decreasing_ratio": None,
            "pass_A": False,
            "pass_B": False,
            "pass_C": False,
            "pass_D": False,
            "pass_E": False,
            "pass_BCD": False,
            "pass_all": False,
            "fail_reasons": ["empty_rows"],
        }

    jump_abs = pd.to_numeric(df.get("t0_jump_abs_ratio"), errors="coerce")
    early_drop = pd.to_numeric(df.get("early_drop"), errors="coerce")
    end_ratio = pd.to_numeric(df.get("end_ratio"), errors="coerce")
    mismatch = pd.to_numeric(df.get("direction_mismatch_flag"), errors="coerce")
    interval_ok = pd.to_numeric(df.get("interval_non_decreasing_flag"), errors="coerce")
    cov = pd.to_numeric(df.get("coverage_p10p90"), errors="coerce")
    cov_n = pd.to_numeric(df.get("coverage_n_points"), errors="coerce").fillna(0.0)

    valid_cov = cov.notna() & (cov_n > 0)
    if bool(valid_cov.any()):
        coverage_global = float(
            np.average(cov[valid_cov].to_numpy(dtype=float), weights=cov_n[valid_cov].to_numpy(dtype=float))
        )
    else:
        coverage_global = None

    jump_med = float(np.median(jump_abs.dropna())) if int(jump_abs.notna().sum()) else None
    early_med = float(np.median(early_drop.dropna())) if int(early_drop.notna().sum()) else None
    end_med = float(np.median(end_ratio.dropna())) if int(end_ratio.notna().sum()) else None
    mismatch_ratio = float(np.mean(mismatch.dropna())) if int(mismatch.notna().sum()) else None
    interval_ratio = float(np.mean(interval_ok.dropna())) if int(interval_ok.notna().sum()) else None

    pass_a = bool(jump_med is not None and jump_med < float(thresholds["a_t0_jump_median_max"]))
    pass_b = bool(early_med is not None and early_med >= float(thresholds["b_early_drop_median_min"]))
    pass_c = bool(
        end_med is not None
        and end_med >= float(thresholds["c_end_ratio_median_min"])
        and end_med <= float(thresholds["c_end_ratio_median_max"])
    )
    pass_d = bool(
        mismatch_ratio is not None
        and mismatch_ratio <= float(thresholds["d_direction_mismatch_ratio_max"])
    )
    pass_e = bool(
        coverage_global is not None
        and coverage_global >= float(thresholds["e_coverage_global_min"])
        and interval_ratio is not None
        and interval_ratio >= float(thresholds["e_interval_non_decreasing_ratio_min"])
    )

    fail_reasons: List[str] = []
    attribution_hints: List[str] = []
    if not pass_a:
        fail_reasons.append("A_t0_jump")
        attribution_hints.append("tighten trend_guard_t0_jump_max or increase t0 jump damping")
    if not pass_b:
        fail_reasons.append("B_early_drop")
        attribution_hints.append("reduce seasonal drift weight or raise trend_guard_early_drop_floor")
    if not pass_c:
        fail_reasons.append("C_end_ratio")
        if end_med is not None and end_med > float(thresholds["c_end_ratio_median_max"]):
            attribution_hints.append("end_ratio too high: strengthen end tail guard or reduce terminal/seasonal pull-up")
        elif end_med is not None and end_med < float(thresholds["c_end_ratio_median_min"]):
            attribution_hints.append("end_ratio too low: relax low-end pull-down and/or increase terminal support")
    if not pass_d:
        fail_reasons.append("D_direction")
        attribution_hints.append("increase direction guard horizon or reduce opposite-direction drift")
    if not pass_e:
        fail_reasons.append("E_uncertainty")
        attribution_hints.append("raise conformal width scaling or enforce non-decreasing q-step")

    return {
        "n_rolls": int(len(df)),
        "t0_jump_abs_median": jump_med,
        "early_drop_median": early_med,
        "end_ratio_median": end_med,
        "direction_mismatch_ratio": mismatch_ratio,
        "coverage_p10p90_global": coverage_global,
        "interval_non_decreasing_ratio": interval_ratio,
        "pass_A": pass_a,
        "pass_B": pass_b,
        "pass_C": pass_c,
        "pass_D": pass_d,
        "pass_E": pass_e,
        "pass_BCD": bool(pass_b and pass_c and pass_d),
        "pass_all": bool(pass_a and pass_b and pass_c and pass_d and pass_e),
        "fail_reasons": fail_reasons,
        "attribution_hints": attribution_hints,
    }


def _summarize_trend_overall(crop_summaries: List[Dict[str, Any]], thresholds: Dict[str, float]) -> Dict[str, Any]:
    valid = [x for x in crop_summaries if bool(x.get("ok")) and isinstance(x.get("trend"), dict)]
    if not valid:
        return {
            "n_crops": 0,
            "pass_BCD_count": 0,
            "pass_BCD_ratio": 0.0,
            "pass_BCD_required_ratio": float(thresholds["majority_crop_ratio_min"]),
            "pass_BCD_majority": False,
            "failing_crops": [],
        }

    bcd_count = int(sum(1 for x in valid if bool((x.get("trend") or {}).get("pass_BCD"))))
    bcd_ratio = float(bcd_count) / float(len(valid))
    pass_a_ratio = float(np.mean([1.0 if bool((x.get("trend") or {}).get("pass_A")) else 0.0 for x in valid]))
    pass_b_ratio = float(np.mean([1.0 if bool((x.get("trend") or {}).get("pass_B")) else 0.0 for x in valid]))
    pass_c_ratio = float(np.mean([1.0 if bool((x.get("trend") or {}).get("pass_C")) else 0.0 for x in valid]))
    pass_d_ratio = float(np.mean([1.0 if bool((x.get("trend") or {}).get("pass_D")) else 0.0 for x in valid]))
    pass_e_ratio = float(np.mean([1.0 if bool((x.get("trend") or {}).get("pass_E")) else 0.0 for x in valid]))
    failing = []
    for item in valid:
        trend = item.get("trend") or {}
        if bool(trend.get("pass_BCD")):
            continue
        failing.append(
            {
                "crop": item.get("crop"),
                "fail_reasons": trend.get("fail_reasons", []),
                "attribution_hints": trend.get("attribution_hints", []),
                "early_drop_median": trend.get("early_drop_median"),
                "end_ratio_median": trend.get("end_ratio_median"),
                "direction_mismatch_ratio": trend.get("direction_mismatch_ratio"),
            }
        )

    return {
        "n_crops": int(len(valid)),
        "pass_BCD_count": int(bcd_count),
        "pass_BCD_ratio": float(bcd_ratio),
        "pass_A_ratio": pass_a_ratio,
        "pass_B_ratio": pass_b_ratio,
        "pass_C_ratio": pass_c_ratio,
        "pass_D_ratio": pass_d_ratio,
        "pass_E_ratio": pass_e_ratio,
        "pass_BCD_required_ratio": float(thresholds["majority_crop_ratio_min"]),
        "pass_BCD_majority": bool(bcd_ratio >= float(thresholds["majority_crop_ratio_min"])),
        "failing_crops": failing,
    }


def _evaluate_crop(
    *,
    crop: str,
    price_file: str,
    config: dict,
    horizon_days: int,
    step_days: int,
    origin_start: Optional[pd.Timestamp],
    origin_end: Optional[pd.Timestamp],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    paths = config.get("paths", {})
    time_cfg = config.get("time", {})
    price_cfg_base = config.get("model", {}).get("price", {})
    base_mode = _normalize_prediction_mode(price_cfg_base)
    price_cfg, crop_override_keys, override_status = resolve_price_cfg_for_crop(
        price_cfg_base,
        crop,
        prediction_mode=base_mode,
        as_of_date=origin_end,
        include_status=True,
    )
    lags = list(time_cfg.get("price_lags", [1, 2, 3, 7]))
    windows = list(time_cfg.get("price_roll_windows", [7, 14, 30]))
    backtest_days = int(time_cfg.get("price_backtest_days", 180))
    strict_cutoff_split = bool(time_cfg.get("strict_cutoff_split", True))
    prediction_mode = _normalize_prediction_mode(price_cfg)
    trend_thresholds = _trend_thresholds(price_cfg_base)
    for notice in (override_status or {}).get("warnings", []):
        if not isinstance(notice, dict):
            continue
        logger.warning(
            "roll validation override warning crop=%s code=%s message=%s",
            crop,
            str(notice.get("code") or ""),
            str(notice.get("message") or ""),
        )

    raw_df = load_price_series(paths["price_dir"], price_file)
    if "date" in raw_df.columns:
        raw_df["date"] = _to_naive_day(raw_df["date"])
    price_window = resolve_price_window_from_df(raw_df, time_cfg=time_cfg)
    validation_cutoff = str(price_window.get("train_validation_cutoff_date") or "2020-12-31").strip()
    price_series = _daily_price_series(raw_df)
    if price_series.empty:
        return [], {"crop": crop, "price_file": price_file, "ok": False, "error": "empty price series"}

    cutoff_ts = _safe_timestamp(validation_cutoff) or pd.Timestamp("2020-12-31")
    max_origin = price_series.index.max() - pd.Timedelta(days=max(1, int(horizon_days)))
    if origin_start is None:
        eval_start = cutoff_ts + pd.Timedelta(days=1)
    else:
        eval_start = max(cutoff_ts + pd.Timedelta(days=1), origin_start)
    eval_end = max_origin if origin_end is None else min(max_origin, origin_end)
    if eval_start > eval_end:
        return [], {
            "crop": crop,
            "price_file": price_file,
            "ok": False,
            "error": "no valid rolling origins in selected date range",
            "origin_start": eval_start.strftime("%Y-%m-%d"),
            "origin_end": eval_end.strftime("%Y-%m-%d"),
        }

    version = str(config.get("serving", {}).get("model_cache_version", "v2"))
    model_dir = Path(config["output"]["out_dir"]) / "模型"

    train_hist_df = raw_df[_to_naive_day(raw_df["date"]) <= cutoff_ts].copy()
    if train_hist_df.empty:
        return [], {
            "crop": crop,
            "price_file": price_file,
            "ok": False,
            "error": "no training rows at or before cutoff",
        }

    step_model, step_meta, step_source = _load_step_model(
        model_dir=model_dir,
        crop=crop,
        price_cfg=price_cfg,
        version=version,
        history_df=train_hist_df,
        lags=lags,
        windows=windows,
        backtest_days=backtest_days,
        validation_cutoff=validation_cutoff,
        strict_cutoff_split=strict_cutoff_split,
    )
    if step_model is None:
        return [], {"crop": crop, "price_file": price_file, "ok": False, "error": "step model unavailable"}

    direct_model, direct_meta = _load_first_available_model(
        price_model_candidates(model_dir, crop, price_cfg, version)
    )

    artifacts = step_meta.get("artifacts", {}) if isinstance(step_meta, dict) else {}
    if not isinstance(artifacts, dict):
        artifacts = {}
    clip_art = artifacts.get("clip", {}) if isinstance(artifacts.get("clip"), dict) else {}
    bias_art = artifacts.get("bias_correction", {}) if isinstance(artifacts.get("bias_correction"), dict) else {}
    conformal_art = artifacts.get("conformal", {}) if isinstance(artifacts.get("conformal"), dict) else {}
    seasonal_art = artifacts.get("seasonal_anchor", {}) if isinstance(artifacts.get("seasonal_anchor"), dict) else {}

    include_raw_time_features = bool(
        (step_meta or {}).get(
            "include_raw_time_features",
            False if prediction_mode in {"return_recursive_v3", "return_recursive_v2"} else True,
        )
    )
    time_raw_mode = (step_meta or {}).get(
        "time_raw_mode",
        "none"
        if prediction_mode in {"return_recursive_v3", "return_recursive_v2"}
        else str(price_cfg.get("legacy_time_raw_mode", "raw")),
    )

    origins = pd.date_range(eval_start, eval_end, freq=f"{max(1, int(step_days))}D")
    rows: List[Dict[str, Any]] = []
    for origin in origins:
        if origin not in price_series.index:
            continue
        history_df = _history_df_from_series(price_series, origin)
        if history_df.empty:
            continue

        latest = _safe_float(pd.to_numeric(history_df["modal_price"], errors="coerce").iloc[-1])
        if latest is None or latest <= 0:
            continue
        target_end = origin + pd.Timedelta(days=int(horizon_days))
        future = price_series.loc[(origin + pd.Timedelta(days=1)) : target_end]
        if len(future) < int(horizon_days):
            continue
        actual = future.to_numpy(dtype=float)

        direct_end = _direct_predict(
            model=direct_model,
            meta=direct_meta,
            history_df=history_df,
            horizon_days=int(horizon_days),
            lags=lags,
            windows=windows,
            price_cfg=price_cfg,
        )
        use_terminal_anchor = bool(price_cfg.get("recursive_use_terminal_anchor", True))
        anchor_end = _effective_anchor(direct_end, history_df, price_cfg) if use_terminal_anchor else None
        diag: Dict[str, Any] = {}

        rec_rows = recursive_multi_step_forecast(
            model=step_model,
            history_df=history_df,
            horizon_days=int(horizon_days),
            lags=lags,
            windows=windows,
            feature_cols=_feature_cols_from_meta_or_model(step_meta, step_model) or None,
            max_daily_move_pct=_safe_float(price_cfg.get("recursive_max_daily_move_pct")),
            anchor_end_value=anchor_end,
            anchor_max_blend=float(price_cfg.get("recursive_anchor_max_blend", 0.65)),
            mean_reversion_strength=float(price_cfg.get("recursive_mean_reversion_strength", 0.18)),
            seasonal_strength=float(price_cfg.get("recursive_seasonal_strength", 0.22)),
            corridor_low_quantile=float(price_cfg.get("recursive_corridor_low_quantile", 0.05)),
            corridor_high_quantile=float(price_cfg.get("recursive_corridor_high_quantile", 0.95)),
            corridor_low_multiplier=float(price_cfg.get("recursive_corridor_low_multiplier", 0.75)),
            corridor_high_multiplier=float(price_cfg.get("recursive_corridor_high_multiplier", 1.35)),
            terminal_low_ratio_vs_latest=float(price_cfg.get("recursive_terminal_low_ratio_vs_latest", 0.6)),
            terminal_high_ratio_vs_latest=float(price_cfg.get("recursive_terminal_high_ratio_vs_latest", 1.8)),
            prediction_mode=prediction_mode,
            clip_r_max=_safe_float(price_cfg.get("return_clip_r_max")) or _safe_float(clip_art.get("r_max")),
            return_clip_quantile=float(price_cfg.get("return_clip_quantile", clip_art.get("quantile", 0.98))),
            return_clip_safety_factor=float(price_cfg.get("return_clip_safety_factor", clip_art.get("safety_factor", 1.2))),
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
            include_raw_time_features=include_raw_time_features,
            time_raw_mode=str(time_raw_mode) if time_raw_mode is not None else None,
            diagnostics=diag,
        )
        if len(rec_rows) < int(horizon_days):
            continue

        rec_rows = rec_rows[: int(horizon_days)]
        p50 = np.asarray([_safe_float(r.get("p50", r.get("value"))) for r in rec_rows], dtype=float)
        p10 = np.asarray([_safe_float(r.get("p10")) for r in rec_rows], dtype=float)
        p90 = np.asarray([_safe_float(r.get("p90")) for r in rec_rows], dtype=float)
        interval_q_step = np.asarray([_safe_float(r.get("interval_q_step")) for r in rec_rows], dtype=float)
        if not np.all(np.isfinite(p50)) or p50.size != actual.size:
            continue

        jump_1 = float(p50[0] / latest - 1.0)
        t0_jump_abs_ratio = float(abs(jump_1))
        early_slice = p50[: min(29, p50.size)]
        early_drop = float(np.min(early_slice) / max(latest, 1e-6) - 1.0) if early_slice.size else None
        ma30 = float(pd.to_numeric(history_df["modal_price"], errors="coerce").tail(30).mean())
        end_ratio = float(p50[-1] / max(ma30, 1e-6))
        hist_vals = pd.to_numeric(history_df["modal_price"], errors="coerce").dropna().to_numpy(dtype=float)
        hist_sign = _slope_sign(np.log(np.clip(hist_vals[-min(90, hist_vals.size) :], 1e-6, None)))
        pred_sign = _slope_sign(np.log(np.clip(p50[: min(60, p50.size)], 1e-6, None)))
        direction_mismatch_flag = (
            1
            if (hist_sign != 0 and pred_sign != 0 and int(hist_sign) != int(pred_sign))
            else 0
        )
        interval_ratio_price = _interval_non_decreasing_ratio(p10, p90)
        interval_ratio_qstep = _non_decreasing_ratio(interval_q_step)
        interval_roll_min = float(price_cfg.get("trend_e_interval_non_decreasing_roll_min", 0.95))
        interval_non_decreasing_flag = interval_ratio_qstep >= interval_roll_min
        end_shift_last = float(p50[-1] / latest - 1.0)
        end_bias = float((p50[-1] - actual[-1]) / max(actual[-1], 1e-6))
        gap_rd = None
        if direct_end is not None and p50[-1] > 0:
            gap_rd = float(abs(p50[-1] - direct_end) / max(abs(p50[-1]), 1e-6))

        cov = _coverage(actual, p10, p90)
        mape_all = _mape(actual, p50)
        smape_all = _smape(actual, p50)

        row: Dict[str, Any] = {
            "crop": crop,
            "price_file": price_file,
            "origin_date": origin.strftime("%Y-%m-%d"),
            "horizon_days": int(horizon_days),
            "step_model_source": step_source,
            "y_t0": float(latest),
            "y_t0_h_actual": float(actual[-1]),
            "y_t0_h_recursive": float(p50[-1]),
            "y_t0_h_direct": _safe_float(direct_end),
            "jump_1": float(jump_1),
            "t0_jump_abs_ratio": float(t0_jump_abs_ratio),
            "early_drop": _safe_float(early_drop),
            "ma30_t0": float(ma30),
            "end_ratio": float(end_ratio),
            "hist_slope_sign90": int(hist_sign),
            "pred_slope_sign60": int(pred_sign),
            "direction_mismatch_flag": int(direction_mismatch_flag),
            "interval_non_decreasing_flag": int(interval_non_decreasing_flag),
            "interval_non_decreasing_ratio_price": float(interval_ratio_price),
            "interval_non_decreasing_ratio_qstep": float(interval_ratio_qstep),
            "end_shift_last": float(end_shift_last),
            "end_bias": float(end_bias),
            "gap_rd": _safe_float(gap_rd),
            "clip_rate": _safe_float(diag.get("clip_rate")),
            "clip_amount_mean": _safe_float(diag.get("clip_amount_mean")),
            "blend_weight_mean": _safe_float(diag.get("blend_weight_mean")),
            "conformal_interval_source": str(diag.get("conformal_interval_source") or ""),
            "conformal_local_vol_ratio_mean": _safe_float(diag.get("conformal_local_vol_ratio_mean")),
            "smooth_adjustment_mean": _safe_float(diag.get("smooth_adjustment_mean")),
            "jerk_adjustment_mean": _safe_float(diag.get("jerk_adjustment_mean")),
            "crop_override_keys": "|".join(crop_override_keys or []),
            "crop_override_warning_codes": "|".join(
                [str(x.get("code") or "") for x in (override_status or {}).get("warnings", []) if isinstance(x, dict)]
            ),
            "crop_override_applied": int(bool((override_status or {}).get("applied", False))),
            "quality_flag": str(diag.get("quality_flag") or "OK"),
            "path_mape_all": _safe_float(mape_all),
            "path_smape_all": _safe_float(smape_all),
            "coverage_p10p90": _safe_float(cov),
            "coverage_n_points": int(actual.size),
        }
        row.update(_bucket_metrics(actual, p50, p10, p90))
        rows.append(row)

    if not rows:
        return [], {
            "crop": crop,
            "price_file": price_file,
            "ok": False,
            "error": "no valid rolling rows after evaluation",
            "step_model_source": step_source,
        }

    df_rows = pd.DataFrame(rows)
    trend_summary = _summarize_trend_rows(df_rows, trend_thresholds)
    return rows, {
        "crop": crop,
        "price_file": price_file,
        "ok": True,
        "n_rolls": int(len(df_rows)),
        "origin_start": str(df_rows["origin_date"].min()),
        "origin_end": str(df_rows["origin_date"].max()),
        "step_model_source": step_source,
        "override_status": override_status,
        "summary": _summarize_rows(df_rows),
        "trend": trend_summary,
    }


def _collect_crops(config: dict, *, crop_filter: str, max_crops: int) -> List[Tuple[str, str]]:
    name_map = resolve_names(load_name_map(config["paths"]["name_map"]))
    crops: List[Tuple[str, str]] = []
    for crop, info in sorted(name_map.items(), key=lambda x: x[0]):
        price_file = str((info or {}).get("price_file", "")).strip()
        if not price_file:
            continue
        crops.append((crop, price_file))
    if crop_filter:
        target = str(crop_filter).strip().lower()
        crops = [(c, pf) for c, pf in crops if str(c).lower() == target]
    if max_crops and int(max_crops) > 0:
        crops = crops[: int(max_crops)]
    return crops


def _apply_price_overrides(config: dict, overrides: Dict[str, Any]) -> dict:
    cfg = deepcopy(config)
    price_cfg = cfg.setdefault("model", {}).setdefault("price", {})
    for k, v in (overrides or {}).items():
        price_cfg[str(k)] = v
    return cfg


def _run_validation_once(
    *,
    config: dict,
    crops: List[Tuple[str, str]],
    horizon_days: int,
    step_days: int,
    origin_start: Optional[pd.Timestamp],
    origin_end: Optional[pd.Timestamp],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    all_rows: List[Dict[str, Any]] = []
    crop_summaries: List[Dict[str, Any]] = []
    for crop, price_file in crops:
        try:
            rows, summary = _evaluate_crop(
                crop=crop,
                price_file=price_file,
                config=config,
                horizon_days=horizon_days,
                step_days=step_days,
                origin_start=origin_start,
                origin_end=origin_end,
            )
        except Exception as exc:
            rows = []
            summary = {"crop": crop, "price_file": price_file, "ok": False, "error": str(exc)}
        all_rows.extend(rows)
        crop_summaries.append(summary)

    rows_df = pd.DataFrame(all_rows)
    overall_summary = _summarize_rows(rows_df) if not rows_df.empty else _summarize_rows(pd.DataFrame())
    trend_thresholds = _trend_thresholds(config.get("model", {}).get("price", {}))
    trend_overall = _summarize_trend_overall(crop_summaries, trend_thresholds)
    report = {
        "ok": bool(not rows_df.empty),
        "horizon_days": int(horizon_days),
        "step_days": int(step_days),
        "origin_start": origin_start.strftime("%Y-%m-%d") if isinstance(origin_start, pd.Timestamp) else None,
        "origin_end": origin_end.strftime("%Y-%m-%d") if isinstance(origin_end, pd.Timestamp) else None,
        "n_crops_requested": int(len(crops)),
        "n_crops_with_rows": int(sum(1 for x in crop_summaries if bool(x.get("ok")))),
        "n_rows": int(len(rows_df)),
        "overall": overall_summary,
        "trend_thresholds": trend_thresholds,
        "trend_overall": trend_overall,
        "per_crop": crop_summaries,
    }
    return report, rows_df


def _score_report(report: Dict[str, Any]) -> float:
    trend = report.get("trend_overall", {}) if isinstance(report, dict) else {}
    overall = report.get("overall", {}) if isinstance(report, dict) else {}
    bcd_ratio = _safe_float(trend.get("pass_BCD_ratio")) or 0.0
    a_ratio = _safe_float(trend.get("pass_A_ratio")) or 0.0
    e_ratio = _safe_float(trend.get("pass_E_ratio")) or 0.0
    coverage = _safe_float(overall.get("coverage_p10p90_global")) or 0.0
    penalty_cov = max(0.0, 0.75 - coverage)
    return float(100.0 * bcd_ratio + 10.0 * a_ratio + 5.0 * e_ratio - 80.0 * penalty_cov)


def _build_round_candidates(
    *,
    round_idx: int,
    best_overrides: Dict[str, Any],
    last_report: Optional[Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    if round_idx <= 1:
        return [
            ("baseline", {}),
            (
                "balanced_guard",
                {
                    "prediction_mode": "return_recursive_v3",
                    "seasonal_drift_max_alpha": 0.14,
                    "seasonal_drift_growth_power": 2.0,
                    "recursive_terminal_anchor_weight": 0.22,
                    "recursive_terminal_anchor_tail_days": 45,
                    "trend_guard_t0_jump_max": 0.03,
                    "trend_guard_t0_jump_steps": 1,
                    "trend_guard_end_ratio_low": 0.85,
                    "trend_guard_end_ratio_high": 1.15,
                    "trend_guard_end_tail_days": 35,
                    "trend_guard_end_max_weight": 0.92,
                    "trend_guard_end_pressure_scale": 6.5,
                    "trend_guard_end_backstop": True,
                    "trend_guard_early_drop_floor": -0.08,
                },
            ),
            (
                "strong_end_guard",
                {
                    "prediction_mode": "return_recursive_v3",
                    "seasonal_drift_max_alpha": 0.12,
                    "seasonal_drift_growth_power": 2.3,
                    "recursive_terminal_anchor_weight": 0.18,
                    "recursive_terminal_anchor_tail_days": 50,
                    "trend_guard_t0_jump_max": 0.028,
                    "trend_guard_t0_jump_steps": 1,
                    "trend_guard_end_ratio_low": 0.85,
                    "trend_guard_end_ratio_high": 1.12,
                    "trend_guard_end_tail_days": 42,
                    "trend_guard_end_max_weight": 0.95,
                    "trend_guard_end_pressure_scale": 7.5,
                    "trend_guard_end_backstop": True,
                    "trend_guard_early_drop_floor": -0.08,
                },
            ),
            (
                "anti_drop",
                {
                    "prediction_mode": "return_recursive_v3",
                    "seasonal_drift_max_alpha": 0.10,
                    "seasonal_drift_growth_power": 2.6,
                    "recursive_terminal_anchor_weight": 0.16,
                    "recursive_terminal_anchor_tail_days": 42,
                    "trend_guard_t0_jump_max": 0.028,
                    "trend_guard_t0_jump_steps": 1,
                    "trend_guard_end_ratio_low": 0.85,
                    "trend_guard_end_ratio_high": 1.15,
                    "trend_guard_end_tail_days": 35,
                    "trend_guard_end_max_weight": 0.90,
                    "trend_guard_end_pressure_scale": 6.0,
                    "trend_guard_end_backstop": True,
                    "trend_guard_early_drop_floor": -0.07,
                },
            ),
        ]

    base = dict(best_overrides or {})
    fail_items = (((last_report or {}).get("trend_overall") or {}).get("failing_crops") or [])
    fail_reasons = [r for item in fail_items for r in (item.get("fail_reasons") or [])]
    fail_a = int(sum(1 for x in fail_reasons if str(x) == "A_t0_jump"))
    fail_b = int(sum(1 for x in fail_reasons if str(x) == "B_early_drop"))
    fail_c = int(sum(1 for x in fail_reasons if str(x) == "C_end_ratio"))
    fail_d = int(sum(1 for x in fail_reasons if str(x) == "D_direction"))

    tuned = dict(base)
    tuned["prediction_mode"] = "return_recursive_v3"
    tuned["trend_guard_end_backstop"] = bool(base.get("trend_guard_end_backstop", True))
    if fail_a > 0:
        tuned["trend_guard_t0_jump_max"] = min(0.03, float(base.get("trend_guard_t0_jump_max", 0.03)))
        tuned["trend_guard_t0_jump_steps"] = max(1, int(base.get("trend_guard_t0_jump_steps", 1)))
    if fail_b > 0:
        tuned["seasonal_drift_max_alpha"] = max(0.05, float(base.get("seasonal_drift_max_alpha", 0.12)) - 0.02)
        tuned["seasonal_drift_growth_power"] = min(3.4, float(base.get("seasonal_drift_growth_power", 2.1)) + 0.4)
        tuned["trend_guard_early_drop_floor"] = max(-0.08, float(base.get("trend_guard_early_drop_floor", -0.08)))
    if fail_c > 0:
        tuned["trend_guard_end_ratio_high"] = min(1.15, float(base.get("trend_guard_end_ratio_high", 1.15)))
        tuned["trend_guard_end_tail_days"] = min(
            60,
            int(base.get("trend_guard_end_tail_days", 35)) + 10,
        )
        tuned["trend_guard_end_max_weight"] = min(
            0.97,
            float(base.get("trend_guard_end_max_weight", 0.90)) + 0.03,
        )
        tuned["trend_guard_end_pressure_scale"] = min(
            9.0,
            float(base.get("trend_guard_end_pressure_scale", 6.0)) + 1.0,
        )
        tuned["recursive_terminal_anchor_weight"] = max(0.12, float(base.get("recursive_terminal_anchor_weight", 0.20)) - 0.03)
        tuned["seasonal_drift_max_alpha"] = max(0.06, float(base.get("seasonal_drift_max_alpha", 0.12)) - 0.01)
    if fail_d > 0:
        tuned["endpoint_direction_guard"] = True
        tuned["trend_guard_future_window_days"] = min(
            120,
            int(base.get("trend_guard_future_window_days", 60)) + 14,
        )

    aggressive = dict(tuned)
    if fail_c > 0:
        aggressive["trend_guard_end_ratio_high"] = min(1.12, float(tuned.get("trend_guard_end_ratio_high", 1.15)))
        aggressive["trend_guard_end_max_weight"] = min(0.97, float(tuned.get("trend_guard_end_max_weight", 0.90)) + 0.03)
        aggressive["trend_guard_end_tail_days"] = min(70, int(tuned.get("trend_guard_end_tail_days", 35)) + 10)
    if fail_b > 0:
        aggressive["trend_guard_early_drop_floor"] = max(-0.07, float(tuned.get("trend_guard_early_drop_floor", -0.08)))
        aggressive["seasonal_drift_growth_power"] = min(3.6, float(tuned.get("seasonal_drift_growth_power", 2.3)) + 0.3)

    return [
        ("tuned_from_failures", tuned),
        ("aggressive_guard", aggressive),
    ]


def _run_cotton_attribution(
    *,
    config: dict,
    horizon_days: int,
    step_days: int,
    max_origins: int,
) -> Dict[str, Any]:
    crops = _collect_crops(config, crop_filter="cotton", max_crops=1)
    if not crops:
        return {"ok": False, "error": "cotton missing from crop map"}
    crop, price_file = crops[0]
    paths = config.get("paths", {})
    time_cfg = config.get("time", {})
    price_cfg_base = config.get("model", {}).get("price", {})
    base_mode = _normalize_prediction_mode(price_cfg_base)
    price_cfg, _, _ = resolve_price_cfg_for_crop(
        price_cfg_base,
        crop,
        prediction_mode=base_mode,
        include_status=True,
    )
    prediction_mode = _normalize_prediction_mode(price_cfg)
    lags = list(time_cfg.get("price_lags", [1, 2, 3, 7]))
    windows = list(time_cfg.get("price_roll_windows", [7, 14, 30]))
    backtest_days = int(time_cfg.get("price_backtest_days", 180))
    strict_cutoff_split = bool(time_cfg.get("strict_cutoff_split", True))

    raw_df = load_price_series(paths["price_dir"], price_file)
    if "date" in raw_df.columns:
        raw_df["date"] = _to_naive_day(raw_df["date"])
    price_window = resolve_price_window_from_df(raw_df, time_cfg=time_cfg)
    validation_cutoff = str(price_window.get("train_validation_cutoff_date") or "2020-12-31").strip()
    series = _daily_price_series(raw_df)
    if series.empty:
        return {"ok": False, "error": "cotton series empty"}

    cutoff_ts = _safe_timestamp(validation_cutoff) or pd.Timestamp("2020-12-31")
    max_origin = series.index.max() - pd.Timedelta(days=max(1, int(horizon_days)))
    origins = pd.date_range(cutoff_ts + pd.Timedelta(days=1), max_origin, freq=f"{max(1, int(step_days))}D")
    origins = origins[-max(1, int(max_origins)) :]
    if len(origins) == 0:
        return {"ok": False, "error": "cotton origins empty"}

    version = str(config.get("serving", {}).get("model_cache_version", "v2"))
    model_dir = Path(config["output"]["out_dir"]) / "模型"
    train_hist_df = raw_df[_to_naive_day(raw_df["date"]) <= cutoff_ts].copy()
    step_model, step_meta, step_source = _load_step_model(
        model_dir=model_dir,
        crop=crop,
        price_cfg=price_cfg,
        version=version,
        history_df=train_hist_df,
        lags=lags,
        windows=windows,
        backtest_days=backtest_days,
        validation_cutoff=validation_cutoff,
        strict_cutoff_split=strict_cutoff_split,
    )
    if step_model is None:
        return {"ok": False, "error": "cotton step model unavailable"}
    direct_model, direct_meta = _load_first_available_model(price_model_candidates(model_dir, crop, price_cfg, version))

    artifacts = step_meta.get("artifacts", {}) if isinstance(step_meta, dict) else {}
    if not isinstance(artifacts, dict):
        artifacts = {}
    clip_art = artifacts.get("clip", {}) if isinstance(artifacts.get("clip"), dict) else {}
    bias_art = artifacts.get("bias_correction", {}) if isinstance(artifacts.get("bias_correction"), dict) else {}
    conformal_art = artifacts.get("conformal", {}) if isinstance(artifacts.get("conformal"), dict) else {}
    seasonal_art = artifacts.get("seasonal_anchor", {}) if isinstance(artifacts.get("seasonal_anchor"), dict) else {}

    include_raw_time_features = bool(
        (step_meta or {}).get(
            "include_raw_time_features",
            False if prediction_mode in {"return_recursive_v3", "return_recursive_v2"} else True,
        )
    )
    time_raw_mode = (step_meta or {}).get(
        "time_raw_mode",
        "none" if prediction_mode in {"return_recursive_v3", "return_recursive_v2"} else str(price_cfg.get("legacy_time_raw_mode", "raw")),
    )

    base_kwargs = {
        "model": step_model,
        "horizon_days": int(horizon_days),
        "lags": lags,
        "windows": windows,
        "feature_cols": _feature_cols_from_meta_or_model(step_meta, step_model) or None,
        "max_daily_move_pct": _safe_float(price_cfg.get("recursive_max_daily_move_pct")),
        "prediction_mode": prediction_mode,
        "clip_r_max": _safe_float(price_cfg.get("return_clip_r_max")) or _safe_float(clip_art.get("r_max")),
        "return_clip_quantile": float(price_cfg.get("return_clip_quantile", clip_art.get("quantile", 0.98))),
        "return_clip_safety_factor": float(price_cfg.get("return_clip_safety_factor", clip_art.get("safety_factor", 1.2))),
        "return_bias_mean": _safe_float(bias_art.get("return_bias_mean")),
        "enable_bias_correction": bool(price_cfg.get("enable_bias_correction", True)),
        "conformal_abs_q": _safe_float(conformal_art.get("abs_q")),
        "enable_conformal_interval": bool(price_cfg.get("enable_conformal_interval", True)),
        "enable_seasonal_anchor": bool(price_cfg.get("enable_seasonal_anchor", True)),
        "seasonal_y_by_doy": seasonal_art.get("dayofyear_log_price_median"),
        "seasonal_tau_days": float(price_cfg.get("seasonal_anchor_tau_days", 55.0)),
        "seasonal_min_ml_weight": float(price_cfg.get("seasonal_anchor_min_ml_weight", 0.15)),
        "seasonal_drift_max_alpha": float(price_cfg.get("seasonal_drift_max_alpha", price_cfg.get("recursive_seasonal_strength", 0.22))),
        "seasonal_drift_growth_power": float(price_cfg.get("seasonal_drift_growth_power", 1.6)),
        "clip_rate_warn_threshold": float(price_cfg.get("clip_rate_warn_threshold", 0.10)),
        "endpoint_direction_guard": bool(price_cfg.get("endpoint_direction_guard", True)),
        "endpoint_opposite_slack": float(price_cfg.get("endpoint_opposite_slack", 1.5)),
        "endpoint_same_dir_cap": float(price_cfg.get("endpoint_same_dir_cap", 4.0)),
        "terminal_anchor_enable": bool(price_cfg.get("recursive_use_terminal_anchor", True)),
        "terminal_anchor_weight": float(price_cfg.get("recursive_terminal_anchor_weight", 0.35)),
        "terminal_anchor_tail_days": int(price_cfg.get("recursive_terminal_anchor_tail_days", 45)),
        "terminal_anchor_tail_power": float(price_cfg.get("recursive_terminal_anchor_tail_power", 1.6)),
        "trend_guard_early_drop_floor": float(price_cfg.get("trend_guard_early_drop_floor", -0.08)),
        "trend_guard_early_window_days": int(price_cfg.get("trend_guard_early_window_days", 28)),
        "trend_guard_end_ratio_low": float(price_cfg.get("trend_guard_end_ratio_low", 0.85)),
        "trend_guard_end_ratio_high": float(price_cfg.get("trend_guard_end_ratio_high", 1.15)),
        "trend_guard_hist_window_days": int(price_cfg.get("trend_guard_hist_window_days", 90)),
        "trend_guard_future_window_days": int(price_cfg.get("trend_guard_future_window_days", 60)),
        "trend_guard_t0_jump_max": float(price_cfg.get("trend_guard_t0_jump_max", 0.03)),
        "trend_guard_t0_jump_steps": int(price_cfg.get("trend_guard_t0_jump_steps", 1)),
        "trend_guard_end_tail_days": int(price_cfg.get("trend_guard_end_tail_days", 30)),
        "trend_guard_end_max_weight": float(price_cfg.get("trend_guard_end_max_weight", 0.90)),
        "trend_guard_end_pressure_scale": float(price_cfg.get("trend_guard_end_pressure_scale", 6.0)),
        "trend_guard_end_backstop": bool(price_cfg.get("trend_guard_end_backstop", True)),
        "interval_growth_power": float(price_cfg.get("conformal_interval_growth_power", 0.5)),
        "interval_growth_scale": float(price_cfg.get("conformal_interval_growth_scale", 1.0)),
        "interval_growth_cap": float(price_cfg.get("conformal_interval_growth_cap", 200.0)),
        "conformal_horizon_scale_points": conformal_art.get("horizon_scale_points"),
        "conformal_use_horizon_scale": bool(price_cfg.get("conformal_use_horizon_scale", True)),
        "conformal_local_vol_window": int(price_cfg.get("conformal_local_vol_window_days", 45)),
        "conformal_local_vol_ratio_low": float(price_cfg.get("conformal_local_vol_ratio_low", 0.70)),
        "conformal_local_vol_ratio_high": float(price_cfg.get("conformal_local_vol_ratio_high", 1.40)),
        "conformal_local_vol_reference": _safe_float(conformal_art.get("local_vol_reference")),
        "return_smooth_alpha": float(price_cfg.get("recursive_return_smooth_alpha", 0.15)),
        "return_jerk_clip_sigma": float(price_cfg.get("recursive_return_jerk_clip_sigma", 2.5)),
        "return_smooth_warmup_steps": int(price_cfg.get("recursive_return_smooth_warmup_steps", 5)),
        "include_raw_time_features": include_raw_time_features,
        "time_raw_mode": str(time_raw_mode) if time_raw_mode is not None else None,
    }
    if bool(price_cfg.get("attribution_disable_trend_guards", True)):
        base_kwargs.update(
            {
                "trend_guard_early_drop_floor": -0.5,
                "trend_guard_early_window_days": 5,
                "trend_guard_end_ratio_low": 0.4,
                "trend_guard_end_ratio_high": 2.5,
                "trend_guard_t0_jump_max": 0.25,
                "trend_guard_t0_jump_steps": 1,
                "trend_guard_end_tail_days": 5,
                "trend_guard_end_max_weight": 0.2,
                "trend_guard_end_pressure_scale": 0.1,
                "trend_guard_end_backstop": False,
            }
        )

    variants = {
        "01_ml_only": {
            "enable_seasonal_anchor": False,
            "terminal_anchor_enable": False,
            "endpoint_direction_guard": False,
            "enable_conformal_interval": False,
            "return_smooth_alpha": 0.0,
            "return_jerk_clip_sigma": 0.0,
            "clip_r_max": 0.35,
        },
        "02_ml_plus_seasonal": {
            "enable_seasonal_anchor": True,
            "terminal_anchor_enable": False,
            "endpoint_direction_guard": False,
            "enable_conformal_interval": False,
            "return_smooth_alpha": 0.0,
            "return_jerk_clip_sigma": 0.0,
            "clip_r_max": 0.35,
        },
        "03_ml_plus_terminal": {
            "enable_seasonal_anchor": False,
            "terminal_anchor_enable": True,
            "endpoint_direction_guard": False,
            "enable_conformal_interval": False,
            "return_smooth_alpha": 0.0,
            "return_jerk_clip_sigma": 0.0,
            "clip_r_max": 0.35,
        },
        "04_seasonal_plus_terminal": {
            "enable_seasonal_anchor": True,
            "terminal_anchor_enable": True,
            "endpoint_direction_guard": False,
            "enable_conformal_interval": False,
            "return_smooth_alpha": 0.0,
            "return_jerk_clip_sigma": 0.0,
            "clip_r_max": 0.35,
        },
        "05_add_direction_guard": {
            "enable_seasonal_anchor": True,
            "terminal_anchor_enable": True,
            "endpoint_direction_guard": True,
            "enable_conformal_interval": False,
            "return_smooth_alpha": 0.0,
            "return_jerk_clip_sigma": 0.0,
            "clip_r_max": 0.35,
        },
        "06_add_smoothing": {
            "enable_seasonal_anchor": True,
            "terminal_anchor_enable": True,
            "endpoint_direction_guard": True,
            "enable_conformal_interval": False,
        },
        "07_add_clip": {
            "enable_seasonal_anchor": True,
            "terminal_anchor_enable": True,
            "endpoint_direction_guard": True,
            "enable_conformal_interval": False,
        },
        "08_add_interval_full": {
            "enable_seasonal_anchor": True,
            "terminal_anchor_enable": True,
            "endpoint_direction_guard": True,
            "enable_conformal_interval": True,
        },
    }

    records: List[Dict[str, Any]] = []
    for origin in origins:
        if origin not in series.index:
            continue
        history_df = _history_df_from_series(series, origin)
        latest = _safe_float(pd.to_numeric(history_df["modal_price"], errors="coerce").iloc[-1])
        if latest is None or latest <= 0:
            continue
        ma30 = float(pd.to_numeric(history_df["modal_price"], errors="coerce").tail(30).mean())
        direct_end = _direct_predict(
            model=direct_model,
            meta=direct_meta,
            history_df=history_df,
            horizon_days=int(horizon_days),
            lags=lags,
            windows=windows,
            price_cfg=price_cfg,
        )
        anchor_end = _effective_anchor(direct_end, history_df, price_cfg)
        for variant_name, override in variants.items():
            diag: Dict[str, Any] = {}
            kwargs = dict(base_kwargs)
            kwargs.update(override)
            kwargs["history_df"] = history_df
            kwargs["anchor_end_value"] = anchor_end if kwargs.get("terminal_anchor_enable", False) else None
            kwargs["diagnostics"] = diag
            out = recursive_multi_step_forecast(**kwargs)
            if len(out) < int(horizon_days):
                continue
            p50 = np.asarray([_safe_float(x.get("p50", x.get("value"))) for x in out[: int(horizon_days)]], dtype=float)
            if p50.size != int(horizon_days) or not np.all(np.isfinite(p50)):
                continue
            idx7 = min(6, p50.size - 1)
            idx30 = min(29, p50.size - 1)
            idx90 = min(89, p50.size - 1)
            dif = np.diff(np.log(np.clip(p50, 1e-6, None)))
            smoothness = float(np.mean(np.abs(np.diff(dif)))) if dif.size >= 2 else 0.0
            records.append(
                {
                    "variant": variant_name,
                    "origin_date": origin.strftime("%Y-%m-%d"),
                    "t0_jump_abs_ratio": float(abs(p50[0] - latest) / max(latest, 1e-6)),
                    "early_drop": float(np.min(p50[: min(29, p50.size)]) / max(latest, 1e-6) - 1.0),
                    "end_ratio": float(p50[-1] / max(ma30, 1e-6)),
                    "path_smoothness": smoothness,
                    "yhat_t0": float(p50[0]),
                    "yhat_t0p7": float(p50[idx7]),
                    "yhat_t0p30": float(p50[idx30]),
                    "yhat_t0p90": float(p50[idx90]),
                    "yhat_t0pH": float(p50[-1]),
                    "blend_weight_mean": _safe_float(diag.get("blend_weight_mean")),
                    "clip_rate": _safe_float(diag.get("clip_rate")),
                    "direction_guard_rate": _safe_float(diag.get("direction_guard_rate", diag.get("trend_sign_guard_rate"))),
                }
            )

    if not records:
        return {"ok": False, "error": "cotton attribution empty"}

    rows_df = pd.DataFrame(records)
    summary_df = (
        rows_df.groupby("variant", as_index=False)
        .agg(
            n_rolls=("origin_date", "count"),
            t0_jump_median=("t0_jump_abs_ratio", "median"),
            early_drop_median=("early_drop", "median"),
            end_ratio_median=("end_ratio", "median"),
            path_smoothness_median=("path_smoothness", "median"),
            yhat_t0_median=("yhat_t0", "median"),
            yhat_t0p7_median=("yhat_t0p7", "median"),
            yhat_t0p30_median=("yhat_t0p30", "median"),
            yhat_t0p90_median=("yhat_t0p90", "median"),
            yhat_t0pH_median=("yhat_t0pH", "median"),
            blend_weight_mean=("blend_weight_mean", "mean"),
            clip_rate_mean=("clip_rate", "mean"),
            direction_guard_rate_mean=("direction_guard_rate", "mean"),
        )
        .sort_values("variant")
    )
    return {
        "ok": True,
        "crop": crop,
        "price_file": price_file,
        "step_model_source": step_source,
        "n_origins": int(len(origins)),
        "summary": summary_df.to_dict(orient="records"),
        "rows": rows_df.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling validation for recursive price forecast")
    parser.add_argument("--config", default="后端/配置.yaml")
    parser.add_argument("--crop", default="", help="optional env_label crop")
    parser.add_argument("--all-crops", action="store_true", help="explicitly evaluate all crops")
    parser.add_argument("--max-crops", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=0, help="rolling horizon days, default from config")
    parser.add_argument("--step-days", type=int, default=7, help="rolling origin step in days")
    parser.add_argument("--origin-start", default="", help="optional rolling origin start date (YYYY-MM-DD)")
    parser.add_argument("--origin-end", default="", help="optional rolling origin end date (YYYY-MM-DD)")
    parser.add_argument("--out-json", default="", help="legacy alias of --report")
    parser.add_argument("--out-csv", default="", help="legacy alias of --rows-csv")
    parser.add_argument("--report", default="", help="output json report path")
    parser.add_argument("--rows-csv", default="", help="output rows csv path")
    parser.add_argument("--compare-config", action="append", default=[], help="extra config path for side-by-side run")
    parser.add_argument("--auto-iterate", action="store_true", help="auto search price config for trend consistency")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--search-step-days", type=int, default=28)
    parser.add_argument("--cotton-attribution-origins", type=int, default=1)
    args = parser.parse_args()

    base_config = load_config(args.config)
    out_dir = Path(base_config["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    horizon_days = int(args.horizon) if int(args.horizon) > 0 else int(base_config.get("time", {}).get("price_forecast_horizon_days", 181))
    step_days = max(1, int(args.step_days))
    search_step_days = max(1, int(args.search_step_days))
    origin_start = _safe_timestamp(args.origin_start)
    origin_end = _safe_timestamp(args.origin_end)

    report_arg = args.report or args.out_json
    rows_arg = args.rows_csv or args.out_csv

    if report_arg:
        out_json = Path(report_arg)
        if not out_json.is_absolute():
            out_json = Path(ROOT) / out_json
    else:
        out_json = out_dir / "价格递归滚动验证报告.json"

    if rows_arg:
        out_csv = Path(rows_arg)
        if not out_csv.is_absolute():
            out_csv = Path(ROOT) / out_csv
    else:
        out_csv = out_dir / "价格递归滚动验证明细.csv"

    crops = _collect_crops(base_config, crop_filter=args.crop, max_crops=args.max_crops)
    if args.all_crops and not args.crop:
        crops = _collect_crops(base_config, crop_filter="", max_crops=args.max_crops)

    iterations: List[Dict[str, Any]] = []
    best_overrides: Dict[str, Any] = {}
    final_config = deepcopy(base_config)
    final_report: Optional[Dict[str, Any]] = None
    rows_df: pd.DataFrame = pd.DataFrame()

    if args.auto_iterate:
        last_report: Optional[Dict[str, Any]] = None
        total_rounds = max(2, int(args.max_rounds))
        for round_idx in range(1, total_rounds + 1):
            this_step = step_days if round_idx >= total_rounds else max(step_days, search_step_days)
            candidates = _build_round_candidates(
                round_idx=round_idx,
                best_overrides=best_overrides,
                last_report=last_report,
            )
            round_results: List[Dict[str, Any]] = []
            for candidate_name, candidate_overrides in candidates:
                merged = dict(best_overrides)
                merged.update(candidate_overrides or {})
                cfg_try = _apply_price_overrides(base_config, merged)
                report_try, rows_try = _run_validation_once(
                    config=cfg_try,
                    crops=crops,
                    horizon_days=horizon_days,
                    step_days=this_step,
                    origin_start=origin_start,
                    origin_end=origin_end,
                )
                round_results.append(
                    {
                        "candidate": candidate_name,
                        "overrides": merged,
                        "score": _score_report(report_try),
                        "step_days": int(this_step),
                        "trend_overall": report_try.get("trend_overall", {}),
                        "coverage_p10p90_global": (report_try.get("overall") or {}).get("coverage_p10p90_global"),
                        "n_rows": int(len(rows_try)),
                        "_report": report_try,
                        "_rows_df": rows_try,
                    }
                )
            round_results = sorted(round_results, key=lambda x: float(x.get("score", 0.0)), reverse=True)
            selected = round_results[0]
            best_overrides = dict(selected.get("overrides") or {})
            final_config = _apply_price_overrides(base_config, best_overrides)
            last_report = selected.get("_report")
            rows_df = selected.get("_rows_df") if isinstance(selected.get("_rows_df"), pd.DataFrame) else pd.DataFrame()
            final_report = dict(last_report or {})
            iterations.append(
                {
                    "round": int(round_idx),
                    "step_days": int(this_step),
                    "candidates": [
                        {
                            k: v
                            for k, v in item.items()
                            if k not in {"_report", "_rows_df"}
                        }
                        for item in round_results
                    ],
                    "selected_candidate": selected.get("candidate"),
                    "selected_overrides": best_overrides,
                    "selected_trend_overall": (last_report or {}).get("trend_overall", {}),
                    "selected_score": _score_report(last_report or {}),
                }
            )
            if (
                round_idx >= 2
                and this_step == step_days
                and bool(((last_report or {}).get("trend_overall") or {}).get("pass_BCD_majority"))
            ):
                break

    if final_report is None or int(final_report.get("step_days", step_days)) != int(step_days):
        final_report, rows_df = _run_validation_once(
            config=final_config,
            crops=crops,
            horizon_days=horizon_days,
            step_days=step_days,
            origin_start=origin_start,
            origin_end=origin_end,
        )

    compare_results: List[Dict[str, Any]] = []
    for cfg_path in args.compare_config:
        if not str(cfg_path).strip():
            continue
        cfg_other = load_config(cfg_path)
        rep_other, _ = _run_validation_once(
            config=cfg_other,
            crops=_collect_crops(cfg_other, crop_filter=args.crop, max_crops=args.max_crops),
            horizon_days=horizon_days,
            step_days=step_days,
            origin_start=origin_start,
            origin_end=origin_end,
        )
        compare_results.append(
            {
                "config_path": str(Path(cfg_path).resolve()),
                "trend_overall": rep_other.get("trend_overall", {}),
                "overall": rep_other.get("overall", {}),
            }
        )

    cotton_attr = _run_cotton_attribution(
        config=final_config,
        horizon_days=horizon_days,
        step_days=step_days,
        max_origins=max(1, int(args.cotton_attribution_origins)),
    )

    rows_df.to_csv(out_csv, index=False)
    output_report = dict(final_report)
    output_report.update(
        {
            "config_path": str(Path(args.config).resolve()),
            "final_price_overrides": best_overrides,
            "iterations": iterations,
            "compare_results": compare_results,
            "cotton_attribution": cotton_attr,
            "rows_csv": out_csv.as_posix(),
        }
    )
    out_json.write_text(json.dumps(output_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(out_json.as_posix())
    print(out_csv.as_posix())


if __name__ == "__main__":
    main()
