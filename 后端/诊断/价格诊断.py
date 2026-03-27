from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from 后端.兼容层 import tune_loaded_model
from 后端.数据加载 import load_config, load_name_map, load_price_series, resolve_names
from 后端.模型产物 import price_recursive_model_candidates
from 后端.价格作物覆盖 import resolve_price_cfg_for_crop
from 后端.模型.价格模型 import train_one_crop as train_price_one_crop
from 后端.价格递归预测 import recursive_multi_step_forecast
from 后端.时间策略 import resolve_price_window_from_df
from 后端.特征工程 import make_supervised


logger = logging.getLogger(__name__)


def _to_naive_day(values: Any) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(parsed, pd.Series):
        return parsed.dt.tz_convert(None).dt.normalize()
    return pd.Series(dtype="datetime64[ns]")

def _safe_float(v: Any) -> Optional[float]:
    try:
        out = float(v)
    except Exception:
        return None
    if pd.isna(out):
        return None
    return out


def _normalize_prediction_mode(price_cfg: dict) -> str:
    mode = str((price_cfg or {}).get("prediction_mode", "return_recursive_v3")).strip().lower()
    if mode not in {"return_recursive_v3", "return_recursive_v2", "price_recursive_v1", "direct_horizon_v1"}:
        return "return_recursive_v3"
    return mode


def _build_step_cfg(price_cfg: dict) -> dict:
    cfg = dict(price_cfg or {})
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


def _resolve_meta_path(meta_path: Path) -> Path:
    candidates = [
        meta_path,
        meta_path.with_name(meta_path.name.replace("_指标.json", "_metrics.json")) if meta_path.name.endswith("_指标.json") else meta_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return meta_path


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
) -> Tuple[object, dict]:
    requested_mode = _normalize_prediction_mode(price_cfg)
    expected_target = "log_return" if requested_mode in {"return_recursive_v3", "return_recursive_v2"} else "price"

    for model_path, meta_path in price_recursive_model_candidates(model_dir, crop, price_cfg, version):
        if not model_path.exists():
            continue
        try:
            model = tune_loaded_model(load(model_path))
        except Exception:
            continue
        meta: dict = {}
        meta_path = _resolve_meta_path(meta_path)
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
                except Exception:
                    meta = {}

        target_mode = _meta_target_mode(meta)
        if requested_mode in {"return_recursive_v3", "return_recursive_v2"} and target_mode != "log_return":
            continue
        if requested_mode == "price_recursive_v1" and target_mode == "log_return":
            continue
        return model, meta

    # Inline fallback.
    step_cfg = _build_step_cfg(price_cfg)
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
        label=f"diag:{crop}",
    )
    meta = {
        "feature_cols": list(res.feature_cols or []),
        "target_mode": step_cfg.get("target_mode", expected_target),
        "feature_space": step_cfg.get("feature_space", "price"),
        "time_raw_mode": step_cfg.get("time_raw_mode", "none"),
        "include_raw_time_features": bool(step_cfg.get("include_raw_time_features", False)),
        "artifacts": res.artifacts or {},
    }
    return res.model, meta


def _metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    if y_true.size == 0 or y_pred.size == 0:
        return {"mae": None, "rmse": None, "mape": None}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs(err) / denom))
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _max_down_streak(values: np.ndarray) -> int:
    if values.size <= 1:
        return 0
    dif = np.diff(values)
    best = 0
    cur = 0
    for d in dif:
        if d < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _teacher_forcing_rows(
    *,
    df: pd.DataFrame,
    model,
    meta: dict,
    cutoff_ts: pd.Timestamp,
    lags: List[int],
    windows: List[int],
    price_cfg: dict,
) -> pd.DataFrame:
    target_mode = str(meta.get("target_mode", "price")).strip().lower() or "price"
    if target_mode in {"return", "ret", "delta"}:
        target_mode = "log_return"
    feature_space = str(meta.get("feature_space", "price")).strip().lower() or "price"
    include_raw = bool(meta.get("include_raw_time_features", False if target_mode == "log_return" else True))
    time_raw_mode = str(meta.get("time_raw_mode", "none" if target_mode == "log_return" else "raw"))

    X_all, y_all, feat_dates, target_dates = make_supervised(
        df,
        "modal_price",
        horizon=1,
        lags=lags,
        windows=windows,
        return_dates=True,
        return_target_dates=True,
        target_mode=target_mode,
        feature_space=feature_space,
        include_raw_time_features=include_raw,
        time_raw_mode=time_raw_mode,
        price_floor=float(price_cfg.get("price_floor", 1e-6)),
    )
    if X_all.empty:
        return pd.DataFrame()

    feat_cols = meta.get("feature_cols", []) if isinstance(meta, dict) else []
    X_model = X_all.reindex(columns=feat_cols, fill_value=0.0) if feat_cols else X_all
    pred_raw = np.asarray(model.predict(X_model), dtype=float)

    artifacts = meta.get("artifacts", {}) if isinstance(meta, dict) else {}
    bias_art = artifacts.get("bias_correction", {}) if isinstance(artifacts.get("bias_correction"), dict) else {}
    clip_art = artifacts.get("clip", {}) if isinstance(artifacts.get("clip"), dict) else {}
    b = _safe_float(bias_art.get("return_bias_mean")) or 0.0
    r_max = _safe_float(price_cfg.get("return_clip_r_max"))
    if r_max is None:
        r_max = _safe_float(clip_art.get("r_max"))

    work = df[["date", "modal_price"]].copy()
    work["date"] = _to_naive_day(work["date"])
    work["modal_price"] = pd.to_numeric(work["modal_price"], errors="coerce")
    work = work.dropna(subset=["date", "modal_price"]).sort_values("date")
    p_map = {pd.Timestamp(d): float(v) for d, v in zip(work["date"], work["modal_price"])}

    rows: List[Dict[str, Any]] = []
    for idx in range(len(X_all)):
        f_date = pd.Timestamp(feat_dates.iloc[idx]).normalize()
        t_date = pd.Timestamp(target_dates.iloc[idx]).normalize()
        if t_date <= cutoff_ts:
            continue
        actual_t = _safe_float(p_map.get(t_date))
        actual_prev = _safe_float(p_map.get(f_date))
        if actual_t is None or actual_prev is None or actual_prev <= 0:
            continue

        clip_applied = False
        clip_amount = 0.0
        pred_price = None
        pred_ret = None
        true_ret = float(np.log(max(actual_t, 1e-6) / max(actual_prev, 1e-6)))

        if target_mode == "log_return":
            r_pred = float(pred_raw[idx] - b)
            if r_max is not None:
                clipped = float(np.clip(r_pred, -abs(r_max), abs(r_max)))
                clip_amount = float(abs(clipped - r_pred))
                clip_applied = clip_amount > 0.0
                r_pred = clipped
            pred_ret = float(r_pred)
            pred_price = float(np.exp(np.log(max(actual_prev, 1e-6)) + pred_ret))
        else:
            pred_price = float(max(0.0, pred_raw[idx]))
            pred_ret = float(np.log(max(pred_price, 1e-6) / max(actual_prev, 1e-6)))

        rows.append(
            {
                "date": t_date.strftime("%Y-%m-%d"),
                "actual": float(actual_t),
                "teacher_forcing": float(pred_price),
                "teacher_forcing_return": float(pred_ret),
                "true_return": float(true_ret),
                "clip_applied": bool(clip_applied),
                "clip_amount": float(clip_amount),
            }
        )

    return pd.DataFrame(rows)


def _diagnose_crop(
    *,
    crop: str,
    price_file: str,
    config: dict,
    out_dir: Path,
    horizon_cap: Optional[int],
) -> Dict[str, Any]:
    paths = config["paths"]
    time_cfg = config.get("time", {})
    price_cfg_base = config.get("model", {}).get("price", {})
    base_mode = _normalize_prediction_mode(price_cfg_base)
    price_cfg, crop_override_keys, override_status = resolve_price_cfg_for_crop(
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
    for notice in (override_status or {}).get("warnings", []):
        if not isinstance(notice, dict):
            continue
        logger.warning(
            "price diagnostics override warning crop=%s code=%s message=%s",
            crop,
            str(notice.get("code") or ""),
            str(notice.get("message") or ""),
        )

    df = load_price_series(paths["price_dir"], price_file)
    df["date"] = _to_naive_day(df["date"])
    df["modal_price"] = pd.to_numeric(df["modal_price"], errors="coerce")
    df = df.dropna(subset=["date", "modal_price"]).sort_values("date").reset_index(drop=True)
    price_window = resolve_price_window_from_df(df, time_cfg=time_cfg)
    validation_cutoff = str(price_window.get("train_validation_cutoff_date") or "2020-12-31").strip()
    cutoff_ts = pd.to_datetime(validation_cutoff, errors="coerce")
    if pd.isna(cutoff_ts):
        cutoff_ts = pd.Timestamp("2020-12-31")
    cutoff_ts = cutoff_ts.normalize()

    val_df = df[df["date"] > cutoff_ts].copy()
    if val_df.empty:
        return {
            "crop": crop,
            "price_file": price_file,
            "prediction_mode": prediction_mode,
            "crop_override_keys": crop_override_keys,
            "crop_override_status": override_status,
            "ok": False,
            "error": "no post-cutoff validation rows",
        }

    if horizon_cap is not None and horizon_cap > 0:
        val_df = val_df.head(int(horizon_cap)).copy()
    if val_df.empty:
        return {
            "crop": crop,
            "price_file": price_file,
            "prediction_mode": prediction_mode,
            "crop_override_keys": crop_override_keys,
            "crop_override_status": override_status,
            "ok": False,
            "error": "validation rows empty after horizon cap",
        }

    train_hist = df[df["date"] <= cutoff_ts].copy()
    if train_hist.empty:
        return {
            "crop": crop,
            "price_file": price_file,
            "prediction_mode": prediction_mode,
            "crop_override_keys": crop_override_keys,
            "crop_override_status": override_status,
            "ok": False,
            "error": "no training rows at/before cutoff",
        }

    version = str(config.get("serving", {}).get("model_cache_version", "v2"))
    model_dir = Path(config["output"]["out_dir"]) / "模型"
    model, meta = _load_step_model(
        model_dir=model_dir,
        crop=crop,
        price_cfg=price_cfg,
        version=version,
        history_df=train_hist,
        lags=lags,
        windows=windows,
        backtest_days=backtest_days,
        validation_cutoff=validation_cutoff,
        strict_cutoff_split=strict_cutoff_split,
    )

    teacher_df = _teacher_forcing_rows(
        df=df,
        model=model,
        meta=meta,
        cutoff_ts=cutoff_ts,
        lags=lags,
        windows=windows,
        price_cfg=price_cfg,
    )
    if teacher_df.empty:
        return {
            "crop": crop,
            "price_file": price_file,
            "prediction_mode": prediction_mode,
            "crop_override_keys": crop_override_keys,
            "crop_override_status": override_status,
            "ok": False,
            "error": "teacher forcing empty",
        }

    actual_dates = [
        pd.Timestamp(x).normalize()
        for x in _to_naive_day(teacher_df["date"]).dropna().tolist()
    ]
    horizon_days = len(actual_dates)
    diag_rec: Dict[str, Any] = {}

    art = meta.get("artifacts", {}) if isinstance(meta, dict) else {}
    bias_art = art.get("bias_correction", {}) if isinstance(art.get("bias_correction"), dict) else {}
    conformal_art = art.get("conformal", {}) if isinstance(art.get("conformal"), dict) else {}
    seasonal_art = art.get("seasonal_anchor", {}) if isinstance(art.get("seasonal_anchor"), dict) else {}
    clip_art = art.get("clip", {}) if isinstance(art.get("clip"), dict) else {}

    rec_rows = recursive_multi_step_forecast(
        model=model,
        history_df=train_hist,
        horizon_days=horizon_days,
        lags=lags,
        windows=windows,
        feature_cols=meta.get("feature_cols") if isinstance(meta, dict) else None,
        max_daily_move_pct=_safe_float(price_cfg.get("recursive_max_daily_move_pct")),
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
        include_raw_time_features=bool(meta.get("include_raw_time_features", False)),
        time_raw_mode=meta.get("time_raw_mode"),
        diagnostics=diag_rec,
    )

    rec_map = {
        pd.Timestamp(str(r.get("date"))).normalize(): r
        for r in rec_rows
        if isinstance(r, dict) and str(r.get("date", "")).strip()
    }

    train_last_price = float(train_hist["modal_price"].iloc[-1])
    season_map_log = seasonal_art.get("dayofyear_log_price_median", {}) if isinstance(seasonal_art, dict) else {}

    curve_rows: List[Dict[str, Any]] = []
    for _, row in teacher_df.iterrows():
        ts = pd.Timestamp(row["date"]).normalize()
        actual = _safe_float(row.get("actual"))
        tf = _safe_float(row.get("teacher_forcing"))
        rec = rec_map.get(ts, {})
        rec_p50 = _safe_float(rec.get("p50", rec.get("value")))
        rec_p10 = _safe_float(rec.get("p10"))
        rec_p90 = _safe_float(rec.get("p90"))
        naive = train_last_price

        seas = None
        doy = int(ts.dayofyear)
        if isinstance(season_map_log, dict) and str(doy) in season_map_log:
            seas = float(np.exp(float(season_map_log[str(doy)])))

        curve_rows.append(
            {
                "date": ts.strftime("%Y-%m-%d"),
                "actual": actual,
                "teacher_forcing": tf,
                "recursive_p50": rec_p50,
                "recursive_p10": rec_p10,
                "recursive_p90": rec_p90,
                "naive": float(naive),
                "seasonal_naive": seas,
                "teacher_forcing_return": _safe_float(row.get("teacher_forcing_return")),
                "true_return": _safe_float(row.get("true_return")),
            }
        )

    curve_df = pd.DataFrame(curve_rows)
    curve_df = curve_df.dropna(subset=["actual"]).copy()

    def _arr(col: str) -> np.ndarray:
        return pd.to_numeric(curve_df.get(col), errors="coerce").to_numpy(dtype=float)

    actual = _arr("actual")
    tf_pred = _arr("teacher_forcing")
    rec_pred = _arr("recursive_p50")
    naive_pred = _arr("naive")
    seas_pred = _arr("seasonal_naive")

    def _finite_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.isfinite(a) & np.isfinite(b)
        return a[mask], b[mask]

    a_tf, p_tf = _finite_pair(actual, tf_pred)
    a_rec, p_rec = _finite_pair(actual, rec_pred)
    a_na, p_na = _finite_pair(actual, naive_pred)
    a_sn, p_sn = _finite_pair(actual, seas_pred)

    log_actual = np.log(np.clip(a_rec, 1e-6, None))
    log_rec = np.log(np.clip(p_rec, 1e-6, None))

    tf_ret = pd.to_numeric(curve_df.get("teacher_forcing_return"), errors="coerce").to_numpy(dtype=float)
    true_ret = pd.to_numeric(curve_df.get("true_return"), errors="coerce").to_numpy(dtype=float)
    ret_mask = np.isfinite(tf_ret) & np.isfinite(true_ret)
    ret_bias = tf_ret[ret_mask] - true_ret[ret_mask]

    down_ratio = None
    max_down_streak_ratio = None
    if p_rec.size > 1:
        dif = np.diff(p_rec)
        down_ratio = float(np.mean(dif < 0))
        max_down = _max_down_streak(p_rec)
        max_down_streak_ratio = float(max_down / max(1, len(dif)))

    out_curve_path = out_dir / f"价格诊断曲线_{crop}.csv"
    curve_df.to_csv(out_curve_path, index=False)

    summary = {
        "crop": crop,
        "price_file": price_file,
        "prediction_mode": prediction_mode,
        "crop_override_keys": list(crop_override_keys),
        "crop_override_status": override_status,
        "ok": True,
        "n_points": int(len(curve_df)),
        "curve_path": out_curve_path.as_posix(),
        "teacher_forcing_metrics": _metric_pack(a_tf, p_tf),
        "recursive_metrics": _metric_pack(a_rec, p_rec),
        "naive_metrics": _metric_pack(a_na, p_na),
        "seasonal_naive_metrics": _metric_pack(a_sn, p_sn),
        "recursive_log_metrics": _metric_pack(log_actual, log_rec) if log_actual.size and log_rec.size else {"mae": None, "rmse": None, "mape": None},
        "return_bias_mean": float(np.mean(ret_bias)) if ret_bias.size else None,
        "return_bias_std": float(np.std(ret_bias)) if ret_bias.size else None,
        "clip_rate": _safe_float(diag_rec.get("clip_rate")),
        "clip_amount_mean": _safe_float(diag_rec.get("clip_amount_mean")),
        "quality_flag": str(diag_rec.get("quality_flag") or "OK"),
        "blend_weight_mean": _safe_float(diag_rec.get("blend_weight_mean")),
        "conformal_interval_source": str(diag_rec.get("conformal_interval_source") or ""),
        "conformal_local_vol_ratio_mean": _safe_float(diag_rec.get("conformal_local_vol_ratio_mean")),
        "smooth_adjustment_mean": _safe_float(diag_rec.get("smooth_adjustment_mean")),
        "jerk_adjustment_mean": _safe_float(diag_rec.get("jerk_adjustment_mean")),
        "monotonic_down_ratio": down_ratio,
        "max_down_streak_ratio": max_down_streak_ratio,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Price diagnostics: teacher forcing vs recursive")
    parser.add_argument("--config", default="后端/配置.yaml")
    parser.add_argument("--crop", default="", help="optional env_label crop name")
    parser.add_argument("--max-crops", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=0, help="cap validation horizon days")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(config["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = resolve_names(load_name_map(config["paths"]["name_map"]))
    rows: List[Tuple[str, str]] = []
    for crop, m in sorted(mapping.items(), key=lambda x: x[0]):
        pf = str((m or {}).get("price_file", "")).strip()
        if not pf:
            continue
        rows.append((crop, pf))

    if args.crop:
        target = str(args.crop).strip().lower()
        rows = [(c, pf) for c, pf in rows if str(c).lower() == target]

    if args.max_crops and args.max_crops > 0:
        rows = rows[: int(args.max_crops)]

    summaries: List[Dict[str, Any]] = []
    for crop, price_file in rows:
        try:
            summary = _diagnose_crop(
                crop=crop,
                price_file=price_file,
                config=config,
                out_dir=out_dir,
                horizon_cap=(args.horizon if args.horizon > 0 else None),
            )
        except Exception as exc:
            summary = {
                "crop": crop,
                "price_file": price_file,
                "ok": False,
                "error": str(exc),
            }
        summaries.append(summary)

    out_json = out_dir / "价格诊断报告.json"
    out_csv = out_dir / "价格诊断汇总.csv"
    out_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(out_csv, index=False)

    print(out_json.as_posix())
    print(out_csv.as_posix())


if __name__ == "__main__":
    main()
