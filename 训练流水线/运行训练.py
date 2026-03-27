from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from 后端.发布治理 import build_release_from_output
from 后端.环境桥接 import load_env_scenario_library, resolve_env_scenario_path
from 后端.反馈回流 import get_feedback_training_status
from 训练流水线.数据流水线.加载器 import build_panel_dataset, load_config
from 训练流水线.数据流水线.输出生命周期 import apply_output_lifecycle
from 训练流水线.集成.融合 import apply_score, build_uncertainty_risk, optimize_score_weights
from 训练流水线.评估.指标 import metrics_by_group, profit_mae, ranking_metrics_by_year, split_metrics_all_real
from 训练流水线.特征.面板特征 import build_task_frames
from 训练流水线.模型.任务训练 import (
    TaskTrainResult,
    add_crop_residual_correction,
    apply_task_bias_calibration,
    train_global_task,
    train_price_with_gate,
)


def _repo_abs(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (Path(ROOT) / path).resolve()


def _resolve_backend_config_path(config: dict, backend_config_path: str | None) -> Optional[Path]:
    if backend_config_path:
        return _repo_abs(backend_config_path)
    closed_loop_cfg = config.get("closed_loop", {}) if isinstance(config, dict) else {}
    if isinstance(closed_loop_cfg, dict):
        text = str(closed_loop_cfg.get("backend_config", "")).strip()
        if text:
            return _repo_abs(text)
    return _repo_abs("后端/配置.yaml")


def _build_env_probability_summary(
    *,
    panel: pd.DataFrame,
    env_prior: Dict[str, float],
    config: dict,
    backend_config_path: str | None = None,
) -> dict:
    source = "empirical_prior"
    if isinstance(panel, pd.DataFrame) and not panel.empty and "env_prob_source" in panel.columns:
        raw_source = str(panel["env_prob_source"].iloc[0] or "").strip()
        if raw_source:
            source = raw_source

    summary = {
        "source": source,
        "crop_count": int(len(env_prior)),
        "probability_sum": float(sum(float(v) for v in env_prior.values())) if env_prior else 0.0,
        "backend_config_path": None,
        "scenario_file": None,
        "scenario_count": 0,
    }

    resolved_backend_config = _resolve_backend_config_path(config, backend_config_path)
    if not resolved_backend_config or not resolved_backend_config.exists():
        return summary

    summary["backend_config_path"] = resolved_backend_config.as_posix()
    try:
        backend_cfg = load_config(str(resolved_backend_config))
        scenario_payload = load_env_scenario_library(root=Path(ROOT), config=backend_cfg, rebuild_if_missing=True)
        items = scenario_payload.get("items", []) if isinstance(scenario_payload, dict) else []
        summary["scenario_file"] = resolve_env_scenario_path(Path(ROOT), backend_cfg).as_posix()
        summary["scenario_count"] = int(len(items)) if isinstance(items, list) else 0
    except Exception as exc:
        summary["scenario_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def _build_closed_loop_summary(config: dict, backend_config_path: str | None = None) -> dict:
    summary = {
        "backend_config_path": None,
        "feedback_training": {},
    }
    resolved_backend_config = _resolve_backend_config_path(config, backend_config_path)
    if not resolved_backend_config or not resolved_backend_config.exists():
        return summary

    summary["backend_config_path"] = resolved_backend_config.as_posix()
    try:
        backend_cfg = load_config(str(resolved_backend_config))
        summary["feedback_training"] = get_feedback_training_status(root=Path(ROOT), config=backend_cfg, refresh=True)
    except Exception as exc:
        summary["feedback_training"] = {"error": f"{type(exc).__name__}: {exc}"}
    return summary


def finalize_closed_loop_release(
    *,
    train_config: dict,
    output_dir: str | Path,
    training_report: dict,
    backtest_report: dict,
    backend_config_path: str | None = None,
    release_run_id: str | None = None,
) -> dict:
    resolved_backend_config = _resolve_backend_config_path(train_config, backend_config_path)
    if not resolved_backend_config or not resolved_backend_config.exists():
        raise FileNotFoundError(f"backend config not found: {resolved_backend_config}")

    backend_cfg = load_config(str(resolved_backend_config))
    closed_loop_cfg = train_config.get("closed_loop", {}) if isinstance(train_config, dict) else {}
    source = "training_pipeline_embedded"
    if isinstance(closed_loop_cfg, dict):
        text = str(closed_loop_cfg.get("release_source", "")).strip()
        if text:
            source = text

    return build_release_from_output(
        root=Path(ROOT),
        config=backend_cfg,
        output_dir=_repo_abs(output_dir),
        run_id=str(release_run_id).strip() or None if release_run_id is not None else None,
        source=source,
        training_report=training_report,
        backtest_report=backtest_report,
    )


def _ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rename_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(
        columns={
            "target": f"{task}_true",
            "pred": f"{task}_hat",
            "p10": f"{task}_p10",
            "p50": f"{task}_p50",
            "p90": f"{task}_p90",
        }
    )
    return out


def _merge_three(price_df: pd.DataFrame, yield_df: pd.DataFrame, cost_df: pd.DataFrame) -> pd.DataFrame:
    out = _rename_task(price_df, "price")
    out = out.merge(_rename_task(yield_df, "yield"), on=["crop", "year", "env_prob"], how="inner")
    out = out.merge(_rename_task(cost_df, "cost"), on=["crop", "year", "env_prob"], how="inner")
    out["profit_true"] = out["price_true"] * out["yield_true"] - out["cost_true"]
    out["profit_hat"] = out["price_hat"] * out["yield_hat"] - out["cost_hat"]
    return out


def _postprocess_cost_predictions(pred_df: pd.DataFrame, name_map: pd.DataFrame) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pred_df

    mapping = name_map[["env_label", "cost_name"]].copy()
    mapping["crop"] = mapping["env_label"].astype(str).str.strip().str.lower()
    mapping["cost_name"] = mapping["cost_name"].astype(str).str.strip()
    mapping = mapping[["crop", "cost_name"]].drop_duplicates(subset=["crop"])

    out = pred_df.copy()
    out["crop"] = out["crop"].astype(str).str.strip().str.lower()
    out = out.merge(mapping, on="crop", how="left")

    # Only smooth shared cost series when multiple crops still map to the same cost_name.
    cnt = mapping["cost_name"].value_counts()
    shared_names = set(cnt[cnt > 1].index.tolist())
    shared = out[out["cost_name"].isin(shared_names)].copy()
    solo = out[~out["cost_name"].isin(shared_names)].copy()

    if not shared.empty:
        agg = (
            shared.groupby(["cost_name", "year"], as_index=False)[["pred", "p10", "p50", "p90"]]
            .mean()
            .sort_values(["cost_name", "year"])
        )
        # Cost is structurally inflation-like; enforce non-decreasing yearly trend per shared cost_name.
        for col in ["pred", "p10", "p50", "p90"]:
            agg[col] = agg.groupby("cost_name")[col].cummax()

        # Keep uncertainty band while ensuring ordering.
        width_low = (agg["p50"] - agg["p10"]).abs()
        width_high = (agg["p90"] - agg["p50"]).abs()
        width = np.maximum(0.05 * np.maximum(agg["p50"], 1.0), np.maximum(width_low, width_high))
        agg["pred"] = agg["p50"]
        agg["p10"] = np.maximum(0.0, agg["p50"] - width)
        agg["p90"] = agg["p50"] + width

        shared = shared.drop(columns=["pred", "p10", "p50", "p90"]).merge(agg, on=["cost_name", "year"], how="left")

    out = pd.concat([solo, shared], ignore_index=True, sort=False)
    out = out.drop(columns=["cost_name"], errors="ignore")
    return out


def _blend_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mae_weight: float = 0.35,
    rmse_weight: float = 0.50,
    mape_weight: float = 0.15,
) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if y.size == 0:
        return float("inf")
    scale = max(1.0, float(np.mean(np.abs(y))))
    mae_n = float(np.mean(np.abs(y - p)) / scale)
    rmse_n = float(np.sqrt(np.mean((y - p) ** 2)) / scale)
    mape = float(np.mean(np.abs(y - p) / np.maximum(np.abs(y), 1e-6)))
    return float(mae_weight * mae_n + rmse_weight * rmse_n + mape_weight * mape)


def _cost_trend_objective(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if y.size == 0:
        return float("inf")
    scale = max(1.0, float(np.mean(np.abs(y))))
    mae_n = float(np.mean(np.abs(y - p)) / scale)
    rmse_n = float(np.sqrt(np.mean((y - p) ** 2)) / scale)
    mape = float(np.mean(np.abs(y - p) / np.maximum(np.abs(y), 1e-6)))
    return 0.35 * mae_n + 0.50 * rmse_n + 0.15 * mape


def _normalize_mix_weights(values) -> tuple[float, float, float]:
    arr = np.asarray(list(values or []), dtype=float).reshape(-1)
    if arr.size != 3:
        arr = np.asarray([0.35, 0.35, 0.30], dtype=float)
    arr = np.where(np.isfinite(arr) & (arr >= 0.0), arr, 0.0)
    total = float(np.sum(arr))
    if total <= 1e-12:
        arr = np.asarray([0.35, 0.35, 0.30], dtype=float)
        total = float(np.sum(arr))
    arr = arr / total
    return float(arr[0]), float(arr[1]), float(arr[2])


def _cost_trend_search_config(config: dict) -> dict:
    train_cfg = config.get("training", {}) if isinstance(config, dict) else {}
    raw = train_cfg.get("cost_trend_search", {}) if isinstance(train_cfg, dict) else {}
    if not isinstance(raw, dict):
        raw = {}

    def _float_list(key: str, default: list[float]) -> list[float]:
        vals = raw.get(key, default)
        out = []
        for item in list(vals or []):
            try:
                val = float(item)
            except Exception:
                continue
            if np.isfinite(val):
                out.append(float(val))
        return out or list(default)

    def _mix_candidates() -> list[tuple[float, float, float]]:
        default = [
            (0.35, 0.35, 0.30),
            (0.50, 0.25, 0.25),
            (0.25, 0.50, 0.25),
            (0.25, 0.25, 0.50),
            (0.40, 0.40, 0.20),
            (0.20, 0.40, 0.40),
            (0.40, 0.20, 0.40),
        ]
        raw_items = raw.get("mix_weight_candidates", default)
        out: list[tuple[float, float, float]] = []
        seen = set()
        for item in list(raw_items or []):
            mix = _normalize_mix_weights(item)
            key = tuple(round(v, 6) for v in mix)
            if key in seen:
                continue
            seen.add(key)
            out.append(mix)
        return out or default

    return {
        "enabled": bool(raw.get("enabled", True)),
        "trials": max(1, int(raw.get("trials", 18))),
        "mix_weight_candidates": _mix_candidates(),
        "clip_low_ratio_candidates": _float_list("clip_low_ratio_candidates", [0.55, 0.60, 0.70, 0.80]),
        "clip_high_ratio_candidates": _float_list("clip_high_ratio_candidates", [2.20, 2.80, 3.20]),
        "recent_yoy_low_candidates": _float_list("recent_yoy_low_candidates", [0.80, 0.85, 0.90]),
        "recent_yoy_high_candidates": _float_list("recent_yoy_high_candidates", [1.25, 1.35, 1.45]),
        "q75_yoy_low_candidates": _float_list("q75_yoy_low_candidates", [0.85, 0.90, 0.95]),
        "q75_yoy_high_candidates": _float_list("q75_yoy_high_candidates", [1.35, 1.50, 1.65]),
        "sigma_floor_candidates": _float_list("sigma_floor_candidates", [0.05, 0.08, 0.10, 0.12]),
    }


def _sample_cost_trend_hparams(config: dict, seed: int) -> list[dict]:
    search_cfg = _cost_trend_search_config(config)
    default = {
        "mix_weights": (0.35, 0.35, 0.30),
        "clip_low_ratio": 0.60,
        "clip_high_ratio": 2.80,
        "recent_yoy_bounds": (0.85, 1.35),
        "q75_yoy_bounds": (0.90, 1.50),
        "sigma_floor": 0.08,
    }
    if not bool(search_cfg.get("enabled", True)):
        return [default]

    trials = int(search_cfg.get("trials", 18))
    rng = np.random.RandomState(int(seed))
    out = [default]
    seen = {
        (
            round(default["clip_low_ratio"], 6),
            round(default["clip_high_ratio"], 6),
            round(default["recent_yoy_bounds"][0], 6),
            round(default["recent_yoy_bounds"][1], 6),
            round(default["q75_yoy_bounds"][0], 6),
            round(default["q75_yoy_bounds"][1], 6),
            round(default["sigma_floor"], 6),
            tuple(round(v, 6) for v in default["mix_weights"]),
        )
    }

    max_attempts = max(32, trials * 12)
    attempts = 0
    while len(out) < trials and attempts < max_attempts:
        attempts += 1
        clip_low = float(rng.choice(search_cfg["clip_low_ratio_candidates"]))
        clip_high = float(rng.choice(search_cfg["clip_high_ratio_candidates"]))
        recent_low = float(rng.choice(search_cfg["recent_yoy_low_candidates"]))
        recent_high = float(rng.choice(search_cfg["recent_yoy_high_candidates"]))
        q75_low = float(rng.choice(search_cfg["q75_yoy_low_candidates"]))
        q75_high = float(rng.choice(search_cfg["q75_yoy_high_candidates"]))
        sigma_floor = float(rng.choice(search_cfg["sigma_floor_candidates"]))
        mix_weights = tuple(search_cfg["mix_weight_candidates"][int(rng.randint(len(search_cfg["mix_weight_candidates"])))])
        if clip_low >= clip_high or recent_low >= recent_high or q75_low >= q75_high:
            continue
        key = (
            round(clip_low, 6),
            round(clip_high, 6),
            round(recent_low, 6),
            round(recent_high, 6),
            round(q75_low, 6),
            round(q75_high, 6),
            round(sigma_floor, 6),
            tuple(round(v, 6) for v in mix_weights),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "mix_weights": mix_weights,
                "clip_low_ratio": clip_low,
                "clip_high_ratio": clip_high,
                "recent_yoy_bounds": (recent_low, recent_high),
                "q75_yoy_bounds": (q75_low, q75_high),
                "sigma_floor": sigma_floor,
            }
        )
    return out


def _prefer_real_history_rows(
    history: pd.DataFrame,
    *,
    entity_col: str,
    value_col: str,
) -> pd.DataFrame:
    base = history[[entity_col, "year", value_col]].copy()
    base[entity_col] = base[entity_col].astype(str).str.strip()
    base["year"] = pd.to_numeric(base["year"], errors="coerce")
    base[value_col] = pd.to_numeric(base[value_col], errors="coerce")
    base = base.dropna(subset=[entity_col, "year", value_col])
    return base


def _yearly_entity_trend_table(
    history: pd.DataFrame,
    *,
    entity_col: str,
    value_col: str,
    train_start: int,
    cutoff_year: int,
    pred_years: list,
    method: str = "poly",
    clip_low_ratio: float = 0.60,
    clip_high_ratio: float = 2.80,
    recent_yoy_bounds: tuple[float, float] = (0.85, 1.35),
    q75_yoy_bounds: tuple[float, float] = (0.90, 1.50),
    sigma_floor: float = 0.08,
    mix_weights: tuple[float, float, float] = (0.35, 0.35, 0.30),
) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=[entity_col, "year", "pred_trend", "p10_trend", "p50_trend", "p90_trend"])

    base = _prefer_real_history_rows(
        history,
        entity_col=entity_col,
        value_col=value_col,
    )
    base = base[(base["year"] >= int(train_start)) & (base["year"] <= int(cutoff_year))].copy()
    if base.empty:
        return pd.DataFrame(columns=[entity_col, "year", "pred_trend", "p10_trend", "p50_trend", "p90_trend"])

    base = (
        base.groupby([entity_col, "year"], as_index=False)[value_col]
        .mean()
        .sort_values([entity_col, "year"])
    )

    method = str(method).strip().lower()
    if method not in {"poly", "recent", "q75", "mix"}:
        method = "poly"
    mix_poly, mix_recent, mix_q75 = _normalize_mix_weights(mix_weights)

    recent_lo, recent_hi = recent_yoy_bounds
    q75_lo, q75_hi = q75_yoy_bounds
    rows = []

    for entity, g in base.groupby(entity_col):
        yrs = g["year"].to_numpy(dtype=float)
        vals = np.maximum(g[value_col].to_numpy(dtype=float), 1e-6)
        if yrs.size == 0:
            continue

        logv = np.log(vals)
        last = float(vals[-1])
        last_year = int(yrs[-1])
        yoy = vals[1:] / vals[:-1] if len(vals) >= 2 else np.asarray([], dtype=float)
        log_yoy = np.log(np.clip(yoy, 1e-6, None)) if yoy.size else np.asarray([], dtype=float)
        sigma = float(np.nanstd(log_yoy))
        if (not np.isfinite(sigma)) or sigma < float(sigma_floor):
            sigma = float(sigma_floor)

        if yrs.size >= 4:
            coef = np.polyfit(yrs, logv, deg=2)
            pred_poly = lambda y: float(np.exp(np.polyval(coef, float(y))))
        elif yrs.size >= 2:
            coef = np.polyfit(yrs, logv, deg=1)
            pred_poly = lambda y: float(np.exp(np.polyval(coef, float(y))))
        else:
            pred_poly = lambda y: float(last)

        recent_yoy = float(np.nanmean(yoy[-5:])) if yoy.size else 1.03
        recent_yoy = float(np.clip(recent_yoy, recent_lo, recent_hi))
        q75_yoy = float(np.nanquantile(yoy, 0.75)) if yoy.size else recent_yoy
        q75_yoy = float(np.clip(q75_yoy, q75_lo, q75_hi))

        def pred_recent(y: int) -> float:
            h = max(1, int(y) - int(last_year))
            return float(last * (recent_yoy ** h))

        def pred_q75(y: int) -> float:
            h = max(1, int(y) - int(last_year))
            return float(last * (q75_yoy ** h))

        for yy in pred_years:
            yint = int(yy)
            p_poly = pred_poly(yint)
            p_recent = pred_recent(yint)
            p_q75 = pred_q75(yint)
            if method == "poly":
                pred = p_poly
            elif method == "recent":
                pred = p_recent
            elif method == "q75":
                pred = p_q75
            else:
                pred = mix_poly * p_poly + mix_recent * p_recent + mix_q75 * p_q75
            pred = float(np.clip(pred, clip_low_ratio * last, clip_high_ratio * last))
            p10 = float(max(0.0, pred * np.exp(-1.2816 * sigma)))
            p90 = float(pred * np.exp(1.2816 * sigma))
            rows.append(
                {
                    entity_col: str(entity),
                    "year": int(yy),
                    "pred_trend": pred,
                    "p10_trend": min(p10, pred),
                    "p50_trend": pred,
                    "p90_trend": max(p90, pred),
                }
            )
    return pd.DataFrame(rows)


def _cost_name_trend_table(
    panel: pd.DataFrame,
    train_start: int,
    cutoff_year: int,
    pred_years: list,
    method: str = "poly",
    trend_params: Optional[dict] = None,
) -> pd.DataFrame:
    params = dict(trend_params or {})
    return _yearly_entity_trend_table(
        panel,
        entity_col="cost_name",
        value_col="cost",
        train_start=train_start,
        cutoff_year=cutoff_year,
        pred_years=pred_years,
        method=method,
        clip_low_ratio=float(params.get("clip_low_ratio", 0.60)),
        clip_high_ratio=float(params.get("clip_high_ratio", 2.80)),
        recent_yoy_bounds=tuple(params.get("recent_yoy_bounds", (0.85, 1.35))),
        q75_yoy_bounds=tuple(params.get("q75_yoy_bounds", (0.90, 1.50))),
        sigma_floor=float(params.get("sigma_floor", 0.08)),
        mix_weights=tuple(params.get("mix_weights", (0.35, 0.35, 0.30))),
    )


def _attach_cost_name(df: pd.DataFrame, name_map: pd.DataFrame) -> pd.DataFrame:
    mapping = name_map[["env_label", "cost_name"]].copy()
    mapping["crop"] = mapping["env_label"].astype(str).str.strip().str.lower()
    mapping["cost_name"] = mapping["cost_name"].astype(str).str.strip()
    mapping = mapping[["crop", "cost_name"]].drop_duplicates(subset=["crop"])
    out = df.copy()
    out["crop"] = out["crop"].astype(str).str.strip().str.lower()
    out = out.merge(mapping, on="crop", how="left")
    return out


def _cost_trend_predict_split(
    split_df: pd.DataFrame,
    panel: pd.DataFrame,
    name_map: pd.DataFrame,
    train_start: int,
    dynamic_cutoff: bool,
    fixed_cutoff_year: int,
    trend_params: Optional[dict] = None,
) -> pd.DataFrame:
    if split_df is None or split_df.empty:
        return split_df

    out = _attach_cost_name(split_df, name_map)
    years = sorted([int(y) for y in out["year"].dropna().unique().tolist()])

    methods = ["poly", "recent", "q75", "mix"]

    def attach_methods(base_df: pd.DataFrame, cutoff: int, pred_years: list) -> pd.DataFrame:
        cur = base_df.copy()
        for m in methods:
            tab = _cost_name_trend_table(
                panel,
                train_start=train_start,
                cutoff_year=cutoff,
                pred_years=pred_years,
                method=m,
                trend_params=trend_params,
            )
            ren = {
                "pred_trend": f"pred_trend_{m}",
                "p10_trend": f"p10_trend_{m}",
                "p50_trend": f"p50_trend_{m}",
                "p90_trend": f"p90_trend_{m}",
            }
            cur = cur.merge(tab.rename(columns=ren), on=["cost_name", "year"], how="left")
        return cur

    chunks = []
    if dynamic_cutoff:
        for y in years:
            sub = out[out["year"] == y].copy()
            sub = attach_methods(sub, cutoff=y - 1, pred_years=[y])
            chunks.append(sub)
    else:
        out = attach_methods(out, cutoff=fixed_cutoff_year, pred_years=years)
        chunks = [out]

    pred = pd.concat(chunks, ignore_index=True) if chunks else out
    return pred


def _fit_cost_trend_strategy(
    val_aug: pd.DataFrame,
    *,
    shared_names: set,
    methods: list[str],
    weight_grid: list[float],
    shared_alpha_grid: list[float],
) -> dict:
    strategy = {
        "global_method": "poly",
        "method_map": {},
        "global_w": 0.5,
        "w_map": {},
        "shared_alpha": 0.0,
        "objective": float("inf"),
    }
    if val_aug is None or val_aug.empty:
        return strategy

    method_score = {}
    y_all = pd.to_numeric(val_aug["target"], errors="coerce").to_numpy(dtype=float)
    for m in methods:
        pcol = f"pred_trend_{m}"
        if pcol not in val_aug.columns:
            continue
        p_all = pd.to_numeric(val_aug[pcol], errors="coerce").to_numpy(dtype=float)
        method_score[m] = _cost_trend_objective(y_all, p_all)
    finite_method_score = {k: v for k, v in method_score.items() if np.isfinite(v)}
    global_method = min(finite_method_score.items(), key=lambda kv: kv[1])[0] if finite_method_score else "poly"

    method_map = {}
    for cname, g in val_aug.groupby("cost_name"):
        y = pd.to_numeric(g["target"], errors="coerce").to_numpy(dtype=float)
        cand = {}
        for m in methods:
            pcol = f"pred_trend_{m}"
            if pcol not in g.columns:
                continue
            p = pd.to_numeric(g[pcol], errors="coerce").to_numpy(dtype=float)
            cand[m] = _cost_trend_objective(y, p)
        cand = {k: v for k, v in cand.items() if np.isfinite(v)}
        method_map[str(cname)] = min(cand.items(), key=lambda kv: kv[1])[0] if cand else global_method

    def pick_trend_col(df_in: pd.DataFrame) -> pd.DataFrame:
        out = df_in.copy()
        out["trend_method"] = out["cost_name"].astype(str).map(method_map).fillna(global_method)
        for c in ["pred", "p10", "p50", "p90"]:
            out[f"{c}_trend"] = np.nan
            for m in methods:
                src = f"{c}_trend_{m}"
                if src not in out.columns:
                    continue
                mask = out["trend_method"] == m
                if mask.any():
                    out.loc[mask, f"{c}_trend"] = pd.to_numeric(out.loc[mask, src], errors="coerce")
        return out

    val_pick = pick_trend_col(val_aug)
    base_all = pd.to_numeric(val_pick["pred"], errors="coerce").to_numpy(dtype=float)
    trend_all = pd.to_numeric(val_pick["pred_trend"], errors="coerce").to_numpy(dtype=float)
    trend_all = np.where(np.isfinite(trend_all), trend_all, base_all)

    global_w = 0.5
    g_best_obj = float("inf")
    for w in weight_grid:
        pred = (1.0 - float(w)) * base_all + float(w) * trend_all
        o = _cost_trend_objective(y_all, pred)
        if o < g_best_obj:
            g_best_obj = o
            global_w = float(w)

    w_map = {}
    for cname, g in val_pick.groupby("cost_name"):
        y = pd.to_numeric(g["target"], errors="coerce").to_numpy(dtype=float)
        base = pd.to_numeric(g["pred"], errors="coerce").to_numpy(dtype=float)
        trend = pd.to_numeric(g["pred_trend"], errors="coerce").to_numpy(dtype=float)
        trend = np.where(np.isfinite(trend), trend, base)
        c_base_obj = _cost_trend_objective(y, base)
        c_best_obj = c_base_obj
        c_best_w = 0.0
        for w in weight_grid:
            pred = (1.0 - float(w)) * base + float(w) * trend
            o = _cost_trend_objective(y, pred)
            if o < c_best_obj:
                c_best_obj = o
                c_best_w = float(w)
        n = int(len(g))
        shrink = float(n / (n + 6.0))
        w = float(np.clip(shrink * c_best_w + (1.0 - shrink) * global_w, 0.0, 0.95))
        w_map[str(cname)] = w

    def apply_without_shared(frame: pd.DataFrame) -> pd.DataFrame:
        out = pick_trend_col(frame)
        out["w_trend"] = out["cost_name"].astype(str).map(w_map).fillna(global_w).astype(float)
        for c in ["pred_trend", "p10_trend", "p50_trend", "p90_trend"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        out["pred_trend"] = out["pred_trend"].fillna(out["pred"])
        out["p10_trend"] = out["p10_trend"].fillna(out["p10"])
        out["p50_trend"] = out["p50_trend"].fillna(out["p50"])
        out["p90_trend"] = out["p90_trend"].fillna(out["p90"])

        w = out["w_trend"].to_numpy(dtype=float)
        out["pred"] = (1.0 - w) * pd.to_numeric(out["pred"], errors="coerce").to_numpy(dtype=float) + w * out["pred_trend"].to_numpy(dtype=float)
        out["p50"] = (1.0 - w) * pd.to_numeric(out["p50"], errors="coerce").to_numpy(dtype=float) + w * out["p50_trend"].to_numpy(dtype=float)
        out["p10"] = (1.0 - w) * pd.to_numeric(out["p10"], errors="coerce").to_numpy(dtype=float) + w * out["p10_trend"].to_numpy(dtype=float)
        out["p90"] = (1.0 - w) * pd.to_numeric(out["p90"], errors="coerce").to_numpy(dtype=float) + w * out["p90_trend"].to_numpy(dtype=float)
        return out

    val_core = apply_without_shared(val_aug)
    y_val = pd.to_numeric(val_core["target"], errors="coerce").to_numpy(dtype=float)
    pred_base = pd.to_numeric(val_core["pred"], errors="coerce").to_numpy(dtype=float)
    best_obj = _cost_trend_objective(y_val, pred_base)
    best_alpha = 0.0
    mask_shared_val = val_core["cost_name"].isin(shared_names).to_numpy(dtype=bool)
    if mask_shared_val.any():
        grp_pred = val_core.groupby(["cost_name", "year"])["pred"].transform("mean").to_numpy(dtype=float)
        for alpha in shared_alpha_grid:
            a = float(alpha)
            pred = pred_base.copy()
            if a > 0.0:
                pred[mask_shared_val] = (1.0 - a) * pred[mask_shared_val] + a * grp_pred[mask_shared_val]
            o = _cost_trend_objective(y_val, pred)
            if o < best_obj:
                best_obj = o
                best_alpha = a

    strategy.update(
        {
            "global_method": global_method,
            "method_map": method_map,
            "global_w": float(global_w),
            "w_map": w_map,
            "shared_alpha": float(best_alpha),
            "objective": float(best_obj),
        }
    )
    return strategy


def _apply_cost_trend_strategy(
    frame: pd.DataFrame,
    *,
    shared_names: set,
    methods: list[str],
    strategy: dict,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame

    global_method = str(strategy.get("global_method", "poly"))
    method_map = dict(strategy.get("method_map", {}))
    global_w = float(strategy.get("global_w", 0.5))
    w_map = dict(strategy.get("w_map", {}))
    shared_alpha = float(strategy.get("shared_alpha", 0.0))

    out = frame.copy()
    out["trend_method"] = out["cost_name"].astype(str).map(method_map).fillna(global_method)
    for c in ["pred", "p10", "p50", "p90"]:
        out[f"{c}_trend"] = np.nan
        for m in methods:
            src = f"{c}_trend_{m}"
            if src not in out.columns:
                continue
            mask = out["trend_method"] == m
            if mask.any():
                out.loc[mask, f"{c}_trend"] = pd.to_numeric(out.loc[mask, src], errors="coerce")

    out["w_trend"] = out["cost_name"].astype(str).map(w_map).fillna(global_w).astype(float)
    for c in ["pred_trend", "p10_trend", "p50_trend", "p90_trend"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["pred_trend"] = out["pred_trend"].fillna(out["pred"])
    out["p10_trend"] = out["p10_trend"].fillna(out["p10"])
    out["p50_trend"] = out["p50_trend"].fillna(out["p50"])
    out["p90_trend"] = out["p90_trend"].fillna(out["p90"])

    w = out["w_trend"].to_numpy(dtype=float)
    out["pred"] = (1.0 - w) * pd.to_numeric(out["pred"], errors="coerce").to_numpy(dtype=float) + w * out["pred_trend"].to_numpy(dtype=float)
    out["p50"] = (1.0 - w) * pd.to_numeric(out["p50"], errors="coerce").to_numpy(dtype=float) + w * out["p50_trend"].to_numpy(dtype=float)
    out["p10"] = (1.0 - w) * pd.to_numeric(out["p10"], errors="coerce").to_numpy(dtype=float) + w * out["p10_trend"].to_numpy(dtype=float)
    out["p90"] = (1.0 - w) * pd.to_numeric(out["p90"], errors="coerce").to_numpy(dtype=float) + w * out["p90_trend"].to_numpy(dtype=float)

    mask_shared = out["cost_name"].isin(shared_names)
    if mask_shared.any() and shared_alpha > 0.0:
        for col in ["pred", "p10", "p50", "p90"]:
            grp = out.loc[mask_shared].groupby(["cost_name", "year"])[col].transform("mean")
            cur = pd.to_numeric(out.loc[mask_shared, col], errors="coerce").to_numpy(dtype=float)
            out.loc[mask_shared, col] = (1.0 - shared_alpha) * cur + shared_alpha * grp.to_numpy(dtype=float)

    out["p50"] = pd.to_numeric(out["pred"], errors="coerce")
    out["p10"] = np.minimum(out["p10"], out["p50"])
    out["p90"] = np.maximum(out["p90"], out["p50"])
    out = out.drop(
        columns=[
            "pred_trend",
            "p10_trend",
            "p50_trend",
            "p90_trend",
            "w_trend",
            "trend_method",
        ]
        + [f"{c}_trend_{m}" for c in ["pred", "p10", "p50", "p90"] for m in methods],
        errors="ignore",
    )
    return out


def _blend_cost_with_trend(
    cost_res: TaskTrainResult,
    panel: pd.DataFrame,
    name_map: pd.DataFrame,
    config: dict,
) -> TaskTrainResult:
    time_cfg = config.get("time", {})
    train_start = int(time_cfg.get("train_start_year", 2010))
    val_years = [int(y) for y in time_cfg.get("val_years", [2019, 2020, 2021])]
    test_years = [int(y) for y in time_cfg.get("test_years", [2022, 2023, 2024])]
    mapping = name_map[["env_label", "cost_name"]].copy()
    mapping["crop"] = mapping["env_label"].astype(str).str.strip().str.lower()
    shared_names = set(mapping["cost_name"].value_counts()[lambda s: s > 1].index.tolist())

    val_aug = _cost_trend_predict_split(
        cost_res.val,
        panel=panel,
        name_map=name_map,
        train_start=train_start,
        dynamic_cutoff=True,
        fixed_cutoff_year=max(val_years) - 1,
    )
    test_aug = _cost_trend_predict_split(
        cost_res.test,
        panel=panel,
        name_map=name_map,
        train_start=train_start,
        dynamic_cutoff=False,
        fixed_cutoff_year=max(val_years),
    )
    infer_aug = _cost_trend_predict_split(
        cost_res.infer,
        panel=panel,
        name_map=name_map,
        train_start=train_start,
        dynamic_cutoff=False,
        fixed_cutoff_year=max(test_years),
    )

    methods = ["poly", "recent", "q75", "mix"]
    weight_grid = np.linspace(0.0, 0.95, 20).tolist()
    shared_alpha_grid = [0.0, 0.25, 0.50, 0.75, 1.0]
    search_seed = int(config.get("training", {}).get("random_seed", 42)) + 17067
    trend_candidates = _sample_cost_trend_hparams(config, seed=search_seed)
    search_rows = []
    best_strategy = None
    best_params = None
    best_val_aug = None
    best_obj = float("inf")

    for trial_idx, trend_params in enumerate(trend_candidates):
        cur_val_aug = _cost_trend_predict_split(
            cost_res.val,
            panel=panel,
            name_map=name_map,
            train_start=train_start,
            dynamic_cutoff=True,
            fixed_cutoff_year=max(val_years) - 1,
            trend_params=trend_params,
        )
        strategy = _fit_cost_trend_strategy(
            cur_val_aug,
            shared_names=shared_names,
            methods=methods,
            weight_grid=weight_grid,
            shared_alpha_grid=shared_alpha_grid,
        )
        score = float(strategy.get("objective", float("inf")))
        row = {
            "trial": int(trial_idx),
            "objective": score,
            "global_method": str(strategy.get("global_method", "poly")),
            "global_w": float(strategy.get("global_w", 0.5)),
            "shared_alpha": float(strategy.get("shared_alpha", 0.0)),
            "clip_low_ratio": float(trend_params.get("clip_low_ratio", 0.60)),
            "clip_high_ratio": float(trend_params.get("clip_high_ratio", 2.80)),
            "recent_yoy_low": float(trend_params.get("recent_yoy_bounds", (0.85, 1.35))[0]),
            "recent_yoy_high": float(trend_params.get("recent_yoy_bounds", (0.85, 1.35))[1]),
            "q75_yoy_low": float(trend_params.get("q75_yoy_bounds", (0.90, 1.50))[0]),
            "q75_yoy_high": float(trend_params.get("q75_yoy_bounds", (0.90, 1.50))[1]),
            "sigma_floor": float(trend_params.get("sigma_floor", 0.08)),
            "mix_weights": [float(v) for v in trend_params.get("mix_weights", (0.35, 0.35, 0.30))],
        }
        search_rows.append(row)
        if score < best_obj:
            best_obj = score
            best_strategy = strategy
            best_params = dict(trend_params)
            best_val_aug = cur_val_aug

    if best_strategy is None:
        best_params = {
            "mix_weights": (0.35, 0.35, 0.30),
            "clip_low_ratio": 0.60,
            "clip_high_ratio": 2.80,
            "recent_yoy_bounds": (0.85, 1.35),
            "q75_yoy_bounds": (0.90, 1.50),
            "sigma_floor": 0.08,
        }
        best_strategy = {
            "global_method": "poly",
            "method_map": {},
            "global_w": 0.5,
            "w_map": {},
            "shared_alpha": 0.0,
            "objective": float("inf"),
        }
        best_val_aug = _cost_trend_predict_split(
            cost_res.val,
            panel=panel,
            name_map=name_map,
            train_start=train_start,
            dynamic_cutoff=True,
            fixed_cutoff_year=max(val_years) - 1,
            trend_params=best_params,
        )

    val_aug = best_val_aug
    test_aug = _cost_trend_predict_split(
        cost_res.test,
        panel=panel,
        name_map=name_map,
        train_start=train_start,
        dynamic_cutoff=False,
        fixed_cutoff_year=max(val_years),
        trend_params=best_params,
    )
    infer_aug = _cost_trend_predict_split(
        cost_res.infer,
        panel=panel,
        name_map=name_map,
        train_start=train_start,
        dynamic_cutoff=False,
        fixed_cutoff_year=max(test_years),
        trend_params=best_params,
    )

    out_val = _apply_cost_trend_strategy(val_aug, shared_names=shared_names, methods=methods, strategy=best_strategy)
    out_test = _apply_cost_trend_strategy(test_aug, shared_names=shared_names, methods=methods, strategy=best_strategy)
    out_infer = _apply_cost_trend_strategy(infer_aug, shared_names=shared_names, methods=methods, strategy=best_strategy)

    report = dict(cost_res.report)
    postprocess = dict(report.get("postprocess", {}))
    postprocess.update(
        {
            "real_history_preferred": True,
            "trend_source": "cost_name",
            "cost_trend_blend": True,
            "global_trend_method": str(best_strategy.get("global_method", "poly")),
            "cost_name_trend_method": dict(best_strategy.get("method_map", {})),
            "global_trend_weight": float(best_strategy.get("global_w", 0.5)),
            "cost_name_weights": dict(best_strategy.get("w_map", {})),
            "shared_cost_smooth_alpha": float(best_strategy.get("shared_alpha", 0.0)),
            "cost_trend_hpo_enabled": bool(_cost_trend_search_config(config).get("enabled", True)),
            "cost_trend_hpo_best_params": best_params,
            "cost_trend_hpo_objective": float(best_strategy.get("objective", float("inf"))),
            "cost_trend_hpo_search_history": search_rows,
        }
    )
    report["postprocess"] = postprocess
    return TaskTrainResult(val=out_val, test=out_test, infer=out_infer, report=report)


def _yield_trend_table(
    panel: pd.DataFrame,
    train_start: int,
    cutoff_year: int,
    pred_years: list,
    method: str = "poly",
) -> pd.DataFrame:
    return _yearly_entity_trend_table(
        panel,
        entity_col="crop",
        value_col="yield",
        train_start=train_start,
        cutoff_year=cutoff_year,
        pred_years=pred_years,
        method=method,
        clip_low_ratio=0.55,
        clip_high_ratio=2.60,
        recent_yoy_bounds=(0.80, 1.35),
        q75_yoy_bounds=(0.85, 1.45),
        sigma_floor=0.10,
    )


def _yield_trend_predict_split(
    split_df: pd.DataFrame,
    panel: pd.DataFrame,
    train_start: int,
    dynamic_cutoff: bool,
    fixed_cutoff_year: int,
) -> pd.DataFrame:
    if split_df is None or split_df.empty:
        return split_df

    out = split_df.copy()
    years = sorted([int(y) for y in out["year"].dropna().unique().tolist()])
    methods = ["poly", "recent", "q75", "mix"]

    def attach_methods(base_df: pd.DataFrame, cutoff: int, pred_years: list) -> pd.DataFrame:
        cur = base_df.copy()
        for m in methods:
            tab = _yield_trend_table(
                panel,
                train_start=train_start,
                cutoff_year=cutoff,
                pred_years=pred_years,
                method=m,
            )
            ren = {
                "pred_trend": f"pred_trend_{m}",
                "p10_trend": f"p10_trend_{m}",
                "p50_trend": f"p50_trend_{m}",
                "p90_trend": f"p90_trend_{m}",
            }
            cur = cur.merge(tab.rename(columns=ren), on=["crop", "year"], how="left")
        return cur

    chunks = []
    if dynamic_cutoff:
        for y in years:
            sub = out[out["year"] == y].copy()
            sub = attach_methods(sub, cutoff=y - 1, pred_years=[y])
            chunks.append(sub)
    else:
        out = attach_methods(out, cutoff=fixed_cutoff_year, pred_years=years)
        chunks = [out]

    return pd.concat(chunks, ignore_index=True) if chunks else out


def _blend_yield_with_trend(
    yield_res: TaskTrainResult,
    panel: pd.DataFrame,
    config: dict,
) -> TaskTrainResult:
    train_cfg = config.get("training", {})
    time_cfg = config.get("time", {})
    train_start = int(time_cfg.get("train_start_year", 2010))
    val_years = [int(y) for y in time_cfg.get("val_years", [2019, 2020, 2021])]
    test_years = [int(y) for y in time_cfg.get("test_years", [2022, 2023, 2024])]
    min_crop_val_points = int(train_cfg.get("yield_trend_min_crop_val_points", 5))
    crop_improvement_floor = float(train_cfg.get("yield_trend_min_obj_gain", 0.002))
    crop_shrink_k = max(0.0, float(train_cfg.get("yield_trend_crop_shrink_k", 6.0)))

    val_aug = _yield_trend_predict_split(
        yield_res.val,
        panel=panel,
        train_start=train_start,
        dynamic_cutoff=True,
        fixed_cutoff_year=max(val_years) - 1,
    )
    test_aug = _yield_trend_predict_split(
        yield_res.test,
        panel=panel,
        train_start=train_start,
        dynamic_cutoff=False,
        fixed_cutoff_year=max(val_years),
    )
    infer_aug = _yield_trend_predict_split(
        yield_res.infer,
        panel=panel,
        train_start=train_start,
        dynamic_cutoff=False,
        fixed_cutoff_year=max(test_years),
    )

    methods = ["poly", "recent", "q75", "mix"]
    weight_grid = np.linspace(0.0, 0.90, 19).tolist()
    global_method = "poly"
    global_w = 0.0
    method_map = {}
    weight_map = {}
    skipped_crops = {}

    if val_aug is not None and not val_aug.empty:
        y_all = pd.to_numeric(val_aug["target"], errors="coerce").to_numpy(dtype=float)
        base_all = pd.to_numeric(val_aug["pred"], errors="coerce").to_numpy(dtype=float)

        best_obj = _blend_objective(y_all, base_all, mae_weight=0.35, rmse_weight=0.35, mape_weight=0.30)
        for m in methods:
            pcol = f"pred_trend_{m}"
            if pcol not in val_aug.columns:
                continue
            trend_all = pd.to_numeric(val_aug[pcol], errors="coerce").to_numpy(dtype=float)
            trend_all = np.where(np.isfinite(trend_all), trend_all, base_all)
            obj = _blend_objective(y_all, trend_all, mae_weight=0.35, rmse_weight=0.35, mape_weight=0.30)
            if obj < best_obj:
                best_obj = obj
                global_method = m

        base_for_global = pd.to_numeric(val_aug["pred"], errors="coerce").to_numpy(dtype=float)
        global_trend = pd.to_numeric(val_aug.get(f"pred_trend_{global_method}"), errors="coerce").to_numpy(dtype=float)
        global_trend = np.where(np.isfinite(global_trend), global_trend, base_for_global)
        best_obj = _blend_objective(y_all, base_for_global, mae_weight=0.35, rmse_weight=0.35, mape_weight=0.30)
        for w in weight_grid:
            pred = (1.0 - float(w)) * base_for_global + float(w) * global_trend
            obj = _blend_objective(y_all, pred, mae_weight=0.35, rmse_weight=0.35, mape_weight=0.30)
            if obj < best_obj:
                best_obj = obj
                global_w = float(w)

        for crop, g in val_aug.groupby("crop"):
            y = pd.to_numeric(g["target"], errors="coerce").to_numpy(dtype=float)
            base = pd.to_numeric(g["pred"], errors="coerce").to_numpy(dtype=float)
            n = int(len(g))
            crop_base_obj = _blend_objective(y, base, mae_weight=0.35, rmse_weight=0.35, mape_weight=0.30)
            crop_best_obj = crop_base_obj
            crop_method = global_method
            crop_weight = 0.0
            if n < int(max(1, min_crop_val_points)):
                method_map[str(crop)] = global_method
                weight_map[str(crop)] = float(global_w)
                skipped_crops[str(crop)] = {
                    "reason": "insufficient_validation_points",
                    "n_points": n,
                }
                continue
            for m in methods:
                pcol = f"pred_trend_{m}"
                if pcol not in g.columns:
                    continue
                trend = pd.to_numeric(g[pcol], errors="coerce").to_numpy(dtype=float)
                trend = np.where(np.isfinite(trend), trend, base)
                for w in weight_grid:
                    pred = (1.0 - float(w)) * base + float(w) * trend
                    obj = _blend_objective(y, pred, mae_weight=0.35, rmse_weight=0.35, mape_weight=0.30)
                    if obj < crop_best_obj:
                        crop_best_obj = obj
                        crop_method = m
                        crop_weight = float(w)
            gain = float(crop_base_obj - crop_best_obj)
            if gain < crop_improvement_floor:
                method_map[str(crop)] = global_method
                weight_map[str(crop)] = float(global_w)
                skipped_crops[str(crop)] = {
                    "reason": "improvement_below_floor",
                    "n_points": n,
                    "obj_gain": gain,
                }
                continue
            shrink = 1.0 if crop_shrink_k <= 0.0 else float(n / (n + crop_shrink_k))
            method_map[str(crop)] = crop_method
            weight_map[str(crop)] = float(np.clip(shrink * crop_weight + (1.0 - shrink) * global_w, 0.0, 0.90))

    def _pick_trend_columns(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame
        out = frame.copy()
        out["trend_method"] = out["crop"].astype(str).map(method_map).fillna(global_method)
        for col in ["pred", "p10", "p50", "p90"]:
            out[f"{col}_trend"] = np.nan
            for m in methods:
                src = f"{col}_trend_{m}"
                if src not in out.columns:
                    continue
                mask = out["trend_method"] == m
                if mask.any():
                    out.loc[mask, f"{col}_trend"] = pd.to_numeric(out.loc[mask, src], errors="coerce")
            out[f"{col}_trend"] = pd.to_numeric(out[f"{col}_trend"], errors="coerce").fillna(
                pd.to_numeric(out[col], errors="coerce")
            )
        return out

    val_aug = _pick_trend_columns(val_aug)
    test_aug = _pick_trend_columns(test_aug)
    infer_aug = _pick_trend_columns(infer_aug)

    def _apply(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame
        out = frame.copy()
        out["w_trend"] = out["crop"].astype(str).map(weight_map).fillna(global_w).astype(float)
        w = out["w_trend"].to_numpy(dtype=float)

        base_pred = pd.to_numeric(out["pred"], errors="coerce").to_numpy(dtype=float)
        base_p10 = pd.to_numeric(out["p10"], errors="coerce").to_numpy(dtype=float)
        base_p50 = pd.to_numeric(out["p50"], errors="coerce").to_numpy(dtype=float)
        base_p90 = pd.to_numeric(out["p90"], errors="coerce").to_numpy(dtype=float)

        trend_pred = pd.to_numeric(out["pred_trend"], errors="coerce").to_numpy(dtype=float)
        trend_p10 = pd.to_numeric(out["p10_trend"], errors="coerce").to_numpy(dtype=float)
        trend_p50 = pd.to_numeric(out["p50_trend"], errors="coerce").to_numpy(dtype=float)
        trend_p90 = pd.to_numeric(out["p90_trend"], errors="coerce").to_numpy(dtype=float)

        out["pred"] = (1.0 - w) * base_pred + w * trend_pred
        out["p50"] = (1.0 - w) * base_p50 + w * trend_p50
        out["p10"] = (1.0 - w) * base_p10 + w * trend_p10
        out["p90"] = (1.0 - w) * base_p90 + w * trend_p90
        out["pred"] = np.clip(pd.to_numeric(out["pred"], errors="coerce").to_numpy(dtype=float), 0.0, None)
        out["p50"] = out["pred"]
        out["p10"] = np.minimum(pd.to_numeric(out["p10"], errors="coerce"), out["p50"])
        out["p90"] = np.maximum(pd.to_numeric(out["p90"], errors="coerce"), out["p50"])
        out = out.drop(
            columns=[
                "pred_trend",
                "p10_trend",
                "p50_trend",
                "p90_trend",
                "w_trend",
                "trend_method",
            ]
            + [f"{c}_trend_{m}" for c in ["pred", "p10", "p50", "p90"] for m in methods],
            errors="ignore",
        )
        return out

    out_val = _apply(val_aug)
    out_test = _apply(test_aug)
    out_infer = _apply(infer_aug)

    report = dict(yield_res.report)
    postprocess = dict(report.get("postprocess", {}))
    postprocess.update(
        {
            "real_history_preferred": True,
            "trend_source": "crop",
            "yield_trend_blend": True,
            "yield_global_trend_method": global_method,
            "yield_crop_trend_method": method_map,
            "yield_global_trend_weight": float(global_w),
            "yield_crop_weights": weight_map,
            "yield_trend_min_crop_val_points": int(min_crop_val_points),
            "yield_trend_min_obj_gain": float(crop_improvement_floor),
            "yield_trend_crop_shrink_k": float(crop_shrink_k),
            "yield_trend_skipped_crops": skipped_crops,
        }
    )
    report["postprocess"] = postprocess
    return TaskTrainResult(val=out_val, test=out_test, infer=out_infer, report=report)


def _task_metrics(split_df: pd.DataFrame, task: str) -> Dict[str, Dict[str, float]]:
    return split_metrics_all_real(
        split_df,
        target_col=f"{task}_true",
        pred_col=f"{task}_hat",
    )


def _robustness_metrics(split_df: pd.DataFrame, task: str) -> Dict[str, object]:
    return {
        "by_crop": metrics_by_group(
            split_df,
            group_col="crop",
            target_col=f"{task}_true",
            pred_col=f"{task}_hat",
        ),
        "by_year": metrics_by_group(
            split_df,
            group_col="year",
            target_col=f"{task}_true",
            pred_col=f"{task}_hat",
        ),
    }


def _business_metrics(split_df: pd.DataFrame, top_k: int) -> Dict[str, object]:
    rank = ranking_metrics_by_year(split_df, year_col="year", score_col="score", profit_col="profit_true", k=top_k)
    return {
        "profit_mae": float(profit_mae(split_df, true_col="profit_true", pred_col="profit_hat")),
        "topk_avg_profit": float(rank["topk_avg_profit"]),
        "ndcg_at_k": float(rank["ndcg_at_k"]),
        "hit_rate_at_k": float(rank["hit_rate_at_k"]),
        "by_year": rank["years"],
    }


def _safe_float(v) -> float:
    try:
        if v is None:
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _load_repo_baseline(outputs_dir: str) -> Dict[str, Dict[str, float]]:
    out = {}
    price_file = os.path.join(outputs_dir, "价格回测.csv")
    cost_file = os.path.join(outputs_dir, "成本回测.csv")
    yield_file = os.path.join(outputs_dir, "产量回测.json")

    if os.path.exists(price_file):
        p = pd.read_csv(price_file)
        out["price"] = {
            "mae": _safe_float(pd.to_numeric(p.get("mae"), errors="coerce").mean()),
            "rmse": _safe_float(pd.to_numeric(p.get("rmse"), errors="coerce").mean()),
            "mape": _safe_float(pd.to_numeric(p.get("mape"), errors="coerce").mean()),
        }

    if os.path.exists(cost_file):
        c = pd.read_csv(cost_file)
        out["cost"] = {
            "mae": _safe_float(pd.to_numeric(c.get("mae"), errors="coerce").mean()),
            "rmse": _safe_float(pd.to_numeric(c.get("rmse"), errors="coerce").mean()),
            "mape": _safe_float(pd.to_numeric(c.get("mape"), errors="coerce").mean()),
        }

    if os.path.exists(yield_file):
        with open(yield_file, "r", encoding="utf-8") as f:
            y = json.load(f)
        m = y.get("metrics", {}) if isinstance(y, dict) else {}
        out["yield"] = {
            "mae": _safe_float(m.get("mae")),
            "rmse": _safe_float(m.get("rmse")),
            "mape": _safe_float(m.get("mape")),
        }
    return out


def _compare_with_baseline(current: Dict[str, Dict[str, float]], baseline: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    out = {}
    for task in ["price", "yield", "cost"]:
        c = current.get(task, {})
        b = baseline.get(task, {})
        row = {}
        for metric in ["mae", "rmse", "mape"]:
            cv = _safe_float(c.get(metric))
            bv = _safe_float(b.get(metric))
            if np.isfinite(cv) and np.isfinite(bv):
                diff = bv - cv
                pct = (diff / bv * 100.0) if abs(bv) > 1e-9 else float("nan")
                row[metric] = {
                    "baseline": bv,
                    "current": cv,
                    "improvement_abs": diff,
                    "improvement_pct": pct,
                }
        out[task] = row
    return out


def _summarize_baseline_gap(baseline_cmp: Dict[str, object]) -> Dict[str, object]:
    task_labels = {"price": "价格", "yield": "产量", "cost": "成本"}
    task_rows: Dict[str, object] = {}
    underperforming_tasks: List[str] = []
    task_mape_gap: List[tuple] = []

    for task in ["price", "yield", "cost"]:
        task_cmp = baseline_cmp.get(task, {}) if isinstance(baseline_cmp, dict) else {}
        improving_metrics: List[str] = []
        underperforming_metrics: List[str] = []
        missing_metrics: List[str] = []
        metric_improvement_pct: Dict[str, float] = {}

        for metric in ["mae", "rmse", "mape"]:
            row = task_cmp.get(metric, {}) if isinstance(task_cmp, dict) else {}
            pct = _safe_float(row.get("improvement_pct"))
            if np.isfinite(pct):
                metric_improvement_pct[metric] = float(pct)
                if pct < 0:
                    underperforming_metrics.append(metric)
                else:
                    improving_metrics.append(metric)
            else:
                missing_metrics.append(metric)

        if underperforming_metrics:
            underperforming_tasks.append(task)

        mape_gap = _safe_float(metric_improvement_pct.get("mape"))
        if np.isfinite(mape_gap):
            task_mape_gap.append((task, float(mape_gap)))

        task_rows[task] = {
            "task_label": task_labels.get(task, task),
            "wins_all_available_metrics": bool(metric_improvement_pct) and not underperforming_metrics,
            "improving_metrics": improving_metrics,
            "underperforming_metrics": underperforming_metrics,
            "missing_metrics": missing_metrics,
            "metric_improvement_pct": metric_improvement_pct,
            "mape_improvement_pct": mape_gap,
        }

    primary_candidates = [item for item in task_mape_gap if item[0] in underperforming_tasks] or task_mape_gap
    primary_bottleneck = min(primary_candidates, key=lambda item: item[1])[0] if primary_candidates else None
    ordered_by_mape_gap = [task for task, _ in sorted(task_mape_gap, key=lambda item: item[1])]

    return {
        "underperforming_tasks": underperforming_tasks,
        "primary_bottleneck": primary_bottleneck,
        "ordered_by_mape_gap": ordered_by_mape_gap,
        "task_status": task_rows,
    }


def _ablation(task: str, panel: pd.DataFrame, config: dict) -> Dict[str, object]:
    cfg = copy.deepcopy(config)
    cfg["training"]["hpo_trials"] = min(4, int(cfg["training"].get("hpo_trials", 8)))

    variants = {
        "full": {"cross": True, "hier": True, "models": ["hgb", "rf", "etr"], "price_gate": task == "price"},
        "hgb_only": {"cross": True, "hier": True, "models": ["hgb"], "price_gate": False},
        "no_cross_task": {"cross": False, "hier": True, "models": ["hgb"], "price_gate": False},
        "no_crop_hierarchy": {"cross": True, "hier": False, "models": ["hgb"], "price_gate": False},
    }

    rows = {}
    for name, opt in variants.items():
        local_cfg = copy.deepcopy(cfg)
        local_cfg["training"]["base_models"] = opt["models"]
        by_task = dict(local_cfg["training"].get("base_models_by_task", {}))
        by_task[str(task)] = list(opt["models"])
        local_cfg["training"]["base_models_by_task"] = by_task
        frames = build_task_frames(
            panel,
            local_cfg,
            include_cross_task_lags=bool(opt["cross"]),
            include_crop_hierarchy=bool(opt["hier"]),
        )
        df = frames[task]
        if task == "price" and bool(opt["price_gate"]):
            res = train_price_with_gate(df, local_cfg)
        else:
            res = train_global_task(task, df, local_cfg, seed_offset=777)
        val_metrics = split_metrics_all_real(res.val, target_col="target", pred_col="pred")
        rows[name] = {
            "val": val_metrics,
            "n_val": int(len(res.val)),
        }
    return rows


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(payload), f, ensure_ascii=False, indent=2)


def _format_year_list(years) -> str:
    items = [str(int(y)) for y in years if y is not None]
    return "、".join(items) if items else "-"


def _build_markdown_report(path: str, report: dict, config_path: str) -> None:
    time_policy = report.get("time_policy", {}) if isinstance(report, dict) else {}
    env_probability = report.get("env_probability", {}) if isinstance(report, dict) else {}
    closed_loop = report.get("closed_loop", {}) if isinstance(report, dict) else {}
    feedback_training = closed_loop.get("feedback_training", {}) if isinstance(closed_loop, dict) else {}
    train_start_year = int(time_policy.get("train_start_year", 2010))
    val_years = _format_year_list(time_policy.get("walk_forward_val_years", []))
    test_years = _format_year_list(time_policy.get("final_test_years", []))
    inference_year = int(time_policy.get("inference_year", 2025))

    lines = []
    lines.append("# 高精度重构训练报告")
    lines.append("")
    lines.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 训练配置: `{config_path}`")
    lines.append(
        f"- 时间策略: 训练起点 {train_start_year}；walk-forward 验证 {val_years}；最终测试 {test_years}；{inference_year} 仅推理"
    )
    lines.append("")
    lines.append("## 方法")
    lines.append("- 三任务拆分建模（price / yield / cost），每任务 3 类基模型（HGB/RF/ETR）+ stacking。")
    lines.append("- Price 使用全局 panel + 作物专属双轨，并通过门控模型做动态融合。")
    lines.append("- Yield/Cost 使用全局主模型，并叠加作物残差模型。")
    lines.append("- 所有特征均使用历史滞后与历史统计，避免时间泄漏。")
    lines.append("- 输出分位数 P10/P50/P90，并在最终评分中加入不确定性与风险惩罚。")
    lines.append("")

    lines.append("## 环境口径")
    lines.append(
        f"- env_prob 来源: {env_probability.get('source', 'unknown')}；"
        f"作物数={env_probability.get('crop_count', 0)}；"
        f"概率和={env_probability.get('probability_sum', 0.0):.6f}"
    )
    if env_probability.get("scenario_count"):
        lines.append(
            f"- 环境场景库: {env_probability.get('scenario_count')} 个场景；"
            f"文件=`{env_probability.get('scenario_file', '-')}`"
        )
    if env_probability.get("backend_config_path"):
        lines.append(f"- 后端配置: `{env_probability.get('backend_config_path')}`")
    if env_probability.get("scenario_error"):
        lines.append(f"- 场景库加载告警: {env_probability.get('scenario_error')}")
    lines.append("")

    lines.append("## 闭环反馈")
    lines.append(
        f"- 推理事件={feedback_training.get('inference_event_count', 0)}；"
        f"匹配反馈={feedback_training.get('matched_feedback_count', 0)}；"
        f"可训练标签样本={feedback_training.get('labeled_sample_count', 0)}"
    )
    if feedback_training.get("outcome_sample_count"):
        lines.append(f"- 含真实结果回填样本={feedback_training.get('outcome_sample_count', 0)}")
    if feedback_training.get("training_sample_file"):
        lines.append(f"- 反馈样本文件: `{feedback_training.get('training_sample_file')}`")
    if feedback_training.get("latest_feedback_at"):
        lines.append(f"- 最近反馈时间: {feedback_training.get('latest_feedback_at')}")
    if isinstance(feedback_training, dict) and feedback_training.get("error"):
        lines.append(f"- 反馈样本摘要告警: {feedback_training.get('error')}")
    lines.append("")

    back = report.get("backtest", {})
    lines.append("## 关键结果")
    for split in ["val", "test"]:
        task_metrics = back.get("task_metrics", {}).get(split, {})
        biz = back.get("business_metrics", {}).get(split, {})
        lines.append(f"### {split.upper()}")
        for task in ["price", "yield", "cost"]:
            m = task_metrics.get(task, {})
            lines.append(
                f"- {task}: MAE={m.get('mae'):.4f} RMSE={m.get('rmse'):.4f} MAPE={m.get('mape'):.4f}"
                if all(isinstance(m.get(k), (int, float)) and m.get(k) is not None for k in ["mae", "rmse", "mape"])
                else f"- {task}: 指标不可用"
            )
        lines.append(
            f"- business: profit_MAE={biz.get('profit_mae')} topK_profit={biz.get('topk_avg_profit')} "
            f"NDCG@K={biz.get('ndcg_at_k')} hit@K={biz.get('hit_rate_at_k')}"
        )
        lines.append("")

    lines.append("## 与仓库原方案对比")
    cmp = back.get("baseline_comparison", {})
    for task in ["price", "yield", "cost"]:
        t = cmp.get(task, {})
        if not t:
            lines.append(f"- {task}: 无可用基线")
            continue
        for metric in ["mae", "rmse", "mape"]:
            r = t.get(metric, {})
            if not r:
                continue
            lines.append(
                f"- {task}.{metric}: baseline={r.get('baseline'):.4f}, current={r.get('current'):.4f}, "
                f"improve={r.get('improvement_abs'):.4f} ({r.get('improvement_pct'):.2f}%)"
            )
    lines.append("")

    gap_summary = back.get("baseline_gap_summary", {})
    under = gap_summary.get("underperforming_tasks", []) if isinstance(gap_summary, dict) else []
    primary = gap_summary.get("primary_bottleneck") if isinstance(gap_summary, dict) else None
    if under or primary:
        lines.append("## 基线短板摘要")
        if primary:
            lines.append(f"- 当前按 MAPE 看最明确的短板: {primary}")
        if under:
            lines.append(f"- 未全面超过基线的任务: {', '.join(under)}")
        lines.append("")

    lines.append("## 改动文件")
    files = report.get("changed_files", [])
    for f in files:
        lines.append(f"- `{f}`")
    lines.append("")

    lines.append("## 复现命令")
    lines.append("```bash")
    lines.append(f"python -m 训练流水线.运行训练 --config {config_path}")
    lines.append("```")
    if env_probability.get("backend_config_path"):
        lines.append("")
        lines.append("```bash")
        lines.append(
            "python 后端/训练/闭环高精度训练.py "
            f"--train-config {config_path} --backend-config {env_probability.get('backend_config_path')}"
        )
        lines.append("```")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(
    config_path: str,
    *,
    build_release: Optional[bool] = None,
    backend_config_path: str | None = None,
    release_run_id: str | None = None,
) -> Dict[str, object]:
    config = load_config(config_path)
    out_dir = config.get("paths", {}).get("output_dir", "输出")
    _ensure_out_dir(out_dir)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    panel, name_map, env_prior = build_panel_dataset(config)
    env_probability = _build_env_probability_summary(
        panel=panel,
        env_prior=env_prior,
        config=config,
        backend_config_path=backend_config_path,
    )
    closed_loop = _build_closed_loop_summary(config=config, backend_config_path=backend_config_path)

    # Main feature frame (full feature set for yield/cost).
    feat_cfg = config.get("features", {})
    frames_full = build_task_frames(
        panel,
        config,
        include_cross_task_lags=bool(feat_cfg.get("include_cross_task_lags", True)),
        include_crop_hierarchy=bool(feat_cfg.get("include_crop_hierarchy", True)),
    )

    # Price can use a dedicated feature policy.
    price_cross = bool(feat_cfg.get("price_include_cross_task_lags", feat_cfg.get("include_cross_task_lags", True)))
    price_hier = bool(feat_cfg.get("price_include_crop_hierarchy", feat_cfg.get("include_crop_hierarchy", True)))
    if price_cross == bool(feat_cfg.get("include_cross_task_lags", True)) and price_hier == bool(feat_cfg.get("include_crop_hierarchy", True)):
        price_frame = frames_full["price"]
    else:
        price_frame = build_task_frames(
            panel,
            config,
            include_cross_task_lags=price_cross,
            include_crop_hierarchy=price_hier,
        )["price"]

    # Train 3 tasks.
    price_res = train_price_with_gate(price_frame, config)

    yield_global = train_global_task("yield", frames_full["yield"], config, seed_offset=31)
    yield_res = add_crop_residual_correction("yield", yield_global, frames_full["yield"], config)
    yield_res = _blend_yield_with_trend(yield_res, panel=panel, config=config)
    yield_res = apply_task_bias_calibration("yield", yield_res, config=config)

    cost_global = train_global_task("cost", frames_full["cost"], config, seed_offset=67)
    cost_res = add_crop_residual_correction("cost", cost_global, frames_full["cost"], config)
    cost_res = _blend_cost_with_trend(cost_res, panel=panel, name_map=name_map, config=config)
    cost_res = apply_task_bias_calibration("cost", cost_res, config=config)

    # Merge val/test/infer predictions.
    val_df = _merge_three(price_res.val, yield_res.val, cost_res.val)
    test_df = _merge_three(price_res.test, yield_res.test, cost_res.test)
    infer_df = _merge_three(price_res.infer, yield_res.infer, cost_res.infer)

    top_k = int(config.get("evaluation", {}).get("top_k", 5))
    score_trials = int(config.get("evaluation", {}).get("score_search_trials", 300))
    seed = int(config.get("training", {}).get("random_seed", 42))

    score_weights, score_stats = optimize_score_weights(
        val_df,
        top_k=top_k,
        trials=score_trials,
        seed=seed + 999,
        year_col="year",
        true_profit_col="profit_true",
    )

    # Add uncertainty + risk using the searched component weights.
    val_df = build_uncertainty_risk(val_df, component_weights=score_weights)
    test_df = build_uncertainty_risk(test_df, component_weights=score_weights)
    infer_df = build_uncertainty_risk(infer_df, component_weights=score_weights)

    val_df = apply_score(val_df, score_weights)
    test_df = apply_score(test_df, score_weights)
    infer_df = apply_score(infer_df, score_weights)

    # Regression metrics.
    task_metrics_val = {task: _task_metrics(val_df, task) for task in ["price", "yield", "cost"]}
    task_metrics_test = {task: _task_metrics(test_df, task) for task in ["price", "yield", "cost"]}

    # Business metrics.
    business_val = _business_metrics(val_df, top_k=top_k)
    business_test = _business_metrics(test_df, top_k=top_k)

    # Robustness metrics.
    robustness = {
        "val": {task: _robustness_metrics(val_df, task) for task in ["price", "yield", "cost"]},
        "test": {task: _robustness_metrics(test_df, task) for task in ["price", "yield", "cost"]},
    }

    # Baseline compare (repo current outputs).
    repo_baseline = _load_repo_baseline(out_dir)
    current_test_all = {
        task: task_metrics_test[task] for task in ["price", "yield", "cost"]
    }
    baseline_cmp = _compare_with_baseline(current=current_test_all, baseline=repo_baseline)
    baseline_gap_summary = _summarize_baseline_gap(baseline_cmp)

    # Ablation study (can be disabled for faster full training runs).
    run_ablation = bool(config.get("evaluation", {}).get("run_ablation", True))
    if run_ablation:
        ablation = {
            "price": _ablation("price", panel, config),
            "yield": _ablation("yield", panel, config),
            "cost": _ablation("cost", panel, config),
        }
    else:
        ablation = {"enabled": False, "reason": "disabled_by_config"}

    # Recommendation for 2025 inference only.
    rec = infer_df.copy()
    rec = rec.sort_values("score", ascending=False).reset_index(drop=True)
    rec.insert(0, "rank", np.arange(1, len(rec) + 1))
    rec_out = rec[
        [
            "rank",
            "crop",
            "year",
            "score",
            "profit_hat",
            "price_hat",
            "yield_hat",
            "cost_hat",
            "price_p10",
            "price_p50",
            "price_p90",
            "yield_p10",
            "yield_p50",
            "yield_p90",
            "cost_p10",
            "cost_p50",
            "cost_p90",
            "env_prob",
            "risk",
            "uncertainty",
        ]
    ].copy()
    rec_out.to_csv(os.path.join(out_dir, "推荐结果.csv"), index=False)

    model_train_report = {
        "time_policy": {
            "train_start_year": int(config.get("time", {}).get("train_start_year", 2010)),
            "walk_forward_val_years": [int(y) for y in config.get("time", {}).get("val_years", [])],
            "final_test_years": [int(y) for y in config.get("time", {}).get("test_years", [])],
            "inference_year": int(config.get("time", {}).get("inference_year", 2025)),
        },
        "env_probability": env_probability,
        "closed_loop": closed_loop,
        "task_reports": {
            "price": price_res.report,
            "yield": yield_res.report,
            "cost": cost_res.report,
        },
        "score_fusion": {
            "weights": score_weights,
            "validation_objective": score_stats,
        },
        "ablation": ablation,
    }

    backtest_report = {
        "task_metrics": {
            "val": task_metrics_val,
            "test": task_metrics_test,
        },
        "business_metrics": {
            "val": business_val,
            "test": business_test,
        },
        "robustness": robustness,
        "baseline_source": {
            "repo_files": {
                "price": os.path.join(out_dir, "价格回测.csv"),
                "yield": os.path.join(out_dir, "产量回测.json"),
                "cost": os.path.join(out_dir, "成本回测.csv"),
            }
        },
        "baseline_metrics": repo_baseline,
        "baseline_comparison": baseline_cmp,
        "baseline_gap_summary": baseline_gap_summary,
        "score_weights": score_weights,
        "env_probability": env_probability,
        "closed_loop": closed_loop,
    }

    _write_json(os.path.join(out_dir, "模型训练报告.json"), model_train_report)
    _write_json(os.path.join(out_dir, "回测报告.json"), backtest_report)

    changed_files = [
        "训练流水线/配置/高精度.yaml",
        "训练流水线/数据流水线/加载器.py",
        "训练流水线/数据流水线/输出生命周期.py",
        "训练流水线/管理输出.py",
        "训练流水线/输出生命周期.md",
        "训练流水线/特征/面板特征.py",
        "训练流水线/模型/基础模型.py",
        "训练流水线/模型/搜索.py",
        "训练流水线/模型/任务训练.py",
        "训练流水线/集成/融合.py",
        "训练流水线/评估/指标.py",
        "训练流水线/运行训练.py",
        "后端/发布治理.py",
        "后端/反馈回流.py",
        "后端/训练/构建反馈训练样本.py",
        "后端/训练/闭环演示.py",
        "后端/训练/闭环高精度训练.py",
        "后端/训练/兼容重训练.py",
        "后端/路由/核心路由.py",
        "后端/推荐数据源.py",
        "后端/配置.yaml",
        "输出/模型训练报告.json",
        "输出/回测报告.json",
        "输出/推荐结果.csv",
        "输出/高精度训练报告.md",
        "输出/输出生命周期报告.json",
    ]

    markdown_payload = {
        "backtest": backtest_report,
        "changed_files": changed_files,
        "env_probability": env_probability,
        "closed_loop": closed_loop,
        "time_policy": model_train_report["time_policy"],
    }
    _build_markdown_report(
        path=os.path.join(out_dir, "高精度训练报告.md"),
        report=markdown_payload,
        config_path=config_path,
    )

    lifecycle_report = apply_output_lifecycle(out_dir=out_dir, config=config, run_tag=run_tag)
    _write_json(os.path.join(out_dir, "输出生命周期报告.json"), lifecycle_report)

    closed_loop_cfg = config.get("closed_loop", {}) if isinstance(config, dict) else {}
    should_build_release = bool(build_release) if build_release is not None else bool(
        isinstance(closed_loop_cfg, dict) and closed_loop_cfg.get("auto_build_release", False)
    )
    release_summary = None
    if should_build_release:
        release_summary = finalize_closed_loop_release(
            train_config=config,
            output_dir=out_dir,
            training_report=model_train_report,
            backtest_report=backtest_report,
            backend_config_path=backend_config_path,
            release_run_id=release_run_id,
        )

    return {
        "model_train_report": os.path.join(out_dir, "模型训练报告.json"),
        "backtest_report": os.path.join(out_dir, "回测报告.json"),
        "recommendation": os.path.join(out_dir, "推荐结果.csv"),
        "markdown": os.path.join(out_dir, "高精度训练报告.md"),
        "output_lifecycle_report": os.path.join(out_dir, "输出生命周期报告.json"),
        "env_probability": env_probability,
        "closed_loop": closed_loop,
        "release": release_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="High-accuracy crop decision training pipeline")
    parser.add_argument("--config", default="训练流水线/配置/高精度.yaml")
    parser.add_argument("--backend-config", default="")
    parser.add_argument("--release-run-id", default="")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--build-release", action="store_true")
    group.add_argument("--skip-release", action="store_true")
    args = parser.parse_args()

    build_release = None
    if bool(args.build_release):
        build_release = True
    elif bool(args.skip_release):
        build_release = False

    result = run(
        args.config,
        build_release=build_release,
        backend_config_path=str(args.backend_config or "").strip() or None,
        release_run_id=str(args.release_run_id or "").strip() or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



