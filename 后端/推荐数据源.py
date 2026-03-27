from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from 后端.发布治理 import resolve_active_release
from 后端.推荐器 import recommend as recommend_online
from 后端.时间策略 import (
    prediction_window_payload,
    resolve_price_window_from_dates,
    resolve_price_window_from_price_dir,
    resolve_target_year,
)

_PRECOMPUTED_CACHE: Dict[str, Tuple[int, pd.DataFrame]] = {}


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        out = float(v)
        if pd.isna(out):
            return None
        return out
    except Exception:
        return None


def _safe_int(v: Any) -> Optional[int]:
    fv = _safe_float(v)
    if fv is None:
        return None
    try:
        return int(round(fv))
    except Exception:
        return None


def _parse_date_str(text: Any) -> Optional[pd.Timestamp]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        ts = pd.Timestamp(raw)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.normalize()


def _resolve_prediction_window(config: dict, root: Optional[Path] = None) -> Dict[str, Any]:
    time_cfg = config.get("time", {})
    paths_cfg = config.get("paths", {})
    price_dir = None
    if isinstance(root, Path):
        raw = str(paths_cfg.get("price_dir", "")).strip()
        if raw:
            p = Path(raw)
            price_dir = p if p.is_absolute() else (root / p)

    if isinstance(price_dir, Path):
        policy = resolve_price_window_from_price_dir(price_dir, time_cfg=time_cfg)
    else:
        policy = resolve_price_window_from_dates(None, time_cfg=time_cfg)
    return prediction_window_payload(policy)


def _normalized_confidence(prob: Optional[float]) -> str:
    p = _safe_float(prob)
    if p is None:
        return "low"
    if p >= 0.75:
        return "high"
    if p >= 0.45:
        return "mid"
    return "low"


def _resolve_precomputed_path(root: Path, output_dir: Path, config: dict) -> Path:
    serving_cfg = config.get("serving", {})
    text = str(serving_cfg.get("precomputed_recommendation_file", "")).strip()
    if text:
        p = Path(text)
        return p if p.is_absolute() else (root / p)
    return output_dir / "推荐结果.csv"


def _load_precomputed_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"precomputed recommendation file not found: {path.as_posix()}")
    key = str(path.resolve())
    mtime = path.stat().st_mtime_ns
    hit = _PRECOMPUTED_CACHE.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    df = pd.read_csv(path)
    _PRECOMPUTED_CACHE[key] = (mtime, df)
    return df


def _row_value(row: Dict[str, Any], cols: List[str]) -> Any:
    for col in cols:
        if col in row:
            val = row.get(col)
            if val is None:
                continue
            text = str(val).strip()
            if text.lower() == "nan":
                continue
            return val
    return None


def _canonical_result_row(
    row: Dict[str, Any],
    config: dict,
    prediction_window: Dict[str, Any],
) -> Dict[str, Any]:
    alignment_cfg = config.get("alignment", {})
    target_year = _safe_int(_row_value(row, ["target_year", "year"]))
    if target_year is None:
        target_year = _safe_int(alignment_cfg.get("target_year"))
    if target_year is None:
        end_ts = _parse_date_str(prediction_window.get("end_date"))
        if end_ts is not None:
            target_year = int(end_ts.year)
    if target_year is None:
        target_year = int(pd.Timestamp.today().year)

    crop = str(_row_value(row, ["crop", "crop_name", "label"]) or "").strip().lower()
    env_prob = _safe_float(_row_value(row, ["env_prob", "prob_best", "best_prob"]))
    prob_best = _safe_float(_row_value(row, ["prob_best", "env_prob", "best_prob"]))
    score = _safe_float(_row_value(row, ["score", "score_total"]))
    profit = _safe_float(_row_value(row, ["profit", "profit_hat"]))
    price_pred = _safe_float(_row_value(row, ["price_pred", "price_hat"]))
    price_p10 = _safe_float(_row_value(row, ["price_p10", "pred_p10"]))
    price_p50 = _safe_float(_row_value(row, ["price_p50", "pred_p50", "price_pred", "price_hat"]))
    price_p90 = _safe_float(_row_value(row, ["price_p90", "pred_p90"]))
    yield_pred = _safe_float(_row_value(row, ["yield", "yield_hat"]))
    cost_pred = _safe_float(_row_value(row, ["cost_pred", "cost_hat"]))
    cost_pred_raw = _safe_float(_row_value(row, ["cost_pred_raw", "cost_pred", "cost_hat"]))
    risk = _safe_float(_row_value(row, ["risk"]))
    uncertainty = _safe_float(_row_value(row, ["uncertainty"]))

    if risk is None:
        risk = max(0.0, min(1.0, uncertainty or 0.0))
    if prob_best is None and env_prob is not None:
        prob_best = env_prob

    return {
        "crop": crop,
        "env_prob": env_prob,
        "price_pred": price_pred,
        "price_p10": price_p10,
        "price_p50": price_p50 if price_p50 is not None else price_pred,
        "price_p90": price_p90,
        "price_forecast": [],
        "price_forecast_mode": "precomputed",
        "price_clip_rate": None,
        "price_quality_flag": None,
        "price_blend_weight": None,
        "cost_pred": cost_pred,
        "cost_pred_raw": cost_pred_raw,
        "cost_adjustment": None,
        "yield": yield_pred,
        "yield_source": "precomputed",
        "profit": profit,
        "volatility": None,
        "risk": risk,
        "score": score,
        "score_raw": score,
        "prob_best": prob_best,
        "prob_best_source": "fallback_env_prob",
        "price_file": _row_value(row, ["price_file"]),
        "cost_name": _row_value(row, ["cost_name"]),
        "target_year": int(target_year),
        "prediction_window": prediction_window,
        "time_alignment": {
            "target_year": int(target_year),
            "frequency": "year",
            "strategy": "precomputed_rank",
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


def recommend_from_precomputed(env_input: dict, config: dict, root: Path, output_dir: Path) -> dict:
    _ = env_input  # precomputed mode intentionally ignores request environment values
    t0 = time.perf_counter()
    path = _resolve_precomputed_path(root=root, output_dir=output_dir, config=config)
    source_df = _load_precomputed_df(path)
    if source_df.empty:
        raise ValueError(f"precomputed recommendation file is empty: {path.as_posix()}")

    prediction_window = _resolve_prediction_window(config, root=root)
    rows = source_df.to_dict(orient="records")
    results = [_canonical_result_row(row, config=config, prediction_window=prediction_window) for row in rows]
    results = [r for r in results if r.get("crop")]

    min_prob = _safe_float(config.get("scoring", {}).get("min_env_prob")) or 0.0
    before = len(results)
    filtered_results = [r for r in results if (_safe_float(r.get("env_prob")) or 0.0) >= min_prob]
    fallback_used = False
    if before > 0 and not filtered_results:
        filtered_results = list(results)
        fallback_used = True
    results = filtered_results
    after = len(results)
    results.sort(
        key=lambda r: (
            r.get("score") is None,
            -(_safe_float(r.get("score")) or -1e9),
            -(_safe_float(r.get("env_prob")) or -1e9),
            str(r.get("crop") or ""),
        )
    )

    topk_limit = max(1, int(config.get("scoring", {}).get("max_candidates", 8)))
    final_topk = [{"crop": r["crop"], "env_prob": float(_safe_float(r.get("env_prob")) or 0.0)} for r in results[:8]]
    topk_pairs = [[r["crop"], float(_safe_float(r.get("env_prob")) or 0.0)] for r in results[:topk_limit]]

    best_label = results[0]["crop"] if results else None
    best_prob = float(_safe_float(results[0].get("env_prob")) or 0.0) if results else 0.0
    conf_norm = _normalized_confidence(best_prob)
    target_year = max((int(r.get("target_year")) for r in results if r.get("target_year") is not None), default=None)
    if target_year is None:
        target_year = resolve_target_year(config, fallback_year=int(pd.Timestamp.today().year))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "env": {
            "best_label": best_label,
            "best_prob": best_prob,
            "topk": topk_pairs,
            "warnings": ["precomputed_source_env_not_conditioned"],
            "source": "precomputed_recommendation",
        },
        "env_confidence_norm": conf_norm,
        "final_topk": final_topk,
        "results": results,
        "yield_missing": [],
        "runtime": {
            "elapsed_ms": round(elapsed_ms, 3),
            "model_version": str(config.get("serving", {}).get("model_cache_version", "v2")),
            "missing_models": [],
            "source": "precomputed",
            "precomputed_file": path.as_posix(),
            "precomputed_env_ignored": True,
            "filters": {
                "min_env_prob": min_prob,
                "before": before,
                "after": after,
                "fallback_used": fallback_used,
            },
            "calibrator": {
                "enabled": False,
                "degraded": True,
                "reasons": ["precomputed_source"],
                "fallback_mode": "env_prob",
                "history_stats": {},
            },
            "time_alignment": {
                "target_year": int(target_year),
                "frequency": "year",
                "strategy": "precomputed_rank",
                "score_penalty_per_year": 0.0,
                "max_score_penalty": 0.0,
            },
            "prediction_window": prediction_window,
        },
    }


def recommend_with_source(
    env_input: dict,
    config: dict,
    root: Path,
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> dict:
    effective_config = copy.deepcopy(config)
    release_info: Dict[str, Any] = {"enabled": False}
    try:
        effective_config, release_info = resolve_active_release(root=root, config=config)
    except Exception:
        if logger is not None:
            logger.exception("resolve active release failed, fallback to base config")
        effective_config = copy.deepcopy(config)
        release_info = {"enabled": False, "reason": "resolve_active_release_failed"}

    serving_cfg = effective_config.get("serving", {})
    strategy = str(serving_cfg.get("recommend_strategy", "online")).strip().lower()
    if strategy not in {"online", "precomputed"}:
        strategy = "online"

    if strategy == "online":
        payload = recommend_online(env_input, effective_config)
    else:
        try:
            payload = recommend_from_precomputed(env_input=env_input, config=effective_config, root=root, output_dir=output_dir)
        except Exception:
            fallback_online = bool(serving_cfg.get("precomputed_fallback_online", True))
            if not fallback_online:
                raise
            if logger is not None:
                logger.exception("precomputed recommend failed, fallback to online strategy")
            payload = recommend_online(env_input, effective_config)

    runtime = payload.setdefault("runtime", {})
    runtime["release"] = release_info
    return payload
