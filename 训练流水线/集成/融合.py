from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from 训练流水线.评估.指标 import ranking_metrics_by_year


def _normalize_uncertainty_component_weights(weights: Dict[str, float] | None = None) -> Dict[str, float]:
    raw = {
        "w_unc_price": float((weights or {}).get("w_unc_price", 1.0 / 3.0)),
        "w_unc_yield": float((weights or {}).get("w_unc_yield", 1.0 / 3.0)),
        "w_unc_cost": float((weights or {}).get("w_unc_cost", 1.0 / 3.0)),
    }
    arr = np.asarray([max(0.0, raw["w_unc_price"]), max(0.0, raw["w_unc_yield"]), max(0.0, raw["w_unc_cost"])], dtype=float)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 1e-12:
        arr = np.asarray([1.0, 1.0, 1.0], dtype=float)
        total = 3.0
    arr = arr / total
    return {
        "w_unc_price": float(arr[0]),
        "w_unc_yield": float(arr[1]),
        "w_unc_cost": float(arr[2]),
    }


def build_uncertainty_risk(df: pd.DataFrame, component_weights: Dict[str, float] | None = None) -> pd.DataFrame:
    out = df.copy()

    def rel_width(low_col: str, mid_col: str, high_col: str) -> np.ndarray:
        low = pd.to_numeric(out.get(low_col), errors="coerce").to_numpy(dtype=float)
        mid = pd.to_numeric(out.get(mid_col), errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(out.get(high_col), errors="coerce").to_numpy(dtype=float)
        return (high - low) / np.maximum(np.abs(mid), 1e-6)

    u_price = rel_width("price_p10", "price_hat", "price_p90")
    u_yield = rel_width("yield_p10", "yield_hat", "yield_p90")
    u_cost = rel_width("cost_p10", "cost_hat", "cost_p90")

    def _fill_component(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        finite = np.isfinite(arr)
        fill = float(np.nanmedian(arr[finite])) if finite.any() else 0.0
        return np.where(finite, arr, fill)

    u_price = _fill_component(u_price)
    u_yield = _fill_component(u_yield)
    u_cost = _fill_component(u_cost)

    comp_w = _normalize_uncertainty_component_weights(component_weights)
    unc = (
        float(comp_w["w_unc_price"]) * u_price
        + float(comp_w["w_unc_yield"]) * u_yield
        + float(comp_w["w_unc_cost"]) * u_cost
    )
    unc = np.where(np.isfinite(unc), unc, np.nanmedian(unc[np.isfinite(unc)]) if np.isfinite(unc).any() else 0.0)

    out["uncertainty_price"] = np.clip(u_price, 0.0, None)
    out["uncertainty_yield"] = np.clip(u_yield, 0.0, None)
    out["uncertainty_cost"] = np.clip(u_cost, 0.0, None)
    out["uncertainty"] = np.clip(unc, 0.0, None)

    unc_norm = out["uncertainty"].to_numpy(dtype=float)
    if np.nanmax(unc_norm) > np.nanmin(unc_norm):
        unc_norm = (unc_norm - np.nanmin(unc_norm)) / (np.nanmax(unc_norm) - np.nanmin(unc_norm))
    else:
        unc_norm = np.zeros_like(unc_norm)

    out["risk"] = np.clip(unc_norm, 0.0, 1.0)
    return out


def _profit_zscore_by_year(df: pd.DataFrame, year_col: str, profit_col: str) -> np.ndarray:
    out = np.zeros(len(df), dtype=float)
    for _, idx in df.groupby(year_col).groups.items():
        vals = pd.to_numeric(df.loc[idx, profit_col], errors="coerce").to_numpy(dtype=float)
        mu = np.nanmean(vals)
        std = np.nanstd(vals)
        if not np.isfinite(std) or std < 1e-9:
            out[idx] = vals - mu
        else:
            out[idx] = (vals - mu) / std
    return out


def apply_score(
    df: pd.DataFrame,
    weights: Dict[str, float],
    year_col: str = "year",
    profit_col: str = "profit_hat",
    env_col: str = "env_prob",
    risk_col: str = "risk",
    unc_col: str = "uncertainty",
) -> pd.DataFrame:
    out = df.copy()
    pz = _profit_zscore_by_year(out, year_col=year_col, profit_col=profit_col)
    env = pd.to_numeric(out.get(env_col), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    risk = pd.to_numeric(out.get(risk_col), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    unc = pd.to_numeric(out.get(unc_col), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    out["score"] = (
        float(weights.get("w_profit", 1.0)) * pz
        + float(weights.get("w_env", 0.2)) * env
        - float(weights.get("w_risk", 0.4)) * risk
        - float(weights.get("w_uncertainty", 0.3)) * unc
    )
    return out


def optimize_score_weights(
    val_df: pd.DataFrame,
    top_k: int,
    trials: int,
    seed: int,
    year_col: str = "year",
    true_profit_col: str = "profit_true",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if val_df.empty:
        w = {
            "w_profit": 1.0,
            "w_env": 0.2,
            "w_risk": 0.4,
            "w_uncertainty": 0.3,
            "w_unc_price": 1.0 / 3.0,
            "w_unc_yield": 1.0 / 3.0,
            "w_unc_cost": 1.0 / 3.0,
        }
        return w, {"objective": 0.0}

    rng = np.random.RandomState(seed)
    best_w = None
    best_obj = -1e18
    best_stats = {}

    profit_scale = float(np.nanmean(np.abs(pd.to_numeric(val_df[true_profit_col], errors="coerce").to_numpy(dtype=float))))
    profit_scale = max(1.0, profit_scale)

    default_w = {
        "w_profit": 1.0,
        "w_env": 0.2,
        "w_risk": 0.4,
        "w_uncertainty": 0.3,
        "w_unc_price": 1.0 / 3.0,
        "w_unc_yield": 1.0 / 3.0,
        "w_unc_cost": 1.0 / 3.0,
    }

    for trial_idx in range(int(max(1, trials))):
        if trial_idx == 0:
            w = dict(default_w)
        else:
            unc_arr = rng.dirichlet(np.ones(3))
            w = {
            "w_profit": float(rng.uniform(0.6, 2.2)),
            "w_env": float(rng.uniform(0.0, 1.2)),
            "w_risk": float(rng.uniform(0.0, 1.5)),
            "w_uncertainty": float(rng.uniform(0.0, 1.5)),
                "w_unc_price": float(unc_arr[0]),
                "w_unc_yield": float(unc_arr[1]),
                "w_unc_cost": float(unc_arr[2]),
            }
        val_scored_base = build_uncertainty_risk(val_df, component_weights=w)
        scored = apply_score(val_scored_base, weights=w, year_col=year_col, profit_col="profit_hat")
        rank_stats = ranking_metrics_by_year(scored, year_col=year_col, score_col="score", profit_col=true_profit_col, k=top_k)
        topk_profit_norm = float(rank_stats["topk_avg_profit"]) / profit_scale
        obj = float(rank_stats["ndcg_at_k"]) + 0.35 * float(rank_stats["hit_rate_at_k"]) + 0.25 * topk_profit_norm

        if obj > best_obj:
            best_obj = obj
            best_w = w
            best_stats = {
                "objective": float(obj),
                "ndcg_at_k": float(rank_stats["ndcg_at_k"]),
                "hit_rate_at_k": float(rank_stats["hit_rate_at_k"]),
                "topk_avg_profit": float(rank_stats["topk_avg_profit"]),
                "topk_avg_profit_norm": float(topk_profit_norm),
                "uncertainty_weights": {
                    "w_unc_price": float(w["w_unc_price"]),
                    "w_unc_yield": float(w["w_unc_yield"]),
                    "w_unc_cost": float(w["w_unc_cost"]),
                },
            }

    if best_w is None:
        best_w = dict(default_w)
        best_stats = {"objective": 0.0}

    return best_w, best_stats
