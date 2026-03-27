from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _to_float_array(s) -> np.ndarray:
    return np.asarray(pd.to_numeric(s, errors="coerce"), dtype=float)


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if yt.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n": 0.0}
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(math.sqrt(mean_squared_error(yt, yp)))
    denom = np.maximum(np.abs(yt), 1e-6)
    mape = float(np.mean(np.abs((yt - yp) / denom)))
    return {"mae": mae, "rmse": rmse, "mape": mape, "n": float(yt.size)}


def split_metrics_all_real(df: pd.DataFrame, target_col: str, pred_col: str) -> Dict[str, float]:
    return regression_metrics(df[target_col], df[pred_col])


def metrics_by_group(df: pd.DataFrame, group_col: str, target_col: str, pred_col: str) -> Dict[str, Dict[str, float]]:
    out = {}
    if df.empty or group_col not in df.columns:
        return out
    for key, g in df.groupby(group_col):
        out[str(key)] = regression_metrics(g[target_col], g[pred_col])
    return out


def _dcg(relevance: np.ndarray) -> float:
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum((2.0 ** relevance - 1.0) / discounts))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if y_true.size == 0:
        return 0.0
    kk = min(int(k), int(y_true.size))
    if kk <= 0:
        return 0.0

    t = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    t = np.where(np.isfinite(t), t, 0.0)
    s = np.where(np.isfinite(s), s, -1e12)

    # Shift yearly profit to non-negative relevance.
    t_rel = t - np.min(t)
    if np.max(t_rel) > 0:
        t_rel = t_rel / np.max(t_rel)

    idx_pred = np.argsort(-s)[:kk]
    idx_ideal = np.argsort(-t_rel)[:kk]
    dcg = _dcg(t_rel[idx_pred])
    idcg = _dcg(t_rel[idx_ideal])
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def ranking_metrics_by_year(df: pd.DataFrame, year_col: str, score_col: str, profit_col: str, k: int) -> Dict[str, object]:
    if df.empty:
        return {"ndcg_at_k": 0.0, "hit_rate_at_k": 0.0, "topk_avg_profit": 0.0, "years": {}}

    ndcgs: List[float] = []
    hits: List[float] = []
    topk_profit: List[float] = []
    year_rows = {}

    for year, g in df.groupby(year_col):
        g = g.copy()
        g = g[np.isfinite(pd.to_numeric(g[profit_col], errors="coerce"))]
        if g.empty:
            continue
        actual = pd.to_numeric(g[profit_col], errors="coerce").to_numpy(dtype=float)
        score = pd.to_numeric(g[score_col], errors="coerce").to_numpy(dtype=float)
        kk = min(int(k), len(g))
        if kk <= 0:
            continue

        ndcg = ndcg_at_k(actual, score, kk)

        pred_idx = np.argsort(-score)[:kk]
        true_idx = np.argsort(-actual)[:kk]
        pred_set = set(pred_idx.tolist())
        true_set = set(true_idx.tolist())
        hit = len(pred_set & true_set) / float(kk)

        pred_profit = float(np.mean(actual[pred_idx]))
        ndcgs.append(ndcg)
        hits.append(hit)
        topk_profit.append(pred_profit)

        year_rows[str(year)] = {
            "ndcg_at_k": float(ndcg),
            "hit_rate_at_k": float(hit),
            "topk_avg_profit": float(pred_profit),
            "n_candidates": int(len(g)),
        }

    if not ndcgs:
        return {"ndcg_at_k": 0.0, "hit_rate_at_k": 0.0, "topk_avg_profit": 0.0, "years": {}}

    return {
        "ndcg_at_k": float(np.mean(ndcgs)),
        "hit_rate_at_k": float(np.mean(hits)),
        "topk_avg_profit": float(np.mean(topk_profit)),
        "years": year_rows,
    }


def profit_mae(df: pd.DataFrame, true_col: str = "profit_true", pred_col: str = "profit_hat") -> float:
    metrics = regression_metrics(df[true_col], df[pred_col])
    return float(metrics["mae"])
