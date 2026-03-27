from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

from 训练流水线.评估.指标 import ndcg_at_k, regression_metrics
from 训练流水线.模型.基础模型 import create_model
from 训练流水线.模型.搜索 import random_search


@dataclass
class TaskTrainResult:
    val: pd.DataFrame
    test: pd.DataFrame
    infer: pd.DataFrame
    report: Dict[str, object]


def _resolve_transform_name(name: object) -> str:
    text = str(name or "none").strip().lower()
    return "log1p" if text == "log1p" else "none"


def _transform_target(y, transform_name: str) -> np.ndarray:
    arr = np.asarray(pd.to_numeric(y, errors="coerce"), dtype=float)
    if _resolve_transform_name(transform_name) == "log1p":
        arr = np.log1p(np.maximum(arr, 0.0))
    return arr


def _inverse_target(y, transform_name: str) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if _resolve_transform_name(transform_name) == "log1p":
        arr = np.expm1(arr)
        arr = np.maximum(arr, 0.0)
    return arr


def _target_transform_for_task(train_cfg: dict, task_name: str) -> str:
    by_task = train_cfg.get("target_transform_by_task", {})
    default_name = train_cfg.get("target_transform", "none")
    return _resolve_transform_name(by_task.get(task_name, default_name))


def _recency_alpha_for_task(train_cfg: dict, task_name: str) -> float:
    by_task = train_cfg.get("recency_weight_alpha_by_task", {})
    alpha = float(by_task.get(task_name, train_cfg.get("recency_weight_alpha", 0.0)))
    return float(np.clip(alpha, 0.0, 2.0))


def _objective_weights_for_task(train_cfg: dict, task_name: str) -> Dict[str, float]:
    default_weights = {"mae": 0.45, "rmse": 0.30, "mape": 0.15, "rank": 0.10}
    cfg = train_cfg.get("cv_objective_weights", {})
    by_task = train_cfg.get("cv_objective_weights_by_task", {})
    if isinstance(by_task, dict) and task_name in by_task:
        cfg = by_task.get(task_name, cfg)
    if not isinstance(cfg, dict):
        cfg = {}

    out = {}
    for k, v in default_weights.items():
        try:
            out[k] = max(0.0, float(cfg.get(k, v)))
        except Exception:
            out[k] = float(v)
    s = float(sum(out.values()))
    if s <= 1e-9:
        return dict(default_weights)
    return {k: float(v / s) for k, v in out.items()}


def _split_masks(df: pd.DataFrame, config: dict) -> Tuple[List[int], List[int], int]:
    time_cfg = config.get("time", {})
    val_years = [int(y) for y in time_cfg.get("val_years", [2017, 2018, 2019, 2020, 2021])]
    test_years = [int(y) for y in time_cfg.get("test_years", [2022, 2023, 2024])]
    infer_year = int(time_cfg.get("inference_year", 2025))
    return val_years, test_years, infer_year


def _folds(df: pd.DataFrame, train_start: int, val_years: List[int]) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    out = []
    for y in val_years:
        train_idx = df[(df["year"] >= train_start) & (df["year"] <= y - 1)].index.to_numpy()
        val_idx = df[df["year"] == y].index.to_numpy()
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        out.append((train_idx, val_idx, int(y)))
    return out


def _fit_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: np.ndarray,
    X_pred: pd.DataFrame,
    model_name: str,
    params: Dict[str, object],
    seed: int,
    target_transform: str = "none",
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    Xt = imputer.fit_transform(X_train)
    Xp = imputer.transform(X_pred)
    model = create_model(model_name, params, random_state=seed)
    yt = _transform_target(y_train, transform_name=target_transform)
    try:
        model.fit(Xt, yt, sample_weight=w_train)
    except TypeError:
        model.fit(Xt, yt)
    pred = model.predict(Xp)
    return _inverse_target(np.asarray(pred, dtype=float), transform_name=target_transform)


def _fit_model_bundle(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: np.ndarray,
    model_name: str,
    params: Dict[str, object],
    seed: int,
    target_transform: str = "none",
):
    imputer = SimpleImputer(strategy="median")
    Xt = imputer.fit_transform(X_train)
    model = create_model(model_name, params, random_state=seed)
    yt = _transform_target(y_train, transform_name=target_transform)
    try:
        model.fit(Xt, yt, sample_weight=w_train)
    except TypeError:
        model.fit(Xt, yt)
    return imputer, model, _resolve_transform_name(target_transform)


def _predict_bundle(bundle, X: pd.DataFrame) -> np.ndarray:
    if len(bundle) >= 3:
        imputer, model, target_transform = bundle
    else:
        imputer, model = bundle
        target_transform = "none"
    pred = np.asarray(model.predict(imputer.transform(X)), dtype=float)
    return _inverse_target(pred, transform_name=target_transform)


def _sample_weight(
    rows,
    real_w: float,
    years: Optional[pd.Series] = None,
    recency_alpha: float = 0.0,
) -> np.ndarray:
    w = np.full(len(rows), float(real_w), dtype=float)

    alpha = float(np.clip(recency_alpha, 0.0, 2.0))
    if years is None or alpha <= 1e-9:
        return w

    y = pd.to_numeric(years, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(y)
    if not finite.any():
        return w
    yv = y[finite]
    y_min = float(np.min(yv))
    y_max = float(np.max(yv))
    if y_max <= y_min + 1e-9:
        return w

    pos = (y - y_min) / (y_max - y_min)
    pos = np.where(np.isfinite(pos), pos, 0.5)
    return w * (1.0 + alpha * pos)


def _cv_objective(
    df: pd.DataFrame,
    feature_cols: List[str],
    folds: List[Tuple[np.ndarray, np.ndarray, int]],
    model_name: str,
    params: Dict[str, object],
    top_k: int,
    seed: int,
    real_w: float,
    target_transform: str,
    recency_alpha: float,
    objective_weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    maes = []
    rmses = []
    mapes = []
    ndcgs = []
    for train_idx, val_idx, year in folds:
        tr = df.loc[train_idx]
        va = df.loc[val_idx]
        if tr.empty or va.empty:
            continue
        X_train = tr[feature_cols]
        y_train = tr["target"]
        X_val = va[feature_cols]
        y_val = va["target"]
        w = _sample_weight(
            tr,
            real_w=real_w,
            years=tr.get("year"),
            recency_alpha=recency_alpha,
        )
        pred = _fit_predict(
            X_train,
            y_train,
            w,
            X_val,
            model_name,
            params,
            seed + int(year),
            target_transform=target_transform,
        )
        m = regression_metrics(y_val, pred)
        scale = max(1.0, float(np.mean(np.abs(pd.to_numeric(y_val, errors="coerce")))))
        maes.append(float(m["mae"]) / scale)
        rmses.append(float(m["rmse"]) / scale)
        mapes.append(float(m["mape"]))
        ndcgs.append(float(ndcg_at_k(np.asarray(y_val, dtype=float), np.asarray(pred, dtype=float), k=min(top_k, len(y_val)))))

    if not maes:
        return float("inf"), {"cv_mae_norm": float("inf"), "cv_rmse_norm": float("inf"), "cv_mape": float("inf"), "cv_ndcg": 0.0}

    mae_norm = float(np.mean(maes))
    rmse_norm = float(np.mean(rmses))
    mape = float(np.mean(mapes))
    ndcg = float(np.mean(ndcgs))
    w_mae = float(objective_weights.get("mae", 0.45))
    w_rmse = float(objective_weights.get("rmse", 0.30))
    w_mape = float(objective_weights.get("mape", 0.15))
    w_rank = float(objective_weights.get("rank", 0.10))
    score = w_mae * mae_norm + w_rmse * rmse_norm + w_mape * mape + w_rank * (1.0 - ndcg)
    return score, {"cv_mae_norm": mae_norm, "cv_rmse_norm": rmse_norm, "cv_mape": mape, "cv_ndcg": ndcg}


def _collect_oof_predictions(
    df: pd.DataFrame,
    feature_cols: List[str],
    folds: List[Tuple[np.ndarray, np.ndarray, int]],
    model_name: str,
    params: Dict[str, object],
    seed: int,
    real_w: float,
    target_transform: str,
    recency_alpha: float,
) -> pd.Series:
    pred = pd.Series(np.nan, index=df.index, dtype=float)
    for train_idx, val_idx, year in folds:
        tr = df.loc[train_idx]
        va = df.loc[val_idx]
        w = _sample_weight(
            tr,
            real_w=real_w,
            years=tr.get("year"),
            recency_alpha=recency_alpha,
        )
        p = _fit_predict(
            tr[feature_cols],
            tr["target"],
            w,
            va[feature_cols],
            model_name,
            params,
            seed + int(year),
            target_transform=target_transform,
        )
        pred.loc[val_idx] = p
    return pred


def _blend_weights(oof_mat: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = np.isfinite(oof_mat).all(axis=1) & np.isfinite(y)
    if mask.sum() < max(5, oof_mat.shape[1] * 2):
        return np.ones(oof_mat.shape[1], dtype=float) / float(oof_mat.shape[1])
    w, _ = nnls(oof_mat[mask], y[mask])
    if np.sum(w) <= 1e-12:
        w = np.ones(oof_mat.shape[1], dtype=float)
    return w / np.sum(w)


def _quantile_predictions(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    seed: int,
    real_w: float,
    recency_alpha: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr = df.loc[train_idx]
    pr = df.loc[pred_idx]
    if tr.empty or pr.empty:
        z = np.zeros(len(pr), dtype=float)
        return z, z, z

    X_train = tr[feature_cols]
    y_train = tr["target"]
    X_pred = pr[feature_cols]
    w = _sample_weight(
        tr,
        real_w=real_w,
        years=tr.get("year"),
        recency_alpha=recency_alpha,
    )

    params = {"n_estimators": 320, "max_depth": 3, "learning_rate": 0.05, "min_samples_leaf": 2}

    def fit_q(q: float) -> np.ndarray:
        imputer = SimpleImputer(strategy="median")
        Xt = imputer.fit_transform(X_train)
        Xp = imputer.transform(X_pred)
        model = create_model("gbr_quantile", params=params, random_state=seed + int(q * 1000), quantile=q)
        try:
            model.fit(Xt, y_train, sample_weight=w)
        except TypeError:
            model.fit(Xt, y_train)
        return np.asarray(model.predict(Xp), dtype=float)

    p10 = fit_q(0.10)
    p50 = fit_q(0.50)
    p90 = fit_q(0.90)

    lo = np.minimum(np.minimum(p10, p50), p90)
    hi = np.maximum(np.maximum(p10, p50), p90)
    mid = np.clip(p50, lo, hi)
    return lo, mid, hi


def _build_output(df: pd.DataFrame, idx: np.ndarray, pred: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> pd.DataFrame:
    out = df.loc[idx, ["crop", "year", "target", "env_prob"]].copy()
    out["pred"] = np.asarray(pred, dtype=float)
    out["p10"] = np.asarray(p10, dtype=float)
    out["p50"] = np.asarray(p50, dtype=float)
    out["p90"] = np.asarray(p90, dtype=float)
    return out.reset_index(drop=True)


def train_global_task(task_name: str, df: pd.DataFrame, config: dict, seed_offset: int = 0) -> TaskTrainResult:
    train_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    time_cfg = config.get("time", {})

    seed = int(train_cfg.get("random_seed", 42)) + int(seed_offset)
    hpo_trials_map = train_cfg.get("hpo_trials_by_task", {})
    hpo_trials = int(hpo_trials_map.get(task_name, train_cfg.get("hpo_trials", 12)))
    by_task = train_cfg.get("base_models_by_task", {})
    task_models = by_task.get(task_name, train_cfg.get("base_models", ["hgb", "rf", "etr"]))
    model_names = [str(x) for x in task_models]
    real_map = train_cfg.get("real_sample_weight_by_task", {})
    real_w = float(real_map.get(task_name, train_cfg.get("real_sample_weight", 1.0)))
    target_transform = _target_transform_for_task(train_cfg, task_name)
    recency_alpha = _recency_alpha_for_task(train_cfg, task_name)
    objective_weights = _objective_weights_for_task(train_cfg, task_name)
    top_k = int(eval_cfg.get("top_k", 5))

    train_start = int(time_cfg.get("train_start_year", 2010))
    val_years, test_years, infer_year = _split_masks(df, config)
    folds = _folds(df, train_start=train_start, val_years=val_years)
    if len(folds) < 3:
        raise ValueError(f"{task_name}: walk-forward folds < 3")

    feature_cols = [
        c
        for c in df.columns
        if c not in {"crop", "year", "target"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    search_rows = []
    best_params = {}
    oof_preds = {}
    best_scores = {}

    for i, model_name in enumerate(model_names):
        def objective(params):
            return _cv_objective(
                df=df,
                feature_cols=feature_cols,
                folds=folds,
                model_name=model_name,
                params=params,
                top_k=top_k,
                seed=seed + i * 17,
                real_w=real_w,
                target_transform=target_transform,
                recency_alpha=recency_alpha,
                objective_weights=objective_weights,
            )

        params, score, history = random_search(
            model_name=model_name,
            objective_fn=objective,
            trials=hpo_trials,
            seed=seed + 100 + i,
        )
        best_params[model_name] = params
        best_scores[model_name] = float(score)
        search_rows.extend(history)

        oof = _collect_oof_predictions(
            df=df,
            feature_cols=feature_cols,
            folds=folds,
            model_name=model_name,
            params=params,
            seed=seed + 200 + i,
            real_w=real_w,
            target_transform=target_transform,
            recency_alpha=recency_alpha,
        )
        oof_preds[model_name] = oof

    val_mask = df["year"].isin(val_years).to_numpy()
    val_idx = df.index[val_mask].to_numpy()

    oof_mat = np.vstack([oof_preds[m].loc[val_idx].to_numpy(dtype=float) for m in model_names]).T
    y_val = df.loc[val_idx, "target"].to_numpy(dtype=float)
    blend_w = _blend_weights(oof_mat, y_val)

    val_point = np.nansum(oof_mat * blend_w.reshape(1, -1), axis=1)

    max_val_year = int(max(val_years))
    eval_train_idx = df[(df["year"] >= train_start) & (df["year"] <= max_val_year)].index.to_numpy()
    test_idx = df[df["year"].isin(test_years)].index.to_numpy()
    infer_train_idx = df[(df["year"] >= train_start) & (df["year"] <= max(test_years))].index.to_numpy()
    infer_idx = df[df["year"] == infer_year].index.to_numpy()

    # Final models for test.
    eval_bundles = {}
    for i, model_name in enumerate(model_names):
        tr = df.loc[eval_train_idx]
        w = _sample_weight(
            tr,
            real_w=real_w,
            years=tr.get("year"),
            recency_alpha=recency_alpha,
        )
        eval_bundles[model_name] = _fit_model_bundle(
            tr[feature_cols],
            tr["target"],
            w,
            model_name,
            best_params[model_name],
            seed=seed + 300 + i,
            target_transform=target_transform,
        )

    test_preds_list = []
    for model_name in model_names:
        p = _predict_bundle(eval_bundles[model_name], df.loc[test_idx, feature_cols]) if len(test_idx) else np.asarray([], dtype=float)
        test_preds_list.append(p)
    test_mat = np.vstack(test_preds_list).T if test_preds_list else np.zeros((len(test_idx), len(model_names)))
    test_point = np.nansum(test_mat * blend_w.reshape(1, -1), axis=1) if len(test_idx) else np.asarray([], dtype=float)

    # Final models for inference year.
    infer_bundles = {}
    for i, model_name in enumerate(model_names):
        tr = df.loc[infer_train_idx]
        w = _sample_weight(
            tr,
            real_w=real_w,
            years=tr.get("year"),
            recency_alpha=recency_alpha,
        )
        infer_bundles[model_name] = _fit_model_bundle(
            tr[feature_cols],
            tr["target"],
            w,
            model_name,
            best_params[model_name],
            seed=seed + 400 + i,
            target_transform=target_transform,
        )

    infer_preds_list = []
    for model_name in model_names:
        p = _predict_bundle(infer_bundles[model_name], df.loc[infer_idx, feature_cols]) if len(infer_idx) else np.asarray([], dtype=float)
        infer_preds_list.append(p)
    infer_mat = np.vstack(infer_preds_list).T if infer_preds_list else np.zeros((len(infer_idx), len(model_names)))
    infer_point = np.nansum(infer_mat * blend_w.reshape(1, -1), axis=1) if len(infer_idx) else np.asarray([], dtype=float)

    # Quantile predictions.
    val_q10 = np.full(len(val_idx), np.nan, dtype=float)
    val_q50 = np.full(len(val_idx), np.nan, dtype=float)
    val_q90 = np.full(len(val_idx), np.nan, dtype=float)
    val_pos = {int(idx): i for i, idx in enumerate(val_idx.tolist())}
    for train_idx, fold_val_idx, year in folds:
        p10, p50, p90 = _quantile_predictions(
            df=df,
            feature_cols=feature_cols,
            train_idx=train_idx,
            pred_idx=fold_val_idx,
        seed=seed + int(year),
        real_w=real_w,
        recency_alpha=recency_alpha,
    )
        for j, ridx in enumerate(fold_val_idx.tolist()):
            pos = val_pos.get(int(ridx))
            if pos is None:
                continue
            val_q10[pos] = p10[j]
            val_q50[pos] = p50[j]
            val_q90[pos] = p90[j]

    if np.isnan(val_q50).any():
        val_q50 = np.where(np.isnan(val_q50), val_point, val_q50)
    val_q10 = np.where(np.isnan(val_q10), np.minimum(val_q50, val_point), val_q10)
    val_q90 = np.where(np.isnan(val_q90), np.maximum(val_q50, val_point), val_q90)

    test_q10, test_q50, test_q90 = _quantile_predictions(
        df=df,
        feature_cols=feature_cols,
        train_idx=eval_train_idx,
        pred_idx=test_idx,
        seed=seed + 500,
        real_w=real_w,
        recency_alpha=recency_alpha,
    )
    infer_q10, infer_q50, infer_q90 = _quantile_predictions(
        df=df,
        feature_cols=feature_cols,
        train_idx=infer_train_idx,
        pred_idx=infer_idx,
        seed=seed + 600,
        real_w=real_w,
        recency_alpha=recency_alpha,
    )

    # Blend median with stacked point for stability.
    val_mid = 0.5 * val_point + 0.5 * val_q50
    test_mid = 0.5 * test_point + 0.5 * test_q50
    infer_mid = 0.5 * infer_point + 0.5 * infer_q50

    val_lo = np.minimum(val_q10, val_mid)
    val_hi = np.maximum(val_q90, val_mid)
    test_lo = np.minimum(test_q10, test_mid)
    test_hi = np.maximum(test_q90, test_mid)
    infer_lo = np.minimum(infer_q10, infer_mid)
    infer_hi = np.maximum(infer_q90, infer_mid)

    out_val = _build_output(df, val_idx, val_point, val_lo, val_mid, val_hi)
    out_test = _build_output(df, test_idx, test_point, test_lo, test_mid, test_hi)
    out_infer = _build_output(df, infer_idx, infer_point, infer_lo, infer_mid, infer_hi)

    report = {
        "task": task_name,
        "n_rows": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "target_transform": target_transform,
        "recency_weight_alpha": float(recency_alpha),
        "cv_objective_weights": objective_weights,
        "hpo_best_params": best_params,
        "hpo_best_scores": best_scores,
        "stacking_weights": {model_names[i]: float(blend_w[i]) for i in range(len(model_names))},
        "search_history": search_rows,
    }

    return TaskTrainResult(val=out_val, test=out_test, infer=out_infer, report=report)


def _local_price_track(
    df: pd.DataFrame,
    feature_cols: List[str],
    config: dict,
    model_names: List[str],
    params_by_model: Dict[str, Dict[str, object]],
    seed: int,
) -> Dict[str, pd.DataFrame]:
    time_cfg = config.get("time", {})
    train_cfg = config.get("training", {})
    train_start = int(time_cfg.get("train_start_year", 2010))
    val_years = [int(y) for y in time_cfg.get("val_years", [2017, 2018, 2019, 2020, 2021])]
    test_years = [int(y) for y in time_cfg.get("test_years", [2022, 2023, 2024])]
    infer_year = int(time_cfg.get("inference_year", 2025))
    real_map = train_cfg.get("real_sample_weight_by_task", {})
    real_w = float(real_map.get("price", train_cfg.get("real_sample_weight", 1.0)))
    target_transform = _target_transform_for_task(train_cfg, "price")
    recency_alpha = _recency_alpha_for_task(train_cfg, "price")
    min_rows = int(train_cfg.get("min_crop_rows_for_local_price", 8))

    all_val = []
    all_test = []
    all_infer = []

    for crop, g in df.groupby("crop"):
        g = g.sort_values("year").copy()
        if len(g) < min_rows:
            continue

        val_parts = []
        for y in val_years:
            tr = g[(g["year"] >= train_start) & (g["year"] <= y - 1)]
            va = g[g["year"] == y]
            if tr.empty or va.empty:
                continue

            preds = []
            for i, m in enumerate(model_names):
                w = _sample_weight(
                    tr,
                    real_w=real_w,
                    years=tr.get("year"),
                    recency_alpha=recency_alpha,
                )
                p = _fit_predict(
                    tr[feature_cols],
                    tr["target"],
                    w,
                    va[feature_cols],
                    m,
                    params_by_model.get(m, {}),
                    seed + i + y,
                    target_transform=target_transform,
                )
                preds.append(p)
            mat = np.vstack(preds).T
            w_crop = _blend_weights(mat, va["target"].to_numpy(dtype=float))
            point = np.nansum(mat * w_crop.reshape(1, -1), axis=1)
            part = va[["crop", "year", "target", "env_prob"]].copy()
            part["pred_local"] = point
            val_parts.append(part)

        if val_parts:
            all_val.append(pd.concat(val_parts, ignore_index=True))

        # test
        tr_eval = g[(g["year"] >= train_start) & (g["year"] <= max(val_years))]
        te = g[g["year"].isin(test_years)]
        if not tr_eval.empty and not te.empty:
            preds = []
            for i, m in enumerate(model_names):
                w = _sample_weight(
                    tr_eval,
                    real_w=real_w,
                    years=tr_eval.get("year"),
                    recency_alpha=recency_alpha,
                )
                p = _fit_predict(
                    tr_eval[feature_cols],
                    tr_eval["target"],
                    w,
                    te[feature_cols],
                    m,
                    params_by_model.get(m, {}),
                    seed + 100 + i,
                    target_transform=target_transform,
                )
                preds.append(p)
            mat = np.vstack(preds).T
            w_crop = _blend_weights(mat, te["target"].to_numpy(dtype=float))
            point = np.nansum(mat * w_crop.reshape(1, -1), axis=1)
            part = te[["crop", "year", "target", "env_prob"]].copy()
            part["pred_local"] = point
            all_test.append(part)

        # infer
        tr_inf = g[(g["year"] >= train_start) & (g["year"] <= max(test_years))]
        inf = g[g["year"] == infer_year]
        if not tr_inf.empty and not inf.empty:
            preds = []
            for i, m in enumerate(model_names):
                w = _sample_weight(
                    tr_inf,
                    real_w=real_w,
                    years=tr_inf.get("year"),
                    recency_alpha=recency_alpha,
                )
                p = _fit_predict(
                    tr_inf[feature_cols],
                    tr_inf["target"],
                    w,
                    inf[feature_cols],
                    m,
                    params_by_model.get(m, {}),
                    seed + 200 + i,
                    target_transform=target_transform,
                )
                preds.append(p)
            mat = np.vstack(preds).T
            w_crop = _blend_weights(mat, inf["target"].to_numpy(dtype=float))
            point = np.nansum(mat * w_crop.reshape(1, -1), axis=1)
            part = inf[["crop", "year", "target", "env_prob"]].copy()
            part["pred_local"] = point
            all_infer.append(part)

    return {
        "val": pd.concat(all_val, ignore_index=True) if all_val else pd.DataFrame(),
        "test": pd.concat(all_test, ignore_index=True) if all_test else pd.DataFrame(),
        "infer": pd.concat(all_infer, ignore_index=True) if all_infer else pd.DataFrame(),
    }


def _price_trend_table(
    source_df: pd.DataFrame,
    train_start: int,
    cutoff_year: int,
    pred_years: List[int],
    method: str,
) -> pd.DataFrame:
    method = str(method).strip().lower()
    if method not in {"poly", "recent", "q75", "mix"}:
        method = "poly"

    base = source_df[["crop", "year", "target"]].copy()
    base = base[(base["year"] >= int(train_start)) & (base["year"] <= int(cutoff_year))]
    base = base.dropna(subset=["crop", "year", "target"]).sort_values(["crop", "year"])
    if base.empty:
        return pd.DataFrame(columns=["crop", "year", "pred_trend", "p10_trend", "p50_trend", "p90_trend"])

    rows = []
    for crop, g in base.groupby("crop"):
        yrs = g["year"].to_numpy(dtype=float)
        vals = np.maximum(g["target"].to_numpy(dtype=float), 1e-6)
        if len(vals) == 0:
            continue

        logv = np.log(vals)
        if len(vals) >= 4:
            coef = np.polyfit(yrs, logv, deg=2)
            pred_poly = lambda y: float(np.exp(np.polyval(coef, float(y))))
        elif len(vals) >= 2:
            coef = np.polyfit(yrs, logv, deg=1)
            pred_poly = lambda y: float(np.exp(np.polyval(coef, float(y))))
        else:
            pred_poly = lambda y: float(vals[-1])

        yoy = vals[1:] / vals[:-1] if len(vals) >= 2 else np.asarray([], dtype=float)
        recent = float(np.nanmean(yoy[-5:])) if len(yoy) else 1.03
        recent = float(np.clip(recent, 0.80, 1.40))
        q75 = float(np.nanquantile(yoy, 0.75)) if len(yoy) else recent
        q75 = float(np.clip(q75, 0.85, 1.60))
        last = float(vals[-1])
        last_year = int(yrs[-1])

        log_yoy = np.log(np.clip(yoy, 1e-6, None)) if len(yoy) else np.asarray([], dtype=float)
        sigma = float(np.nanstd(log_yoy))
        if not np.isfinite(sigma) or sigma < 1e-6:
            sigma = 0.10

        def p_recent(y: int) -> float:
            h = max(1, int(y) - last_year)
            return float(last * (recent ** h))

        def p_q75(y: int) -> float:
            h = max(1, int(y) - last_year)
            return float(last * (q75 ** h))

        for yy in pred_years:
            yint = int(yy)
            poly = pred_poly(yint)
            rc = p_recent(yint)
            q = p_q75(yint)
            if method == "poly":
                pred = poly
            elif method == "recent":
                pred = rc
            elif method == "q75":
                pred = q
            else:
                pred = 0.35 * poly + 0.35 * rc + 0.30 * q
            pred = float(np.clip(pred, 0.50 * last, 3.20 * last))
            p10 = float(max(0.0, pred * np.exp(-1.2816 * sigma)))
            p90 = float(pred * np.exp(1.2816 * sigma))
            rows.append(
                {
                    "crop": str(crop),
                    "year": yint,
                    "pred_trend": pred,
                    "p10_trend": min(p10, pred),
                    "p50_trend": pred,
                    "p90_trend": max(p90, pred),
                }
            )
    return pd.DataFrame(rows)


def _attach_price_trend_candidates(
    pred_df: pd.DataFrame,
    source_df: pd.DataFrame,
    train_start: int,
    dynamic_cutoff: bool,
    fixed_cutoff_year: int,
) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pred_df
    out = pred_df.copy()
    years = sorted([int(y) for y in out["year"].dropna().unique().tolist()])
    methods = ["poly", "recent", "q75", "mix"]

    def attach(base: pd.DataFrame, cutoff: int, pred_years: List[int]) -> pd.DataFrame:
        cur = base.copy()
        for m in methods:
            tab = _price_trend_table(source_df, train_start=train_start, cutoff_year=cutoff, pred_years=pred_years, method=m)
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
            sub = attach(sub, cutoff=y - 1, pred_years=[y])
            chunks.append(sub)
        return pd.concat(chunks, ignore_index=True) if chunks else out
    return attach(out, cutoff=fixed_cutoff_year, pred_years=years)


def _price_trend_blend(
    base_val: pd.DataFrame,
    base_test: pd.DataFrame,
    base_infer: pd.DataFrame,
    source_df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    time_cfg = config.get("time", {})
    train_start = int(time_cfg.get("train_start_year", 2010))
    val_years = [int(y) for y in time_cfg.get("val_years", [2017, 2018, 2019, 2020, 2021])]
    test_years = [int(y) for y in time_cfg.get("test_years", [2022, 2023, 2024])]

    val_aug = _attach_price_trend_candidates(
        base_val,
        source_df=source_df,
        train_start=train_start,
        dynamic_cutoff=True,
        fixed_cutoff_year=max(val_years) - 1,
    )
    test_aug = _attach_price_trend_candidates(
        base_test,
        source_df=source_df,
        train_start=train_start,
        dynamic_cutoff=False,
        fixed_cutoff_year=max(val_years),
    )
    infer_aug = _attach_price_trend_candidates(
        base_infer,
        source_df=source_df,
        train_start=train_start,
        dynamic_cutoff=False,
        fixed_cutoff_year=max(test_years),
    )

    methods = ["poly", "recent", "q75", "mix"]
    global_method = "none"
    crop_method = {}
    crop_weight = {}
    global_w = 0.0
    weight_grid = np.linspace(0.0, 0.90, 19)

    def _obj(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
        return 0.25 * mae_n + 0.40 * rmse_n + 0.35 * mape

    if val_aug is not None and not val_aug.empty:
        y_all = pd.to_numeric(val_aug["target"], errors="coerce").to_numpy(dtype=float)
        base_all = pd.to_numeric(val_aug["pred_fused"], errors="coerce").to_numpy(dtype=float)

        best_obj = _obj(y_all, base_all)
        global_method = "none"
        global_w = 0.0
        for m in methods:
            trend_all = pd.to_numeric(val_aug[f"pred_trend_{m}"], errors="coerce").to_numpy(dtype=float)
            trend_all = np.where(np.isfinite(trend_all), trend_all, base_all)
            for w in weight_grid:
                pred = (1.0 - float(w)) * base_all + float(w) * trend_all
                obj = _obj(y_all, pred)
                if obj < best_obj:
                    best_obj = obj
                    global_method = m
                    global_w = float(w)

        for crop, g in val_aug.groupby("crop"):
            y = pd.to_numeric(g["target"], errors="coerce").to_numpy(dtype=float)
            base = pd.to_numeric(g["pred_fused"], errors="coerce").to_numpy(dtype=float)
            c_best_obj = _obj(y, base)
            c_method = "none"
            c_w = 0.0
            for m in methods:
                trend = pd.to_numeric(g[f"pred_trend_{m}"], errors="coerce").to_numpy(dtype=float)
                trend = np.where(np.isfinite(trend), trend, base)
                for w in weight_grid:
                    pred = (1.0 - float(w)) * base + float(w) * trend
                    obj = _obj(y, pred)
                    if obj < c_best_obj:
                        c_best_obj = obj
                        c_method = m
                        c_w = float(w)

            n = int(len(g))
            shrink = float(n / (n + 6.0))
            crop_method[str(crop)] = c_method if c_method != "none" else global_method
            crop_weight[str(crop)] = float(np.clip(shrink * c_w + (1.0 - shrink) * global_w, 0.0, 0.90))

        def pick(df_in: pd.DataFrame) -> pd.DataFrame:
            out = df_in.copy()
            out["trend_method"] = out["crop"].astype(str).map(crop_method).fillna(global_method)
            out["pred_trend"] = np.nan
            out["p10_trend"] = np.nan
            out["p50_trend"] = np.nan
            out["p90_trend"] = np.nan
            for m in methods:
                mask = out["trend_method"] == m
                if not mask.any():
                    continue
                out.loc[mask, "pred_trend"] = pd.to_numeric(out.loc[mask, f"pred_trend_{m}"], errors="coerce")
                out.loc[mask, "p10_trend"] = pd.to_numeric(out.loc[mask, f"p10_trend_{m}"], errors="coerce")
                out.loc[mask, "p50_trend"] = pd.to_numeric(out.loc[mask, f"p50_trend_{m}"], errors="coerce")
                out.loc[mask, "p90_trend"] = pd.to_numeric(out.loc[mask, f"p90_trend_{m}"], errors="coerce")
            none_mask = out["trend_method"].astype(str) == "none"
            if none_mask.any():
                out.loc[none_mask, "pred_trend"] = pd.to_numeric(out.loc[none_mask, "pred_fused"], errors="coerce")
                out.loc[none_mask, "p10_trend"] = pd.to_numeric(out.loc[none_mask, "p10_fused"], errors="coerce")
                out.loc[none_mask, "p50_trend"] = pd.to_numeric(out.loc[none_mask, "p50_fused"], errors="coerce")
                out.loc[none_mask, "p90_trend"] = pd.to_numeric(out.loc[none_mask, "p90_fused"], errors="coerce")
            return out

        val_aug = pick(val_aug)
        test_aug = pick(test_aug)
        infer_aug = pick(infer_aug)
    else:
        # fallback
        for frame in [val_aug, test_aug, infer_aug]:
            if frame is None or frame.empty:
                continue
            frame["trend_method"] = "none"
            frame["pred_trend"] = pd.to_numeric(frame.get("pred_fused"), errors="coerce")
            frame["p10_trend"] = pd.to_numeric(frame.get("p10_fused"), errors="coerce")
            frame["p50_trend"] = pd.to_numeric(frame.get("p50_fused"), errors="coerce")
            frame["p90_trend"] = pd.to_numeric(frame.get("p90_fused"), errors="coerce")

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame
        out = frame.copy()
        out["w_trend"] = out["crop"].astype(str).map(crop_weight).fillna(global_w).astype(float)
        out["pred_trend"] = pd.to_numeric(out["pred_trend"], errors="coerce").fillna(pd.to_numeric(out["pred_fused"], errors="coerce"))
        out["p10_trend"] = pd.to_numeric(out["p10_trend"], errors="coerce").fillna(pd.to_numeric(out["p10_fused"], errors="coerce"))
        out["p50_trend"] = pd.to_numeric(out["p50_trend"], errors="coerce").fillna(pd.to_numeric(out["p50_fused"], errors="coerce"))
        out["p90_trend"] = pd.to_numeric(out["p90_trend"], errors="coerce").fillna(pd.to_numeric(out["p90_fused"], errors="coerce"))
        w = out["w_trend"].to_numpy(dtype=float)
        out["pred_final"] = (1.0 - w) * pd.to_numeric(out["pred_fused"], errors="coerce").to_numpy(dtype=float) + w * out["pred_trend"].to_numpy(dtype=float)
        out["p50_final"] = out["pred_final"]
        out["p10_final"] = (1.0 - w) * pd.to_numeric(out["p10_fused"], errors="coerce").to_numpy(dtype=float) + w * out["p10_trend"].to_numpy(dtype=float)
        out["p90_final"] = (1.0 - w) * pd.to_numeric(out["p90_fused"], errors="coerce").to_numpy(dtype=float) + w * out["p90_trend"].to_numpy(dtype=float)
        out["p10_final"] = np.minimum(out["p10_final"], out["p50_final"])
        out["p90_final"] = np.maximum(out["p90_final"], out["p50_final"])
        return out

    val_out = apply(val_aug)
    test_out = apply(test_aug)
    infer_out = apply(infer_aug)

    report = {
        "enabled": True,
        "global_method": global_method,
        "crop_method": crop_method,
        "global_weight": float(global_w),
        "crop_weight": crop_weight,
    }
    return val_out, test_out, infer_out, report


def _price_stability_candidates(
    pred_df: pd.DataFrame,
    source_df: pd.DataFrame,
    train_start: int,
    dynamic_cutoff: bool,
    fixed_cutoff_year: int,
) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pred_df

    out = pred_df.copy()
    src = source_df[["crop", "year", "target"]].copy()
    src["crop"] = src["crop"].astype(str)
    src = src[src["year"] >= int(train_start)].sort_values(["crop", "year"])

    rows = []
    for _, r in out.iterrows():
        crop = str(r.get("crop", ""))
        year = int(r.get("year"))
        cutoff = (year - 1) if dynamic_cutoff else int(fixed_cutoff_year)
        hist = src[(src["crop"] == crop) & (src["year"] <= cutoff)].sort_values("year")

        pred = float(r.get("pred_final", np.nan))
        p10 = float(r.get("p10_final", np.nan))
        p50 = float(r.get("p50_final", np.nan))
        p90 = float(r.get("p90_final", np.nan))

        if len(hist) >= 2 and np.isfinite(pred):
            vals = np.maximum(hist["target"].to_numpy(dtype=float), 1e-6)
            last = float(vals[-1])
            yoy = vals[1:] / vals[:-1]
            q_low = float(np.nanquantile(yoy, 0.15))
            q_high = float(np.nanquantile(yoy, 0.85))
            q_low = max(0.70, q_low)
            q_high = min(1.45, q_high)
            if q_high <= q_low:
                q_high = q_low + 0.05

            ratio = float(pred / max(last, 1e-6))
            ratio_clip = float(np.clip(ratio, q_low, q_high))
            pred_s = float(last * ratio_clip)

            w_lo = max(0.0, p50 - p10) if np.isfinite(p50) and np.isfinite(p10) else 0.08 * pred_s
            w_hi = max(0.0, p90 - p50) if np.isfinite(p90) and np.isfinite(p50) else 0.08 * pred_s
            p50_s = pred_s
            p10_s = max(0.0, p50_s - w_lo)
            p90_s = p50_s + w_hi
        else:
            pred_s = pred
            p10_s = p10
            p50_s = p50 if np.isfinite(p50) else pred
            p90_s = p90

        rows.append(
            {
                "crop": crop,
                "year": year,
                "pred_stable": pred_s,
                "p10_stable": p10_s,
                "p50_stable": p50_s,
                "p90_stable": p90_s,
            }
        )

    tab = pd.DataFrame(rows)
    return out.merge(tab, on=["crop", "year"], how="left")


def _price_stability_blend(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    source_df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    time_cfg = config.get("time", {})
    train_start = int(time_cfg.get("train_start_year", 2010))
    val_years = [int(y) for y in time_cfg.get("val_years", [2017, 2018, 2019, 2020, 2021])]
    test_years = [int(y) for y in time_cfg.get("test_years", [2022, 2023, 2024])]

    val_aug = _price_stability_candidates(
        val_df,
        source_df=source_df,
        train_start=train_start,
        dynamic_cutoff=True,
        fixed_cutoff_year=max(val_years) - 1,
    )
    test_aug = _price_stability_candidates(
        test_df,
        source_df=source_df,
        train_start=train_start,
        dynamic_cutoff=False,
        fixed_cutoff_year=max(val_years),
    )
    infer_aug = _price_stability_candidates(
        infer_df,
        source_df=source_df,
        train_start=train_start,
        dynamic_cutoff=False,
        fixed_cutoff_year=max(test_years),
    )

    w_grid = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]
    crop_w = {}
    global_w = 0.0

    def obj(y: np.ndarray, pred: np.ndarray) -> float:
        mask = np.isfinite(y) & np.isfinite(pred)
        y = y[mask]
        pred = pred[mask]
        if y.size == 0:
            return float("inf")
        scale = max(1.0, float(np.mean(np.abs(y))))
        mae_n = float(np.mean(np.abs(y - pred)) / scale)
        rmse_n = float(np.sqrt(np.mean((y - pred) ** 2)) / scale)
        mape = float(np.mean(np.abs(y - pred) / np.maximum(np.abs(y), 1e-6)))
        return 0.20 * mae_n + 0.60 * rmse_n + 0.20 * mape

    if val_aug is not None and not val_aug.empty:
        y_all = pd.to_numeric(val_aug["target"], errors="coerce").to_numpy(dtype=float)
        base_all = pd.to_numeric(val_aug["pred_final"], errors="coerce").to_numpy(dtype=float)
        stb_all = pd.to_numeric(val_aug["pred_stable"], errors="coerce").to_numpy(dtype=float)
        best_obj = float("inf")
        for w in w_grid:
            pred = (1.0 - float(w)) * base_all + float(w) * stb_all
            o = obj(y_all, pred)
            if o < best_obj:
                best_obj = o
                global_w = float(w)

        for crop, g in val_aug.groupby("crop"):
            y = pd.to_numeric(g["target"], errors="coerce").to_numpy(dtype=float)
            base = pd.to_numeric(g["pred_final"], errors="coerce").to_numpy(dtype=float)
            stb = pd.to_numeric(g["pred_stable"], errors="coerce").to_numpy(dtype=float)
            best_obj = float("inf")
            best_w = global_w
            for w in w_grid:
                pred = (1.0 - float(w)) * base + float(w) * stb
                o = obj(y, pred)
                if o < best_obj:
                    best_obj = o
                    best_w = float(w)
            n = int(len(g))
            shrink = float(n / (n + 5.0))
            crop_w[str(crop)] = float(np.clip(shrink * best_w + (1.0 - shrink) * global_w, 0.0, 0.9))

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame
        out = frame.copy()
        out["w_stable"] = out["crop"].astype(str).map(crop_w).fillna(global_w).astype(float)
        for c in ["pred_stable", "p10_stable", "p50_stable", "p90_stable"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out["pred_stable"] = out["pred_stable"].fillna(out["pred_final"])
        out["p10_stable"] = out["p10_stable"].fillna(out["p10_final"])
        out["p50_stable"] = out["p50_stable"].fillna(out["p50_final"])
        out["p90_stable"] = out["p90_stable"].fillna(out["p90_final"])

        w = out["w_stable"].to_numpy(dtype=float)
        out["pred_final"] = (1.0 - w) * pd.to_numeric(out["pred_final"], errors="coerce").to_numpy(dtype=float) + w * out["pred_stable"].to_numpy(dtype=float)
        out["p50_final"] = out["pred_final"]
        out["p10_final"] = (1.0 - w) * pd.to_numeric(out["p10_final"], errors="coerce").to_numpy(dtype=float) + w * out["p10_stable"].to_numpy(dtype=float)
        out["p90_final"] = (1.0 - w) * pd.to_numeric(out["p90_final"], errors="coerce").to_numpy(dtype=float) + w * out["p90_stable"].to_numpy(dtype=float)
        out["p10_final"] = np.minimum(out["p10_final"], out["p50_final"])
        out["p90_final"] = np.maximum(out["p90_final"], out["p50_final"])
        out = out.drop(columns=["pred_stable", "p10_stable", "p50_stable", "p90_stable", "w_stable"], errors="ignore")
        return out

    val_out = apply(val_aug)
    test_out = apply(test_aug)
    infer_out = apply(infer_aug)
    report = {"enabled": True, "global_weight": float(global_w), "crop_weight": crop_w}
    return val_out, test_out, infer_out, report


def _price_bias_calibration(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    infer_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if val_df is None or val_df.empty:
        report = {"enabled": False}
        return val_df, test_df, infer_df, report

    def obj(y: np.ndarray, pred: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        pred = np.asarray(pred, dtype=float)
        mask = np.isfinite(y) & np.isfinite(pred)
        y = y[mask]
        pred = pred[mask]
        if y.size == 0:
            return float("inf")
        scale = max(1.0, float(np.mean(np.abs(y))))
        mae_n = float(np.mean(np.abs(y - pred)) / scale)
        rmse_n = float(np.sqrt(np.mean((y - pred) ** 2)) / scale)
        mape = float(np.mean(np.abs(y - pred) / np.maximum(np.abs(y), 1e-6)))
        return 0.30 * mae_n + 0.45 * rmse_n + 0.25 * mape

    def build_ratio_map(df_hist: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        y = pd.to_numeric(df_hist["target"], errors="coerce").to_numpy(dtype=float)
        p = pd.to_numeric(df_hist["pred_final"], errors="coerce").to_numpy(dtype=float)
        r = y / np.maximum(np.abs(p), 1e-6)
        r = r[np.isfinite(r) & (r > 0.0)]
        if r.size == 0:
            return 1.0, {}

        global_ratio_local = float(np.nanmedian(r))
        global_ratio_local = float(np.clip(global_ratio_local, 0.85, 1.15))
        crop_ratio_local = {}
        for crop, g in df_hist.groupby("crop"):
            y_c = pd.to_numeric(g["target"], errors="coerce").to_numpy(dtype=float)
            p_c = pd.to_numeric(g["pred_final"], errors="coerce").to_numpy(dtype=float)
            r_c = y_c / np.maximum(np.abs(p_c), 1e-6)
            r_c = r_c[np.isfinite(r_c) & (r_c > 0.0)]
            if r_c.size == 0:
                crop_ratio_local[str(crop)] = global_ratio_local
                continue
            raw = float(np.nanmedian(r_c))
            raw = float(np.clip(raw, 0.80, 1.20))
            n = int(len(r_c))
            shrink = float(n / (n + 10.0))
            adj = float(np.clip(shrink * raw + (1.0 - shrink) * global_ratio_local, 0.85, 1.15))
            crop_ratio_local[str(crop)] = adj
        return global_ratio_local, crop_ratio_local

    gamma_grid = np.linspace(0.0, 1.0, 11).tolist()
    seq_true = []
    seq_pred = {float(g): [] for g in gamma_grid}
    years = sorted([int(y) for y in val_df["year"].dropna().unique().tolist()])
    for year in years:
        hist = val_df[val_df["year"] < int(year)]
        cur = val_df[val_df["year"] == int(year)]
        if hist.empty or cur.empty:
            continue
        global_ratio_hist, crop_ratio_hist = build_ratio_map(hist)
        y_cur = pd.to_numeric(cur["target"], errors="coerce").to_numpy(dtype=float)
        p_cur = pd.to_numeric(cur["pred_final"], errors="coerce").to_numpy(dtype=float)
        ratio_cur = cur["crop"].astype(str).map(crop_ratio_hist).fillna(global_ratio_hist).to_numpy(dtype=float)
        seq_true.append(y_cur)
        for gamma in gamma_grid:
            g = float(gamma)
            fac = 1.0 + g * (ratio_cur - 1.0)
            fac = np.clip(fac, 0.90, 1.10)
            seq_pred[g].append(p_cur * fac)

    best_gamma = 0.0
    if seq_true:
        y_seq = np.concatenate(seq_true, axis=0)
        best_obj = float("inf")
        for gamma in gamma_grid:
            g = float(gamma)
            pred_seq = np.concatenate(seq_pred[g], axis=0)
            o = obj(y_seq, pred_seq)
            if o < best_obj:
                best_obj = o
                best_gamma = g

    global_ratio, crop_ratio = build_ratio_map(val_df)
    if not crop_ratio and not np.isfinite(global_ratio):
        report = {"enabled": False}
        return val_df, test_df, infer_df, report

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame
        out = frame.copy()
        ratio = out["crop"].astype(str).map(crop_ratio).fillna(global_ratio).to_numpy(dtype=float)
        fac = 1.0 + float(best_gamma) * (ratio - 1.0)
        fac = np.clip(fac, 0.90, 1.10)
        for col in ["pred_final", "p10_final", "p50_final", "p90_final"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float) * fac
        out["p50_final"] = out["pred_final"]
        out["p10_final"] = np.minimum(out["p10_final"], out["p50_final"])
        out["p90_final"] = np.maximum(out["p90_final"], out["p50_final"])
        return out

    val_out = apply(val_df)
    test_out = apply(test_df)
    infer_out = apply(infer_df)
    report = {
        "enabled": True,
        "global_ratio": float(global_ratio),
        "gamma": float(best_gamma),
        "crop_ratio": crop_ratio,
    }
    return val_out, test_out, infer_out, report


def train_price_with_gate(df: pd.DataFrame, config: dict) -> TaskTrainResult:
    global_res = train_global_task("price", df, config, seed_offset=11)

    params_by_model = global_res.report.get("hpo_best_params", {})
    model_names = list(params_by_model.keys())
    if not model_names:
        model_names = [str(x) for x in config.get("training", {}).get("base_models", ["hgb", "rf", "etr"])]
    feature_cols = global_res.report.get("feature_cols", [])
    seed = int(config.get("training", {}).get("random_seed", 42)) + 501

    local = _local_price_track(
        df=df,
        feature_cols=feature_cols,
        config=config,
        model_names=model_names,
        params_by_model=params_by_model,
        seed=seed,
    )

    def merge_track(base: pd.DataFrame, loc: pd.DataFrame) -> pd.DataFrame:
        out = base.copy()
        if loc is None or loc.empty:
            out["pred_local"] = out["pred"]
            return out
        key = ["crop", "year"]
        out = out.merge(loc[["crop", "year", "pred_local"]], on=key, how="left")
        out["pred_local"] = out["pred_local"].fillna(out["pred"])
        return out

    val_df = merge_track(global_res.val, local.get("val"))
    test_df = merge_track(global_res.test, local.get("test"))
    infer_df = merge_track(global_res.infer, local.get("infer"))

    # Crop-level gate learned from validation errors.
    val_err_global = np.abs(pd.to_numeric(val_df["target"], errors="coerce") - pd.to_numeric(val_df["pred"], errors="coerce"))
    val_err_local = np.abs(pd.to_numeric(val_df["target"], errors="coerce") - pd.to_numeric(val_df["pred_local"], errors="coerce"))
    mae_g = float(np.nanmean(val_err_global))
    mae_l = float(np.nanmean(val_err_local))
    if not np.isfinite(mae_g):
        mae_g = 1.0
    if not np.isfinite(mae_l):
        mae_l = 1.0
    global_w = float(np.clip(mae_g / max(mae_g + mae_l, 1e-9), 0.05, 0.95))

    crop_gate = {}
    for crop, g in val_df.groupby("crop"):
        eg = np.abs(pd.to_numeric(g["target"], errors="coerce") - pd.to_numeric(g["pred"], errors="coerce"))
        el = np.abs(pd.to_numeric(g["target"], errors="coerce") - pd.to_numeric(g["pred_local"], errors="coerce"))
        mg = float(np.nanmean(eg))
        ml = float(np.nanmean(el))
        if not np.isfinite(mg):
            mg = mae_g
        if not np.isfinite(ml):
            ml = mae_l
        w_crop = float(np.clip(mg / max(mg + ml, 1e-9), 0.05, 0.95))
        n = int(len(g))
        shrink = float(n / (n + 3.0))
        w = float(np.clip(shrink * w_crop + (1.0 - shrink) * global_w, 0.05, 0.95))
        crop_gate[str(crop)] = w

    def attach_gate_weight(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["gate_w"] = out["crop"].astype(str).map(crop_gate).fillna(global_w).astype(float)
        return out

    val_df = attach_gate_weight(val_df)
    test_df = attach_gate_weight(test_df)
    infer_df = attach_gate_weight(infer_df)

    for frame in (val_df, test_df, infer_df):
        frame["pred_fused"] = frame["gate_w"] * frame["pred_local"] + (1.0 - frame["gate_w"]) * frame["pred"]
        spread = np.abs(frame["pred_local"] - frame["pred"])
        frame["p50_fused"] = frame["pred_fused"]
        frame["p10_fused"] = np.minimum(frame["p10"], frame["p50_fused"] - 1.2816 * spread)
        frame["p90_fused"] = np.maximum(frame["p90"], frame["p50_fused"] + 1.2816 * spread)

    val_df, test_df, infer_df, trend_report = _price_trend_blend(
        base_val=val_df,
        base_test=test_df,
        base_infer=infer_df,
        source_df=df,
        config=config,
    )
    val_df, test_df, infer_df, stability_report = _price_stability_blend(
        val_df=val_df,
        test_df=test_df,
        infer_df=infer_df,
        source_df=df,
        config=config,
    )
    val_df, test_df, infer_df, calibration_report = _price_bias_calibration(
        val_df=val_df,
        test_df=test_df,
        infer_df=infer_df,
    )

    out_val = val_df[["crop", "year", "target", "env_prob", "pred_final", "p10_final", "p50_final", "p90_final"]].copy()
    out_test = test_df[["crop", "year", "target", "env_prob", "pred_final", "p10_final", "p50_final", "p90_final"]].copy()
    out_infer = infer_df[["crop", "year", "target", "env_prob", "pred_final", "p10_final", "p50_final", "p90_final"]].copy()

    out_val = out_val.rename(columns={"pred_final": "pred", "p10_final": "p10", "p50_final": "p50", "p90_final": "p90"})
    out_test = out_test.rename(columns={"pred_final": "pred", "p10_final": "p10", "p50_final": "p50", "p90_final": "p90"})
    out_infer = out_infer.rename(columns={"pred_final": "pred", "p10_final": "p10", "p50_final": "p50", "p90_final": "p90"})

    report = dict(global_res.report)
    report["task"] = "price"
    report["gate"] = {
        "mode": "crop_mae_weighted",
        "global_weight_local": global_w,
        "val_mae_global": mae_g,
        "val_mae_local": mae_l,
        "crop_weight_local": crop_gate,
    }
    report["trend_blend"] = trend_report
    report["stability_blend"] = stability_report
    report["bias_calibration"] = calibration_report
    report["dual_track"] = True
    report["local_track_rows"] = {
        "val": int(len(local.get("val", pd.DataFrame()))),
        "test": int(len(local.get("test", pd.DataFrame()))),
        "infer": int(len(local.get("infer", pd.DataFrame()))),
    }

    return TaskTrainResult(val=out_val.reset_index(drop=True), test=out_test.reset_index(drop=True), infer=out_infer.reset_index(drop=True), report=report)


def add_crop_residual_correction(task_name: str, base_res: TaskTrainResult, df: pd.DataFrame, config: dict) -> TaskTrainResult:
    train_cfg = config.get("training", {})
    time_cfg = config.get("time", {})
    enabled_map = train_cfg.get("enable_residual_correction_by_task", {})
    enabled = bool(enabled_map.get(str(task_name), task_name in {"yield", "cost"}))
    if not enabled:
        report = dict(base_res.report)
        report["task"] = task_name
        report["residual_models"] = 0
        report["residual_correction"] = {
            "enabled": False,
            "reason": "disabled_by_config",
        }
        return TaskTrainResult(val=base_res.val, test=base_res.test, infer=base_res.infer, report=report)

    min_hist_years = int(train_cfg.get("min_crop_rows_for_residual", 2))
    val_years = sorted(int(y) for y in time_cfg.get("val_years", [2017, 2018, 2019, 2020, 2021]))

    val_df = base_res.val.copy()
    test_df = base_res.test.copy()
    infer_df = base_res.infer.copy()
    if val_df.empty:
        return base_res

    name = str(task_name or "").strip().lower()
    gamma_grid = np.linspace(0.0, 1.2, 13).tolist()
    shrink_grid = [0.5, 1.0, 2.0, 4.0, 8.0]

    def _objective(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
        if name == "yield":
            return 0.25 * mae_n + 0.25 * rmse_n + 0.50 * mape
        if name == "cost":
            return 0.20 * mae_n + 0.55 * rmse_n + 0.25 * mape
        return 0.35 * mae_n + 0.40 * rmse_n + 0.25 * mape

    def _build_offset_map(hist_df: pd.DataFrame, shrink_k: float):
        work = hist_df.copy()
        work["resid"] = pd.to_numeric(work["target"], errors="coerce") - pd.to_numeric(work["pred"], errors="coerce")
        work = work.dropna(subset=["crop", "resid"])
        if work.empty:
            return 0.0, {}

        global_offset = float(np.nanmedian(work["resid"].to_numpy(dtype=float)))
        crop_offsets = {}
        for crop, grp in work.groupby("crop"):
            vals = pd.to_numeric(grp["resid"], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size < int(max(1, min_hist_years)):
                continue
            local_offset = float(np.nanmedian(vals))
            shrink = float(vals.size / (vals.size + float(shrink_k)))
            crop_offsets[str(crop)] = shrink * local_offset + (1.0 - shrink) * global_offset
        return global_offset, crop_offsets

    def _apply_with_history(pred_df: pd.DataFrame, hist_df: pd.DataFrame, gamma: float, shrink_k: float) -> pd.DataFrame:
        out = pred_df.copy()
        if out.empty:
            return out
        global_offset, crop_offsets = _build_offset_map(hist_df, shrink_k)
        offset = out["crop"].astype(str).map(crop_offsets).fillna(global_offset).to_numpy(dtype=float)
        offset = float(gamma) * offset
        for col in ["pred", "p10", "p50", "p90"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float) + offset
        out["pred"] = np.clip(pd.to_numeric(out["pred"], errors="coerce").to_numpy(dtype=float), 0.0, None)
        out["p50"] = pd.to_numeric(out["pred"], errors="coerce")
        out["p10"] = np.minimum(pd.to_numeric(out["p10"], errors="coerce"), out["p50"])
        out["p90"] = np.maximum(pd.to_numeric(out["p90"], errors="coerce"), out["p50"])
        return out

    base_val_obj = _objective(
        pd.to_numeric(val_df["target"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(val_df["pred"], errors="coerce").to_numpy(dtype=float),
    )
    best = {
        "enabled": False,
        "gamma": 0.0,
        "shrink_k": None,
        "objective": float(base_val_obj),
        "base_objective": float(base_val_obj),
    }

    if len(val_years) >= 2:
        for shrink_k in shrink_grid:
            for gamma in gamma_grid:
                pred_parts = []
                true_parts = []
                for year in sorted(int(y) for y in val_df["year"].dropna().unique().tolist()):
                    cur = val_df[val_df["year"] == int(year)].copy()
                    hist = val_df[val_df["year"] < int(year)].copy()
                    if hist.empty:
                        adj = cur
                    else:
                        adj = _apply_with_history(cur, hist, gamma=gamma, shrink_k=shrink_k)
                    pred_parts.append(pd.to_numeric(adj["pred"], errors="coerce").to_numpy(dtype=float))
                    true_parts.append(pd.to_numeric(cur["target"], errors="coerce").to_numpy(dtype=float))
                score = _objective(np.concatenate(true_parts, axis=0), np.concatenate(pred_parts, axis=0))
                if score + 1e-9 < best["objective"]:
                    best.update(
                        {
                            "enabled": True,
                            "gamma": float(gamma),
                            "shrink_k": float(shrink_k),
                            "objective": float(score),
                        }
                    )

    def _apply_sequential_val(frame: pd.DataFrame, gamma: float, shrink_k: float) -> pd.DataFrame:
        if frame.empty or gamma <= 0.0 or shrink_k is None:
            return frame
        rows = []
        for year in sorted(int(y) for y in frame["year"].dropna().unique().tolist()):
            cur = frame[frame["year"] == int(year)].copy()
            hist = frame[frame["year"] < int(year)].copy()
            rows.append(_apply_with_history(cur, hist, gamma=gamma, shrink_k=shrink_k) if not hist.empty else cur)
        return pd.concat(rows, ignore_index=True) if rows else frame

    out_val = _apply_sequential_val(val_df, gamma=best["gamma"], shrink_k=best["shrink_k"])
    out_test = _apply_with_history(test_df, val_df, gamma=best["gamma"], shrink_k=best["shrink_k"]) if best["enabled"] else test_df
    infer_hist = pd.concat([out_val, out_test], ignore_index=True, sort=False) if best["enabled"] else val_df
    out_infer = _apply_with_history(infer_df, infer_hist, gamma=best["gamma"], shrink_k=best["shrink_k"]) if best["enabled"] else infer_df

    final_global_offset, final_crop_offsets = _build_offset_map(val_df, best["shrink_k"] if best["shrink_k"] is not None else 1.0)

    report = dict(base_res.report)
    report["task"] = task_name
    report["residual_models"] = int(len(final_crop_offsets))
    report["residual_correction"] = {
        "enabled": bool(best["enabled"]),
        "method": "crop_shrunk_median_offset",
        "gamma": float(best["gamma"]),
        "shrink_k": best["shrink_k"],
        "base_objective": float(best["base_objective"]),
        "selected_objective": float(best["objective"]),
        "global_offset": float(final_global_offset),
        "crop_offsets": final_crop_offsets,
        "history_source": "validation_only_for_test; validation_plus_test_for_infer",
        "min_history_rows_per_crop": int(min_hist_years),
    }

    return TaskTrainResult(
        val=out_val.reset_index(drop=True),
        test=out_test.reset_index(drop=True),
        infer=out_infer.reset_index(drop=True),
        report=report,
    )


def _task_bias_calibration(
    task_name: str,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    config: dict | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if val_df is None or val_df.empty:
        return val_df, test_df, infer_df, {"enabled": False}

    name = str(task_name).strip().lower()
    train_cfg = (config or {}).get("training", {}) if isinstance(config, dict) else {}
    bias_cfg_all = train_cfg.get("bias_calibration_by_task", {})
    bias_cfg = bias_cfg_all.get(name, {}) if isinstance(bias_cfg_all, dict) else {}
    ratio_bounds = {
        "price": (0.85, 1.15),
        "yield": (0.75, 1.25),
        "cost": (0.85, 1.18),
    }
    fac_bounds = {
        "price": (0.90, 1.10),
        "yield": (0.88, 1.12),
        "cost": (0.92, 1.08),
    }
    obj_w_map = {
        "price": (0.30, 0.45, 0.25),
        "yield": (0.40, 0.35, 0.25),
        "cost": (0.30, 0.50, 0.20),
    }

    def _resolve_bounds(raw, default):
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            try:
                lo = float(raw[0])
                hi = float(raw[1])
            except Exception:
                lo, hi = default
        else:
            lo, hi = default
        if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi <= lo:
            lo, hi = default
        return float(lo), float(hi)

    default_ratio_bounds = ratio_bounds.get(name, (0.85, 1.15))
    default_fac_bounds = fac_bounds.get(name, (0.90, 1.10))
    r_lo, r_hi = _resolve_bounds(bias_cfg.get("ratio_bounds"), default_ratio_bounds)
    f_lo, f_hi = _resolve_bounds(bias_cfg.get("factor_bounds"), default_fac_bounds)

    default_weights = obj_w_map.get(name, (0.35, 0.40, 0.25))
    weight_cfg = bias_cfg.get("objective_weights", {})
    if isinstance(weight_cfg, dict):
        w_mae = float(weight_cfg.get("mae", default_weights[0]))
        w_rmse = float(weight_cfg.get("rmse", default_weights[1]))
        w_mape = float(weight_cfg.get("mape", default_weights[2]))
    elif isinstance(weight_cfg, (list, tuple)) and len(weight_cfg) >= 3:
        w_mae = float(weight_cfg[0])
        w_rmse = float(weight_cfg[1])
        w_mape = float(weight_cfg[2])
    else:
        w_mae, w_rmse, w_mape = default_weights

    mode = str(bias_cfg.get("mode", "linear")).strip().lower()
    if mode not in {"linear", "power"}:
        mode = "linear"
    per_crop_gamma = bool(bias_cfg.get("per_crop_gamma", False))
    gamma_max = max(0.0, float(bias_cfg.get("gamma_max", 1.0)))
    gamma_step = max(0.01, float(bias_cfg.get("gamma_step", 0.1)))
    if gamma_max <= 1e-9:
        gamma_grid = [0.0]
    else:
        gamma_steps = max(1, int(round(gamma_max / gamma_step)))
        gamma_grid = sorted({float(x) for x in np.linspace(0.0, gamma_max, gamma_steps + 1).tolist()})
    ratio_shrink_k = max(0.0, float(bias_cfg.get("ratio_shrink_k", 8.0)))
    gamma_shrink_k = max(0.0, float(bias_cfg.get("gamma_shrink_k", 8.0)))
    low_target_threshold = bias_cfg.get("low_target_threshold")
    low_target_threshold = float(low_target_threshold) if low_target_threshold is not None else None

    def obj(y: np.ndarray, pred: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        pred = np.asarray(pred, dtype=float)
        mask = np.isfinite(y) & np.isfinite(pred)
        y = y[mask]
        pred = pred[mask]
        if y.size == 0:
            return float("inf")
        scale = max(1.0, float(np.mean(np.abs(y))))
        mae_n = float(np.mean(np.abs(y - pred)) / scale)
        rmse_n = float(np.sqrt(np.mean((y - pred) ** 2)) / scale)
        mape = float(np.mean(np.abs(y - pred) / np.maximum(np.abs(y), 1e-6)))
        return w_mae * mae_n + w_rmse * rmse_n + w_mape * mape

    def build_ratio_map(df_hist: pd.DataFrame) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        y = pd.to_numeric(df_hist["target"], errors="coerce").to_numpy(dtype=float)
        p = pd.to_numeric(df_hist["pred"], errors="coerce").to_numpy(dtype=float)
        ratio = y / np.maximum(np.abs(p), 1e-6)
        ratio = ratio[np.isfinite(ratio) & (ratio > 0.0)]
        if ratio.size == 0:
            return 1.0, {}, {}

        global_ratio = float(np.nanmedian(ratio))
        global_ratio = float(np.clip(global_ratio, r_lo, r_hi))
        crop_ratio = {}
        crop_target_median = {}
        for crop, g in df_hist.groupby("crop"):
            yc = pd.to_numeric(g["target"], errors="coerce").to_numpy(dtype=float)
            pc = pd.to_numeric(g["pred"], errors="coerce").to_numpy(dtype=float)
            rc = yc / np.maximum(np.abs(pc), 1e-6)
            rc = rc[np.isfinite(rc) & (rc > 0.0)]
            finite_y = yc[np.isfinite(yc)]
            crop_target_median[str(crop)] = float(np.nanmedian(finite_y)) if finite_y.size else float("nan")
            if rc.size == 0:
                crop_ratio[str(crop)] = global_ratio
                continue
            raw = float(np.nanmedian(rc))
            raw = float(np.clip(raw, r_lo * 0.95, r_hi * 1.05))
            n = int(rc.size)
            shrink = 1.0 if ratio_shrink_k <= 0.0 else float(n / (n + ratio_shrink_k))
            crop_ratio[str(crop)] = float(np.clip(shrink * raw + (1.0 - shrink) * global_ratio, r_lo, r_hi))
        return global_ratio, crop_ratio, crop_target_median

    def _factor_from_ratio(ratio: np.ndarray, gamma: np.ndarray | float) -> np.ndarray:
        ratio_arr = np.maximum(np.asarray(ratio, dtype=float), 1e-6)
        gamma_arr = np.asarray(gamma, dtype=float)
        if mode == "power":
            fac = np.power(ratio_arr, gamma_arr)
        else:
            fac = 1.0 + gamma_arr * (ratio_arr - 1.0)
        return np.clip(fac, f_lo, f_hi)

    def _active_mask(crops: pd.Series, crop_target_median: Dict[str, float]) -> np.ndarray:
        if low_target_threshold is None:
            return np.ones(len(crops), dtype=bool)
        med = crops.astype(str).map(crop_target_median)
        med = pd.to_numeric(med, errors="coerce")
        return (med <= float(low_target_threshold)).fillna(False).to_numpy(dtype=bool)

    seq_true = []
    seq_pred = {float(g): [] for g in gamma_grid}
    crop_seq_pred = {str(c): {float(g): [] for g in gamma_grid} for c in val_df["crop"].astype(str).unique()}
    crop_seq_true = {str(c): [] for c in val_df["crop"].astype(str).unique()}
    years = sorted([int(y) for y in val_df["year"].dropna().unique().tolist()])
    for year in years:
        hist = val_df[val_df["year"] < int(year)]
        cur = val_df[val_df["year"] == int(year)]
        if hist.empty or cur.empty:
            continue
        g_ratio, c_ratio, c_target_median = build_ratio_map(hist)
        y_cur = pd.to_numeric(cur["target"], errors="coerce").to_numpy(dtype=float)
        p_cur = pd.to_numeric(cur["pred"], errors="coerce").to_numpy(dtype=float)
        ratio_cur = cur["crop"].astype(str).map(c_ratio).fillna(g_ratio).to_numpy(dtype=float)
        active_cur = _active_mask(cur["crop"], c_target_median)
        seq_true.append(y_cur)
        for gamma in gamma_grid:
            fac = _factor_from_ratio(ratio_cur, float(gamma))
            fac = np.where(active_cur, fac, 1.0)
            seq_pred[float(gamma)].append(p_cur * fac)
        if per_crop_gamma:
            for crop, g in cur.groupby("crop"):
                crop_key = str(crop)
                crop_true = pd.to_numeric(g["target"], errors="coerce").to_numpy(dtype=float)
                crop_pred = pd.to_numeric(g["pred"], errors="coerce").to_numpy(dtype=float)
                crop_ratio = g["crop"].astype(str).map(c_ratio).fillna(g_ratio).to_numpy(dtype=float)
                crop_active = _active_mask(g["crop"], c_target_median)
                crop_seq_true[crop_key].append(crop_true)
                for gamma in gamma_grid:
                    fac = _factor_from_ratio(crop_ratio, float(gamma))
                    fac = np.where(crop_active, fac, 1.0)
                    crop_seq_pred[crop_key][float(gamma)].append(crop_pred * fac)

    best_gamma = 0.0
    if seq_true:
        y_seq = np.concatenate(seq_true, axis=0)
        best_obj = float("inf")
        for gamma in gamma_grid:
            pred_seq = np.concatenate(seq_pred[float(gamma)], axis=0)
            o = obj(y_seq, pred_seq)
            if o < best_obj:
                best_obj = o
                best_gamma = float(gamma)

    crop_gamma = {}
    if per_crop_gamma:
        for crop_key, pred_map in crop_seq_pred.items():
            true_parts = crop_seq_true.get(crop_key, [])
            if not true_parts:
                continue
            y_crop = np.concatenate(true_parts, axis=0)
            best_crop_gamma = float(best_gamma)
            best_crop_obj = float("inf")
            for gamma in gamma_grid:
                pred_crop = np.concatenate(pred_map[float(gamma)], axis=0)
                o = obj(y_crop, pred_crop)
                if o < best_crop_obj:
                    best_crop_obj = o
                    best_crop_gamma = float(gamma)
            n = int(y_crop.size)
            shrink = 1.0 if gamma_shrink_k <= 0.0 else float(n / (n + gamma_shrink_k))
            crop_gamma[crop_key] = float(shrink * best_crop_gamma + (1.0 - shrink) * best_gamma)

    global_ratio, crop_ratio, crop_target_median = build_ratio_map(val_df)

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame
        out = frame.copy()
        ratio = out["crop"].astype(str).map(crop_ratio).fillna(global_ratio).to_numpy(dtype=float)
        gamma = out["crop"].astype(str).map(crop_gamma).fillna(best_gamma).to_numpy(dtype=float)
        fac = _factor_from_ratio(ratio, gamma)
        fac = np.where(_active_mask(out["crop"], crop_target_median), fac, 1.0)
        for col in ["pred", "p10", "p50", "p90"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float) * fac
        out["p50"] = pd.to_numeric(out["pred"], errors="coerce")
        out["p10"] = np.minimum(out["p10"], out["p50"])
        out["p90"] = np.maximum(out["p90"], out["p50"])
        return out

    val_out = apply(val_df)
    test_out = apply(test_df)
    infer_out = apply(infer_df)
    report = {
        "enabled": True,
        "mode": mode,
        "per_crop_gamma": bool(per_crop_gamma),
        "global_ratio": float(global_ratio),
        "gamma": float(best_gamma),
        "global_gamma": float(best_gamma),
        "crop_gamma": crop_gamma,
        "crop_ratio": crop_ratio,
        "objective_weights": {"mae": float(w_mae), "rmse": float(w_rmse), "mape": float(w_mape)},
        "ratio_bounds": [float(r_lo), float(r_hi)],
        "factor_bounds": [float(f_lo), float(f_hi)],
        "ratio_shrink_k": float(ratio_shrink_k),
        "gamma_shrink_k": float(gamma_shrink_k),
        "low_target_threshold": low_target_threshold,
    }
    return val_out, test_out, infer_out, report


def apply_task_bias_calibration(task_name: str, task_res: TaskTrainResult, config: dict | None = None) -> TaskTrainResult:
    out_val, out_test, out_infer, calib_report = _task_bias_calibration(
        task_name=task_name,
        val_df=task_res.val,
        test_df=task_res.test,
        infer_df=task_res.infer,
        config=config,
    )
    report = dict(task_res.report)
    report["bias_calibration"] = calib_report
    report["task"] = str(task_name)
    return TaskTrainResult(val=out_val, test=out_test, infer=out_infer, report=report)
