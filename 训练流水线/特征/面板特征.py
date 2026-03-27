from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


TASKS = ["price", "yield", "cost"]
TARGET_COL = {"price": "price", "yield": "yield", "cost": "cost"}


def _year_features(df: pd.DataFrame, train_start_year: int) -> pd.DataFrame:
    out = df.copy()
    out["year_idx"] = out["year"].astype(float) - float(train_start_year)
    out["year_idx2"] = out["year_idx"] ** 2
    out["year_sin"] = np.sin(2.0 * np.pi * out["year_idx"] / 12.0)
    out["year_cos"] = np.cos(2.0 * np.pi * out["year_idx"] / 12.0)
    return out


def _add_lag_family(df: pd.DataFrame, value_col: str, prefix: str, lags: list, windows: list) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("crop")[value_col]
    for lag in lags:
        out[f"{prefix}_lag{lag}"] = g.shift(lag)
    shifted = g.shift(1)
    for w in windows:
        out[f"{prefix}_roll_mean_{w}"] = shifted.groupby(out["crop"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
        out[f"{prefix}_roll_std_{w}"] = shifted.groupby(out["crop"]).transform(lambda s: s.rolling(w, min_periods=1).std()).fillna(0.0)
    out[f"{prefix}_exp_mean"] = shifted.groupby(out["crop"]).transform(lambda s: s.expanding().mean())
    out[f"{prefix}_exp_std"] = shifted.groupby(out["crop"]).transform(lambda s: s.expanding().std()).fillna(0.0)

    if f"{prefix}_lag1" in out.columns and f"{prefix}_lag2" in out.columns:
        lag1 = pd.to_numeric(out[f"{prefix}_lag1"], errors="coerce")
        lag2 = pd.to_numeric(out[f"{prefix}_lag2"], errors="coerce")
        denom = np.maximum(np.abs(lag2.to_numpy(dtype=float)), 1e-6)
        out[f"{prefix}_diff_1_2"] = lag1 - lag2
        out[f"{prefix}_ratio_1_2"] = lag1.to_numpy(dtype=float) / denom
        out[f"{prefix}_yoy_1_2"] = out[f"{prefix}_ratio_1_2"] - 1.0

    if f"{prefix}_lag1" in out.columns and f"{prefix}_exp_mean" in out.columns and f"{prefix}_exp_std" in out.columns:
        lag1 = pd.to_numeric(out[f"{prefix}_lag1"], errors="coerce").to_numpy(dtype=float)
        exp_mean = pd.to_numeric(out[f"{prefix}_exp_mean"], errors="coerce").to_numpy(dtype=float)
        exp_std = pd.to_numeric(out[f"{prefix}_exp_std"], errors="coerce").to_numpy(dtype=float)
        z = (lag1 - exp_mean) / np.maximum(exp_std, 1e-6)
        out[f"{prefix}_zscore_lag1"] = np.clip(z, -8.0, 8.0)

    if windows:
        w_min = min(windows)
        w_max = max(windows)
        c_min = f"{prefix}_roll_mean_{w_min}"
        c_max = f"{prefix}_roll_mean_{w_max}"
        if c_min in out.columns and c_max in out.columns and w_min != w_max:
            out[f"{prefix}_roll_mean_gap_{w_min}_{w_max}"] = out[c_min] - out[c_max]
    return out


def _with_crop_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["crop"] = out["crop"].astype(str)
    out["crop_group"] = out["crop_group"].astype(str)
    crop_d = pd.get_dummies(out["crop"], prefix="crop", dtype=float)
    group_d = pd.get_dummies(out["crop_group"], prefix="group", dtype=float)
    return pd.concat([out, crop_d, group_d], axis=1)


def build_task_frames(
    panel: pd.DataFrame,
    config: dict,
    include_cross_task_lags: bool = True,
    include_crop_hierarchy: bool = True,
) -> Dict[str, pd.DataFrame]:
    feat_cfg = config.get("features", {})
    time_cfg = config.get("time", {})
    train_start = int(time_cfg.get("train_start_year", 2010))
    lags = list(feat_cfg.get("task_lags", [1, 2, 3]))
    windows = list(feat_cfg.get("rolling_windows", [3, 5]))

    df = panel.copy().sort_values(["crop", "year"]).reset_index(drop=True)
    df = _year_features(df, train_start_year=train_start)

    for task in TASKS:
        tcol = TARGET_COL[task]
        df = _add_lag_family(df, value_col=tcol, prefix=tcol, lags=lags, windows=windows)

    if include_crop_hierarchy:
        df = _with_crop_features(df)

    task_frames: Dict[str, pd.DataFrame] = {}

    for task in TASKS:
        tcol = TARGET_COL[task]
        base_cols = ["crop", "year", tcol, "env_prob"]
        feature_cols = ["year_idx", "year_idx2", "year_sin", "year_cos", "env_prob"]

        if include_crop_hierarchy:
            feature_cols.extend([c for c in df.columns if c.startswith("crop_") or c.startswith("group_")])

        own_prefix = tcol + "_"

        if include_cross_task_lags:
            lag_cols = [
                c
                for c in df.columns
                if c.endswith("_exp_mean")
                or c.endswith("_exp_std")
                or "_roll_mean_" in c
                or "_roll_std_" in c
                or "_lag" in c
                or "_diff_" in c
                or "_ratio_" in c
                or "_yoy_" in c
                or "_zscore_" in c
            ]
            lag_cols = [
                c
                for c in lag_cols
                if c.startswith("price_")
                or c.startswith("yield_")
                or c.startswith("cost_")
            ]
            feature_cols.extend(lag_cols)
        else:
            lag_cols = [c for c in df.columns if c.startswith(own_prefix)]
            feature_cols.extend(lag_cols)

        feature_cols = sorted(set([c for c in feature_cols if c in df.columns]))

        ordered_cols = base_cols + [c for c in feature_cols if c not in base_cols]
        sub = df[ordered_cols].copy()
        sub = sub.rename(columns={tcol: "target"})
        sub = sub[sub["year"] >= train_start].copy()
        sub = sub.dropna(subset=["target"]).reset_index(drop=True)
        task_frames[task] = sub

    return task_frames
