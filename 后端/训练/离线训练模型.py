from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List
import os
import sys

from joblib import dump
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from 后端.数据加载 import (
    load_config,
    load_cost_data,
    load_name_map,
    load_price_series,
    load_yield_history,
    resolve_names,
)
from 后端.模型产物 import (
    expected_cost_model_path,
    expected_price_model_path,
    expected_price_recursive_model_path,
    expected_yield_model_path,
    price_direct_artifact_tag,
    price_recursive_artifact_tag,
)
from 后端.模型.价格模型 import train_one_crop as train_price
from 后端.模型.成本模型 import (
    crop_group_from_cost_name,
    make_panel_lite_serving_model,
    train_one_crop as train_cost,
    train_panel_model as train_cost_panel,
)
from 后端.模型.产量模型 import train_yield_model
from 后端.模型.概率校准器 import save_calibrator, train_calibrator
from 后端.时间策略 import resolve_price_window_from_df, resolve_target_year, resolve_year_window_from_series


from 后端.训练.环境模型训练 import build_bundle as train_env_bundle


logger = logging.getLogger("offline_train")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _save_meta(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _synonym_cost_name(crop: str) -> str | None:
    return {"lentil": "Masur", "rice": "Paddy", "chickpea": "Gram"}.get(crop)


def _safe_last_year(series) -> int | None:
    try:
        cleaned = series.dropna()
        if len(cleaned) == 0:
            return None
        return int(cleaned.max())
    except Exception:
        return None


def _aggregate_price_yearly(price_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(price_df, pd.DataFrame) or price_df.empty or "date" not in price_df.columns:
        return pd.DataFrame(columns=["year", "price", "price_year_std", "price_obs_days"])
    price_col = "modal_price" if "modal_price" in price_df.columns else "price"
    work = price_df[["date", price_col]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work = work.dropna(subset=["date", price_col])
    if work.empty:
        return pd.DataFrame(columns=["year", "price", "price_year_std", "price_obs_days"])
    work["year"] = work["date"].dt.year.astype(int)
    out = work.groupby("year", as_index=False).agg(
        price=(price_col, "mean"),
        price_year_std=(price_col, "std"),
        price_obs_days=(price_col, "count"),
    )
    out["price_year_std"] = pd.to_numeric(out["price_year_std"], errors="coerce").fillna(0.0)
    out["price_obs_days"] = pd.to_numeric(out["price_obs_days"], errors="coerce").fillna(0.0)
    return out


def _prepare_yield_yearly(yield_history_all: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(yield_history_all, pd.DataFrame) or yield_history_all.empty:
        return pd.DataFrame(columns=["crop", "year", "yield"])
    crop_col = "crop_name" if "crop_name" in yield_history_all.columns else "crop"
    target_col = "yield_quintal_per_hectare" if "yield_quintal_per_hectare" in yield_history_all.columns else "yield"
    if crop_col not in yield_history_all.columns or "year" not in yield_history_all.columns or target_col not in yield_history_all.columns:
        return pd.DataFrame(columns=["crop", "year", "yield"])
    work = yield_history_all[[crop_col, "year", target_col]].copy()
    work["crop"] = work[crop_col].astype(str).str.strip().str.lower()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["yield"] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=["crop", "year", "yield"])
    if work.empty:
        return pd.DataFrame(columns=["crop", "year", "yield"])
    out = work.groupby(["crop", "year"], as_index=False).agg(
        yield_value=("yield", "mean"),
    )
    out = out.rename(columns={"yield_value": "yield"})
    out["year"] = out["year"].astype(int)
    return out


def _build_cost_panel_frame(
    *,
    mapping: Dict[str, Dict[str, str]],
    paths: dict,
    cost_all: pd.DataFrame,
    yield_history_all: pd.DataFrame,
) -> pd.DataFrame:
    yield_yearly = _prepare_yield_yearly(yield_history_all)
    rows: List[pd.DataFrame] = []

    for crop, m in sorted(mapping.items(), key=lambda x: x[0]):
        cost_name = str(m.get("cost_name", "")).strip()
        if not cost_name:
            continue

        cost_df = cost_all[cost_all["crop_name"].str.lower() == cost_name.lower()].copy()
        if cost_df.empty:
            alt = _synonym_cost_name(crop)
            if alt:
                cost_df = cost_all[cost_all["crop_name"].str.lower() == alt.lower()].copy()
        if cost_df.empty:
            continue

        cost_rows = cost_df.copy()
        cost_rows["year"] = pd.to_numeric(cost_rows["year_start"], errors="coerce")
        cost_rows["cost"] = pd.to_numeric(cost_rows["india_cost_wavg_sample"], errors="coerce")
        for col in ["n_states", "n_rows_cost", "sum_sample_weight"]:
            if col not in cost_rows.columns:
                cost_rows[col] = pd.NA
            cost_rows[col] = pd.to_numeric(cost_rows[col], errors="coerce")
        cost_rows = (
            cost_rows.groupby("year", as_index=False)
            .agg(
                cost=("cost", "mean"),
                n_states=("n_states", "mean"),
                n_rows_cost=("n_rows_cost", "mean"),
                sum_sample_weight=("sum_sample_weight", "mean"),
            )
            .dropna(subset=["year"])
        )
        if cost_rows.empty:
            continue
        cost_rows["year"] = cost_rows["year"].astype(int)

        price_yearly = pd.DataFrame(columns=["year", "price", "price_year_std", "price_obs_days"])
        price_file = str(m.get("price_file", "")).strip()
        if price_file:
            try:
                price_df = load_price_series(paths["price_dir"], price_file)
                price_yearly = _aggregate_price_yearly(price_df)
            except Exception:
                logger.exception("failed loading yearly price history for crop=%s price_file=%s", crop, price_file)

        yield_rows = yield_yearly[yield_yearly["crop"] == str(crop).strip().lower()].copy()

        year_pool = set(cost_rows["year"].tolist())
        year_pool.update(pd.to_numeric(price_yearly.get("year"), errors="coerce").dropna().astype(int).tolist())
        year_pool.update(pd.to_numeric(yield_rows.get("year"), errors="coerce").dropna().astype(int).tolist())
        if not year_pool:
            continue

        base = pd.DataFrame(
            {
                "crop": [str(crop).strip().lower()] * len(year_pool),
                "year": sorted(year_pool),
                "cost_name": [cost_name] * len(year_pool),
                "crop_group": [crop_group_from_cost_name(cost_name)] * len(year_pool),
            }
        )
        base = base.merge(cost_rows, on="year", how="left")
        base = base.merge(price_yearly, on="year", how="left")
        base = base.merge(yield_rows[["year", "yield"]], on="year", how="left")
        rows.append(base)

    if not rows:
        return pd.DataFrame(
            columns=[
                "crop",
                "year",
                "cost_name",
                "crop_group",
                "cost",
                "n_states",
                "n_rows_cost",
                "sum_sample_weight",
                "price",
                "price_year_std",
                "price_obs_days",
                "yield",
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["crop", "year"]).reset_index(drop=True)
    return out


def _normalize_price_prediction_mode(price_cfg: dict) -> str:
    mode = str((price_cfg or {}).get("prediction_mode", "return_recursive_v3")).strip().lower()
    if mode not in {"return_recursive_v3", "return_recursive_v2", "price_recursive_v1", "direct_horizon_v1"}:
        return "return_recursive_v3"
    return mode


def _build_recursive_step_cfg(price_cfg: dict) -> dict:
    cfg = dict(price_cfg or {})
    mode = _normalize_price_prediction_mode(cfg)
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


def train_all(config_path: str, train_calibrator_flag: bool = False) -> dict:
    if not logging.getLogger().handlers:
        _setup_logging()

    config = load_config(config_path)
    serving_cfg = config.get("serving", {})
    version = str(serving_cfg.get("model_cache_version", "v2"))

    paths = config["paths"]
    model_dir = Path(config["output"]["out_dir"]) / "模型"
    model_dir.mkdir(parents=True, exist_ok=True)

    mapping = resolve_names(load_name_map(paths["name_map"]))
    cost_all = load_cost_data(paths["cost_file"])
    yield_history_all = load_yield_history(paths.get("yield_history", ""))
    time_cfg = config.get("time", {})
    alignment_cfg = config.get("alignment", {})
    target_year = resolve_target_year(
        config,
        cost_df=cost_all,
        yield_df=yield_history_all,
        fallback_year=time.localtime().tm_year,
    )
    crop_alignment: Dict[str, Dict[str, int | None]] = {
        crop: {"price_last_year": None, "cost_last_year": None, "yield_last_year": None}
        for crop in mapping.keys()
    }

    if not yield_history_all.empty and "crop_name" in yield_history_all.columns and "year" in yield_history_all.columns:
        ymap = yield_history_all[["crop_name", "year"]].copy()
        ymap["crop_name"] = ymap["crop_name"].astype(str).str.strip().str.lower()
        ymap["year"] = ymap["year"].astype(float)
        for crop, grp in ymap.groupby("crop_name"):
            if crop in crop_alignment:
                crop_alignment[crop]["yield_last_year"] = _safe_last_year(grp["year"])

    lags = config["time"]["price_lags"]
    windows = config["time"]["price_roll_windows"]
    strict_cutoff_split = bool(time_cfg.get("strict_cutoff_split", True))

    report: Dict[str, List[dict]] = {
        "env": [],
        "price": [],
        "price_recursive": [],
        "cost": [],
        "yield": [],
        "calibrator": [],
    }

    # Train environment recommendation bundle.
    t0 = time.perf_counter()
    try:
        logger.info("training env model bundle ...")
        env_res = train_env_bundle(config_path)
        report["env"].append(
            {
                "ok": True,
                "bundle_path": env_res.get("bundle_path"),
                "train_file": env_res.get("train_file"),
                "metrics": env_res.get("metrics"),
                "holdout_metrics": env_res.get("holdout_metrics"),
                "cv_metrics": env_res.get("cv_metrics"),
                "best_params": env_res.get("best_params"),
                "selection": env_res.get("selection"),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }
        )
        logger.info("trained env model bundle in %.2f ms", (time.perf_counter() - t0) * 1000.0)
    except Exception as exc:
        report["env"].append({"ok": False, "error": str(exc)})
        logger.exception("failed training env model bundle")

    # Train price models per mapped crop.
    for crop, m in sorted(mapping.items(), key=lambda x: x[0]):
        price_file = m.get("price_file", "")
        if not price_file:
            continue
        t0 = time.perf_counter()
        try:
            logger.info("training price model for crop=%s", crop)
            model_path, meta_path = expected_price_model_path(model_dir, crop, config["model"]["price"], version)
            train_recursive_step = bool(config.get("model", {}).get("price", {}).get("train_recursive_step_model", True))
            step_model_path = step_meta_path = None
            if train_recursive_step:
                step_model_path, step_meta_path = expected_price_recursive_model_path(
                    model_dir,
                    crop,
                    config["model"]["price"],
                    version,
                )
            direct_ready = model_path.exists() and meta_path.exists()
            step_ready = (not train_recursive_step) or ((step_model_path is not None) and step_model_path.exists() and (step_meta_path is not None) and step_meta_path.exists())
            if direct_ready and step_ready:
                report["price"].append(
                    {
                        "crop": crop,
                        "price_file": price_file,
                        "ok": True,
                        "skipped": True,
                        "reason": "artifacts_already_exist",
                        "model_path": model_path.as_posix(),
                        "elapsed_ms": 0.0,
                    }
                )
                if train_recursive_step and step_model_path is not None:
                    report["price_recursive"].append(
                        {
                            "crop": crop,
                            "price_file": price_file,
                            "ok": True,
                            "skipped": True,
                            "reason": "artifacts_already_exist",
                            "model_path": step_model_path.as_posix(),
                            "elapsed_ms": 0.0,
                        }
                    )
                logger.info("skipping price model for crop=%s: artifacts already exist", crop)
                continue
            df = load_price_series(paths["price_dir"], price_file)
            if "date" in df.columns:
                crop_alignment.setdefault(crop, {})
                crop_alignment[crop]["price_last_year"] = _safe_last_year(
                    pd.to_datetime(df["date"], errors="coerce").dt.year
                )
            price_window = resolve_price_window_from_df(df, time_cfg=time_cfg)
            horizon = int(price_window["price_horizon_days"])
            backtest_days = int(time_cfg.get("price_backtest_days", 180))
            validation_cutoff = str(price_window["train_validation_cutoff_date"]).strip()
            res = train_price(
                df,
                config["model"]["price"],
                lags,
                windows,
                horizon,
                backtest_days,
                test_ratio=config["model"]["price"].get("test_ratio"),
                validation_cutoff=validation_cutoff,
                strict_cutoff_split=strict_cutoff_split,
                verbose=config["model"]["price"].get("verbose", False),
                label=crop,
            )
            model_path, meta_path = expected_price_model_path(model_dir, crop, config["model"]["price"], version)
            dump(res.model, model_path)
            _save_meta(
                meta_path,
                {
                    "metrics": res.metrics,
                    "feature_cols": res.feature_cols,
                    "artifact_role": "price_direct_main",
                    "artifact_name_tag": price_direct_artifact_tag(config["model"]["price"]),
                    "model_architecture": str(
                        ((res.artifacts or {}).get("model_family"))
                        or (res.metrics or {}).get("model_family")
                        or "legacy_price_model"
                    ),
                    "training_horizon_days": int(horizon),
                    "forecast_mode": "direct_horizon",
                    "target_mode": "price",
                    "feature_space": "price",
                    "artifacts": res.artifacts or {},
                },
            )
            report["price"].append(
                {
                    "crop": crop,
                    "price_file": price_file,
                    "ok": True,
                    "model_path": model_path.as_posix(),
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                    "prediction_mode": "direct_horizon_v1",
                    "prediction_window": {
                        "start_date": str(price_window.get("start_date").strftime("%Y-%m-%d")),
                        "end_date": str(price_window.get("end_date").strftime("%Y-%m-%d")),
                        "price_horizon_days": int(price_window.get("price_horizon_days", horizon)),
                    },
                    "train_validation_cutoff_date": validation_cutoff,
                }
            )

            if bool(config.get("model", {}).get("price", {}).get("train_recursive_step_model", True)):
                t_step = time.perf_counter()
                step_horizon = int(config.get("model", {}).get("price", {}).get("recursive_step_days", 1) or 1)
                step_horizon = max(1, step_horizon)
                step_cfg = _build_recursive_step_cfg(config["model"]["price"])
                step_prediction_mode = _normalize_price_prediction_mode(config["model"]["price"])
                try:
                    res_step = train_price(
                        df,
                        step_cfg,
                        lags,
                        windows,
                        step_horizon,
                        backtest_days,
                        test_ratio=step_cfg.get("test_ratio", config["model"]["price"].get("test_ratio")),
                        validation_cutoff=validation_cutoff,
                        strict_cutoff_split=strict_cutoff_split,
                        verbose=False,
                        label=f"{crop}:step{step_horizon}:{step_prediction_mode}",
                    )
                    step_model_path, step_meta_path = expected_price_recursive_model_path(
                        model_dir,
                        crop,
                        config["model"]["price"],
                        version,
                    )
                    dump(res_step.model, step_model_path)
                    _save_meta(
                        step_meta_path,
                        {
                            "metrics": res_step.metrics,
                            "feature_cols": res_step.feature_cols,
                            "artifact_role": "price_recursive_step",
                            "artifact_name_tag": price_recursive_artifact_tag(config["model"]["price"]),
                            "model_architecture": str(step_prediction_mode),
                            "training_horizon_days": int(step_horizon),
                            "forecast_mode": "recursive_step",
                            "prediction_mode": step_prediction_mode,
                            "target_mode": step_cfg.get("target_mode"),
                            "feature_space": step_cfg.get("feature_space"),
                            "time_raw_mode": step_cfg.get("time_raw_mode"),
                            "include_raw_time_features": bool(step_cfg.get("include_raw_time_features", False)),
                            "artifacts": res_step.artifacts or {},
                        },
                    )
                    report["price_recursive"].append(
                        {
                            "crop": crop,
                            "price_file": price_file,
                            "ok": True,
                            "model_path": step_model_path.as_posix(),
                            "elapsed_ms": round((time.perf_counter() - t_step) * 1000.0, 2),
                            "training_horizon_days": int(step_horizon),
                            "prediction_mode": step_prediction_mode,
                        }
                    )
                except Exception as step_exc:
                    report["price_recursive"].append(
                        {"crop": crop, "price_file": price_file, "ok": False, "error": str(step_exc)}
                    )
                    logger.exception("failed training recursive step price model for crop=%s", crop)
            logger.info("trained price model for crop=%s in %.2f ms", crop, (time.perf_counter() - t0) * 1000.0)
        except Exception as exc:
            report["price"].append({"crop": crop, "price_file": price_file, "ok": False, "error": str(exc)})
            logger.exception("failed training price model for crop=%s", crop)

    # Train cost models.
    cost_cfg = config["model"]["cost"]
    cost_feature_set = str(cost_cfg.get("feature_set", "legacy")).strip().lower()
    if cost_feature_set == "panel_lite":
        t0 = time.perf_counter()
        try:
            logger.info("training shared panel-lite cost model ...")
            panel_df = _build_cost_panel_frame(
                mapping=mapping,
                paths=paths,
                cost_all=cost_all,
                yield_history_all=yield_history_all,
            )
            if panel_df.empty:
                raise ValueError("cost panel frame is empty")
            cost_window = resolve_year_window_from_series(
                pd.to_numeric(panel_df["year"], errors="coerce").dropna().tolist(),
                time_cfg=time_cfg,
                window_years_key="cost_prediction_window_years",
                preferred_end_year=target_year,
            )
            validation_cutoff_cost = str(cost_window.get("train_validation_cutoff_date") or "2020-12-31")
            shared_res = train_cost_panel(
                panel_df,
                cost_cfg,
                test_ratio=cost_cfg.get("test_ratio"),
                validation_cutoff=validation_cutoff_cost,
                strict_cutoff_split=strict_cutoff_split,
                verbose=cost_cfg.get("verbose", False),
                label="shared_panel",
            )

            crops_with_panel = set(panel_df["crop"].astype(str).str.strip().str.lower().unique().tolist())
            for crop, m in sorted(mapping.items(), key=lambda x: x[0]):
                cost_name = str(m.get("cost_name", "")).strip()
                if not cost_name:
                    continue
                if crop not in crops_with_panel:
                    report["cost"].append(
                        {"crop": crop, "cost_name": cost_name, "ok": False, "error": "crop_missing_from_cost_panel"}
                    )
                    continue
                crop_df = panel_df[panel_df["crop"] == str(crop).strip().lower()].copy()
                crop_alignment.setdefault(crop, {})
                crop_alignment[crop]["cost_last_year"] = _safe_last_year(
                    pd.to_numeric(crop_df.loc[crop_df["cost"].notna(), "year"], errors="coerce")
                )
                serving_model = make_panel_lite_serving_model(
                    shared_res.model,
                    crop=crop,
                    cost_name=cost_name,
                    crop_group=crop_group_from_cost_name(cost_name),
                )
                model_path, meta_path = expected_cost_model_path(model_dir, crop, cost_cfg, version)
                dump(serving_model, model_path)
                _save_meta(
                    meta_path,
                    {
                        "metrics": shared_res.metrics,
                        "feature_cols": shared_res.feature_cols,
                        "feature_meta": shared_res.feature_meta,
                        "shared_panel_model": True,
                        "crop": crop,
                        "cost_name": cost_name,
                    },
                )
                report["cost"].append(
                    {
                        "crop": crop,
                        "cost_name": cost_name,
                        "ok": True,
                        "shared_panel_model": True,
                        "model_path": model_path.as_posix(),
                        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                        "prediction_year_window": {
                            "start_year": int(cost_window.get("start_year")),
                            "end_year": int(cost_window.get("end_year")),
                            "window_years": int(cost_window.get("window_years")),
                        },
                        "train_validation_cutoff_date": validation_cutoff_cost,
                    }
                )
            logger.info("trained shared panel-lite cost model in %.2f ms", (time.perf_counter() - t0) * 1000.0)
        except Exception as exc:
            report["cost"].append({"ok": False, "shared_panel_model": True, "error": str(exc)})
            logger.exception("failed training shared panel-lite cost model")
    else:
        for crop, m in sorted(mapping.items(), key=lambda x: x[0]):
            cost_name = m.get("cost_name", "")
            if not cost_name:
                continue
            t0 = time.perf_counter()
            try:
                logger.info("training cost model for crop=%s", crop)
                cost_df = cost_all[cost_all["crop_name"].str.lower() == cost_name.lower()].copy()
                if cost_df.empty:
                    alt = _synonym_cost_name(crop)
                    if alt:
                        cost_df = cost_all[cost_all["crop_name"].str.lower() == alt.lower()].copy()
                if cost_df.empty:
                    raise ValueError(f"no cost data for crop={crop} mapped_cost={cost_name}")
                crop_alignment.setdefault(crop, {})
                crop_alignment[crop]["cost_last_year"] = _safe_last_year(pd.to_numeric(cost_df["year_start"], errors="coerce"))
                cost_window = resolve_year_window_from_series(
                    pd.to_numeric(cost_df["year_start"], errors="coerce").dropna().tolist(),
                    time_cfg=time_cfg,
                    window_years_key="cost_prediction_window_years",
                    preferred_end_year=target_year,
                )
                validation_cutoff_cost = str(cost_window.get("train_validation_cutoff_date") or "2020-12-31")

                res = train_cost(
                    cost_df,
                    cost_cfg,
                    test_ratio=cost_cfg.get("test_ratio"),
                    validation_cutoff=validation_cutoff_cost,
                    strict_cutoff_split=strict_cutoff_split,
                    verbose=cost_cfg.get("verbose", False),
                    label=crop,
                )
                model_path, meta_path = expected_cost_model_path(model_dir, crop, cost_cfg, version)
                dump(res.model, model_path)
                _save_meta(meta_path, {"metrics": res.metrics, "feature_cols": res.feature_cols})
                report["cost"].append(
                    {
                        "crop": crop,
                        "cost_name": cost_name,
                        "ok": True,
                        "model_path": model_path.as_posix(),
                        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                        "prediction_year_window": {
                            "start_year": int(cost_window.get("start_year")),
                            "end_year": int(cost_window.get("end_year")),
                            "window_years": int(cost_window.get("window_years")),
                        },
                        "train_validation_cutoff_date": validation_cutoff_cost,
                    }
                )
                logger.info("trained cost model for crop=%s in %.2f ms", crop, (time.perf_counter() - t0) * 1000.0)
            except Exception as exc:
                report["cost"].append({"crop": crop, "cost_name": cost_name, "ok": False, "error": str(exc)})
                logger.exception("failed training cost model for crop=%s", crop)

    # Train global yield model.
    t0 = time.perf_counter()
    try:
        logger.info("training global yield model ...")
        yhist = load_yield_history(paths.get("yield_history", ""))
        if yhist.empty:
            raise ValueError("yield history is empty")
        yield_window = resolve_year_window_from_series(
            pd.to_numeric(yhist.get("year"), errors="coerce").dropna().tolist(),
            time_cfg=time_cfg,
            window_years_key="yield_prediction_window_years",
            preferred_end_year=target_year,
        )
        validation_cutoff_yield = str(yield_window.get("train_validation_cutoff_date") or "2020-12-31")
        res = train_yield_model(
            yhist,
            config["model"]["yield"],
            test_ratio=config["model"]["yield"].get("test_ratio"),
            validation_cutoff=validation_cutoff_yield,
            strict_cutoff_split=strict_cutoff_split,
            verbose=config["model"]["yield"].get("verbose", False),
            label="global",
        )
        model_path, meta_path = expected_yield_model_path(model_dir, config["model"]["yield"], version)
        dump(res.model, model_path)
        _save_meta(meta_path, {"metrics": res.metrics, "feature_cols": res.feature_cols})
        report["yield"].append(
            {
                "ok": True,
                "model_path": model_path.as_posix(),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "prediction_year_window": {
                    "start_year": int(yield_window.get("start_year")),
                    "end_year": int(yield_window.get("end_year")),
                    "window_years": int(yield_window.get("window_years")),
                },
                "train_validation_cutoff_date": validation_cutoff_yield,
            }
        )
        logger.info("trained global yield model in %.2f ms", (time.perf_counter() - t0) * 1000.0)
    except Exception as exc:
        report["yield"].append({"ok": False, "error": str(exc)})
        logger.exception("failed training yield model")

    if train_calibrator_flag:
        t0 = time.perf_counter()
        try:
            prob_cfg = config.get("probability", {})
            if not bool(prob_cfg.get("enable_calibrator", False)):
                report["calibrator"].append({"ok": True, "skipped": True, "reason": "disabled_by_config"})
                logger.info("skipping probability calibrator training: disabled by config")
                t0 = None
            else:
                logger.info("training probability calibrator ...")
                hist_path = prob_cfg.get("history_file", "")
                if not hist_path:
                    report["calibrator"].append({"ok": True, "skipped": True, "reason": "history_file_empty"})
                    logger.info("skipping probability calibrator training: history_file is empty")
                    t0 = None
                else:
                    hdf = pd.read_csv(hist_path)
                    cal_res = train_calibrator(hdf, prob_cfg)
                    save_calibrator(config["output"]["out_dir"], cal_res)
                    report["calibrator"].append(
                        {"ok": True, "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2), "history": hist_path}
                    )
                    logger.info("trained calibrator in %.2f ms", (time.perf_counter() - t0) * 1000.0)
        except Exception as exc:
            report["calibrator"].append({"ok": False, "error": str(exc)})
            logger.exception("failed training calibrator")

    alignment_rows: List[dict] = []
    for crop in sorted(crop_alignment.keys()):
        row = crop_alignment.get(crop, {})
        price_last = row.get("price_last_year")
        yield_last = row.get("yield_last_year")
        cost_last = row.get("cost_last_year")
        price_gap = None if price_last is None else max(0, target_year - int(price_last))
        yield_gap = None if yield_last is None else max(0, target_year - int(yield_last))
        cost_gap = None if cost_last is None else max(0, target_year - int(cost_last))
        alignment_rows.append(
            {
                "crop": crop,
                "price_last_year": price_last,
                "yield_last_year": yield_last,
                "cost_last_year": cost_last,
                "target_year": target_year,
                "price_gap_years": price_gap,
                "yield_gap_years": yield_gap,
                "cost_gap_years": cost_gap,
            }
        )
    report["time_alignment"] = {
        "frequency": "year",
        "strategy": str(alignment_cfg.get("strategy", "trend_extrapolate_with_uncertainty")),
        "target_year": int(target_year),
        "train_validation_cutoff_date": "dynamic_by_task_window",
        "strict_cutoff_split": bool(strict_cutoff_split),
        "policy": {
            "use_dynamic_in_sample_windows": bool(time_cfg.get("use_dynamic_in_sample_windows", True)),
            "price_forecast_horizon_days": int(time_cfg.get("price_forecast_horizon_days", 181)),
            "price_prediction_window_months": int(time_cfg.get("price_prediction_window_months", 6)),
            "cost_prediction_window_years": int(time_cfg.get("cost_prediction_window_years", 3)),
            "yield_prediction_window_years": int(time_cfg.get("yield_prediction_window_years", 3)),
        },
        "rows": alignment_rows,
    }

    report_path = Path(config["output"]["out_dir"]) / "模型训练报告.json"
    _save_meta(report_path, report)
    logger.info("wrote training report: %s", report_path.as_posix())
    return report


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(description="Offline train models and export artifacts")
    parser.add_argument("--config", default="后端/配置.yaml")
    parser.add_argument("--train-calibrator", action="store_true")
    args = parser.parse_args()
    train_all(args.config, train_calibrator_flag=args.train_calibrator)


if __name__ == "__main__":
    main()

