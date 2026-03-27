from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from 后端.质量评分 import evaluate_project_quality


def _safe_read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


def _safe_float(v: Any) -> float | None:
    try:
        val = float(v)
    except Exception:
        return None
    if pd.isna(val):
        return None
    return val


def _safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


def _baseline_gap_summary_from_compare(compare: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(compare, dict):
        return {}

    task_status: Dict[str, Any] = {}
    underperforming_tasks: List[str] = []
    ordered_pairs: List[Tuple[str, float]] = []
    labels = {"price": "价格", "yield": "产量", "cost": "成本"}

    for task in ["price", "yield", "cost"]:
        node = compare.get(task, {}) if isinstance(compare.get(task), dict) else {}
        improving_metrics: List[str] = []
        underperforming_metrics: List[str] = []
        metric_improvement_pct: Dict[str, float] = {}

        for metric in ["mae", "rmse", "mape"]:
            row = node.get(metric, {}) if isinstance(node.get(metric), dict) else {}
            pct = _safe_float(row.get("improvement_pct"))
            if pct is None:
                continue
            metric_improvement_pct[metric] = pct
            if pct < 0:
                underperforming_metrics.append(metric)
            else:
                improving_metrics.append(metric)

        if underperforming_metrics:
            underperforming_tasks.append(task)

        mape_gap = metric_improvement_pct.get("mape")
        if mape_gap is not None:
            ordered_pairs.append((task, mape_gap))

        task_status[task] = {
            "task_label": labels.get(task, task),
            "wins_all_available_metrics": bool(metric_improvement_pct) and not underperforming_metrics,
            "improving_metrics": improving_metrics,
            "underperforming_metrics": underperforming_metrics,
            "metric_improvement_pct": metric_improvement_pct,
            "mape_improvement_pct": mape_gap,
        }

    primary_candidates = [item for item in ordered_pairs if item[0] in underperforming_tasks] or ordered_pairs
    primary_bottleneck = min(primary_candidates, key=lambda item: item[1])[0] if primary_candidates else None
    ordered_by_mape_gap = [task for task, _ in sorted(ordered_pairs, key=lambda item: item[1])]
    return {
        "underperforming_tasks": underperforming_tasks,
        "primary_bottleneck": primary_bottleneck,
        "ordered_by_mape_gap": ordered_by_mape_gap,
        "task_status": task_status,
    }


def metrics_summary(out_dir: Path) -> Dict[str, Any]:
    env_metrics = _safe_read_json(out_dir / "环境回测.json") or {}
    yield_metrics_all = _safe_read_json(out_dir / "产量回测.json") or {}
    yield_metrics = yield_metrics_all.get("metrics", {}) if isinstance(yield_metrics_all, dict) else {}
    ml_backtest = _safe_read_json(out_dir / "回测报告.json") or {}

    price_path = out_dir / "价格回测.csv"
    cost_path = out_dir / "成本回测.csv"
    prob_metrics_all = _safe_read_json(out_dir / "概率校准器指标.json") or {}

    price_summary: Dict[str, Any] = {}
    if price_path.exists():
        price_df = pd.read_csv(price_path)
        for c in ["mae", "rmse", "mape"]:
            if c in price_df.columns:
                price_summary[f"{c}_mean"] = float(price_df[c].mean())
                price_summary[f"{c}_median"] = float(price_df[c].median())
        price_summary["n_crops"] = int(len(price_df))

    cost_summary: Dict[str, Any] = {}
    if cost_path.exists():
        cost_df = pd.read_csv(cost_path)
        for c in ["mae", "rmse", "mape"]:
            if c in cost_df.columns:
                cost_summary[f"{c}_mean"] = float(cost_df[c].mean())
                cost_summary[f"{c}_median"] = float(cost_df[c].median())
        cost_summary["n_crops"] = int(len(cost_df))

    if isinstance(ml_backtest, dict):
        task_metrics_test = (
            ml_backtest.get("task_metrics", {}).get("test", {})
            if isinstance(ml_backtest.get("task_metrics"), dict)
            else {}
        )

        def _task_all(task_name: str) -> Dict[str, Any]:
            node = task_metrics_test.get(task_name, {})
            if isinstance(node, dict):
                all_node = node.get("all", {})
                if isinstance(all_node, dict):
                    return all_node
            return {}

        if not price_summary:
            p = _task_all("price")
            if p:
                price_summary = {
                    "mae_mean": _safe_float(p.get("mae")),
                    "rmse_mean": _safe_float(p.get("rmse")),
                    "mape_mean": _safe_float(p.get("mape")),
                    "n_crops": _safe_int(p.get("n")),
                    "source": "训练流水线.回测报告.test.price.all",
                }

        if not cost_summary:
            c = _task_all("cost")
            if c:
                cost_summary = {
                    "mae_mean": _safe_float(c.get("mae")),
                    "rmse_mean": _safe_float(c.get("rmse")),
                    "mape_mean": _safe_float(c.get("mape")),
                    "n_crops": _safe_int(c.get("n")),
                    "source": "训练流水线.回测报告.test.cost.all",
                }

        if not yield_metrics:
            y = _task_all("yield")
            if y:
                yield_metrics = {
                    "mae": _safe_float(y.get("mae")),
                    "rmse": _safe_float(y.get("rmse")),
                    "mape": _safe_float(y.get("mape")),
                    "n": _safe_int(y.get("n")),
                    "source": "训练流水线.回测报告.test.yield.all",
                }

    prob_summary: Dict[str, Any] = {}
    if isinstance(prob_metrics_all, dict):
        nested = prob_metrics_all.get("metrics", {}) if isinstance(prob_metrics_all.get("metrics"), dict) else {}
        for key in ["cv_mean_logloss", "cv_mean_brier", "cv_mean_ece"]:
            val = prob_metrics_all.get(key)
            if val is None:
                val = nested.get(key)
            if val is not None:
                prob_summary[key] = val
        cv = nested.get("cv", [])
        if isinstance(cv, list):
            prob_summary["walk_forward_folds"] = len(cv)
        n_train = prob_metrics_all.get("n_train")
        n_valid = prob_metrics_all.get("n_valid")
        if n_train is not None:
            prob_summary["n_train"] = n_train
        if n_valid is not None:
            prob_summary["n_valid"] = n_valid
        diagnostics = prob_metrics_all.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            alerts = diagnostics.get("alerts", [])
            if isinstance(alerts, list):
                prob_summary["alerts"] = alerts
                prob_summary["alert_count"] = int(len(alerts))

    baseline_gap_summary: Dict[str, Any] = {}
    if isinstance(ml_backtest, dict):
        baseline_gap_summary = ml_backtest.get("baseline_gap_summary", {}) or {}
        if not baseline_gap_summary:
            baseline_gap_summary = _baseline_gap_summary_from_compare(ml_backtest.get("baseline_comparison", {}))

    quality = evaluate_project_quality(
        env_summary=env_metrics if isinstance(env_metrics, dict) else {},
        price_summary=price_summary,
        cost_summary=cost_summary,
        yield_summary=yield_metrics if isinstance(yield_metrics, dict) else {},
        prob_summary=prob_summary,
    )

    return {
        "env": env_metrics,
        "price": price_summary,
        "cost": cost_summary,
        "yield": yield_metrics,
        "probability_calibrator": prob_summary,
        "baseline_gap_summary": baseline_gap_summary,
        "quality": quality,
    }


def latest_recommendation(out_dir: Path, logger: Any | None = None) -> Dict[str, Any]:
    raw = _safe_read_json(out_dir / "推荐结果.json") or {}
    rows = raw.get("results", [])
    if not isinstance(rows, list):
        rows = []

    cleaned_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cleaned_rows.append(
            {
                "crop": row.get("crop"),
                "env_prob": _safe_float(row.get("env_prob")),
                "price_pred": _safe_float(row.get("price_pred")),
                "cost_pred": _safe_float(row.get("cost_pred")),
                "yield": _safe_float(row.get("yield")),
                "profit": _safe_float(row.get("profit")),
                "risk": _safe_float(row.get("risk")),
                "score": _safe_float(row.get("score")),
                "prob_best": _safe_float(row.get("prob_best")),
                "prob_best_source": row.get("prob_best_source"),
            }
        )

    if not cleaned_rows:
        csv_path = out_dir / "推荐结果.csv"
        if csv_path.exists():
            try:
                rec_df = pd.read_csv(csv_path)
                for _, row in rec_df.iterrows():
                    crop = row.get("crop")
                    if not crop:
                        continue
                    price_pred = row.get("price_pred", row.get("price_hat"))
                    cost_pred = row.get("cost_pred", row.get("cost_hat"))
                    yield_pred = row.get("yield", row.get("yield_hat"))
                    profit_pred = row.get("profit", row.get("profit_hat"))
                    cleaned_rows.append(
                        {
                            "crop": crop,
                            "env_prob": _safe_float(row.get("env_prob")),
                            "price_pred": _safe_float(price_pred),
                            "cost_pred": _safe_float(cost_pred),
                            "yield": _safe_float(yield_pred),
                            "profit": _safe_float(profit_pred),
                            "risk": _safe_float(row.get("risk")),
                            "score": _safe_float(row.get("score")),
                            "prob_best": _safe_float(row.get("prob_best")),
                            "prob_best_source": row.get("prob_best_source") or "训练流水线",
                        }
                    )
            except Exception:
                if logger is not None and hasattr(logger, "exception"):
                    logger.exception("failed parsing 推荐结果.csv fallback")

    return {
        "env": raw.get("env", {}),
        "runtime": raw.get("runtime", {}),
        "results": cleaned_rows,
        "top5": cleaned_rows[:5],
    }


def insights_summary(out_dir: Path, logger: Any | None = None) -> Dict[str, Any]:
    latest = latest_recommendation(out_dir=out_dir, logger=logger)
    results = latest["results"]
    profits = [x["profit"] for x in results if isinstance(x.get("profit"), (int, float))]
    risks = [x["risk"] for x in results if isinstance(x.get("risk"), (int, float))]

    def _safe_stats(vals: List[float]) -> Dict[str, float | None]:
        if not vals:
            return {"min": None, "max": None, "mean": None}
        return {
            "min": float(min(vals)),
            "max": float(max(vals)),
            "mean": float(sum(vals) / len(vals)),
        }

    price_best: List[Dict[str, Any]] = []
    price_worst: List[Dict[str, Any]] = []
    price_path = out_dir / "价格回测.csv"
    if price_path.exists():
        price_df = pd.read_csv(price_path)
        if "mae" in price_df.columns and "crop" in price_df.columns and len(price_df):
            df = price_df[["crop", "mae", "mape"]].copy()
            df["mae"] = pd.to_numeric(df["mae"], errors="coerce")
            df["mape"] = pd.to_numeric(df["mape"], errors="coerce")
            # Cross-crop MAE is scale-sensitive (high-price crops are naturally larger in absolute error),
            # so rank by relative error first and use MAE only as tie-break/reference.
            rank_base = df.copy()
            rank_base["rank_metric"] = rank_base["mape"]
            rank_base["rank_metric"] = rank_base["rank_metric"].where(rank_base["rank_metric"].notna(), rank_base["mae"])

            best_df = rank_base.sort_values(["rank_metric", "mae"], ascending=[True, True], na_position="last").head(5)
            worst_df = rank_base.sort_values(["rank_metric", "mae"], ascending=[False, False], na_position="last").head(5)
            price_best = [
                {"crop": str(r.crop), "mae": _safe_float(r.mae), "mape": _safe_float(r.mape)}
                for r in best_df.itertuples(index=False)
            ]
            price_worst = [
                {"crop": str(r.crop), "mae": _safe_float(r.mae), "mape": _safe_float(r.mape)}
                for r in worst_df.itertuples(index=False)
            ]

    return {
        "latest_recommendation": latest,
        "profit_stats": _safe_stats(profits),
        "risk_stats": _safe_stats(risks),
        "price_best_5": price_best,
        "price_worst_5": price_worst,
    }
