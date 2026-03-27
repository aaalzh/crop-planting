from __future__ import annotations

import copy
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from 后端.环境桥接 import load_env_scenario_library, resolve_env_scenario_path
from 后端.反馈回流 import get_feedback_training_status


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_abs(root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if out != out:
            return None
        return out
    except Exception:
        return None


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_file(src: Path, dst: Path) -> Optional[str]:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.as_posix()


def _copy_tree(src: Path, dst: Path) -> Optional[str]:
    if not src.exists():
        return None
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst.as_posix()


def _output_dir_from_config(root: Path, config: dict) -> Path:
    output_cfg = config.get("output", {})
    if isinstance(output_cfg, dict):
        text = str(output_cfg.get("out_dir", "")).strip()
        if text:
            return _to_abs(root, text)
    paths_cfg = config.get("paths", {})
    if isinstance(paths_cfg, dict):
        text = str(paths_cfg.get("output_dir", "")).strip()
        if text:
            return _to_abs(root, text)
    return _to_abs(root, "输出")


def _release_root(root: Path, config: dict) -> Path:
    release_cfg = config.get("release", {})
    text = str(release_cfg.get("root_dir", "")).strip()
    if text:
        return _to_abs(root, text)
    return _output_dir_from_config(root, config) / "发布"


def _registry_path(root: Path, config: dict) -> Path:
    return _release_root(root, config) / "发布索引.json"


def _manifest_path(root: Path, config: dict, run_id: str) -> Path:
    return _release_root(root, config) / run_id / "发布清单.json"


def _load_registry(root: Path, config: dict) -> dict:
    payload = _read_json(_registry_path(root, config))
    if not isinstance(payload, dict):
        return {
            "champion_run_id": None,
            "challenger_run_id": None,
            "archived_run_ids": [],
            "history": [],
            "updated_at": None,
        }
    payload.setdefault("champion_run_id", None)
    payload.setdefault("challenger_run_id", None)
    payload.setdefault("archived_run_ids", [])
    payload.setdefault("history", [])
    payload.setdefault("updated_at", None)
    return payload


def _save_registry(root: Path, config: dict, registry: dict) -> None:
    registry = dict(registry)
    registry["updated_at"] = _now_iso()
    _write_json(_registry_path(root, config), registry)


def _load_manifest(root: Path, config: dict, run_id: str | None) -> Optional[dict]:
    if not run_id:
        return None
    payload = _read_json(_manifest_path(root, config, run_id))
    return payload if isinstance(payload, dict) else None


def _set_manifest_status(root: Path, config: dict, run_id: str, status: str) -> Optional[dict]:
    manifest = _load_manifest(root, config, run_id)
    if not manifest:
        return None
    manifest["status"] = str(status)
    manifest["updated_at"] = _now_iso()
    _write_json(_manifest_path(root, config, run_id), manifest)
    return manifest


def _artifact_summary(path: Optional[str]) -> dict:
    if not path:
        return {"path": None, "exists": False}
    file_path = Path(path)
    return {"path": path, "exists": file_path.exists()}


def _report_metrics(training_report: dict, backtest_report: dict) -> dict:
    return {
        "task_metrics_test": ((backtest_report.get("task_metrics") or {}).get("test") or {}),
        "business_metrics_test": ((backtest_report.get("business_metrics") or {}).get("test") or {}),
        "baseline_gap_summary": backtest_report.get("baseline_gap_summary") or {},
        "score_fusion": training_report.get("score_fusion") or {},
        "time_policy": training_report.get("time_policy") or {},
        "env_probability": training_report.get("env_probability") or backtest_report.get("env_probability") or {},
        "closed_loop": training_report.get("closed_loop") or backtest_report.get("closed_loop") or {},
    }


def _derive_closed_loop_summary(root: Path, config: dict) -> dict:
    try:
        return {
            "feedback_training": get_feedback_training_status(root=root, config=config, refresh=True),
        }
    except Exception as exc:
        return {
            "feedback_training": {"error": f"{type(exc).__name__}: {exc}"},
        }


def _derive_env_probability_from_scenarios(root: Path, config: dict) -> dict:
    payload = load_env_scenario_library(root, config, rebuild_if_missing=True)
    items = payload.get("items") or [] if isinstance(payload, dict) else []
    scores: Dict[str, List[float]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        probabilities = item.get("probabilities") or {}
        if not isinstance(probabilities, dict):
            continue
        for crop, prob in probabilities.items():
            value = _safe_float(prob)
            key = str(crop or "").strip().lower()
            if value is None or not key:
                continue
            scores.setdefault(key, []).append(float(value))
    averages = {
        crop: float(sum(values) / max(len(values), 1))
        for crop, values in scores.items()
        if values
    }
    return {
        "source": "scenario_library" if averages else "empirical_prior",
        "crop_count": int(len(averages)),
        "probability_sum": float(sum(averages.values())) if averages else 0.0,
        "backend_config_path": None,
        "scenario_file": resolve_env_scenario_path(root, config).as_posix(),
        "scenario_count": int(len(items)) if isinstance(items, list) else 0,
    }


def _default_env_input(root: Path) -> dict:
    for rel in (
        "数据/样例/环境示例.json",
        "示例输入/最小可测输入/数据/样例/环境示例.json",
    ):
        path = _to_abs(root, rel)
        payload = _read_json(path)
        if isinstance(payload, dict):
            return payload
    return {
        "N": 63,
        "P": 55,
        "K": 42,
        "temperature": 25.96,
        "humidity": 83.5,
        "ph": 5.27,
        "rainfall": 295.73,
    }


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _feedback_log_paths(root: Path, config: dict) -> Dict[str, Path]:
    feedback_cfg = config.get("feedback", {}) if isinstance(config, dict) else {}
    return {
        "inference": _to_abs(root, str(feedback_cfg.get("inference_log_file", "输出/闭环/推理事件.jsonl"))),
        "feedback": _to_abs(root, str(feedback_cfg.get("feedback_log_file", "输出/闭环/用户反馈.jsonl"))),
    }


def _build_release_runtime_config(root: Path, config: dict, release_dir: Path, manifest: dict) -> Tuple[dict, str]:
    effective = copy.deepcopy(config)
    output_cfg = effective.setdefault("output", {})
    output_cfg["out_dir"] = release_dir.as_posix()

    serving_cfg = effective.setdefault("serving", {})
    manifest_serving = manifest.get("serving") or {}
    default_strategy = str(manifest_serving.get("default_strategy", "precomputed")).strip().lower()
    if default_strategy not in {"online", "precomputed"}:
        default_strategy = "precomputed"
    serving_cfg["recommend_strategy"] = default_strategy

    precomputed_file = str(manifest_serving.get("precomputed_recommendation_file", "")).strip()
    if precomputed_file:
        serving_cfg["precomputed_recommendation_file"] = precomputed_file

    env_bundle = str(manifest_serving.get("env_model_bundle", "")).strip()
    if env_bundle:
        effective.setdefault("paths", {})["env_model_bundle"] = env_bundle

    return effective, default_strategy


def _run_release_inference(root: Path, config: dict, release_dir: Path, manifest: dict, env_input: dict) -> dict:
    effective, default_strategy = _build_release_runtime_config(root, config, release_dir, manifest)
    try:
        if default_strategy == "online":
            from 后端.推荐器 import recommend as recommend_online

            payload = recommend_online(env_input, effective)
        else:
            from 后端.推荐数据源 import recommend_from_precomputed

            payload = recommend_from_precomputed(env_input=env_input, config=effective, root=root, output_dir=release_dir)

        results = payload.get("results") if isinstance(payload, dict) else None
        rows = results if isinstance(results, list) else []
        topk = [
            str(row.get("crop") or "").strip().lower()
            for row in rows
            if isinstance(row, dict) and str(row.get("crop") or "").strip()
        ]
        return {
            "ok": bool(rows),
            "strategy": default_strategy,
            "candidate_count": len(rows),
            "best_crop": topk[0] if topk else None,
            "topk": topk[:8],
            "error": None,
            "checked_at": _now_iso(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "strategy": default_strategy,
            "candidate_count": 0,
            "best_crop": None,
            "topk": [],
            "error": f"{type(exc).__name__}: {exc}",
            "checked_at": _now_iso(),
        }


def _logged_topk_names(inference: dict) -> List[str]:
    names: List[str] = []
    final_topk = inference.get("final_topk") or []
    if isinstance(final_topk, list):
        for item in final_topk:
            if isinstance(item, dict):
                crop = str(item.get("crop") or "").strip().lower()
            elif isinstance(item, (list, tuple)) and item:
                crop = str(item[0] or "").strip().lower()
            else:
                crop = ""
            if crop:
                names.append(crop)
    if names:
        return names

    results = inference.get("results") or []
    return [
        str(row.get("crop") or "").strip().lower()
        for row in results
        if isinstance(row, dict) and str(row.get("crop") or "").strip()
    ]


def _run_release_smoke(root: Path, config: dict, release_dir: Path, manifest: dict) -> dict:
    sample_env = _default_env_input(root)
    result = _run_release_inference(root, config, release_dir, manifest, sample_env)
    return {
        "ok": bool(result.get("ok", False)),
        "strategy": result.get("strategy"),
        "candidate_count": result.get("candidate_count", 0),
        "best_crop": result.get("best_crop"),
        "error": result.get("error"),
        "checked_at": result.get("checked_at") or _now_iso(),
    }


def _run_shadow_replay(root: Path, config: dict, release_dir: Path, manifest: dict) -> dict:
    release_cfg = config.get("release", {})
    gate_cfg = release_cfg.get("gate", {}) if isinstance(release_cfg, dict) else {}
    max_events = max(0, int(gate_cfg.get("shadow_replay_max_events", 20)))
    topk_limit = max(1, int(gate_cfg.get("shadow_topk", 5)))
    min_top1_match_rate = float(gate_cfg.get("min_shadow_top1_match_rate", 0.5))
    min_selected_hit_rate = float(gate_cfg.get("min_shadow_selected_hit_rate", 0.5))

    if max_events <= 0:
        return {
            "enabled": False,
            "ok": True,
            "available_events": 0,
            "attempted_events": 0,
            "replayed_events": 0,
            "reason": "disabled",
            "checked_at": _now_iso(),
        }

    log_paths = _feedback_log_paths(root, config)
    inference_rows = [row for row in _read_jsonl(log_paths["inference"]) if isinstance(row.get("env_input"), dict)]
    if not inference_rows:
        return {
            "enabled": True,
            "ok": True,
            "available_events": 0,
            "attempted_events": 0,
            "replayed_events": 0,
            "reason": "no_inference_logs",
            "inference_log_file": log_paths["inference"].as_posix(),
            "feedback_log_file": log_paths["feedback"].as_posix(),
            "checked_at": _now_iso(),
        }

    feedback_map: Dict[str, dict] = {}
    for row in _read_jsonl(log_paths["feedback"]):
        event_id = str(row.get("event_id") or "").strip()
        if not event_id:
            continue
        current = feedback_map.get(event_id)
        created_at = str(row.get("created_at") or "")
        current_at = str((current or {}).get("created_at") or "")
        if current is None or created_at >= current_at:
            feedback_map[event_id] = row

    recent_rows = inference_rows[-max_events:]
    comparisons: List[dict] = []
    replayed_events = 0
    failure_count = 0
    top1_match_count = 0
    selected_eval_count = 0
    selected_hit_count = 0

    for row in recent_rows:
        event_id = str(row.get("event_id") or "").strip() or None
        replay = _run_release_inference(root, config, release_dir, manifest, row.get("env_input") or {})
        logged_topk = _logged_topk_names(row)
        candidate_topk = [str(item or "").strip().lower() for item in (replay.get("topk") or []) if str(item or "").strip()]
        logged_top1 = logged_topk[0] if logged_topk else None
        candidate_top1 = candidate_topk[0] if candidate_topk else None
        top1_match = bool(replay.get("ok", False) and logged_top1 and candidate_top1 and logged_top1 == candidate_top1)
        if replay.get("ok", False):
            replayed_events += 1
        else:
            failure_count += 1
        if top1_match:
            top1_match_count += 1

        feedback = feedback_map.get(event_id or "") or {}
        selected_crop = str(feedback.get("selected_crop") or "").strip().lower() or None
        selected_in_topk = None
        if selected_crop:
            selected_eval_count += 1
            selected_in_topk = selected_crop in candidate_topk[:topk_limit]
            if selected_in_topk:
                selected_hit_count += 1

        if len(comparisons) < 12:
            comparisons.append(
                {
                    "event_id": event_id,
                    "logged_top1": logged_top1,
                    "candidate_top1": candidate_top1,
                    "top1_match": top1_match,
                    "selected_crop": selected_crop,
                    "selected_in_topk": selected_in_topk,
                    "candidate_count": replay.get("candidate_count", 0),
                    "error": replay.get("error"),
                }
            )

    top1_match_rate = float(top1_match_count / replayed_events) if replayed_events else None
    selected_hit_rate = float(selected_hit_count / selected_eval_count) if selected_eval_count else None
    ok = failure_count == 0
    if replayed_events:
        ok = ok and (top1_match_rate is None or top1_match_rate >= min_top1_match_rate)
    if selected_eval_count:
        ok = ok and (selected_hit_rate is None or selected_hit_rate >= min_selected_hit_rate)

    return {
        "enabled": True,
        "ok": bool(ok),
        "available_events": len(inference_rows),
        "attempted_events": len(recent_rows),
        "replayed_events": replayed_events,
        "failure_count": failure_count,
        "topk_limit": topk_limit,
        "top1_match_rate": top1_match_rate,
        "selected_eval_count": selected_eval_count,
        "selected_hit_rate": selected_hit_rate,
        "min_top1_match_rate": min_top1_match_rate,
        "min_selected_hit_rate": min_selected_hit_rate,
        "inference_log_file": log_paths["inference"].as_posix(),
        "feedback_log_file": log_paths["feedback"].as_posix(),
        "comparisons": comparisons,
        "checked_at": _now_iso(),
    }


def _task_metric(metrics: dict, task: str, metric_name: str) -> Optional[float]:
    task_node = (metrics.get("task_metrics_test") or {}).get(task) or {}
    if not isinstance(task_node, dict):
        return None
    candidate = task_node.get("all") if isinstance(task_node.get("all"), dict) else task_node
    if not isinstance(candidate, dict):
        return None
    return _safe_float(candidate.get(metric_name))


def _business_metric(metrics: dict, metric_name: str) -> Optional[float]:
    node = metrics.get("business_metrics_test") or {}
    if not isinstance(node, dict):
        return None
    return _safe_float(node.get(metric_name))


def _release_report_payload(manifest: dict) -> dict:
    gating = manifest.get("gating") or {}
    metrics = manifest.get("metrics") or {}
    score_fusion = manifest.get("score_fusion") or {}
    return {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "created_at": manifest.get("created_at"),
        "updated_at": manifest.get("updated_at"),
        "source": manifest.get("source"),
        "release_dir": manifest.get("release_dir"),
        "manifest_path": manifest.get("manifest_path"),
        "baseline": gating.get("baseline"),
        "allowed": gating.get("allowed"),
        "summary": gating.get("summary"),
        "smoke": manifest.get("smoke") or {},
        "shadow": manifest.get("shadow") or {},
        "gate": {
            "smoke_checks": gating.get("smoke_checks") or [],
            "shadow_checks": gating.get("shadow_checks") or [],
            "module_checks": gating.get("module_checks") or [],
            "business_checks": gating.get("business_checks") or [],
            "health_checks": gating.get("health_checks") or [],
            "provenance_checks": gating.get("provenance_checks") or [],
            "failed_checks": gating.get("failed_checks") or [],
        },
        "env_probability": metrics.get("env_probability") or {},
        "closed_loop": metrics.get("closed_loop") or manifest.get("closed_loop") or {},
        "score_fusion": {
            "weights": score_fusion.get("weights") or {},
            "validation_objective": score_fusion.get("validation_objective") or {},
        },
        "artifacts": manifest.get("artifacts") or {},
    }


def _evaluate_gate(challenger: dict, champion: Optional[dict], config: dict) -> dict:
    release_cfg = config.get("release", {})
    gate_cfg = release_cfg.get("gate", {}) if isinstance(release_cfg, dict) else {}
    max_metric_regression_ratio = float(gate_cfg.get("max_metric_regression_ratio", 0.03))
    max_profit_mae_regression_ratio = float(gate_cfg.get("max_profit_mae_regression_ratio", 0.03))
    min_topk_profit_ratio = float(gate_cfg.get("min_topk_profit_ratio", 0.0))
    min_ndcg_delta = float(gate_cfg.get("min_ndcg_delta", -0.01))
    min_hit_rate_delta = float(gate_cfg.get("min_hit_rate_delta", -0.01))
    require_env_scenario_library = bool(gate_cfg.get("require_env_scenario_library", True))
    min_env_scenario_count = int(gate_cfg.get("min_env_scenario_count", 1))
    env_prob_sum_tolerance = float(gate_cfg.get("env_prob_sum_tolerance", 0.02))
    require_score_weight_provenance = bool(gate_cfg.get("require_score_weight_provenance", True))
    require_shadow_replay_when_logs_exist = bool(gate_cfg.get("require_shadow_replay_when_logs_exist", True))

    artifacts = challenger.get("artifacts") or {}
    serving = challenger.get("serving") or {}
    required_smoke = [
        artifacts.get("manifest"),
        artifacts.get("recommendation_csv"),
        artifacts.get("training_report"),
        artifacts.get("backtest_report"),
    ]
    if str(serving.get("default_strategy", "precomputed")).strip().lower() == "online":
        required_smoke.append(artifacts.get("model_dir"))

    smoke_checks = []
    smoke_ok = True
    for item in required_smoke:
        entry = item if isinstance(item, dict) else {"path": None, "exists": False}
        check = {
            "path": entry.get("path"),
            "exists": bool(entry.get("exists", False)),
        }
        smoke_checks.append(check)
        smoke_ok = smoke_ok and check["exists"]

    smoke_inference = challenger.get("smoke") or {}
    if isinstance(smoke_inference, dict):
        smoke_checks.append(
            {
                "path": "release_inference",
                "exists": bool(smoke_inference.get("ok", False)),
                "detail": smoke_inference,
            }
        )
        smoke_ok = smoke_ok and bool(smoke_inference.get("ok", False))

    failed_checks: List[str] = []
    module_checks: List[dict] = []
    business_checks: List[dict] = []
    health_checks: List[dict] = []
    provenance_checks: List[dict] = []
    shadow_checks: List[dict] = []

    shadow_replay = challenger.get("shadow") or {}
    shadow_available = int(shadow_replay.get("available_events", 0) or 0) if isinstance(shadow_replay, dict) else 0
    shadow_attempted = int(shadow_replay.get("attempted_events", 0) or 0) if isinstance(shadow_replay, dict) else 0
    shadow_expected = bool(require_shadow_replay_when_logs_exist and shadow_available > 0)
    if isinstance(shadow_replay, dict) and (shadow_expected or shadow_attempted > 0):
        passed = bool(shadow_replay.get("ok", False))
        shadow_checks.append(
            {
                "metric": "shadow_replay",
                "direction": "pass",
                "required": shadow_expected,
                "available_events": shadow_available,
                "attempted_events": shadow_attempted,
                "replayed_events": int(shadow_replay.get("replayed_events", 0) or 0),
                "top1_match_rate": shadow_replay.get("top1_match_rate"),
                "selected_hit_rate": shadow_replay.get("selected_hit_rate"),
                "passed": passed,
            }
        )
        if not passed:
            failed_checks.append("shadow.replay")

    challenger_metrics = challenger.get("metrics") or {}
    env_probability = challenger_metrics.get("env_probability") or challenger.get("environment_bridge") or {}
    env_source = str(env_probability.get("source", "")).strip().lower()
    env_prob_sum = _safe_float(env_probability.get("probability_sum"))
    try:
        env_scenario_count = int(env_probability.get("scenario_count", challenger.get("environment_bridge", {}).get("scenario_count", 0)) or 0)
    except Exception:
        env_scenario_count = 0

    if require_env_scenario_library:
        passed = env_source == "scenario_library"
        health_checks.append(
            {
                "metric": "env_probability_source",
                "direction": "equals",
                "challenger": env_source or None,
                "expected": "scenario_library",
                "passed": passed,
            }
        )
        if not passed:
            failed_checks.append("health.env_probability_source")

    passed = env_scenario_count >= max(1, min_env_scenario_count)
    health_checks.append(
        {
            "metric": "env_scenario_count",
            "direction": "greater_or_equal",
            "challenger": env_scenario_count,
            "threshold": max(1, min_env_scenario_count),
            "passed": passed,
        }
    )
    if not passed:
        failed_checks.append("health.env_scenario_count")

    passed = env_prob_sum is not None and abs(env_prob_sum - 1.0) <= max(1e-6, env_prob_sum_tolerance)
    health_checks.append(
        {
            "metric": "env_probability_sum",
            "direction": "near_one",
            "challenger": env_prob_sum,
            "threshold": env_prob_sum_tolerance,
            "passed": passed,
        }
    )
    if not passed:
        failed_checks.append("health.env_probability_sum")

    score_fusion = challenger.get("score_fusion") or {}
    score_weights = score_fusion.get("weights") or {}
    validation_objective = score_fusion.get("validation_objective") or {}

    passed = isinstance(score_weights, dict) and bool(score_weights)
    provenance_checks.append(
        {
            "metric": "score_weights_present",
            "direction": "exists",
            "challenger": bool(passed),
            "passed": passed,
        }
    )
    if not passed:
        failed_checks.append("provenance.score_weights")

    passed = True
    if require_score_weight_provenance:
        passed = isinstance(validation_objective, dict) and bool(validation_objective)
    provenance_checks.append(
        {
            "metric": "score_weight_validation_objective",
            "direction": "exists",
            "challenger": bool(isinstance(validation_objective, dict) and bool(validation_objective)),
            "required": require_score_weight_provenance,
            "passed": passed,
        }
    )
    if not passed:
        failed_checks.append("provenance.score_weight_validation")

    if not champion:
        allowed = bool(smoke_ok) and not failed_checks
        return {
            "allowed": allowed,
            "smoke_ok": bool(smoke_ok),
            "smoke_checks": smoke_checks,
            "shadow_checks": shadow_checks,
            "baseline": "bootstrap",
            "module_checks": [],
            "business_checks": [],
            "health_checks": health_checks,
            "provenance_checks": provenance_checks,
            "failed_checks": failed_checks if failed_checks else ([] if smoke_ok else ["smoke"]),
            "summary": "no_champion_baseline" if allowed else "bootstrap_blocked",
            "evaluated_at": _now_iso(),
        }

    champion_metrics = champion.get("metrics") or {}

    for task in ("price", "yield", "cost"):
        for metric_name in ("mae", "rmse", "mape"):
            current = _task_metric(challenger_metrics, task, metric_name)
            baseline = _task_metric(champion_metrics, task, metric_name)
            if current is None or baseline is None:
                continue
            threshold = baseline * (1.0 + max_metric_regression_ratio)
            passed = current <= threshold
            item = {
                "scope": task,
                "metric": metric_name,
                "direction": "lower_is_better",
                "challenger": current,
                "champion": baseline,
                "threshold": threshold,
                "passed": passed,
            }
            module_checks.append(item)
            if not passed:
                failed_checks.append(f"{task}.{metric_name}")

    profit_mae_ch = _business_metric(challenger_metrics, "profit_mae")
    profit_mae_cp = _business_metric(champion_metrics, "profit_mae")
    if profit_mae_ch is not None and profit_mae_cp is not None:
        threshold = profit_mae_cp * (1.0 + max_profit_mae_regression_ratio)
        passed = profit_mae_ch <= threshold
        business_checks.append(
            {
                "metric": "profit_mae",
                "direction": "lower_is_better",
                "challenger": profit_mae_ch,
                "champion": profit_mae_cp,
                "threshold": threshold,
                "passed": passed,
            }
        )
        if not passed:
            failed_checks.append("business.profit_mae")

    topk_profit_ch = _business_metric(challenger_metrics, "topk_avg_profit")
    topk_profit_cp = _business_metric(champion_metrics, "topk_avg_profit")
    if topk_profit_ch is not None and topk_profit_cp is not None:
        threshold = topk_profit_cp * (1.0 + min_topk_profit_ratio)
        passed = topk_profit_ch >= threshold
        business_checks.append(
            {
                "metric": "topk_avg_profit",
                "direction": "higher_is_better",
                "challenger": topk_profit_ch,
                "champion": topk_profit_cp,
                "threshold": threshold,
                "passed": passed,
            }
        )
        if not passed:
            failed_checks.append("business.topk_avg_profit")

    ndcg_ch = _business_metric(challenger_metrics, "ndcg_at_k")
    ndcg_cp = _business_metric(champion_metrics, "ndcg_at_k")
    if ndcg_ch is not None and ndcg_cp is not None:
        threshold = ndcg_cp + min_ndcg_delta
        passed = ndcg_ch >= threshold
        business_checks.append(
            {
                "metric": "ndcg_at_k",
                "direction": "higher_is_better",
                "challenger": ndcg_ch,
                "champion": ndcg_cp,
                "threshold": threshold,
                "passed": passed,
            }
        )
        if not passed:
            failed_checks.append("business.ndcg_at_k")

    hit_rate_ch = _business_metric(challenger_metrics, "hit_rate_at_k")
    hit_rate_cp = _business_metric(champion_metrics, "hit_rate_at_k")
    if hit_rate_ch is not None and hit_rate_cp is not None:
        threshold = hit_rate_cp + min_hit_rate_delta
        passed = hit_rate_ch >= threshold
        business_checks.append(
            {
                "metric": "hit_rate_at_k",
                "direction": "higher_is_better",
                "challenger": hit_rate_ch,
                "champion": hit_rate_cp,
                "threshold": threshold,
                "passed": passed,
            }
        )
        if not passed:
            failed_checks.append("business.hit_rate_at_k")

    allowed = bool(smoke_ok) and not failed_checks
    return {
        "allowed": allowed,
        "smoke_ok": bool(smoke_ok),
        "smoke_checks": smoke_checks,
        "shadow_checks": shadow_checks,
        "baseline": champion.get("run_id"),
        "module_checks": module_checks,
        "business_checks": business_checks,
        "health_checks": health_checks,
        "provenance_checks": provenance_checks,
        "failed_checks": failed_checks,
        "summary": "pass" if allowed else "blocked",
        "evaluated_at": _now_iso(),
    }


def _summary_from_manifest(manifest: Optional[dict]) -> Optional[dict]:
    if not manifest:
        return None
    gating = manifest.get("gating") or {}
    serving = manifest.get("serving") or {}
    shadow = manifest.get("shadow") or {}
    feedback_training = ((manifest.get("closed_loop") or {}).get("feedback_training") or {})
    return {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "created_at": manifest.get("created_at"),
        "updated_at": manifest.get("updated_at"),
        "source": manifest.get("source"),
        "release_dir": manifest.get("release_dir"),
        "manifest_path": manifest.get("manifest_path"),
        "default_strategy": serving.get("default_strategy"),
        "allowed": gating.get("allowed"),
        "summary": gating.get("summary"),
        "shadow_ok": shadow.get("ok"),
        "shadow_attempted_events": shadow.get("attempted_events"),
        "feedback_labeled_sample_count": feedback_training.get("labeled_sample_count"),
    }


def promote_release(root: Path, config: dict, run_id: str, reason: str = "manual") -> dict:
    registry = _load_registry(root, config)
    manifest = _load_manifest(root, config, run_id)
    if not manifest:
        raise FileNotFoundError(f"release manifest not found: {run_id}")

    previous = registry.get("champion_run_id")
    if previous and previous != run_id:
        _set_manifest_status(root, config, previous, "archived")
        archived = [rid for rid in registry.get("archived_run_ids", []) if rid != run_id]
        archived.insert(0, previous)
        keep_archived = max(1, int(config.get("release", {}).get("keep_archived", 12)))
        registry["archived_run_ids"] = archived[:keep_archived]

    manifest["status"] = "champion"
    manifest["updated_at"] = _now_iso()
    _write_json(_manifest_path(root, config, run_id), manifest)

    registry["champion_run_id"] = run_id
    if registry.get("challenger_run_id") == run_id:
        registry["challenger_run_id"] = None
    registry.setdefault("history", []).insert(
        0,
        {
            "action": "promote",
            "run_id": run_id,
            "reason": reason,
            "created_at": _now_iso(),
        },
    )
    _save_registry(root, config, registry)
    return {
        "ok": True,
        "champion_run_id": run_id,
        "previous_champion_run_id": previous,
        "reason": reason,
    }


def rollback_release(root: Path, config: dict, run_id: str | None = None) -> dict:
    registry = _load_registry(root, config)
    target = run_id or (registry.get("archived_run_ids") or [None])[0]
    if not target:
        raise ValueError("no archived release available for rollback")
    return promote_release(root, config, str(target), reason="rollback")


def build_release_from_output(
    *,
    root: Path,
    config: dict,
    output_dir: Path | None = None,
    run_id: str | None = None,
    source: str = "high_precision_training",
    training_report: Optional[dict] = None,
    backtest_report: Optional[dict] = None,
) -> dict:
    release_root = _release_root(root, config)
    release_root.mkdir(parents=True, exist_ok=True)
    resolved_output_dir = output_dir if isinstance(output_dir, Path) else _output_dir_from_config(root, config)
    resolved_output_dir = resolved_output_dir.resolve()
    run_id = str(run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    release_dir = (release_root / run_id).resolve()
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    training_report = training_report or _read_json(resolved_output_dir / "模型训练报告.json") or {}
    backtest_report = backtest_report or _read_json(resolved_output_dir / "回测报告.json") or {}
    score_weights = ((training_report.get("score_fusion") or {}).get("weights") or {})
    reported_env_probability = training_report.get("env_probability") or backtest_report.get("env_probability") or {}
    fallback_env_probability = _derive_env_probability_from_scenarios(root, config)
    env_probability = dict(fallback_env_probability)
    if isinstance(reported_env_probability, dict):
        env_probability.update({k: v for k, v in reported_env_probability.items() if v not in (None, "", {})})
    reported_closed_loop = training_report.get("closed_loop") or backtest_report.get("closed_loop") or {}
    fallback_closed_loop = _derive_closed_loop_summary(root, config)
    closed_loop = dict(fallback_closed_loop)
    if isinstance(reported_closed_loop, dict):
        for key, value in reported_closed_loop.items():
            if key == "feedback_training" and isinstance(value, dict):
                merged_feedback = dict(fallback_closed_loop.get("feedback_training") or {})
                merged_feedback.update({k: v for k, v in value.items() if v not in (None, "", {})})
                closed_loop["feedback_training"] = merged_feedback
                continue
            if value not in (None, "", {}):
                closed_loop[key] = value
    scenario_payload = load_env_scenario_library(root, config, rebuild_if_missing=True)
    scenario_items = scenario_payload.get("items") or [] if isinstance(scenario_payload, dict) else []

    artifacts: Dict[str, dict] = {}
    copied_model_dir = _copy_tree(resolved_output_dir / "模型", release_dir / "模型")
    artifacts["model_dir"] = _artifact_summary(copied_model_dir)
    model_dir_has_files = False
    if copied_model_dir:
        model_dir_path = Path(copied_model_dir)
        try:
            model_dir_has_files = any(path.is_file() for path in model_dir_path.rglob("*"))
        except Exception:
            model_dir_has_files = False

    env_bundle_src = _to_abs(root, str((config.get("paths") or {}).get("env_model_bundle", "环境推荐/模型/作物推荐模型管道.pkl")))
    copied_env_bundle = _copy_file(env_bundle_src, release_dir / "环境模型包.pkl")
    artifacts["env_bundle"] = _artifact_summary(copied_env_bundle)

    copied_rec_csv = _copy_file(resolved_output_dir / "推荐结果.csv", release_dir / "推荐结果.csv")
    copied_rec_json = _copy_file(resolved_output_dir / "推荐结果.json", release_dir / "推荐结果.json")
    copied_train_report = _copy_file(resolved_output_dir / "模型训练报告.json", release_dir / "模型训练报告.json")
    copied_backtest_report = _copy_file(resolved_output_dir / "回测报告.json", release_dir / "回测报告.json")
    copied_markdown = _copy_file(resolved_output_dir / "高精度训练报告.md", release_dir / "高精度训练报告.md")
    copied_lifecycle = _copy_file(resolved_output_dir / "输出生命周期报告.json", release_dir / "输出生命周期报告.json")
    copied_calibrator = _copy_file(resolved_output_dir / "概率校准器.pkl", release_dir / "概率校准器.pkl")
    copied_calibrator_meta = _copy_file(resolved_output_dir / "概率校准器指标.json", release_dir / "概率校准器指标.json")
    copied_score_weights = None
    if isinstance(score_weights, dict) and score_weights:
        copied_score_weights = (release_dir / "评分权重.json").as_posix()
        _write_json(release_dir / "评分权重.json", score_weights)

    artifacts["recommendation_csv"] = _artifact_summary(copied_rec_csv)
    artifacts["recommendation_json"] = _artifact_summary(copied_rec_json)
    artifacts["training_report"] = _artifact_summary(copied_train_report)
    artifacts["backtest_report"] = _artifact_summary(copied_backtest_report)
    artifacts["markdown_report"] = _artifact_summary(copied_markdown)
    artifacts["lifecycle_report"] = _artifact_summary(copied_lifecycle)
    artifacts["calibrator_model"] = _artifact_summary(copied_calibrator)
    artifacts["calibrator_meta"] = _artifact_summary(copied_calibrator_meta)
    artifacts["score_weights"] = _artifact_summary(copied_score_weights)

    default_strategy = "online" if model_dir_has_files else "precomputed"
    manifest_path = release_dir / "发布清单.json"
    metrics = _report_metrics(training_report, backtest_report)
    metrics["env_probability"] = env_probability

    manifest = {
        "run_id": run_id,
        "status": "challenger",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "source": source,
        "release_dir": release_dir.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "source_output_dir": resolved_output_dir.as_posix(),
        "time_policy": training_report.get("time_policy") or {},
        "metrics": metrics,
        "closed_loop": closed_loop,
        "score_fusion": training_report.get("score_fusion") or {},
        "environment_bridge": {
            "source": env_probability.get("source"),
            "crop_count": env_probability.get("crop_count"),
            "probability_sum": env_probability.get("probability_sum"),
            "backend_config_path": env_probability.get("backend_config_path"),
            "scenario_file": resolve_env_scenario_path(root, config).as_posix(),
            "scenario_count": int(len(scenario_items)) if isinstance(scenario_items, list) else 0,
        },
        "serving": {
            "default_strategy": default_strategy,
            "output_dir": release_dir.as_posix(),
            "precomputed_recommendation_file": copied_rec_csv,
            "env_model_bundle": copied_env_bundle or env_bundle_src.as_posix(),
            "score_weights": score_weights if isinstance(score_weights, dict) else {},
        },
        "artifacts": artifacts,
        "provenance": {
            "model_cache_version": str((config.get("serving") or {}).get("model_cache_version", "v2")),
            "recommend_strategy": str((config.get("serving") or {}).get("recommend_strategy", "online")),
            "env_probability": env_probability,
            "score_fusion": {
                "weights_present": bool(isinstance(score_weights, dict) and score_weights),
                "validation_objective_present": bool((training_report.get("score_fusion") or {}).get("validation_objective")),
            },
        },
    }
    _write_json(manifest_path, manifest)
    artifacts["manifest"] = _artifact_summary(manifest_path.as_posix())

    registry = _load_registry(root, config)
    previous_challenger = registry.get("challenger_run_id")
    if previous_challenger and previous_challenger != run_id and previous_challenger != registry.get("champion_run_id"):
        _set_manifest_status(root, config, previous_challenger, "archived")
        archived = [rid for rid in registry.get("archived_run_ids", []) if rid not in {previous_challenger, run_id}]
        archived.insert(0, previous_challenger)
        keep_archived = max(1, int(config.get("release", {}).get("keep_archived", 12)))
        registry["archived_run_ids"] = archived[:keep_archived]

    manifest["smoke"] = _run_release_smoke(root, config, release_dir, manifest)
    manifest["shadow"] = _run_shadow_replay(root, config, release_dir, manifest)
    _write_json(manifest_path, manifest)

    champion = _load_manifest(root, config, registry.get("champion_run_id"))
    gate_result = _evaluate_gate(manifest, champion, config)
    manifest["gating"] = gate_result
    _write_json(manifest_path, manifest)

    registry["challenger_run_id"] = run_id
    registry.setdefault("history", []).insert(
        0,
        {
            "action": "build",
            "run_id": run_id,
            "created_at": _now_iso(),
            "allowed": bool(gate_result.get("allowed", False)),
            "summary": gate_result.get("summary"),
        },
    )
    _save_registry(root, config, registry)

    promoted = False
    if bool(config.get("release", {}).get("auto_promote_on_pass", True)) and bool(gate_result.get("allowed", False)):
        promote_release(root, config, run_id, reason="gate_passed")
        manifest = _load_manifest(root, config, run_id) or manifest
        promoted = True

    release_report_path = release_dir / "发布报告.json"
    release_report = _release_report_payload(manifest)
    _write_json(release_report_path, release_report)
    artifacts["release_report"] = _artifact_summary(release_report_path.as_posix())
    manifest["artifacts"] = artifacts
    manifest["release_report"] = {
        "path": release_report_path.as_posix(),
        "summary": release_report.get("summary"),
        "allowed": release_report.get("allowed"),
    }
    _write_json(manifest_path, manifest)

    return {
        "ok": True,
        "run_id": run_id,
        "release_dir": release_dir.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "status": manifest.get("status"),
        "gating": gate_result,
        "release_report": release_report_path.as_posix(),
        "promoted": promoted,
    }


def get_release_status(root: Path, config: dict) -> dict:
    registry = _load_registry(root, config)
    champion = _load_manifest(root, config, registry.get("champion_run_id"))
    challenger = _load_manifest(root, config, registry.get("challenger_run_id"))
    archived_ids = [str(item) for item in registry.get("archived_run_ids", [])]
    archived = [_summary_from_manifest(_load_manifest(root, config, run_id)) for run_id in archived_ids[:8]]
    archived = [item for item in archived if item]
    return {
        "release_root": _release_root(root, config).as_posix(),
        "registry": {
            "champion_run_id": registry.get("champion_run_id"),
            "challenger_run_id": registry.get("challenger_run_id"),
            "archived_run_ids": archived_ids,
            "updated_at": registry.get("updated_at"),
        },
        "champion": _summary_from_manifest(champion),
        "challenger": _summary_from_manifest(challenger),
        "archived": archived,
    }


def resolve_active_release(root: Path, config: dict) -> Tuple[dict, dict]:
    release_cfg = config.get("release", {})
    serving_cfg = config.get("serving", {})
    if not bool(release_cfg.get("enabled", True)):
        return copy.deepcopy(config), {"enabled": False, "reason": "release_disabled"}

    policy = str(serving_cfg.get("active_release_policy", "champion")).strip().lower()
    if policy not in {"champion", "challenger", "none"}:
        policy = "champion"
    if policy == "none":
        return copy.deepcopy(config), {"enabled": False, "reason": "policy_none"}

    registry = _load_registry(root, config)
    run_id = registry.get("champion_run_id") if policy == "champion" else registry.get("challenger_run_id") or registry.get("champion_run_id")
    manifest = _load_manifest(root, config, run_id)
    if not manifest:
        return copy.deepcopy(config), {"enabled": False, "reason": "active_release_missing", "policy": policy}

    effective = copy.deepcopy(config)
    release_dir = Path(str(manifest.get("release_dir", "")))
    if not release_dir.is_absolute():
        release_dir = (_release_root(root, config) / str(manifest.get("run_id"))).resolve()

    output_cfg = effective.setdefault("output", {})
    output_cfg["out_dir"] = release_dir.as_posix()

    paths_cfg = effective.setdefault("paths", {})
    serving_manifest = manifest.get("serving") or {}
    env_bundle = str(serving_manifest.get("env_model_bundle", "")).strip()
    if env_bundle:
        paths_cfg["env_model_bundle"] = env_bundle

    serving_target = effective.setdefault("serving", {})
    precomputed_path = str(serving_manifest.get("precomputed_recommendation_file", "")).strip()
    if precomputed_path:
        serving_target["precomputed_recommendation_file"] = precomputed_path
    serving_target["active_release"] = {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "created_at": manifest.get("created_at"),
        "manifest_path": manifest.get("manifest_path"),
        "release_dir": release_dir.as_posix(),
        "score_weights": (manifest.get("score_fusion") or {}).get("weights") or {},
        "gating": manifest.get("gating") or {},
    }

    score_weights = (manifest.get("score_fusion") or {}).get("weights") or {}
    if isinstance(score_weights, dict) and score_weights:
        scoring_cfg = effective.setdefault("scoring", {})
        scoring_cfg["release_score_weights"] = score_weights

    return effective, {
        "enabled": True,
        "policy": policy,
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "created_at": manifest.get("created_at"),
        "manifest_path": manifest.get("manifest_path"),
        "release_dir": release_dir.as_posix(),
        "default_strategy": serving_manifest.get("default_strategy"),
        "score_weights": score_weights,
        "gating": manifest.get("gating") or {},
    }
