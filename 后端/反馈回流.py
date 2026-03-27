from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if out != out:
            return None
        return out
    except Exception:
        return None


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _to_abs(root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (root / path).resolve()


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


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _collect_fieldnames(rows: List[dict]) -> List[str]:
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            fieldnames.append(key)
            seen.add(key)
    return fieldnames


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _collect_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as fh:
        if not fieldnames:
            return
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _feedback_paths(root: Path, config: dict) -> Dict[str, Path]:
    feedback_cfg = config.get("feedback", {}) if isinstance(config, dict) else {}
    return {
        "inference_log_file": _to_abs(root, str(feedback_cfg.get("inference_log_file", "输出/闭环/推理事件.jsonl"))),
        "feedback_log_file": _to_abs(root, str(feedback_cfg.get("feedback_log_file", "输出/闭环/用户反馈.jsonl"))),
        "training_sample_file": _to_abs(root, str(feedback_cfg.get("training_sample_file", "输出/闭环/反馈训练样本.jsonl"))),
        "training_sample_csv": _to_abs(root, str(feedback_cfg.get("training_sample_csv", "输出/闭环/反馈训练样本.csv"))),
        "training_summary_file": _to_abs(root, str(feedback_cfg.get("training_summary_file", "输出/闭环/反馈训练摘要.json"))),
    }


def _find_result_by_crop(results: List[dict], crop: str | None) -> dict | None:
    target = str(crop or "").strip().lower()
    if not target:
        return None
    for row in results:
        if str(row.get("crop") or "").strip().lower() == target:
            return row
    return None


def _result_snapshot(row: dict | None, prefix: str) -> dict:
    if not isinstance(row, dict):
        return {
            f"{prefix}_crop": None,
            f"{prefix}_env_prob": None,
            f"{prefix}_profit": None,
            f"{prefix}_price_pred": None,
            f"{prefix}_yield": None,
            f"{prefix}_cost_pred": None,
            f"{prefix}_risk": None,
            f"{prefix}_uncertainty": None,
            f"{prefix}_score": None,
            f"{prefix}_target_year": None,
        }
    return {
        f"{prefix}_crop": row.get("crop"),
        f"{prefix}_env_prob": _safe_float(row.get("env_prob")),
        f"{prefix}_profit": _safe_float(row.get("profit")),
        f"{prefix}_price_pred": _safe_float(row.get("price_pred")),
        f"{prefix}_yield": _safe_float(row.get("yield")),
        f"{prefix}_cost_pred": _safe_float(row.get("cost_pred")),
        f"{prefix}_risk": _safe_float(row.get("risk")),
        f"{prefix}_uncertainty": _safe_float(row.get("uncertainty")),
        f"{prefix}_score": _safe_float(row.get("score")),
        f"{prefix}_target_year": row.get("target_year"),
    }


def _selection_rank(results: List[dict], selected_crop: str | None) -> int | None:
    target = str(selected_crop or "").strip().lower()
    if not target:
        return None
    for idx, row in enumerate(results, start=1):
        crop = str(row.get("crop") or "").strip().lower()
        if crop == target:
            return idx
    return None


def _topk_names(inference: dict, results: List[dict]) -> List[str]:
    final_topk = inference.get("final_topk") or []
    names: List[str] = []
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
    return [str(row.get("crop") or "").strip().lower() for row in results if isinstance(row, dict) and str(row.get("crop") or "").strip()]


def _feedback_label(selected_crop: str | None, recommended_crop: str | None, accepted: bool | None, has_actuals: bool) -> str:
    if accepted is True and selected_crop and selected_crop == recommended_crop:
        return "accepted_top1"
    if accepted is True and selected_crop:
        return "accepted_other"
    if accepted is True:
        return "accepted"
    if accepted is False and selected_crop:
        return "rejected_with_selection"
    if accepted is False:
        return "rejected"
    if selected_crop and selected_crop == recommended_crop:
        return "selected_top1"
    if selected_crop:
        return "selected_other"
    if has_actuals:
        return "actuals_only"
    return "pending"


def build_feedback_training_dataset(root: Path, config: dict, *, save: bool = True) -> dict:
    root = root.resolve()
    paths = _feedback_paths(root, config)
    inference_rows = _read_jsonl(paths["inference_log_file"])
    feedback_rows = _read_jsonl(paths["feedback_log_file"])

    inference_map: Dict[str, dict] = {}
    for row in inference_rows:
        event_id = str(row.get("event_id") or "").strip()
        if not event_id:
            continue
        inference_map[event_id] = row

    feedback_map: Dict[str, dict] = {}
    for row in feedback_rows:
        event_id = str(row.get("event_id") or "").strip()
        if not event_id:
            continue
        current = feedback_map.get(event_id)
        created_at = str(row.get("created_at") or "")
        current_at = str((current or {}).get("created_at") or "")
        if current is None or created_at >= current_at:
            feedback_map[event_id] = row

    joined_rows: List[dict] = []
    labeled_rows: List[dict] = []
    outcome_rows: List[dict] = []
    label_distribution: Dict[str, int] = {}
    selected_distribution: Dict[str, int] = {}
    release_distribution: Dict[str, int] = {}
    accepted_true_count = 0
    accepted_false_count = 0
    accepted_unknown_count = 0

    for event_id, inference in inference_map.items():
        feedback = feedback_map.get(event_id) or {}
        results = [row for row in (inference.get("results") or []) if isinstance(row, dict)]
        release = inference.get("release") or {}
        env = inference.get("env") or {}
        env_input = inference.get("env_input") or {}
        recommended_result = results[0] if results else None
        recommended_crop = str((recommended_result or {}).get("crop") or "").strip().lower() or None
        topk_names = _topk_names(inference, results)

        selected_crop = str(feedback.get("selected_crop") or "").strip().lower() or None
        accepted = _safe_bool(feedback.get("accepted"))
        has_actuals = any(
            _safe_float(feedback.get(key)) is not None
            for key in ("actual_profit", "actual_price", "actual_yield", "actual_cost")
        )
        label = _feedback_label(selected_crop, recommended_crop, accepted, has_actuals)
        selected_result = _find_result_by_crop(results, selected_crop)
        selected_rank = _selection_rank(results, selected_crop)
        is_training_label = label != "pending"

        release_run_id = str(release.get("run_id") or "").strip() or None
        if release_run_id:
            release_distribution[release_run_id] = release_distribution.get(release_run_id, 0) + 1

        if selected_crop:
            selected_distribution[selected_crop] = selected_distribution.get(selected_crop, 0) + 1

        label_distribution[label] = label_distribution.get(label, 0) + 1
        if accepted is True:
            accepted_true_count += 1
        elif accepted is False:
            accepted_false_count += 1
        else:
            accepted_unknown_count += 1

        row = {
            "event_id": event_id,
            "inference_at": inference.get("created_at"),
            "feedback_at": feedback.get("created_at"),
            "has_feedback": event_id in feedback_map,
            "is_training_label": is_training_label,
            "feedback_label": label,
            "accepted": accepted,
            "selected_crop": selected_crop,
            "selected_rank": selected_rank,
            "selected_in_topk": selected_crop in topk_names if selected_crop else None,
            "recommended_crop": recommended_crop,
            "recommended_rank": 1 if recommended_crop else None,
            "recommended_topk": "|".join(topk_names[:8]),
            "candidate_count": len(results),
            "release_run_id": release_run_id,
            "release_status": release.get("status"),
            "release_manifest_path": release.get("manifest_path"),
            "env_best_label": env.get("best_label"),
            "env_best_prob": _safe_float(env.get("best_prob")),
            "env_confidence": env.get("confidence"),
            "env_risk": env.get("risk"),
            "actual_profit": _safe_float(feedback.get("actual_profit")),
            "actual_price": _safe_float(feedback.get("actual_price")),
            "actual_yield": _safe_float(feedback.get("actual_yield")),
            "actual_cost": _safe_float(feedback.get("actual_cost")),
            "notes": feedback.get("notes"),
            "user_name": (feedback.get("user") or inference.get("user") or {}).get("username"),
            "env_input_json": json.dumps(env_input, ensure_ascii=False, sort_keys=True) if isinstance(env_input, dict) else None,
        }
        for key, value in (env_input.items() if isinstance(env_input, dict) else []):
            row[f"env_{key}"] = _safe_float(value)
        row.update(_result_snapshot(recommended_result, "recommended"))
        row.update(_result_snapshot(selected_result, "selected"))

        joined_rows.append(row)
        if is_training_label:
            labeled_rows.append(row)
        if has_actuals:
            outcome_rows.append(row)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inference_log_file": paths["inference_log_file"].as_posix(),
        "feedback_log_file": paths["feedback_log_file"].as_posix(),
        "training_sample_file": paths["training_sample_file"].as_posix(),
        "training_sample_csv": paths["training_sample_csv"].as_posix(),
        "training_summary_file": paths["training_summary_file"].as_posix(),
        "inference_event_count": len(inference_rows),
        "unique_inference_event_count": len(inference_map),
        "feedback_event_count": len(feedback_rows),
        "matched_feedback_count": sum(1 for row in joined_rows if row.get("has_feedback")),
        "pending_inference_count": sum(1 for row in joined_rows if not row.get("has_feedback")),
        "labeled_sample_count": len(labeled_rows),
        "outcome_sample_count": len(outcome_rows),
        "accepted_true_count": accepted_true_count,
        "accepted_false_count": accepted_false_count,
        "accepted_unknown_count": accepted_unknown_count,
        "latest_inference_at": max((str(row.get("created_at") or "") for row in inference_map.values()), default=None),
        "latest_feedback_at": max((str(row.get("created_at") or "") for row in feedback_map.values()), default=None),
        "label_distribution": label_distribution,
        "selected_crop_distribution": dict(sorted(selected_distribution.items(), key=lambda item: (-item[1], item[0]))),
        "release_distribution": dict(sorted(release_distribution.items(), key=lambda item: (-item[1], item[0]))),
        "preview_event_ids": [row.get("event_id") for row in labeled_rows[:10]],
    }

    if save:
        _write_jsonl(paths["training_sample_file"], labeled_rows)
        _write_csv(paths["training_sample_csv"], labeled_rows)
        _write_json(paths["training_summary_file"], summary)

    return summary


def get_feedback_training_status(root: Path, config: dict, *, refresh: bool = False) -> dict:
    root = root.resolve()
    paths = _feedback_paths(root, config)
    if refresh:
        return build_feedback_training_dataset(root=root, config=config, save=True)

    summary = _read_json(paths["training_summary_file"])
    if isinstance(summary, dict):
        return summary

    return {
        "generated_at": None,
        "inference_log_file": paths["inference_log_file"].as_posix(),
        "feedback_log_file": paths["feedback_log_file"].as_posix(),
        "training_sample_file": paths["training_sample_file"].as_posix(),
        "training_sample_csv": paths["training_sample_csv"].as_posix(),
        "training_summary_file": paths["training_summary_file"].as_posix(),
        "inference_event_count": 0,
        "unique_inference_event_count": 0,
        "feedback_event_count": 0,
        "matched_feedback_count": 0,
        "pending_inference_count": 0,
        "labeled_sample_count": 0,
        "outcome_sample_count": 0,
        "accepted_true_count": 0,
        "accepted_false_count": 0,
        "accepted_unknown_count": 0,
        "latest_inference_at": None,
        "latest_feedback_at": None,
        "label_distribution": {},
        "selected_crop_distribution": {},
        "release_distribution": {},
        "preview_event_ids": [],
    }
