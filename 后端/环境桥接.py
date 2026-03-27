from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _to_abs(root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (root / path).resolve()


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


def _safe_read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


def _safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_env_dataset_path(root: Path, config: dict) -> Path:
    env_cfg = config.get("env_bridge", {})
    text = str(env_cfg.get("dataset_file", "")).strip()
    if text:
        return _to_abs(root, text)

    paths_cfg = config.get("paths", {})
    env_dir = str(paths_cfg.get("env_model_dir", "环境推荐/数据")).strip() or "环境推荐/数据"
    return _to_abs(root, Path(env_dir) / "作物推荐数据.csv")


def resolve_env_scenario_path(root: Path, config: dict) -> Path:
    env_cfg = config.get("env_bridge", {})
    text = str(env_cfg.get("scenario_file", "数据/样例/环境场景库.json")).strip()
    return _to_abs(root, text)


def _load_env_module(root: Path, config: dict):
    import importlib.util

    paths_cfg = config.get("paths", {})
    predict_py = str(paths_cfg.get("env_predict_py", "环境推荐/脚本/环境预测.py")).strip()
    path = _to_abs(root, predict_py)
    spec = importlib.util.spec_from_file_location("env_predict_bridge", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import env predictor: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_env_bundle_path(root: Path, config: dict) -> Path:
    paths_cfg = config.get("paths", {})
    text = str(paths_cfg.get("env_model_bundle", "环境推荐/模型/作物推荐模型管道.pkl")).strip()
    return _to_abs(root, text)


def _topk_for_row(classes: List[str], probs: np.ndarray, k: int) -> List[dict]:
    idx = np.argsort(probs)[::-1][: max(1, int(k))]
    return [{"crop": str(classes[i]), "prob": float(probs[i])} for i in idx]


def _probability_map(classes: List[str], probs: np.ndarray) -> Dict[str, float]:
    return {str(classes[i]): float(probs[i]) for i in range(len(classes))}


def build_env_scenario_library(root: Path, config: dict, *, save: bool = True) -> dict:
    dataset_path = resolve_env_dataset_path(root, config)
    if not dataset_path.exists():
        raise FileNotFoundError(f"env dataset not found: {dataset_path.as_posix()}")

    env_cfg = config.get("env_bridge", {})
    scenario_per_crop = max(1, int(env_cfg.get("scenarios_per_crop", 2)))
    topk = max(1, int(env_cfg.get("topk", 3)))

    env_mod = _load_env_module(root, config)
    bundle = env_mod.load_bundle(_resolve_env_bundle_path(root, config))
    model = bundle["model"]
    meta = bundle["meta"]
    raw_features = list(meta["raw_features"])
    feature_order = list(meta["feature_order"])
    train_stats = copy.deepcopy(meta.get("train_raw_stats", {}))

    df = pd.read_csv(dataset_path)
    required = [col for col in raw_features if col not in df.columns]
    if required:
        raise ValueError(f"env dataset missing columns: {required}")

    source = df[raw_features + [c for c in ["label"] if c in df.columns]].copy()
    for col in raw_features:
        source[col] = pd.to_numeric(source[col], errors="coerce")
    source = source.dropna(subset=raw_features).reset_index(drop=True)
    if source.empty:
        raise ValueError("env dataset is empty after cleaning")

    feat = env_mod.add_engineered_features(source[raw_features].copy())[feature_order]
    proba = model.predict_proba(feat)
    classes = [str(c) for c in list(model.classes_)]

    for idx, cls in enumerate(classes):
        source[f"prob_{cls}"] = proba[:, idx]
    source["best_label"] = [classes[int(np.argmax(row))] for row in proba]
    source["best_prob"] = [float(np.max(row)) for row in proba]
    source["confidence"] = [env_mod.confidence_level(float(np.max(row))) for row in proba]

    def _select_rows(group: pd.DataFrame, label: str) -> List[Tuple[str, pd.Series]]:
        if group.empty:
            return []
        prob_col = f"prob_{label}"
        sorted_group = group.sort_values(prob_col, ascending=False).reset_index(drop=True)
        picks: List[Tuple[str, pd.Series]] = [("anchor", sorted_group.iloc[0])]
        if scenario_per_crop >= 2 and len(sorted_group) > 1:
            picks.append(("median", sorted_group.iloc[len(sorted_group) // 2]))
        if scenario_per_crop >= 3 and len(sorted_group) > 2:
            picks.append(("tail", sorted_group.iloc[-1]))
        return picks[:scenario_per_crop]

    scenarios: List[dict] = []
    for label in classes:
        preferred = source[source["best_label"] == label].copy()
        if preferred.empty:
            preferred = source.sort_values(f"prob_{label}", ascending=False).head(max(3, scenario_per_crop)).copy()
        for role, row in _select_rows(preferred, label):
            raw_input = {col: float(row[col]) for col in raw_features}
            warnings = env_mod.ood_warnings(pd.DataFrame([raw_input]), train_stats)
            probs = np.asarray([float(row[f"prob_{cls}"]) for cls in classes], dtype=float)
            scenarios.append(
                {
                    "scenario_id": f"{label}_{role}",
                    "label": label,
                    "role": role,
                    "env_input": raw_input,
                    "best_label": str(row["best_label"]),
                    "best_prob": float(row["best_prob"]),
                    "label_prob": float(row.get(f"prob_{label}", 0.0)),
                    "confidence": str(row["confidence"]),
                    "warnings": warnings,
                    "topk": _topk_for_row(classes, probs, topk),
                    "probabilities": _probability_map(classes, probs),
                }
            )

    payload = {
        "created_at": pd.Timestamp.now().isoformat(),
        "dataset_path": dataset_path.as_posix(),
        "scenario_file": resolve_env_scenario_path(root, config).as_posix(),
        "scenario_count": len(scenarios),
        "scenarios_per_crop": scenario_per_crop,
        "topk": topk,
        "items": scenarios,
    }

    if save:
        _safe_write_json(resolve_env_scenario_path(root, config), payload)
    return payload


def load_env_scenario_library(root: Path, config: dict, *, rebuild_if_missing: bool = False) -> dict:
    path = resolve_env_scenario_path(root, config)
    payload = _safe_read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return payload
    if not rebuild_if_missing:
        return {
            "created_at": None,
            "dataset_path": resolve_env_dataset_path(root, config).as_posix(),
            "scenario_file": path.as_posix(),
            "scenario_count": 0,
            "items": [],
        }
    return build_env_scenario_library(root, config, save=True)
