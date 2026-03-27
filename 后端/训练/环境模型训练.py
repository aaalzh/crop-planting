from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from 后端.数据加载 import load_config


RAW_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
DEFAULT_ENV_OBJECTIVE_WEIGHTS = {
    "log_loss": 0.45,
    "macro_f1": 0.30,
    "balanced_accuracy": 0.15,
    "top3_accuracy": 0.10,
}


def _to_abs(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def add_engineered_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    eps = 1e-6

    df["N_K_ratio"] = df["N"] / (df["K"] + eps)
    df["N_P_ratio"] = df["N"] / (df["P"] + eps)
    df["P_K_ratio"] = df["P"] / (df["K"] + eps)

    df["npk_sum"] = df["N"] + df["P"] + df["K"]
    df["soil_index"] = df["npk_sum"] / 3.0
    df["npk_std"] = df[["N", "P", "K"]].std(axis=1)

    df["ph_neutral_dist"] = (df["ph"] - 7.0).abs()
    df["heat_stress"] = df["temperature"] / (df["humidity"] + eps)

    df["rain_humidity_ratio"] = df["rainfall"] / (df["humidity"] + eps)
    df["temp_rain_interaction"] = df["temperature"] * np.log1p(df["rainfall"])
    return df


def resolve_env_training_csv(env_dir: Path) -> Path:
    candidates = []
    for path in sorted(env_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        cols = set(df.columns)
        if "label" not in cols:
            continue
        if not all(col in cols for col in RAW_FEATURES):
            continue
        candidates.append((len(df), path))

    if not candidates:
        raise FileNotFoundError(f"no training csv found under {env_dir}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _normalize_prob(y_prob: np.ndarray) -> np.ndarray:
    arr = np.asarray(y_prob, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    row_sum = np.clip(arr.sum(axis=1, keepdims=True), 1e-12, None)
    arr = arr / row_sum
    if arr.ndim == 2 and arr.shape[1] >= 2:
        arr[:, -1] = np.clip(1.0 - arr[:, :-1].sum(axis=1), 0.0, 1.0)
        arr = arr / np.clip(arr.sum(axis=1, keepdims=True), 1e-12, None)
    return arr


def _multiclass_log_loss(y_true: pd.Series, y_prob: np.ndarray, classes: List[str]) -> float:
    y_prob = _normalize_prob(y_prob)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    y_prob = y_prob / np.clip(y_prob.sum(axis=1, keepdims=True), 1e-12, None)
    class_to_idx = {str(label): idx for idx, label in enumerate(classes)}
    target_idx = np.asarray([class_to_idx[str(label)] for label in y_true], dtype=int)
    picked = y_prob[np.arange(len(target_idx)), target_idx]
    return float(np.mean(-np.log(np.clip(picked, 1e-15, 1.0))))


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value in [None, "", "none", "None", 0, "0", "null", "Null"]:
        return None
    return int(value)


def _coerce_max_features(value: Any) -> Optional[Union[str, float, int]]:
    text = str(value).strip().lower()
    if value is None or text in {"", "none", "null"}:
        return None
    if text in {"sqrt", "log2"}:
        return text
    try:
        num = float(value)
    except Exception:
        return value
    if abs(num - round(num)) <= 1e-12 and num > 1.0:
        return int(round(num))
    return num


def _coerce_class_weight(value: Any) -> Optional[Union[str, dict]]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return value


def _normalize_objective_weights(raw: Any) -> Dict[str, float]:
    cfg = raw if isinstance(raw, dict) else {}
    out: Dict[str, float] = {}
    for key, default_value in DEFAULT_ENV_OBJECTIVE_WEIGHTS.items():
        try:
            out[key] = max(0.0, float(cfg.get(key, default_value)))
        except Exception:
            out[key] = float(default_value)
    total = float(sum(out.values()))
    if total <= 1e-12:
        return dict(DEFAULT_ENV_OBJECTIVE_WEIGHTS)
    return {k: float(v / total) for k, v in out.items()}


def _resolve_env_model_params(env_cfg: dict) -> Dict[str, Any]:
    return {
        "n_estimators": int(env_cfg.get("n_estimators", 1374)),
        "max_depth": _coerce_optional_int(env_cfg.get("max_depth", 30)),
        "min_samples_split": max(2, int(env_cfg.get("min_samples_split", 11))),
        "min_samples_leaf": max(1, int(env_cfg.get("min_samples_leaf", 1))),
        "max_features": _coerce_max_features(env_cfg.get("max_features", None)),
        "class_weight": _coerce_class_weight(env_cfg.get("class_weight", "balanced")),
    }


def _resolve_search_cfg(env_cfg: dict, random_state: int) -> Dict[str, Any]:
    raw = env_cfg.get("search", {})
    cfg = raw if isinstance(raw, dict) else {}

    trials = max(1, int(cfg.get("trials", 16)))
    n_estimators_choices = [int(x) for x in cfg.get("n_estimators_choices", [96, 128, 192, 256, 320, 480, 640, 800, 1000, 1200, 1400])]
    max_depth_choices = [_coerce_optional_int(x) for x in cfg.get("max_depth_choices", [None, 8, 12, 16, 20, 24, 30, 36])]
    min_samples_split_choices = [max(2, int(x)) for x in cfg.get("min_samples_split_choices", [2, 4, 6, 8, 10, 12, 16, 20])]
    min_samples_leaf_choices = [max(1, int(x)) for x in cfg.get("min_samples_leaf_choices", [1, 2, 3, 4, 6, 8])]
    max_features_choices = [_coerce_max_features(x) for x in cfg.get("max_features_choices", [None, "sqrt", "log2", 0.4, 0.6, 0.8, 1.0])]

    return {
        "enabled": bool(cfg.get("enabled", True)),
        "trials": trials,
        "coarse_trials": max(1, int(cfg.get("coarse_trials", trials))),
        "fine_trials": max(0, int(cfg.get("fine_trials", 0))),
        "fine_neighbors": max(1, int(cfg.get("fine_neighbors", 1))),
        "cv_folds": max(3, int(cfg.get("cv_folds", 5))),
        "seed": int(cfg.get("seed", random_state + 101)),
        "objective_weights": _normalize_objective_weights(cfg.get("objective_weights", {})),
        "n_estimators_choices": list(dict.fromkeys(n_estimators_choices)),
        "max_depth_choices": list(dict.fromkeys(max_depth_choices)),
        "min_samples_split_choices": list(dict.fromkeys(min_samples_split_choices)),
        "min_samples_leaf_choices": list(dict.fromkeys(min_samples_leaf_choices)),
        "max_features_choices": list(dict.fromkeys(max_features_choices)),
    }


def _topk_accuracy_from_proba(y_true: pd.Series, y_prob: np.ndarray, classes: List[str], k: int) -> float:
    if y_prob.size == 0:
        return 0.0
    class_to_idx = {str(label): idx for idx, label in enumerate(classes)}
    target_idx = np.asarray([class_to_idx[str(label)] for label in y_true], dtype=int)
    top_idx = np.argsort(y_prob, axis=1)[:, -max(1, min(int(k), len(classes))):]
    hit = np.any(top_idx == target_idx.reshape(-1, 1), axis=1)
    return float(np.mean(hit))


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray, classes: List[str]) -> Dict[str, float]:
    y_prob = _normalize_prob(y_prob)
    class_count = max(2, len(classes))
    log_loss_value = _multiclass_log_loss(y_true, y_prob, classes=classes)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "log_loss": log_loss_value,
        "log_loss_norm": float(log_loss_value / np.log(class_count)),
        "top3_accuracy": _topk_accuracy_from_proba(y_true, y_prob, classes, k=3),
        "top5_accuracy": _topk_accuracy_from_proba(y_true, y_prob, classes, k=5),
    }


def _mean_metric_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    out: Dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            out[key] = float(np.mean(values))
    return out


def _objective_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    return float(
        weights["log_loss"] * float(metrics.get("log_loss_norm", 1.0))
        + weights["macro_f1"] * (1.0 - float(metrics.get("macro_f1", 0.0)))
        + weights["balanced_accuracy"] * (1.0 - float(metrics.get("balanced_accuracy", 0.0)))
        + weights["top3_accuracy"] * (1.0 - float(metrics.get("top3_accuracy", 0.0)))
    )


def _make_env_model(params: Dict[str, Any], n_jobs: int, random_state: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        class_weight=params["class_weight"],
        n_jobs=n_jobs,
        random_state=random_state,
    )


def _cv_metrics_for_params(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, Any],
    search_cfg: Dict[str, Any],
    n_jobs: int,
    random_state: int,
) -> Tuple[Dict[str, float], int]:
    min_class_count = int(y.astype(str).value_counts().min())
    cv_folds = min(int(search_cfg["cv_folds"]), max(2, min_class_count))
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=int(search_cfg["seed"]))
    classes = sorted(y.astype(str).unique().tolist())
    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        model = _make_env_model(params, n_jobs=n_jobs, random_state=random_state + fold_idx * 17)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)
        fold_metrics.append(_classification_metrics(y_val, y_pred, y_prob, classes=classes))

    metrics = _mean_metric_rows(fold_metrics)
    metrics["objective_score"] = _objective_score(metrics, search_cfg["objective_weights"])
    return metrics, cv_folds


def _sample_search_params(
    rng: np.random.RandomState,
    baseline_params: Dict[str, Any],
    search_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "n_estimators": int(rng.choice(search_cfg["n_estimators_choices"])),
        "max_depth": search_cfg["max_depth_choices"][int(rng.randint(len(search_cfg["max_depth_choices"])))],
        "min_samples_split": int(rng.choice(search_cfg["min_samples_split_choices"])),
        "min_samples_leaf": int(rng.choice(search_cfg["min_samples_leaf_choices"])),
        "max_features": search_cfg["max_features_choices"][int(rng.randint(len(search_cfg["max_features_choices"])))],
        "class_weight": baseline_params["class_weight"],
    }


def _params_key(params: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        params["n_estimators"],
        params["max_depth"],
        params["min_samples_split"],
        params["min_samples_leaf"],
        params["max_features"],
        params["class_weight"],
    )


def _value_equals(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, str) or isinstance(right, str):
        return str(left) == str(right)
    try:
        return abs(float(left) - float(right)) <= 1e-12
    except Exception:
        return left == right


def _find_choice_index(choices: List[Any], value: Any) -> Optional[int]:
    if not choices:
        return None
    for idx, choice in enumerate(choices):
        if _value_equals(choice, value):
            return idx
    if value is None:
        return None
    numeric_positions: List[Tuple[int, float]] = []
    for idx, choice in enumerate(choices):
        if choice is None:
            continue
        try:
            numeric_positions.append((idx, float(choice)))
        except Exception:
            continue
    if not numeric_positions:
        return None
    try:
        target = float(value)
    except Exception:
        return None
    return min(numeric_positions, key=lambda item: abs(item[1] - target))[0]


def _unique_preserve(values: List[Any]) -> List[Any]:
    out: List[Any] = []
    for value in values:
        if any(_value_equals(value, existing) for existing in out):
            continue
        out.append(value)
    return out


def _numeric_boundary_extension(
    choices: List[Any],
    best_value: Any,
    *,
    extend_low: bool,
    integral: bool,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Optional[Any]:
    if best_value is None:
        return None

    numeric_positions: List[Tuple[int, float]] = []
    for idx, choice in enumerate(choices):
        if choice is None:
            continue
        try:
            numeric_positions.append((idx, float(choice)))
        except Exception:
            continue
    if not numeric_positions:
        return None
    best_num = float(best_value)
    numeric_idx = None
    numeric_values = [value for _, value in numeric_positions]
    for pos, (_, value) in enumerate(numeric_positions):
        if _value_equals(value, best_num):
            numeric_idx = pos
            break
    if numeric_idx is None:
        return None
    if extend_low and numeric_idx != 0:
        return None
    if not extend_low and numeric_idx != len(numeric_positions) - 1:
        return None

    if extend_low:
        ref = numeric_values[1] if len(numeric_values) >= 2 else best_num
        delta = abs(ref - best_num)
        candidate = best_num - delta if delta > 1e-12 else best_num * 0.75
    else:
        ref = numeric_values[-2] if len(numeric_values) >= 2 else best_num
        delta = abs(best_num - ref)
        candidate = best_num + delta if delta > 1e-12 else best_num * 1.25

    if min_value is not None:
        candidate = max(float(min_value), candidate)
    if max_value is not None:
        candidate = min(float(max_value), candidate)

    if integral:
        candidate = int(round(candidate))
        if min_value is not None:
            candidate = max(int(round(min_value)), candidate)
        if max_value is not None:
            candidate = min(int(round(max_value)), candidate)
    if _value_equals(candidate, best_value):
        return None
    return candidate


def _refine_choice_space(
    choices: List[Any],
    best_value: Any,
    *,
    neighbors: int,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    integral: bool = False,
    extend_boundary: bool = True,
) -> List[Any]:
    if not choices:
        return [best_value]
    idx = _find_choice_index(choices, best_value)
    if idx is None:
        idx = 0
    lo = max(0, idx - neighbors)
    hi = min(len(choices), idx + neighbors + 1)
    local = list(choices[lo:hi])
    if not any(_value_equals(best_value, item) for item in local):
        local.append(best_value)
    if extend_boundary:
        ext_low = _numeric_boundary_extension(
            choices,
            best_value,
            extend_low=True,
            integral=integral,
            min_value=min_value,
            max_value=max_value,
        )
        ext_high = _numeric_boundary_extension(
            choices,
            best_value,
            extend_low=False,
            integral=integral,
            min_value=min_value,
            max_value=max_value,
        )
        if ext_low is not None:
            local.append(ext_low)
        if ext_high is not None:
            local.append(ext_high)
    return _unique_preserve(local)


def _build_fine_search_space(best_params: Dict[str, Any], search_cfg: Dict[str, Any]) -> Dict[str, List[Any]]:
    neighbors = int(search_cfg["fine_neighbors"])
    return {
        "n_estimators_choices": _refine_choice_space(
            list(search_cfg["n_estimators_choices"]),
            best_params["n_estimators"],
            neighbors=neighbors,
            min_value=32,
            max_value=4096,
            integral=True,
            extend_boundary=True,
        ),
        "max_depth_choices": _refine_choice_space(
            list(search_cfg["max_depth_choices"]),
            best_params["max_depth"],
            neighbors=neighbors,
            min_value=4,
            max_value=128,
            integral=True,
            extend_boundary=True,
        ),
        "min_samples_split_choices": _refine_choice_space(
            list(search_cfg["min_samples_split_choices"]),
            best_params["min_samples_split"],
            neighbors=neighbors,
            min_value=2,
            max_value=64,
            integral=True,
            extend_boundary=True,
        ),
        "min_samples_leaf_choices": _refine_choice_space(
            list(search_cfg["min_samples_leaf_choices"]),
            best_params["min_samples_leaf"],
            neighbors=neighbors,
            min_value=1,
            max_value=32,
            integral=True,
            extend_boundary=True,
        ),
        "max_features_choices": _refine_choice_space(
            list(search_cfg["max_features_choices"]),
            best_params["max_features"],
            neighbors=neighbors,
            extend_boundary=False,
        ),
    }


def _candidate_distance(candidate: Dict[str, Any], best_params: Dict[str, Any], fine_space: Dict[str, List[Any]]) -> float:
    score = 0.0
    for key in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"):
        choice_key = f"{key}_choices"
        choices = list(fine_space.get(choice_key, []))
        best_idx = _find_choice_index(choices, best_params.get(key))
        cur_idx = _find_choice_index(choices, candidate.get(key))
        if best_idx is not None and cur_idx is not None:
            score += abs(cur_idx - best_idx)
            continue
        if candidate.get(key) is None or best_params.get(key) is None:
            score += 1.0
            continue
        try:
            score += abs(float(candidate.get(key)) - float(best_params.get(key)))
        except Exception:
            score += 1.0 if not _value_equals(candidate.get(key), best_params.get(key)) else 0.0
    return float(score)


def _generate_fine_candidates(
    best_params: Dict[str, Any],
    fine_space: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    grids = [
        list(fine_space.get("n_estimators_choices", [best_params["n_estimators"]])),
        list(fine_space.get("max_depth_choices", [best_params["max_depth"]])),
        list(fine_space.get("min_samples_split_choices", [best_params["min_samples_split"]])),
        list(fine_space.get("min_samples_leaf_choices", [best_params["min_samples_leaf"]])),
        list(fine_space.get("max_features_choices", [best_params["max_features"]])),
    ]
    out: List[Dict[str, Any]] = []
    for n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features in product(*grids):
        out.append(
            {
                "n_estimators": int(n_estimators),
                "max_depth": max_depth,
                "min_samples_split": int(min_samples_split),
                "min_samples_leaf": int(min_samples_leaf),
                "max_features": max_features,
                "class_weight": best_params["class_weight"],
            }
        )
    out.sort(key=lambda item: (_candidate_distance(item, best_params, fine_space), repr(_params_key(item))))
    return out


def _search_best_params(
    X: pd.DataFrame,
    y: pd.Series,
    baseline_params: Dict[str, Any],
    search_cfg: Dict[str, Any],
    n_jobs: int,
    random_state: int,
) -> Tuple[Dict[str, Any], Optional[Dict[str, float]], List[Dict[str, Any]], int, Dict[str, Any]]:
    if not search_cfg["enabled"]:
        return dict(baseline_params), None, [], 0, {
            "strategy": "fixed_config",
            "coarse_trials_requested": 0,
            "coarse_trials_run": 0,
            "fine_trials_requested": 0,
            "fine_trials_run": 0,
            "fine_search_space": {},
        }

    rng = np.random.RandomState(int(search_cfg["seed"]))
    history: List[Dict[str, Any]] = []
    seen = set()
    best_params = dict(baseline_params)
    best_metrics: Optional[Dict[str, float]] = None
    actual_cv_folds = 0
    coarse_trials_requested = max(1, int(search_cfg.get("coarse_trials", search_cfg["trials"])))
    fine_trials_requested = max(0, int(search_cfg.get("fine_trials", 0)))
    coarse_trials_run = 0
    fine_trials_run = 0
    fine_search_space: Dict[str, Any] = {}

    def _evaluate(params: Dict[str, Any], *, stage: str, trial_in_stage: int, source: str) -> None:
        nonlocal best_params, best_metrics, actual_cv_folds
        seen.add(_params_key(params))
        metrics, cv_folds = _cv_metrics_for_params(
            X=X,
            y=y,
            params=params,
            search_cfg=search_cfg,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        actual_cv_folds = cv_folds
        history.append(
            {
                "trial": int(len(history)),
                "trial_in_stage": int(trial_in_stage),
                "stage": stage,
                "source": source,
                "params": dict(params),
                **metrics,
            }
        )
        if best_metrics is None or float(metrics["objective_score"]) < float(best_metrics["objective_score"]):
            best_params = dict(params)
            best_metrics = dict(metrics)

    for trial in range(coarse_trials_requested):
        if trial == 0:
            params = dict(baseline_params)
            source = "baseline_config"
        else:
            params = _sample_search_params(rng, baseline_params, search_cfg)
            for _ in range(20):
                if _params_key(params) not in seen:
                    break
                params = _sample_search_params(rng, baseline_params, search_cfg)
            source = "coarse_random"
        if _params_key(params) in seen:
            continue
        _evaluate(params, stage="coarse", trial_in_stage=trial, source=source)
        coarse_trials_run += 1

    if fine_trials_requested > 0 and best_metrics is not None:
        fine_search_space = _build_fine_search_space(best_params, search_cfg)
        fine_candidates = _generate_fine_candidates(best_params, fine_search_space)
        for trial, params in enumerate(fine_candidates):
            if fine_trials_run >= fine_trials_requested:
                break
            if _params_key(params) in seen:
                continue
            _evaluate(params, stage="fine", trial_in_stage=trial, source="fine_local_grid")
            fine_trials_run += 1

    search_meta = {
        "strategy": "coarse_to_fine_stratified_cv" if fine_trials_run > 0 else "random_search_stratified_cv",
        "coarse_trials_requested": int(coarse_trials_requested),
        "coarse_trials_run": int(coarse_trials_run),
        "fine_trials_requested": int(fine_trials_requested),
        "fine_trials_run": int(fine_trials_run),
        "fine_search_space": {key: list(values) for key, values in fine_search_space.items()},
    }
    return best_params, best_metrics, history, actual_cv_folds, search_meta


def build_bundle(config_path: str = "后端/配置.yaml") -> dict:
    cfg = load_config(str(_to_abs(config_path)))
    paths = cfg["paths"]
    env_cfg = cfg.get("model", {}).get("env", {})
    env_dir = _to_abs(paths["env_model_dir"])
    bundle_path = _to_abs(paths["env_model_bundle"])
    data_path = resolve_env_training_csv(env_dir)

    df = pd.read_csv(data_path)
    df = df.dropna(subset=RAW_FEATURES + ["label"]).copy()
    for col in RAW_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=RAW_FEATURES + ["label"]).copy()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"] != ""].copy()
    if df.empty:
        raise ValueError("environment training data is empty after cleaning")

    X_raw = df[RAW_FEATURES].copy()
    y = df["label"].copy()
    X_feat = add_engineered_features(X_raw)
    feature_order = list(X_feat.columns)

    test_size = float(env_cfg.get("test_size", 0.2))
    test_size = min(max(test_size, 0.05), 0.5)
    random_state = int(env_cfg.get("random_state", 42))
    n_jobs = int(env_cfg.get("n_jobs", 1))
    base_params = _resolve_env_model_params(env_cfg)
    search_cfg = _resolve_search_cfg(env_cfg, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat[feature_order],
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    best_params, cv_metrics, search_history, actual_cv_folds, search_meta = _search_best_params(
        X=X_train,
        y=y_train,
        baseline_params=base_params,
        search_cfg=search_cfg,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    eval_model = _make_env_model(best_params, n_jobs=n_jobs, random_state=random_state)
    eval_model.fit(X_train, y_train)
    y_pred = eval_model.predict(X_test)
    y_prob = eval_model.predict_proba(X_test)
    holdout_metrics = _classification_metrics(y_test, y_pred, y_prob, classes=list(eval_model.classes_))

    model = _make_env_model(best_params, n_jobs=n_jobs, random_state=random_state)
    model.fit(X_feat[feature_order], y)

    train_raw_stats = {
        "min": {col: float(X_raw[col].min()) for col in RAW_FEATURES},
        "max": {col: float(X_raw[col].max()) for col in RAW_FEATURES},
        "mean": {col: float(X_raw[col].mean()) for col in RAW_FEATURES},
        "std": {col: float(X_raw[col].std(ddof=0)) for col in RAW_FEATURES},
    }

    selection = {
        "strategy": str(search_meta.get("strategy", "fixed_config")),
        "search_enabled": bool(search_cfg["enabled"]),
        "search_trials": int(search_cfg["trials"]),
        "coarse_trials": int(search_cfg["coarse_trials"]),
        "fine_trials": int(search_cfg["fine_trials"]),
        "coarse_trials_run": int(search_meta.get("coarse_trials_run", 0)),
        "fine_trials_run": int(search_meta.get("fine_trials_run", 0)),
        "fine_neighbors": int(search_cfg["fine_neighbors"]),
        "cv_folds": int(actual_cv_folds),
        "search_seed": int(search_cfg["seed"]),
        "objective_weights": dict(search_cfg["objective_weights"]),
        "best_objective_score": None if cv_metrics is None else float(cv_metrics.get("objective_score", 0.0)),
        "search_space": {
            "n_estimators_choices": list(search_cfg["n_estimators_choices"]),
            "max_depth_choices": list(search_cfg["max_depth_choices"]),
            "min_samples_split_choices": list(search_cfg["min_samples_split_choices"]),
            "min_samples_leaf_choices": list(search_cfg["min_samples_leaf_choices"]),
            "max_features_choices": list(search_cfg["max_features_choices"]),
        },
        "fine_search_space": dict(search_meta.get("fine_search_space", {})),
    }

    bundle = {
        "model": model,
        "meta": {
            "created_at_unix": int(time.time()),
            "random_state": random_state,
            "raw_features": RAW_FEATURES,
            "feature_order": feature_order,
            "classes": [str(item) for item in model.classes_],
            "test_metrics": holdout_metrics,
            "holdout_metrics": holdout_metrics,
            "cv_metrics": cv_metrics,
            "selected_params": {
                **best_params,
                "n_jobs": n_jobs,
                "random_state": random_state,
            },
            "selection": selection,
            "search_history": search_history,
            "holdout_split": {
                "test_size": float(test_size),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
            },
            "train_raw_stats": train_raw_stats,
            "train_file": data_path.as_posix(),
        },
    }

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    dump(bundle, bundle_path)
    return {
        "bundle_path": bundle_path.as_posix(),
        "metrics": holdout_metrics,
        "holdout_metrics": holdout_metrics,
        "cv_metrics": cv_metrics,
        "best_params": {
            **best_params,
            "n_jobs": n_jobs,
            "random_state": random_state,
        },
        "selection": selection,
        "train_file": data_path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export environment recommendation model bundle")
    parser.add_argument("--config", default="后端/配置.yaml")
    args = parser.parse_args()
    out = build_bundle(args.config)
    print("saved:", out["bundle_path"])
    print("train_file:", out["train_file"])
    print("best_params:", out["best_params"])
    print("holdout_metrics:", out["holdout_metrics"])
    print("cv_metrics:", out["cv_metrics"])


if __name__ == "__main__":
    main()
