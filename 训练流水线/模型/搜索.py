from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np


def sample_params(model_name: str, rng: np.random.RandomState) -> Dict[str, object]:
    name = str(model_name).strip().lower()
    if name == "ridge":
        return {"alpha": float(10.0 ** rng.uniform(-3.0, 2.0))}

    if name == "hgb":
        return {
            "max_iter": int(rng.randint(180, 620)),
            "learning_rate": float(rng.uniform(0.02, 0.15)),
            "max_depth": int(rng.choice([3, 4, 5, 6, 8])),
            "min_samples_leaf": int(rng.randint(5, 45)),
            "l2_regularization": float(rng.uniform(0.0, 0.2)),
        }

    if name == "rf":
        return {
            "n_estimators": int(rng.randint(240, 900)),
            "max_depth": None if rng.rand() < 0.4 else int(rng.choice([4, 6, 8, 10, 12])),
            "min_samples_leaf": int(rng.randint(1, 6)),
            "max_features": rng.choice(["sqrt", "log2", 0.6, 0.8, 1.0]),
        }

    if name == "etr":
        return {
            "n_estimators": int(rng.randint(260, 1000)),
            "max_depth": None if rng.rand() < 0.5 else int(rng.choice([4, 6, 8, 10, 12])),
            "min_samples_leaf": int(rng.randint(1, 5)),
            "max_features": rng.choice(["sqrt", "log2", 0.6, 0.8, 1.0]),
        }

    raise ValueError(f"unsupported model: {model_name}")


def random_search(
    model_name: str,
    objective_fn: Callable[[Dict[str, object]], Tuple[float, Dict[str, float]]],
    trials: int,
    seed: int,
) -> Tuple[Dict[str, object], float, List[Dict[str, object]]]:
    rng = np.random.RandomState(seed)
    best_params: Dict[str, object] = {}
    best_score = float("inf")
    history: List[Dict[str, object]] = []

    for trial in range(int(trials)):
        params = sample_params(model_name, rng)
        score, stats = objective_fn(params)
        row = {
            "trial": int(trial),
            "model": model_name,
            "score": float(score),
            "params": params,
        }
        row.update({k: float(v) for k, v in stats.items()})
        history.append(row)
        if float(score) < best_score:
            best_score = float(score)
            best_params = dict(params)

    if not best_params:
        best_params = sample_params(model_name, rng)
    return best_params, best_score, history
