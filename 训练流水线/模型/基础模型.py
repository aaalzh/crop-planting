from __future__ import annotations

from typing import Dict, Optional

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge


def _resolve_max_features(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"sqrt", "log2"}:
            return text
        if text in {"none", "null", ""}:
            return None
        try:
            return float(text)
        except Exception:
            return "sqrt"
    return value


def create_model(model_name: str, params: Dict[str, object], random_state: int = 42, quantile: Optional[float] = None):
    name = str(model_name).strip().lower()
    if name == "ridge":
        return Ridge(alpha=float(params.get("alpha", 1.0)), random_state=random_state)

    if name == "rf":
        max_features = _resolve_max_features(params.get("max_features", "sqrt"))
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=max_features,
            random_state=random_state,
            n_jobs=1,
        )

    if name == "etr":
        max_features = _resolve_max_features(params.get("max_features", "sqrt"))
        return ExtraTreesRegressor(
            n_estimators=int(params.get("n_estimators", 500)),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=max_features,
            random_state=random_state,
            n_jobs=1,
        )

    if name == "gbr_quantile":
        alpha = 0.5 if quantile is None else float(quantile)
        return GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=int(params.get("n_estimators", 350)),
            max_depth=int(params.get("max_depth", 3)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            random_state=random_state,
        )

    loss = "squared_error"
    kwargs = {}
    if quantile is not None:
        loss = "quantile"
        kwargs["quantile"] = float(quantile)

    return HistGradientBoostingRegressor(
        loss=loss,
        max_iter=int(params.get("max_iter", 350)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_depth=params.get("max_depth", 6),
        min_samples_leaf=int(params.get("min_samples_leaf", 10)),
        l2_regularization=float(params.get("l2_regularization", 0.0)),
        random_state=random_state,
        **kwargs,
    )
