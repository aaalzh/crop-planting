from __future__ import annotations

import os
import sys
import types
from typing import Any, Iterable

_ORIG_NUMPY_BITGEN_CTOR = None


def _ensure_numpy_private_core_alias() -> None:
    """
    Bridge NumPy 1.x/2.x pickle module paths.

    Some artifacts saved with newer NumPy reference ``numpy._core`` while older
    runtimes expose ``numpy.core`` only.
    """
    try:
        import numpy as np  # noqa: F401
        import numpy.core as np_core
    except Exception:
        return

    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np_core

    # Mirror already-imported submodules (e.g. numpy.core.multiarray).
    for mod_name, mod_obj in list(sys.modules.items()):
        if not mod_name.startswith("numpy.core."):
            continue
        alias = mod_name.replace("numpy.core.", "numpy._core.", 1)
        if alias not in sys.modules:
            sys.modules[alias] = mod_obj


def _configure_loky_cpu_count() -> None:
    """
    Reduce noisy loky core-detection warnings in restricted environments.

    When physical-core probing fails, setting LOKY_MAX_CPU_COUNT avoids repeated
    warning spam while preserving deterministic single-process behavior.
    """
    if os.environ.get("LOKY_MAX_CPU_COUNT"):
        return
    try:
        cpu_count = max(1, int(os.cpu_count() or 1))
    except Exception:
        cpu_count = 1
    os.environ["LOKY_MAX_CPU_COUNT"] = str(cpu_count)


def _normalize_bitgen_name(raw: Any) -> str:
    if isinstance(raw, str):
        text = raw.strip()
    elif isinstance(raw, type):
        text = getattr(raw, "__name__", str(raw))
    else:
        text = getattr(raw, "__name__", "") or str(raw)
    text = text.strip()
    if text.startswith("<class '") and text.endswith("'>"):
        # "<class 'numpy.random._pcg64.PCG64'>" -> "PCG64"
        text = text[len("<class '") : -len("'>")]
    if "." in text:
        text = text.split(".")[-1]
    return text


def _numpy_bitgenerator_ctor_compat(bit_generator_name: Any = "MT19937"):
    """
    Pickle-safe wrapper for NumPy bit-generator constructor.

    Must stay at module scope (not a local closure), otherwise estimators
    containing random-state internals can fail to serialize.
    """
    global _ORIG_NUMPY_BITGEN_CTOR

    name = _normalize_bitgen_name(bit_generator_name)
    original = _ORIG_NUMPY_BITGEN_CTOR
    if original is not None:
        try:
            return original(name)
        except Exception:
            try:
                return original("MT19937")
            except Exception:
                pass

    # Last-resort direct constructor path.
    try:
        import numpy.random as npr

        ctor = getattr(npr, name, None) or getattr(npr, "MT19937")
        return ctor()
    except Exception as exc:
        raise ValueError(f"{name} is not a known BitGenerator module.") from exc


def _patch_numpy_bitgenerator_ctor() -> None:
    """
    Compat for NumPy pickle payloads that store bit-generator class objects.

    Older NumPy runtimes expect a plain string name in
    ``numpy.random._pickle.__bit_generator_ctor``.
    """
    try:
        import numpy.random._pickle as np_pickle
    except Exception:
        return

    original = getattr(np_pickle, "__bit_generator_ctor", None)
    if original is None:
        return

    if original is _numpy_bitgenerator_ctor_compat:
        return

    global _ORIG_NUMPY_BITGEN_CTOR
    _ORIG_NUMPY_BITGEN_CTOR = original
    np_pickle.__bit_generator_ctor = _numpy_bitgenerator_ctor_compat


def _ensure_sklearn_frozen_module() -> None:
    """
    Provide a runtime shim for sklearn>=1.8 `sklearn.frozen.FrozenEstimator`.

    Old artifacts may reference this module when loaded on older sklearn
    versions where it does not exist.
    """
    try:
        import sklearn.frozen  # noqa: F401
        return
    except Exception:
        pass

    mod = types.ModuleType("sklearn.frozen")

    class FrozenEstimator:  # pragma: no cover - shim for artifact loading
        def __init__(self, estimator: Any):
            self.estimator = estimator

        def fit(self, X, y=None, **kwargs):
            return self

        def predict(self, X):
            return self.estimator.predict(X)

        def predict_proba(self, X):
            return self.estimator.predict_proba(X)

        def decision_function(self, X):
            fn = getattr(self.estimator, "decision_function", None)
            if fn is None:
                raise AttributeError("underlying estimator has no decision_function")
            return fn(X)

        def get_params(self, deep: bool = True):
            return {"estimator": self.estimator}

        def set_params(self, **params):
            if "estimator" in params:
                self.estimator = params["estimator"]
            return self

    mod.FrozenEstimator = FrozenEstimator
    sys.modules["sklearn.frozen"] = mod


def apply_runtime_compat() -> None:
    _ensure_numpy_private_core_alias()
    _configure_loky_cpu_count()
    _patch_numpy_bitgenerator_ctor()
    _ensure_sklearn_frozen_module()


def _iter_children(model: Any) -> Iterable[Any]:
    raw_dict = getattr(model, "__dict__", {})

    for attr in ("model", "regressor", "regressor_", "estimator", "estimator_"):
        child = raw_dict.get(attr) if isinstance(raw_dict, dict) and attr in raw_dict else getattr(model, attr, None)
        if child is not None:
            yield child

    # Avoid touching sklearn's deprecated `base_estimator_` property unless it is
    # already materialized on the instance.
    for attr in ("base_estimator", "base_estimator_"):
        child = raw_dict.get(attr) if isinstance(raw_dict, dict) else None
        if child is not None:
            yield child

    members = getattr(model, "models", None)
    if isinstance(members, (list, tuple)):
        for child in members:
            if child is not None:
                yield child

    steps = getattr(model, "steps", None)
    if isinstance(steps, (list, tuple)):
        for item in steps:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                child = item[1]
                if child is not None:
                    yield child

    estimators = getattr(model, "estimators", None)
    if isinstance(estimators, (list, tuple)):
        for item in estimators:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                child = item[1]
            else:
                child = item
            if child is not None:
                yield child

    estimators_fitted = getattr(model, "estimators_", None)
    if isinstance(estimators_fitted, (list, tuple)):
        for child in estimators_fitted:
            if child is not None:
                yield child


def tune_loaded_model(model: Any) -> Any:
    """
    Make persisted estimators safer in constrained local environments.

    Some sandboxes deny multiprocessing primitives used by joblib when
    estimators were trained with ``n_jobs=-1``. For serving/inference we force
    single-thread prediction.
    """
    if model is None:
        return model

    stack = [model]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        cur_id = id(cur)
        if cur_id in seen:
            continue
        seen.add(cur_id)

        try:
            if hasattr(cur, "get_params") and hasattr(cur, "set_params"):
                params = cur.get_params(deep=False)
                patch = {}
                if isinstance(params, dict):
                    if "n_jobs" in params:
                        patch["n_jobs"] = 1
                    if "nthread" in params:
                        patch["nthread"] = 1
                if patch:
                    cur.set_params(**patch)
            else:
                if hasattr(cur, "n_jobs"):
                    setattr(cur, "n_jobs", 1)
                if hasattr(cur, "nthread"):
                    setattr(cur, "nthread", 1)
        except Exception:
            pass

        for child in _iter_children(cur):
            stack.append(child)

    return model


apply_runtime_compat()
