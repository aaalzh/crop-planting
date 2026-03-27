from importlib import import_module

_impl = import_module("后端.兼容层")

apply_runtime_compat = _impl.apply_runtime_compat
tune_loaded_model = _impl.tune_loaded_model
_numpy_bitgenerator_ctor_compat = _impl._numpy_bitgenerator_ctor_compat


def __getattr__(name):
    return getattr(_impl, name)
