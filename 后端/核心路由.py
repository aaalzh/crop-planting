from pathlib import Path as _Path
import runpy as _runpy

_TARGET = _Path(__file__).resolve().parent / r'路由\\核心路由.py'
_EXPORTS = _runpy.run_path(str(_TARGET))
globals().update({k: v for k, v in _EXPORTS.items() if not k.startswith('__')})

if __name__ == '__main__':
    _runpy.run_path(str(_TARGET), run_name='__main__')
