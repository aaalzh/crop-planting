from pathlib import Path as _Path
import runpy as _runpy

_TARGET = _Path(__file__).resolve().parent / r'脚本\\清洗价格数据.py'
_EXPORTS = _runpy.run_path(str(_TARGET))
globals().update({k: v for k, v in _EXPORTS.items() if not k.startswith('__')})

if __name__ == '__main__':
    _runpy.run_path(str(_TARGET), run_name='__main__')
