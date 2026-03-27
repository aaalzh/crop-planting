from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from 后端.数据加载 import load_config
from 后端.环境桥接 import build_env_scenario_library


def _abs(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build environment scenario library from the real environment model")
    parser.add_argument("--config", default="后端/配置.yaml")
    args = parser.parse_args()

    config = load_config(str(_abs(args.config)))
    result = build_env_scenario_library(root=ROOT, config=config, save=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
