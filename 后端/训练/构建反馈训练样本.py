from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from 后端.数据加载 import load_config
from 后端.反馈回流 import build_feedback_training_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build closed-loop feedback training samples")
    parser.add_argument("--config", default="后端/配置.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    config = load_config(str(config_path))
    result = build_feedback_training_dataset(root=ROOT, config=config, save=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
