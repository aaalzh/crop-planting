from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, int(os.cpu_count() or 1))))

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from 后端.发布治理 import get_release_status, rollback_release
from 后端.数据加载 import load_config
from 后端.闭环反馈 import ClosedLoopRecorder
from 后端.反馈回流 import build_feedback_training_dataset, get_feedback_training_status
from 训练流水线.运行训练 import run as run_training


def _abs(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop demo CLI")
    parser.add_argument("--backend-config", default="后端/配置.yaml")
    parser.add_argument("--train-config", default="训练流水线/配置/高精度.yaml")
    parser.add_argument(
        "--action",
        default="status",
        choices=["status", "build-feedback", "train-release", "rollback"],
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--skip-release", action="store_true")
    parser.add_argument("--refresh-feedback", action="store_true")
    args = parser.parse_args()

    backend_config_path = _abs(args.backend_config)
    train_config_path = _abs(args.train_config)
    backend_config = load_config(str(backend_config_path))

    if args.action == "status":
        recorder = ClosedLoopRecorder(root=ROOT, config=backend_config, logger=None)
        payload = {
            "closed_loop": recorder.get_status(),
            "feedback_training": get_feedback_training_status(
                root=ROOT,
                config=backend_config,
                refresh=bool(args.refresh_feedback),
            ),
            "release": get_release_status(root=ROOT, config=backend_config),
        }
    elif args.action == "build-feedback":
        payload = build_feedback_training_dataset(root=ROOT, config=backend_config, save=True)
    elif args.action == "train-release":
        payload = run_training(
            str(train_config_path),
            build_release=not bool(args.skip_release),
            backend_config_path=str(backend_config_path),
            release_run_id=str(args.run_id).strip() or None,
        )
    else:
        payload = rollback_release(
            root=ROOT,
            config=backend_config,
            run_id=str(args.run_id).strip() or None,
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
