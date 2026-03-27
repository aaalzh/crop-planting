from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from 后端.数据加载 import load_config
from 后端.发布治理 import build_release_from_output, get_release_status, promote_release, rollback_release


def _abs(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or manage closed-loop release bundles")
    parser.add_argument("--config", default="后端/配置.yaml")
    parser.add_argument("--action", default="build", choices=["build", "status", "promote", "rollback"])
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    config = load_config(str(_abs(args.config)))
    action = str(args.action)

    if action == "status":
        result = get_release_status(root=ROOT, config=config)
    elif action == "promote":
        if not str(args.run_id).strip():
            raise ValueError("--run-id is required for promote")
        result = promote_release(root=ROOT, config=config, run_id=str(args.run_id).strip())
    elif action == "rollback":
        result = rollback_release(root=ROOT, config=config, run_id=str(args.run_id).strip() or None)
    else:
        output_dir = _abs(args.output_dir) if str(args.output_dir).strip() else None
        result = build_release_from_output(
            root=ROOT,
            config=config,
            output_dir=output_dir,
            run_id=str(args.run_id).strip() or None,
            source="manual_release_builder",
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
