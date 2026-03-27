from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hf_space.runtime import prepare_runtime_environment, summary_as_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare runtime config and start the crop Space server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    parser.add_argument("--runtime-root", default="hf_runtime")
    parser.add_argument("--artifacts-local-dir", default=os.environ.get("HF_ARTIFACTS_LOCAL_DIR", ""))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-artifact-sync", action="store_true")
    args = parser.parse_args()

    local_artifacts_dir = Path(args.artifacts_local_dir).resolve() if str(args.artifacts_local_dir).strip() else None
    summary = prepare_runtime_environment(
        runtime_root=Path(args.runtime_root),
        local_artifacts_dir=local_artifacts_dir,
        force_artifact_sync=bool(args.force_artifact_sync),
    )

    os.environ["CROP_CONFIG_PATH"] = summary["runtime_config_path"]
    os.environ["CROP_OUTPUT_DIR"] = summary["output_root"]
    os.environ["CROP_USERS_PATH"] = str((Path(summary["data_root"]) / "system" / "users.json").as_posix())

    if args.prepare_only:
        print(summary_as_json(summary))
        return

    uvicorn.run(
        "后端.入口.本地服务:app",
        host=args.host,
        port=int(args.port),
        reload=False,
    )


if __name__ == "__main__":
    main()
