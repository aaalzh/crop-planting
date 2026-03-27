from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, int(os.cpu_count() or 1))))

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from 后端.数据加载 import load_config
from 训练流水线.运行训练 import run as run_training


def _abs(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _resolve_output_dir(train_cfg: dict, backend_cfg: dict) -> Path:
    train_paths = train_cfg.get("paths", {}) if isinstance(train_cfg, dict) else {}
    text = str(train_paths.get("output_dir", "")).strip()
    if text:
        return _abs(text)
    output_cfg = backend_cfg.get("output", {}) if isinstance(backend_cfg, dict) else {}
    text = str(output_cfg.get("out_dir", "")).strip()
    if text:
        return _abs(text)
    return _abs("输出")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run high-precision training and build a closed-loop release bundle")
    parser.add_argument("--train-config", default="训练流水线/配置/高精度.yaml")
    parser.add_argument("--backend-config", default="后端/配置.yaml")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--skip-release", action="store_true")
    args = parser.parse_args()

    train_config_path = _abs(args.train_config)
    backend_config_path = _abs(args.backend_config)
    train_cfg = load_config(str(train_config_path))
    backend_cfg = load_config(str(backend_config_path))
    output_dir = _resolve_output_dir(train_cfg, backend_cfg)

    t0 = time.perf_counter()
    train_result = run_training(
        str(train_config_path),
        build_release=not bool(args.skip_release),
        backend_config_path=str(backend_config_path),
        release_run_id=str(args.run_id).strip() or None,
    )

    payload = {
        "ok": True,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "train_config": train_config_path.as_posix(),
        "backend_config": backend_config_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "train": train_result,
        "release": train_result.get("release"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
