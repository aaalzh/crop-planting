import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from 后端.数据加载 import load_config
from 后端.训练.环境模型训练 import build_bundle


def _to_abs(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Environment model backtest")
    parser.add_argument("--config", default="后端/配置.yaml")
    args = parser.parse_args()

    config = load_config(str(_to_abs(args.config)))
    out = build_bundle(args.config)

    payload = {
        "train_file": out.get("train_file"),
        "bundle_path": out.get("bundle_path"),
        "best_params": out.get("best_params"),
        "selection": out.get("selection"),
        "holdout_metrics": out.get("holdout_metrics"),
        "cv_metrics": out.get("cv_metrics"),
    }

    print("bundle_path", payload["bundle_path"])
    print("train_file", payload["train_file"])
    print("best_params", payload["best_params"])
    print("holdout_metrics", payload["holdout_metrics"])
    print("cv_metrics", payload["cv_metrics"])

    out_dir = _to_abs(config["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "环境回测.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path.as_posix())


if __name__ == "__main__":
    main()
