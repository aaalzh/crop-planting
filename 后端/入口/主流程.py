import os
import json
import argparse
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, int(os.cpu_count() or 1))))

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from 后端.数据加载 import load_config, read_env_json
from 后端.推荐器 import recommend


def _to_abs(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def _ensure_out_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def _write_outputs(out_dir: Path, payload: dict):
    json_path = out_dir / "推荐结果.json"
    csv_path = out_dir / "推荐结果.csv"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.DataFrame(payload["results"])
    df.to_csv(csv_path, index=False)

    return json_path.as_posix(), csv_path.as_posix()


def main():
    parser = argparse.ArgumentParser(description="Crop decision system")
    parser.add_argument("--config", default="后端/配置.yaml")
    sub = parser.add_subparsers(dest="cmd")

    r = sub.add_parser("recommend")
    r.add_argument("--env-json", required=True, help="path to environment json")

    args = parser.parse_args()
    if args.cmd != "recommend":
        parser.print_help()
        return

    config = load_config(args.config)
    env = read_env_json(args.env_json)

    payload = recommend(env, config)
    out_dir = _to_abs(config["output"]["out_dir"])
    _ensure_out_dir(out_dir)
    json_path, csv_path = _write_outputs(out_dir, payload)

    print("Saved:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
