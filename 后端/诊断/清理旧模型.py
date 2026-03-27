import argparse
from pathlib import Path
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from 后端.数据加载 import load_config
from 后端.数据加载 import load_name_map, resolve_names
from 后端.模型产物 import (
    expected_cost_model_path,
    expected_price_model_path,
    expected_price_recursive_model_path,
    expected_yield_model_path,
)


def clean_legacy_models(config_path: str, execute: bool = False) -> dict:
    config = load_config(config_path)
    version = str(config.get("serving", {}).get("model_cache_version", "v2"))
    model_dir = Path(config["output"]["out_dir"]) / "模型"
    model_dir.mkdir(parents=True, exist_ok=True)
    mapping = resolve_names(load_name_map(config["paths"]["name_map"]))

    keep_names = set()
    y_model, y_meta = expected_yield_model_path(model_dir, config["model"]["yield"], version)
    keep_names.add(y_model.name)
    keep_names.add(y_meta.name)
    for crop in sorted(mapping.keys()):
        p_model, p_meta = expected_price_model_path(model_dir, crop, config["model"]["price"], version)
        keep_names.add(p_model.name)
        keep_names.add(p_meta.name)
        step_model, step_meta = expected_price_recursive_model_path(model_dir, crop, config["model"]["price"], version)
        keep_names.add(step_model.name)
        keep_names.add(step_meta.name)
        c_model, c_meta = expected_cost_model_path(model_dir, crop, config["model"]["cost"], version)
        keep_names.add(c_model.name)
        keep_names.add(c_meta.name)

    removed = []
    kept = []
    for p in sorted(model_dir.iterdir(), key=lambda x: x.name):
        if not p.is_file():
            continue
        name = p.name
        if name in keep_names:
            kept.append(p)
            continue
        if not (name.endswith(".pkl") or name.endswith("_指标.json")):
            kept.append(p)
            continue

        size_mb = p.stat().st_size / 1024 / 1024
        if execute:
            p.unlink(missing_ok=True)
        removed.append({"name": name, "size_mb": round(size_mb, 2)})

    total_removed_mb = round(sum(x["size_mb"] for x in removed), 2)
    return {
        "model_dir": model_dir.as_posix(),
        "version_kept": version,
        "removed_count": len(removed),
        "removed_total_mb": total_removed_mb,
        "removed": removed,
        "execute": execute,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean legacy model artifacts under 输出/模型")
    parser.add_argument("--config", default="后端/配置.yaml")
    parser.add_argument("--execute", action="store_true", help="Actually delete files. Without this flag it's dry-run.")
    args = parser.parse_args()

    summary = clean_legacy_models(args.config, execute=args.execute)
    print(summary)


if __name__ == "__main__":
    main()
