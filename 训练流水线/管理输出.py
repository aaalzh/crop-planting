from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from 训练流水线.数据流水线.加载器 import load_config
from 训练流水线.数据流水线.输出生命周期 import apply_output_lifecycle


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage outputs lifecycle without retraining")
    parser.add_argument("--config", default="训练流水线/配置/高精度.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = args.out_dir or config.get("paths", {}).get("output_dir", "输出")
    os.makedirs(out_dir, exist_ok=True)

    if args.dry_run:
        config.setdefault("outputs_lifecycle", {})
        config["outputs_lifecycle"]["dry_run"] = True

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    report = apply_output_lifecycle(out_dir=out_dir, config=config, run_tag=run_tag)

    report_path = os.path.join(out_dir, "输出生命周期报告.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({"output_lifecycle_report": report_path, "summary": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
