import os
import sys
import json
import argparse
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from 后端.数据加载 import load_config
from 后端.模型.概率校准器 import train_calibrator, save_calibrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="后端/配置.yaml")
    parser.add_argument("--history", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    prob_cfg = config.get("probability", {})
    if not bool(prob_cfg.get("enable_calibrator", False)):
        raise ValueError("probability calibrator is disabled by config")

    history_path = args.history or prob_cfg.get("history_file", "")
    if not history_path:
        raise ValueError("probability.history_file is empty")

    if not os.path.exists(history_path):
        raise FileNotFoundError(f"history file not found: {history_path}")

    df = pd.read_csv(history_path)
    res = train_calibrator(df, prob_cfg)

    out_dir = config["output"]["out_dir"]
    model_path, meta_path = save_calibrator(out_dir, res)

    print("saved:")
    print(model_path)
    print(meta_path)


if __name__ == "__main__":
    main()
