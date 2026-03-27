import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from 后端.数据加载 import load_config, load_yield_history
from 后端.模型.产量模型 import train_yield_model
from 后端.时间策略 import resolve_year_window_from_series


def main() -> None:
    parser = argparse.ArgumentParser(description="Yield model backtest")
    parser.add_argument("--config", default="后端/配置.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ypath = config["paths"].get("yield_history", "")
    time_cfg = config.get("time", {})
    strict_cutoff_split = bool(time_cfg.get("strict_cutoff_split", True))
    df = load_yield_history(ypath)
    out_dir = config["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "产量回测.json")

    if df.empty:
        payload = {"error": "yield_history is empty", "path": ypath}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(out_path)
        return

    if "yield_quintal_per_hectare" not in df.columns:
        payload = {"error": "missing yield_quintal_per_hectare", "columns": list(df.columns)}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(out_path)
        return

    window = resolve_year_window_from_series(
        pd.to_numeric(df.get("year"), errors="coerce").dropna().tolist(),
        time_cfg=time_cfg,
        window_years_key="yield_prediction_window_years",
    )
    validation_cutoff = str(window.get("train_validation_cutoff_date") or "2020-12-31").strip()
    res = train_yield_model(
        df,
        config["model"]["yield"],
        test_ratio=config["model"]["yield"].get("test_ratio"),
        validation_cutoff=validation_cutoff,
        strict_cutoff_split=strict_cutoff_split,
        verbose=config["model"]["yield"].get("verbose", False),
        label="global",
    )
    payload = {
        "metrics": res.metrics,
        "feature_cols": res.feature_cols,
        "prediction_year_window": {
            "start_year": int(window.get("start_year")),
            "end_year": int(window.get("end_year")),
            "window_years": int(window.get("window_years")),
            "train_validation_cutoff_date": validation_cutoff,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
