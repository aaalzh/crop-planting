import argparse
import glob
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from 后端.数据加载 import load_config, load_price_series
from 后端.模型.价格模型 import train_one_crop
from 后端.时间策略 import resolve_price_window_from_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Price model backtest")
    parser.add_argument("--config", default="后端/配置.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    price_dir = config["paths"]["price_dir"]
    lags = config["time"]["price_lags"]
    windows = config["time"]["price_roll_windows"]
    time_cfg = config.get("time", {})
    backtest_days = int(time_cfg.get("price_backtest_days", 180))
    strict_cutoff_split = bool(time_cfg.get("strict_cutoff_split", True))

    rows = []
    for f in glob.glob(os.path.join(price_dir, "**", "*.csv"), recursive=True):
        crop = os.path.splitext(os.path.basename(f))[0]
        df = load_price_series(price_dir, crop)
        price_window = resolve_price_window_from_df(df, time_cfg=time_cfg)
        horizon = int(price_window["price_horizon_days"])
        validation_cutoff = str(price_window["train_validation_cutoff_date"]).strip()
        res = train_one_crop(
            df,
            config["model"]["price"],
            lags,
            windows,
            horizon,
            backtest_days,
            test_ratio=config["model"]["price"].get("test_ratio"),
            validation_cutoff=validation_cutoff,
            strict_cutoff_split=strict_cutoff_split,
            verbose=config["model"]["price"].get("verbose", False),
            label=crop,
        )
        row = {"crop": crop}
        row.update(res.metrics)
        row.update(
            {
                "prediction_window_start_date": str(price_window["start_date"].strftime("%Y-%m-%d")),
                "prediction_window_end_date": str(price_window["end_date"].strftime("%Y-%m-%d")),
                "price_horizon_days": int(price_window["price_horizon_days"]),
                "train_validation_cutoff_date": validation_cutoff,
            }
        )
        rows.append(row)

    out_dir = config["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "价格回测.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()
