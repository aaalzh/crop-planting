import argparse
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from 后端.数据加载 import load_config, load_cost_data, load_name_map, load_yield_history, resolve_names
from 后端.模型.成本模型 import train_one_crop, train_panel_model
from 后端.时间策略 import resolve_year_window_from_series
from 后端.训练.离线训练模型 import _build_cost_panel_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost model backtest")
    parser.add_argument("--config", default="后端/配置.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    cost_df = load_cost_data(config["paths"]["cost_file"])
    time_cfg = config.get("time", {})
    strict_cutoff_split = bool(time_cfg.get("strict_cutoff_split", True))
    rows = []
    cost_cfg = config["model"]["cost"]
    feature_set = str(cost_cfg.get("feature_set", "legacy")).strip().lower()
    if feature_set == "panel_lite":
        mapping = resolve_names(load_name_map(config["paths"]["name_map"]))
        yield_history_all = load_yield_history(config["paths"].get("yield_history", ""))
        panel_df = _build_cost_panel_frame(
            mapping=mapping,
            paths=config["paths"],
            cost_all=cost_df,
            yield_history_all=yield_history_all,
        )
        window = resolve_year_window_from_series(
            pd.to_numeric(panel_df.get("year"), errors="coerce").dropna().tolist(),
            time_cfg=time_cfg,
            window_years_key="cost_prediction_window_years",
        )
        validation_cutoff = str(window.get("train_validation_cutoff_date") or "2020-12-31").strip()
        try:
            res = train_panel_model(
                panel_df,
                cost_cfg,
                test_ratio=cost_cfg.get("test_ratio"),
                validation_cutoff=validation_cutoff,
                strict_cutoff_split=strict_cutoff_split,
                verbose=cost_cfg.get("verbose", False),
                label="shared_panel",
            )
            row = {"crop": "__shared_panel__"}
            row.update(res.metrics)
            row.update(
                {
                    "prediction_window_start_year": int(window.get("start_year")),
                    "prediction_window_end_year": int(window.get("end_year")),
                    "prediction_window_years": int(window.get("window_years")),
                    "train_validation_cutoff_date": validation_cutoff,
                }
            )
            rows.append(row)
        except Exception as exc:
            rows.append({"crop": "__shared_panel__", "error": str(exc)})
    else:
        for crop, g in cost_df.groupby("crop_name"):
            g = g.copy().sort_values("year_start")
            window = resolve_year_window_from_series(
                pd.to_numeric(g["year_start"], errors="coerce").dropna().tolist(),
                time_cfg=time_cfg,
                window_years_key="cost_prediction_window_years",
            )
            validation_cutoff = str(window.get("train_validation_cutoff_date") or "2020-12-31").strip()
            try:
                res = train_one_crop(
                    g,
                    cost_cfg,
                    test_ratio=cost_cfg.get("test_ratio"),
                    validation_cutoff=validation_cutoff,
                    strict_cutoff_split=strict_cutoff_split,
                    verbose=cost_cfg.get("verbose", False),
                    label=crop,
                )
                row = {"crop": crop}
                row.update(res.metrics)
                row.update(
                    {
                        "prediction_window_start_year": int(window.get("start_year")),
                        "prediction_window_end_year": int(window.get("end_year")),
                        "prediction_window_years": int(window.get("window_years")),
                        "train_validation_cutoff_date": validation_cutoff,
                    }
                )
                rows.append(row)
            except Exception as exc:
                rows.append({"crop": crop, "error": str(exc)})

    out_dir = config["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "成本回测.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()
