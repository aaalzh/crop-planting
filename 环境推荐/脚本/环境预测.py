from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from joblib import load

from 后端.兼容层 import tune_loaded_model


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "环境推荐" / "模型" / "作物推荐模型管道.pkl"


def load_bundle(model_path: str | Path = MODEL_PATH) -> dict:
    path = Path(model_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"找不到模型文件：{path}")

    bundle = load(path)
    if isinstance(bundle, dict) and "model" in bundle:
        bundle["model"] = tune_loaded_model(bundle.get("model"))
    if not isinstance(bundle, dict) or "model" not in bundle or "meta" not in bundle:
        raise ValueError("模型文件格式不正确：应为包含 model/meta 的 bundle dict。")
    return bundle


def add_engineered_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    eps = 1e-6

    df["N_K_ratio"] = df["N"] / (df["K"] + eps)
    df["N_P_ratio"] = df["N"] / (df["P"] + eps)
    df["P_K_ratio"] = df["P"] / (df["K"] + eps)

    df["npk_sum"] = df["N"] + df["P"] + df["K"]
    df["soil_index"] = df["npk_sum"] / 3.0
    df["npk_std"] = df[["N", "P", "K"]].std(axis=1)

    df["ph_neutral_dist"] = (df["ph"] - 7.0).abs()
    df["heat_stress"] = df["temperature"] / (df["humidity"] + eps)

    df["rain_humidity_ratio"] = df["rainfall"] / (df["humidity"] + eps)
    df["temp_rain_interaction"] = df["temperature"] * np.log1p(df["rainfall"])
    return df


def to_input_df(input_data: Union[Dict[str, Any], pd.DataFrame], raw_features: List[str]) -> pd.DataFrame:
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    missing = [col for col in raw_features if col not in df.columns]
    if missing:
        raise ValueError(f"输入数据缺少列：{missing}")

    for col in raw_features:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df[raw_features]


def ood_warnings(df_raw: pd.DataFrame, train_stats: dict) -> List[str]:
    warns = []
    mins = train_stats["min"]
    maxs = train_stats["max"]

    margin = 0.05
    for col in df_raw.columns:
        value = float(df_raw[col].iloc[0])
        min_value = float(mins[col])
        max_value = float(maxs[col])
        span = max_value - min_value
        if span == 0:
            continue
        low = min_value - margin * span
        high = max_value + margin * span
        if value < low or value > high:
            warns.append(f"{col}={value:g} 超出训练范围[{min_value:g}, {max_value:g}]（可能是分布外输入）")
    return warns


def confidence_level(max_prob: float) -> str:
    if max_prob >= 0.80:
        return "高"
    if max_prob >= 0.55:
        return "中"
    return "低"


def risk_level(conf_lvl: str, ood_msgs: List[str]) -> str:
    if ood_msgs or conf_lvl == "低":
        return "高"
    if conf_lvl == "中":
        return "中"
    return "低"


def predict_topk(input_data: Union[Dict[str, Any], pd.DataFrame], k: int = 3, model_path: str | Path = MODEL_PATH) -> dict:
    bundle = load_bundle(model_path)
    model = bundle["model"]
    meta = bundle["meta"]

    raw_features = meta["raw_features"]
    feature_order = meta["feature_order"]
    train_stats = meta["train_raw_stats"]

    df_raw = to_input_df(input_data, raw_features)
    df_feat = add_engineered_features(df_raw)[feature_order]

    proba = model.predict_proba(df_feat)[0]
    classes = list(model.classes_)

    k = max(1, min(int(k), len(classes)))
    idx = np.argsort(proba)[::-1][:k]
    topk = [(classes[i], float(proba[i])) for i in idx]

    best_label, best_p = topk[0]
    conf = confidence_level(best_p)
    ood_msgs = ood_warnings(df_raw, train_stats)
    risk = risk_level(conf, ood_msgs)

    return {
        "best_label": best_label,
        "best_prob": float(best_p),
        "topk": topk,
        "confidence": conf,
        "risk": risk,
        "warnings": ood_msgs,
    }


if __name__ == "__main__":
    sample = {
        "N": 63,
        "P": 55,
        "K": 42,
        "temperature": 25.96,
        "humidity": 83.5,
        "ph": 5.27,
        "rainfall": 295.73,
    }

    out = predict_topk(sample, k=5)

    print("输入：", sample)
    print("\nTop-K 推荐：")
    for rank, (label, prob) in enumerate(out["topk"], start=1):
        print(f"  #{rank:<2d} {label:<15s}  prob={prob:.4f}")

    print(f"\n最佳作物：{out['best_label']}  (prob={out['best_prob']:.4f})")
    print(f"置信度：{out['confidence']}   风险：{out['risk']}")

    if out["warnings"]:
        print("\n警告：")
        for warning in out["warnings"]:
            print(" -", warning)
