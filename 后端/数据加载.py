import os
import os
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import yaml
import importlib.util

from 后端.数据清洗 import clean_cost_history_frame, clean_price_series_frame, clean_yield_history_frame


@dataclass
class Paths:
    env_model_dir: str
    env_model_bundle: str
    env_predict_py: str
    price_dir: str
    cost_file: str
    yield_file: str
    yield_history: str
    name_map: str


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_paths(config: dict) -> Paths:
    p = config["paths"]
    return Paths(
        env_model_dir=p["env_model_dir"],
        env_model_bundle=p["env_model_bundle"],
        env_predict_py=p["env_predict_py"],
        price_dir=p["price_dir"],
        cost_file=p["cost_file"],
        yield_file=p["yield_file"],
        yield_history=p.get("yield_history", ""),
        name_map=p["name_map"],
    )


def load_name_map(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.fillna("")
    return df


PRICE_FILE_ALIASES = {
    "apple": "苹果",
    "banana": "香蕉",
    "blackgram": "黑豆",
    "chickpea": "鹰嘴豆",
    "coconut": "椰子",
    "cocount": "椰子",
    "coffee": "咖啡",
    "cotton": "棉花",
    "grapes": "葡萄",
    "jute": "黄麻",
    "kidneybeans": "芸豆",
    "lentil": "扁豆",
    "maize": "玉米",
    "mango": "芒果",
    "mothbeans": "木豆",
    "mungbean": "绿豆",
    "muskmelon": "香瓜",
    "orange": "橙子",
    "papaya": "木瓜",
    "pigeonpeas": "豌豆",
    "pomegranate": "石榴",
    "rice": "水稻",
    "watermelon": "西瓜",
    "water melon": "西瓜",
}


def _normalize_name_token(value: str) -> str:
    text = str(value or "").strip().lower()
    if text.endswith(".csv"):
        text = text[:-4]
    return "".join(text.replace("_", " ").replace("-", " ").split())


def canonicalize_price_file(price_file: str) -> str:
    raw = str(price_file or "").strip()
    if not raw:
        return ""
    normalized = _normalize_name_token(raw)
    return PRICE_FILE_ALIASES.get(normalized, raw[:-4] if raw.lower().endswith(".csv") else raw)


def _iter_price_csvs(base_dir: Path):
    if not base_dir.exists():
        return []
    return [path for path in sorted(base_dir.rglob("*.csv")) if path.is_file()]


def resolve_price_file_path(price_dir: str, price_file: str) -> Tuple[str, Path]:
    raw = str(price_file or "").strip()
    canonical = canonicalize_price_file(raw)
    base_dir = Path(price_dir)
    candidates = []
    for item in (canonical, raw):
        stem = str(item or "").strip()
        if not stem:
            continue
        if stem.lower().endswith(".csv"):
            stem = stem[:-4]
        if stem and stem not in candidates:
            candidates.append(stem)

    for stem in candidates:
        path = base_dir / f"{stem}.csv"
        if path.exists():
            return stem, path

    candidate_tokens = {_normalize_name_token(item) for item in candidates if item}
    for path in _iter_price_csvs(base_dir):
        if _normalize_name_token(path.stem) in candidate_tokens:
            return path.stem, path

    fallback_stem = candidates[0] if candidates else ""
    return fallback_stem, base_dir / f"{fallback_stem}.csv"


def load_price_series(price_dir: str, price_file: str) -> pd.DataFrame:
    resolved_stem, file_path = resolve_price_file_path(price_dir, price_file)
    if not file_path.exists():
        raise FileNotFoundError(f"price file not found: requested={price_file} resolved={resolved_stem} path={file_path}")
    df = pd.read_csv(file_path)
    return clean_price_series_frame(df)


def _resolve_internal_cost_stats_file(cost_file: str) -> Path:
    cost_path = Path(cost_file)
    if str(cost_path.parent.name).strip() == "原始":
        return cost_path.parent.parent / "内部" / "加权平均成本统计.csv"
    return cost_path.with_name(f"{cost_path.stem}_统计{cost_path.suffix}")


def _load_internal_cost_stats(stats_file: Path) -> pd.DataFrame:
    if not stats_file.exists():
        return pd.DataFrame(columns=["crop_name", "year_start", "n_states", "n_rows_cost", "sum_sample_weight"])
    df = pd.read_csv(stats_file)
    if df.empty:
        return pd.DataFrame(columns=["crop_name", "year_start", "n_states", "n_rows_cost", "sum_sample_weight"])
    keep_cols = ["crop_name", "year_start", "n_states", "n_rows_cost", "sum_sample_weight"]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = pd.NA
    out = df[keep_cols].copy()
    out["crop_name"] = out["crop_name"].astype(str).str.strip()
    out["year_start"] = pd.to_numeric(out["year_start"], errors="coerce")
    for col in ["n_states", "n_rows_cost", "sum_sample_weight"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["crop_name", "year_start"])
    if out.empty:
        return pd.DataFrame(columns=keep_cols)
    out["year_start"] = out["year_start"].astype(int)
    out = out[out["crop_name"] != ""].copy()
    return out.drop_duplicates(subset=["crop_name", "year_start"], keep="last").reset_index(drop=True)


def load_cost_data(cost_file: str) -> pd.DataFrame:
    df = pd.read_csv(cost_file)
    base = clean_cost_history_frame(df)
    if base.empty:
        return base

    stats_file = _resolve_internal_cost_stats_file(cost_file)
    stats_df = _load_internal_cost_stats(stats_file)
    if stats_df.empty:
        return base
    return base.merge(stats_df, on=["crop_name", "year_start"], how="left")


def load_yield_data(yield_file: str) -> pd.DataFrame:
    if not os.path.exists(yield_file):
        return pd.DataFrame(columns=["crop_name", "yield_quintal_per_hectare"])
    df = pd.read_csv(yield_file, comment="#")
    if "crop_name" not in df.columns:
        return pd.DataFrame(columns=["crop_name", "yield_quintal_per_hectare"])
    if "yield_quintal_per_hectare" not in df.columns:
        df["yield_quintal_per_hectare"] = None
    return df


def load_yield_history(yield_history: str) -> pd.DataFrame:
    if not yield_history or not os.path.exists(yield_history):
        return pd.DataFrame()
    df = pd.read_csv(yield_history, comment="#")
    return clean_yield_history_frame(df)


def resolve_names(name_map: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    out = {}
    for _, row in name_map.iterrows():
        env_label = str(row.get("env_label", "")).strip()
        if not env_label:
            continue
        price_file = canonicalize_price_file(str(row.get("price_file", "")).strip() or env_label)
        out[env_label] = {
            "price_file": price_file,
            "cost_name": str(row.get("cost_name", "")).strip(),
        }
    return out


def _load_module_from_path(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_env_predictor(env_predict_py: str):
    return _load_module_from_path(env_predict_py, "env_predict")


def read_env_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)
