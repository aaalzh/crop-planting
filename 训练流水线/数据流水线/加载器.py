from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml

from 后端.数据清洗 import clean_cost_history_frame, clean_price_series_frame, clean_yield_history_frame
from 后端.环境桥接 import load_env_scenario_library


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith(".json"):
        return json.loads(text)
    return yaml.safe_load(text)


def _norm_text(v: object) -> str:
    return str(v or "").strip().lower()


def _to_int_year(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def load_name_map(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = ["env_label", "price_file", "cost_name"]
    for col in keep:
        if col not in df.columns:
            df[col] = ""
    out = df[keep].copy()
    out["env_label"] = out["env_label"].astype(str).str.strip().str.lower()
    out["price_file"] = out["price_file"].astype(str).str.strip()
    out["cost_name"] = out["cost_name"].astype(str).str.strip()
    out = out[out["env_label"] != ""].drop_duplicates(subset=["env_label"])
    return out.reset_index(drop=True)


def load_env_prior(env_file: str) -> Dict[str, float]:
    df = pd.read_csv(env_file)
    if "label" not in df.columns or df.empty:
        return {}
    s = df["label"].astype(str).str.strip().str.lower()
    freq = s.value_counts(normalize=True)
    return {str(k): float(v) for k, v in freq.items()}


def load_env_prob_from_scenarios(root_dir: str, backend_config_path: str = "后端/配置.yaml") -> Dict[str, float]:
    try:
        root = os.path.abspath(root_dir or ".")
        config_path = backend_config_path
        if not os.path.isabs(config_path):
            config_path = os.path.join(root, config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            backend_cfg = yaml.safe_load(f.read())
        payload = load_env_scenario_library(root=Path(root), config=backend_cfg, rebuild_if_missing=True)
        items = payload.get("items", [])
        if not isinstance(items, list) or not items:
            return {}
        scores: Dict[str, list] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            probabilities = item.get("probabilities", {})
            if isinstance(probabilities, dict) and probabilities:
                for crop, prob in probabilities.items():
                    crop_norm = _norm_text(crop)
                    prob_num = pd.to_numeric(pd.Series([prob]), errors="coerce").iloc[0]
                    if not crop_norm or pd.isna(prob_num):
                        continue
                    scores.setdefault(crop_norm, []).append(float(prob_num))
                continue
            topk = item.get("topk", [])
            if not isinstance(topk, list):
                continue
            for row in topk:
                if not isinstance(row, dict):
                    continue
                crop = _norm_text(row.get("crop"))
                prob = pd.to_numeric(pd.Series([row.get("prob")]), errors="coerce").iloc[0]
                if not crop or pd.isna(prob):
                    continue
                scores.setdefault(crop, []).append(float(prob))
        if not scores:
            return {}
        return {crop: float(sum(vals) / max(len(vals), 1)) for crop, vals in scores.items() if vals}
    except Exception:
        return {}


def _read_price_one(price_path: str) -> pd.DataFrame:
    df = pd.read_csv(price_path)
    work = clean_price_series_frame(df)
    if work.empty or "date" not in work.columns or "modal_price" not in work.columns:
        return pd.DataFrame(columns=["year", "price", "price_obs_days", "price_year_std"])
    work["year"] = work["date"].dt.year.astype(int)
    agg = work.groupby("year", as_index=False).agg(
        price=("modal_price", "mean"),
        price_obs_days=("modal_price", "count"),
        price_year_std=("modal_price", "std"),
    )
    agg["price_year_std"] = agg["price_year_std"].fillna(0.0)
    return agg


def load_price_yearly(price_dir: str, name_map: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, m in name_map.iterrows():
        crop = str(m["env_label"]).strip().lower()
        price_file = str(m["price_file"]).strip()
        if not crop or not price_file:
            continue
        fp = os.path.join(price_dir, f"{price_file}.csv")
        if not os.path.exists(fp):
            continue
        agg = _read_price_one(fp)
        if agg.empty:
            continue
        agg["crop"] = crop
        agg["price_file"] = price_file
        rows.append(agg)
    if not rows:
        return pd.DataFrame(columns=["crop", "year", "price", "price_obs_days", "price_year_std", "price_file"])
    out = pd.concat(rows, ignore_index=True)
    out["year"] = _to_int_year(out["year"])
    out = out.dropna(subset=["year", "price"])
    out["year"] = out["year"].astype(int)
    return out.reset_index(drop=True)


def load_yield_history(yield_file: str, name_map: pd.DataFrame) -> pd.DataFrame:
    df = clean_yield_history_frame(pd.read_csv(yield_file))
    cols = {"crop_name", "year", "yield_quintal_per_hectare"}
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"yield file missing column: {c}")
    out = df[["crop_name", "year", "yield_quintal_per_hectare"]].copy()
    out["crop"] = out["crop_name"].astype(str).str.strip().str.lower()
    out["year"] = _to_int_year(out["year"])
    out["yield"] = pd.to_numeric(out["yield_quintal_per_hectare"], errors="coerce")
    out = out.dropna(subset=["crop", "year", "yield"])
    out["year"] = out["year"].astype(int)

    crop_set = set(name_map["env_label"].astype(str).str.lower())
    out = out[out["crop"].isin(crop_set)].copy()

    out = out.groupby(["crop", "year"], as_index=False).agg(yield_value=("yield", "mean"))
    out = out.rename(columns={"yield_value": "yield"})
    return out


def load_cost_history(cost_file: str, name_map: pd.DataFrame) -> pd.DataFrame:
    df = clean_cost_history_frame(pd.read_csv(cost_file))
    required = {"crop_name", "year_start", "india_cost_wavg_sample"}
    for c in required:
        if c not in df.columns:
            raise ValueError(f"cost file missing column: {c}")

    base = df[["crop_name", "year_start", "india_cost_wavg_sample"]].copy()
    base["cost_name"] = base["crop_name"].astype(str).str.strip()
    base["cost_name_norm"] = base["cost_name"].str.lower()
    base["year"] = _to_int_year(base["year_start"])
    base["cost"] = pd.to_numeric(base["india_cost_wavg_sample"], errors="coerce")
    base = base.dropna(subset=["year", "cost"]).copy()
    base["year"] = base["year"].astype(int)

    rows = []
    for _, m in name_map.iterrows():
        crop = str(m["env_label"]).strip().lower()
        cname = str(m["cost_name"]).strip()
        if not crop or not cname:
            continue
        sub = base[base["cost_name_norm"] == _norm_text(cname)].copy()
        if sub.empty:
            continue
        sub["crop"] = crop
        sub["cost_name"] = cname
        rows.append(sub[["crop", "year", "cost", "cost_name"]])

    if not rows:
        return pd.DataFrame(columns=["crop", "year", "cost", "cost_name"])

    out = pd.concat(rows, ignore_index=True)
    out = out.groupby(["crop", "year", "cost_name"], as_index=False).agg(cost=("cost", "mean"))
    return out


def _crop_group(cost_name: str) -> str:
    n = _norm_text(cost_name)
    if "proxy" in n:
        if "horticulture" in n:
            return "horticulture"
        if "plantation" in n:
            return "plantation"
        if "pulse" in n:
            return "pulses"
        return "proxy"
    if any(k in n for k in ["gram", "lentil", "moong", "urad", "tur", "bean"]):
        return "pulses"
    if any(k in n for k in ["paddy", "rice", "maize", "wheat", "bajra", "barley", "jowar", "ragi"]):
        return "cereal"
    if any(k in n for k in ["cotton", "jute", "sugarcane"]):
        return "industrial"
    if any(k in n for k in ["onion", "potato", "tapioca", "vegetable"]):
        return "vegetable"
    return "other"


def build_panel_dataset(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    paths = config["paths"]

    name_map = load_name_map(paths["name_map"])
    env_prior = load_env_prob_from_scenarios(root_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
    if not env_prior:
        env_prior = load_env_prior(paths["env_file"])

    price = load_price_yearly(paths["price_dir"], name_map)
    yield_df = load_yield_history(paths["yield_file"], name_map)
    cost = load_cost_history(paths["cost_file"], name_map)

    panel = price.merge(yield_df, on=["crop", "year"], how="outer")
    panel = panel.merge(cost, on=["crop", "year"], how="outer")

    meta = name_map.rename(columns={"env_label": "crop"}).copy()
    panel = panel.merge(meta, on="crop", how="left", suffixes=("", "_map"))

    panel["crop"] = panel["crop"].astype(str).str.strip().str.lower()
    panel["year"] = _to_int_year(panel["year"])
    panel = panel.dropna(subset=["crop", "year"]).copy()
    panel["year"] = panel["year"].astype(int)

    panel["env_prob"] = panel["crop"].map(env_prior).fillna(0.0)
    panel["env_prob_source"] = "scenario_library" if env_prior else "empirical_prior"
    panel["cost_name"] = panel["cost_name"].fillna("")
    panel["crop_group"] = panel["cost_name"].map(_crop_group)

    panel = panel.sort_values(["crop", "year"]).reset_index(drop=True)
    return panel, name_map, env_prior
