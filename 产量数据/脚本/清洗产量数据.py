import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


TARGET_START_YEAR = 2010
TARGET_END_YEAR = 2025

# Source crop names -> model crop labels used across this project.
CROP_NAME_MAP: Dict[str, str] = {
    "banana": "banana",
    "cotton(lint)": "cotton",
    "gram": "chickpea",
    "jute": "jute",
    "maize": "maize",
    "masoor": "lentil",
    "moong(green gram)": "mungbean",
    "moth": "mothbeans",
    "arhar/tur": "pigeonpeas",
    "rice": "rice",
    "urad": "blackgram",
}

DONOR_OVERRIDES: Dict[str, str] = {
    "apple": "banana",
    "coconut": "jute",
    "coffee": "mungbean",
    "grapes": "banana",
    "kidneybeans": "chickpea",
    "mango": "banana",
    "muskmelon": "banana",
    "orange": "banana",
    "papaya": "banana",
    "pomegranate": "banana",
    "watermelon": "banana",
}


def _norm_text(v) -> str:
    return str(v).strip().lower()


def _safe_float(v) -> Optional[float]:
    try:
        out = float(v)
    except Exception:
        return None
    if pd.isna(out):
        return None
    return float(out)


def _year_to_int(v) -> Optional[int]:
    m = re.search(r"(\d{4})", str(v))
    if not m:
        return None
    return int(m.group(1))


def _convert_yield_to_quintal_per_hectare(df: pd.DataFrame) -> pd.Series:
    y = pd.to_numeric(df["yield"], errors="coerce")
    if "yield_unit" not in df.columns:
        return y * 10.0

    u = df["yield_unit"].astype(str).str.strip().str.lower()
    ton_mask = u.str.contains("ton", na=False)
    out = y.copy()
    out[ton_mask] = out[ton_mask] * 10.0
    return out


def _default_source(script_dir: Path) -> Path:
    reserved = {"产量历史数据.csv", "单位面积产量表.csv"}
    data_dir = script_dir.parent / "原始"
    cands = [p for p in data_dir.glob("*.csv") if p.name not in reserved]
    if not cands:
        return data_dir / "yield_source.csv"
    cands.sort(key=lambda p: (-p.stat().st_size, p.name))
    return cands[0]


def _crop_group(cost_name: str) -> str:
    name = _norm_text(cost_name)
    if any(k in name for k in ("apple", "banana", "grape", "mango", "muskmelon", "orange", "papaya", "pomegranate", "watermelon")):
        return "horticulture"
    if any(k in name for k in ("coffee", "coconut")):
        return "plantation"
    if any(k in name for k in ("gram", "lentil", "moong", "moth", "tur", "urad", "kidneybean")):
        return "pulses"
    if any(k in name for k in ("cotton", "jute")):
        return "industrial"
    if any(k in name for k in ("maize", "paddy", "rice")):
        return "cereal"
    return "other"


def _load_anchor_yield_map(repo_root: Path) -> Dict[str, float]:
    candidates: List[Path] = []
    registry_path = repo_root / "输出" / "发布" / "发布索引.json"
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:
            registry = {}
        champion_run_id = str((registry or {}).get("champion_run_id") or "").strip()
        if champion_run_id:
            candidates.append(repo_root / "输出" / "发布" / champion_run_id / "推荐结果.csv")

    release_root = repo_root / "输出" / "发布"
    if release_root.exists():
        release_files = sorted(release_root.glob("*/推荐结果.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.extend(release_files[:5])

    candidates.append(repo_root / "输出" / "推荐结果.csv")

    seen = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "crop" not in df.columns:
            continue
        yield_col = None
        for col in ("yield_hat", "yield", "yield_pred"):
            if col in df.columns:
                yield_col = col
                break
        if not yield_col:
            continue
        out: Dict[str, float] = {}
        for _, row in df.iterrows():
            crop = _norm_text(row.get("crop"))
            value = _safe_float(row.get(yield_col))
            if crop and value is not None and value > 0:
                out[crop] = float(value)
        if out:
            return out
    return {}


def _build_observed_series(
    crop_df: pd.DataFrame,
    *,
    anchor_2025: float,
    years: List[int],
) -> pd.Series:
    year_index = pd.Index(years, dtype=int)
    actual = crop_df[["year", "yield_quintal_per_hectare"]].copy()
    actual["year"] = pd.to_numeric(actual["year"], errors="coerce").astype(int)
    actual["yield_quintal_per_hectare"] = pd.to_numeric(actual["yield_quintal_per_hectare"], errors="coerce")
    actual = actual.dropna(subset=["yield_quintal_per_hectare"]).drop_duplicates(subset=["year"], keep="last")
    actual = actual.sort_values("year")

    if actual.empty:
        raise ValueError("observed crop series is empty")

    series = pd.Series(index=year_index, dtype=float)
    actual_map = actual.set_index("year")["yield_quintal_per_hectare"]
    common_years = [y for y in actual_map.index.tolist() if y in year_index]
    if common_years:
        series.loc[common_years] = actual_map.loc[common_years].to_numpy(dtype=float)

    series = series.sort_index().interpolate(limit_direction="both")
    latest_year = int(actual_map.index.max())
    latest_value = float(actual_map.loc[latest_year])

    if latest_year < years[-1]:
        end_value = anchor_2025 if anchor_2025 > 0 else latest_value
        span = max(1, years[-1] - latest_year)
        for year in years:
            if year <= latest_year:
                continue
            alpha = float(year - latest_year) / float(span)
            series.loc[year] = max(0.0, latest_value * (1.0 - alpha) + end_value * alpha)

    return series.clip(lower=0.0)


def _pick_donor(
    crop: str,
    target_anchor: float,
    target_group: str,
    donor_meta: Dict[str, dict],
) -> str:
    preferred = DONOR_OVERRIDES.get(crop)
    if preferred in donor_meta:
        return preferred

    same_group = [name for name, meta in donor_meta.items() if str(meta.get("group")) == target_group]
    pool = same_group or list(donor_meta.keys())
    if not pool:
        raise ValueError("no donor crops available for yield proxy generation")

    def distance(name: str) -> float:
        donor_anchor = float(donor_meta[name]["anchor_2025"])
        return abs(donor_anchor - target_anchor)

    return min(pool, key=distance)


def _expand_history(
    hist: pd.DataFrame,
    *,
    name_map: pd.DataFrame,
    anchor_yield_map: Dict[str, float],
) -> pd.DataFrame:
    years = list(range(TARGET_START_YEAR, TARGET_END_YEAR + 1))
    all_crops = sorted(name_map["env_label"].astype(str).str.strip().str.lower().unique())
    crop_group_map = {
        _norm_text(row["env_label"]): _crop_group(str(row.get("cost_name", "")))
        for _, row in name_map.iterrows()
    }

    observed = hist[(hist["year"] >= TARGET_START_YEAR) & (hist["year"] <= TARGET_END_YEAR)].copy()
    observed["crop_name"] = observed["crop_name"].astype(str).str.strip().str.lower()
    observed["year"] = pd.to_numeric(observed["year"], errors="coerce").astype(int)
    observed["yield_quintal_per_hectare"] = pd.to_numeric(observed["yield_quintal_per_hectare"], errors="coerce")
    observed = observed.dropna(subset=["yield_quintal_per_hectare"])

    observed_mean_map = (
        observed.groupby("crop_name", as_index=False)["yield_quintal_per_hectare"]
        .mean()
        .set_index("crop_name")["yield_quintal_per_hectare"]
        .to_dict()
    )

    anchor_candidates = [float(v) for v in anchor_yield_map.values() if _safe_float(v) is not None]
    anchor_candidates.extend(float(v) for v in observed_mean_map.values() if _safe_float(v) is not None)
    default_anchor = float(np.median(anchor_candidates)) if anchor_candidates else 1.0

    donor_meta: Dict[str, dict] = {}
    dense_rows = []

    for crop, crop_df in observed.groupby("crop_name"):
        crop_anchor = _safe_float(anchor_yield_map.get(crop))
        if crop_anchor is None or crop_anchor <= 0:
            crop_anchor = _safe_float(observed_mean_map.get(crop)) or default_anchor

        series = _build_observed_series(crop_df, anchor_2025=float(crop_anchor), years=years)
        anchor_value = max(float(series.loc[TARGET_END_YEAR]), 1e-6)
        donor_meta[crop] = {
            "anchor_2025": anchor_value,
            "group": crop_group_map.get(crop, "other"),
            "ratio": (series / anchor_value).astype(float),
        }

        observed_years = set(crop_df["year"].astype(int).tolist())
        latest_observed_year = max(observed_years)
        for year in years:
            if year in observed_years:
                source = "yield_source_raw"
            elif year <= latest_observed_year:
                source = "yield_gap_fill"
            else:
                source = "yield_trend_bridge"
            dense_rows.append(
                {
                    "crop_name": crop,
                    "year": int(year),
                    "yield_quintal_per_hectare": float(series.loc[year]),
                    "source": source,
                }
            )

    for crop in all_crops:
        if crop in donor_meta:
            continue
        crop_anchor = _safe_float(anchor_yield_map.get(crop))
        if crop_anchor is None or crop_anchor <= 0:
            crop_anchor = _safe_float(observed_mean_map.get(crop)) or default_anchor

        donor = _pick_donor(
            crop=crop,
            target_anchor=float(crop_anchor),
            target_group=crop_group_map.get(crop, "other"),
            donor_meta=donor_meta,
        )
        donor_ratio = donor_meta[donor]["ratio"]
        series = donor_ratio * float(crop_anchor)
        for year in years:
            dense_rows.append(
                {
                    "crop_name": crop,
                    "year": int(year),
                    "yield_quintal_per_hectare": float(series.loc[year]),
                    "source": f"yield_proxy:{donor}",
                }
            )

    dense = pd.DataFrame(dense_rows).sort_values(["crop_name", "year"]).reset_index(drop=True)
    return dense


def prepare(source_path: Path, map_path: Path, out_history: Path, out_table: Path):
    if not source_path.exists():
        raise FileNotFoundError(f"yield source file not found: {source_path}")
    if not map_path.exists():
        raise FileNotFoundError(f"name map file not found: {map_path}")

    repo_root = map_path.resolve().parents[2]
    src = pd.read_csv(source_path)
    required_cols = {"year", "crop_name", "yield"}
    missing = [c for c in required_cols if c not in src.columns]
    if missing:
        raise ValueError(f"yield source missing required columns: {missing}")

    df = src.copy()
    df["crop_name_src"] = df["crop_name"].astype(str).str.strip()
    df["crop_name_norm"] = df["crop_name_src"].apply(_norm_text)
    df["crop_name"] = df["crop_name_norm"].map(CROP_NAME_MAP)
    df["year"] = df["year"].apply(_year_to_int)
    df["yield_quintal_per_hectare"] = _convert_yield_to_quintal_per_hectare(df)

    if "area" not in df.columns:
        df["area"] = np.nan
    df["area"] = pd.to_numeric(df["area"], errors="coerce")

    df = df.dropna(subset=["crop_name", "year", "yield_quintal_per_hectare"])
    df = df[df["yield_quintal_per_hectare"] > 0].copy()
    df["year"] = df["year"].astype(int)

    if df.empty:
        raise ValueError("no valid rows after cleaning; check source data or crop mapping")

    df["area_pos"] = df["area"].where(df["area"] > 0, 0.0)
    df["yield_x_area"] = df["yield_quintal_per_hectare"] * df["area_pos"]
    hist = (
        df.groupby(["crop_name", "year"], as_index=False)
        .agg(
            area_sum=("area_pos", "sum"),
            yield_x_area_sum=("yield_x_area", "sum"),
            yield_mean=("yield_quintal_per_hectare", "mean"),
            n_rows=("yield_quintal_per_hectare", "size"),
        )
    )
    hist["yield_quintal_per_hectare"] = np.where(
        hist["area_sum"] > 0,
        hist["yield_x_area_sum"] / hist["area_sum"],
        hist["yield_mean"],
    )
    hist = hist.dropna(subset=["yield_quintal_per_hectare"])
    hist = hist.sort_values(["crop_name", "year"]).reset_index(drop=True)

    name_map = pd.read_csv(map_path)
    anchor_yield_map = _load_anchor_yield_map(repo_root)
    hist_out = _expand_history(hist, name_map=name_map, anchor_yield_map=anchor_yield_map)

    latest_rows = (
        hist_out.sort_values(["crop_name", "year"])
        .groupby("crop_name", as_index=False)
        .tail(1)
        .sort_values("crop_name")
        .reset_index(drop=True)
    )
    table_out = latest_rows[["crop_name", "yield_quintal_per_hectare", "source"]].copy()
    table_out["note"] = "aligned to 2025 anchor with raw history or donor template"

    out_history.parent.mkdir(parents=True, exist_ok=True)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    hist_out.to_csv(out_history, index=False, encoding="utf-8")
    table_out.to_csv(out_table, index=False, encoding="utf-8")

    print(f"source_rows={len(src)}")
    print(f"raw_history_rows={len(hist)}")
    print(f"history_rows={len(hist_out)}")
    print(f"history_crops={hist_out['crop_name'].nunique()}")
    print(f"history_year_min={int(hist_out['year'].min())}")
    print(f"history_year_max={int(hist_out['year'].max())}")
    print(f"table_rows={len(table_out)}")
    print(f"table_filled={int(table_out['yield_quintal_per_hectare'].notna().sum())}")
    print(str(out_history))
    print(str(out_table))


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    # script_dir = <repo>/产量数据/脚本, so one level up from 产量数据 is the repo root.
    project_root = script_dir.parents[1]
    data_dir = script_dir.parent / "原始"
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(_default_source(script_dir)))
    parser.add_argument("--name-map", default=str(project_root / "数据" / "映射" / "作物名称映射.csv"))
    parser.add_argument("--out-history", default=str(data_dir / "产量历史数据.csv"))
    parser.add_argument("--out-table", default=str(data_dir / "单位面积产量表.csv"))
    args = parser.parse_args()

    prepare(
        source_path=Path(args.source),
        map_path=Path(args.name_map),
        out_history=Path(args.out_history),
        out_table=Path(args.out_table),
    )
