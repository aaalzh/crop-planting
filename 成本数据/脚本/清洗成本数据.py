import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _split_fiscal_year(x) -> Tuple[Optional[int], Optional[int]]:
    if pd.isna(x):
        return None, None
    s = str(x).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        s = s[:7]
    if re.fullmatch(r"\d{4}", s):
        y = int(s)
        return y, y
    m = re.fullmatch(r"(\d{4})\s*-\s*(\d{2}|\d{4})", s)
    if m:
        y1 = int(m.group(1))
        y2_raw = m.group(2)
        y2 = int(str(y1)[:2] + y2_raw) if len(y2_raw) == 2 else int(y2_raw)
        return y1, y2
    return None, None


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype="float64")
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def _default_source_file(script_dir: Path) -> Path:
    data_dir = script_dir.parent
    raw_dir = data_dir / "原始"
    preferred = raw_dir / "成本源数据.xlsx"
    if preferred.exists():
        return preferred
    for p in sorted(raw_dir.glob("*.xlsx")):
        return p
    for p in sorted(raw_dir.glob("*.csv")):
        if "加权平均成本数据" in p.name:
            continue
        return p
    return preferred


def _default_out_file(script_dir: Path) -> Path:
    return script_dir.parent / "原始" / "加权平均成本数据.csv"


def _default_stats_out_file(script_dir: Path) -> Path:
    return script_dir.parent / "内部" / "加权平均成本统计.csv"


def _load_source(source_file: Path, sheet=0) -> pd.DataFrame:
    if not source_file.exists():
        raise FileNotFoundError(f"source file not found: {source_file}")
    suffix = source_file.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source_file, sheet_name=sheet)
    return pd.read_csv(source_file)


def build_weighted_cost(
    source_file: Path,
    out_file: Path,
    stats_out_file: Optional[Path] = None,
    sheet=0,
    crop_col: str = "crop_name",
    year_col: str = "year",
    cost_col: str = "cul_cost_c2",
    weight_col: str = "num_holdings_sample",
) -> None:
    df = _load_source(source_file=source_file, sheet=sheet)

    required = [crop_col, year_col, cost_col, weight_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}; available={list(df.columns)}")

    all_df = df.copy()
    all_df[crop_col] = all_df[crop_col].astype(str).str.strip()
    all_df = all_df[all_df[crop_col] != ""].copy()

    years = all_df[year_col].apply(_split_fiscal_year)
    all_df["year_start"] = years.apply(lambda t: t[0])
    all_df["year_end"] = years.apply(lambda t: t[1])
    all_df = all_df.dropna(subset=["year_start"]).copy()
    all_df["year_start"] = all_df["year_start"].astype(int)

    all_df[cost_col] = pd.to_numeric(all_df[cost_col], errors="coerce")
    all_df[weight_col] = pd.to_numeric(all_df[weight_col], errors="coerce")

    rows = []
    for (crop_name, year_start), g in all_df.groupby([crop_col, "year_start"], dropna=False):
        rows.append(
            {
                "crop_name": crop_name,
                "year_start": int(year_start),
                "india_cost_unweighted": float(np.nanmean(g[cost_col].values)),
                "india_cost_wavg_sample": _weighted_mean(g[cost_col], g[weight_col]),
                "n_states": int(g["state_name"].nunique()) if "state_name" in g.columns else np.nan,
                "n_rows_cost": int(g[cost_col].notna().sum()),
                "sum_sample_weight": float(np.nansum(g[weight_col].values)),
            }
        )

    out = pd.DataFrame(rows).sort_values(["crop_name", "year_start"]).reset_index(drop=True)
    public_cols = ["crop_name", "year_start", "india_cost_unweighted", "india_cost_wavg_sample"]
    stats_cols = ["crop_name", "year_start", "n_states", "n_rows_cost", "sum_sample_weight"]
    public_out = out[public_cols].copy()
    stats_out = out[stats_cols].copy()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    public_out.to_csv(out_file, index=False, encoding="utf-8")
    if stats_out_file is not None:
        stats_out_file.parent.mkdir(parents=True, exist_ok=True)
        stats_out.to_csv(stats_out_file, index=False, encoding="utf-8")

    print(f"source={source_file}")
    print(f"saved={out_file}")
    if stats_out_file is not None:
        print(f"saved_internal_stats={stats_out_file}")
    print(f"rows={len(public_out)}, crops={public_out['crop_name'].nunique()}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(_default_source_file(script_dir)))
    parser.add_argument("--sheet", default=0, help="excel sheet index or name")
    parser.add_argument("--out", default=str(_default_out_file(script_dir)))
    parser.add_argument("--stats-out", default=str(_default_stats_out_file(script_dir)))
    parser.add_argument("--crop-col", default="crop_name")
    parser.add_argument("--year-col", default="year")
    parser.add_argument("--cost-col", default="cul_cost_c2")
    parser.add_argument("--weight-col", default="num_holdings_sample")
    args = parser.parse_args()

    source_file = Path(args.source)
    if not source_file.is_absolute():
        source_file = script_dir.parent / source_file

    out_file = Path(args.out)
    if not out_file.is_absolute():
        out_file = script_dir.parent / out_file

    stats_out_file = Path(args.stats_out)
    if not stats_out_file.is_absolute():
        stats_out_file = script_dir.parent / stats_out_file

    sheet = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)

    build_weighted_cost(
        source_file=source_file,
        out_file=out_file,
        stats_out_file=stats_out_file,
        sheet=sheet,
        crop_col=args.crop_col,
        year_col=args.year_col,
        cost_col=args.cost_col,
        weight_col=args.weight_col,
    )


if __name__ == "__main__":
    main()
