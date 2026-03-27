import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


STANDARD_COLS = ["date", "modal_price", "min_price", "max_price", "change"]


def _norm_col(s: str) -> str:
    return (
        str(s)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def _guess_column_map(columns: List[str]) -> Dict[str, str]:
    norm_map = {_norm_col(c): c for c in columns}
    candidates = {
        "date": ["date", "arrivaldate", "reporteddate"],
        "modal_price": ["modalprice", "modal", "price", "modelprice"],
        "min_price": ["minprice", "minimumprice", "min"],
        "max_price": ["maxprice", "maximumprice", "max"],
        "change": ["change", "pricechange", "variation"],
    }

    out: Dict[str, str] = {}
    used = set()
    for std, keys in candidates.items():
        for k in keys:
            if k in norm_map and norm_map[k] not in used:
                out[std] = norm_map[k]
                used.add(norm_map[k])
                break

    # Fallback: if required columns are still missing and file has at least 5 columns,
    # use first 5 columns in standard order.
    if len(out) < len(STANDARD_COLS) and len(columns) >= 5:
        for std, src in zip(STANDARD_COLS, columns[:5]):
            out.setdefault(std, src)

    return out


def _clean_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("Rs.", "", regex=False)
        .str.replace("Rs", "", regex=False)
        .str.replace("INR", "", regex=False)
        .str.replace("\u20b9", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def _clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    col_map = _guess_column_map(list(df.columns))
    missing = [c for c in STANDARD_COLS if c not in col_map]
    if missing:
        raise ValueError(f"missing required columns after mapping: {missing}")

    out = pd.DataFrame({k: df[v] for k, v in col_map.items()})
    out["date_dt"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")
    out = out.dropna(subset=["date_dt"])

    for c in ["modal_price", "min_price", "max_price", "change"]:
        out[c] = _clean_number(out[c])

    out = out.dropna(subset=["modal_price"])
    out = out.sort_values("date_dt").drop_duplicates(subset=["date_dt"], keep="last")
    out["date"] = out["date_dt"].dt.strftime("%d/%m/%Y")
    out = out.drop(columns=["date_dt"]).reset_index(drop=True)
    return out[STANDARD_COLS]


def _iter_input_files(source_dir: Path, source_file: Optional[Path]) -> List[Path]:
    if source_file is not None:
        return [source_file]

    files = []
    for p in sorted(source_dir.glob("*.csv")):
        name = p.name.lower()
        if name.endswith("_cleaned.csv"):
            continue
        if name.startswith("price_backtest"):
            continue
        files.append(p)
    return files


def process_file(in_path: Path, out_path: Path) -> Tuple[int, int]:
    df = pd.read_csv(in_path)
    before = len(df)
    cleaned = _clean_price_df(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False, encoding="utf-8")
    return before, len(cleaned)


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "原始"
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(data_dir))
    parser.add_argument("--source-file", default="")
    parser.add_argument("--out-dir", default=str(data_dir))
    parser.add_argument(
        "--suffix",
        default="",
        help="optional filename suffix before .csv, e.g. _cleaned",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    source_file = Path(args.source_file) if args.source_file else None

    files = _iter_input_files(source_dir, source_file)
    if not files:
        raise ValueError(f"no csv files found in {source_dir}")

    ok = 0
    skipped = 0
    for f in files:
        out_name = f"{f.stem}{args.suffix}.csv" if args.suffix else f.name
        out_path = out_dir / out_name
        try:
            before, after = process_file(f, out_path)
            print(f"[ok] {f.name}: {before} -> {after} rows -> {out_path}")
            ok += 1
        except Exception as e:
            print(f"[skip] {f.name}: {e}")
            skipped += 1

    print(f"done: ok={ok}, skipped={skipped}, out_dir={out_dir}")


if __name__ == "__main__":
    main()
