from typing import Dict
import numpy as np
import pandas as pd


def price_volatility(df: pd.DataFrame, window_days: int = 90) -> float:
    if df.empty:
        return 0.0
    df = df.sort_values("date")
    cutoff = df["date"].max() - pd.Timedelta(days=window_days)
    df = df[df["date"] >= cutoff]
    if len(df) < 5:
        return 0.0
    series = df["modal_price"].astype(float)
    mu = float(series.mean())
    sigma = float(series.std())
    if mu <= 0:
        return 0.0
    return float(sigma / mu)


def risk_score(volatility: float, env_confidence: str, ood_warnings: list) -> float:
    # base from volatility
    v = min(max(volatility, 0.0), 2.0)
    v = min(v / 0.5, 1.0)  # 0.5 volatility -> 1.0 risk

    conf_map = {"高": 0.0, "中": 0.2, "低": 0.4, "high": 0.0, "mid": 0.2, "low": 0.4}
    c = conf_map.get(str(env_confidence).strip(), 0.2)

    ood = 0.3 if ood_warnings else 0.0

    return float(min(1.0, v + c + ood))
