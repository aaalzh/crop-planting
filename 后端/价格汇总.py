from __future__ import annotations

from typing import Any, Dict, List, Optional


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def summarize_forecast_tail(
    forecast_rows: List[Dict[str, Any]],
    *,
    window_days: int,
    fallback_price: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    safe_window = max(1, int(window_days))
    rows = [row for row in (forecast_rows or []) if isinstance(row, dict)]
    tail = rows[-safe_window:]

    def _mean(key: str, *, fallback_key: Optional[str] = None) -> Optional[float]:
        values: List[float] = []
        for row in tail:
            value = _safe_float(row.get(key))
            if value is None and fallback_key:
                value = _safe_float(row.get(fallback_key))
            if value is not None:
                values.append(float(value))
        if not values:
            return None
        return float(sum(values) / len(values))

    price_p10 = _mean("p10")
    price_p50 = _mean("p50", fallback_key="value")
    price_p90 = _mean("p90")
    price_pred = price_p50 if price_p50 is not None else _safe_float(fallback_price)

    if price_p50 is None:
        price_p50 = price_pred

    return {
        "window_days": float(safe_window),
        "used_days": float(len(tail)),
        "price_pred": price_pred,
        "price_p10": price_p10,
        "price_p50": price_p50,
        "price_p90": price_p90,
    }
