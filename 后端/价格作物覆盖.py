from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


_META_KEYS = {"created_at", "expires_at", "owner", "reason", "params"}


def _normalize_prediction_mode(mode: Any) -> str:
    raw = str(mode or "").strip().lower()
    if raw in {"return_recursive_v3", "return_recursive_v2", "price_recursive_v1", "direct_horizon_v1"}:
        return raw
    return "return_recursive_v3"


def _safe_timestamp(value: Any) -> Optional[pd.Timestamp]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        ts = pd.Timestamp(text)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if isinstance(ts, pd.Timestamp) and ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _extract_params_and_meta(selected: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(selected, dict):
        return {}, {}
    if isinstance(selected.get("params"), dict):
        params = {str(k): v for k, v in selected["params"].items()}
    else:
        params = {str(k): v for k, v in selected.items() if str(k) not in _META_KEYS}
    meta = {
        "created_at": str(selected.get("created_at") or "").strip(),
        "expires_at": str(selected.get("expires_at") or "").strip(),
        "owner": str(selected.get("owner") or "unknown").strip() or "unknown",
        "reason": str(selected.get("reason") or "").strip(),
    }
    return params, meta


def resolve_price_cfg_for_crop(
    price_cfg: Dict[str, Any],
    crop: str,
    *,
    prediction_mode: str = "",
    as_of_date: Optional[Any] = None,
    include_status: bool = False,
) -> Any:
    base = deepcopy(price_cfg or {})
    mode = _normalize_prediction_mode(prediction_mode or base.get("prediction_mode", "return_recursive_v3"))
    crop_key = str(crop or "").strip().lower()
    status: Dict[str, Any] = {
        "crop": crop_key,
        "prediction_mode": mode,
        "override_enabled": bool(mode == "return_recursive_v3"),
        "strict_override_expiry": bool(base.get("strict_override_expiry", False)),
        "applied": False,
        "applied_keys": [],
        "warnings": [],
        "metadata": None,
        "reason": "",
    }
    if not crop_key:
        status["reason"] = "empty_crop"
        return (base, [], status) if include_status else (base, [])
    if mode != "return_recursive_v3":
        status["reason"] = "prediction_mode_not_v3"
        return (base, [], status) if include_status else (base, [])

    overrides_map = (
        base.get("per_crop_overrides")
        if isinstance(base.get("per_crop_overrides"), dict)
        else {}
    )
    if not overrides_map:
        status["reason"] = "no_per_crop_overrides"
        return (base, [], status) if include_status else (base, [])

    applied: List[str] = []
    selected = overrides_map.get(crop_key)
    if not isinstance(selected, dict):
        status["reason"] = "crop_override_missing"
        return (base, applied, status) if include_status else (base, applied)

    params, meta = _extract_params_and_meta(selected)
    status["metadata"] = dict(meta)
    as_of_ts = _safe_timestamp(as_of_date) or pd.Timestamp.utcnow().tz_localize(None).normalize()

    created_ts = _safe_timestamp(meta.get("created_at"))
    expires_ts = _safe_timestamp(meta.get("expires_at"))
    if created_ts is not None and expires_ts is None:
        expires_ts = created_ts + pd.Timedelta(days=30)
    if expires_ts is not None:
        status["metadata"]["expires_at_resolved"] = expires_ts.strftime("%Y-%m-%d")
    if created_ts is None:
        status["warnings"].append(
            {
                "level": "warning",
                "code": "override_metadata_missing_created_at",
                "crop": crop_key,
                "message": f"{crop_key} override missing/invalid created_at",
            }
        )
    if expires_ts is None:
        status["warnings"].append(
            {
                "level": "warning",
                "code": "override_metadata_missing_expires_at",
                "crop": crop_key,
                "message": f"{crop_key} override missing/invalid expires_at",
            }
        )

    strict = bool(base.get("strict_override_expiry", False))
    allow_apply = True
    if expires_ts is not None:
        days_to_expiry = int((expires_ts - as_of_ts).days)
        status["days_to_expiry"] = int(days_to_expiry)
        if days_to_expiry < 0:
            overdue_days = int(abs(days_to_expiry))
            status["warnings"].append(
                {
                    "level": "error" if strict else "warning",
                    "code": "override_expired_strict_block" if strict else "override_expired",
                    "crop": crop_key,
                    "overdue_days": overdue_days,
                    "message": (
                        f"{crop_key} override expired {overdue_days} day(s) ago; "
                        "run 历史/experiments/scripts/targeted_tune_three_crops.py for re-evaluation"
                    ),
                }
            )
            if strict:
                allow_apply = False
        elif days_to_expiry <= 7:
            status["warnings"].append(
                {
                    "level": "warning",
                    "code": "override_expiring_soon",
                    "crop": crop_key,
                    "days_to_expiry": int(days_to_expiry),
                    "message": (
                        f"{crop_key} override expires in {days_to_expiry} day(s); "
                        "schedule targeted re-evaluation"
                    ),
                }
            )

    if allow_apply:
        for k, v in params.items():
            base[str(k)] = v
            applied.append(str(k))
        status["applied"] = bool(applied)
        status["applied_keys"] = list(applied)
        status["reason"] = "applied" if applied else "no_effective_params"
    else:
        status["applied"] = False
        status["applied_keys"] = []
        status["reason"] = "expired_strict_block"
    return (base, applied, status) if include_status else (base, applied)
