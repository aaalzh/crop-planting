from __future__ import annotations

from typing import Any, Dict, List, Optional


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:  # NaN
        return None
    return out


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _score_higher_better(
    value: Optional[float],
    *,
    excellent: float,
    acceptable: float,
    floor: float = 40.0,
) -> float:
    if value is None:
        return floor
    if value >= excellent:
        return 100.0
    if value <= acceptable:
        return floor
    ratio = (value - acceptable) / max(excellent - acceptable, 1e-9)
    return floor + ratio * (100.0 - floor)


def _score_lower_better(
    value: Optional[float],
    *,
    excellent: float,
    acceptable: float,
    floor: float = 40.0,
) -> float:
    if value is None:
        return floor
    if value <= excellent:
        return 100.0
    if value >= acceptable:
        return floor
    ratio = (acceptable - value) / max(acceptable - excellent, 1e-9)
    return floor + ratio * (100.0 - floor)


def evaluate_project_quality(
    env_summary: Dict[str, Any],
    price_summary: Dict[str, Any],
    cost_summary: Dict[str, Any],
    yield_summary: Dict[str, Any],
    prob_summary: Dict[str, Any],
) -> Dict[str, Any]:
    env_acc = _to_float(env_summary.get("accuracy"))
    price_mape = _to_float(price_summary.get("mape_mean"))
    cost_mape = _to_float(cost_summary.get("mape_mean"))
    yield_mape = _to_float(yield_summary.get("mape"))
    cal_ece = _to_float(prob_summary.get("cv_mean_ece"))
    cal_logloss = _to_float(prob_summary.get("cv_mean_logloss"))

    # Availability / coverage score.
    price_n = _to_float(price_summary.get("n_crops"))
    cost_n = _to_float(cost_summary.get("n_crops"))
    yield_ok = 1.0 if yield_mape is not None else 0.0
    prob_ok = 1.0 if cal_ece is not None else 0.0
    coverage_ratio = min(
        1.0,
        (
            (0.0 if price_n is None else min(price_n / 16.0, 1.0))
            + (0.0 if cost_n is None else min(cost_n / 6.0, 1.0))
            + yield_ok
            + prob_ok
        )
        / 4.0,
    )

    dimensions: List[Dict[str, Any]] = [
        {
            "key": "env_accuracy",
            "label": "环境识别",
            "weight": 0.24,
            "value": env_acc,
            "score": _score_higher_better(env_acc, excellent=0.995, acceptable=0.90),
            "direction": "higher_better",
            "unit": "ratio",
        },
        {
            "key": "price_mape",
            "label": "价格预测",
            "weight": 0.18,
            "value": price_mape,
            # Price MAPE has crop-level long tails; keep scoring informative but less punitive in mid-range.
            "score": _score_lower_better(price_mape, excellent=0.07, acceptable=0.36, floor=45.0),
            "direction": "lower_better",
            "unit": "ratio",
        },
        {
            "key": "cost_mape",
            "label": "成本预测",
            "weight": 0.14,
            "value": cost_mape,
            "score": _score_lower_better(cost_mape, excellent=0.08, acceptable=0.30),
            "direction": "lower_better",
            "unit": "ratio",
        },
        {
            "key": "yield_mape",
            "label": "产量预测",
            "weight": 0.14,
            "value": yield_mape,
            "score": _score_lower_better(yield_mape, excellent=0.12, acceptable=0.35),
            "direction": "lower_better",
            "unit": "ratio",
        },
        {
            "key": "calibration_ece",
            "label": "概率校准(ECE)",
            "weight": 0.14,
            "value": cal_ece,
            "score": _score_lower_better(cal_ece, excellent=0.005, acceptable=0.08),
            "direction": "lower_better",
            "unit": "ratio",
        },
        {
            "key": "calibration_logloss",
            "label": "概率校准(LogLoss)",
            "weight": 0.08,
            "value": cal_logloss,
            "score": _score_lower_better(cal_logloss, excellent=0.05, acceptable=0.45),
            "direction": "lower_better",
            "unit": "ratio",
        },
        {
            "key": "artifact_coverage",
            "label": "产物覆盖度",
            "weight": 0.08,
            "value": coverage_ratio,
            "score": _score_higher_better(coverage_ratio, excellent=0.95, acceptable=0.60),
            "direction": "higher_better",
            "unit": "ratio",
        },
    ]

    weighted_sum = sum(dim["score"] * float(dim["weight"]) for dim in dimensions)
    overall = round(_clamp(weighted_sum, 0.0, 100.0), 2)
    target = 90.0

    weakest = sorted(dimensions, key=lambda x: x["score"])[:3]
    focus = [dim["label"] for dim in weakest]

    advice_map = {
        "环境识别": "环境模型建议增加近年样本并做分层重采样，提升低频类别稳定性。",
        "价格预测": "价格模型建议强化节假日/季节特征并做滚动重训，压低 MAPE。",
        "成本预测": "成本模型建议继续做趋势约束和异常值鲁棒回归，降低波动放大。",
        "产量预测": "产量模型建议引入作物-年份交互与稳健先验融合，提升长期外推能力。",
        "概率校准(ECE)": "校准器建议扩充历史样本并监控校准曲线，避免过度自信。",
        "概率校准(LogLoss)": "校准器建议优化特征覆盖并减少标签噪声，提升概率分辨率。",
        "产物覆盖度": "建议补齐缺失作物模型并统一产物版本，减少线上降级预测。",
    }

    actions: List[str] = []
    for dim in weakest:
        label = str(dim["label"])
        if dim["score"] >= 90.0:
            continue
        action = advice_map.get(label)
        if action and action not in actions:
            actions.append(action)
    if not actions:
        actions.append("当前评分已达到目标，建议保持每周离线回训并持续监控漂移。")

    grade = "A+" if overall >= 95 else "A" if overall >= 90 else "B" if overall >= 80 else "C"
    return {
        "overall_score": overall,
        "target_score": target,
        "target_met": bool(overall >= target),
        "grade": grade,
        "dimensions": dimensions,
        "focus_areas": focus,
        "next_actions": actions,
    }
