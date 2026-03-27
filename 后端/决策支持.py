from __future__ import annotations

import copy
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from 后端.深度求索客户端 import llm_client_ready, request_llm_chat


class AssistantUnavailableError(RuntimeError):
    def __init__(self, *, code: str, detail: str, status_code: int) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail
        self.status_code = int(status_code)


def _assistant_error_from_llm(exc: Exception) -> AssistantUnavailableError:
    text = str(exc or "")
    if text.startswith("llm_http_error:") or text.startswith("deepseek_http_error:"):
        parts = text.split(":", 2)
        status_code = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 502
        detail = parts[2].strip() if len(parts) > 2 else ""
        if status_code == 401:
            return AssistantUnavailableError(
                code="llm_auth_failed",
                detail="AI 服务暂时不可用，请稍后重试。",
                status_code=401,
            )
        if status_code == 429:
            return AssistantUnavailableError(
                code="llm_rate_limited",
                detail=detail or "AI 服务当前较忙，请稍后重试。",
                status_code=429,
            )
        return AssistantUnavailableError(
            code="llm_failed",
            detail=f"AI 服务调用失败{f'：{detail}' if detail else '。'}",
            status_code=status_code,
        )
    if "429" in text:
        return AssistantUnavailableError(
            code="llm_rate_limited",
            detail="AI 服务当前较忙，请稍后重试。",
            status_code=429,
        )
    if "missing_api_key" in text or "llm_disabled" in text or "llm_unavailable" in text:
        return AssistantUnavailableError(
            code="llm_unavailable",
            detail="AI 服务暂未配置，请联系管理员。",
            status_code=503,
        )
    if "llm_network_error" in text or "deepseek_network_error" in text or "network" in text:
        return AssistantUnavailableError(
            code="llm_network_error",
            detail="AI 服务当前不可达，请稍后重试。",
            status_code=503,
        )
    if "llm_empty_answer" in text:
        return AssistantUnavailableError(
            code="llm_empty_answer",
            detail="AI 服务暂时没有返回有效内容，请稍后重试。",
            status_code=502,
        )
    return AssistantUnavailableError(
        code="llm_failed",
        detail="AI 服务调用失败，请稍后重试。",
        status_code=502,
    )


CROP_LABELS = {
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

CROP_CATEGORIES = {
    "apple": "fruit",
    "banana": "fruit",
    "blackgram": "pulse",
    "chickpea": "pulse",
    "coconut": "plantation",
    "coffee": "plantation",
    "cotton": "fiber",
    "grapes": "fruit",
    "jute": "fiber",
    "kidneybeans": "pulse",
    "lentil": "pulse",
    "maize": "grain",
    "mango": "fruit",
    "mothbeans": "pulse",
    "mungbean": "pulse",
    "muskmelon": "fruit",
    "orange": "fruit",
    "papaya": "fruit",
    "pigeonpeas": "pulse",
    "pomegranate": "fruit",
    "rice": "grain",
    "watermelon": "fruit",
}

QUESTION_PRESETS = [
    {"id": "why_recommended", "label": "为什么推荐这个作物？"},
    {"id": "risk_sources", "label": "风险主要来自哪里？"},
    {"id": "stable_option", "label": "哪个候选更稳？"},
    {"id": "profit_option", "label": "哪个候选可能赚得更多？"},
    {"id": "low_budget", "label": "如果预算更少，应该选哪个？"},
    {"id": "low_risk", "label": "如果更看重稳妥，应该选哪个？"},
    {"id": "price_trend", "label": "最近价格走势说明了什么？"},
]

STORE_TEMPLATES = {
    "grain": {
        "bundle_title": "粮食稳产基础包",
        "budget": "基础预算",
        "items": [
            "高发芽率良种",
            "底肥组合（氮磷钾平衡型）",
            "播前土壤改良剂",
            "基础病虫害防控包",
        ],
    },
    "pulse": {
        "bundle_title": "豆类稳健起步包",
        "budget": "控本预算",
        "items": [
            "精选豆类种子",
            "根瘤菌剂与拌种包",
            "控草基础组合",
            "追肥小规格补充包",
        ],
    },
    "fruit": {
        "bundle_title": "果类收益提升包",
        "budget": "进阶预算",
        "items": [
            "优选果类种苗",
            "保花保果营养组合",
            "滴灌/喷施基础工具",
            "品质提升叶面肥",
        ],
    },
    "fiber": {
        "bundle_title": "纤维作物效率包",
        "budget": "中等预算",
        "items": [
            "优质种子",
            "长势促进套餐",
            "中后期补肥包",
            "采收前风险检查清单",
        ],
    },
    "plantation": {
        "bundle_title": "经济作物耐久管理包",
        "budget": "长期预算",
        "items": [
            "优选苗木/种苗",
            "土壤改良与保水材料",
            "病害预警检查包",
            "年度养分补充方案",
        ],
    },
}

COMMUNITY_TEMPLATES = {
    "grain": [
        {
            "title": "播种窗口怎么把握更稳？",
            "tag": "稳产经验",
            "summary": "重点看土壤含水量、短期降雨和底肥配比，避免一次性加大面积。",
        },
        {
            "title": "控本情况下先保哪些投入？",
            "tag": "控本策略",
            "summary": "优先保证种子质量和底肥，再决定是否加码增产型投入。",
        },
    ],
    "pulse": [
        {
            "title": "豆类作物如何降低前期失败率？",
            "tag": "实战经验",
            "summary": "建议先核验土壤酸碱度和拌种环节，再决定是否扩大种植面积。",
        },
        {
            "title": "价格波动大时怎样留出回旋余地？",
            "tag": "行情判断",
            "summary": "可以保留替代作物备选，并按两阶段采购农资减少一次性押注。",
        },
    ],
    "fruit": [
        {
            "title": "高收益果类为什么更需要分阶段投入？",
            "tag": "收益策略",
            "summary": "果类通常收益空间更高，但品质管理和中后期投入更决定最终利润。",
        },
        {
            "title": "什么时候该优先看价格趋势而不是环境适配？",
            "tag": "决策方法",
            "summary": "当候选作物环境匹配都不差时，应重点看行情趋势、波动和现金流压力。",
        },
    ],
    "fiber": [
        {
            "title": "纤维作物怎么判断是稳还是赚？",
            "tag": "候选比较",
            "summary": "重点看价格波动、成本占比和保守收益，不要只看单点利润。",
        },
        {
            "title": "高波动作物是否适合新手？",
            "tag": "风险提示",
            "summary": "通常更适合有一定风险承受能力、能持续跟踪行情的人群。",
        },
    ],
    "plantation": [
        {
            "title": "经济作物为何要更关注长期管理成本？",
            "tag": "长期经营",
            "summary": "长期类作物的收益不只看售价，还要看持续养护、病害和现金周转。",
        },
        {
            "title": "如何判断是否值得继续追踪这个作物？",
            "tag": "跟踪建议",
            "summary": "先看价格方向和环境匹配，再看风险来源是否可控。",
        },
    ],
}

DEFAULT_SETTINGS = [
    {"label": "默认展示单位", "value": "价格按 ₹/公担，成本按 ₹/公顷"},
    {"label": "首页偏好", "value": "优先展示最近一次推荐摘要"},
    {"label": "风险提醒", "value": "高风险候选作物自动高亮"},
]


def _safe_float(value: Any) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    if val != val:
        return None
    return val


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _crop_label(crop: str) -> str:
    key = str(crop or "").strip()
    return CROP_LABELS.get(key, key or "-")


def _money_text(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "-"
    return f"₹{value:,.{digits}f}"


def _pct_text(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}%"


def _risk_level(risk: float | None) -> str:
    score = _safe_float(risk)
    if score is None:
        return "未知"
    if score <= 0.28:
        return "低"
    if score <= 0.58:
        return "中"
    return "高"


def _fit_level(prob: float | None) -> str:
    score = _safe_float(prob)
    if score is None:
        return "未知"
    if score >= 0.75:
        return "很强"
    if score >= 0.45:
        return "较强"
    return "一般"


def _probability_strength(prob: float | None, *, default: float = 42.0) -> float:
    score = _safe_float(prob)
    if score is None:
        return _clamp(default, 3.0, 97.0)
    softened = 0.03 + _clamp(score, 0.0, 1.0) * 0.94
    return _clamp(softened * 100.0, 3.0, 97.0)


def _risk_safety_strength(risk: float | None) -> float:
    val = max(0.0, _safe_float(risk) or 0.0)
    baseline = 100.0 / (1.0 + math.pow(val / 0.2, 1.25))
    return _clamp(baseline, 3.0, 97.0)


def _margin_strength(margin_pct: float | None, *, default: float = 52.0) -> float:
    margin = _safe_float(margin_pct)
    if margin is None:
        return _clamp(default, 3.0, 97.0)
    centered = (margin - 10.0) / 18.0
    score = 50.0 + 42.0 * math.tanh(centered)
    return _clamp(score, 3.0, 97.0)


def _score_signal_strength(score: float | None, *, default: float = 50.0) -> float:
    raw = _safe_float(score)
    if raw is None:
        return _clamp(default, 3.0, 97.0)
    scaled = 50.0 + 45.0 * math.tanh(raw / 18000.0)
    return _clamp(scaled, 3.0, 97.0)


def _strength_level(score: float | None) -> str:
    val = _safe_float(score)
    if val is None:
        return "未知"
    if val >= 82.0:
        return "很强"
    if val >= 68.0:
        return "较强"
    if val >= 55.0:
        return "中等"
    return "谨慎"


def _recommend_strength(row: Dict[str, Any], *, margin_pct: float | None = None) -> Dict[str, Any]:
    margin = _safe_float(margin_pct)
    if margin is None:
        margin = _margin_pct(row)

    env_strength = _probability_strength(row.get("env_prob"), default=42.0)
    confidence_strength = _probability_strength(
        row.get("prob_best"),
        default=max(38.0, env_strength * 0.76),
    )
    risk_safety = _risk_safety_strength(row.get("risk"))
    profit_buffer = _margin_strength(margin, default=52.0)
    score_signal = _score_signal_strength(row.get("score"), default=50.0)

    overall = (
        env_strength * 0.34
        + confidence_strength * 0.20
        + risk_safety * 0.20
        + profit_buffer * 0.14
        + score_signal * 0.12
    )
    overall = round(_clamp(overall, 0.0, 100.0), 1)
    level = _strength_level(overall)

    return {
        "overall": overall,
        "level": level,
        "summary": f"推荐强度 {overall:.1f}/100，属于{level}。",
        "components": {
            "env_fit": round(env_strength, 1),
            "decision_confidence": round(confidence_strength, 1),
            "risk_safety": round(risk_safety, 1),
            "profit_buffer": round(profit_buffer, 1),
            "rank_signal": round(score_signal, 1),
        },
    }


def _market_direction(value: float | None) -> str:
    val = _safe_float(value)
    if val is None:
        return "横盘"
    if val >= 3.0:
        return "走强"
    if val <= -3.0:
        return "走弱"
    return "横盘"


def _story_style(*, profit_rank: int, risk_rank: int, env_rank: int, risk: float | None) -> str:
    level = _risk_level(risk)
    if profit_rank == 1 and level in {"中", "高"}:
        return "更赚"
    if risk_rank == 1 or (level == "低" and env_rank <= 3):
        return "更稳"
    return "更平衡"


def _forecast_change_pct(row: Dict[str, Any]) -> float | None:
    forecast = row.get("price_forecast")
    if not isinstance(forecast, list) or len(forecast) < 2:
        return None
    first = forecast[0] if isinstance(forecast[0], dict) else {}
    last = forecast[-1] if isinstance(forecast[-1], dict) else {}
    first_val = _safe_float(first.get("p50"))
    if first_val is None:
        first_val = _safe_float(first.get("value"))
    last_val = _safe_float(last.get("p50"))
    if last_val is None:
        last_val = _safe_float(last.get("value"))
    if first_val is None or last_val is None or abs(first_val) < 1e-9:
        return None
    return ((last_val / first_val) - 1.0) * 100.0


def _revenue(row: Dict[str, Any]) -> float | None:
    price = _safe_float(row.get("price_pred"))
    yield_val = _safe_float(row.get("yield"))
    if price is None or yield_val is None:
        return None
    return price * yield_val


def _margin_pct(row: Dict[str, Any]) -> float | None:
    revenue = _revenue(row)
    profit = _safe_float(row.get("profit"))
    if revenue is None or revenue <= 0 or profit is None:
        return None
    return (profit / revenue) * 100.0


def _cost_ratio_pct(row: Dict[str, Any]) -> float | None:
    revenue = _revenue(row)
    cost = _safe_float(row.get("cost_pred"))
    if revenue is None or revenue <= 0 or cost is None:
        return None
    return (cost / revenue) * 100.0


def _profit_band(row: Dict[str, Any]) -> Dict[str, Any]:
    price_p10 = _safe_float(row.get("price_p10"))
    price_p50 = _safe_float(row.get("price_p50")) or _safe_float(row.get("price_pred"))
    price_p90 = _safe_float(row.get("price_p90"))
    yield_val = _safe_float(row.get("yield"))
    cost = _safe_float(row.get("cost_pred"))
    base_profit = _safe_float(row.get("profit"))

    conservative = None
    balanced = base_profit
    optimistic = None
    if yield_val is not None and cost is not None:
        if price_p10 is not None:
            conservative = price_p10 * yield_val - cost
        if price_p50 is not None and balanced is None:
            balanced = price_p50 * yield_val - cost
        if price_p90 is not None:
            optimistic = price_p90 * yield_val - cost

    width_pct = None
    if price_p10 is not None and price_p90 is not None and price_p50 is not None and abs(price_p50) > 1e-9:
        width_pct = ((price_p90 - price_p10) / price_p50) * 100.0

    return {
        "conservative": conservative,
        "balanced": balanced,
        "optimistic": optimistic,
        "width_pct": width_pct,
        "summary": (
            f"保守约 {_money_text(conservative)}，中性约 {_money_text(balanced)}，乐观约 {_money_text(optimistic)}"
            if conservative is not None or optimistic is not None
            else f"当前可参考中性收益 {_money_text(balanced)}"
        ),
    }


def _risk_breakdown(
    row: Dict[str, Any],
    *,
    env_warnings: List[str],
    env_confidence_norm: str,
) -> List[Dict[str, Any]]:
    env_prob = _safe_float(row.get("env_prob")) or 0.0
    risk = _safe_float(row.get("risk")) or 0.0
    volatility = _safe_float(row.get("volatility")) or 0.0
    interval = _profit_band(row)
    width_pct = _safe_float(interval.get("width_pct"))
    cost_ratio = _cost_ratio_pct(row)
    gap_info = ((row.get("time_alignment") or {}).get("gaps") or {}) if isinstance(row, dict) else {}

    confidence_penalty = {"high": 4.0, "mid": 18.0, "low": 34.0}.get(str(env_confidence_norm or "mid"), 18.0)
    price_risk = _clamp(volatility * 220.0, 0.0, 100.0)
    env_risk = _clamp((1.0 - env_prob) * 100.0 + confidence_penalty, 0.0, 100.0)
    cost_risk = _clamp((cost_ratio or 52.0) - 45.0, 0.0, 100.0)
    interval_risk = _clamp(width_pct or (risk * 100.0), 0.0, 100.0)
    data_gap = max(
        _safe_int(gap_info.get("price_gap_years")) or 0,
        _safe_int(gap_info.get("yield_gap_years")) or 0,
        _safe_int(gap_info.get("cost_gap_years")) or 0,
    )
    data_risk = _clamp(data_gap * 14.0 + (18.0 if env_warnings else 0.0), 0.0, 100.0)

    def _level(score: float) -> str:
        if score >= 65.0:
            return "高"
        if score >= 35.0:
            return "中"
        return "低"

    return [
        {
            "label": "价格波动风险",
            "score": round(price_risk, 1),
            "level": _level(price_risk),
            "detail": (
                f"近阶段价格波动水平约 {_pct_text(volatility * 100.0)}，行情越活跃，实际收益越容易偏离预期。"
                if volatility > 0
                else "当前缺少明显波动信号，价格层面的不确定性暂时不高。"
            ),
        },
        {
            "label": "环境匹配不确定性",
            "score": round(env_risk, 1),
            "level": _level(env_risk),
            "detail": f"当前环境匹配度约 {env_prob * 100.0:.0f} 分，匹配越弱，落地表现越容易打折。",
        },
        {
            "label": "成本压力",
            "score": round(cost_risk, 1),
            "level": _level(cost_risk),
            "detail": (
                f"预计成本约占收入的 {_pct_text(cost_ratio)}，成本占比越高，对现金流和利润缓冲越不友好。"
                if cost_ratio is not None
                else "当前缺少完整的收入成本占比，成本压力按中性水平提示。"
            ),
        },
        {
            "label": "预测区间不确定性",
            "score": round(interval_risk, 1),
            "level": _level(interval_risk),
            "detail": (
                f"价格区间宽度约 {_pct_text(width_pct)}，区间越宽，说明未来价格上下摆动空间越大。"
                if width_pct is not None
                else "当前缺少完整价格区间，预测区间不确定性按综合风险提示。"
            ),
        },
        {
            "label": "数据可信度提醒",
            "score": round(data_risk, 1),
            "level": _level(data_risk),
            "detail": (
                f"最近可用历史与目标年份存在约 {data_gap} 年空档，部分结论属于趋势外推。"
                if data_gap > 0
                else ("当前环境输入存在超出常见样本分布的提醒，建议把结果当作决策参考而非唯一依据。" if env_warnings else "当前没有明显的数据补齐风险提醒。")
            ),
        },
    ]


def _top_risk_labels(items: List[Dict[str, Any]], limit: int = 2) -> List[str]:
    ordered = sorted(items, key=lambda item: _safe_float(item.get("score")) or 0.0, reverse=True)
    return [str(item.get("label")) for item in ordered[:limit] if item.get("label")]


def _clean_text(text: str) -> str:
    return "\n".join([line.rstrip() for line in str(text or "").strip().splitlines() if line.strip()]).strip()


def _truncate_text(text: str, limit: int = 240) -> str:
    raw = _clean_text(text)
    if len(raw) <= limit:
        return raw
    return f"{raw[: max(0, limit - 1)].rstrip()}…"


class DecisionSupportService:
    def __init__(self, *, root: Path, output_dir: Path, logger: Any | None = None) -> None:
        self.root = root
        self.output_dir = output_dir
        self.logger = logger
        self.default_env_path = self.root / "数据" / "环境示例.json"
        self.history_path = self.output_dir / "推荐历史.json"

    def _safe_read_json(self, path: Path) -> Dict[str, Any] | List[Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            try:
                return json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception:
                return None

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _ranking_maps(self, rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        def _rank(key: str, reverse: bool) -> Dict[str, int]:
            sortable = [
                (str(row.get("crop") or ""), _safe_float(row.get(key)))
                for row in rows
                if str(row.get("crop") or "").strip()
            ]
            sortable = [(crop, value) for crop, value in sortable if value is not None]
            sortable.sort(key=lambda item: item[1], reverse=reverse)
            return {crop: idx + 1 for idx, (crop, _) in enumerate(sortable)}

        return {
            "profit": _rank("profit", True),
            "risk": _rank("risk", False),
            "env_prob": _rank("env_prob", True),
            "cost_pred": _rank("cost_pred", False),
        }

    def _comparison_marks(self, row: Dict[str, Any], rows: List[Dict[str, Any]], ranking_maps: Dict[str, Dict[str, int]]) -> List[str]:
        crop = str(row.get("crop") or "")
        marks: List[str] = []
        if ranking_maps["profit"].get(crop) == 1:
            marks.append("候选中收益上限最高")
        if ranking_maps["risk"].get(crop) == 1:
            marks.append("候选中最稳")
        if ranking_maps["env_prob"].get(crop) == 1:
            marks.append("环境匹配最强")
        if ranking_maps["cost_pred"].get(crop) == 1:
            marks.append("初始成本压力最轻")

        trend_pct = _forecast_change_pct(row)
        if trend_pct is not None:
            if trend_pct >= 4.0:
                marks.append("价格短期趋势偏强")
            elif trend_pct <= -4.0:
                marks.append("短期价格需要继续观察")

        if not marks:
            risk_rank = ranking_maps["risk"].get(crop, len(rows))
            profit_rank = ranking_maps["profit"].get(crop, len(rows))
            if risk_rank <= max(2, min(3, len(rows))):
                marks.append("风险水平处于候选前列")
            elif profit_rank <= max(2, min(3, len(rows))):
                marks.append("收益空间处于候选前列")
            else:
                marks.append("属于均衡型备选")
        return marks[:3]

    def _build_row_story(
        self,
        row: Dict[str, Any],
        rows: List[Dict[str, Any]],
        *,
        env: Dict[str, Any],
        env_confidence_norm: str,
        ranking_maps: Dict[str, Dict[str, int]],
    ) -> Dict[str, Any]:
        crop = str(row.get("crop") or "")
        crop_label = _crop_label(crop)
        profit_rank = ranking_maps["profit"].get(crop, len(rows))
        risk_rank = ranking_maps["risk"].get(crop, len(rows))
        env_rank = ranking_maps["env_prob"].get(crop, len(rows))
        style = _story_style(
            profit_rank=profit_rank,
            risk_rank=risk_rank,
            env_rank=env_rank,
            risk=_safe_float(row.get("risk")),
        )

        fit_prob = _safe_float(row.get("env_prob"))
        fit_text = _fit_level(fit_prob)
        risk_level = _risk_level(_safe_float(row.get("risk")))
        trend_pct = _forecast_change_pct(row)
        direction = _market_direction(trend_pct)
        margin_pct = _margin_pct(row)
        profit_band = _profit_band(row)
        risk_breakdown = _risk_breakdown(
            row,
            env_warnings=list(env.get("warnings") or []),
            env_confidence_norm=env_confidence_norm,
        )
        recommend_strength = _recommend_strength(row, margin_pct=margin_pct)

        one_liner = (
            f"{crop_label}更偏{style}，环境匹配{fit_text}，预计收益 {_money_text(_safe_float(row.get('profit')))}，总体风险{risk_level}。"
        )
        reasons = [
            f"环境适配：匹配度约 {fit_prob * 100.0:.0f} 分，属于{fit_text}，适合作为当前条件下的重点备选。"
            if fit_prob is not None
            else "环境适配：当前缺少完整适配评分，建议结合土壤和气候再确认。",
            f"收益判断：{profit_band['summary']}。"
            if profit_band.get("balanced") is not None
            else "收益判断：当前可先把它视为潜力候选，再结合行情变化确认种植面积。",
            (
                f"行情信号：未来价格窗口预计{direction}，变化约 {_pct_text(trend_pct)}。"
                if trend_pct is not None
                else "行情信号：当前缺少明确方向性变化，建议继续观察近期价格走势。"
            ),
        ]
        if margin_pct is not None:
            reasons.append(f"利润缓冲：预计利润率约 {_pct_text(margin_pct)}，利润缓冲越厚，对价格波动越有承受力。")

        comparison_marks = self._comparison_marks(row, rows, ranking_maps)
        next_steps = []
        if style == "更稳":
            next_steps.append("如果你更看重稳妥，可优先把它列为主选作物并尽快确认品种和底肥。")
        elif style == "更赚":
            next_steps.append("如果你愿意承担一定波动，可把它作为冲高收益方案，但建议先核算现金流。")
        else:
            next_steps.append("如果你想在稳妥和收益之间做平衡，可以把它作为默认方案继续比较。")

        if direction == "走弱":
            next_steps.append("短期价格信号偏弱，建议先控制面积或继续观察 1 到 2 周行情。")
        elif direction == "走强":
            next_steps.append("短期价格有走强迹象，可以同步查看农资包和历史趋势，尽快进入执行准备。")

        if risk_level == "高":
            next_steps.append("整体风险偏高，建议保留替代作物并优先小面积试种。")
        elif risk_level == "中":
            next_steps.append("风险可控但不低，适合继续看成本占比和价格区间，再决定投入力度。")
        else:
            next_steps.append("整体风险较低，适合进入更具体的播种、农资和时机确认。")

        return {
            "style": style,
            "one_liner": one_liner,
            "fit_summary": f"环境匹配{fit_text}，当前风险{risk_level}。",
            "recommend_strength": recommend_strength,
            "market_signal": {
                "direction": direction,
                "change_pct": trend_pct,
                "summary": (
                    f"短期价格预计{direction}，变化约 {_pct_text(trend_pct)}。"
                    if trend_pct is not None
                    else "短期价格暂无明确方向，建议继续跟踪。"
                ),
            },
            "profit_band": profit_band,
            "reasons": reasons[:4],
            "risk_breakdown": risk_breakdown,
            "risk_focus": _top_risk_labels(risk_breakdown),
            "comparison_marks": comparison_marks,
            "next_steps": next_steps[:4],
        }

    def _enrich_rows(
        self,
        rows: List[Dict[str, Any]],
        *,
        env: Dict[str, Any],
        env_confidence_norm: str,
    ) -> List[Dict[str, Any]]:
        ranking_maps = self._ranking_maps(rows)
        enriched: List[Dict[str, Any]] = []
        for row in rows:
            copied = copy.deepcopy(row)
            copied["crop_label"] = _crop_label(str(copied.get("crop") or ""))
            copied["margin_pct"] = _margin_pct(copied)
            copied["revenue_pred"] = _revenue(copied)
            profile = self._build_row_story(
                copied,
                rows,
                env=env,
                env_confidence_norm=env_confidence_norm,
                ranking_maps=ranking_maps,
            )
            copied["decision_profile"] = profile
            strength_meta = profile.get("recommend_strength", {}) if isinstance(profile, dict) else {}
            copied["recommend_strength"] = _safe_float(strength_meta.get("overall"))
            copied["recommend_strength_level"] = strength_meta.get("level")
            copied["recommend_strength_components"] = copy.deepcopy(strength_meta.get("components") or {})
            enriched.append(copied)
        return enriched

    def enrich_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        safe_payload = copy.deepcopy(payload or {})
        rows = [row for row in safe_payload.get("results", []) if isinstance(row, dict)]
        env = safe_payload.get("env", {}) if isinstance(safe_payload.get("env"), dict) else {}
        env_confidence_norm = str(safe_payload.get("env_confidence_norm") or "mid")
        safe_payload["results"] = self._enrich_rows(rows, env=env, env_confidence_norm=env_confidence_norm)
        safe_payload["decision_summary"] = self._build_decision_summary(safe_payload)
        safe_payload["quick_questions"] = copy.deepcopy(QUESTION_PRESETS)
        return safe_payload

    def _build_decision_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rows = [row for row in payload.get("results", []) if isinstance(row, dict)]
        best = rows[0] if rows else {}
        env = payload.get("env", {}) if isinstance(payload.get("env"), dict) else {}
        top_cards = []
        for idx, row in enumerate(rows[:5], start=1):
            profile = row.get("decision_profile", {}) if isinstance(row.get("decision_profile"), dict) else {}
            band = profile.get("profit_band", {}) if isinstance(profile.get("profit_band"), dict) else {}
            top_cards.append(
                {
                    "rank": idx,
                    "crop": row.get("crop"),
                    "crop_label": _crop_label(str(row.get("crop") or "")),
                    "style": profile.get("style"),
                    "recommend_strength": _safe_float(row.get("recommend_strength")),
                    "recommend_strength_level": row.get("recommend_strength_level"),
                    "profit": _safe_float(row.get("profit")),
                    "risk_level": _risk_level(_safe_float(row.get("risk"))),
                    "fit_level": _fit_level(_safe_float(row.get("env_prob"))),
                    "summary": profile.get("one_liner"),
                    "profit_band": {
                        "conservative": _safe_float(band.get("conservative")),
                        "balanced": _safe_float(band.get("balanced")),
                        "optimistic": _safe_float(band.get("optimistic")),
                    },
                }
            )

        best_profile = best.get("decision_profile", {}) if isinstance(best.get("decision_profile"), dict) else {}
        best_market = best_profile.get("market_signal", {}) if isinstance(best_profile.get("market_signal"), dict) else {}
        headline = (
            f"现在更适合优先考虑 {_crop_label(str(best.get('crop') or ''))}"
            if best
            else "先运行一次智能推荐，系统会给出今日优先候选"
        )
        return {
            "headline": headline,
            "subheadline": best_profile.get("one_liner") or "系统会综合环境、收益和风险给出决策建议。",
            "best_crop": {
                "crop": best.get("crop"),
                "crop_label": _crop_label(str(best.get("crop") or "")) if best else None,
                "style": best_profile.get("style"),
                "recommend_strength": _safe_float(best.get("recommend_strength")),
                "recommend_strength_level": best.get("recommend_strength_level"),
                "profit": _safe_float(best.get("profit")),
                "risk": _safe_float(best.get("risk")),
                "risk_level": _risk_level(_safe_float(best.get("risk"))),
                "env_prob": _safe_float(best.get("env_prob")),
                "margin_pct": _safe_float(best.get("margin_pct")),
                "market_summary": best_market.get("summary"),
                "reasons": best_profile.get("reasons") or [],
                "next_steps": best_profile.get("next_steps") or [],
            },
            "top_cards": top_cards,
            "risk_summary": {
                "level": _risk_level(_safe_float(best.get("risk"))),
                "focus": best_profile.get("risk_focus") or [],
            },
            "market_summary": {
                "direction": best_market.get("direction"),
                "change_pct": _safe_float(best_market.get("change_pct")),
                "summary": best_market.get("summary"),
            },
            "quick_actions": [
                {"label": "继续完善推荐", "href": "/recommend"},
                {"label": "打开 AI 助手问答", "href": "/assistant"},
                {"label": "查看农资建议", "href": "/store"},
            ],
            "env_summary": {
                "best_label": env.get("best_label"),
                "confidence": env.get("confidence"),
                "risk": env.get("risk"),
                "warnings": list(env.get("warnings") or []),
            },
        }

    def load_history(self) -> List[Dict[str, Any]]:
        raw = self._safe_read_json(self.history_path)
        if not isinstance(raw, list):
            return []
        rows = [row for row in raw if isinstance(row, dict)]
        rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return rows

    def append_history(self, payload: Dict[str, Any]) -> None:
        rows = [row for row in payload.get("results", []) if isinstance(row, dict)]
        best = rows[0] if rows else {}
        runtime = payload.get("runtime") or {}
        release = runtime.get("release") if isinstance(runtime, dict) else {}
        record = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "env": payload.get("env", {}),
            "release": {
                "run_id": (release.get("run_id") if isinstance(release, dict) else None),
                "status": (release.get("status") if isinstance(release, dict) else None),
            },
            "feedback_event_id": runtime.get("feedback_event_id") if isinstance(runtime, dict) else None,
            "best_crop": {
                "crop": best.get("crop"),
                "crop_label": _crop_label(str(best.get("crop") or "")) if best else None,
                "recommend_strength": _safe_float(best.get("recommend_strength")),
                "profit": _safe_float(best.get("profit")),
                "risk": _safe_float(best.get("risk")),
            },
            "top_candidates": [
                {
                    "crop": row.get("crop"),
                    "crop_label": _crop_label(str(row.get("crop") or "")),
                    "recommend_strength": _safe_float(row.get("recommend_strength")),
                    "profit": _safe_float(row.get("profit")),
                    "risk": _safe_float(row.get("risk")),
                    "style": ((row.get("decision_profile") or {}).get("style") if isinstance(row.get("decision_profile"), dict) else None),
                }
                for row in rows[:3]
            ],
        }
        history = self.load_history()
        history.insert(0, record)
        self._write_json(self.history_path, history[:30])

    def _ensure_payload(
        self,
        *,
        payload: Dict[str, Any] | None,
        config: Dict[str, Any] | None,
        recommend_with_source: Callable[..., Dict[str, Any]] | None,
        logger: Any | None,
    ) -> Dict[str, Any]:
        if payload and payload.get("results"):
            return self.enrich_payload(payload)

        existing = self._safe_read_json(self.output_dir / "推荐结果.json")
        if isinstance(existing, dict) and existing.get("results"):
            return self.enrich_payload(existing)

        env = self._safe_read_json(self.default_env_path)
        if isinstance(env, dict) and recommend_with_source and config is not None:
            try:
                preview = recommend_with_source(env, config=config, root=self.root, output_dir=self.output_dir, logger=logger)
                enriched = self.enrich_payload(preview)
                enriched["preview_mode"] = "default_env"
                return enriched
            except Exception:
                if logger is not None and hasattr(logger, "exception"):
                    logger.exception("failed building default preview payload")

        return {
            "env": {},
            "results": [],
            "decision_summary": self._build_decision_summary({"results": [], "env": {}}),
            "quick_questions": copy.deepcopy(QUESTION_PRESETS),
        }

    def build_home_summary(
        self,
        *,
        payload: Dict[str, Any] | None = None,
        config: Dict[str, Any] | None = None,
        recommend_with_source: Callable[..., Dict[str, Any]] | None = None,
        logger: Any | None = None,
    ) -> Dict[str, Any]:
        resolved = self._ensure_payload(
            payload=payload,
            config=config,
            recommend_with_source=recommend_with_source,
            logger=logger,
        )
        summary = resolved.get("decision_summary", {}) if isinstance(resolved.get("decision_summary"), dict) else {}
        history = self.load_history()
        top_cards = summary.get("top_cards", []) if isinstance(summary.get("top_cards"), list) else []
        market_watch = [
            {
                "crop": card.get("crop"),
                "crop_label": card.get("crop_label"),
                "style": card.get("style"),
                "summary": card.get("summary"),
                "risk_level": card.get("risk_level"),
                "fit_level": card.get("fit_level"),
            }
            for card in top_cards[:3]
        ]
        store_preview = self.build_store_summary(payload=resolved)["bundles"][:2]
        community_preview = self.build_community_summary(payload=resolved)["highlights"][:3]
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "headline": summary.get("headline"),
            "subheadline": summary.get("subheadline"),
            "best_crop": {
                "crop": ((summary.get("best_crop") or {}).get("crop") if isinstance(summary.get("best_crop"), dict) else None),
                "crop_label": ((summary.get("best_crop") or {}).get("crop_label") if isinstance(summary.get("best_crop"), dict) else None),
                "style": ((summary.get("best_crop") or {}).get("style") if isinstance(summary.get("best_crop"), dict) else None),
                "risk_level": ((summary.get("best_crop") or {}).get("risk_level") if isinstance(summary.get("best_crop"), dict) else None),
                "market_summary": _truncate_text(((summary.get("best_crop") or {}).get("market_summary") if isinstance(summary.get("best_crop"), dict) else "") or "", 90),
            },
            "top_cards": top_cards[:5],
            "risk_summary": summary.get("risk_summary", {}),
            "market_summary": summary.get("market_summary", {}),
            "market_watch": market_watch,
            "quick_questions": copy.deepcopy(QUESTION_PRESETS),
            "store_preview": store_preview,
            "community_preview": community_preview,
            "history_preview": history[:3],
            "actions": summary.get("quick_actions", []),
            "preview_mode": resolved.get("preview_mode"),
        }

    def build_store_summary(
        self,
        *,
        payload: Dict[str, Any] | None = None,
        config: Dict[str, Any] | None = None,
        recommend_with_source: Callable[..., Dict[str, Any]] | None = None,
        logger: Any | None = None,
    ) -> Dict[str, Any]:
        resolved = self._ensure_payload(
            payload=payload,
            config=config,
            recommend_with_source=recommend_with_source,
            logger=logger,
        )
        rows = [row for row in resolved.get("results", []) if isinstance(row, dict)]
        primary_crop = str(rows[0].get("crop") or "") if rows else ""
        primary_label = _crop_label(primary_crop) if primary_crop else ""
        bundles = [
            {
                "id": "rice-seed-premium",
                "crop": "rice",
                "crop_label": "水稻",
                "title": "优选水稻良种",
                "budget": "展示价",
                "price": "¥68",
                "unit_price": 68,
                "spec": "1kg / 袋",
                "unit_label": "袋",
                "badge": "适合试种",
                "shop_name": "田穗良种店",
                "shop_badge": "品牌种子",
                "rating": "4.8",
                "market_price": "¥79",
                "coupon_text": "首单减 ¥6",
                "reason": "适合做小面积试播和补播，先把发芽率和整齐度稳住。",
                "seller": "田穗良种店",
                "stock": 126,
                "sold": "近30天 52袋",
                "ship_in": "48小时内发货",
                "origin": "黑龙江建三江",
                "shipping_fee": 0,
                "items": ["净含量 1kg", "建议浸种催芽后播种", "页面展示价，不含运费"],
                "tags": ["种子", "主粮作物", "试种常备"],
                "gallery": [
                    {"image": "/assets/前端/资源/图片/商城/水稻种子.jpg", "alt": "水稻种子特写"},
                    {"image": "/assets/前端/资源/图片/交流/水稻田.jpg", "alt": "水稻田景象"},
                    {"image": "/assets/前端/资源/图片/交流/农田灌溉.jpg", "alt": "农田灌溉场景"},
                ],
                "variants": [
                    {"id": "trial", "label": "1kg 试种装", "price": 68, "market_price": 79, "note": "1 袋，小地块试播"},
                    {"id": "farmer", "label": "5kg 农户装", "price": 318, "market_price": 348, "note": "5 袋，适合连片试种"},
                    {"id": "coop", "label": "10kg 合作社装", "price": 598, "market_price": 658, "note": "10 袋，二次补单压力更小"},
                ],
                "delivery_lines": ["48 小时内发货", "快递免运费", "支持批次咨询和售后补寄"],
                "overview": [
                    "适合做直播与育秧前的小面积试种。",
                    "先观察出苗整齐度和秧苗长势，再决定是否扩大备货。"
                ],
                "description": [
                    "这款水稻良种更适合拿来做试播、补播和示范田，不追求一上来大量铺货，先把出苗率、整齐度和苗势看清楚。",
                    "如果你当前对水稻更感兴趣，可以先买 1 到 2 袋小规模验证，再结合地温、墒情和管理节奏决定后续补单。"
                ],
                "service": ["支持种植咨询", "破损包赔", "支持拍照确认批次信息"],
                "specs": [
                    {"label": "适合作物", "value": "水稻直播 / 育秧"},
                    {"label": "建议场景", "value": "试种、补播、小地块验证"},
                    {"label": "发货节奏", "value": "48 小时内"},
                    {"label": "售后说明", "value": "破损和漏发可补寄"}
                ],
                "image": "/assets/前端/资源/图片/商城/水稻种子.jpg",
                "image_alt": "水稻种子特写",
                "source_url": "https://commons.wikimedia.org/wiki/File:Rice_seed.JPG",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "license_label": "CC BY-SA 4.0",
                "credit": "Ranjithsiji / Wikimedia Commons",
            },
            {
                "id": "maize-seed-high-germ",
                "crop": "maize",
                "crop_label": "玉米",
                "title": "高发芽玉米种子",
                "budget": "展示价",
                "price": "¥88",
                "unit_price": 88,
                "spec": "1kg / 袋",
                "unit_label": "袋",
                "badge": "热销种子",
                "shop_name": "北垦农资直营",
                "shop_badge": "旗舰店",
                "rating": "4.7",
                "market_price": "¥96",
                "coupon_text": "店铺券减 ¥8",
                "reason": "适合做春播备种，先保证出苗整齐，再考虑追肥投入。",
                "seller": "北垦农资直营",
                "stock": 84,
                "sold": "近30天 41袋",
                "ship_in": "24小时内发货",
                "origin": "吉林松原",
                "shipping_fee": 0,
                "items": ["净含量 1kg", "适合小地块试种", "页面展示价，不含运费"],
                "tags": ["种子", "高发芽率", "玉米"],
                "gallery": [
                    {"image": "/assets/前端/资源/图片/商城/玉米种子.jpg", "alt": "红色玉米种子特写"},
                    {"image": "/assets/前端/资源/图片/交流/玉米地.jpg", "alt": "玉米地景象"},
                    {"image": "/assets/前端/资源/图片/交流/农田作业.jpg", "alt": "农田作业场景"},
                ],
                "variants": [
                    {"id": "trial", "label": "1kg 试播装", "price": 88, "market_price": 96, "note": "1 袋，适合春播验证"},
                    {"id": "farm", "label": "5kg 农户装", "price": 408, "market_price": 456, "note": "5 袋，适合家庭农场"},
                    {"id": "bulk", "label": "10kg 备货装", "price": 768, "market_price": 860, "note": "10 袋，适合大田集中播种"},
                ],
                "delivery_lines": ["24 小时内发货", "默认包邮", "支持到货后 24 小时内反馈"],
                "overview": [
                    "更适合在春播前先做小规模备种。",
                    "先稳出苗，再决定后续化肥和追肥投放节奏。"
                ],
                "description": [
                    "玉米种子最怕的是前期出苗不齐，一旦苗不齐，后面水肥管理会被动。这款更适合作为早期备种品项来展示。",
                    "如果你准备做春播，建议先在代表性地块试播，观察发芽、苗情和土壤保水情况，再补齐后续采购。"
                ],
                "service": ["支持批次咨询", "支持到货后 24 小时内反馈", "可按袋购买"],
                "specs": [
                    {"label": "适合作物", "value": "玉米春播"},
                    {"label": "建议场景", "value": "备种、试播、对比测试"},
                    {"label": "库存状态", "value": "现货"},
                    {"label": "售后说明", "value": "到货异常可反馈"}
                ],
                "image": "/assets/前端/资源/图片/商城/玉米种子.jpg",
                "image_alt": "红色玉米种子特写",
                "source_url": "https://commons.wikimedia.org/wiki/File:Red_Maize_Seed.jpg",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "license_label": "CC0",
                "credit": "Kkagwema / Wikimedia Commons",
            },
            {
                "id": "cotton-seed-coated",
                "crop": "cotton",
                "crop_label": "棉花",
                "title": "包衣棉花种子",
                "budget": "展示价",
                "price": "¥59",
                "unit_price": 59,
                "spec": "500g / 袋",
                "unit_label": "袋",
                "badge": "轻量备货",
                "shop_name": "棉丰植保馆",
                "shop_badge": "产地直供",
                "rating": "4.6",
                "market_price": "¥66",
                "coupon_text": "下单减 ¥5",
                "reason": "适合先做小批量备货，观察地温和墒情后再追加。",
                "seller": "棉丰植保馆",
                "stock": 63,
                "sold": "近30天 29袋",
                "ship_in": "48小时内发货",
                "origin": "新疆阿克苏",
                "shipping_fee": 8,
                "items": ["净含量 500g", "适合做早期试播", "页面展示价，不含运费"],
                "tags": ["种子", "包衣处理", "棉花"],
                "gallery": [
                    {"image": "/assets/前端/资源/图片/商城/棉花种子.jpg", "alt": "棉花种子特写"},
                    {"image": "/assets/前端/资源/图片/交流/农田作业.jpg", "alt": "农田作业场景"},
                    {"image": "/assets/前端/资源/图片/交流/农田灌溉.jpg", "alt": "农田灌溉场景"},
                ],
                "variants": [
                    {"id": "light", "label": "500g 轻量装", "price": 59, "market_price": 66, "note": "1 袋，适合试播"},
                    {"id": "farmer", "label": "2kg 农户装", "price": 218, "market_price": 248, "note": "4 袋，适合分批播种"},
                    {"id": "bulk", "label": "5kg 备货装", "price": 498, "market_price": 560, "note": "10 袋，适合集中备货"},
                ],
                "delivery_lines": ["48 小时内发货", "新疆产地发出", "支持批次追溯"],
                "overview": [
                    "更适合做小规模试播和补播。",
                    "先看地温、墒情和苗期管理，再决定是否加大采购。"
                ],
                "description": [
                    "棉花前期最怕播后条件跟不上，尤其是地温和土壤湿度不稳时，盲目大量备种容易压住周转。",
                    "先买小规格，观察 1 到 2 个地块的出苗和整齐度，更符合你现在这个项目的轻量采购逻辑。"
                ],
                "service": ["支持批次追溯", "支持售后咨询", "适合试播"],
                "specs": [
                    {"label": "适合作物", "value": "棉花"},
                    {"label": "建议场景", "value": "试播、轻量备货"},
                    {"label": "包装规格", "value": "500g / 袋"},
                    {"label": "发货时效", "value": "48 小时内"}
                ],
                "image": "/assets/前端/资源/图片/商城/棉花种子.jpg",
                "image_alt": "棉花种子特写",
                "source_url": "https://commons.wikimedia.org/wiki/File:Cotton_seeds_-_01.jpg",
                "license_url": "https://commons.wikimedia.org/wiki/File:Cotton_seeds_-_01.jpg",
                "license_label": "Public domain",
                "credit": "Keith Weller, USDA ARS / Wikimedia Commons",
            },
            {
                "id": "compound-fertilizer-balance",
                "crop": "general",
                "crop_label": "通用肥料",
                "title": "平衡型复合肥",
                "budget": "展示价",
                "price": "¥128",
                "unit_price": 128,
                "spec": "25kg / 袋",
                "unit_label": "袋",
                "badge": "通用底肥",
                "shop_name": "丰地农资仓",
                "shop_badge": "农资仓配",
                "rating": "4.9",
                "market_price": "¥138",
                "coupon_text": "满 2 袋减 ¥10",
                "reason": "更适合做播前和移栽前的基础底肥，不追求花哨，先保稳。",
                "seller": "丰地农资仓",
                "stock": 210,
                "sold": "近30天 73袋",
                "ship_in": "次日发货",
                "origin": "山东临沂",
                "shipping_fee": 12,
                "items": ["25kg 大包装", "适合播前或移栽前使用", "页面展示价，不含运费"],
                "tags": ["化肥", "底肥", "复合肥"],
                "gallery": [
                    {"image": "/assets/前端/资源/图片/商城/复合肥.jpg", "alt": "复合肥颗粒特写"},
                    {"image": "/assets/前端/资源/图片/商城/复合肥包装.jpg", "alt": "复合肥包装展示"},
                    {"image": "/assets/前端/资源/图片/交流/农田作业.jpg", "alt": "农田作业场景"},
                ],
                "variants": [
                    {"id": "single", "label": "25kg 单袋装", "price": 128, "market_price": 138, "note": "1 袋，常规底肥"},
                    {"id": "double", "label": "25kg x 2 袋", "price": 246, "market_price": 276, "note": "2 袋，更适合连片地块"},
                    {"id": "bulk", "label": "25kg x 5 袋", "price": 598, "market_price": 690, "note": "5 袋，仓配批量发货"},
                ],
                "delivery_lines": ["次日发货", "大件按袋发出", "支持运费估算和仓配咨询"],
                "overview": [
                    "适合作基础底肥，先把底盘打稳。",
                    "适合预算清晰、追求稳定管理的地块。"
                ],
                "description": [
                    "底肥不是越复杂越好，关键是搭配得稳。这款平衡型复合肥更适合放在播前或移栽前使用，做基础营养补充。",
                    "如果你不想把采购做得太花哨，这种通用型产品更适合做页面里的标准化上架商品。"
                ],
                "service": ["整袋发货", "支持运费估算", "适合常规底肥场景"],
                "specs": [
                    {"label": "适合作物", "value": "粮食作物 / 果蔬通用"},
                    {"label": "建议场景", "value": "底肥、播前整地"},
                    {"label": "净含量", "value": "25kg"},
                    {"label": "发货说明", "value": "整袋发出"}
                ],
                "image": "/assets/前端/资源/图片/商城/复合肥.jpg",
                "image_alt": "复合肥颗粒特写",
                "source_url": "https://commons.wikimedia.org/wiki/File:Compound_fertiliser.jpg",
                "license_url": "https://creativecommons.org/licenses/by-sa/2.0/",
                "license_label": "CC BY-SA 2.0",
                "credit": "Fir0002 / Wikimedia Commons",
            },
            {
                "id": "npk-water-soluble-191919",
                "crop": "general",
                "crop_label": "通用肥料",
                "title": "19-19-19 水溶肥",
                "budget": "展示价",
                "price": "¥46",
                "unit_price": 46,
                "spec": "1kg / 袋",
                "unit_label": "袋",
                "badge": "追肥快补",
                "shop_name": "叶丰水肥店",
                "shop_badge": "水肥专营",
                "rating": "4.8",
                "market_price": "¥52",
                "coupon_text": "满 3 袋减 ¥6",
                "reason": "适合在中前期做一次快补，方便观察作物长势反应。",
                "seller": "叶丰水肥店",
                "stock": 158,
                "sold": "近30天 96袋",
                "ship_in": "24小时内发货",
                "origin": "河北石家庄",
                "shipping_fee": 6,
                "items": ["1kg 小规格", "适合叶面或冲施场景", "页面展示价，不含运费"],
                "tags": ["化肥", "水溶肥", "NPK"],
                "gallery": [
                    {"image": "/assets/前端/资源/图片/商城/复合肥包装.jpg", "alt": "19-19-19 水溶肥包装"},
                    {"image": "/assets/前端/资源/图片/商城/复合肥.jpg", "alt": "复合肥颗粒特写"},
                    {"image": "/assets/前端/资源/图片/交流/农田灌溉.jpg", "alt": "农田灌溉场景"},
                ],
                "variants": [
                    {"id": "one", "label": "1kg 体验装", "price": 46, "market_price": 52, "note": "1 袋，适合快补测试"},
                    {"id": "five", "label": "1kg x 5 袋", "price": 218, "market_price": 246, "note": "5 袋，适合一季分批追肥"},
                    {"id": "ten", "label": "1kg x 10 袋", "price": 398, "market_price": 460, "note": "10 袋，适合长期备货"},
                ],
                "delivery_lines": ["24 小时内发货", "小规格包邮", "适合冲施和叶面补充"],
                "overview": [
                    "适合追肥快补和阶段性观察苗情反应。",
                    "小包装更适合做演示型商城的轻量上架商品。"
                ],
                "description": [
                    "水溶肥的好处是灵活，适合在中前期长势判断后做一次快补，不用像大包装底肥那样压库存。",
                    "这一类小规格商品更容易被普通用户理解，也更适合你当前的页面展示逻辑。"
                ],
                "service": ["小规格发货", "适合冲施 / 叶面场景", "支持咨询"],
                "specs": [
                    {"label": "适合作物", "value": "果蔬 / 经济作物通用"},
                    {"label": "建议场景", "value": "冲施、叶面补充"},
                    {"label": "净含量", "value": "1kg"},
                    {"label": "发货说明", "value": "24 小时内"}
                ],
                "image": "/assets/前端/资源/图片/商城/复合肥包装.jpg",
                "image_alt": "19-19-19 水溶肥包装",
                "source_url": "https://commons.wikimedia.org/wiki/File:NPK_19-19-19_Fertilizer_in_1_KG_Commercial_Packing_(1).jpg",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "license_label": "CC BY-SA 4.0",
                "credit": "Mahir Chowdhury / Wikimedia Commons",
            },
        ]
        if primary_crop:
            bundles.sort(key=lambda item: 0 if item.get("crop") == primary_crop else 1)
        return {
            "headline": f"{primary_label}优先采购清单" if primary_label else "农资小商城",
            "subheadline": "按当前作物整理了常用种子、肥料和配套农资。",
            "bundles": bundles,
            "preview_mode": resolved.get("preview_mode"),
        }

    def build_community_summary(
        self,
        *,
        payload: Dict[str, Any] | None = None,
        config: Dict[str, Any] | None = None,
        recommend_with_source: Callable[..., Dict[str, Any]] | None = None,
        logger: Any | None = None,
    ) -> Dict[str, Any]:
        resolved = self._ensure_payload(
            payload=payload,
            config=config,
            recommend_with_source=recommend_with_source,
            logger=logger,
        )
        highlights = [
            {
                "id": "seedling-check",
                "title": "老赵：返青期别急着催，先把苗情看明白",
                "tag": "热帖",
                "summary": "论坛里这两天讨论最多的还是苗情判断。大家更关心的是先看整齐度、根系和地表墒情，再决定追肥和补苗节奏。",
                "author": "老赵",
                "stats": "46 赞 · 17 条回复",
                "date": "2026-03-10",
                "location": "山东德州",
                "read_time": "5 分钟",
                "cover_image": "/assets/前端/资源/图片/交流/农田作业.jpg",
                "cover_alt": "农田作业场景",
                "cover_source_url": "https://commons.wikimedia.org/wiki/File:Farmers_field.jpg",
                "cover_license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "cover_license_label": "CC0",
                "cover_credit": "Knulclunk / Wikimedia Commons",
                "body": [
                    "老赵说现在最容易犯的错，就是一看气温上来了就着急催苗，结果根系还没站稳，后面不是徒长就是缺苗。",
                    "他这条帖子底下，不少人都在补充自己这两年的经验：先看整齐度、根系颜色、地表墒情，再决定要不要追肥、补苗或者加快管理动作。",
                    "论坛里比较一致的意见是，返青期宁可多观察两天，也别因为着急把后面的管理空间压没了。"
                ],
                "key_points": [
                    "先看苗情，再定追肥节奏。",
                    "弱苗先稳根，不要一把猛催。",
                    "同一块地里也要分区判断，不要按平均状态操作。"
                ],
                "comments": [
                    {"author": "王师傅", "time": "2小时前", "text": "前期太急最容易把后面节奏带乱，先把苗看稳最重要。"},
                    {"author": "小陈", "time": "今天 09:18", "text": "我们今年就是先巡田两天，再决定追不追，心里明显更稳。"},
                ],
            },
            {
                "id": "fertilizer-rhythm",
                "title": "老周：追肥别图快，先看叶色和墒情",
                "tag": "用户交流",
                "summary": "这周地里回温快，但大家普遍建议别一上来就猛追氮，先看叶色和土壤含水，再决定追肥量。",
                "author": "老周",
                "stats": "38 赞 · 12 条回复",
                "date": "2026-03-08",
                "location": "河南周口",
                "read_time": "5 分钟",
                "cover_image": "/assets/前端/资源/图片/交流/农田灌溉.jpg",
                "cover_alt": "农田灌溉场景",
                "cover_source_url": "https://commons.wikimedia.org/wiki/File:USDA_Irrigation_2011.jpg",
                "cover_license_url": "https://commons.wikimedia.org/wiki/File:USDA_Irrigation_2011.jpg",
                "cover_license_label": "Public domain",
                "cover_credit": "USDA / Wikimedia Commons",
                "body": [
                    "老周这条帖子说得很实在：很多人一看到天气转暖，就想赶紧追肥，但土壤含水和叶色没看清楚，追得越快越容易浪费。",
                    "交流区里不少人也认同这个判断，尤其是墒情偏差或者阴雨将来的时候，急追氮肥往往不如先等一天，或者先做小剂量试补。",
                    "他的建议不是不追肥，而是把追肥从“按日历操作”改成“看苗、看墒、看天气”三件事一起决定。"
                ],
                "key_points": [
                    "叶色偏浅但土壤偏干时，先补水再追肥更稳。",
                    "阴雨前不急着喷施和冲施，减少流失。",
                    "先做一块试补，观察 2 到 3 天的反应。"
                ],
                "comments": [
                    {"author": "阿涛", "time": "昨天 20:16", "text": "我们这边也是，先看墒情，不然追了也白追。"},
                    {"author": "静姐", "time": "昨天 18:44", "text": "叶色和天气一起看，确实比只看天数靠谱。"},
                ],
            },
            {
                "id": "test-small-plot",
                "title": "小李：种子先试播一小块，能少走很多弯路",
                "tag": "用户交流",
                "summary": "先用一小块田测出苗整齐度和病害反应，再决定是否整块铺开，是群里公认更稳的做法。",
                "author": "小李",
                "stats": "25 赞 · 8 条回复",
                "date": "2026-03-06",
                "location": "吉林松原",
                "read_time": "4 分钟",
                "cover_image": "/assets/前端/资源/图片/交流/玉米地.jpg",
                "cover_alt": "玉米地景象",
                "cover_source_url": "https://commons.wikimedia.org/wiki/File:Corn_field_in_Colorado.jpg",
                "cover_license_url": "https://commons.wikimedia.org/wiki/File:Corn_field_in_Colorado.jpg",
                "cover_license_label": "Public domain",
                "cover_credit": "USDA / Wikimedia Commons",
                "body": [
                    "小李提到的这个办法其实特别接地气：别一上来整块田都铺开，先拿一小块田测试种子表现，能省掉后面很多补救动作。",
                    "试播最有价值的地方，不只是看出苗率，还能看整齐度、苗势、苗期病害和田块对管理动作的反应。",
                    "如果前面这一小块田都不顺，后面再大面积推进，只会把问题放大。"
                ],
                "key_points": [
                    "先试播，再补单。",
                    "重点看出苗整齐度和苗期病害反应。",
                    "代表性地块比随机试播更有参考价值。"
                ],
                "comments": [
                    {"author": "阿峰", "time": "3天前", "text": "去年我没试播，后面补苗补得很累，今年肯定先试一块。"},
                    {"author": "春兰", "time": "3天前", "text": "这个方法省钱也省心，很适合新手。"},
                ],
            },
            {
                "id": "rain-spray-window",
                "title": "阿芳：雨前别急着喷，等叶面干爽再上药",
                "tag": "用户交流",
                "summary": "如果短时降雨概率高，大家更倾向先等 1 天，避免药液被冲掉导致补喷。",
                "author": "阿芳",
                "stats": "31 赞 · 10 条回复",
                "date": "2026-03-05",
                "location": "湖南岳阳",
                "read_time": "5 分钟",
                "cover_image": "/assets/前端/资源/图片/交流/水稻田.jpg",
                "cover_alt": "水稻田景象",
                "cover_source_url": "https://commons.wikimedia.org/wiki/File:Rice_Field_(22788739238).jpg",
                "cover_license_url": "https://creativecommons.org/publicdomain/mark/1.0/",
                "cover_license_label": "Public Domain Mark 1.0",
                "cover_credit": "RhemaPrabhata / Wikimedia Commons",
                "body": [
                    "阿芳这条帖子讨论的是喷施窗口。核心意思很简单：药不是越早喷越好，短时降雨概率高的时候，急着上药往往会变成补喷。",
                    "很多种植户也会忽视叶面状态，叶面过湿或气温不稳时，药效和附着都会受影响。",
                    "所以她给的建议是，先看 24 小时内天气，再看叶面状态，尽量把喷施动作放在更稳的窗口。"
                ],
                "key_points": [
                    "短时降雨前不急着喷。",
                    "优先选择早晨或傍晚的稳定时段。",
                    "喷后要复查，不是喷完就结束。"
                ],
                "comments": [
                    {"author": "海哥", "time": "4天前", "text": "喷完遇雨真的很伤，药费和人工都白搭。"},
                    {"author": "玉兰", "time": "4天前", "text": "我们现在都先看天气雷达，再决定当天喷不喷。"},
                ],
            },
        ]
        return {
            "headline": "经验交流",
            "subheadline": "看看农友的经验分享、过程复盘和评论交流。",
            "highlights": highlights[:4],
            "tips": [],
            "faq": [],
            "preview_mode": resolved.get("preview_mode"),
        }
    def build_profile_summary(
        self,
        *,
        payload: Dict[str, Any] | None = None,
        config: Dict[str, Any] | None = None,
        recommend_with_source: Callable[..., Dict[str, Any]] | None = None,
        logger: Any | None = None,
    ) -> Dict[str, Any]:
        resolved = self._ensure_payload(
            payload=payload,
            config=config,
            recommend_with_source=recommend_with_source,
            logger=logger,
        )
        history = self.load_history()
        counts: Dict[str, int] = {}
        for item in history:
            crop = str(((item.get("best_crop") or {}).get("crop")) or "")
            if not crop:
                continue
            counts[crop] = counts.get(crop, 0) + 1

        favorites = [
            {
                "crop": crop,
                "crop_label": _crop_label(crop),
                "count": count,
            }
            for crop, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
        ]

        latest_summary = resolved.get("decision_summary", {}) if isinstance(resolved.get("decision_summary"), dict) else {}
        return {
            "headline": "查看推荐记录和常用设置",
            "subheadline": "方便回看最近结果，也能快速找到常看作物。",
            "history": history[:8],
            "favorites": favorites,
            "settings": copy.deepcopy(DEFAULT_SETTINGS),
            "latest": latest_summary.get("best_crop", {}),
            "preview_mode": resolved.get("preview_mode"),
        }

    def _question_label(self, question_id: Optional[str], question_text: Optional[str]) -> str:
        if question_text and str(question_text).strip():
            return str(question_text).strip()
        return next((item["label"] for item in QUESTION_PRESETS if item["id"] == question_id), str(question_id or "自定义提问"))

    def _answer_by_llm(
        self,
        *,
        config: Dict[str, Any],
        question_text: str,
        logger: Any | None,
    ) -> Dict[str, Any]:
        if not llm_client_ready(config):
            raise RuntimeError("llm_unavailable")

        response = request_llm_chat(
            config=config,
            user_message=question_text,
            logger=logger,
        )
        answer = str(response.get("text") or "").strip()
        if not answer:
            raise RuntimeError("llm_empty_answer")

        return {
            "answer": answer,
            "bullets": [],
            "source": {
                "mode": "llm",
                "label": "AI 回答",
                "provider": response.get("provider"),
                "model": response.get("model"),
            },
        }

    def answer_question(
        self,
        *,
        question_id: Optional[str],
        crop: Optional[str] = None,
        question_text: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        recommend_with_source: Optional[Callable[..., Dict[str, Any]]] = None,
        logger: Any | None = None,
    ) -> Dict[str, Any]:
        display_question = self._question_label(question_id, question_text)

        if config is None:
            raise AssistantUnavailableError(
                code="llm_unavailable",
                detail="AI 服务暂未配置，请联系管理员。",
                status_code=503,
            )
        try:
            chosen = self._answer_by_llm(
                config=config,
                question_text=display_question,
                logger=logger,
            )
        except Exception as exc:
            if logger is not None and hasattr(logger, "warning"):
                logger.warning("assistant llm answer failed", exc_info=True)
            raise _assistant_error_from_llm(exc) from exc

        return {
            "question_id": question_id,
            "question": display_question,
            "question_scope": "general",
            "crop": None,
            "crop_label": None,
            "answer": chosen.get("answer"),
            "bullets": chosen.get("bullets", []),
            "related": [],
            "source": chosen.get("source", {}),
            "preview_mode": None,
        }



