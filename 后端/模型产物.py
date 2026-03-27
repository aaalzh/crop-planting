from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

PRICE_DIRECT_ARTIFACT_TAG = "直接趋势残差_v3"
PRICE_DIRECT_LEGACY_ARTIFACT_TAGS = ["混合直推_v3", "direct_trend_residual_v3", "hybrid_direct_v3"]

作物中文名 = {
    "apple": "苹果",
    "banana": "香蕉",
    "blackgram": "黑豆",
    "chickpea": "鹰嘴豆",
    "cocount": "椰子",
    "coconut": "椰子",
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
    "water melon": "西瓜",
    "watermelon": "西瓜",
}


def _price_step_reg_tag(cfg: dict) -> str:
    reg = str(cfg.get("regressor", "hgb"))
    if reg == "ensemble":
        members = "_".join(cfg.get("ensemble_members", []))
        return f"集成_{members}" if members else "集成"
    return reg


def _去重标签(tags: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for tag in tags:
        text = str(tag or "").strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def price_direct_artifact_tag(cfg: dict) -> str:
    return PRICE_DIRECT_ARTIFACT_TAG


def price_direct_legacy_artifact_tags(cfg: dict) -> List[str]:
    return list(PRICE_DIRECT_LEGACY_ARTIFACT_TAGS)


def price_recursive_artifact_tag(cfg: dict) -> str:
    return _price_step_reg_tag(cfg)


def price_recursive_legacy_artifact_tags(cfg: dict) -> List[str]:
    reg = str(cfg.get("regressor", "hgb"))
    if reg == "ensemble":
        members = "_".join(cfg.get("ensemble_members", []))
        return _去重标签([f"ensemble_{members}" if members else "ensemble"])
    return []


def _cost_reg_tag(cfg: dict) -> str:
    reg = str(cfg.get("regressor", "ridge"))
    feature_set = str(cfg.get("feature_set", "legacy")).strip().lower()
    if feature_set == "panel_lite":
        return "轻量面板集成" if reg == "ensemble" else f"{reg}_panel_lite"
    return reg


def _cost_legacy_reg_tags(cfg: dict) -> List[str]:
    reg = str(cfg.get("regressor", "ridge"))
    feature_set = str(cfg.get("feature_set", "legacy")).strip().lower()
    if feature_set == "panel_lite":
        return _去重标签(["ensemble_panel_lite" if reg == "ensemble" else f"{reg}_panel_lite", reg, "集成" if reg == "ensemble" else ""])
    if reg == "ensemble":
        return ["集成", "ensemble"]
    return []


def _yield_reg_tag(cfg: dict) -> str:
    reg = str(cfg.get("regressor", "hgb"))
    if reg == "ensemble":
        members = "_".join(cfg.get("ensemble_members", []))
        return f"集成_{members}" if members else "集成"
    return reg


def _yield_legacy_reg_tags(cfg: dict) -> List[str]:
    reg = str(cfg.get("regressor", "hgb"))
    if reg == "ensemble":
        members = "_".join(cfg.get("ensemble_members", []))
        return _去重标签([f"ensemble_{members}" if members else "ensemble"])
    return []


def _作物名(crop: str) -> str:
    text = str(crop or "").strip()
    if not text:
        return text
    return 作物中文名.get(text.lower(), text)


def _中文指标路径(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + "_指标.json")


def price_model_candidates(model_dir: Path, crop: str, cfg: dict, version: str) -> List[Tuple[Path, Path]]:
    canonical_tag = price_direct_artifact_tag(cfg)
    crop_cn = _作物名(crop)
    tags = _去重标签([canonical_tag] + price_direct_legacy_artifact_tags(cfg))
    out: List[Tuple[Path, Path]] = []
    for tag in tags:
        p = model_dir / f"价格_{crop_cn}_{tag}_{version}.pkl"
        out.append((p, _中文指标路径(p)))
    return out


def price_recursive_model_candidates(model_dir: Path, crop: str, cfg: dict, version: str) -> List[Tuple[Path, Path]]:
    crop_cn = _作物名(crop)
    out: List[Tuple[Path, Path]] = []
    for tag in _去重标签([price_recursive_artifact_tag(cfg)] + price_recursive_legacy_artifact_tags(cfg)):
        for step_suffix in ("_第1步", "_step1"):
            p = model_dir / f"价格_{crop_cn}_{tag}_{version}{step_suffix}.pkl"
            out.append((p, _中文指标路径(p)))
    return out


def cost_model_candidates(model_dir: Path, crop: str, cfg: dict, version: str) -> List[Tuple[Path, Path]]:
    crop_cn = _作物名(crop)
    tags = _去重标签([_cost_reg_tag(cfg)] + _cost_legacy_reg_tags(cfg))
    out: List[Tuple[Path, Path]] = []
    for tag in tags:
        p = model_dir / f"成本_{crop_cn}_{tag}_{version}.pkl"
        out.append((p, _中文指标路径(p)))
    return out


def yield_model_candidates(model_dir: Path, cfg: dict, version: str) -> List[Tuple[Path, Path]]:
    out: List[Tuple[Path, Path]] = []
    for tag in _去重标签([_yield_reg_tag(cfg)] + _yield_legacy_reg_tags(cfg)):
        p = model_dir / f"产量_全局_{tag}_{version}.pkl"
        out.append((p, _中文指标路径(p)))
    return out


def expected_price_model_path(model_dir: Path, crop: str, cfg: dict, version: str) -> Tuple[Path, Path]:
    p = model_dir / f"价格_{_作物名(crop)}_{price_direct_artifact_tag(cfg)}_{version}.pkl"
    return p, _中文指标路径(p)


def expected_price_recursive_model_path(model_dir: Path, crop: str, cfg: dict, version: str) -> Tuple[Path, Path]:
    p = model_dir / f"价格_{_作物名(crop)}_{price_recursive_artifact_tag(cfg)}_{version}_第1步.pkl"
    return p, _中文指标路径(p)


def expected_cost_model_path(model_dir: Path, crop: str, cfg: dict, version: str) -> Tuple[Path, Path]:
    p = model_dir / f"成本_{_作物名(crop)}_{_cost_reg_tag(cfg)}_{version}.pkl"
    return p, _中文指标路径(p)


def expected_yield_model_path(model_dir: Path, cfg: dict, version: str) -> Tuple[Path, Path]:
    p = model_dir / f"产量_全局_{_yield_reg_tag(cfg)}_{version}.pkl"
    return p, _中文指标路径(p)
