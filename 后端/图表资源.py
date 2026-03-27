from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote


def list_chart_images(chart_dir: Path) -> List[Dict[str, str]]:
    if not chart_dir.exists():
        return []
    files = sorted(
        [
            f
            for f in chart_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ],
        key=lambda p: p.name,
    )
    return [{"name": f.name, "url": f"/assets/charts/{quote(f.name)}"} for f in files]


def group_chart_images(images: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, str]]] = OrderedDict()
    for img in images:
        name = img.get("name", "")
        key = name.split("_", 1)[0] if "_" in name else "其他"
        groups.setdefault(key, []).append(img)
    return [{"group": k, "count": len(v), "images": v} for k, v in groups.items()]


def resolve_chart_dir(primary: Path, secondary: Path) -> Path:
    image_ext = {".png", ".jpg", ".jpeg", ".webp"}

    def _has_images(p: Path) -> bool:
        if not p.exists():
            return False
        return any(f.is_file() and f.suffix.lower() in image_ext for f in p.iterdir())

    if _has_images(primary):
        return primary
    if _has_images(secondary):
        return secondary

    primary.mkdir(parents=True, exist_ok=True)
    return primary

