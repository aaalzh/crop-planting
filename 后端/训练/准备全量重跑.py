from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_abs(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return payload if isinstance(payload, dict) else {}


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def _artifact_summary(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {"path": None, "exists": False}
    return {"path": path.resolve().as_posix(), "exists": True}


def _build_release_report(manifest: dict) -> dict:
    gating = manifest.get("gating") or {}
    metrics = manifest.get("metrics") or {}
    score_fusion = manifest.get("score_fusion") or {}
    return {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "created_at": manifest.get("created_at"),
        "updated_at": manifest.get("updated_at"),
        "source": manifest.get("source"),
        "release_dir": manifest.get("release_dir"),
        "manifest_path": manifest.get("manifest_path"),
        "baseline": gating.get("baseline"),
        "allowed": gating.get("allowed"),
        "summary": gating.get("summary"),
        "smoke": manifest.get("smoke") or {},
        "shadow": manifest.get("shadow") or {},
        "gate": {
            "smoke_checks": gating.get("smoke_checks") or [],
            "shadow_checks": gating.get("shadow_checks") or [],
            "module_checks": gating.get("module_checks") or [],
            "business_checks": gating.get("business_checks") or [],
            "health_checks": gating.get("health_checks") or [],
            "provenance_checks": gating.get("provenance_checks") or [],
            "failed_checks": gating.get("failed_checks") or [],
        },
        "env_probability": metrics.get("env_probability") or {},
        "closed_loop": metrics.get("closed_loop") or manifest.get("closed_loop") or {},
        "score_fusion": {
            "weights": score_fusion.get("weights") or {},
            "validation_objective": score_fusion.get("validation_objective") or {},
        },
        "artifacts": manifest.get("artifacts") or {},
    }


def repair_release_metadata(backend_config_path: str) -> dict:
    backend_cfg = _read_yaml(_to_abs(backend_config_path))
    release_root = _to_abs(str((backend_cfg.get("release") or {}).get("root_dir", "输出/发布")))
    registry_path = release_root / "发布索引.json"
    registry = _read_json(registry_path)

    manifests: List[tuple[Path, Path, dict, str]] = []
    alias_map: Dict[str, str] = {}
    for path in sorted(release_root.iterdir()) if release_root.exists() else []:
        if not path.is_dir():
            continue
        manifest_path = path / "发布清单.json"
        if not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        old_run_id = str(manifest.get("run_id") or path.name).strip() or path.name
        alias_map[old_run_id] = path.name
        alias_map[path.name] = path.name
        manifests.append((path, manifest_path, manifest, old_run_id))

    def translate_id(value: Any) -> Optional[str]:
        text = str(value or "").strip()
        if not text:
            return None
        return alias_map.get(text, text)

    champion_run_id = translate_id(registry.get("champion_run_id"))
    challenger_run_id = translate_id(registry.get("challenger_run_id"))
    archived_run_ids = _unique_keep_order(translate_id(item) or "" for item in registry.get("archived_run_ids", []))

    registry["champion_run_id"] = champion_run_id
    registry["challenger_run_id"] = challenger_run_id
    registry["archived_run_ids"] = archived_run_ids
    history_rows = registry.get("history", [])
    if isinstance(history_rows, list):
        for row in history_rows:
            if not isinstance(row, dict):
                continue
            row_run_id = translate_id(row.get("run_id"))
            if row_run_id:
                row["run_id"] = row_run_id
    registry["history"] = history_rows if isinstance(history_rows, list) else []
    registry["updated_at"] = _now_iso()
    _write_json(registry_path, registry)

    repaired = []
    for release_dir, manifest_path, manifest, old_run_id in manifests:
        release_report_path = release_dir / "发布报告.json"
        updated_at = _now_iso()

        manifest["run_id"] = release_dir.name
        manifest["release_dir"] = release_dir.resolve().as_posix()
        manifest["manifest_path"] = manifest_path.resolve().as_posix()
        manifest["updated_at"] = updated_at

        if champion_run_id and release_dir.name == champion_run_id:
            manifest["status"] = "champion"
        elif challenger_run_id and release_dir.name == challenger_run_id:
            manifest["status"] = "challenger"
        else:
            manifest["status"] = "archived"

        serving = manifest.setdefault("serving", {})
        serving["output_dir"] = release_dir.resolve().as_posix()
        env_bundle_path = release_dir / "环境模型包.pkl"
        rec_csv_path = release_dir / "推荐结果.csv"
        if rec_csv_path.exists():
            serving["precomputed_recommendation_file"] = rec_csv_path.resolve().as_posix()
        if env_bundle_path.exists():
            serving["env_model_bundle"] = env_bundle_path.resolve().as_posix()

        artifacts = manifest.setdefault("artifacts", {})
        artifacts["model_dir"] = _artifact_summary(release_dir / "模型")
        artifacts["env_bundle"] = _artifact_summary(env_bundle_path)
        artifacts["recommendation_csv"] = _artifact_summary(release_dir / "推荐结果.csv")
        artifacts["recommendation_json"] = _artifact_summary(release_dir / "推荐结果.json")
        artifacts["training_report"] = _artifact_summary(release_dir / "模型训练报告.json")
        artifacts["backtest_report"] = _artifact_summary(release_dir / "回测报告.json")
        artifacts["markdown_report"] = _artifact_summary(release_dir / "高精度训练报告.md")
        artifacts["lifecycle_report"] = _artifact_summary(release_dir / "输出生命周期报告.json")
        artifacts["calibrator_model"] = _artifact_summary(release_dir / "概率校准器.pkl")
        artifacts["calibrator_meta"] = _artifact_summary(release_dir / "概率校准器指标.json")
        artifacts["score_weights"] = _artifact_summary(release_dir / "评分权重.json")
        artifacts["manifest"] = _artifact_summary(manifest_path)
        artifacts["release_report"] = _artifact_summary(release_report_path)
        manifest["artifacts"] = artifacts

        release_report = manifest.get("release_report")
        if not isinstance(release_report, dict):
            release_report = {}
        release_report["path"] = release_report_path.resolve().as_posix()
        release_report["summary"] = ((manifest.get("gating") or {}).get("summary"))
        release_report["allowed"] = ((manifest.get("gating") or {}).get("allowed"))
        manifest["release_report"] = release_report

        _write_json(manifest_path, manifest)
        if release_report_path.exists():
            _write_json(release_report_path, _build_release_report(manifest))

        repaired.append(
            {
                "dir": release_dir.name,
                "old_run_id": old_run_id,
                "new_run_id": release_dir.name,
                "status": manifest.get("status"),
            }
        )

    return {
        "release_root": release_root.resolve().as_posix(),
        "registry_path": registry_path.resolve().as_posix(),
        "champion_run_id": champion_run_id,
        "challenger_run_id": challenger_run_id,
        "archived_run_ids": archived_run_ids,
        "repaired_count": len(repaired),
        "repaired": repaired,
    }


def write_stage_configs(
    *,
    run_id: str,
    backend_config_path: str,
    train_config_path: str,
    stage_root: str,
) -> dict:
    stage_dir = Path(stage_root) / run_id
    stage_dir_text = stage_dir.as_posix()
    env_bundle_text = (stage_dir / "环境推荐" / "作物推荐模型管道.pkl").as_posix()
    backend_out_path = ROOT / "后端" / f"配置.{run_id}.yaml"
    train_out_path = ROOT / "训练流水线" / "配置" / f"高精度.{run_id}.yaml"

    backend_cfg = _read_yaml(_to_abs(backend_config_path))
    backend_cfg.setdefault("output", {})["out_dir"] = stage_dir_text
    backend_cfg.setdefault("paths", {})["env_model_bundle"] = env_bundle_text
    serving_cfg = backend_cfg.setdefault("serving", {})
    serving_cfg["precomputed_recommendation_file"] = (stage_dir / "推荐结果.csv").as_posix()
    serving_cfg["log_file"] = (stage_dir / "服务日志.log").as_posix()
    _write_yaml(backend_out_path, backend_cfg)

    train_cfg = _read_yaml(_to_abs(train_config_path))
    train_cfg.setdefault("paths", {})["output_dir"] = stage_dir_text
    lifecycle_cfg = train_cfg.setdefault("outputs_lifecycle", {})
    lifecycle_cfg["archive_dir"] = (stage_dir / "归档运行").as_posix()
    lifecycle_cfg["backup_dir"] = (stage_dir / "备份").as_posix()
    closed_loop_cfg = train_cfg.setdefault("closed_loop", {})
    closed_loop_cfg["backend_config"] = Path("后端", backend_out_path.name).as_posix()
    _write_yaml(train_out_path, train_cfg)

    return {
        "run_id": run_id,
        "stage_dir": (ROOT / stage_dir).resolve().as_posix(),
        "backend_config": backend_out_path.resolve().as_posix(),
        "train_config": train_out_path.resolve().as_posix(),
        "env_model_bundle": (ROOT / env_bundle_text).resolve().as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair release metadata and prepare stage configs for a full rerun")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backend-config", default="后端/配置.yaml")
    parser.add_argument("--train-config", default="训练流水线/配置/高精度.yaml")
    parser.add_argument("--stage-root", default="输出/stage")
    parser.add_argument("--skip-repair-release", action="store_true")
    parser.add_argument("--skip-write-configs", action="store_true")
    args = parser.parse_args()

    payload: Dict[str, Any] = {
        "ok": True,
        "run_id": str(args.run_id).strip(),
    }

    if not bool(args.skip_repair_release):
        payload["release_repair"] = repair_release_metadata(str(args.backend_config))
    if not bool(args.skip_write_configs):
        payload["stage_configs"] = write_stage_configs(
            run_id=str(args.run_id).strip(),
            backend_config_path=str(args.backend_config),
            train_config_path=str(args.train_config),
            stage_root=str(args.stage_root),
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
