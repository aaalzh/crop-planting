from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from joblib import load
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from 后端.兼容层 import apply_runtime_compat, tune_loaded_model
from 后端.数据加载 import load_config, read_env_json
from 后端.发布治理 import build_release_from_output
from 后端.反馈回流 import build_feedback_training_dataset
from 后端.模型产物 import (
    expected_cost_model_path,
    expected_price_model_path,
    expected_price_recursive_model_path,
    expected_yield_model_path,
)
from 后端.推荐器 import recommend

apply_runtime_compat()


def _abs(path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return ROOT / p


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _safe_rmtree(path: Path) -> None:
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception:
        pass


def _log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[retrain][{ts}] {message}", flush=True)


def _get_nested(config: dict, dotted: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _set_nested(config: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _cap_int(config: dict, dotted: str, upper: int, applied: Dict[str, Any]) -> None:
    raw = _get_nested(config, dotted, None)
    try:
        current = int(raw)
    except Exception:
        current = upper
    new_value = min(current, upper)
    _set_nested(config, dotted, new_value)
    applied[dotted] = new_value


def _apply_training_profile(config: dict, profile: str) -> tuple[dict, Dict[str, Any]]:
    tuned = copy.deepcopy(config)
    applied: Dict[str, Any] = {}
    cost_feature_set = str(_get_nested(tuned, "model.cost.feature_set", "legacy")).strip().lower()

    # Always enforce single-thread training in restricted Windows environments.
    for key in ("model.env.n_jobs", "model.price.n_jobs", "model.cost.n_jobs", "model.yield.n_jobs"):
        _set_nested(tuned, key, 1)
        applied[key] = 1

    if profile == "full":
        return tuned, applied

    if profile == "balanced":
        _cap_int(tuned, "model.env.n_estimators", 900, applied)

        _cap_int(tuned, "model.price.max_iter", 450, applied)
        _cap_int(tuned, "model.price.rf_n_estimators", 320, applied)
        _cap_int(tuned, "model.price.etr_n_estimators", 420, applied)
        if str(_get_nested(tuned, "model.price.regressor", "hgb")).lower() == "ensemble":
            members = ["hgb", "rf"]
            _set_nested(tuned, "model.price.ensemble_members", members)
            applied["model.price.ensemble_members"] = members

        _cap_int(tuned, "model.cost.rf_n_estimators", 260, applied)
        _cap_int(tuned, "model.cost.etr_n_estimators", 360, applied)
        _cap_int(tuned, "model.cost.hgb_max_iter", 360, applied)
        _cap_int(tuned, "model.cost.qgb_n_estimators", 220, applied)
        _cap_int(tuned, "model.cost.gbr_n_estimators", 280, applied)
        if str(_get_nested(tuned, "model.cost.regressor", "auto")).lower() == "ensemble":
            members = ["hgb", "etr", "rf"] if cost_feature_set == "panel_lite" else ["ridge", "huber", "hgb", "etr"]
            _set_nested(tuned, "model.cost.ensemble_members", members)
            applied["model.cost.ensemble_members"] = members

        _cap_int(tuned, "model.yield.max_iter", 700, applied)
        _cap_int(tuned, "model.yield.rf_n_estimators", 220, applied)
        _cap_int(tuned, "model.yield.etr_n_estimators", 380, applied)
        if str(_get_nested(tuned, "model.yield.regressor", "hgb")).lower() == "ensemble":
            members = ["hgb", "rf"]
            _set_nested(tuned, "model.yield.ensemble_members", members)
            applied["model.yield.ensemble_members"] = members

        if bool(_get_nested(tuned, "probability.walk_forward.enable", True)):
            step_size = int(_get_nested(tuned, "probability.walk_forward.step_size", 30))
            new_step = max(step_size, 60)
            _set_nested(tuned, "probability.walk_forward.step_size", new_step)
            applied["probability.walk_forward.step_size"] = new_step
        return tuned, applied

    if profile == "fast":
        _cap_int(tuned, "model.env.n_estimators", 480, applied)

        _cap_int(tuned, "model.price.max_iter", 280, applied)
        _cap_int(tuned, "model.price.rf_n_estimators", 180, applied)
        _cap_int(tuned, "model.price.etr_n_estimators", 240, applied)
        if str(_get_nested(tuned, "model.price.regressor", "hgb")).lower() == "ensemble":
            members = ["hgb"]
            _set_nested(tuned, "model.price.ensemble_members", members)
            applied["model.price.ensemble_members"] = members

        _cap_int(tuned, "model.cost.rf_n_estimators", 160, applied)
        _cap_int(tuned, "model.cost.etr_n_estimators", 220, applied)
        _cap_int(tuned, "model.cost.hgb_max_iter", 220, applied)
        _cap_int(tuned, "model.cost.qgb_n_estimators", 140, applied)
        _cap_int(tuned, "model.cost.gbr_n_estimators", 180, applied)
        if str(_get_nested(tuned, "model.cost.regressor", "auto")).lower() == "ensemble":
            members = ["hgb", "rf"] if cost_feature_set == "panel_lite" else ["ridge", "huber", "hgb"]
            _set_nested(tuned, "model.cost.ensemble_members", members)
            applied["model.cost.ensemble_members"] = members

        _cap_int(tuned, "model.yield.max_iter", 360, applied)
        _cap_int(tuned, "model.yield.rf_n_estimators", 140, applied)
        _cap_int(tuned, "model.yield.etr_n_estimators", 220, applied)
        if str(_get_nested(tuned, "model.yield.regressor", "hgb")).lower() == "ensemble":
            members = ["hgb"]
            _set_nested(tuned, "model.yield.ensemble_members", members)
            applied["model.yield.ensemble_members"] = members

        _set_nested(tuned, "probability.walk_forward.enable", False)
        applied["probability.walk_forward.enable"] = False
        return tuned, applied

    raise ValueError(f"unsupported profile: {profile}")


def _write_effective_config(config: dict, backup_root: Path, profile: str) -> Path:
    backup_root.mkdir(parents=True, exist_ok=True)
    out_path = backup_root / f"生效配置_{profile}.yaml"
    out_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return out_path


def _run_train_stage(config_path: str, train_calibrator: bool, timeout_minutes: int) -> None:
    cmd = [
        sys.executable,
        "-u",
        str(_abs("后端/训练/离线训练模型.py")),
        "--config",
        str(_abs(config_path)),
    ]
    if train_calibrator:
        cmd.append("--train-calibrator")
    timeout_seconds = None if int(timeout_minutes) <= 0 else int(timeout_minutes) * 60
    subprocess.run(
        cmd,
        cwd=str(ROOT),
        check=True,
        text=True,
        timeout=timeout_seconds,
    )


def _load_train_report(config: dict) -> dict:
    report_path = _abs(config["output"]["out_dir"]) / "模型训练报告.json"
    if not report_path.exists():
        raise FileNotFoundError(f"missing training report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _same_path(a: Path, b: Path) -> bool:
    try:
        return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))
    except Exception:
        return os.path.normcase(str(a)) == os.path.normcase(str(b))


def _meta_path_for_model(model_path: Path) -> Path:
    candidates = [
        model_path.with_name(model_path.stem + "_指标.json"),
        model_path.with_name(model_path.stem + "_metrics.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_model_checked(path: Path) -> object:
    return tune_loaded_model(load(path))


def _normalize_legacy_model_path(path: Path) -> Path:
    text = path.as_posix()
    replacements = [
        ("/output/", "/输出/"),
        ("_direct_trend_residual_v3_", "_直接趋势残差_v3_"),
        ("_hybrid_direct_v3_", "_混合直推_v3_"),
        ("_ensemble_hgb_rf_etr_", "_集成_hgb_rf_etr_"),
        ("_ensemble_panel_lite_", "_轻量面板集成_"),
        ("_ensemble_", "_集成_"),
        ("_step1.pkl", "_第1步.pkl"),
        ("_metrics.json", "_指标.json"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return Path(text)


def _resolve_report_model_path(key: str, row: dict, base_config: dict) -> Path | None:
    raw_text = str(row.get("model_path") or "").strip()
    if not raw_text:
        return None

    version = str(base_config.get("serving", {}).get("model_cache_version", "v2"))
    model_dir = _abs(base_config["output"]["out_dir"]) / "模型"
    crop = str(row.get("crop") or "").strip()
    candidates: List[Path] = []

    if key == "price" and crop:
        candidates.append(expected_price_model_path(model_dir, crop, base_config["model"]["price"], version)[0])
    elif key == "price_recursive" and crop:
        candidates.append(expected_price_recursive_model_path(model_dir, crop, base_config["model"]["price"], version)[0])
    elif key == "cost" and crop:
        candidates.append(expected_cost_model_path(model_dir, crop, base_config["model"]["cost"], version)[0])
    elif key == "yield":
        candidates.append(expected_yield_model_path(model_dir, base_config["model"]["yield"], version)[0])

    raw_path = _abs(raw_text)
    candidates.extend([raw_path, _normalize_legacy_model_path(raw_path)])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _model_paths_from_report(train_report: dict, base_config: dict) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {"price": [], "price_recursive": [], "cost": [], "yield": []}
    for key in ("price", "price_recursive", "cost", "yield"):
        rows = train_report.get(key, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not bool(row.get("ok", False)):
                continue
            model_path = _resolve_report_model_path(key, row, base_config)
            if model_path is None:
                continue
            out[key].append(model_path)
    return out


def _build_serving_alias_plan(base_config: dict, train_report: dict) -> List[Dict[str, Path]]:
    version = str(base_config.get("serving", {}).get("model_cache_version", "v2"))
    model_dir = _abs(base_config["output"]["out_dir"]) / "模型"
    plan: List[Dict[str, Path]] = []

    for row in train_report.get("price", []):
        if not isinstance(row, dict) or not bool(row.get("ok", False)):
            continue
        crop = str(row.get("crop", "")).strip()
        model_path = str(row.get("model_path", "")).strip()
        if not crop or not model_path:
            continue
        src_model = _resolve_report_model_path("price", row, base_config)
        if src_model is None:
            continue
        dst_model, dst_meta = expected_price_model_path(model_dir, crop, base_config["model"]["price"], version)
        src_meta = _meta_path_for_model(src_model)
        if _same_path(src_model, dst_model):
            continue
        plan.append(
            {
                "src_model": src_model,
                "src_meta": src_meta,
                "dst_model": dst_model,
                "dst_meta": dst_meta,
            }
        )

    for row in train_report.get("price_recursive", []):
        if not isinstance(row, dict) or not bool(row.get("ok", False)):
            continue
        crop = str(row.get("crop", "")).strip()
        model_path = str(row.get("model_path", "")).strip()
        if not crop or not model_path:
            continue
        src_model = _resolve_report_model_path("price_recursive", row, base_config)
        if src_model is None:
            continue
        dst_model, dst_meta = expected_price_recursive_model_path(model_dir, crop, base_config["model"]["price"], version)
        src_meta = _meta_path_for_model(src_model)
        if _same_path(src_model, dst_model):
            continue
        plan.append(
            {
                "src_model": src_model,
                "src_meta": src_meta,
                "dst_model": dst_model,
                "dst_meta": dst_meta,
            }
        )

    for row in train_report.get("cost", []):
        if not isinstance(row, dict) or not bool(row.get("ok", False)):
            continue
        crop = str(row.get("crop", "")).strip()
        model_path = str(row.get("model_path", "")).strip()
        if not crop or not model_path:
            continue
        src_model = _resolve_report_model_path("cost", row, base_config)
        if src_model is None:
            continue
        dst_model, dst_meta = expected_cost_model_path(model_dir, crop, base_config["model"]["cost"], version)
        src_meta = _meta_path_for_model(src_model)
        if _same_path(src_model, dst_model):
            continue
        plan.append(
            {
                "src_model": src_model,
                "src_meta": src_meta,
                "dst_model": dst_model,
                "dst_meta": dst_meta,
            }
        )

    for row in train_report.get("yield", []):
        if not isinstance(row, dict) or not bool(row.get("ok", False)):
            continue
        model_path = str(row.get("model_path", "")).strip()
        if not model_path:
            continue
        src_model = _resolve_report_model_path("yield", row, base_config)
        if src_model is None:
            continue
        dst_model, dst_meta = expected_yield_model_path(model_dir, base_config["model"]["yield"], version)
        src_meta = _meta_path_for_model(src_model)
        if _same_path(src_model, dst_model):
            continue
        plan.append(
            {
                "src_model": src_model,
                "src_meta": src_meta,
                "dst_model": dst_model,
                "dst_meta": dst_meta,
            }
        )
        break

    return plan


def _publish_serving_aliases(base_config: dict, train_report: dict) -> Dict[str, Any]:
    plan = _build_serving_alias_plan(base_config, train_report)
    applied = 0
    examples: List[Dict[str, str]] = []

    for item in plan:
        src_model = item["src_model"]
        src_meta = item["src_meta"]
        dst_model = item["dst_model"]
        dst_meta = item["dst_meta"]
        if not src_model.exists():
            raise FileNotFoundError(f"alias source model missing: {src_model}")

        dst_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_model, dst_model)
        applied += 1

        if src_meta.exists():
            dst_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_meta, dst_meta)

        if len(examples) < 10:
            examples.append({"src": src_model.as_posix(), "dst": dst_model.as_posix()})

    return {
        "alias_count": applied,
        "planned_alias_count": len(plan),
        "examples": examples,
    }


def _backup_paths(config: dict, backup_root: Path) -> List[Dict[str, Any]]:
    out_dir = _abs(config["output"]["out_dir"])
    targets = [
        {"path": _abs(config["paths"]["env_model_bundle"]), "kind": "file"},
        {"path": out_dir / "模型", "kind": "dir"},
        {"path": out_dir / "概率校准器.pkl", "kind": "file"},
        {"path": out_dir / "概率校准器指标.json", "kind": "file"},
        {"path": out_dir / "模型训练报告.json", "kind": "file"},
        {"path": out_dir / "环境回测.json", "kind": "file"},
        {"path": out_dir / "价格回测.csv", "kind": "file"},
        {"path": out_dir / "成本回测.csv", "kind": "file"},
        {"path": out_dir / "产量回测.json", "kind": "file"},
    ]

    manifest: List[Dict[str, Any]] = []
    tree_root = backup_root / "tree"
    tree_root.mkdir(parents=True, exist_ok=True)

    for item in targets:
        src: Path = item["path"]
        kind = str(item["kind"])
        rel = src.relative_to(ROOT)
        dst = tree_root / rel
        existed = src.exists()

        entry = {
            "src": str(src),
            "backup": str(dst),
            "kind": kind,
            "existed": existed,
        }
        manifest.append(entry)

        if not existed:
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        if kind == "dir":
            _safe_rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    (backup_root / "备份清单.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _restore_from_backup(manifest: List[Dict[str, Any]]) -> None:
    for entry in reversed(manifest):
        src = Path(entry["src"])
        backup = Path(entry["backup"])
        kind = str(entry["kind"])
        existed = bool(entry["existed"])

        if not existed:
            if kind == "dir":
                _safe_rmtree(src)
            else:
                _safe_unlink(src)
            continue

        if not backup.exists():
            continue

        src.parent.mkdir(parents=True, exist_ok=True)
        if kind == "dir":
            _safe_rmtree(src)
            shutil.copytree(backup, src)
        else:
            shutil.copy2(backup, src)


def _summarize_train_failures(report: dict, train_calibrator: bool) -> List[str]:
    failures: List[str] = []
    sections = ["env", "price", "cost", "yield"] + (["calibrator"] if train_calibrator else [])
    for section in sections:
        rows = report.get(section, [])
        if not isinstance(rows, list):
            failures.append(f"{section}: invalid report format")
            continue
        if len(rows) == 0:
            failures.append(f"{section}: no training rows")
            continue
        for row in rows:
            if bool(row.get("ok", False)):
                continue
            failures.append(f"{section}: {row.get('crop') or row.get('cost_name') or row.get('error') or 'unknown error'}")
    return failures


def _run_backtests(config_path: str, train_calibrator: bool) -> None:
    scripts = [
        "后端/诊断/回测环境.py",
        "后端/诊断/回测价格.py",
        "后端/诊断/回测成本.py",
        "后端/诊断/回测产量.py",
    ]
    if train_calibrator:
        scripts.append("后端/训练/训练概率校准器.py")

    for rel in scripts:
        script = _abs(rel)
        subprocess.run(
            [sys.executable, str(script), "--config", str(_abs(config_path))],
            cwd=str(ROOT),
            check=True,
            text=True,
        )


def _validate_artifacts(
    serve_config: dict,
    train_report: dict,
    config_path_for_smoke: str,
) -> Dict[str, Any]:
    errors: List[str] = []
    summary: Dict[str, Any] = {}

    env_bundle = _abs(serve_config["paths"]["env_model_bundle"])
    if not env_bundle.exists():
        errors.append(f"missing env bundle: {env_bundle}")
    else:
        try:
            _load_model_checked(env_bundle)
            summary["env_bundle"] = env_bundle.as_posix()
        except Exception as exc:
            errors.append(f"env bundle unreadable: {exc}")

    model_paths = _model_paths_from_report(train_report, serve_config)
    summary["price_models"] = len(model_paths["price"])
    summary["cost_models"] = len(model_paths["cost"])
    summary["yield_models"] = len(model_paths["yield"])

    for kind, paths in model_paths.items():
        if not paths:
            errors.append(f"training report has no successful {kind} models")
            continue
        for p in paths:
            if not p.exists():
                errors.append(f"missing trained {kind} model: {p}")
                continue
            try:
                _load_model_checked(p)
            except Exception as exc:
                errors.append(f"unreadable trained {kind} model {p.name}: {exc}")

    try:
        env_input = read_env_json(str(_abs("数据/样例/环境示例.json")))
        cfg = load_config(str(_abs(config_path_for_smoke)))
        payload = recommend(env_input, cfg)
        runtime = payload.get("runtime", {})
        missing_models = runtime.get("missing_models", [])
        strict_loading = bool(serve_config.get("serving", {}).get("strict_model_loading", False))
        if missing_models:
            summary["missing_models"] = missing_models
            if strict_loading:
                errors.append(f"recommendation reports missing models: {len(missing_models)}")
        results = payload.get("results", [])
        summary["recommend_candidates"] = len(results) if isinstance(results, list) else 0
    except Exception as exc:
        errors.append(f"recommend smoke test failed: {exc}")

    if errors:
        raise RuntimeError("; ".join(errors))

    return summary


def run_retrain_replace(
    config_path: str,
    *,
    profile: str,
    train_calibrator: bool,
    run_backtests: bool,
    keep_backup: bool,
    train_timeout_minutes: int,
) -> Dict[str, Any]:
    base_config = load_config(config_path)
    out_dir = _abs(base_config["output"]["out_dir"])
    backup_root = out_dir / "备份" / f"兼容重训练_{time.strftime('%Y%m%d_%H%M%S')}"
    tuned_config, applied_overrides = _apply_training_profile(base_config, profile)
    effective_config_path = _write_effective_config(tuned_config, backup_root, profile)

    t0 = time.perf_counter()
    feedback_dataset_summary: Dict[str, Any] = {}
    try:
        _log("[0/6] feedback dataset build start")
        feedback_dataset_summary = build_feedback_training_dataset(root=ROOT, config=base_config, save=True)
        _log(
            f"[0/6] feedback dataset ready "
            f"(labeled={feedback_dataset_summary.get('labeled_sample_count', 0)}, "
            f"outcome={feedback_dataset_summary.get('outcome_sample_count', 0)})"
        )
    except Exception as exc:
        feedback_dataset_summary = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _log(f"[0/6] feedback dataset build skipped: {feedback_dataset_summary['error']}")

    _log(f"[1/4] backup start -> {backup_root.as_posix()}")
    manifest = _backup_paths(base_config, backup_root)
    _log("[1/4] backup completed")
    report: Dict[str, Any] = {}
    publish_summary: Dict[str, Any] = {}
    validation_summary: Dict[str, Any] = {}
    release_summary: Dict[str, Any] = {}
    stage = "backup"

    try:
        stage = "train"
        _log(
            f"[2/4] training start (profile={profile}, calibrator={train_calibrator}, "
            f"timeout_minutes={train_timeout_minutes or 0})"
        )
        _run_train_stage(str(effective_config_path), train_calibrator, timeout_minutes=train_timeout_minutes)
        report = _load_train_report(tuned_config)
        failures = _summarize_train_failures(report, train_calibrator=train_calibrator)
        if failures:
            raise RuntimeError("training contains failures: " + " | ".join(failures[:10]))
        _log("[2/4] training completed")

        if run_backtests:
            stage = "backtest"
            _log("[3/4] backtests start")
            _run_backtests(config_path=str(effective_config_path), train_calibrator=train_calibrator)
            _log("[3/4] backtests completed")
        else:
            _log("[3/4] backtests skipped")

        stage = "publish"
        _log("[4/5] publish serving aliases start")
        publish_summary = _publish_serving_aliases(base_config, report)
        _log(f"[4/5] publish serving aliases completed (aliases={publish_summary.get('alias_count', 0)})")

        stage = "validate"
        _log("[5/5] validation start")
        validation_summary = _validate_artifacts(
            serve_config=base_config,
            train_report=report,
            config_path_for_smoke=str(_abs(config_path)),
        )
        _log("[5/5] validation completed")

        stage = "release"
        _log("[6/6] closed-loop release build start")
        release_summary = build_release_from_output(
            root=ROOT,
            config=base_config,
            output_dir=out_dir,
            source="compatible_retrain",
        )
        _log(
            f"[6/6] closed-loop release build completed "
            f"(run_id={release_summary.get('run_id')}, status={release_summary.get('status')})"
        )

    except Exception as exc:
        _log(f"failure at stage={stage}, starting rollback")
        _restore_from_backup(manifest)
        raise RuntimeError(f"retrain failed at stage '{stage}', rollback completed: {exc}") from exc

    elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
    result = {
        "ok": True,
        "elapsed_ms": elapsed,
        "base_config": str(_abs(config_path)),
        "effective_config": effective_config_path.as_posix(),
        "profile": profile,
        "applied_overrides": applied_overrides,
        "backup_dir": backup_root.as_posix(),
        "train_calibrator": train_calibrator,
        "run_backtests": run_backtests,
        "feedback_dataset": feedback_dataset_summary,
        "publish": publish_summary,
        "validation": validation_summary,
        "release": release_summary,
    }

    out_report = out_dir / "兼容重训练报告.json"
    out_report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if not keep_backup:
        _safe_rmtree(backup_root)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click compatible retrain + artifact replacement with rollback")
    parser.add_argument("--config", default="后端/配置.yaml")
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=["full", "balanced", "fast"],
        help="training profile: full(quality-first), balanced(default), fast(runtime-first)",
    )
    parser.add_argument("--no-calibrator", action="store_true", help="skip probability calibrator retraining")
    parser.add_argument("--with-backtests", action="store_true", help="run backtest scripts after training")
    parser.add_argument("--no-backtests", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--train-timeout-minutes",
        type=int,
        default=0,
        help="stop training stage after N minutes; 0 means no timeout",
    )
    parser.add_argument("--delete-backup", action="store_true", help="delete backup after successful replacement")
    args = parser.parse_args()

    result = run_retrain_replace(
        config_path=str(_abs(args.config)),
        profile=str(args.profile),
        train_calibrator=not args.no_calibrator,
        run_backtests=bool(args.with_backtests and not args.no_backtests),
        keep_backup=not args.delete_backup,
        train_timeout_minutes=int(args.train_timeout_minutes),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

