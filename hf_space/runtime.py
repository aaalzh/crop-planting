from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from huggingface_hub import snapshot_download

from hf_space.hf_auth import resolve_hf_token


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_ROOT = ROOT / "hf_runtime"


def _bool_env(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _nonempty_env(name: str) -> Optional[str]:
    value = str(os.environ.get(name, "")).strip()
    return value or None


def _apply_llm_env_overrides(llm_cfg: Dict[str, Any]) -> None:
    enabled_env = _nonempty_env("CROP_LLM_ENABLED")
    if enabled_env is not None:
        llm_cfg["enabled"] = enabled_env.lower() in {"1", "true", "yes", "on"}

    for cfg_key, env_name in [
        ("provider", "CROP_LLM_PROVIDER"),
        ("model", "CROP_LLM_MODEL"),
        ("endpoint", "CROP_LLM_ENDPOINT"),
        ("api_key_env", "CROP_LLM_API_KEY_ENV"),
    ]:
        value = _nonempty_env(env_name)
        if value is not None:
            llm_cfg[cfg_key] = value

    for cfg_key, env_name in [
        ("timeout_seconds", "CROP_LLM_TIMEOUT_SECONDS"),
        ("max_tokens", "CROP_LLM_MAX_TOKENS"),
        ("temperature", "CROP_LLM_TEMPERATURE"),
    ]:
        value = _nonempty_env(env_name)
        if value is not None:
            llm_cfg[cfg_key] = value

    if _nonempty_env("CROP_LLM_API_KEY") is not None:
        llm_cfg["api_key"] = ""
        llm_cfg["api_key_env"] = "CROP_LLM_API_KEY"

    api_key_env = str(llm_cfg.get("api_key_env", "")).strip() or "DEEPSEEK_API_KEY"
    has_direct_key = bool(str(llm_cfg.get("api_key", "")).strip())
    has_env_key = bool(_nonempty_env(api_key_env))
    llm_cfg["enabled"] = bool(llm_cfg.get("enabled", False)) and (has_direct_key or has_env_key)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _sync_from_local(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"local artifacts dir not found: {source_dir.as_posix()}")
    _copy_tree(source_dir, target_dir)


def _sync_from_hub(repo_id: str, repo_type: str, revision: Optional[str], target_dir: Path) -> None:
    token = resolve_hf_token()
    if not token:
        raise RuntimeError("missing Hugging Face token; set HF_TOKEN or HUGGINGFACE_HUB_TOKEN")

    target_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "local_dir": str(target_dir),
        "token": token,
    }
    try:
        snapshot_download(local_dir_use_symlinks=False, **kwargs)
    except TypeError:
        snapshot_download(**kwargs)


def _replace_path_prefix(text: str, replacements: list[tuple[str, str]]) -> str:
    normalized = text.replace("\\", "/")
    for old_prefix, new_prefix in replacements:
        old_norm = old_prefix.replace("\\", "/").rstrip("/")
        new_norm = new_prefix.replace("\\", "/").rstrip("/")
        if not old_norm:
            continue
        if normalized == old_norm:
            return new_norm
        if normalized.startswith(old_norm + "/"):
            suffix = normalized[len(old_norm) + 1 :]
            return f"{new_norm}/{suffix}" if new_norm else suffix
    return text


def _rewrite_paths(payload: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(payload, str):
        return _replace_path_prefix(payload, replacements)
    if isinstance(payload, list):
        return [_rewrite_paths(item, replacements) for item in payload]
    if isinstance(payload, dict):
        return {key: _rewrite_paths(value, replacements) for key, value in payload.items()}
    return payload


def _set_artifact_path(artifacts: dict[str, Any], key: str, path: Path) -> None:
    if key not in artifacts:
        return
    item = artifacts.get(key)
    if isinstance(item, dict):
        item["path"] = path.as_posix()
        item["exists"] = path.exists()


def _repair_release_manifests(release_root: Path, output_root: Path) -> None:
    if not release_root.exists():
        return

    for run_dir in sorted(path for path in release_root.iterdir() if path.is_dir()):
        manifest_path = run_dir / "发布清单.json"
        if not manifest_path.exists():
            continue

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_release_dir_text = str(payload.get("release_dir", "")).strip()
        replacements: list[tuple[str, str]] = []
        if old_release_dir_text:
            old_release_dir = Path(old_release_dir_text)
            replacements.append((old_release_dir.as_posix(), run_dir.as_posix()))

            old_release_root = old_release_dir.parent
            replacements.append((old_release_root.as_posix(), release_root.as_posix()))

            if old_release_root.name == "发布":
                old_output_root = old_release_root.parent
                replacements.append((old_output_root.as_posix(), output_root.as_posix()))
                old_repo_root = old_output_root.parent
                replacements.append((old_repo_root.as_posix(), ROOT.as_posix()))

        replacements = sorted(replacements, key=lambda item: len(item[0]), reverse=True)
        repaired = _rewrite_paths(payload, replacements)

        repaired["release_dir"] = run_dir.as_posix()
        repaired["manifest_path"] = manifest_path.as_posix()
        if "source_output_dir" in repaired:
            repaired["source_output_dir"] = run_dir.as_posix()

        serving = repaired.setdefault("serving", {})
        if isinstance(serving, dict):
            serving["output_dir"] = run_dir.as_posix()
            env_bundle = run_dir / "环境模型包.pkl"
            if env_bundle.exists():
                serving["env_model_bundle"] = env_bundle.as_posix()
            precomputed_file = run_dir / "推荐结果.csv"
            if precomputed_file.exists():
                serving["precomputed_recommendation_file"] = precomputed_file.as_posix()

        artifacts = repaired.get("artifacts")
        if isinstance(artifacts, dict):
            _set_artifact_path(artifacts, "model_dir", run_dir / "模型")
            _set_artifact_path(artifacts, "env_model_bundle", run_dir / "环境模型包.pkl")
            _set_artifact_path(artifacts, "recommendation_csv", run_dir / "推荐结果.csv")
            _set_artifact_path(artifacts, "recommendation_json", run_dir / "推荐结果.json")
            _set_artifact_path(artifacts, "training_report", run_dir / "模型训练报告.json")
            _set_artifact_path(artifacts, "backtest_report", run_dir / "回测报告.json")
            _set_artifact_path(artifacts, "markdown_report", run_dir / "高精度训练报告.md")
            _set_artifact_path(artifacts, "lifecycle_report", run_dir / "输出生命周期报告.json")
            _set_artifact_path(artifacts, "calibrator_model", run_dir / "概率校准器.pkl")
            _set_artifact_path(artifacts, "calibrator_meta", run_dir / "概率校准器指标.json")
            _set_artifact_path(artifacts, "score_weights", run_dir / "评分权重.json")

        manifest_path.write_text(
            json.dumps(repaired, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def prepare_artifacts(
    *,
    target_dir: Path,
    local_artifacts_dir: Optional[Path] = None,
    repo_id: Optional[str] = None,
    repo_type: str = "dataset",
    revision: Optional[str] = None,
    force: bool = False,
) -> Path:
    target_dir = target_dir.resolve()
    registry_path = target_dir / "发布索引.json"
    if registry_path.exists() and not force:
        return target_dir

    if target_dir.exists() and force:
        shutil.rmtree(target_dir)

    if local_artifacts_dir is not None:
        _sync_from_local(local_artifacts_dir.resolve(), target_dir)
    else:
        repo_name = str(repo_id or os.environ.get("HF_ARTIFACTS_REPO_ID", "")).strip()
        if not repo_name:
            raise RuntimeError("HF_ARTIFACTS_REPO_ID is required when local artifacts are not provided")
        repo_kind = str(os.environ.get("HF_ARTIFACTS_REPO_TYPE", repo_type)).strip() or "dataset"
        repo_revision = str(os.environ.get("HF_ARTIFACTS_REVISION", revision or "")).strip() or None
        _sync_from_hub(repo_name, repo_kind, repo_revision, target_dir)

    if not registry_path.exists():
        raise FileNotFoundError(f"artifacts registry missing after sync: {registry_path.as_posix()}")
    return target_dir


def build_runtime_config(
    *,
    base_config_path: Path,
    runtime_config_path: Path,
    output_root: Path,
    release_root: Path,
    data_root: Path,
) -> Path:
    with base_config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    output_root = output_root.resolve()
    release_root = release_root.resolve()
    data_root = data_root.resolve()

    serving_cfg = config.setdefault("serving", {})
    feedback_cfg = config.setdefault("feedback", {})
    release_cfg = config.setdefault("release", {})
    output_cfg = config.setdefault("output", {})
    auth_cfg = config.setdefault("auth", {})
    llm_cfg = config.setdefault("llm", {})

    release_cfg["enabled"] = True
    release_cfg["root_dir"] = release_root.as_posix()

    output_cfg["out_dir"] = output_root.as_posix()

    serving_cfg["active_release_policy"] = str(os.environ.get("CROP_ACTIVE_RELEASE_POLICY", "champion")).strip() or "champion"
    serving_cfg["strict_model_loading"] = True
    serving_cfg["log_file"] = (output_root / "service.log").as_posix()

    feedback_root = output_root / "feedback"
    feedback_cfg["inference_log_file"] = (feedback_root / "inference_events.jsonl").as_posix()
    feedback_cfg["feedback_log_file"] = (feedback_root / "user_feedback.jsonl").as_posix()
    feedback_cfg["training_sample_file"] = (feedback_root / "training_samples.jsonl").as_posix()
    feedback_cfg["training_sample_csv"] = (feedback_root / "training_samples.csv").as_posix()
    feedback_cfg["training_summary_file"] = (feedback_root / "training_summary.json").as_posix()
    feedback_cfg["competition_overview_file"] = (feedback_root / "competition_overview.json").as_posix()

    auth_cfg["users_file"] = (data_root / "system" / "users.json").as_posix()
    _apply_llm_env_overrides(llm_cfg)

    _ensure_parent(runtime_config_path)
    runtime_config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return runtime_config_path


def prepare_runtime_environment(
    *,
    runtime_root: Optional[Path] = None,
    local_artifacts_dir: Optional[Path] = None,
    force_artifact_sync: bool = False,
) -> Dict[str, Any]:
    runtime_root = (runtime_root or DEFAULT_RUNTIME_ROOT).resolve()
    output_root = (runtime_root / "output").resolve()
    release_root = (output_root / "release").resolve()
    data_root = (runtime_root / "data").resolve()
    runtime_config_path = (runtime_root / "runtime_config.hf.yaml").resolve()

    force = force_artifact_sync or _bool_env("HF_ARTIFACTS_FORCE_DOWNLOAD", False)
    artifacts_dir = prepare_artifacts(
        target_dir=release_root,
        local_artifacts_dir=local_artifacts_dir,
        force=force,
    )
    _repair_release_manifests(artifacts_dir, output_root)
    config_path = build_runtime_config(
        base_config_path=ROOT / "后端" / "配置.yaml",
        runtime_config_path=runtime_config_path,
        output_root=output_root,
        release_root=artifacts_dir,
        data_root=data_root,
    )

    return {
        "runtime_root": runtime_root.as_posix(),
        "output_root": output_root.as_posix(),
        "release_root": artifacts_dir.as_posix(),
        "data_root": data_root.as_posix(),
        "runtime_config_path": config_path.as_posix(),
    }


def summary_as_json(summary: Dict[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2)
