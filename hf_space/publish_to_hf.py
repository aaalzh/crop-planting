from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hf_space.hf_auth import resolve_hf_token

DEFAULT_ARTIFACT_SOURCE = ROOT / "输出" / "发布"
SPACE_README_TEMPLATE = ROOT / "hf_space" / "space_readme.md"
EXTRA_SPACE_PATHS = [
    Path(".dockerignore"),
    Path("Dockerfile"),
    Path("requirements-space.txt"),
    Path("hf_space"),
]


def _tracked_files(root: Path) -> List[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=str(root))
    items = [part for part in output.decode("utf-8").split("\0") if part]
    files = {root / item for item in items}

    for rel in EXTRA_SPACE_PATHS:
        candidate = root / rel
        if candidate.is_file():
            files.add(candidate)
            continue
        if candidate.is_dir():
            for path in candidate.rglob("*"):
                if not path.is_file():
                    continue
                if "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                files.add(path)

    return sorted(files)


def _copy_tracked_files(root: Path, dest: Path, replacements: dict[str, str]) -> None:
    for src in _tracked_files(root):
        rel = src.relative_to(root)
        if rel.as_posix() == "README.md":
            continue
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    readme = SPACE_README_TEMPLATE.read_text(encoding="utf-8")
    for key, value in replacements.items():
        readme = readme.replace(key, value)
    (dest / "README.md").write_text(readme, encoding="utf-8")


def _ensure_repo(api: HfApi, repo_id: str, repo_type: str, private: bool, space_sdk: str | None = None) -> None:
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
        space_sdk=space_sdk,
    )


def _upload_artifacts(api: HfApi, repo_id: str, source_dir: Path) -> None:
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(source_dir),
        print_report=True,
        print_report_every=30,
    )


def _upload_space_code(api: HfApi, repo_id: str, staging_dir: Path) -> None:
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(staging_dir),
        delete_patterns="*",
        commit_message="Deploy Docker Space",
    )


def _set_space_runtime(api: HfApi, repo_id: str, artifacts_repo_id: str, token: str) -> None:
    api.add_space_variable(repo_id=repo_id, key="HF_ARTIFACTS_REPO_ID", value=artifacts_repo_id, description="Artifacts dataset repo")
    api.add_space_variable(repo_id=repo_id, key="HF_ARTIFACTS_REPO_TYPE", value="dataset", description="Artifacts repo type")
    api.add_space_variable(repo_id=repo_id, key="HF_ARTIFACTS_REVISION", value="main", description="Artifacts revision")
    api.add_space_variable(repo_id=repo_id, key="CROP_ACTIVE_RELEASE_POLICY", value="champion", description="Release policy")
    api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value=token, description="Used by the Space to download private artifacts")


def publish(
    *,
    artifacts_repo_name: str,
    space_repo_name: str,
    artifacts_private: bool,
    space_private: bool,
    artifacts_source: Path,
) -> dict:
    token = resolve_hf_token()
    if not token:
        raise RuntimeError("Unable to resolve Hugging Face token")

    api = HfApi(token=token)
    whoami = api.whoami(token=token)
    owner = str(whoami["name"])

    artifacts_repo_id = f"{owner}/{artifacts_repo_name}"
    space_repo_id = f"{owner}/{space_repo_name}"

    if not artifacts_source.exists():
        raise FileNotFoundError(f"artifacts source not found: {artifacts_source.as_posix()}")

    _ensure_repo(api, artifacts_repo_id, "dataset", artifacts_private)
    _upload_artifacts(api, artifacts_repo_id, artifacts_source)

    _ensure_repo(api, space_repo_id, "space", space_private, space_sdk="docker")

    with tempfile.TemporaryDirectory(prefix="hf-space-stage-") as tmp_dir:
        staging_dir = Path(tmp_dir)
        _copy_tracked_files(
            ROOT,
            staging_dir,
            replacements={"__ARTIFACT_REPO_ID__": artifacts_repo_id},
        )
        _upload_space_code(api, space_repo_id, staging_dir)

    _set_space_runtime(api, space_repo_id, artifacts_repo_id, token)

    return {
        "owner": owner,
        "artifacts_repo_id": artifacts_repo_id,
        "space_repo_id": space_repo_id,
        "artifacts_url": f"https://huggingface.co/datasets/{artifacts_repo_id}",
        "space_url": f"https://huggingface.co/spaces/{space_repo_id}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish artifacts + Docker Space to Hugging Face")
    parser.add_argument("--artifacts-repo-name", default="crop-planting-artifacts")
    parser.add_argument("--space-repo-name", default="crop-planting-space")
    parser.add_argument("--artifacts-source", default=str(DEFAULT_ARTIFACT_SOURCE))
    parser.add_argument("--artifacts-public", action="store_true")
    parser.add_argument("--space-public", action="store_true")
    args = parser.parse_args()

    result = publish(
        artifacts_repo_name=args.artifacts_repo_name,
        space_repo_name=args.space_repo_name,
        artifacts_private=not bool(args.artifacts_public),
        space_private=not bool(args.space_public),
        artifacts_source=Path(args.artifacts_source).resolve(),
    )
    for key, value in result.items():
        print(f"{key}\t{value}")


if __name__ == "__main__":
    main()
