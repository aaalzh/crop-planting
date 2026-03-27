from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _bytes_of_path(path: str) -> int:
    if not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        try:
            return int(os.path.getsize(path))
        except OSError:
            return 0

    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += int(os.path.getsize(fp))
            except OSError:
                continue
    return int(total)


def _list_dirs_sorted(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    dirs = []
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.isdir(p):
            dirs.append(p)
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return dirs


def _delete_path(path: str, dry_run: bool) -> Tuple[bool, int, str]:
    size = _bytes_of_path(path)
    if dry_run:
        return True, size, ""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
        return True, size, ""
    except Exception as exc:
        return False, 0, str(exc)


def _snapshot_outputs(out_dir: str, archive_dir: str, files: List[str], run_tag: str) -> Dict[str, object]:
    _safe_mkdir(archive_dir)
    run_dir = os.path.join(archive_dir, run_tag)
    _safe_mkdir(run_dir)

    copied = []
    missing = []
    total_bytes = 0
    for name in files:
        src = os.path.join(out_dir, name)
        if not os.path.exists(src):
            missing.append(name)
            continue
        dst = os.path.join(run_dir, name)
        _safe_mkdir(os.path.dirname(dst))
        shutil.copy2(src, dst)
        copied.append(name)
        total_bytes += _bytes_of_path(dst)

    return {
        "run_dir": run_dir,
        "copied_files": copied,
        "missing_files": missing,
        "snapshot_bytes": int(total_bytes),
    }


def apply_output_lifecycle(
    out_dir: str,
    config: Dict[str, object],
    run_tag: str,
) -> Dict[str, object]:
    cfg = config.get("outputs_lifecycle", {}) if isinstance(config, dict) else {}
    enabled = bool(cfg.get("enabled", True))
    if not enabled:
        return {"enabled": False}

    dry_run = bool(cfg.get("dry_run", False))
    archive_dir = str(cfg.get("archive_dir", os.path.join(out_dir, "归档运行")))
    keep_latest_runs = int(cfg.get("keep_latest_runs", 12))
    keep_latest_runs = max(1, keep_latest_runs)

    backup_dir = str(cfg.get("backup_dir", os.path.join(out_dir, "备份")))
    keep_latest_backups = int(cfg.get("keep_latest_backups", 8))
    keep_latest_backups = max(0, keep_latest_backups)
    max_backup_age_days = int(cfg.get("max_backup_age_days", 30))
    max_backup_age_days = max(1, max_backup_age_days)
    require_age_for_backup_delete = bool(cfg.get("require_age_for_backup_delete", False))
    age_cutoff = datetime.now() - timedelta(days=max_backup_age_days)

    snapshot_files = cfg.get(
        "snapshot_files",
        [
            "模型训练报告.json",
            "回测报告.json",
            "推荐结果.csv",
            "高精度训练报告.md",
        ],
    )
    snapshot_files = [str(x) for x in snapshot_files]

    snap = _snapshot_outputs(out_dir, archive_dir=archive_dir, files=snapshot_files, run_tag=run_tag)

    deleted_runs = []
    deleted_backups = []
    failed = []
    freed_bytes = 0

    run_dirs = _list_dirs_sorted(archive_dir)
    for p in run_dirs[keep_latest_runs:]:
        ok, b, err = _delete_path(p, dry_run=dry_run)
        if ok:
            deleted_runs.append(p)
            freed_bytes += int(b)
        else:
            failed.append({"path": p, "error": err})

    backup_dirs = _list_dirs_sorted(backup_dir)
    for idx, p in enumerate(backup_dirs):
        if idx < keep_latest_backups:
            continue
        if require_age_for_backup_delete:
            mtime = datetime.fromtimestamp(os.path.getmtime(p))
            if mtime > age_cutoff:
                continue
        ok, b, err = _delete_path(p, dry_run=dry_run)
        if ok:
            deleted_backups.append(p)
            freed_bytes += int(b)
        else:
            failed.append({"path": p, "error": err})

    return {
        "enabled": True,
        "dry_run": dry_run,
        "run_tag": run_tag,
        "snapshot": snap,
        "archive_dir": archive_dir,
        "backup_dir": backup_dir,
        "keep_latest_runs": keep_latest_runs,
        "keep_latest_backups": keep_latest_backups,
        "max_backup_age_days": max_backup_age_days,
        "require_age_for_backup_delete": require_age_for_backup_delete,
        "deleted_archive_runs": deleted_runs,
        "deleted_backups": deleted_backups,
        "freed_bytes": int(freed_bytes),
        "failed": failed,
    }
