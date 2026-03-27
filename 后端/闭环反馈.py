from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if out != out:
            return None
        return out
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _to_abs(root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (root / path).resolve()


class ClosedLoopRecorder:
    def __init__(self, *, root: Path, config: dict, logger: Any | None = None):
        feedback_cfg = config.get("feedback", {}) if isinstance(config, dict) else {}
        self.root = root.resolve()
        self.logger = logger
        self.inference_path = _to_abs(self.root, str(feedback_cfg.get("inference_log_file", "输出/闭环/推理事件.jsonl")))
        self.feedback_path = _to_abs(self.root, str(feedback_cfg.get("feedback_log_file", "输出/闭环/用户反馈.jsonl")))
        self.inference_path.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _user_stub(self, user: Dict[str, Any] | None) -> Dict[str, Any]:
        if not isinstance(user, dict):
            return {}
        return {
            "username": user.get("username"),
            "display_name": user.get("display_name"),
            "role": user.get("role"),
        }

    def record_inference(self, *, env_input: Dict[str, Any], payload: Dict[str, Any], user: Dict[str, Any] | None = None) -> Dict[str, Any]:
        event_id = f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        runtime = payload.get("runtime") or {}
        release = runtime.get("release") if isinstance(runtime, dict) else {}
        results = payload.get("results") or []
        record = {
            "event_id": event_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "user": self._user_stub(user),
            "env_input": env_input,
            "release": release if isinstance(release, dict) else {},
            "env": payload.get("env") or {},
            "final_topk": payload.get("final_topk") or [],
            "results": [
                {
                    "crop": row.get("crop"),
                    "env_prob": _safe_float(row.get("env_prob")),
                    "profit": _safe_float(row.get("profit")),
                    "price_pred": _safe_float(row.get("price_pred")),
                    "yield": _safe_float(row.get("yield")),
                    "cost_pred": _safe_float(row.get("cost_pred")),
                    "risk": _safe_float(row.get("risk")),
                    "uncertainty": _safe_float(row.get("uncertainty")),
                    "score": _safe_float(row.get("score")),
                    "target_year": _safe_int(row.get("target_year")),
                }
                for row in results
                if isinstance(row, dict)
            ],
        }
        self._append_jsonl(self.inference_path, record)
        if self.logger is not None and hasattr(self.logger, "info"):
            self.logger.info("closed-loop inference event recorded: %s", event_id)
        return {
            "event_id": event_id,
            "inference_log_file": self.inference_path.as_posix(),
        }

    def record_feedback(
        self,
        *,
        event_id: str,
        selected_crop: str | None = None,
        accepted: bool | None = None,
        actual_profit: float | None = None,
        actual_price: float | None = None,
        actual_yield: float | None = None,
        actual_cost: float | None = None,
        notes: str | None = None,
        user: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        record = {
            "event_id": str(event_id or "").strip(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "user": self._user_stub(user),
            "selected_crop": str(selected_crop or "").strip().lower() or None,
            "accepted": accepted if accepted is None else bool(accepted),
            "actual_profit": _safe_float(actual_profit),
            "actual_price": _safe_float(actual_price),
            "actual_yield": _safe_float(actual_yield),
            "actual_cost": _safe_float(actual_cost),
            "notes": str(notes or "").strip() or None,
        }
        self._append_jsonl(self.feedback_path, record)
        if self.logger is not None and hasattr(self.logger, "info"):
            self.logger.info("closed-loop user feedback recorded: %s", record["event_id"])
        return {
            "ok": True,
            "event_id": record["event_id"],
            "feedback_log_file": self.feedback_path.as_posix(),
            "created_at": record["created_at"],
        }

    def get_status(self) -> Dict[str, Any]:
        inference_lines = 0
        feedback_lines = 0
        latest_inference = None
        latest_feedback = None

        if self.inference_path.exists():
            with self.inference_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        inference_lines += 1
                        latest_inference = line
        if self.feedback_path.exists():
            with self.feedback_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        feedback_lines += 1
                        latest_feedback = line

        def _created_at(raw: str | None) -> str | None:
            if not raw:
                return None
            try:
                return (json.loads(raw) or {}).get("created_at")
            except Exception:
                return None

        return {
            "inference_log_file": self.inference_path.as_posix(),
            "feedback_log_file": self.feedback_path.as_posix(),
            "inference_event_count": inference_lines,
            "feedback_event_count": feedback_lines,
            "latest_inference_at": _created_at(latest_inference),
            "latest_feedback_at": _created_at(latest_feedback),
        }
