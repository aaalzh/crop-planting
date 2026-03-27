from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from fastapi import Response

AUTH_COOKIE_NAME = "crop_session"
PBKDF2_ITERS = 240000
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,32}$")
VALID_ROLES = {"admin", "user"}
USERS_PATH_ENV = "CROP_USERS_PATH"


def resolve_users_path(root: Path, config: Dict[str, Any]) -> Path:
    env_path = str(os.environ.get(USERS_PATH_ENV, "")).strip()
    if env_path:
        return Path(env_path).expanduser().resolve()

    auth_cfg = config.get("auth", {}) if isinstance(config, dict) else {}
    cfg_path = str(auth_cfg.get("users_file", "")).strip()
    if cfg_path:
        p = Path(cfg_path)
        if not p.is_absolute():
            p = (root / cfg_path).resolve()
        return p
    return (root / "数据" / "用户.json").resolve()


def set_auth_cookie(response: Response, token: str, ttl_seconds: int) -> None:
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        max_age=ttl_seconds,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(key=AUTH_COOKIE_NAME, path="/")


def _safe_read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


class SessionStore:
    def __init__(self, ttl_seconds: int = 3600 * 12, max_entries: int = 4096):
        self.ttl_seconds = max(120, int(ttl_seconds))
        self.max_entries = max(16, int(max_entries))
        self._sessions: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = Lock()

    def _cleanup(self, now: float) -> None:
        dead = [token for token, (exp, _) in self._sessions.items() if exp < now]
        for token in dead:
            self._sessions.pop(token, None)

    def issue(self, username: str) -> str:
        now = time.monotonic()
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._cleanup(now)
            self._sessions[token] = (now + self.ttl_seconds, username)
            self._sessions.move_to_end(token)
            while len(self._sessions) > self.max_entries:
                self._sessions.popitem(last=False)
        return token

    def get_user(self, token: str | None) -> str | None:
        if not token:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._sessions.get(token)
            if not item:
                return None
            exp, username = item
            if exp < now:
                self._sessions.pop(token, None)
                return None
            self._sessions.move_to_end(token)
            return username

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)


class UserStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"users": []}, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _normalize_enabled(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if value is None:
            return True
        text = str(value).strip().lower()
        if text in {"", "0", "false", "no", "off"}:
            return False
        return True

    def _read_all(self) -> Dict[str, Any]:
        self._ensure_file()
        data = _safe_read_json(self.path)
        if not isinstance(data, dict):
            return {"users": []}
        users = data.get("users")
        if not isinstance(users, list):
            data["users"] = []
            users = data["users"]

        changed = False
        normalized_users: List[Dict[str, Any]] = []
        for user in users:
            if not isinstance(user, dict):
                changed = True
                continue

            username = str(user.get("username", "")).strip()
            if not username:
                changed = True
                continue

            record = dict(user)
            record["username"] = username

            display_name = str(record.get("display_name") or username).strip()[:32]
            if record.get("display_name") != display_name:
                changed = True
            record["display_name"] = display_name or username

            role = str(record.get("role") or "").strip().lower()
            if role not in VALID_ROLES:
                role = "user"
                changed = True
            record["role"] = role

            enabled = self._normalize_enabled(record.get("enabled", True))
            if record.get("enabled") != enabled:
                changed = True
            record["enabled"] = enabled

            if not record.get("created_at"):
                record["created_at"] = datetime.now().isoformat(timespec="seconds")
                changed = True
            if "last_login_at" not in record:
                record["last_login_at"] = None
                changed = True

            normalized_users.append(record)

        if normalized_users:
            has_enabled_admin = any(
                str(u.get("role", "")).lower() == "admin" and self._normalize_enabled(u.get("enabled", True))
                for u in normalized_users
            )
            if not has_enabled_admin:
                normalized_users[0]["role"] = "admin"
                normalized_users[0]["enabled"] = True
                changed = True

        data["users"] = normalized_users
        if changed:
            self._write_all(data)
        return data

    def _write_all(self, data: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _hash_password(password: str, salt_hex: str) -> str:
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            PBKDF2_ITERS,
        )
        return digest.hex()

    @staticmethod
    def _validate_password(password: str) -> None:
        errors: List[str] = []
        if len(password) < 8:
            errors.append("长度至少 8 位")
        if not re.search(r"[a-z]", password):
            errors.append("必须包含小写字母")
        if not re.search(r"[A-Z]", password):
            errors.append("必须包含大写字母")
        if not re.search(r"[0-9]", password):
            errors.append("必须包含数字")
        if not re.search(r"[^A-Za-z0-9]", password):
            errors.append("必须包含符号")
        if errors:
            raise ValueError("密码不符合规则：" + "；".join(errors))

    def get_user(self, username: str) -> Dict[str, Any] | None:
        needle = username.strip().lower()
        if not needle:
            return None
        with self._lock:
            data = self._read_all()
            for user in data["users"]:
                if str(user.get("username", "")).lower() == needle:
                    return dict(user)
        return None

    def create_user(self, username: str, password: str, display_name: str | None = None) -> Dict[str, Any]:
        user_name_clean = username.strip()
        if not USERNAME_PATTERN.match(user_name_clean):
            raise ValueError("用户名需为 3-32 位字母/数字/下划线")
        self._validate_password(password)
        safe_display = (display_name or user_name_clean).strip()[:32]
        now = datetime.now().isoformat(timespec="seconds")

        with self._lock:
            data = self._read_all()
            for user in data["users"]:
                if str(user.get("username", "")).lower() == user_name_clean.lower():
                    raise ValueError("用户名已存在")

            has_enabled_admin = any(
                str(u.get("role", "")).lower() == "admin" and self._normalize_enabled(u.get("enabled", True))
                for u in data["users"]
            )

            salt = secrets.token_hex(16)
            pwd_hash = self._hash_password(password, salt)
            record = {
                "username": user_name_clean,
                "display_name": safe_display or user_name_clean,
                "role": "user" if has_enabled_admin else "admin",
                "enabled": True,
                "salt": salt,
                "password_hash": pwd_hash,
                "created_at": now,
                "last_login_at": None,
            }
            data["users"].append(record)
            self._write_all(data)
            return dict(record)

    def verify_user(self, username: str, password: str) -> Dict[str, Any] | None:
        user = self.get_user(username)
        if user is None:
            return None
        if not self._normalize_enabled(user.get("enabled", True)):
            return None
        salt = user.get("salt")
        password_hash = user.get("password_hash")
        if not isinstance(salt, str) or not isinstance(password_hash, str):
            return None
        calc = self._hash_password(password, salt)
        if not hmac.compare_digest(calc, password_hash):
            return None
        return user

    def touch_login(self, username: str) -> Dict[str, Any] | None:
        with self._lock:
            data = self._read_all()
            for user in data["users"]:
                if str(user.get("username", "")).lower() == username.lower():
                    user["last_login_at"] = datetime.now().isoformat(timespec="seconds")
                    self._write_all(data)
                    return dict(user)
        return None

    def list_users(self) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._read_all()
            users = [dict(u) for u in data["users"]]
        users.sort(key=lambda x: (0 if x.get("role") == "admin" else 1, str(x.get("username", "")).lower()))
        return users

    def update_user(
        self,
        username: str,
        *,
        role: str | None = None,
        enabled: bool | None = None,
        display_name: str | None = None,
        actor_username: str | None = None,
    ) -> Dict[str, Any]:
        user_name_clean = username.strip()
        if not user_name_clean:
            raise ValueError("用户名不能为空")

        normalized_role: str | None = None
        if role is not None:
            normalized_role = str(role).strip().lower()
            if normalized_role not in VALID_ROLES:
                raise ValueError("角色只支持 admin 或 user")

        clean_display: str | None = None
        if display_name is not None:
            clean_display = display_name.strip()[:32]

        with self._lock:
            data = self._read_all()
            target: Dict[str, Any] | None = None
            for user in data["users"]:
                if str(user.get("username", "")).lower() == user_name_clean.lower():
                    target = user
                    break
            if target is None:
                raise ValueError("目标用户不存在")

            actor_match = (
                actor_username is not None
                and str(target.get("username", "")).lower() == actor_username.strip().lower()
            )
            if actor_match and normalized_role == "user":
                raise ValueError("不能将当前登录管理员降级为普通用户")
            if actor_match and enabled is False:
                raise ValueError("不能禁用当前登录管理员")

            if normalized_role is not None:
                target["role"] = normalized_role
            if enabled is not None:
                target["enabled"] = bool(enabled)
            if clean_display is not None:
                target["display_name"] = clean_display or target["username"]

            enabled_admins = [
                u
                for u in data["users"]
                if str(u.get("role", "")).lower() == "admin" and self._normalize_enabled(u.get("enabled", True))
            ]
            if not enabled_admins:
                raise ValueError("系统至少保留一个启用中的管理员")

            self._write_all(data)
            return dict(target)
