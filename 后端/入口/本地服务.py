from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import OrderedDict
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Support both startup styles:
# - python -m 后端.本地服务
# - python 后端/入口/本地服务.py
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from 后端.认证存储 import (
    AUTH_COOKIE_NAME,
    SessionStore,
    UserStore,
    clear_auth_cookie,
    resolve_users_path,
    set_auth_cookie,
)
from 后端.图表资源 import resolve_chart_dir
from 后端.闭环反馈 import ClosedLoopRecorder
from 后端.作物可视化服务 import _trend_from_year_rows, build_crop_visual_payload
from 后端.路由.认证路由 import register_auth_routes
from 后端.路由.核心路由 import register_core_routes
from 后端.路由.推荐路由 import register_recommendation_routes
from 后端.决策支持 import DecisionSupportService
from 后端.时间策略 import resolve_price_window_from_price_dir
from 后端.数据加载 import load_config, load_name_map, resolve_names
from 后端.推荐数据源 import recommend_with_source

CONFIG_PATH = ROOT / "后端" / "配置.yaml"
OUTPUT_DIR = ROOT / "输出"
FRONTEND_DIR = ROOT / "前端"
FRONTEND_PAGES_DIR = FRONTEND_DIR / "页面"
CHART_DIR = OUTPUT_DIR / "项目图片"
ALT_CHART_DIR = ROOT / "可视化图片" / "项目图片"
LOGIN_PAGE_URL = "/"

logger = logging.getLogger("crop_server")


def _setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid duplicate handlers when module is imported multiple times.
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


class TTLCache:
    def __init__(self, ttl_seconds: int = 300, max_entries: int = 256):
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._data: OrderedDict[str, tuple[float, Dict[str, Any]]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Dict[str, Any] | None:
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            expire_at, payload = item
            if expire_at < now:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return payload

    def set(self, key: str, payload: Dict[str, Any]) -> None:
        now = time.monotonic()
        expire_at = now + self.ttl_seconds
        with self._lock:
            self._data[key] = (expire_at, payload)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)


class EnvInput(BaseModel):
    N: float = Field(..., ge=0)
    P: float = Field(..., ge=0)
    K: float = Field(..., ge=0)
    temperature: float
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=0, le=14)
    rainfall: float = Field(..., ge=0)


class RecommendRequest(BaseModel):
    env: EnvInput


class RecommendFeedbackRequest(BaseModel):
    event_id: str = Field(..., min_length=8, max_length=128)
    selected_crop: Optional[str] = Field(default=None, max_length=64)
    accepted: Optional[bool] = None
    actual_profit: Optional[float] = None
    actual_price: Optional[float] = None
    actual_yield: Optional[float] = None
    actual_cost: Optional[float] = None
    notes: Optional[str] = Field(default=None, max_length=1000)


class CropVisualRequest(BaseModel):
    crop: str = Field(..., min_length=1, max_length=64)
    price_file: Optional[str] = Field(default=None, max_length=128)
    cost_name: Optional[str] = Field(default=None, max_length=128)
    price_pred: Optional[float] = None
    price_forecast: Optional[List[Dict[str, Any]]] = None
    yield_pred: Optional[float] = None
    cost_pred: Optional[float] = None
    cost_pred_raw: Optional[float] = None
    profit_pred: Optional[float] = None
    env_prob: Optional[float] = None
    prob_best: Optional[float] = None
    risk: Optional[float] = None
    score: Optional[float] = None
    horizon_days: Optional[int] = Field(default=None, ge=1, le=730)
    target_year: Optional[int] = Field(default=None, ge=1900, le=2200)
    history_years: Optional[int] = Field(default=None, ge=3, le=30)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=8, max_length=128)
    display_name: Optional[str] = Field(default=None, max_length=32)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=1, max_length=128)


class AdminUserUpdateRequest(BaseModel):
    role: Optional[str] = None
    enabled: Optional[bool] = None
    display_name: Optional[str] = Field(default=None, max_length=32)


class AssistantAnswerRequest(BaseModel):
    question_id: Optional[str] = Field(default=None, max_length=64)
    question_text: Optional[str] = Field(default=None, max_length=1200)
    crop: Optional[str] = Field(default=None, max_length=64)


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


def _save_recommendation(payload: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "推荐结果.json"
    csv_path = out_dir / "推荐结果.csv"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(payload.get("results", [])).to_csv(csv_path, index=False)


def _safe_float(v: Any) -> float | None:
    try:
        val = float(v)
    except Exception:
        return None
    if pd.isna(val):
        return None
    return val


def _to_abs_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return ROOT / p


def _safe_int(v: Any) -> int | None:
    try:
        val = int(v)
    except Exception:
        return None
    return val


def _safe_timestamp(v: Any) -> pd.Timestamp | None:
    text = str(v or "").strip()
    if not text:
        return None
    try:
        ts = pd.Timestamp(text)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.normalize()


def _resolve_prediction_window(time_cfg: Dict[str, Any], price_dir_path: Path) -> Dict[str, Any]:
    policy = resolve_price_window_from_price_dir(price_dir=price_dir_path, time_cfg=time_cfg)
    return {
        "start_date": policy["start_date"],
        "end_date": policy["end_date"],
        "price_horizon_days": int(policy["price_horizon_days"]),
    }


def _public_user(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "username": user.get("username"),
        "display_name": user.get("display_name") or user.get("username"),
        "role": str(user.get("role") or "user").lower(),
        "enabled": UserStore._normalize_enabled(user.get("enabled", True)),
        "created_at": user.get("created_at"),
        "last_login_at": user.get("last_login_at"),
    }


def create_app() -> FastAPI:
    config = load_config(str(CONFIG_PATH))
    serving_cfg = config.get("serving", {})
    auth_cfg = config.get("auth", {})
    paths_cfg = config.get("paths", {})
    alignment_cfg = config.get("alignment", {})
    time_cfg = config.get("time", {})

    log_file = ROOT / serving_cfg.get("log_file", "输出/服务日志.log")
    _setup_logging(log_file)

    ttl_seconds = int(serving_cfg.get("request_cache_ttl_seconds", 300))
    max_entries = int(serving_cfg.get("request_cache_max_entries", 256))
    cache = TTLCache(ttl_seconds=ttl_seconds, max_entries=max_entries)
    chart_dir = resolve_chart_dir(CHART_DIR, ALT_CHART_DIR)
    price_dir_path = _to_abs_path(str(paths_cfg.get("price_dir", "价格数据")))
    prediction_window = _resolve_prediction_window(time_cfg, price_dir_path=price_dir_path)
    prediction_start_date = prediction_window["start_date"]
    prediction_end_date = prediction_window["end_date"]
    prediction_price_horizon_days = int(prediction_window["price_horizon_days"])
    cost_file_path = _to_abs_path(str(paths_cfg.get("cost_file", "成本数据/原始/加权平均成本数据.csv")))
    yield_history_path = _to_abs_path(str(paths_cfg.get("yield_history", "")))

    resolved_name_map: Dict[str, Dict[str, str]] = {}
    try:
        name_map_path = _to_abs_path(str(paths_cfg.get("name_map", "数据/映射/作物名称映射.csv")))
        resolved_name_map = resolve_names(load_name_map(str(name_map_path)))
    except Exception:
        logger.exception("failed loading name map for crop visuals")

    users_path = resolve_users_path(root=ROOT, config=config)
    user_store = UserStore(users_path)
    logger.info("user store path: %s", users_path.as_posix())
    session_ttl_seconds = int(auth_cfg.get("session_ttl_seconds", 3600 * 12))
    session_store = SessionStore(
        ttl_seconds=session_ttl_seconds,
        max_entries=int(auth_cfg.get("session_max_entries", 4096)),
    )

    app = FastAPI(title="农作物种植推荐服务", version="2.0.0")
    logger.info("chart dir: %s", chart_dir.as_posix())
    decision_service = DecisionSupportService(root=ROOT, output_dir=OUTPUT_DIR, logger=logger)
    closed_loop_recorder = ClosedLoopRecorder(root=ROOT, config=config, logger=logger)

    app.mount("/assets/前端", StaticFiles(directory=str(FRONTEND_DIR), check_dir=True), name="frontend")
    app.mount("/assets/charts", StaticFiles(directory=str(chart_dir), check_dir=False), name="charts")

    def get_current_user(request: Request) -> Dict[str, Any]:
        token = request.cookies.get(AUTH_COOKIE_NAME)
        username = session_store.get_user(token)
        if not username:
            raise HTTPException(status_code=401, detail="请先登录")
        user = user_store.get_user(username)
        if not user:
            raise HTTPException(status_code=401, detail="用户不存在或已失效")
        if not UserStore._normalize_enabled(user.get("enabled", True)):
            session_store.revoke(token)
            raise HTTPException(status_code=403, detail="账号已被禁用，请联系管理员")
        return user

    def get_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        if str(current_user.get("role", "")).lower() != "admin":
            raise HTTPException(status_code=403, detail="需要管理员权限")
        return current_user

    def serve_front_page(file_name: str) -> FileResponse:
        page_path = FRONTEND_PAGES_DIR / file_name
        if not page_path.exists():
            raise HTTPException(status_code=404, detail=f"前端页面不存在: 前端/页面/{file_name}")
        return FileResponse(
            page_path,
            headers={
                "Cache-Control": "no-store, max-age=0",
                "Pragma": "no-cache",
            },
        )

    def login_redirect() -> RedirectResponse:
        return RedirectResponse(url=LOGIN_PAGE_URL, status_code=303)

    def guarded_page(file_name: str, request: Request) -> Response:
        try:
            get_current_user(request)
        except HTTPException:
            return login_redirect()
        return serve_front_page(file_name)

    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
            # Avoid stale frontend scripts/styles in local browser cache.
            if request.url.path.startswith("/assets/前端/"):
                response.headers["Cache-Control"] = "no-store, max-age=0"
                response.headers["Pragma"] = "no-cache"
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.3f}"
            logger.info("%s %s -> %s in %.2f ms", request.method, request.url.path, response.status_code, elapsed_ms)
            return response
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.exception("Unhandled exception for %s %s after %.2f ms", request.method, request.url.path, elapsed_ms)
            return JSONResponse(status_code=500, content={"detail": "服务器内部错误，请查看日志"})

    register_auth_routes(
        app,
        get_current_user=get_current_user,
        get_admin_user=get_admin_user,
        user_store=user_store,
        session_store=session_store,
        session_ttl_seconds=session_ttl_seconds,
        auth_cookie_name=AUTH_COOKIE_NAME,
        login_page_url=LOGIN_PAGE_URL,
        set_auth_cookie=set_auth_cookie,
        clear_auth_cookie=clear_auth_cookie,
        public_user=_public_user,
        normalize_enabled=UserStore._normalize_enabled,
        register_model=RegisterRequest,
        login_model=LoginRequest,
        admin_update_model=AdminUserUpdateRequest,
    )

    register_core_routes(
        app,
        get_current_user=get_current_user,
        get_admin_user=get_admin_user,
        serve_front_page=serve_front_page,
        guarded_page=guarded_page,
        root=ROOT,
        output_dir=OUTPUT_DIR,
        chart_dir=chart_dir,
        safe_read_json=_safe_read_json,
        logger=logger,
        config=config,
        recommend_with_source=recommend_with_source,
        decision_service=decision_service,
        closed_loop_recorder=closed_loop_recorder,
        assistant_answer_model=AssistantAnswerRequest,
    )

    register_recommendation_routes(
        app,
        get_current_user=get_current_user,
        crop_visual_model=CropVisualRequest,
        recommend_model=RecommendRequest,
        feedback_model=RecommendFeedbackRequest,
        resolved_name_map=resolved_name_map,
        prediction_price_horizon_days=int(prediction_price_horizon_days),
        prediction_start_date=prediction_start_date,
        prediction_end_date=prediction_end_date,
        alignment_cfg=alignment_cfg,
        price_dir_path=price_dir_path,
        cost_file_path=cost_file_path,
        yield_history_path=yield_history_path,
        safe_float=_safe_float,
        safe_int=_safe_int,
        build_crop_visual_payload=build_crop_visual_payload,
        cache=cache,
        ttl_seconds=int(ttl_seconds),
        config=config,
        root=ROOT,
        output_dir=OUTPUT_DIR,
        logger=logger,
        recommend_with_source=recommend_with_source,
        save_recommendation=_save_recommendation,
        decision_service=decision_service,
        closed_loop_recorder=closed_loop_recorder,
    )

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local web server for crop recommendation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()




