from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse

from 后端.决策支持 import AssistantUnavailableError
from 后端.环境桥接 import load_env_scenario_library
from 后端.发布治理 import get_release_status, promote_release, rollback_release
from 后端.竞赛概览 import get_competition_overview
from 后端.反馈回流 import build_feedback_training_dataset, get_feedback_training_status
from 后端.图表资源 import group_chart_images, list_chart_images
from 后端.输出洞察 import insights_summary, metrics_summary


def register_core_routes(
    app: FastAPI,
    *,
    get_current_user: Callable[..., Dict[str, Any]],
    get_admin_user: Callable[..., Dict[str, Any]],
    serve_front_page: Callable[[str], FileResponse],
    guarded_page: Callable[[str, Request], Response],
    root: Path,
    output_dir: Path,
    chart_dir: Path,
    safe_read_json: Callable[[Path], Optional[Dict[str, Any]]],
    logger,
    config: Dict[str, Any],
    recommend_with_source: Callable[..., Dict[str, Any]],
    decision_service,
    closed_loop_recorder,
    assistant_answer_model,
) -> None:
    @app.get("/")
    def index(request: Request) -> Response:
        try:
            get_current_user(request)
        except HTTPException:
            return serve_front_page("登录.html")
        return RedirectResponse(url="/recommend", status_code=303)

    @app.get("/dashboard")
    def dashboard() -> Response:
        return RedirectResponse(url="/recommend", status_code=303)

    @app.get("/home")
    def home_page() -> Response:
        return RedirectResponse(url="/recommend", status_code=303)

    @app.get("/recommend")
    def recommend_page(request: Request) -> Response:
        return guarded_page("推荐.html", request)

    @app.get("/analytics")
    def analytics_page() -> Response:
        return RedirectResponse(url="/assistant", status_code=303)

    @app.get("/assistant")
    def assistant_page(request: Request) -> Response:
        return guarded_page("分析.html", request)

    @app.get("/charts")
    def charts_page() -> Response:
        return RedirectResponse(url="/recommend", status_code=303)

    @app.get("/market")
    def market_page() -> Response:
        return RedirectResponse(url="/recommend", status_code=303)

    @app.get("/store")
    def store_page(request: Request) -> Response:
        return guarded_page("商城.html", request)

    @app.get("/store/product/{product_id}")
    def store_product_page(product_id: str, request: Request) -> Response:
        return guarded_page("商品详情.html", request)

    @app.get("/community")
    def community_page(request: Request) -> Response:
        return guarded_page("交流.html", request)

    @app.get("/profile")
    def profile_page(request: Request) -> Response:
        return guarded_page("我的.html", request)

    @app.get("/admin")
    def admin_page(_: Dict[str, Any] = Depends(get_admin_user)) -> FileResponse:
        return serve_front_page("管理.html")

    @app.get("/api/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "time": datetime.now().isoformat(timespec="seconds")}

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/default-env")
    def default_env(_: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        data_path = root / "数据" / "样例" / "环境示例.json"
        data = safe_read_json(data_path)
        if data is None:
            raise HTTPException(status_code=404, detail="默认环境参数不存在")
        return data

    @app.get("/api/metrics")
    def metrics(_: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        return metrics_summary(output_dir)

    @app.get("/api/charts")
    def charts(_: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        images = list_chart_images(chart_dir)
        return {"images": images, "groups": group_chart_images(images)}

    @app.get("/api/insights")
    def insights(_: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        return insights_summary(output_dir, logger=logger)

    @app.get("/api/home-summary")
    def home_summary(_: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        return decision_service.build_home_summary(
            config=config,
            recommend_with_source=recommend_with_source,
            logger=logger,
        )

    @app.get("/api/store-summary")
    def store_summary(_: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        return decision_service.build_store_summary(
            config=config,
            recommend_with_source=recommend_with_source,
            logger=logger,
        )

    @app.get("/api/community-summary")
    def community_summary(_: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        return decision_service.build_community_summary(
            config=config,
            recommend_with_source=recommend_with_source,
            logger=logger,
        )

    @app.get("/api/profile-summary")
    def profile_summary(_: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        return decision_service.build_profile_summary(
            config=config,
            recommend_with_source=recommend_with_source,
            logger=logger,
        )

    @app.get("/api/recommend-history")
    def recommend_history(_: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        return {"items": decision_service.load_history()}

    @app.get("/api/release-status")
    def release_status(_: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        return get_release_status(root=root, config=config)

    @app.get("/api/closed-loop-status")
    def closed_loop_status(_: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        payload = closed_loop_recorder.get_status()
        payload["release"] = get_release_status(root=root, config=config)
        payload["feedback_training"] = get_feedback_training_status(root=root, config=config, refresh=True)
        return payload

    @app.get("/api/feedback-training-status")
    def feedback_training_status(_: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        return get_feedback_training_status(root=root, config=config, refresh=True)

    @app.get("/api/competition-overview")
    def competition_overview(_: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        return get_competition_overview(root=root, config=config, refresh=True)

    @app.post("/api/feedback-training/rebuild")
    def feedback_training_rebuild(_: Dict[str, Any] = Depends(get_admin_user)) -> Dict[str, Any]:
        return build_feedback_training_dataset(root=root, config=config, save=True)

    @app.get("/api/env-scenarios")
    def env_scenarios(_: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        return load_env_scenario_library(root=root, config=config, rebuild_if_missing=True)

    @app.post("/api/release/promote")
    def release_promote(
        run_id: str,
        _: Dict[str, Any] = Depends(get_admin_user),
    ) -> Dict[str, Any]:
        text = str(run_id or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="run_id 不能为空")
        return promote_release(root=root, config=config, run_id=text, reason="api_promote")

    @app.post("/api/release/rollback")
    def release_rollback(
        run_id: Optional[str] = None,
        _: Dict[str, Any] = Depends(get_admin_user),
    ) -> Dict[str, Any]:
        text = str(run_id or "").strip() or None
        return rollback_release(root=root, config=config, run_id=text)

    def _assistant_answer_impl(question_id: Optional[str], question_text: Optional[str], crop: Optional[str]) -> Dict[str, Any]:
        try:
            return decision_service.answer_question(
                question_id=question_id,
                question_text=question_text,
                crop=crop,
                config=config,
                recommend_with_source=recommend_with_source,
                logger=logger,
            )
        except AssistantUnavailableError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.get("/api/assistant-answer")
    def assistant_answer(
        question_id: Optional[str] = None,
        question_text: Optional[str] = None,
        crop: Optional[str] = None,
        _: Dict[str, Any] = Depends(get_current_user),
    ) -> Dict[str, Any]:
        return _assistant_answer_impl(question_id=question_id, question_text=question_text, crop=crop)

    @app.post("/api/assistant-answer")
    async def assistant_answer_post(
        request: Request,
        _: Dict[str, Any] = Depends(get_current_user),
    ) -> Dict[str, Any]:
        raw_payload: Dict[str, Any] = {}
        try:
            parsed = await request.json()
            if isinstance(parsed, dict):
                raw_payload = parsed
        except Exception:
            raw_payload = {}

        question_id = raw_payload.get("question_id")
        question_text = raw_payload.get("question_text")
        crop = raw_payload.get("crop")

        if question_id is not None:
            question_id = str(question_id).strip() or None
        if question_text is not None:
            question_text = str(question_text).strip() or None
        if crop is not None:
            crop = str(crop).strip() or None

        return _assistant_answer_impl(
            question_id=question_id,
            question_text=question_text,
            crop=crop,
        )
