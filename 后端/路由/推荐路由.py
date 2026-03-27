import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException


def register_recommendation_routes(
    app: FastAPI,
    *,
    get_current_user: Callable[..., Dict[str, Any]],
    crop_visual_model,
    recommend_model,
    feedback_model,
    resolved_name_map: Dict[str, Dict[str, str]],
    prediction_price_horizon_days: int,
    prediction_start_date: datetime,
    prediction_end_date: datetime,
    alignment_cfg: Dict[str, Any],
    price_dir_path: Path,
    cost_file_path: Path,
    yield_history_path: Path,
    safe_float: Callable[[Any], Optional[float]],
    safe_int: Callable[[Any], Optional[int]],
    build_crop_visual_payload: Callable[..., Dict[str, Any]],
    cache,
    ttl_seconds: int,
    config: Dict[str, Any],
    root: Path,
    output_dir: Path,
    logger,
    recommend_with_source: Callable[..., Dict[str, Any]],
    save_recommendation: Callable[[Dict[str, Any], Path], None],
    decision_service,
    closed_loop_recorder,
) -> None:
    @app.post("/api/crop-visuals")
    def crop_visuals(req: crop_visual_model, _: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        crop = str(req.crop or "").strip().lower()
        if not crop:
            raise HTTPException(status_code=400, detail="crop 不能为空")

        mapping = resolved_name_map.get(crop, {})
        price_file = str(req.price_file or mapping.get("price_file") or "").strip()
        cost_name = str(req.cost_name or mapping.get("cost_name") or "").strip()

        horizon_days = int(prediction_price_horizon_days)
        price_pred = safe_float(req.price_pred)
        price_forecast = req.price_forecast if isinstance(req.price_forecast, list) else None
        yield_pred = safe_float(req.yield_pred)
        cost_pred = safe_float(req.cost_pred)
        cost_pred_raw = safe_float(req.cost_pred_raw)
        profit_pred = safe_float(req.profit_pred)
        env_prob = safe_float(req.env_prob)
        prob_best = safe_float(req.prob_best)
        risk = safe_float(req.risk)
        score = safe_float(req.score)

        target_year = safe_int(req.target_year)
        if target_year is None:
            target_year = safe_int(alignment_cfg.get("target_year"))
        if target_year is None:
            target_year = int(prediction_end_date.year)

        history_years = safe_int(req.history_years)
        if history_years is None:
            history_years = safe_int(alignment_cfg.get("history_years"))
        if history_years is None:
            history_years = 8
        history_years = max(3, min(30, int(history_years)))

        return build_crop_visual_payload(
            crop=crop,
            price_file=price_file,
            cost_name=cost_name,
            price_pred=price_pred,
            price_forecast=price_forecast,
            yield_pred=yield_pred,
            cost_pred=cost_pred,
            cost_pred_raw=cost_pred_raw,
            profit_pred=profit_pred,
            env_prob=env_prob,
            prob_best=prob_best,
            risk=risk,
            score=score,
            target_year=int(target_year),
            history_years=int(history_years),
            prediction_start_date=prediction_start_date,
            prediction_end_date=prediction_end_date,
            horizon_days=int(horizon_days),
            price_summary_window_days=max(1, int(config.get("time", {}).get("price_summary_window_days", 30))),
            price_dir_path=price_dir_path,
            cost_file_path=cost_file_path,
            yield_history_path=yield_history_path,
            alignment_cfg=alignment_cfg,
        )

    @app.post("/api/recommend")
    def do_recommend(req: recommend_model, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        key = json.dumps(req.env.model_dump(), ensure_ascii=False, sort_keys=True)
        cached = cache.get(key)
        if cached is not None:
            cached_payload = cached
            if not isinstance(cached_payload.get("decision_summary"), dict):
                cached_payload = decision_service.enrich_payload(cached_payload)
            out = copy.deepcopy(cached_payload)
            feedback_meta = closed_loop_recorder.record_inference(
                env_input=req.env.model_dump(),
                payload=out,
                user=current_user,
            )
            out["feedback"] = feedback_meta
            out.setdefault("runtime", {})["feedback_event_id"] = feedback_meta["event_id"]
            decision_service.append_history(out)
            out["cache"] = {"hit": True, "ttl_seconds": ttl_seconds}
            return out

        t0 = time.perf_counter()
        payload = recommend_with_source(
            req.env.model_dump(),
            config=config,
            root=root,
            output_dir=output_dir,
            logger=logger,
        )
        payload = decision_service.enrich_payload(payload)
        save_recommendation(payload, output_dir)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("recommendation executed in %.2f ms", elapsed_ms)

        cache.set(key, payload)
        out = copy.deepcopy(payload)
        feedback_meta = closed_loop_recorder.record_inference(
            env_input=req.env.model_dump(),
            payload=out,
            user=current_user,
        )
        out["feedback"] = feedback_meta
        out.setdefault("runtime", {})["feedback_event_id"] = feedback_meta["event_id"]
        decision_service.append_history(out)
        out["cache"] = {"hit": False, "ttl_seconds": ttl_seconds}
        return out

    @app.post("/api/recommend-feedback")
    def submit_recommend_feedback(req: feedback_model, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        event_id = str(req.event_id or "").strip()
        if not event_id:
            raise HTTPException(status_code=400, detail="event_id 不能为空")
        return closed_loop_recorder.record_feedback(
            event_id=event_id,
            selected_crop=req.selected_crop,
            accepted=req.accepted,
            actual_profit=req.actual_profit,
            actual_price=req.actual_price,
            actual_yield=req.actual_yield,
            actual_cost=req.actual_cost,
            notes=req.notes,
            user=current_user,
        )
