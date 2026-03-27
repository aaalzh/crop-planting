from typing import Any, Callable, Dict

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import RedirectResponse


def register_auth_routes(
    app: FastAPI,
    *,
    get_current_user: Callable[..., Dict[str, Any]],
    get_admin_user: Callable[..., Dict[str, Any]],
    user_store,
    session_store,
    session_ttl_seconds: int,
    auth_cookie_name: str,
    login_page_url: str,
    set_auth_cookie: Callable[[Response, str, int], None],
    clear_auth_cookie: Callable[[Response], None],
    public_user: Callable[[Dict[str, Any]], Dict[str, Any]],
    normalize_enabled: Callable[[Any], bool],
    register_model,
    login_model,
    admin_update_model,
) -> None:
    @app.post("/api/auth/register")
    def register(req: register_model, response: Response) -> Dict[str, Any]:
        try:
            user = user_store.create_user(req.username, req.password, req.display_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        saved = user_store.touch_login(user["username"]) or user
        token = session_store.issue(saved["username"])
        set_auth_cookie(response, token, session_ttl_seconds)
        return {"message": "注册成功", "user": public_user(saved)}

    @app.post("/api/auth/login")
    def login(req: login_model, response: Response) -> Dict[str, Any]:
        current = user_store.get_user(req.username)
        if current and not normalize_enabled(current.get("enabled", True)):
            raise HTTPException(status_code=403, detail="账号已被禁用，请联系管理员")
        user = user_store.verify_user(req.username, req.password)
        if not user:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        saved = user_store.touch_login(user["username"]) or user
        token = session_store.issue(saved["username"])
        set_auth_cookie(response, token, session_ttl_seconds)
        return {"message": "登录成功", "user": public_user(saved)}

    @app.post("/api/auth/logout")
    def logout(request: Request, response: Response) -> Dict[str, str]:
        token = request.cookies.get(auth_cookie_name)
        session_store.revoke(token)
        clear_auth_cookie(response)
        return {"message": "已退出登录"}

    @app.get("/logout")
    def logout_page(request: Request) -> RedirectResponse:
        token = request.cookies.get(auth_cookie_name)
        session_store.revoke(token)
        response = RedirectResponse(url=login_page_url, status_code=303)
        clear_auth_cookie(response)
        return response

    @app.get("/api/auth/me")
    def me(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        return {"user": public_user(current_user)}

    @app.get("/api/admin/users")
    def admin_list_users(_: Dict[str, Any] = Depends(get_admin_user)) -> Dict[str, Any]:
        users = [public_user(user) for user in user_store.list_users()]
        stats = {
            "total": len(users),
            "enabled": sum(1 for u in users if u.get("enabled")),
            "disabled": sum(1 for u in users if not u.get("enabled")),
            "admins": sum(1 for u in users if u.get("role") == "admin"),
            "normal_users": sum(1 for u in users if u.get("role") == "user"),
        }
        return {"users": users, "stats": stats}

    @app.patch("/api/admin/users/{username}")
    def admin_update_user(
        username: str,
        req: admin_update_model,
        admin_user: Dict[str, Any] = Depends(get_admin_user),
    ) -> Dict[str, Any]:
        if req.role is None and req.enabled is None and req.display_name is None:
            raise HTTPException(status_code=400, detail="请至少提交一个可更新字段")
        try:
            updated = user_store.update_user(
                username=username,
                role=req.role,
                enabled=req.enabled,
                display_name=req.display_name,
                actor_username=admin_user.get("username"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"message": "更新成功", "user": public_user(updated)}
