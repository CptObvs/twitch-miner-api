"""
Authenticated reverse proxy for Docker container web UIs.

Routes:
  GET/POST/... /instances/{id}/ui           -> http://127.0.0.1:{port}/
  GET/POST/... /instances/{id}/ui/{path}    -> http://127.0.0.1:{port}/{path}
  WS           /instances/{id}/ui/{path}    -> ws://127.0.0.1:{port}/{path}?token=<jwt>

Auth:
  HTTP  — standard JWT Bearer header
  WS    — token passed as query parameter ?token=<jwt>
         (browsers cannot set custom headers on WebSocket connections)

Admin users can access any instance's UI.
Regular users can only access their own instances.
"""

import logging
import re
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import MinerInstance, User, get_db
from app.models.enums import InstanceState
from app.services.auth import get_current_user, verify_token

router = APIRouter(prefix="/instances", tags=["proxy"])
logger = logging.getLogger("uvicorn.error")

# Shared httpx client (connection-pooled, reused across requests)
_http_client: httpx.AsyncClient | None = None

# Headers that must not be forwarded (hop-by-hop)
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host",
}


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
        )
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()


async def _resolve_port(
    instance_id: str,
    current_user: User,
    db: AsyncSession,
) -> int:
    """Verify ownership and return the container's host port."""
    if current_user.is_admin():
        result = await db.execute(
            select(MinerInstance).where(MinerInstance.id == instance_id)
        )
    else:
        result = await db.execute(
            select(MinerInstance).where(
                MinerInstance.id == instance_id,
                MinerInstance.user_id == current_user.id,
            )
        )
    instance = result.scalar_one_or_none()
    if not instance:
        raise HTTPException(404, "Instance not found")
    if instance.status != InstanceState.RUNNING or instance.port is None:
        raise HTTPException(503, "Instance is not running")
    return instance.port


# ------------------------------------------------------------------
# HTTP proxy
# ------------------------------------------------------------------



async def proxy_http(
    instance_id: str,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    path: str = "",
):
    """Proxy HTTP requests to the Docker container's web UI."""
    # Auth priority: Authorization header > ?token= query param > session cookie
    # The cookie is set on the first ?token= request so subsequent asset requests
    # (CSS/JS/images) are authenticated without needing the token in every URL.
    cookie_name = f"proxy_token_{instance_id}"
    token_from_url = False

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        jwt_token = auth_header[7:]
    else:
        jwt_token = request.query_params.get("token", "")
        if jwt_token:
            token_from_url = True
        else:
            jwt_token = request.cookies.get(cookie_name, "")

    if not jwt_token:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)

    try:
        current_user = await verify_token(jwt_token, db)
    except Exception:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)

    port = await _resolve_port(instance_id, current_user, db)

    # Forward query string but strip our ?token= param
    qs_params = {k: v for k, v in request.query_params.items() if k != "token"}
    qs = "&".join(f"{k}={v}" for k, v in qs_params.items())
    target_url = f"http://127.0.0.1:{port}/{path}"
    if qs:
        target_url += f"?{qs}"

    forward_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP
    }
    forward_headers["host"] = f"127.0.0.1:{port}"

    client = _get_http_client()
    body = await request.body()

    # Base path for this instance's UI (used to rewrite root-absolute links in HTML)
    ui_base = f"/api/instances/{instance_id}/ui"

    try:
        async with client.stream(
            method=request.method,
            url=target_url,
            headers=forward_headers,
            content=body,
        ) as upstream:
            response_headers = {
                k: v for k, v in upstream.headers.items()
                if k.lower() not in HOP_BY_HOP and k.lower() != "content-length"
            }
            content = await upstream.aread()
            content_type = upstream.headers.get("content-type", "")

            # Rewrite root-absolute asset paths in HTML responses so the browser
            # fetches them through the proxy instead of the server root.
            if "text/html" in content_type:
                html = content.decode("utf-8", errors="replace")
                # Rewrite root-absolute paths in HTML attributes
                html = re.sub(r'(href|src|action)="/', rf'\1="{ui_base}/', html)
                html = re.sub(r"(href|src|action)='/", rf"\1='{ui_base}/", html)
                html = re.sub(r'url\(/', rf'url({ui_base}/', html)

                # Inject a JS shim before </head> that rewrites root-relative
                # paths in fetch(), XHR, and WebSocket so the container's JS calls
                # are routed through the proxy instead of hitting our backend.
                shim = f"""<script>
(function(){{
  var base='{ui_base}';
  function rw(u){{
    if(typeof u==='string'&&u.startsWith('/')&&!u.startsWith(base))return base+u;
    return u;
  }}
  // Rewrite HTTP fetch
  var _f=window.fetch;
  window.fetch=function(u,o){{return _f(rw(u),o);}};
  // Rewrite XHR
  var _o=XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open=function(m,u){{
    arguments[1]=rw(u);return _o.apply(this,arguments);
  }};
  // Rewrite WebSocket (Socket.IO uses absolute wss:// URLs from same host)
  var _WS=window.WebSocket;
  function PatchedWS(url,protos){{
    if(typeof url==='string'){{
      try{{
        var u=new URL(url);
        if(u.hostname===location.hostname&&!u.pathname.startsWith(base)){{
          u.pathname=base+u.pathname;url=u.toString();
        }}
      }}catch(e){{}}
    }}
    return protos!==undefined?new _WS(url,protos):new _WS(url);
  }}
  PatchedWS.prototype=_WS.prototype;
  ['CONNECTING','OPEN','CLOSING','CLOSED'].forEach(function(k){{PatchedWS[k]=_WS[k];}});
  window.WebSocket=PatchedWS;
}})();
</script>"""
                # Inject at the START of <head> so the shim runs before any
                # scripts (e.g. socket.io) can cache the original WebSocket ref.
                html = re.sub(r'<head([^>]*)>', lambda m: m.group(0) + shim, html, count=1)
                if '<head' not in html:
                    html = shim + html
                content = html.encode("utf-8")
                response_headers["content-type"] = "text/html; charset=utf-8"

            response = Response(
                content=content,
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=upstream.headers.get("content-type"),
            )

            # Persist token as session cookie so asset requests don't need ?token=
            if token_from_url:
                response.set_cookie(
                    key=cookie_name,
                    value=jwt_token,
                    httponly=True,
                    secure=True,
                    samesite="lax",
                    path=f"/api/instances/{instance_id}/ui",
                    max_age=3600,
                )

            return response
    except httpx.ConnectError:
        raise HTTPException(502, "Container not reachable")
    except httpx.TimeoutException:
        raise HTTPException(504, "Container timed out")

# Register each HTTP method with a unique operation_id to avoid OpenAPI warnings
for method in ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]:
    router.add_api_route(
        "/{instance_id}/ui",
        proxy_http,
        methods=[method],
        operation_id=f"proxy_http_{method.lower()}_root",
    )
    router.add_api_route(
        "/{instance_id}/ui/{path:path}",
        proxy_http,
        methods=[method],
        operation_id=f"proxy_http_{method.lower()}",
    )


# ------------------------------------------------------------------
# WebSocket proxy
# ------------------------------------------------------------------

@router.websocket("/{instance_id}/ui/{path:path}")
async def proxy_websocket(
    instance_id: str,
    websocket: WebSocket,
    path: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Proxy WebSocket connections (Socket.IO) to the Docker container.

    Authentication: pass JWT as query param ?token=<jwt>
    The frontend must include this when opening the WebSocket connection,
    since browsers cannot set Authorization headers for WebSocket.
    """
    import asyncio
    import websockets

    # Auth: ?token= query param or session cookie (set on first HTTP ?token= request)
    token = websocket.query_params.get("token") or websocket.cookies.get(f"proxy_token_{instance_id}")
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return

    try:
        current_user = await verify_token(token, db)
    except HTTPException:
        await websocket.close(code=4003, reason="Unauthorized")
        return

    try:
        port = await _resolve_port(instance_id, current_user, db)
    except HTTPException as e:
        await websocket.close(code=4004, reason=str(e.detail))
        return

    # Build upstream WS URL, forwarding all query params except our token
    upstream_qs = "&".join(
        f"{k}={v}"
        for k, v in websocket.query_params.items()
        if k != "token"
    )
    upstream_url = f"ws://127.0.0.1:{port}/{path}"
    if upstream_qs:
        upstream_url += f"?{upstream_qs}"

    await websocket.accept()

    try:
        # python-engineio (Socket.IO) validates that Origin matches Host.
        # Pass the container's own address as Origin so the check passes.
        async with websockets.connect(
            upstream_url,
            additional_headers={"Origin": f"http://127.0.0.1:{port}"},
            compression=None,  # disable permessage-deflate to avoid negotiation issues
        ) as upstream_ws:

            async def forward_to_upstream():
                # iter_bytes() only gets binary frames; Socket.IO uses TEXT frames.
                # Use receive() directly to handle both text and binary messages.
                try:
                    while True:
                        msg = await websocket.receive()
                        if msg.get("type") == "websocket.disconnect":
                            break
                        if msg.get("bytes") is not None:
                            await upstream_ws.send(msg["bytes"])
                        elif msg.get("text") is not None:
                            await upstream_ws.send(msg["text"])
                except WebSocketDisconnect:
                    pass
                except Exception:
                    pass

            async def forward_to_client():
                try:
                    async for message in upstream_ws:
                        if isinstance(message, bytes):
                            await websocket.send_bytes(message)
                        else:
                            await websocket.send_text(message)
                except Exception:
                    pass

            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(forward_to_upstream()),
                    asyncio.create_task(forward_to_client()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        logger.warning("WebSocket proxy error for instance %s: %s", instance_id, e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
