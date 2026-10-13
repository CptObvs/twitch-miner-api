"""Regression coverage for the v2 dashboard, browser login and instance isolation."""

import asyncio
import gzip
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
import websockets
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.models.database import MinerInstance, get_db
from app.models.enums import InstanceState
from app.routers import proxy
from app.core.config import Settings, settings


@pytest_asyncio.fixture
async def proxy_setup(db_session, normal_user, user_token, monkeypatch):
    db_session.add_all([
        MinerInstance(id="one", user_id=normal_user.id, status=InstanceState.RUNNING, port=5000),
        MinerInstance(id="two", user_id=normal_user.id, status=InstanceState.RUNNING, port=5001),
    ])
    await db_session.commit()
    requests = []
    handlers = {}

    def handle(request):
        requests.append(request)
        return handlers.get("request", lambda r: httpx.Response(200, json={"ok": True}))(request)

    app = FastAPI()
    app.include_router(proxy.router, prefix="/api")

    async def get_test_db():
        yield db_session

    app.dependency_overrides[get_db] = get_test_db
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as upstream:
        monkeypatch.setattr(proxy, "_get_http_client", lambda: upstream)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://public.test") as browser:
            yield browser, user_token, requests, handlers


async def test_browser_login_post_cookie_and_followup(proxy_setup):
    browser, token, requests, handlers = proxy_setup
    handlers["request"] = lambda r: httpx.Response(200, headers={
        "set-cookie": "tdm_session=dashboard-session; HttpOnly; Path=/; SameSite=Strict",
    }, json={"success": True})
    response = await browser.post(
        f"/api/instances/one/ui/api/session/finish?token={token}",
        headers={"Origin": "https://public.test", "X-TDM-Request": "1"},
    )
    assert response.status_code == 200
    sent = requests[-1]
    assert sent.headers["origin"] == "http://127.0.0.1:5000"
    assert sent.headers["host"] == "127.0.0.1:5000"
    assert sent.headers["x-tdm-request"] == "1"
    assert token not in str(sent.url) + str(sent.headers)
    cookies = response.headers.get_list("set-cookie")
    assert len(cookies) == 2
    assert all("Path=/api/instances/one/ui" in value for value in cookies)
    assert all("Secure" in value for value in cookies)
    assert (await browser.get("/api/instances/one/ui/api/session")).status_code == 200
    assert requests[-1].headers["cookie"] == "tdm_session=dashboard-session"


async def test_sessions_are_isolated_across_instances_and_shared_http_client(proxy_setup):
    browser, token, requests, handlers = proxy_setup
    handlers["request"] = lambda r: httpx.Response(200, headers={
        "set-cookie": f"tdm_session=session-{r.url.port}; Path=/; HttpOnly",
    })
    await browser.get(f"/api/instances/one/ui?token={token}")
    await browser.get(f"/api/instances/two/ui?token={token}")
    # An httpx cookie from the first container must not reach the second.
    assert requests[-1].headers["cookie"] == ""
    await browser.get("/api/instances/one/ui/api/session")
    assert requests[-1].headers["cookie"] == "tdm_session=session-5000"
    await browser.get("/api/instances/two/ui/api/session")
    assert requests[-1].headers["cookie"] == "tdm_session=session-5001"


async def test_invalid_origin_is_rejected_before_forwarding(proxy_setup):
    browser, token, requests, _ = proxy_setup
    response = await browser.post(
        f"/api/instances/one/ui/api/session/finish?token={token}",
        headers={"Origin": "https://attacker.test", "X-TDM-Request": "1"},
    )
    assert response.status_code == 403
    assert not requests


async def test_https_termination_and_backend_credentials(proxy_setup):
    browser, token, requests, _ = proxy_setup
    await browser.post("/api/instances/one/ui/api/session/finish", headers={
        "Authorization": f"Bearer {token}", "Origin": "https://public.test",
        "X-Forwarded-Proto": "https", "X-TDM-Request": "1",
        "Cookie": "frontend_secret=keep-private; tdm_session=miner-session",
    })
    sent = requests[-1].headers
    assert "authorization" not in sent and "x-forwarded-proto" not in sent
    assert sent["cookie"] == "tdm_session=miner-session"


async def test_public_https_origin_behind_http_gateway(proxy_setup, monkeypatch):
    browser, token, requests, _ = proxy_setup
    monkeypatch.setattr(settings, "UI_PUBLIC_ORIGIN", "https://public.test")
    response = await browser.post(
        f"http://public.test/api/instances/one/ui/api/session/finish?token={token}",
        headers={"Origin": "https://public.test", "X-TDM-Request": "1"},
    )
    assert response.status_code == 200
    assert requests[-1].headers["origin"] == "http://127.0.0.1:5000"
    assert "Secure" in response.headers["set-cookie"]
    response = await browser.post(
        f"http://public.test/api/instances/one/ui/api/session/finish?token={token}",
        headers={"Origin": "https://attacker.test", "X-TDM-Request": "1"},
    )
    assert response.status_code == 403


@pytest.mark.parametrize("value", ["https://user:pass@example.com", "https://example.com/api", "https://example.com?token=secret", "https://example.com#fragment", "ftp://example.com", "https://example.com/ "])
def test_public_origin_rejects_invalid_configuration(value):
    with pytest.raises(ValueError):
        Settings(_env_file=None, UI_PUBLIC_ORIGIN=value)


def test_public_origin_normalizes_default_port_and_trailing_slash():
    assert Settings(_env_file=None, UI_PUBLIC_ORIGIN="https://EXAMPLE.com:443/").UI_PUBLIC_ORIGIN == "https://example.com"


async def test_query_roundtrip_and_redirects(proxy_setup):
    browser, token, requests, handlers = proxy_setup
    handlers["request"] = lambda r: httpx.Response(303, headers={"location": "/login?next=%2F"})
    response = await browser.get("/api/instances/one/ui", params=[
        ("token", token), ("value", "a&b+c/"), ("value", "other value"),
    ])
    assert requests[-1].url.params.get_list("value") == ["a&b+c/", "other value"]
    assert "token" not in requests[-1].url.params
    assert response.headers["location"] == "/api/instances/one/ui/login?next=%2F"
    assert len(response.headers.get_list("set-cookie")) == 1


async def test_html_is_decoded_and_shim_runs_before_dashboard_scripts(proxy_setup):
    browser, token, _, handlers = proxy_setup
    html = '<html><head><script src="/static/auth.js"></script></head><body><a href="//cdn.test/a">CDN</a></body></html>'
    handlers["request"] = lambda r: httpx.Response(200, content=gzip.compress(html.encode()), headers={
        "content-type": "text/html", "content-encoding": "gzip", "etag": "old",
    })
    response = await browser.get(f"/api/instances/one/ui?token={token}")
    assert response.text.index("window.__tdmProxyURL") < response.text.index('/static/auth.js')
    assert 'src="/api/instances/one/ui/static/auth.js"' in response.text
    assert 'href="//cdn.test/a"' in response.text
    assert "content-encoding" not in response.headers and "etag" not in response.headers


async def test_dynamic_novnc_import_and_auth_redirects(proxy_setup):
    browser, token, _, handlers = proxy_setup
    source = "const RFB = (await import('/static/novnc/core/rfb.js')).default;\nwindow.location.replace(action === 'logout' ? '/login' : '/');"
    handlers["request"] = lambda r: httpx.Response(200, text=source, headers={"content-type": "text/javascript"})
    response = await browser.get(f"/api/instances/one/ui/static/browser-login.js?token={token}")
    assert "import('/api/instances/one/ui/static/novnc/core/rfb.js')" in response.text
    assert "window.location.replace(window.__tdmProxyURL(action === 'logout' ? '/login' : '/'));" in response.text


async def test_proxy_still_checks_instance_ownership(proxy_setup, admin_user):
    browser, token, requests, _ = proxy_setup
    # A valid JWT for another account cannot access these instances.
    from app.services.auth import create_access_token
    from app.models.enums import UserRole
    admin_user.role = UserRole.USER
    response = await browser.get(f"/api/instances/one/ui?token={create_access_token(admin_user.id, admin_user.username)}")
    assert response.status_code == 404
    assert not requests


@pytest.mark.parametrize("tls_gateway", [False, True])
async def test_vnc_websocket_relays_cookies_subprotocol_and_binary_frames(monkeypatch, tls_gateway):
    if tls_gateway:
        monkeypatch.setattr(settings, "UI_PUBLIC_ORIGIN", "https://public.test")
    app = FastAPI()
    app.include_router(proxy.router, prefix="/api")

    async def no_database():
        yield None

    app.dependency_overrides[get_db] = no_database
    monkeypatch.setattr(proxy, "verify_token", AsyncMock(return_value=object()))
    requests = []

    async def echo(connection):
        requests.append(connection.request)
        await connection.send(b"RFB 003.008\n")
        async for message in connection:
            await connection.send(message)

    async with websockets.serve(echo, "127.0.0.1", 0, subprotocols=["binary"]) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setattr(proxy, "_resolve_port", AsyncMock(return_value=port))

        def browser_session():
            with TestClient(app) as browser:
                with browser.websocket_connect(
                    ("ws" if tls_gateway else "wss") + "://public.test/api/instances/one/ui/api/session/vnc?value=a%26b&value=c",
                    headers={"Origin": "https://public.test", "Cookie": "proxy_token_one=backend-jwt; tdm_session=miner-session"},
                    subprotocols=["binary"],
                ) as ws:
                    assert ws.accepted_subprotocol == "binary"
                    assert ws.receive_bytes() == b"RFB 003.008\n"
                    ws.send_bytes(b"binary-vnc-frame")
                    assert ws.receive_bytes() == b"binary-vnc-frame"
                    ws.send_text("socket-io-text")
                    assert ws.receive_text() == "socket-io-text"

        await asyncio.wait_for(asyncio.to_thread(browser_session), timeout=10)
    assert requests[0].headers["origin"] == f"http://127.0.0.1:{port}"
    assert requests[0].headers["cookie"] == "tdm_session=miner-session"
    assert requests[0].path.endswith("?value=a%26b&value=c")
    assert "backend-jwt" not in str(requests[0].headers)


def test_websocket_rejects_untrusted_origin(monkeypatch):
    app = FastAPI()
    app.include_router(proxy.router)

    async def no_database():
        yield None

    app.dependency_overrides[get_db] = no_database
    monkeypatch.setattr(proxy, "verify_token", AsyncMock(return_value=object()))
    monkeypatch.setattr(proxy, "_resolve_port", AsyncMock(return_value=5000))
    with TestClient(app) as browser:
        with pytest.raises(WebSocketDisconnect) as error:
            with browser.websocket_connect("/instances/one/ui/api/session/vnc?token=jwt", headers={"Origin": "https://attacker.test"}):
                pass
    assert error.value.code == 4003
