"""
Twitch Miner Backend API
========================
FastAPI application for managing multiple TwitchDropsMiner Docker instances.
"""

import asyncio
import gc
import io
import logging
import sys
from time import perf_counter
from contextlib import asynccontextmanager

from alembic.config import Config as AlembicConfig
from alembic import command as alembic_command

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from jose import JWTError, jwt

import socketio as _socketio

from app.core.config import settings
from app.routers import admin, auth, codes, instances, proxy
from app.services.miner_manager import miner_manager
from app.routers.proxy import close_http_client
from app.services.socket_manager import sio

logger = logging.getLogger("uvicorn.error")
request_logger = logging.getLogger("uvicorn.error")


if sys.platform == "win32" and "pytest" not in sys.modules:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


async def orphan_container_cleanup() -> None:
    while True:
        await asyncio.sleep(300)
        await miner_manager.reconcile_all_on_startup()


async def memory_gc_cleanup() -> None:
    while True:
        await asyncio.sleep(settings.MEMORY_GC_INTERVAL_SECONDS)
        collected = gc.collect(settings.MEMORY_GC_GENERATION)
        logger.debug(f"Periodic GC run complete, collected objects: {collected}")


def start_background_tasks(app: FastAPI) -> None:
    app.state.cleanup_task = asyncio.create_task(orphan_container_cleanup())
    app.state.memory_gc_task = (
        asyncio.create_task(memory_gc_cleanup()) if settings.MEMORY_GC_ENABLED else None
    )


async def stop_background_tasks(app: FastAPI) -> None:
    tasks = []
    for attr in ("cleanup_task", "memory_gc_task"):
        task = getattr(app.state, attr, None)
        if task:
            task.cancel()
            tasks.append(task)

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _run_migrations() -> None:
    alembic_cfg = AlembicConfig("alembic.ini")
    alembic_command.upgrade(alembic_cfg, "head")


async def run_startup(app: FastAPI) -> None:
    logger.info("Starting API ...")
    if settings.RUN_MIGRATIONS_ON_STARTUP:
        logger.info("Running database migrations ...")
        await asyncio.to_thread(_run_migrations)
        logger.info("Database migrations complete")
    await miner_manager.reconcile_all_on_startup()
    start_background_tasks(app)
    logger.info("API startup complete")


async def run_shutdown(app: FastAPI) -> None:
    logger.info("Stopping API ...")
    await stop_background_tasks(app)
    await miner_manager.shutdown_all()
    await close_http_client()
    logger.info("API stopped cleanly")


def _extract_request_identity(request: Request) -> tuple[str | None, str | None]:
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return None, None

    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        return None, None

    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        return payload.get("sub"), payload.get("username")
    except JWTError:
        return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_startup(app)
    try:
        yield
    finally:
        await run_shutdown(app)


app = FastAPI(
    title="Twitch Miner Backend",
    description="Manage TwitchDropsMiner Docker instances via REST API",
    version="2.0.0",
    root_path="/api",
    docs_url=settings.DOCS_URL if settings.ENABLE_SWAGGER else None,
    redoc_url=settings.REDOC_URL if settings.ENABLE_SWAGGER else None,
    openapi_url="/openapi.json" if settings.ENABLE_SWAGGER else None,
    swagger_ui_parameters={
        "persistAuthorization": True,
        "syntaxHighlight.theme": "monokai",
        "displayRequestDuration": True,
        "filter": True,
        "tryItOutEnabled": True,
    },
    lifespan=lifespan,
)


app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    if not settings.API_REQUEST_LOGGING_ENABLED:
        return await call_next(request)

    start = perf_counter()
    client_ip = request.client.host if request.client else "-"
    user_id, username = _extract_request_identity(request)
    actor = username or user_id or "anonymous"
    method = request.method
    path = request.url.path

    try:
        response = await call_next(request)
        duration_ms = (perf_counter() - start) * 1000
        request_logger.info(
            "API %s %s abgeschlossen: Status %s in %.2f ms (Nutzer: %s, IP: %s)",
            method,
            path,
            response.status_code,
            duration_ms,
            actor,
            client_ip,
        )
        return response
    except Exception:
        duration_ms = (perf_counter() - start) * 1000
        request_logger.exception(
            "API %s %s fehlgeschlagen: Status 500 nach %.2f ms (Nutzer: %s, IP: %s)",
            method,
            path,
            duration_ms,
            actor,
            client_ip,
        )
        raise

app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(codes.router)
app.include_router(instances.router)
app.include_router(proxy.router)

# Socket.IO — mounted at /socket.io (external: /api/socket.io via nginx)
# socketio_path='' because FastAPI strips the mount prefix before calling the ASGI app
app.mount("/socket.io", _socketio.ASGIApp(sio, socketio_path=""))


@app.get("/", tags=["health"])
async def root():
    return {
        "status": "ok",
        "service": "Twitch Miner Backend",
        "version": "2.0.0",
        "docs": settings.DOCS_URL if settings.ENABLE_SWAGGER else "disabled",
    }


@app.get("/health", tags=["health"])
async def health():
    return {"status": "healthy"}
