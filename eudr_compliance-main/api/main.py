"""
FastAPI application entry point.

Startup
-------
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Or via the project script:
    eudr-api
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from eudr.config import settings
from eudr.pipeline import EUDRComplianceSystem

# ── Logging configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("eudr.api")

# ── Singleton model instance ──────────────────────────────────────────────────
_system: "Optional[EUDRComplianceSystem]" = None


def get_system() -> EUDRComplianceSystem:
    """Return the lazily initialised EUDRComplianceSystem singleton."""
    global _system
    if _system is None:
        _system = EUDRComplianceSystem(lazy_load=True)
    return _system


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """
    Pre-warm the system on startup so the first request does not pay the
    full model-loading latency.
    """
    logger.info("Pre-warming EUDR Compliance System …")
    try:
        sys_instance = get_system()
        # Trigger lazy model loading
        sys_instance._ensure_loaded()
        logger.info("System ready.")
    except Exception as exc:
        logger.warning("Pre-warm failed (lazy loading will be used instead): %s", exc)
    yield
    logger.info("Shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Osapiens EUDR Compliance Intelligence API",
    description=(
        "Automated EUDR (EU Deforestation Regulation) compliance verification using "
        "multimodal satellite analysis powered by Prithvi-100M and Gemma."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

from api.routes import router  # noqa: E402  (import after app creation avoids circular refs)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# ── Static UI files ───────────────────────────────────────────────────────────
_STATIC_DIR = Path(__file__).parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_ui() -> FileResponse:
        return FileResponse(str(_STATIC_DIR / "index.html"))


# ── CLI entry point ───────────────────────────────────────────────────────────
def run() -> None:
    import os

    # Ensure CWD is the project root so that `api.main` is importable by uvicorn
    project_root = Path(__file__).parent.parent.resolve()
    os.chdir(project_root)

    host = settings.api_host  # "0.0.0.0" (binds all interfaces)
    port = settings.api_port

    # Always print the browser URL using localhost — never 0.0.0.0
    print()
    print("=" * 60)
    print("  Osapiens EUDR Compliance Intelligence")
    print("=" * 60)
    print(f"  Officer UI  →  http://localhost:{port}")
    print(f"  Swagger     →  http://localhost:{port}/docs")
    print(f"  API         →  http://localhost:{port}/api/v1/verify")
    print("=" * 60)
    print()

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
