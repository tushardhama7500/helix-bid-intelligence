import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.limits import BodySizeLimitMiddleware
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.database import dispose_engine, init_db

settings = get_settings()
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("helix")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting Helix (env=%s)", settings.app_env)
    try:
        await init_db()
    except Exception:
        # Stay up so /health can report the problem instead of crash-looping.
        logger.exception("Database initialisation failed")
    yield
    await dispose_engine()


app = FastAPI(title="Helix", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(BodySizeLimitMiddleware)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: a structured JSON body, never a stack trace or internal detail."""
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, type(exc).__name__, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(api_router)
