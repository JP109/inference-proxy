"""
FastAPI application entrypoint and ASGI lifespan configuration.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.core.config import settings
from src.proxy.client import UpstreamClientManager
from src.proxy.router import router as proxy_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("gateway.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    ASGI Lifespan context manager.
    Handles startup connection initialization and graceful shutdown draining.
    """
    logger.info("Initializing LLM Gateway...")
    # Pre-warm connection pool
    UpstreamClientManager.get_client()
    logger.info(
        f"HTTP connection pool established (max={settings.http_max_connections}, "
        f"keepalive={settings.http_max_keepalive_connections})"
    )
    yield
    logger.info("Draining connections and shutting down LLM Gateway...")
    await UpstreamClientManager.close_client()
    logger.info("Gateway shutdown complete.")


app = FastAPI(
    title="High-Throughput LLM Gateway & Semantic Proxy",
    version="1.0.0",
    lifespan=lifespan,
)

# Healthcheck for load balancers / orchestrators
@app.get("/healthz")
async def health_check():
    return JSONResponse(content={"status": "healthy", "service": "llm-gateway"})


# Mount reverse proxy router
app.include_router(proxy_router)

