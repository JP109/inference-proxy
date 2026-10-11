"""
Persistent HTTP Client manager with connection pooling for upstream LLM providers.
"""
from typing import Optional
import httpx
from src.core.config import settings


class UpstreamClientManager:
    _client: Optional[httpx.AsyncClient] = None

    @classmethod
    def get_client(cls) -> httpx.AsyncClient:
        if cls._client is None or cls._client.is_closed:
            limits = httpx.Limits(
                max_connections=settings.http_max_connections,
                max_keepalive_connections=settings.http_max_keepalive_connections,
                keepalive_expiry=settings.http_keepalive_expiry_seconds,
            )
            timeout = httpx.Timeout(
                connect=settings.http_connect_timeout_seconds,
                read=settings.http_read_timeout_seconds,
                write=10.0,
                pool=5.0,
            )
            cls._client = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
                http2=False,
            )
        return cls._client

    @classmethod
    async def close_client(cls) -> None:
        if cls._client is not None and not cls._client.is_closed:
            await cls._client.aclose()
            cls._client = None
