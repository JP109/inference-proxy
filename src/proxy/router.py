"""
Reverse proxy routing for OpenAI-compatible chat completion requests.
"""
import logging
from typing import Optional
from fastapi import APIRouter, Request, Response, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

from src.core.config import settings
from src.schemas.chat import ChatCompletionRequest
from src.proxy.client import UpstreamClientManager
from src.proxy.streaming import sse_stream_transformer, StreamMetrics

logger = logging.getLogger("gateway.router")

router = APIRouter(prefix="/v1", tags=["Inference Gateway"])


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    payload: ChatCompletionRequest,
) -> Response:
    """
    OpenAI-compatible inference endpoint supporting zero-copy SSE streaming.
    """
    client = UpstreamClientManager.get_client()

    # Prepare upstream URL and headers
    upstream_url = f"{settings.upstream_base_url.rstrip('/')}/chat/completions"
    
    # Forward client auth token if present, otherwise fallback to configured upstream key
    auth_header = request.headers.get("Authorization")
    if not auth_header and settings.upstream_api_key:
        auth_header = f"Bearer {settings.upstream_api_key}"

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if payload.stream else "application/json",
    }
    if auth_header:
        headers["Authorization"] = auth_header

    # High-performance zero-copy forwarding:
    # Use raw body bytes directly to avoid double Pydantic dump & JSON re-serialization overhead
    raw_body = await request.body()

    try:
        if payload.stream:
            # Build and dispatch streaming request with raw body content
            upstream_req = client.build_request(
                method="POST",
                url=upstream_url,
                content=raw_body,
                headers=headers,
            )
            upstream_resp = await client.send(upstream_req, stream=True)

            if upstream_resp.status_code != 200:
                # Upstream error (e.g. 401 Unauthorized, 400 Bad Request, 500)
                error_bytes = await upstream_resp.aread()
                await upstream_resp.aclose()
                return Response(
                    content=error_bytes,
                    status_code=upstream_resp.status_code,
                    media_type="application/json",
                )

            metrics = StreamMetrics()
            return StreamingResponse(
                content=sse_stream_transformer(upstream_resp, metrics),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable proxy buffering in Nginx if upstream
                },
            )

        else:
            # Non-streaming request
            resp = await client.post(
                url=upstream_url,
                content=raw_body,
                headers=headers,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )

    except httpx.TimeoutException as exc:
        logger.error(f"Upstream timeout: {exc}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Upstream provider timed out.",
        )
    except httpx.RequestError as exc:
        logger.error(f"Upstream connection failure: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to communicate with upstream provider: {str(exc)}",
        )

