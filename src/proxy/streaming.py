"""
Zero-copy SSE stream transformer, metrics tracker, and response accumulator.
"""
import json
import time
import logging
from typing import AsyncGenerator, Callable, Optional, Awaitable, List
import httpx
from src.schemas.chat import ChatCompletionChunk

logger = logging.getLogger("gateway.streaming")


class StreamMetrics:
    """
    Latency and token metrics recorded during active streaming.
    """
    def __init__(self) -> None:
        self.start_time: float = time.perf_counter()
        self.time_to_first_token: Optional[float] = None
        self.total_stream_time: Optional[float] = None
        self.token_count: int = 0
        self.raw_chunk_count: int = 0
        self.accumulated_content: List[str] = []
        self.finish_reason: Optional[str] = None
        self.completion_id: Optional[str] = None
        self.model: Optional[str] = None

    def record_chunk(self, content_delta: Optional[str]) -> None:
        self.raw_chunk_count += 1
        if content_delta:
            if self.time_to_first_token is None:
                self.time_to_first_token = time.perf_counter() - self.start_time
            self.token_count += 1
            self.accumulated_content.append(content_delta)

    def finalize(self) -> None:
        self.total_stream_time = time.perf_counter() - self.start_time

    @property
    def full_text(self) -> str:
        return "".join(self.accumulated_content)


async def sse_stream_transformer(
    upstream_response: httpx.Response,
    metrics: StreamMetrics,
    on_complete: Optional[Callable[[StreamMetrics], Awaitable[None]]] = None,
) -> AsyncGenerator[bytes, None]:
    """
    Consumes SSE chunks from an upstream HTTP response chunk-by-chunk,
    yielding them directly downstream without buffering the full response.

    Architectural Invariants:
    1. Zero-Copy: Bytes received from upstream socket are forwarded downstream immediately,
       ensuring minimal TTFT and O(1) memory overhead with respect to total tokens.
    2. Concurrent Accumulation (Tee Pattern): Chunks are parsed on the fly to count tokens
       and accumulate the full response for asynchronous caching upon completion.
    3. Graceful Termination & Cleanup: Closes upstream stream cleanly if downstream disconnects.
    """
    buffer = ""
    try:
        async for raw_bytes in upstream_response.aiter_bytes():
            # 1. Immediately yield raw bytes downstream to keep TTFT minimal
            yield raw_bytes

            # 2. Parse SSE protocol lines in parallel for metrics and caching
            buffer += raw_bytes.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line or not line.startswith("data:"):
                    continue

                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    continue

                try:
                    data = json.loads(payload)
                    chunk = ChatCompletionChunk.model_validate(data)
                    if chunk.id and not metrics.completion_id:
                        metrics.completion_id = chunk.id
                    if chunk.model and not metrics.model:
                        metrics.model = chunk.model

                    for choice in chunk.choices:
                        if choice.delta and choice.delta.content:
                            metrics.record_chunk(choice.delta.content)
                        if choice.finish_reason:
                            metrics.finish_reason = choice.finish_reason
                except Exception:
                    # Non-fatal: if upstream chunk is malformed or non-standard, continue streaming
                    pass

        metrics.finalize()
        logger.info(
            f"Stream complete: tokens={metrics.token_count}, "
            f"ttft={metrics.time_to_first_token:.4f}s if metrics.time_to_first_token else 0s, "
            f"total_time={metrics.total_stream_time:.4f}s"
        )

        # 3. Asynchronous hook (e.g., write accumulated response into cache tiers)
        if on_complete:
            try:
                await on_complete(metrics)
            except Exception as e:
                logger.error(f"Error executing on_complete callback: {e}", exc_info=True)

    finally:
        await upstream_response.aclose()
