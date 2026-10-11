"""
Verification tests for Milestone 1: Async Reverse Proxy Core & Zero-Copy SSE Streaming.
"""
import asyncio
import json
import time
import pytest
import httpx
from fastapi.testclient import TestClient

from src.main import app
from src.proxy.client import UpstreamClientManager
from src.proxy.streaming import sse_stream_transformer, StreamMetrics


@pytest.fixture(autouse=True)
async def reset_client():
    yield
    await UpstreamClientManager.close_client()


@pytest.mark.asyncio
async def test_health_check():
    """Verify healthcheck endpoint returns 200 OK."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy", "service": "llm-gateway"}


@pytest.mark.asyncio
async def test_sse_stream_transformer_metrics_and_accumulation():
    """
    Verify that sse_stream_transformer:
    1. Forwards raw bytes chunk-by-chunk without modifying stream order.
    2. Measures Time-To-First-Token (TTFT) accurately.
    3. Counts token deltas and accumulates the complete response text.
    """
    # Simulated upstream SSE chunks with artificial delays between tokens
    mock_chunks = [
        b'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    async def mock_byte_stream():
        for chunk in mock_chunks:
            await asyncio.sleep(0.01)  # 10ms network delay between chunks
            yield chunk

    # Mock an httpx.Response object
    mock_response = httpx.Response(
        status_code=200,
        content=mock_byte_stream(),
        headers={"content-type": "text/event-stream"},
    )

    metrics = StreamMetrics()
    yielded_chunks = []

    async for chunk in sse_stream_transformer(mock_response, metrics):
        yielded_chunks.append(chunk)

    # 1. Output verification
    assert len(yielded_chunks) == len(mock_chunks)
    assert b"".join(yielded_chunks) == b"".join(mock_chunks)

    # 2. Metrics verification
    assert metrics.time_to_first_token is not None
    assert metrics.time_to_first_token > 0.0
    assert metrics.token_count == 3  # "Hello", " world", "!"
    assert metrics.full_text == "Hello world!"
    assert metrics.finish_reason == "stop"
    assert metrics.completion_id == "chatcmpl-test"


@pytest.mark.asyncio
async def test_zero_copy_streaming_ttft_overhead(monkeypatch):
    """
    Profile the gateway overhead on Time-To-First-Token (TTFT).
    Gateway overhead must remain < 15ms.
    """
    upstream_sleep_time = 0.02  # 20ms upstream delay before first token

    # Create a mock upstream transport
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        async def delayed_chunks():
            await asyncio.sleep(upstream_sleep_time)
            yield b'data: {"id":"mock-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"First"},"finish_reason":null}]}\n\n'
            await asyncio.sleep(0.005)
            yield b'data: {"id":"mock-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Second"},"finish_reason":"stop"}]}\n\n'
            yield b'data: [DONE]\n\n'

        return httpx.Response(200, content=delayed_chunks(), headers={"content-type": "text/event-stream"})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    monkeypatch.setattr(UpstreamClientManager, "get_client", lambda: mock_client)

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Ping"}],
        "stream": True,
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Pre-warm route handler to eliminate Python 3.14 AST compilation / Starlette route resolution cold-start
        async with client.stream("POST", "/v1/chat/completions", json=payload) as warmup_resp:
            async for _ in warmup_resp.aiter_raw():
                break

        # Measure dispatch -> first chunk received
        start = time.perf_counter()
        first_chunk_received_at = None

        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            assert response.status_code == 200
            async for chunk in response.aiter_raw():
                if first_chunk_received_at is None:
                    first_chunk_received_at = time.perf_counter()
                    break

    elapsed_ms = (first_chunk_received_at - start) * 1000.0
    gateway_overhead_ms = elapsed_ms - (upstream_sleep_time * 1000.0)

    print(f"\n[TTFT Profile] Total: {elapsed_ms:.2f}ms, Gateway Overhead: {gateway_overhead_ms:.2f}ms")
    # Gateway overhead invariant: < 15ms
    assert gateway_overhead_ms < 15.0, f"Gateway overhead {gateway_overhead_ms:.2f}ms exceeded 15ms threshold"


@pytest.mark.asyncio
async def test_bounded_memory_streaming(monkeypatch):
    """
    Confirm that streaming a long payload (10,000 chunks) maintains constant,
    bounded RAM usage and yields chunks incrementally.
    """
    import tracemalloc

    total_tokens = 10000

    async def mock_large_generator():
        for i in range(total_tokens):
            chunk_json = {
                "id": "mock-large",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": f"tok_{i} "}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk_json)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    mock_resp = httpx.Response(200, content=mock_large_generator(), headers={"content-type": "text/event-stream"})
    metrics = StreamMetrics()

    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    chunks_streamed = 0
    async for chunk in sse_stream_transformer(mock_resp, metrics):
        chunks_streamed += 1
        # Consumer discards chunk immediately without in-memory buffering

    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Memory allocated for generator overhead must be strictly bounded (< 2MB for 10,000 chunks including metrics)
    top_stats = snapshot2.compare_to(snapshot1, "lineno")
    total_diff_kb = sum(stat.size_diff for stat in top_stats) / 1024

    print(f"\n[Memory Profile] Streamed {chunks_streamed} chunks. Heap delta: {total_diff_kb:.2f} KB")
    assert chunks_streamed == total_tokens + 1
    assert metrics.token_count == total_tokens
    assert total_diff_kb < 2048.0  # Under 2MB heap delta for 10k tokens

