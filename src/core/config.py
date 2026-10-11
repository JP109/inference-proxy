"""
Gateway configuration and runtime settings.
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    """
    Settings for the LLM Gateway and reverse proxy.
    
    Architectural Consideration:
    - max_keepalive_connections & max_connections: Keep-alive reuse is critical for
      eliminating TLS handshake latency on repeated upstream calls.
    - connect_timeout & read_timeout: LLM token generation has high variance; read timeout
      must accommodate long generation cycles or slow first tokens, while connect timeout
      should be aggressive to fail fast if upstream is unreachable.
    """
    model_config = SettingsConfigDict(env_prefix="GATEWAY_", env_file=".env", extra="ignore")

    # Upstream Provider Configuration
    upstream_base_url: str = "https://api.openai.com/v1"
    upstream_api_key: Optional[str] = None
    default_upstream_model: str = "gpt-4o-mini"
    
    # HTTP Connection Pool Tuning
    http_max_connections: int = 100
    http_max_keepalive_connections: int = 20
    http_keepalive_expiry_seconds: float = 30.0
    http_connect_timeout_seconds: float = 5.0
    http_read_timeout_seconds: float = 60.0
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    
    # Gateway Host & Port
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8000


settings = GatewaySettings()

