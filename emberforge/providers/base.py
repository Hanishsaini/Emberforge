"""
EMBERFORGE BaseProvider — all providers inherit from this.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator


# ── Tool call (OpenAI function-calling format) ────────────────────────────────
@dataclass
class ToolCall:
    id:        str
    name:      str
    arguments: str   # raw JSON string, parsed by the tool executor


# ── Response dataclass ────────────────────────────────────────────────────────
@dataclass
class EmberResponse:
    content:        str
    provider:       str
    model:          str
    tokens_in:      int  = 0
    tokens_out:     int  = 0
    tokens_saved:   int  = 0     # from compression
    latency_ms:     int  = 0
    success:        bool = True
    error:          str  = ""
    quality_score:  float= 1.0   # 0-1, used by router for retry decisions
    tool_calls:     list = field(default_factory=list)   # list[ToolCall]

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def cost_usd(self) -> float:
        # All free providers → $0. Override in paid providers.
        return 0.0


# ── Health state ─────────────────────────────────────────────────────────────
@dataclass
class ProviderHealth:
    healthy:          bool  = True
    quota_pct:        float = 100.0   # 0-100
    last_error:       str   = ""
    last_checked:     float = field(default_factory=time.time)
    consecutive_fails:int   = 0
    avg_latency_ms:   int   = 0

    def mark_fail(self, error: str) -> None:
        self.consecutive_fails += 1
        self.last_error = error
        self.last_checked = time.time()
        if self.consecutive_fails >= 3:
            self.healthy = False

    def mark_success(self, latency_ms: int) -> None:
        self.consecutive_fails = 0
        self.healthy = True
        self.last_error = ""
        self.last_checked = time.time()
        # Exponential moving average for latency
        if self.avg_latency_ms == 0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = int(0.8 * self.avg_latency_ms + 0.2 * latency_ms)

    def quota_depleted(self) -> bool:
        return self.quota_pct < 5.0


# ── Abstract base ─────────────────────────────────────────────────────────────
class BaseProvider(ABC):
    """
    All EMBERFORGE providers extend this.
    Providers are responsible for:
      - Calling the LLM API
      - Tracking health + quota
      - Returning EmberResponse
    """

    def __init__(
        self,
        name:      str,
        api_key:   str,
        tier:      str,
        models:    dict,
        base_url:  str,
        rpm_limit: int  = 20,
        supports_tools: bool = True,
    ):
        self.name      = name
        self.api_key   = api_key
        self.tier      = tier
        self.models    = models
        self.base_url  = base_url
        self.rpm_limit = rpm_limit
        self.supports_tools = supports_tools   # native function calling; flips to
                                               # False at runtime if the API rejects tools
        self.health    = ProviderHealth()

        # Rate limiting
        self._call_times: list[float] = []

    # ── Must implement ────────────────────────────────────────────────────────
    @abstractmethod
    async def complete(
        self,
        prompt:     str,
        context:    str    = "",
        system:     str    = "",
        max_tokens: int    = 4096,
        stream:     bool   = False,
    ) -> EmberResponse:
        """Call the provider and return a EmberResponse."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Ping the provider. Return True if alive."""
        ...

    # ── Multi-turn chat with tool calling (agent mode) ────────────────────────
    async def chat(
        self,
        messages:   list[dict],
        tools:      list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> EmberResponse:
        """
        Full-message-list chat with optional OpenAI-format tool schemas.
        Providers that support agent mode must override this.
        """
        raise NotImplementedError(f"{self.name} does not support chat()/agent mode")

    # ── Rate limiting ─────────────────────────────────────────────────────────
    def _check_rate_limit(self) -> bool:
        """Return True if we can make a request right now."""
        now = time.time()
        # Keep only calls in the last 60 seconds
        self._call_times = [t for t in self._call_times if now - t < 60]
        return len(self._call_times) < self.rpm_limit

    def _record_call(self) -> None:
        self._call_times.append(time.time())

    # ── Availability check ────────────────────────────────────────────────────
    def is_available(self) -> bool:
        return (
            self.health.healthy
            and not self.health.quota_depleted()
            and self._check_rate_limit()
            and bool(self.api_key)
        )

    # ── Primary model ─────────────────────────────────────────────────────────
    @property
    def primary_model(self) -> str:
        return self.models.get("primary", "")

    @property
    def fallback_model(self) -> str:
        return self.models.get("fallback", self.primary_model)

    def __repr__(self) -> str:
        status = "✅" if self.health.healthy else "❌"
        return f"{status} {self.name} [{self.tier}] ({self.primary_model})"
