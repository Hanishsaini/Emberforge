"""
EMBERFORGE OpenAI-Compatible Provider
Handles: Groq, Gemini, OpenCode, Mistral, OpenRouter, NVIDIA NIM, Ollama
All use the same OpenAI /chat/completions format. One class, all providers.

Phase 3 hardening:
  - One shared httpx.AsyncClient per provider (loop-aware) — no per-request
    TLS handshakes.
  - Unified error handling with cooldowns: 429 honors Retry-After, 5xx gets a
    short cooldown, auth errors a long one. Providers recover automatically.
  - Real SSE streaming via on_token callback.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

import httpx

from emberforge.compressor.tokens import count_tokens
from emberforge.providers.base import BaseProvider, EmberResponse, ToolCall


# Verbosity trim — injected into every system prompt (Headroom-inspired)
VERBOSITY_SUFFIX = """
RESPONSE RULES:
- Be direct and terse. No preambles like "Sure!" or "Great question!"
- Never restate the task back to me
- Never re-print code I just showed you unless you changed it
- Skip reasoning on routine steps
- Code blocks only when the output IS code
- If fixing a bug: show only the changed lines + context
"""

# Cooldown policy per failure class (seconds)
COOLDOWN_RATE_LIMITED = 60.0    # 429 without Retry-After header
COOLDOWN_SERVER_ERROR = 30.0    # 5xx
COOLDOWN_AUTH_ERROR   = 300.0   # 401/403 — a bad key won't fix itself quickly
COOLDOWN_NETWORK      = 20.0    # timeouts / connection refused


def parse_sse_line(line: str) -> str | None:
    """
    Parse one SSE line from an OpenAI-compatible stream.
    Returns the content delta ("" for valid-but-empty chunks), or None for
    non-data lines, malformed JSON, and the [DONE] sentinel.
    """
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        chunk = json.loads(data)
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("delta", {}).get("content") or ""
    except json.JSONDecodeError:
        return None


class OpenAICompatProvider(BaseProvider):
    """
    Single provider class for all OpenAI-compatible APIs.
    Groq, Gemini (beta), OpenCode Zen, Mistral, OpenRouter, NVIDIA NIM, Ollama.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client:      httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    # ── Shared HTTP client (loop-aware: the chat REPL runs one asyncio.run
    #    per turn, so the client must be rebuilt when the loop changes) ────────
    def _get_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if (
            self._client is None
            or self._client.is_closed
            or self._client_loop is not loop
        ):
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
            self._client_loop = loop
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    def _headers(self) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        if self.name == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/Hanishsaini/emberforge"
            headers["X-Title"]      = "EmberForge"
        return headers

    # ── Unified failure handling: classify → cooldown → EmberResponse ────────
    def _fail(self, error: str, cooldown: float | None, model: str) -> EmberResponse:
        self.health.mark_fail(error, cooldown_seconds=cooldown)
        return EmberResponse(
            content="", provider=self.name, model=model,
            success=False, error=error,
        )

    def _classify_http_error(self, e: httpx.HTTPStatusError) -> tuple[str, float]:
        """Returns (error message, cooldown seconds; 0 = no cooldown)."""
        status = e.response.status_code
        body   = e.response.text[:300]
        error  = f"HTTP {status}: {body}"

        if status == 429:
            retry_after = e.response.headers.get("retry-after", "")
            try:
                cooldown = max(1.0, float(retry_after))
            except ValueError:
                cooldown = COOLDOWN_RATE_LIMITED
            return f"rate/quota limited (retry in {int(cooldown)}s): {body[:120]}", cooldown
        if status in (401, 403):
            return error, COOLDOWN_AUTH_ERROR
        if status >= 500:
            return error, COOLDOWN_SERVER_ERROR
        return error, 0.0   # other 4xx: likely our payload — no cooldown

    # ── One-shot completion (optionally streaming) ────────────────────────────
    async def complete(
        self,
        prompt:     str,
        context:    str  = "",
        system:     str  = "",
        max_tokens: int  = 4096,
        stream:     bool = False,
        on_token:   Callable[[str], None] | None = None,
    ) -> EmberResponse:

        if not self._check_rate_limit():
            return EmberResponse(
                content="", provider=self.name, model=self.primary_model,
                success=False, error=f"Rate limit: {self.rpm_limit} RPM exceeded",
            )

        sys_prompt = (system or "You are EMBERFORGE, an expert coding assistant.") + VERBOSITY_SUFFIX
        messages: list[dict] = [{"role": "system", "content": sys_prompt}]
        if context:
            messages.append({
                "role": "user",
                "content": f"<context>\n{context}\n</context>\n\n{prompt}"
            })
        else:
            messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model":      self.primary_model,
            "messages":   messages,
            "max_tokens": max_tokens,
        }

        t_start = time.time()
        try:
            if stream and on_token is not None:
                return await self._complete_streaming(payload, messages, on_token, t_start)

            payload["stream"] = False
            client = self._get_client()
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload, headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

            latency_ms = int((time.time() - t_start) * 1000)
            content    = data["choices"][0]["message"]["content"]

            self._record_call()
            self.health.mark_success(latency_ms)

            return EmberResponse(
                content=content,
                provider=self.name,
                model=self.primary_model,
                tokens_in=data.get("usage", {}).get("prompt_tokens", 0),
                tokens_out=data.get("usage", {}).get("completion_tokens", 0),
                latency_ms=latency_ms,
                success=True,
            )

        except httpx.HTTPStatusError as e:
            error, cooldown = self._classify_http_error(e)
            # Model-level 4xx: try the fallback model once before giving up
            if e.response.status_code in (400, 404, 422):
                self.health.mark_fail(error)
                return await self._try_fallback(messages, max_tokens, t_start)
            return self._fail(error, cooldown or None, self.primary_model)

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            return self._fail(
                f"Connection error: {str(e)[:100]}", COOLDOWN_NETWORK, self.primary_model,
            )

    async def _complete_streaming(
        self,
        payload:  dict,
        messages: list[dict],
        on_token: Callable[[str], None],
        t_start:  float,
    ) -> EmberResponse:
        """SSE streaming path. Token counts are estimated (streams rarely send usage)."""
        payload["stream"] = True
        client = self._get_client()
        parts: list[str] = []

        async with client.stream(
            "POST", f"{self.base_url}/chat/completions",
            json=payload, headers=self._headers(),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                delta = parse_sse_line(line)
                if delta:
                    parts.append(delta)
                    on_token(delta)

        latency_ms = int((time.time() - t_start) * 1000)
        content = "".join(parts)
        self._record_call()
        self.health.mark_success(latency_ms)

        return EmberResponse(
            content=content,
            provider=self.name,
            model=self.primary_model,
            tokens_in=count_tokens("\n".join(str(m.get("content", "")) for m in messages)),
            tokens_out=count_tokens(content),
            latency_ms=latency_ms,
            success=True,
        )

    # ── Agent mode: multi-turn chat with tool calling ─────────────────────────
    async def chat(
        self,
        messages:   list[dict],
        tools:      list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> EmberResponse:
        """
        Send a full message list (system/user/assistant/tool turns) with optional
        OpenAI-format tool schemas. Returns content and/or parsed tool_calls.
        If the API rejects the tools parameter, flips self.supports_tools to False
        so the agent can fall back to the ReAct text protocol.
        """
        if not self._check_rate_limit():
            return EmberResponse(
                content="", provider=self.name, model=self.primary_model,
                success=False, error=f"Rate limit: {self.rpm_limit} RPM exceeded",
            )

        payload: dict[str, Any] = {
            "model":      self.primary_model,
            "messages":   messages,
            "max_tokens": max_tokens,
            "stream":     False,
        }
        if tools and self.supports_tools:
            payload["tools"]       = tools
            payload["tool_choice"] = "auto"

        t_start = time.time()
        try:
            client = self._get_client()
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload, headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

            latency_ms = int((time.time() - t_start) * 1000)
            message    = data["choices"][0]["message"]
            content    = message.get("content") or ""

            tool_calls: list[ToolCall] = []
            for tc in message.get("tool_calls") or []:
                fn = tc.get("function", {})
                tool_calls.append(ToolCall(
                    id=tc.get("id", f"call_{len(tool_calls)}"),
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", "{}"),
                ))

            self._record_call()
            self.health.mark_success(latency_ms)

            return EmberResponse(
                content=content,
                provider=self.name,
                model=self.primary_model,
                tokens_in=data.get("usage", {}).get("prompt_tokens", 0),
                tokens_out=data.get("usage", {}).get("completion_tokens", 0),
                latency_ms=latency_ms,
                success=True,
                tool_calls=tool_calls,
            )

        except httpx.HTTPStatusError as e:
            body = e.response.text[:300]
            error, cooldown = self._classify_http_error(e)

            # API rejected the tools param → remember, so the agent switches to
            # ReAct. No cooldown: the provider itself is fine, just tool-less.
            if (
                tools
                and e.response.status_code in (400, 404, 422)
                and ("tool" in body.lower() or "function" in body.lower())
            ):
                self.supports_tools = False
                self.health.mark_fail(error)
                return EmberResponse(
                    content="", provider=self.name, model=self.primary_model,
                    success=False, error=f"tools_unsupported: {error}",
                )

            return self._fail(error, cooldown or None, self.primary_model)

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            return self._fail(
                f"Connection error: {str(e)[:100]}", COOLDOWN_NETWORK, self.primary_model,
            )

    async def _try_fallback(
        self,
        messages:   list[dict],
        max_tokens: int,
        t_start:    float,
    ) -> EmberResponse:
        """Try fallback model when primary fails on a model-level 4xx."""
        payload = {
            "model":      self.fallback_model,
            "messages":   messages,
            "max_tokens": max_tokens,
        }
        try:
            client = self._get_client()
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload, headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

            latency_ms = int((time.time() - t_start) * 1000)
            content    = data["choices"][0]["message"]["content"]

            self.health.mark_success(latency_ms)
            return EmberResponse(
                content=content,
                provider=self.name,
                model=self.fallback_model,
                tokens_in=data.get("usage", {}).get("prompt_tokens", 0),
                tokens_out=data.get("usage", {}).get("completion_tokens", 0),
                latency_ms=latency_ms,
                success=True,
            )
        except Exception as e:
            return EmberResponse(
                content="", provider=self.name, model=self.fallback_model,
                success=False, error=str(e)[:100],
            )

    async def health_check(self) -> bool:
        """Quick ping — list models endpoint."""
        try:
            client = self._get_client()
            resp = await client.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0,
            )
            alive = resp.status_code in (200, 401)  # 401 = key wrong but server up
            if alive:
                self.health.mark_success(0)
            return alive
        except Exception:
            self.health.mark_fail("health check failed")
            return False
