"""
FORGE OpenAI-Compatible Provider
Handles: Groq, Gemini, OpenCode, Mistral, OpenRouter, NVIDIA NIM, Ollama
All use the same OpenAI /chat/completions format. One class, all providers.
"""
from __future__ import annotations

import time
import httpx
from typing import Any

from forge.providers.base import BaseProvider, ForgeResponse


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


class OpenAICompatProvider(BaseProvider):
    """
    Single provider class for all OpenAI-compatible APIs.
    Groq, Gemini (beta), OpenCode Zen, Mistral, OpenRouter, NVIDIA NIM, Ollama.
    """

    async def complete(
        self,
        prompt:     str,
        context:    str  = "",
        system:     str  = "",
        max_tokens: int  = 4096,
        stream:     bool = False,
    ) -> ForgeResponse:

        if not self._check_rate_limit():
            return ForgeResponse(
                content="",
                provider=self.name,
                model=self.primary_model,
                success=False,
                error=f"Rate limit: {self.rpm_limit} RPM exceeded",
            )

        # Build system prompt
        sys_prompt = (system or "You are FORGE, an expert coding assistant.") + VERBOSITY_SUFFIX

        # Build messages
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
            "stream":     False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        # OpenRouter extras
        if self.name == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/honeystark/forge"
            headers["X-Title"]      = "FORGE"

        t_start = time.time()
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            latency_ms = int((time.time() - t_start) * 1000)

            content    = data["choices"][0]["message"]["content"]
            tokens_in  = data.get("usage", {}).get("prompt_tokens", 0)
            tokens_out = data.get("usage", {}).get("completion_tokens", 0)

            self._record_call()
            self.health.mark_success(latency_ms)

            return ForgeResponse(
                content=content,
                provider=self.name,
                model=self.primary_model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                success=True,
            )

        except httpx.HTTPStatusError as e:
            error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            self.health.mark_fail(error)

            # Try fallback model if primary fails with 4xx
            if e.response.status_code in (400, 404, 422):
                return await self._try_fallback(messages, headers, max_tokens, t_start)

            return ForgeResponse(
                content="", provider=self.name, model=self.primary_model,
                success=False, error=error,
            )

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            error = f"Connection error: {str(e)[:100]}"
            self.health.mark_fail(error)
            return ForgeResponse(
                content="", provider=self.name, model=self.primary_model,
                success=False, error=error,
            )

    async def _try_fallback(
        self,
        messages:   list[dict],
        headers:    dict,
        max_tokens: int,
        t_start:    float,
    ) -> ForgeResponse:
        """Try fallback model when primary fails."""
        payload = {
            "model":      self.fallback_model,
            "messages":   messages,
            "max_tokens": max_tokens,
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            latency_ms = int((time.time() - t_start) * 1000)
            content    = data["choices"][0]["message"]["content"]

            self.health.mark_success(latency_ms)
            return ForgeResponse(
                content=content,
                provider=self.name,
                model=self.fallback_model,
                tokens_in=data.get("usage", {}).get("prompt_tokens", 0),
                tokens_out=data.get("usage", {}).get("completion_tokens", 0),
                latency_ms=latency_ms,
                success=True,
            )
        except Exception as e:
            return ForgeResponse(
                content="", provider=self.name, model=self.fallback_model,
                success=False, error=str(e)[:100],
            )

    async def health_check(self) -> bool:
        """Quick ping — list models endpoint."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                alive = resp.status_code in (200, 401)  # 401 = key wrong but server up
                if alive:
                    self.health.mark_success(0)
                return alive
        except Exception:
            self.health.mark_fail("health check failed")
            return False
