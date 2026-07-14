"""
FORGE Smart Router — the brain.
Routes tasks to the right provider based on:
  - Task classification (type + min tier)
  - Provider health + quota
  - Rate limits
  - Response quality (retry on bad output)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from rich.console import Console

from forge.providers.base import BaseProvider, ForgeResponse
from forge.providers import get_providers_at_or_above_tier
from forge.router.classifier import TaskClassifier, Classification

console = Console()


@dataclass
class RouterResult:
    response:       ForgeResponse
    classification: Classification
    attempts:       int
    total_ms:       int


class ForgeRouter:
    """
    Main routing engine. Algorithm:

    1. Classify task → task_type + min_tier
    2. Get all providers at min_tier or above, sorted by availability + latency
    3. Try providers in order:
       a. Skip if not available (quota/health/rate)
       b. Call provider
       c. Check response quality
       d. On success → return
       e. On failure/low quality → try next
    4. If all fail → return best partial response with error context
    """

    def __init__(
        self,
        providers: dict[str, BaseProvider],
        verbose:   bool = True,
    ):
        self._providers   = providers
        self._classifier  = TaskClassifier()
        self._verbose     = verbose
        self._call_log:   list[dict] = []

    async def route(
        self,
        prompt:     str,
        context:    str  = "",
        system:     str  = "",
        max_tokens: int  = 4096,
    ) -> RouterResult:
        """Main entry. Classify → route → return."""

        t_start = time.time()

        # Step 1: Classify
        classification = self._classifier.classify(prompt, context)

        if self._verbose:
            console.print(
                f"[dim]→ Task: [cyan]{classification.task_type}[/cyan] | "
                f"Tier: [yellow]{classification.min_tier}[/yellow] | "
                f"Confidence: {classification.confidence:.0%}[/dim]"
            )

        # Step 2: Get candidate providers
        candidates = get_providers_at_or_above_tier(
            self._providers, classification.min_tier
        )

        if not candidates:
            # No providers available — emergency fallback message
            return RouterResult(
                response=ForgeResponse(
                    content="❌ No providers available. Check your API keys in ~/.forge/config.yaml",
                    provider="none",
                    model="none",
                    success=False,
                    error="No available providers",
                ),
                classification=classification,
                attempts=0,
                total_ms=int((time.time() - t_start) * 1000),
            )

        # Step 3: Try providers in order
        attempts = 0
        last_error = ""

        for provider in candidates:
            attempts += 1
            if self._verbose:
                console.print(f"[dim]  Trying [bold]{provider.name}[/bold] ({provider.primary_model})...[/dim]")

            try:
                response = await provider.complete(
                    prompt=prompt,
                    context=context,
                    system=system,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                last_error = str(e)
                provider.health.mark_fail(last_error)
                if self._verbose:
                    console.print(f"[dim red]  ✗ Exception: {last_error[:60]}[/dim red]")
                continue

            if not response.success:
                last_error = response.error
                if self._verbose:
                    console.print(f"[dim red]  ✗ Failed: {last_error[:60]}[/dim red]")
                continue

            # Quality check: response too short for non-autocomplete = suspicious
            if (
                classification.task_type != "autocomplete"
                and len(response.content.strip()) < 10
            ):
                if self._verbose:
                    console.print(f"[dim yellow]  ⚠ Empty response, trying next...[/dim yellow]")
                continue

            # ✅ Success
            total_ms = int((time.time() - t_start) * 1000)
            if self._verbose:
                console.print(
                    f"[dim green]  ✓ {provider.name} | "
                    f"{response.tokens_in}→{response.tokens_out} tokens | "
                    f"{total_ms}ms[/dim green]"
                )

            self._log(classification, provider.name, response, attempts, total_ms)

            return RouterResult(
                response=response,
                classification=classification,
                attempts=attempts,
                total_ms=total_ms,
            )

        # All providers failed
        total_ms = int((time.time() - t_start) * 1000)
        return RouterResult(
            response=ForgeResponse(
                content=f"All {attempts} providers failed. Last error: {last_error}",
                provider="none",
                model="none",
                success=False,
                error=last_error,
            ),
            classification=classification,
            attempts=attempts,
            total_ms=total_ms,
        )

    def _log(
        self,
        classification: Classification,
        provider:        str,
        response:        ForgeResponse,
        attempts:        int,
        total_ms:        int,
    ) -> None:
        self._call_log.append({
            "task_type":  classification.task_type,
            "min_tier":   classification.min_tier,
            "provider":   provider,
            "model":      response.model,
            "tokens_in":  response.tokens_in,
            "tokens_out": response.tokens_out,
            "attempts":   attempts,
            "ms":         total_ms,
            "ts":         time.time(),
        })

    def stats(self) -> dict:
        if not self._call_log:
            return {}
        total_in  = sum(r["tokens_in"] for r in self._call_log)
        total_out = sum(r["tokens_out"] for r in self._call_log)
        providers = {}
        for r in self._call_log:
            providers[r["provider"]] = providers.get(r["provider"], 0) + 1
        return {
            "total_calls":   len(self._call_log),
            "total_tokens":  total_in + total_out,
            "tokens_in":     total_in,
            "tokens_out":    total_out,
            "provider_dist": providers,
        }
