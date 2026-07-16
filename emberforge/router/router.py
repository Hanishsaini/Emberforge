"""
EMBERFORGE Smart Router — the brain.
Routes tasks to the right provider based on:
  - Task classification (type + min tier)
  - Provider health + quota
  - Rate limits
  - Response quality (retry on bad output)
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Callable

from rich.console import Console

from emberforge import TIER_LOCAL
from emberforge.providers.base import BaseProvider, EmberResponse
from emberforge.providers import get_providers_at_or_above_tier
from emberforge.router.classifier import (
    TaskClassifier, Classification, TASK_TIER_MAP,
    build_judge_prompt, parse_judge_output,
)

console = Console()

# Quality gates: things a "successful" HTTP 200 can still get wrong.
_REFUSAL_RE = re.compile(
    r"^(i can'?not|i can'?t|i'?m sorry|sorry, i|as an ai\b|i am unable|i'?m unable)",
    re.IGNORECASE,
)


def quality_issue(content: str, task_type: str) -> str | None:
    """Return a reason string if the response looks unusable, else None."""
    stripped = content.strip()
    if task_type != "autocomplete" and len(stripped) < 10:
        return "empty or too short"
    if _REFUSAL_RE.match(stripped):
        return "model refused the task"
    # An odd number of code fences means the output was cut off mid-block
    if stripped.count("```") % 2 == 1:
        return "truncated mid code-fence"
    return None


@dataclass
class RouterResult:
    response:       EmberResponse
    classification: Classification
    attempts:       int
    total_ms:       int


class EmberRouter:
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
        use_judge: bool = True,
    ):
        self._providers   = providers
        self._classifier  = TaskClassifier()
        self._verbose     = verbose
        self._use_judge   = use_judge
        self._call_log:   list[dict] = []

    def _judge_provider(self) -> BaseProvider | None:
        """A local-tier provider (Ollama) used as router-as-judge — free & fast."""
        for p in self._providers.values():
            if p.tier == TIER_LOCAL and p.is_available():
                return p
        return None

    async def _maybe_judge(
        self, prompt: str, classification: Classification
    ) -> Classification:
        """
        Router-as-judge: when the regex heuristics are unsure, ask a small local
        model to classify. Any failure falls back to the heuristic silently.
        """
        if not self._use_judge or classification.confidence >= 0.6:
            return classification
        judge = self._judge_provider()
        if judge is None:
            return classification
        try:
            resp = await judge.complete(
                prompt=build_judge_prompt(prompt),
                system="You are a task classifier. Reply with exactly one word.",
                max_tokens=8,
            )
            verdict = parse_judge_output(resp.content) if resp.success else None
            if verdict and verdict != classification.task_type:
                if self._verbose:
                    console.print(
                        f"[dim]  ⚖ judge ({judge.name}): "
                        f"{classification.task_type} → {verdict}[/dim]"
                    )
                return Classification(
                    task_type=verdict,
                    min_tier=TASK_TIER_MAP[verdict],
                    confidence=0.85,
                    context_size=classification.context_size,
                    reasoning=f"router-as-judge override (heuristic said "
                              f"{classification.task_type}: {classification.reasoning})",
                )
        except Exception:
            pass
        return classification

    async def route(
        self,
        prompt:     str,
        context:    str  = "",
        system:     str  = "",
        max_tokens: int  = 4096,
        on_token:   Callable[[str], None] | None = None,
    ) -> RouterResult:
        """Main entry. Classify → route → return."""

        t_start = time.time()

        # Step 1: Classify (heuristics, escalated to a local judge if unsure)
        classification = self._classifier.classify(prompt, context)
        classification = await self._maybe_judge(prompt, classification)

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
                response=EmberResponse(
                    content="❌ No providers available. Check your API keys in ~/.emberforge/config.yaml",
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
                if on_token is not None:
                    try:
                        response = await provider.complete(
                            prompt=prompt, context=context, system=system,
                            max_tokens=max_tokens, stream=True, on_token=on_token,
                        )
                    except TypeError:
                        # provider doesn't support streaming — plain call
                        response = await provider.complete(
                            prompt=prompt, context=context, system=system,
                            max_tokens=max_tokens,
                        )
                else:
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

            # Quality check: HTTP 200 can still be unusable (refusal, truncation)
            issue = quality_issue(response.content, classification.task_type)
            if issue:
                last_error = f"quality: {issue}"
                if self._verbose:
                    console.print(f"[dim yellow]  ⚠ {issue} — trying next provider[/dim yellow]")
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
            response=EmberResponse(
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
        response:        EmberResponse,
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
