"""
EmberForge AHE Evolution Loop — the harness improves its own system prompt.

The pieces built in earlier phases interlock here:
  - Phase 7 GEPA gives us failure post-mortems  → the mutation signal
  - Phase 6 eval suite gives us task pass rates → the fitness function
  - This module proposes a prompt variant, scores it against the current
    prompt on the SAME eval tasks, and promotes it only if it is provably
    not worse (higher pass rate, or equal pass rate with fewer tokens).

Guardrails (non-negotiable):
  - never promote on a lower pass rate
  - variants over 1,000 tokens are rejected outright (Pi discipline)
  - variants must keep the tool protocol (tool names + final-reply rule)
  - full history is recorded; the previous prompt is never deleted
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from emberforge.agent import AGENT_SYSTEM_PROMPT
from emberforge.compressor.tokens import count_tokens
from emberforge.config import settings as _settings

MAX_PROMPT_TOKENS = 1_000
REQUIRED_MARKERS  = ("read_file", "edit_file", "run_shell")

# scorer: system_prompt_text -> (pass_rate, total_tokens)
Scorer = Callable[[str], Awaitable[tuple[float, int]]]

PROPOSE_SYSTEM = (
    "You are a harness engineer improving a coding agent's system prompt. "
    "Reply with the new prompt text only — no preamble, no fences."
)

PROPOSE_PROMPT = """Below is the current system prompt of a coding agent, followed by
post-mortems of its recent failures. Rewrite the prompt to prevent those
failure modes.

CONSTRAINTS:
- stay under 900 tokens; terser is better
- keep the tool names exactly: read_file, write_file, edit_file, list_dir, grep_search, run_shell
- keep the rule that the final reply is plain text with NO tool call
- change only what the failures justify changing

CURRENT PROMPT:
{current}

RECENT FAILURE POST-MORTEMS:
{analyses}

Return ONLY the improved prompt text."""


def validate_variant(text: str) -> str | None:
    """Return a rejection reason, or None if the variant is acceptable."""
    if not text or not text.strip():
        return "empty variant"
    if count_tokens(text) > MAX_PROMPT_TOKENS:
        return f"over {MAX_PROMPT_TOKENS} tokens"
    for marker in REQUIRED_MARKERS:
        if marker not in text:
            return f"missing required tool mention: {marker}"
    return None


@dataclass
class EvolveResult:
    baseline_pass:   float
    baseline_tokens: int
    variant_pass:    float = 0.0
    variant_tokens:  int   = 0
    promoted:        bool  = False
    reason:          str   = ""
    variant:         str   = ""


class PromptStore:
    """Versioned storage for the agent's system prompt in ~/.emberforge/prompts/."""

    def __init__(self, home: Path | None = None):
        self.dir = (home or _settings.EMBERFORGE_HOME) / "prompts"

    @property
    def active_path(self) -> Path:
        return self.dir / "active.txt"

    @property
    def history_path(self) -> Path:
        return self.dir / "history.json"

    def get_active(self) -> str:
        if self.active_path.exists():
            text = self.active_path.read_text(encoding="utf-8").strip()
            if text and validate_variant(text) is None:
                return text
        return AGENT_SYSTEM_PROMPT

    def history(self) -> list[dict]:
        if self.history_path.exists():
            try:
                return json.loads(self.history_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return []
        return []

    def promote(self, text: str, record: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        # archive the outgoing prompt — evolution must be reversible
        if self.active_path.exists():
            stamp = time.strftime("%Y%m%d-%H%M%S")
            (self.dir / f"retired-{stamp}.txt").write_text(
                self.active_path.read_text(encoding="utf-8"), encoding="utf-8")
        self.active_path.write_text(text, encoding="utf-8")
        entries = self.history()
        entries.append({**record, "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.history_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


class PromptEvolver:
    """Propose → score → promote-only-if-not-worse."""

    async def propose(self, current: str, analyses: list[str], router) -> str | None:
        if not analyses:
            return None
        try:
            result = await router.route(
                prompt=PROPOSE_PROMPT.format(
                    current=current,
                    analyses="\n".join(f"- {a}" for a in analyses[:8]),
                ),
                system=PROPOSE_SYSTEM,
                max_tokens=1500,
            )
            if not result.response.success:
                return None
            variant = result.response.content.strip()
            if variant.startswith("```"):
                variant = variant.strip("`").strip()
            return variant if validate_variant(variant) is None else None
        except Exception:
            return None

    async def evolve(
        self,
        scorer:  Scorer,
        router,
        memory,
        project: str,
        store:   PromptStore | None = None,
    ) -> EvolveResult:
        store = store or PromptStore()
        current = store.get_active()

        base_pass, base_tokens = await scorer(current)

        analyses = [
            f["analysis"] for f in memory.recent_failures(project, limit=10)
            if f.get("analysis")
        ]
        variant = await self.propose(current, analyses, router)
        if variant is None:
            return EvolveResult(
                baseline_pass=base_pass, baseline_tokens=base_tokens,
                reason="no valid variant proposed"
                       + ("" if analyses else " (no failure analyses to learn from)"),
            )

        var_pass, var_tokens = await scorer(variant)

        promoted = (
            var_pass > base_pass
            or (var_pass == base_pass and var_tokens < base_tokens)
        )
        reason = (
            "higher pass rate" if var_pass > base_pass else
            "equal pass rate, fewer tokens" if promoted else
            "variant not better — kept current prompt"
        )
        if promoted:
            store.promote(variant, {
                "baseline_pass": base_pass, "variant_pass": var_pass,
                "baseline_tokens": base_tokens, "variant_tokens": var_tokens,
                "reason": reason, "analyses_used": len(analyses),
            })

        return EvolveResult(
            baseline_pass=base_pass, baseline_tokens=base_tokens,
            variant_pass=var_pass, variant_tokens=var_tokens,
            promoted=promoted, reason=reason, variant=variant,
        )
