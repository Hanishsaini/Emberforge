"""
EmberForge GEPA — failure analysis (Genetic-Pareto-style reflection, lite).

When an agent run fails we already log THAT it failed. GEPA asks WHY:
a cheap model reads the failure trace (the tools the agent actually called
and what they returned) and produces a root cause + fix hint. The analysis
lands in the failures table and comes back as a <past-failures> warning the
next time a similar task runs — so the agent inherits the post-mortem, not
just the obituary.
"""
from __future__ import annotations

import json

from emberforge.tools import ExecutionRecord

ANALYSIS_SYSTEM = (
    "You are a failure analyst for a coding agent. Be concrete and terse. "
    "Reply with JSON only."
)

ANALYSIS_PROMPT = """A coding agent failed at a task. Diagnose it.

Task: {prompt}

Failure: {error}

Tool trace (what the agent actually did):
{trace}

Return a JSON object with exactly these keys:
{{
  "root_cause": "one sentence: the real reason it failed",
  "fix_hint": "one sentence: what to do differently next attempt"
}}
JSON only. No preamble."""


def build_trace(
    history: list[ExecutionRecord],
    max_records: int = 12,
    max_chars: int = 1800,
) -> str:
    """Compact, human/LLM-readable trace of the agent's tool calls."""
    if not history:
        return "(no tool calls were made)"
    lines = []
    for rec in history[-max_records:]:
        status = "ok" if rec.success else "FAILED"
        args = json.dumps(rec.args)[:100]
        lines.append(f"{rec.tool}({args}) -> {status}: {rec.output_preview[:80]}")
    return "\n".join(lines)[:max_chars]


class FailureAnalyst:
    """Turns a failure trace into a stored root-cause analysis."""

    async def analyze(self, prompt: str, error: str, trace: str, router) -> str | None:
        """
        Returns a compact analysis string ("root cause | fix: hint") or None.
        Any failure in the analysis itself is swallowed — post-mortems must
        never break the caller.
        """
        try:
            result = await router.route(
                prompt=ANALYSIS_PROMPT.format(
                    prompt=prompt[:400], error=error[:300], trace=trace,
                ),
                system=ANALYSIS_SYSTEM,
                max_tokens=300,
            )
            if not result.response.success:
                return None
            content = result.response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content.strip())
            root = str(data.get("root_cause", "")).strip()
            hint = str(data.get("fix_hint", "")).strip()
            if not root:
                return None
            return f"{root} | fix: {hint}" if hint else root
        except Exception:
            return None
