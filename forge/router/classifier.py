"""
FORGE Task Classifier
Determines task type + minimum tier required. Heuristic-first, router-as-judge later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from forge import (
    TASK_AUTOCOMPLETE, TASK_EXPLAIN, TASK_DEBUG,
    TASK_TEST, TASK_REFACTOR, TASK_WRITE,
    TASK_ARCHITECTURE, TASK_RESEARCH, TASK_REVIEW,
    TIER_LOCAL, TIER_FAST_FREE, TIER_SMART_FREE, TIER_BEST_FREE,
)


# Task → minimum tier mapping
TASK_TIER_MAP: dict[str, str] = {
    TASK_AUTOCOMPLETE: TIER_LOCAL,
    TASK_EXPLAIN:      TIER_FAST_FREE,
    TASK_DEBUG:        TIER_FAST_FREE,
    TASK_TEST:         TIER_SMART_FREE,
    TASK_REFACTOR:     TIER_SMART_FREE,
    TASK_WRITE:        TIER_SMART_FREE,
    TASK_ARCHITECTURE: TIER_BEST_FREE,
    TASK_RESEARCH:     TIER_BEST_FREE,
    TASK_REVIEW:       TIER_BEST_FREE,
}


@dataclass
class Classification:
    task_type:    str
    min_tier:     str
    confidence:   float   # 0-1
    context_size: int     # estimated tokens in context
    reasoning:    str     # why this classification


# ── Keyword patterns ──────────────────────────────────────────────────────────
_PATTERNS: list[tuple[str, list[str]]] = [
    (TASK_DEBUG, [
        r'\b(bug|error|exception|traceback|fix|broken|crash|fail|wrong output)\b',
        r'\b(why (is|does|isn\'t)|not working|doesn\'t work)\b',
        r'(TypeError|ValueError|AttributeError|KeyError|ImportError)',
    ]),
    (TASK_TEST, [
        r'\b(test|unittest|pytest|spec|coverage|mock|fixture)\b',
        r'\b(write tests?|add tests?|test this|tdd)\b',
    ]),
    (TASK_ARCHITECTURE, [
        r'\b(architect|design|structure|system design|how should i|best way to build)\b',
        r'\b(pattern|design pattern|microservice|monolith|api design)\b',
        r'\b(scale|scalability|performance|optimize the system)\b',
    ]),
    (TASK_RESEARCH, [
        r'\b(research|compare|benchmark|evaluate|which is better|pros and cons)\b',
        r'\b(paper|arxiv|state of the art|sota|literature)\b',
    ]),
    (TASK_REVIEW, [
        r'\b(review|code review|pr review|feedback|critique|improve)\b',
        r'\b(is this (good|correct|right)|what do you think|any issues)\b',
    ]),
    (TASK_REFACTOR, [
        r'\b(refactor|clean up|reorganize|restructure|simplify|extract)\b',
        r'\b(make (this|it) (cleaner|better|more readable))\b',
    ]),
    (TASK_WRITE, [
        r'\b(write|implement|create|build|add|generate)\b.{0,30}\b(function|class|module|script|file)\b',
        r'\b(implement|code|program)\b',
    ]),
    (TASK_EXPLAIN, [
        r'\b(explain|what (is|does|are)|how does|tell me about|describe)\b',
        r'\b(understand|clarify|what does .* mean)\b',
    ]),
    (TASK_AUTOCOMPLETE, [
        r'^.{0,30}$',   # Very short prompt = autocomplete
    ]),
]


class TaskClassifier:
    """
    Fast heuristic task classifier.
    Phase 2: wire in Ollama qwen:7b as router-as-judge for ambiguous cases.
    """

    def classify(self, prompt: str, context: str = "") -> Classification:
        prompt_lower = prompt.lower().strip()
        context_tokens = len(context) // 4

        # Run all patterns
        scores: dict[str, float] = {}
        for task_type, patterns in _PATTERNS:
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, prompt_lower, re.IGNORECASE):
                    score += 1.0
            if score > 0:
                scores[task_type] = score

        if not scores:
            # Default: classify by context size
            if context_tokens > 2000:
                task_type = TASK_REFACTOR
            elif len(prompt) < 50:
                task_type = TASK_AUTOCOMPLETE
            else:
                task_type = TASK_WRITE
            confidence = 0.5
            reasoning = "default classification (no pattern match)"
        else:
            task_type = max(scores, key=lambda k: scores[k])
            confidence = min(1.0, scores[task_type] / 3)
            reasoning = f"pattern match: {scores}"

        # Context size bump: large context → higher tier
        min_tier = TASK_TIER_MAP[task_type]
        if context_tokens >= 3000 and min_tier == TIER_FAST_FREE:
            min_tier = TIER_SMART_FREE
            reasoning += " | bumped tier: large context"

        return Classification(
            task_type=task_type,
            min_tier=min_tier,
            confidence=confidence,
            context_size=context_tokens,
            reasoning=reasoning,
        )
