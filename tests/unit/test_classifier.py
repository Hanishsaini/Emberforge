"""Unit tests for task classifier."""
import pytest
from emberforge.router.classifier import TaskClassifier
from emberforge import (
    TASK_DEBUG, TASK_TEST, TASK_ARCHITECTURE,
    TASK_RESEARCH, TASK_REFACTOR, TASK_WRITE,
    TASK_EXPLAIN, TASK_AUTOCOMPLETE,
    TIER_LOCAL, TIER_FAST_FREE, TIER_SMART_FREE, TIER_BEST_FREE,
)


class TestTaskClassifier:
    def setup_method(self):
        self.c = TaskClassifier()

    def test_debug_keywords(self):
        result = self.c.classify("why is my code throwing AttributeError?")
        assert result.task_type == TASK_DEBUG
        assert result.min_tier == TIER_FAST_FREE

    def test_test_keywords(self):
        result = self.c.classify("write pytest unit tests for the router module")
        assert result.task_type == TASK_TEST
        assert result.min_tier == TIER_SMART_FREE

    def test_architecture_keywords(self):
        result = self.c.classify("how should I design the multi-provider routing system?")
        assert result.task_type == TASK_ARCHITECTURE
        assert result.min_tier == TIER_BEST_FREE

    def test_research_keywords(self):
        result = self.c.classify("compare BM25 vs semantic search for code retrieval")
        assert result.task_type == TASK_RESEARCH
        assert result.min_tier == TIER_BEST_FREE

    def test_refactor_keywords(self):
        result = self.c.classify("refactor this function to be cleaner")
        assert result.task_type == TASK_REFACTOR
        assert result.min_tier == TIER_SMART_FREE

    def test_short_prompt_autocomplete(self):
        result = self.c.classify("def add(")
        assert result.task_type == TASK_AUTOCOMPLETE
        assert result.min_tier == TIER_LOCAL

    def test_explain_keywords(self):
        result = self.c.classify("explain how the simhash deduplication works")
        assert result.task_type == TASK_EXPLAIN
        assert result.min_tier == TIER_FAST_FREE

    def test_large_context_bumps_tier(self):
        large_context = "x" * 12001  # > 3000 tokens
        result = self.c.classify("fix the bug", context=large_context)
        # debug normally = fast_free, but large context bumps to smart_free
        assert result.min_tier == TIER_SMART_FREE

    def test_confidence_populated(self):
        result = self.c.classify("write unit tests for the memory module")
        assert 0 <= result.confidence <= 1.0

    def test_context_size_tracked(self):
        context = "a" * 400  # ~100 tokens
        result = self.c.classify("fix bug", context=context)
        assert result.context_size > 0
