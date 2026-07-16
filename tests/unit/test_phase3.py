"""Phase 3 tests — cooldown health, error classification, SSE parsing,
quality gates, router-as-judge, streaming passthrough."""
import time

import httpx
import pytest

from emberforge import TIER_LOCAL, TIER_SMART_FREE, TIER_FAST_FREE
from emberforge.providers.base import BaseProvider, EmberResponse, ProviderHealth
from emberforge.providers.openai_compat import (
    OpenAICompatProvider, parse_sse_line,
    COOLDOWN_AUTH_ERROR, COOLDOWN_SERVER_ERROR,
)
from emberforge.router.classifier import build_judge_prompt, parse_judge_output
from emberforge.router.router import EmberRouter, quality_issue


# ── Helpers ───────────────────────────────────────────────────────────────────
class DummyProvider(BaseProvider):
    """Configurable fake provider for router-level tests."""

    def __init__(self, name="dummy", tier=TIER_SMART_FREE, content="ok " * 20,
                 supports_streaming=False):
        super().__init__(name=name, api_key="k", tier=tier,
                         models={"primary": "m1"}, base_url="http://x")
        self.content = content
        self.calls = 0
        self.supports_streaming = supports_streaming

    async def complete(self, prompt, context="", system="", max_tokens=4096,
                       stream=False, **kwargs):
        self.calls += 1
        on_token = kwargs.get("on_token")
        if on_token is not None and not self.supports_streaming:
            raise TypeError("unexpected keyword argument 'on_token'")
        if on_token is not None:
            for chunk in ("hel", "lo ", "world"):
                on_token(chunk)
        return EmberResponse(content=self.content, provider=self.name, model="m1",
                             tokens_in=5, tokens_out=5)

    async def health_check(self):
        return True


class NoStreamProvider(DummyProvider):
    """Strict signature — no **kwargs, streaming call raises TypeError."""

    async def complete(self, prompt, context="", system="", max_tokens=4096,
                       stream=False):
        self.calls += 1
        return EmberResponse(content=self.content, provider=self.name, model="m1")


def http_error(status: int, body: str = "err", headers: dict | None = None):
    req = httpx.Request("POST", "http://api.test/v1/chat/completions")
    resp = httpx.Response(status, request=req, text=body, headers=headers or {})
    return httpx.HTTPStatusError("boom", request=req, response=resp)


# ── Cooldown health ───────────────────────────────────────────────────────────
class TestCooldownHealth:
    def test_explicit_cooldown_blocks_availability(self):
        p = DummyProvider()
        p.health.mark_fail("HTTP 429", cooldown_seconds=60)
        assert p.health.in_cooldown()
        assert not p.is_available()

    def test_cooldown_expiry_restores_availability(self):
        p = DummyProvider()
        p.health.mark_fail("HTTP 429", cooldown_seconds=60)
        p.health.cooldown_until = time.time() - 1   # fast-forward
        assert not p.health.in_cooldown()
        assert p.is_available()

    def test_provider_recovers_after_repeated_failures(self):
        """The old permanent-death bug: 3 fails used to kill a provider forever."""
        p = DummyProvider()
        for _ in range(3):
            p.health.mark_fail("HTTP 500")
        assert p.health.in_cooldown()            # backoff, not death
        p.health.cooldown_until = time.time() - 1
        assert p.is_available()                  # gets another chance

    def test_backoff_grows_with_consecutive_failures(self):
        h = ProviderHealth()
        for _ in range(3):
            h.mark_fail("x")
        first = h.cooldown_until
        h.cooldown_until = 0                     # reset for measurement
        h.mark_fail("x")                         # 4th failure
        assert h.cooldown_until - time.time() > (first - time.time())

    def test_success_clears_cooldown(self):
        h = ProviderHealth()
        h.mark_fail("x", cooldown_seconds=300)
        h.mark_success(50)
        assert not h.in_cooldown()
        assert h.consecutive_fails == 0
        assert h.cooldown_reason == ""

    def test_two_fails_do_not_cooldown(self):
        h = ProviderHealth()
        h.mark_fail("x")
        h.mark_fail("x")
        assert not h.in_cooldown()


# ── HTTP error classification ─────────────────────────────────────────────────
class TestErrorClassification:
    @pytest.fixture
    def provider(self):
        return OpenAICompatProvider(
            name="groq", api_key="k", tier=TIER_SMART_FREE,
            models={"primary": "m"}, base_url="http://x",
        )

    def test_429_honors_retry_after(self, provider):
        err, cooldown = provider._classify_http_error(
            http_error(429, "slow down", {"retry-after": "17"}))
        assert cooldown == 17.0
        assert "rate/quota" in err

    def test_429_default_without_header(self, provider):
        _, cooldown = provider._classify_http_error(http_error(429))
        assert cooldown == 60.0

    def test_auth_error_long_cooldown(self, provider):
        _, cooldown = provider._classify_http_error(http_error(401, "bad key"))
        assert cooldown == COOLDOWN_AUTH_ERROR

    def test_server_error_short_cooldown(self, provider):
        _, cooldown = provider._classify_http_error(http_error(503))
        assert cooldown == COOLDOWN_SERVER_ERROR

    def test_client_error_no_cooldown(self, provider):
        _, cooldown = provider._classify_http_error(http_error(404, "no model"))
        assert cooldown == 0.0


# ── SSE stream parsing ────────────────────────────────────────────────────────
class TestSSEParsing:
    def test_content_delta(self):
        line = 'data: {"choices": [{"delta": {"content": "hello"}}]}'
        assert parse_sse_line(line) == "hello"

    def test_done_sentinel(self):
        assert parse_sse_line("data: [DONE]") is None

    def test_non_data_lines(self):
        assert parse_sse_line("event: ping") is None
        assert parse_sse_line("") is None
        assert parse_sse_line(": comment") is None

    def test_malformed_json(self):
        assert parse_sse_line("data: {broken") is None

    def test_empty_choices(self):
        assert parse_sse_line('data: {"choices": []}') == ""

    def test_delta_without_content(self):
        line = 'data: {"choices": [{"delta": {"role": "assistant"}}]}'
        assert parse_sse_line(line) == ""


# ── Quality gates ─────────────────────────────────────────────────────────────
class TestQuality:
    def test_good_content_passes(self):
        assert quality_issue("Here is the fixed function:\n```py\nx=1\n```", "debug") is None

    def test_too_short(self):
        assert quality_issue("ok", "write") is not None

    def test_short_autocomplete_allowed(self):
        assert quality_issue("x + 1", "autocomplete") is None

    def test_refusal_detected(self):
        assert "refused" in quality_issue(
            "I cannot help with that request because of policies.", "write")

    def test_truncated_code_fence(self):
        content = "Here you go:\n```python\ndef f():\n    return 1"
        assert "truncated" in quality_issue(content, "write")


# ── Router-as-judge ───────────────────────────────────────────────────────────
class TestJudge:
    def test_prompt_lists_categories(self):
        p = build_judge_prompt("do the thing")
        assert "debug" in p and "architecture" in p and "do the thing" in p

    def test_parse_valid(self):
        assert parse_judge_output("Debug.") == "debug"
        assert parse_judge_output("the answer is REFACTOR") == "refactor"

    def test_parse_garbage(self):
        assert parse_judge_output("banana pancakes") is None
        assert parse_judge_output("") is None
        assert parse_judge_output(None) is None

    async def test_judge_overrides_low_confidence(self):
        # prompt with no heuristic pattern match → default 'write' at 0.5 confidence
        vague = "please take care of the thing we discussed yesterday somehow"
        judge = DummyProvider(name="ollama", tier=TIER_LOCAL, content="explain")
        worker = DummyProvider(name="groq", tier=TIER_FAST_FREE,
                               content="A detailed explanation of the thing.")
        router = EmberRouter({"ollama": judge, "groq": worker}, verbose=False)
        result = await router.route(vague)
        assert result.classification.task_type == "explain"
        assert "judge" in result.classification.reasoning
        assert judge.calls == 1

    async def test_judge_garbage_keeps_heuristic(self):
        vague = "please take care of the thing we discussed yesterday somehow"
        judge = DummyProvider(name="ollama", tier=TIER_LOCAL, content="banana")
        worker = DummyProvider(name="groq", tier=TIER_SMART_FREE)
        router = EmberRouter({"ollama": judge, "groq": worker}, verbose=False)
        result = await router.route(vague)
        assert result.classification.task_type == "write"   # heuristic default

    async def test_no_judge_available(self):
        vague = "please take care of the thing we discussed yesterday somehow"
        worker = DummyProvider(name="groq", tier=TIER_SMART_FREE)
        router = EmberRouter({"groq": worker}, verbose=False)
        result = await router.route(vague)
        assert result.classification.task_type == "write"

    async def test_confident_classification_skips_judge(self):
        judge = DummyProvider(name="ollama", tier=TIER_LOCAL, content="explain")
        worker = DummyProvider(name="groq", tier=TIER_SMART_FREE,
                               content="def f():\n    return 1  # implemented")
        router = EmberRouter({"ollama": judge, "groq": worker}, verbose=False)
        # strong multi-pattern match → high confidence → no judge call
        await router.route("fix this TypeError bug, the crash traceback is broken")
        assert judge.calls == 0


# ── Router behavior with cooldowns + streaming ────────────────────────────────
class TestRouterHardening:
    async def test_cooled_down_provider_skipped(self):
        a = DummyProvider(name="a", tier=TIER_SMART_FREE)
        b = DummyProvider(name="b", tier=TIER_SMART_FREE)
        a.health.start_cooldown(60, "quota")
        router = EmberRouter({"a": a, "b": b}, verbose=False, use_judge=False)
        result = await router.route("write a function that parses csv files")
        assert result.response.provider == "b"
        assert a.calls == 0

    async def test_streaming_tokens_flow_through(self):
        p = DummyProvider(name="s", tier=TIER_SMART_FREE,
                          content="hello world", supports_streaming=True)
        router = EmberRouter({"s": p}, verbose=False, use_judge=False)
        received = []
        result = await router.route(
            "write a function that parses csv files",
            on_token=received.append,
        )
        assert result.response.success
        assert "".join(received) == "hello world"

    async def test_streaming_falls_back_for_non_streaming_provider(self):
        p = NoStreamProvider(name="ns", tier=TIER_SMART_FREE)
        router = EmberRouter({"ns": p}, verbose=False, use_judge=False)
        result = await router.route(
            "write a function that parses csv files",
            on_token=lambda t: None,
        )
        assert result.response.success
        assert p.calls == 1

    async def test_low_quality_rotates_provider(self):
        refuser = DummyProvider(name="refuser", tier=TIER_SMART_FREE,
                                content="I cannot help with that request.")
        helper = DummyProvider(name="helper", tier=TIER_SMART_FREE,
                               content="def parse(s):\n    return s.split(',')")
        router = EmberRouter({"refuser": refuser, "helper": helper},
                             verbose=False, use_judge=False)
        result = await router.route("write a function that parses csv files")
        assert result.response.provider == "helper"
        assert result.attempts == 2
