"""Phase 8 tests — AHE evolution loop (prompt store, guardrails, promotion)
and plan-first mode."""
import json
from types import SimpleNamespace

import pytest

from emberforge.agent import AGENT_SYSTEM_PROMPT
from emberforge.evolve import (
    EvolveResult, PromptEvolver, PromptStore, validate_variant, MAX_PROMPT_TOKENS,
)
from emberforge.memory import EmberMemory

from tests.integration.test_agent_loop import MockProvider, answer

GOOD_VARIANT = (
    "You are EmberForge, a coding agent. Explore with grep_search and "
    "read_file (signatures first), change code with edit_file or write_file, "
    "verify with run_shell. Finish with a plain-text summary and no tool call. "
    "Never repeat a failing call unchanged."
)


class FakeRouter:
    def __init__(self, content, success=True):
        self.content, self.success, self.calls = content, success, 0

    async def route(self, prompt, system="", **kwargs):
        self.calls += 1
        return SimpleNamespace(response=SimpleNamespace(
            success=self.success, content=self.content))


# ── Variant guardrails ────────────────────────────────────────────────────────
class TestValidation:
    def test_default_prompt_is_valid(self):
        assert validate_variant(AGENT_SYSTEM_PROMPT) is None

    def test_good_variant_valid(self):
        assert validate_variant(GOOD_VARIANT) is None

    def test_too_long_rejected(self):
        long = GOOD_VARIANT + " padding" * MAX_PROMPT_TOKENS
        assert "tokens" in validate_variant(long)

    def test_missing_tool_rejected(self):
        assert "run_shell" in validate_variant(
            "Use read_file and edit_file wisely.")

    def test_empty_rejected(self):
        assert validate_variant("  ") is not None


# ── PromptStore ───────────────────────────────────────────────────────────────
class TestPromptStore:
    def test_default_is_builtin(self, tmp_path):
        assert PromptStore(tmp_path).get_active() == AGENT_SYSTEM_PROMPT

    def test_promote_and_reload(self, tmp_path):
        store = PromptStore(tmp_path)
        store.promote(GOOD_VARIANT, {"variant_pass": 0.8})
        assert store.get_active() == GOOD_VARIANT
        assert store.history()[0]["variant_pass"] == 0.8

    def test_second_promote_archives_previous(self, tmp_path):
        store = PromptStore(tmp_path)
        store.promote(GOOD_VARIANT, {})
        store.promote(GOOD_VARIANT + " v2", {})
        retired = list(store.dir.glob("retired-*.txt"))
        assert len(retired) == 1
        assert retired[0].read_text(encoding="utf-8") == GOOD_VARIANT
        assert len(store.history()) == 2

    def test_corrupt_active_falls_back(self, tmp_path):
        store = PromptStore(tmp_path)
        store.dir.mkdir(parents=True, exist_ok=True)
        store.active_path.write_text("no tools mentioned here", encoding="utf-8")
        assert store.get_active() == AGENT_SYSTEM_PROMPT   # invalid → builtin


# ── Proposal ──────────────────────────────────────────────────────────────────
class TestPropose:
    async def test_valid_proposal_returned(self):
        router = FakeRouter(GOOD_VARIANT)
        out = await PromptEvolver().propose(AGENT_SYSTEM_PROMPT, ["a failure"], router)
        assert out == GOOD_VARIANT

    async def test_invalid_proposal_rejected(self):
        router = FakeRouter("short prompt with no tool names")
        assert await PromptEvolver().propose(
            AGENT_SYSTEM_PROMPT, ["a failure"], router) is None

    async def test_no_analyses_no_proposal(self):
        router = FakeRouter(GOOD_VARIANT)
        assert await PromptEvolver().propose(AGENT_SYSTEM_PROMPT, [], router) is None
        assert router.calls == 0


# ── The evolution decision ────────────────────────────────────────────────────
def make_scorer(baseline_score, variant_score):
    async def scorer(prompt_text):
        return baseline_score if prompt_text == AGENT_SYSTEM_PROMPT else variant_score
    return scorer


@pytest.fixture
def memory(tmp_path):
    m = EmberMemory(tmp_path / "mem.db")
    fid = m.log_failure("proj", "fix parser", "groq", "budget")
    m.update_failure_analysis(fid, "explored too much | fix: grep first")
    return m


class TestEvolve:
    async def test_better_variant_promoted(self, memory, tmp_path):
        store = PromptStore(tmp_path / "prompts")
        result = await PromptEvolver().evolve(
            make_scorer((0.5, 10_000), (0.8, 9_000)),
            FakeRouter(GOOD_VARIANT), memory, "proj", store)
        assert result.promoted
        assert result.reason == "higher pass rate"
        assert store.get_active() == GOOD_VARIANT

    async def test_worse_variant_rejected(self, memory, tmp_path):
        store = PromptStore(tmp_path / "prompts")
        result = await PromptEvolver().evolve(
            make_scorer((0.8, 10_000), (0.5, 2_000)),   # cheaper but WORSE
            FakeRouter(GOOD_VARIANT), memory, "proj", store)
        assert not result.promoted
        assert store.get_active() == AGENT_SYSTEM_PROMPT   # untouched

    async def test_equal_pass_fewer_tokens_promoted(self, memory, tmp_path):
        store = PromptStore(tmp_path / "prompts")
        result = await PromptEvolver().evolve(
            make_scorer((0.8, 10_000), (0.8, 7_000)),
            FakeRouter(GOOD_VARIANT), memory, "proj", store)
        assert result.promoted
        assert "fewer tokens" in result.reason

    async def test_no_analyses_skips_gracefully(self, tmp_path):
        empty_memory = EmberMemory(tmp_path / "m2.db")
        store = PromptStore(tmp_path / "prompts")
        result = await PromptEvolver().evolve(
            make_scorer((0.5, 1_000), (0.9, 1_000)),
            FakeRouter(GOOD_VARIANT), empty_memory, "proj", store)
        assert not result.promoted
        assert "no failure analyses" in result.reason


# ── Prompt injection through the stack ────────────────────────────────────────
class TestPromptThreading:
    async def test_agent_uses_custom_prompt(self, tmp_path):
        from emberforge.agent import EmberAgent
        from emberforge.tools import EmberTools, ToolExecutor
        provider = MockProvider([answer("done")])
        agent = EmberAgent(
            providers={"mock": provider},
            executor=ToolExecutor(EmberTools(tmp_path), auto_approve=True),
            verbose=False, system_prompt=GOOD_VARIANT,
        )
        await agent.run("anything")
        assert provider.requests[0]["messages"][0]["content"] == GOOD_VARIANT

    async def test_eval_runner_threads_prompt(self, tmp_path):
        import shutil
        from emberforge.evals import EvalRunner
        from emberforge.evals.tasks import get_task
        provider = MockProvider([answer("did nothing")])
        runner = EvalRunner({"mock": provider}, system_prompt=GOOD_VARIANT)
        result = await runner.run_task(get_task("document-install"))
        assert provider.requests[0]["messages"][0]["content"] == GOOD_VARIANT
        shutil.rmtree(result.sandbox, ignore_errors=True)

    def test_core_picks_up_promoted_prompt(self, tmp_path, monkeypatch):
        import emberforge.config.settings as settings
        monkeypatch.setattr(settings, "EMBERFORGE_HOME", tmp_path / "home")
        monkeypatch.setattr(settings, "SKILLS_DIR", tmp_path / "home" / "skills")
        monkeypatch.setattr(settings, "LOG_FILE", tmp_path / "home" / "log")
        PromptStore(tmp_path / "home").promote(GOOD_VARIANT, {})

        from emberforge.config.settings import EmberConfig, MemoryConfig
        from emberforge.core import Ember
        config = EmberConfig(memory=MemoryConfig(path=str(tmp_path / "mem.db")))
        ember = Ember(project="p", repo_path=tmp_path, verbose=False, config=config)
        assert ember.create_agent().system_prompt == GOOD_VARIANT


# ── Plan-first mode ───────────────────────────────────────────────────────────
class TestPlanFirst:
    @pytest.fixture
    def ember(self, tmp_path, monkeypatch):
        import emberforge.config.settings as settings
        monkeypatch.setattr(settings, "EMBERFORGE_HOME", tmp_path / "home")
        monkeypatch.setattr(settings, "SKILLS_DIR", tmp_path / "home" / "skills")
        monkeypatch.setattr(settings, "LOG_FILE", tmp_path / "home" / "log")
        from emberforge.config.settings import EmberConfig, MemoryConfig
        from emberforge.core import Ember
        config = EmberConfig(memory=MemoryConfig(path=str(tmp_path / "mem.db")))
        return Ember(project="p", repo_path=tmp_path, verbose=False, config=config)

    async def test_plan_task_returns_plan(self, ember):
        ember._router = FakeRouter("1. read the file\n2. fix it\n3. run tests")
        plan = await ember.plan_task("fix the bug")
        assert plan.startswith("1.")

    async def test_plan_task_failure_returns_empty(self, ember):
        ember._router = FakeRouter("", success=False)
        assert await ember.plan_task("fix the bug") == ""

    async def test_approved_plan_injected(self, ember):
        from emberforge.agent import AgentResult

        class StubAgent:
            async def run(self, task, context=""):
                self.context = context
                return AgentResult(content="done", success=True, steps=1,
                                   tool_calls_made=0, provider="m", model="m")

        stub = StubAgent()
        await ember.run_agent("fix it", agent=stub,
                              plan="1. read\n2. edit\n3. verify")
        assert "<approved-plan>" in stub.context
        assert "2. edit" in stub.context
