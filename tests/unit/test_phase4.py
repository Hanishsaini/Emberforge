"""Phase 4 tests — memory recall, failure warnings, skill quality gating."""
import json
from types import SimpleNamespace

import pytest

from emberforge.memory import EmberMemory, SessionRecord, Skill
from emberforge.skills import SkillGenerator


@pytest.fixture
def memory(tmp_path):
    return EmberMemory(tmp_path / "mem.db")


def make_session(prompt="fix the parser", task_type="debug", success=True,
                 response="fixed by handling None input"):
    return SessionRecord(
        project="proj", task_type=task_type, prompt=prompt,
        response=response, provider="groq", model="m", success=success,
    )


# ── Context brief (recall) ────────────────────────────────────────────────────
class TestContextBrief:
    def test_empty_project_returns_empty(self, memory):
        assert memory.get_context_brief("proj") == ""

    def test_brief_includes_decisions_and_sessions(self, memory):
        memory.upsert_project("proj")
        memory.log_decision("proj", "use SQLite FTS5 for skill search")
        memory.save_session(make_session(prompt="add cooldown handling to router"))
        brief = memory.get_context_brief("proj")
        assert "use SQLite FTS5" in brief
        assert "add cooldown handling" in brief
        assert "[debug]" in brief

    def test_failed_sessions_excluded(self, memory):
        memory.save_session(make_session(prompt="broken run", success=False))
        brief = memory.get_context_brief("proj")
        assert "broken run" not in brief

    def test_brief_is_capped(self, memory):
        memory.upsert_project("proj")
        for i in range(30):
            memory.log_decision("proj", f"decision {i}: " + "x" * 200)
        assert len(memory.get_context_brief("proj")) <= 1600


# ── Similar-failure recall ────────────────────────────────────────────────────
class TestSimilarFailures:
    def test_matching_failure_found(self, memory):
        memory.log_failure("proj", "fix the router quota bug", "groq", "HTTP 429 quota")
        hits = memory.similar_failures("fix the router quota handling", "proj")
        assert len(hits) == 1
        assert "HTTP 429" in hits[0]["error"]

    def test_unrelated_failure_ignored(self, memory):
        memory.log_failure("proj", "compress typescript signatures", "groq", "timeout")
        hits = memory.similar_failures("fix the router quota handling", "proj")
        assert hits == []

    def test_other_project_ignored(self, memory):
        memory.log_failure("other", "fix the router quota bug", "groq", "HTTP 429")
        assert memory.similar_failures("fix the router quota bug", "proj") == []


# ── FTS sanitization ──────────────────────────────────────────────────────────
class TestFtsSanitize:
    def test_punctuation_stripped(self, memory):
        q = memory.sanitize_fts_query('how do I fix "AST"? (python!)')
        assert '"' not in q and "?" not in q and "(" not in q
        assert "AST" in q and "python" in q

    def test_search_with_hostile_query_finds_skill(self, memory):
        memory.save_skill(Skill(title="AST Python compression tricks",
                                task_type="write", content="use ast.parse"))
        hits = memory.search_skills('fix my "AST"? (python) compression!!')
        assert hits and hits[0]["title"] == "AST Python compression tricks"

    def test_empty_query_returns_nothing(self, memory):
        assert memory.search_skills("?? !! ..") == []


# ── Skill dedupe + use counting ───────────────────────────────────────────────
class TestSkillDedupe:
    def test_similar_title_detected(self, memory):
        memory.save_skill(Skill(title="AST Python compression tricks",
                                task_type="write", content="c"))
        hit = memory.find_similar_skill("AST Python compression tips")
        assert hit is not None

    def test_different_title_not_matched(self, memory):
        memory.save_skill(Skill(title="AST Python compression tricks",
                                task_type="write", content="c"))
        assert memory.find_similar_skill("router cooldown strategy") is None

    def test_find_relevant_skills_bumps_use_count(self, memory):
        memory.save_skill(Skill(title="router cooldown strategy",
                                task_type="debug", content="use Retry-After"))
        gen = SkillGenerator(memory)
        gen.find_relevant_skills("how to handle router cooldown")
        hits = memory.search_skills("router cooldown")
        assert hits[0]["use_count"] == 1


# ── Skill generation gating ───────────────────────────────────────────────────
class FakeRouter:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    async def route(self, prompt, system="", **kwargs):
        self.calls += 1
        return SimpleNamespace(response=SimpleNamespace(
            success=True, content=json.dumps(self.payload)))


SKILL_JSON = {
    "title": "Null-safe parser fixes",
    "when_to_use": "when parsers crash on None",
    "approach": "1. reproduce 2. guard 3. test",
    "patterns": "if x is None: return default",
    "pitfalls": "don't swallow exceptions",
}


class TestSkillGating:
    def _sessions(self, n, task_type="debug", success=True):
        return [
            {"prompt": f"fix parser crash {i}", "response": "guarded None",
             "task_type": task_type, "success": success}
            for i in range(n)
        ]

    async def test_too_few_sessions_skips(self, memory):
        gen = SkillGenerator(memory, threshold=5)
        gen._tool_call_count = 5
        router = FakeRouter(SKILL_JSON)
        result = await gen.maybe_generate("proj", "debug", self._sessions(2), router)
        assert result is None
        assert router.calls == 0

    async def test_failed_sessions_dont_count(self, memory):
        gen = SkillGenerator(memory, threshold=5)
        gen._tool_call_count = 5
        router = FakeRouter(SKILL_JSON)
        result = await gen.maybe_generate(
            "proj", "debug", self._sessions(5, success=False), router)
        assert result is None
        assert router.calls == 0

    async def test_wrong_task_type_doesnt_count(self, memory):
        gen = SkillGenerator(memory, threshold=5)
        gen._tool_call_count = 5
        router = FakeRouter(SKILL_JSON)
        result = await gen.maybe_generate(
            "proj", "debug", self._sessions(5, task_type="write"), router)
        assert result is None

    async def test_enough_good_sessions_generates(self, memory):
        gen = SkillGenerator(memory, threshold=5)
        gen._tool_call_count = 5
        router = FakeRouter(SKILL_JSON)
        result = await gen.maybe_generate("proj", "debug", self._sessions(4), router)
        assert result is not None
        assert result.title == "Null-safe parser fixes"
        assert gen._tool_call_count == 0   # counter reset

    async def test_duplicate_skill_not_saved_twice(self, memory):
        gen = SkillGenerator(memory, threshold=5)
        router = FakeRouter(SKILL_JSON)

        gen._tool_call_count = 5
        first = await gen.maybe_generate("proj", "debug", self._sessions(4), router)
        assert first is not None

        gen._tool_call_count = 5
        second = await gen.maybe_generate("proj", "debug", self._sessions(4), router)
        assert second is None                       # deduped
        hits = memory.search_skills("null safe parser")
        assert len(hits) == 1
        assert hits[0]["use_count"] == 1            # reuse recorded


# ── End-to-end recall through Ember.run_agent ────────────────────────────────
class TestRecallIntegration:
    async def test_agent_receives_memory_and_failure_context(self, tmp_path, monkeypatch):
        import emberforge.config.settings as settings
        monkeypatch.setattr(settings, "EMBERFORGE_HOME", tmp_path / "home")
        monkeypatch.setattr(settings, "SKILLS_DIR", tmp_path / "home" / "skills")
        monkeypatch.setattr(settings, "LOG_FILE", tmp_path / "home" / "log")

        from emberforge.agent import AgentResult
        from emberforge.config.settings import EmberConfig, MemoryConfig
        from emberforge.core import Ember

        config = EmberConfig(memory=MemoryConfig(path=str(tmp_path / "mem.db")))
        ember = Ember(project="proj", repo_path=tmp_path, verbose=False, config=config)

        # seed memory
        ember._memory.log_decision("proj", "router uses cooldown-based health")
        ember._memory.save_session(make_session(prompt="add retry-after parsing"))
        ember._memory.log_failure("proj", "fix the router quota bug", "groq", "HTTP 429")

        class StubAgent:
            def __init__(self):
                self.task = None
                self.context = None

            async def run(self, task, context=""):
                self.task, self.context = task, context
                return AgentResult(
                    content="done", success=True, steps=2, tool_calls_made=3,
                    files_changed=["router.py"], provider="mock", model="m",
                    tokens_in=10, tokens_out=5, latency_ms=7,
                )

        stub = StubAgent()
        result = await ember.run_agent("fix the router quota handling", agent=stub)

        assert result.success
        assert "<memory>" in stub.context
        assert "cooldown-based health" in stub.context
        assert "<past-failures>" in stub.context
        assert "HTTP 429" in stub.context

        # the run itself became memory: session saved + decision logged
        sessions = ember._memory.recent_sessions("proj", limit=5)
        assert any(s["task_type"] == "agent" for s in sessions)
        decisions = ember._memory.get_project("proj")["decisions"]
        assert "router.py" in decisions
