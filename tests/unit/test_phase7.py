"""Phase 7 tests — GEPA failure analysis, lazy skills (Pi steal), trace file."""
import json
from types import SimpleNamespace

import pytest

from emberforge.gepa import FailureAnalyst, build_trace
from emberforge.memory import EmberMemory, Skill
from emberforge.skills import skill_summary
from emberforge.tools import (
    EmberTools, ToolExecutor, ExecutionRecord, TOOL_SCHEMAS,
)


@pytest.fixture
def memory(tmp_path):
    return EmberMemory(tmp_path / "mem.db")


class FakeRouter:
    def __init__(self, content, success=True):
        self.content = content
        self.success = success
        self.calls = 0

    async def route(self, prompt, system="", **kwargs):
        self.calls += 1
        self.last_prompt = prompt
        return SimpleNamespace(response=SimpleNamespace(
            success=self.success, content=self.content))


# ── build_trace ───────────────────────────────────────────────────────────────
class TestBuildTrace:
    def test_empty(self):
        assert "no tool calls" in build_trace([])

    def test_records_rendered(self):
        history = [
            ExecutionRecord(tool="read_file", args={"path": "a.py"},
                            success=True, output_preview="    1| def f():"),
            ExecutionRecord(tool="edit_file", args={"path": "a.py"},
                            success=False, output_preview="old_string not found"),
        ]
        trace = build_trace(history)
        assert "read_file" in trace and "ok" in trace
        assert "edit_file" in trace and "FAILED" in trace
        assert "old_string not found" in trace

    def test_caps_length(self):
        history = [
            ExecutionRecord(tool="read_file", args={"path": f"f{i}.py"},
                            success=True, output_preview="x" * 120)
            for i in range(50)
        ]
        assert len(build_trace(history)) <= 1800


# ── FailureAnalyst ────────────────────────────────────────────────────────────
class TestAnalyst:
    async def test_good_json(self):
        router = FakeRouter(json.dumps({
            "root_cause": "edit_file old_string never matched",
            "fix_hint": "read the file first and copy exact whitespace",
        }))
        out = await FailureAnalyst().analyze("fix bug", "budget", "trace", router)
        assert "old_string never matched" in out
        assert "fix:" in out

    async def test_fenced_json(self):
        router = FakeRouter('```json\n{"root_cause": "quota", "fix_hint": "wait"}\n```')
        out = await FailureAnalyst().analyze("t", "e", "tr", router)
        assert out == "quota | fix: wait"

    async def test_garbage_returns_none(self):
        assert await FailureAnalyst().analyze("t", "e", "tr", FakeRouter("lol no")) is None

    async def test_router_failure_returns_none(self):
        router = FakeRouter("", success=False)
        assert await FailureAnalyst().analyze("t", "e", "tr", router) is None

    async def test_missing_root_cause_returns_none(self):
        router = FakeRouter(json.dumps({"fix_hint": "only a hint"}))
        assert await FailureAnalyst().analyze("t", "e", "tr", router) is None


# ── skill_summary ─────────────────────────────────────────────────────────────
class TestSkillSummary:
    def test_extracts_when_to_use(self):
        content = "# T\n\n## When to Use\nWhen parsers crash on None input.\n\n## Approach\n..."
        assert skill_summary({"title": "T", "content": content}) == \
            "When parsers crash on None input."

    def test_falls_back_to_title(self):
        assert skill_summary({"title": "Router cooldowns", "content": "no sections"}) == \
            "Router cooldowns"


# ── Lazy skills at the executor ───────────────────────────────────────────────
class TestLazySkills:
    def test_schema_only_with_loader(self, tmp_path):
        tools = EmberTools(tmp_path)
        plain = ToolExecutor(tools, auto_approve=True)
        lazy = ToolExecutor(tools, auto_approve=True, skill_loader=lambda n: "content")
        assert all(s["function"]["name"] != "load_skill" for s in plain.schemas)
        assert any(s["function"]["name"] == "load_skill" for s in lazy.schemas)
        assert len(plain.schemas) == len(TOOL_SCHEMAS)

    def test_load_skill_returns_content(self, tmp_path):
        ex = ToolExecutor(EmberTools(tmp_path), auto_approve=True,
                          skill_loader=lambda n: f"FULL INSTRUCTIONS for {n}")
        r = ex.execute("load_skill", json.dumps({"name": "parser fixes"}))
        assert r.success
        assert "FULL INSTRUCTIONS for parser fixes" in r.output

    def test_load_skill_without_loader(self, tmp_path):
        ex = ToolExecutor(EmberTools(tmp_path), auto_approve=True)
        r = ex.execute("load_skill", json.dumps({"name": "x"}))
        assert not r.success

    def test_load_skill_missing_name(self, tmp_path):
        ex = ToolExecutor(EmberTools(tmp_path), auto_approve=True,
                          skill_loader=lambda n: "c")
        r = ex.execute("load_skill", "{}")
        assert not r.success


# ── Failure analysis persistence ──────────────────────────────────────────────
def test_update_failure_analysis_recalled(memory):
    fid = memory.log_failure("proj", "fix the router quota bug", "groq", "HTTP 429")
    memory.update_failure_analysis(fid, "hit free-tier cap | fix: wait for reset")
    hits = memory.similar_failures("fix the router quota handling", "proj")
    assert hits[0]["analysis"] == "hit free-tier cap | fix: wait for reset"


# ── End-to-end through Ember.run_agent ───────────────────────────────────────
class TestPhase7Integration:
    @pytest.fixture
    def ember(self, tmp_path, monkeypatch):
        import emberforge.config.settings as settings
        monkeypatch.setattr(settings, "EMBERFORGE_HOME", tmp_path / "home")
        monkeypatch.setattr(settings, "SKILLS_DIR", tmp_path / "home" / "skills")
        monkeypatch.setattr(settings, "LOG_FILE", tmp_path / "home" / "log")
        from emberforge.config.settings import EmberConfig, MemoryConfig
        from emberforge.core import Ember
        config = EmberConfig(memory=MemoryConfig(path=str(tmp_path / "mem.db")))
        return Ember(project="proj", repo_path=tmp_path, verbose=False, config=config)

    def _stub(self, success=True):
        from emberforge.agent import AgentResult

        class StubAgent:
            def __init__(self):
                self.context = None

            async def run(self, task, context=""):
                self.context = context
                return AgentResult(
                    content="done" if success else "gave up",
                    success=success, steps=2, tool_calls_made=1,
                    files_changed=[], provider="mock", model="m",
                    tokens_in=5, tokens_out=5, latency_ms=3,
                    error="" if success else "step_budget_exhausted",
                )
        return StubAgent()

    async def test_skills_are_lazy_in_context(self, ember):
        ember._memory.save_skill(Skill(
            title="AST compression tricks", task_type="write",
            content="# AST compression tricks\n\n## When to Use\nWhen compressing "
                    "python files.\n\n## Approach\nuse ast.parse with SECRET-SAUCE",
        ))
        stub = self._stub(success=True)
        await ember.run_agent("compress AST python files", agent=stub)
        assert "<skills-available>" in stub.context
        assert "AST compression tricks" in stub.context
        assert "When compressing python files." in stub.context
        assert "SECRET-SAUCE" not in stub.context           # full content stays out
        assert "load_skill" in stub.context                 # the escape hatch is advertised

    async def test_failed_run_gets_analyzed(self, ember):
        ember._providers = {"fake": object()}               # unlock the analysis path
        ember._router = FakeRouter(json.dumps({
            "root_cause": "ran out of steps exploring",
            "fix_hint": "grep before reading whole files",
        }))
        stub = self._stub(success=False)
        await ember.run_agent("refactor the giant module", agent=stub)
        failures = ember._memory.recent_failures("proj")
        assert failures[0]["analysis"].startswith("ran out of steps")
        # and the analysis comes back as a warning on the next similar task
        stub2 = self._stub(success=True)
        await ember.run_agent("refactor the giant module again", agent=stub2)
        assert "analysis: ran out of steps" in stub2.context

    async def test_trace_file_written(self, ember, tmp_path):
        stub = self._stub(success=True)
        await ember.run_agent("do a thing", agent=stub)
        trace_path = tmp_path / "home" / "last_trace.json"
        assert trace_path.exists()
        data = json.loads(trace_path.read_text(encoding="utf-8"))
        assert data["prompt"] == "do a thing"
        assert data["success"] is True
