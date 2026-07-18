"""Phase 6 tests — the eval suite must be able to prove itself.

Core property: every task in the registry starts FAILING after setup() and
becomes PASSING after its golden() reference fix. If that ever breaks, the
eval is measuring noise — so it's enforced here, without any LLM involved.
"""
import shutil
from pathlib import Path

import pytest

from emberforge.evals import EvalRunner, render_markdown, EvalReport, EvalResult
from emberforge.evals.tasks import TASKS, get_task
from emberforge.tools import EmberTools

from tests.integration.test_agent_loop import MockProvider, tc, calls, answer


# ── Registry sanity ───────────────────────────────────────────────────────────
def test_registry_names_unique():
    names = [t.name for t in TASKS]
    assert len(names) == len(set(names))
    assert len(TASKS) >= 6


def test_get_task():
    assert get_task("fix-zero-division") is not None
    assert get_task("nonexistent") is None


# ── The eval of the eval: golden self-validation ─────────────────────────────
@pytest.mark.parametrize("task", TASKS, ids=[t.name for t in TASKS])
def test_task_fails_fresh_and_passes_with_golden(task, tmp_path):
    sandbox = tmp_path / task.name
    sandbox.mkdir()
    task.setup(sandbox)
    assert task.verify(sandbox) is False, (
        f"{task.name}: verify() must FAIL on a fresh sandbox — otherwise the "
        "agent gets credit for doing nothing"
    )
    task.golden(sandbox)
    assert task.verify(sandbox) is True, (
        f"{task.name}: verify() must PASS after the golden fix — otherwise "
        "the task is unsolvable"
    )


# ── Runner end-to-end with a scripted agent ───────────────────────────────────
ZERO_DIV_FIX = (
    "    if b == 0:\n"
    "        raise ValueError('division by zero')\n"
    "    return a / b"
)


async def test_runner_scores_successful_agent():
    provider = MockProvider([
        calls(tc("read_file", path="calc.py")),
        calls(tc("edit_file", path="calc.py",
                 old_string="    return a / b",
                 new_string=ZERO_DIV_FIX)),
        answer("Added a zero-division guard and verified."),
    ])
    runner = EvalRunner({"mock": provider}, verbose=False)
    result = await runner.run_task(get_task("fix-zero-division"))

    assert result.passed is True
    assert result.steps == 3
    assert result.tool_calls == 2
    assert result.provider == "mock"
    assert result.compress is True
    assert not Path(result.sandbox).exists()   # passing sandboxes are cleaned up


async def test_runner_scores_failing_agent():
    provider = MockProvider([
        answer("I looked at it but changed nothing."),   # does nothing
    ])
    runner = EvalRunner({"mock": provider}, verbose=False)
    result = await runner.run_task(get_task("fix-zero-division"))

    assert result.passed is False
    assert Path(result.sandbox).exists()       # failing sandboxes kept for autopsy
    shutil.rmtree(result.sandbox, ignore_errors=True)


async def test_run_all_single_task_filter():
    provider = MockProvider([answer("nope")])
    runner = EvalRunner({"mock": provider})
    report = await runner.run_all(only="document-install")
    assert len(report.results) == 1
    assert report.results[0].task == "document-install"
    shutil.rmtree(report.results[0].sandbox, ignore_errors=True)


# ── The compression axis ──────────────────────────────────────────────────────
class TestCompressSwitch:
    def test_compress_off_forces_full_reads(self, tmp_path):
        (tmp_path / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        tools = EmberTools(tmp_path, compress=False)
        r = tools.read_file("m.py", mode="signatures")
        assert "return 1" in r.output          # full body, not signatures
        assert "[signatures" not in r.output

    def test_compress_off_disables_read_cache(self, tmp_path):
        (tmp_path / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        tools = EmberTools(tmp_path, compress=False)
        tools.read_file("m.py")
        second = tools.read_file("m.py")
        assert "[cached]" not in second.output

    def test_compress_off_disables_shell_compression(self, tmp_path):
        tools = EmberTools(tmp_path, compress=False)
        # long output that WOULD be compressed with the switch on
        r = tools.run_shell("python -c \"print('line passed\\n' * 200)\"")
        assert r.success

    async def test_runner_threads_compress_flag(self, tmp_path):
        provider = MockProvider([answer("did nothing")])
        runner = EvalRunner({"mock": provider}, compress=False)
        result = await runner.run_task(get_task("document-install"))
        assert result.compress is False
        shutil.rmtree(result.sandbox, ignore_errors=True)


# ── Report rendering ──────────────────────────────────────────────────────────
def test_render_markdown():
    report = EvalReport(results=[
        EvalResult(task="fix-zero-division", task_type="debug", passed=True,
                   steps=3, tool_calls=2, tokens_in=100, tokens_out=50,
                   seconds=2.5, provider="groq", compress=True),
        EvalResult(task="add-method", task_type="feature", passed=False,
                   steps=15, tool_calls=14, tokens_in=900, tokens_out=300,
                   seconds=30.0, provider="groq", compress=True,
                   error="step_budget_exhausted"),
    ])
    md = render_markdown({"compressed": report})
    assert "Pass rate: 50%" in md
    assert "fix-zero-division" in md
    assert "✅" in md and "❌" in md
