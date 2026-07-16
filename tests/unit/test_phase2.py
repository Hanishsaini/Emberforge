"""Phase 2 tests — accurate tokens, polyglot signatures, progressive disclosure."""
import json

import pytest

from emberforge.compressor import EmberCompressor
from emberforge.compressor.polyglot import PolyglotCompressor
from emberforge.compressor.tokens import count_tokens, counting_is_exact
from emberforge.tools import EmberTools, ToolExecutor


# ── Token counting ────────────────────────────────────────────────────────────
class TestTokens:
    def test_empty(self):
        assert count_tokens("") == 0

    def test_monotonic(self):
        assert count_tokens("hello world " * 100) > count_tokens("hello world")

    def test_reasonable_range(self):
        # "hello world" is 2-3 tokens in any sane tokenizer/estimate
        assert 1 <= count_tokens("hello world") <= 5

    def test_fallback_when_no_tiktoken(self, monkeypatch):
        import emberforge.compressor.tokens as t
        monkeypatch.setattr(t, "_ENCODER", None)
        monkeypatch.setattr(t, "_TRIED", True)
        assert t.count_tokens("x" * 400) == 100      # chars/4
        assert not t.counting_is_exact()


# ── Polyglot signatures ───────────────────────────────────────────────────────
TS_SRC = """import { db } from './db';
import { Router } from 'express';

export interface User {
    id: string;
}

/** Loads a user. */
export async function loadUser(id: string): Promise<User> {
    const row = await db.query('select * from users where id = $1', [id]);
    if (!row) {
        throw new Error('not found');
    }
    return row;
}

export class UserService {
    async getUser(id: string): Promise<User> {
        return loadUser(id);
    }
}

export const MAX_USERS = 100;
"""

GO_SRC = """package main

import "fmt"

// Greeter says hello.
type Greeter struct {
    name string
}

func NewGreeter(name string) *Greeter {
    return &Greeter{name: name}
}

func (g *Greeter) Greet() string {
    return fmt.Sprintf("hello %s", g.name)
}

const DefaultName = "world"
"""


class TestPolyglot:
    def test_typescript_signatures(self):
        p = PolyglotCompressor()
        r = p.compress(TS_SRC, "service.ts")
        assert r.language == "typescript"
        assert "interface User" in r.compressed
        assert "loadUser" in r.compressed
        assert "class UserService" in r.compressed
        # bodies stripped
        assert "db.query" not in r.compressed
        assert r.reduction_pct > 30

    def test_go_signatures(self):
        p = PolyglotCompressor()
        r = p.compress(GO_SRC, "main.go")
        assert r.language == "go"
        assert "func NewGreeter" in r.compressed
        assert "type Greeter struct" in r.compressed
        assert "Sprintf" not in r.compressed

    def test_doc_comment_kept(self):
        p = PolyglotCompressor()
        r = p.compress(GO_SRC, "main.go")
        assert "// Greeter says hello." in r.compressed

    def test_unknown_language_passthrough(self):
        p = PolyglotCompressor()
        r = p.compress("some random text", "notes.txt")
        assert r.compressed == "some random text"

    def test_too_little_structure_returns_original(self):
        p = PolyglotCompressor()
        src = "console.log('just a script');\n"
        r = p.compress(src, "script.js")
        assert r.compressed == src   # never lose info silently

    def test_pipeline_routes_ts(self):
        c = EmberCompressor()
        result = c.compress(TS_SRC, content_type="code",
                            filename="service.ts", mode="signatures")
        assert any("polyglot-typescript" in s for s in result.stages_applied)
        assert result.final_tokens < result.original_tokens


# ── Progressive disclosure: read cache ────────────────────────────────────────
@pytest.fixture
def repo(tmp_path):
    (tmp_path / "mod.py").write_text(
        "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"
        + "\n".join(f"# padding line {i}" for i in range(40)) + "\n",
        encoding="utf-8",
    )
    return tmp_path


class TestReadCache:
    def test_second_read_is_cached_marker(self, repo):
        tools = EmberTools(repo)
        first = tools.read_file("mod.py")
        second = tools.read_file("mod.py")
        assert "def alpha" in first.output
        assert "[cached]" in second.output
        assert "def alpha" not in second.output
        # the marker is dramatically cheaper
        assert len(second.output) < len(first.output)

    def test_force_reemits(self, repo):
        tools = EmberTools(repo)
        tools.read_file("mod.py")
        forced = tools.read_file("mod.py", force=True)
        assert "def alpha" in forced.output

    def test_file_change_busts_cache(self, repo):
        tools = EmberTools(repo)
        tools.read_file("mod.py")
        (repo / "mod.py").write_text("def gamma():\n    return 3\n", encoding="utf-8")
        again = tools.read_file("mod.py")
        assert "def gamma" in again.output
        assert "[cached]" not in again.output

    def test_own_edit_busts_cache(self, repo):
        tools = EmberTools(repo)
        tools.read_file("mod.py")
        tools.edit_file("mod.py", "return 1", "return 100")
        again = tools.read_file("mod.py")
        assert "return 100" in again.output

    def test_own_write_busts_cache(self, repo):
        tools = EmberTools(repo)
        tools.read_file("mod.py")
        tools.write_file("mod.py", "NEW = True\n")
        again = tools.read_file("mod.py")
        assert "NEW = True" in again.output

    def test_different_modes_cached_separately(self, repo):
        tools = EmberTools(repo)
        tools.read_file("mod.py", mode="full")
        sig = tools.read_file("mod.py", mode="signatures")
        assert "[cached]" not in sig.output   # different request shape

    def test_invalidate_read_via_executor(self, repo):
        tools = EmberTools(repo)
        ex = ToolExecutor(tools, auto_approve=True)
        ex.execute("read_file", json.dumps({"path": "mod.py"}))
        ex.invalidate_read("mod.py")
        r = ex.execute("read_file", json.dumps({"path": "mod.py"}))
        assert "def alpha" in r.output


# ── Shell compressor upgrades (driven by benchmark findings) ─────────────────
class TestShellUpgrades:
    def test_pytest_verbose_passes_collapsed_failures_kept(self):
        from emberforge.compressor.shell import ShellCompressor
        out = "\n".join(
            [f"tests/test_a.py::test_{i} PASSED  [ {i}%]" for i in range(20)]
            + ["tests/test_b.py::test_broken FAILED  [ 99%]"]
            + [f"tests/test_c.py::test_{i} PASSED  [ {i}%]" for i in range(10)]
            + ["", "1 failed, 30 passed in 2.1s"]
        )
        r = ShellCompressor().compress(out, hint="pytest")
        assert "tests/test_a.py: 20 passed" in r.compressed
        assert "tests/test_c.py: 10 passed" in r.compressed
        # failures are signal — kept verbatim
        assert "tests/test_b.py::test_broken FAILED" in r.compressed
        assert r.reduction_pct > 60

    def test_git_hints_stripped(self):
        from emberforge.compressor.shell import ShellCompressor
        status = (
            "On branch main\n"
            "Changes not staged for commit:\n"
            '  (use "git add <file>..." to update what will be committed)\n'
            '  (use "git restore <file>..." to discard changes)\n'
            "\tmodified:   emberforge/core.py\n"
        )
        r = ShellCompressor().compress(status, hint="git")
        assert "modified:   emberforge/core.py" in r.compressed
        assert '(use "git' not in r.compressed

    def test_git_stat_lines_collapsed(self):
        from emberforge.compressor.shell import ShellCompressor
        log = (
            "commit abcd123\n"
            + "\n".join(f" emberforge/file_{i}.py | {i+1} ++--" for i in range(10))
            + "\n 10 files changed, 55 insertions(+), 20 deletions(-)\n"
        )
        r = ShellCompressor().compress(log, hint="git")
        assert "10 files changed" in r.compressed          # summary kept
        assert "emberforge/file_3.py |" not in r.compressed     # stat lines gone
        assert "stat lines collapsed" in r.compressed


# ── Benchmark smoke test ──────────────────────────────────────────────────────
def test_benchmark_runs_and_measures():
    from benchmarks.compression_bench import run_benchmark, render_markdown
    rows = run_benchmark()
    assert len(rows) >= 6
    by_cat = {r["category"] for r in rows}
    assert {"code", "shell", "json", "agent"} <= by_cat
    # every row measured something real
    for r in rows:
        assert r["original_tokens"] > 0
    # the flagship claims hold on real data
    python_row = next(r for r in rows if r["name"].startswith("Python"))
    assert python_row["reduction_pct"] > 40
    cache_row = next(r for r in rows if "read cache" in r["name"])
    assert cache_row["reduction_pct"] > 90
    md = render_markdown(rows)
    assert "| Content |" in md
