"""Unit tests for emberforge.tools — every tool, every safety path."""
import sys

import pytest

from emberforge.tools import (
    EmberTools, ToolExecutor, TOOL_SCHEMAS, MUTATING_TOOLS,
)


@pytest.fixture
def repo(tmp_path):
    """A tiny fake repo."""
    (tmp_path / "app.py").write_text(
        "def greet(name):\n"
        "    return f'hello {name}'\n"
        "\n"
        "def add(a, b):\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Demo\nA demo repo.\n", encoding="utf-8")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "util.py").write_text("VALUE = 42\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def tools(repo):
    return EmberTools(repo)


@pytest.fixture
def executor(tools):
    return ToolExecutor(tools, auto_approve=True)


# ── read_file ─────────────────────────────────────────────────────────────────
class TestReadFile:
    def test_full_read_numbered(self, tools):
        r = tools.read_file("app.py")
        assert r.success
        assert "1| def greet(name):" in r.output

    def test_missing_file(self, tools):
        r = tools.read_file("nope.py")
        assert not r.success
        assert "not found" in r.output.lower()

    def test_directory_rejected(self, tools):
        r = tools.read_file("pkg")
        assert not r.success
        assert "list_dir" in r.output

    def test_signatures_mode(self, tools):
        r = tools.read_file("app.py", mode="signatures")
        assert r.success
        assert "signatures" in r.output

    def test_paging(self, repo, tools):
        (repo / "big.py").write_text(
            "\n".join(f"x{i} = {i}" for i in range(100)), encoding="utf-8"
        )
        r = tools.read_file("big.py", start_line=1, max_lines=10)
        assert "x0 = 0" in r.output
        assert "x50" not in r.output
        assert "start_line=11" in r.output

    def test_path_escape_blocked(self, tools):
        r = tools.read_file("../../etc/passwd")
        assert not r.success
        assert "escapes" in r.output


# ── write_file ────────────────────────────────────────────────────────────────
class TestWriteFile:
    def test_create(self, repo, tools):
        r = tools.write_file("new_module.py", "X = 1\n")
        assert r.success
        assert (repo / "new_module.py").read_text() == "X = 1\n"
        assert r.changed_path == "new_module.py"

    def test_create_nested(self, repo, tools):
        r = tools.write_file("deep/nested/mod.py", "Y = 2\n")
        assert r.success
        assert (repo / "deep" / "nested" / "mod.py").exists()

    def test_overwrite(self, repo, tools):
        tools.write_file("app.py", "# replaced\n")
        assert (repo / "app.py").read_text() == "# replaced\n"

    def test_escape_blocked(self, tools):
        r = tools.write_file("../evil.py", "boom")
        assert not r.success


# ── edit_file ─────────────────────────────────────────────────────────────────
class TestEditFile:
    def test_unique_edit(self, repo, tools):
        r = tools.edit_file("app.py", "return a + b", "return a + b  # sum")
        assert r.success
        assert "# sum" in (repo / "app.py").read_text()

    def test_not_found_string(self, tools):
        r = tools.edit_file("app.py", "does not exist", "x")
        assert not r.success
        assert "not found" in r.output

    def test_ambiguous_requires_replace_all(self, repo, tools):
        (repo / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
        r = tools.edit_file("dup.py", "x = 1", "x = 2")
        assert not r.success
        assert "2 times" in r.output

        r2 = tools.edit_file("dup.py", "x = 1", "x = 2", replace_all=True)
        assert r2.success
        assert (repo / "dup.py").read_text() == "x = 2\nx = 2\n"

    def test_missing_file(self, tools):
        r = tools.edit_file("ghost.py", "a", "b")
        assert not r.success


# ── list_dir ──────────────────────────────────────────────────────────────────
class TestListDir:
    def test_root(self, tools):
        r = tools.list_dir(".")
        assert r.success
        assert "app.py" in r.output
        assert "pkg/" in r.output

    def test_excludes_pycache(self, repo, tools):
        (repo / "__pycache__").mkdir()
        r = tools.list_dir(".")
        assert "__pycache__" not in r.output

    def test_missing(self, tools):
        r = tools.list_dir("nowhere")
        assert not r.success


# ── grep_search ───────────────────────────────────────────────────────────────
class TestGrep:
    def test_finds_matches(self, tools):
        r = tools.grep_search(r"def \w+")
        assert r.success
        assert "app.py:1" in r.output
        assert "app.py:4" in r.output

    def test_glob_filter(self, tools):
        r = tools.grep_search("Demo", glob="*.md")
        assert "README.md" in r.output

    def test_no_match(self, tools):
        r = tools.grep_search("zzz_nothing_zzz")
        assert "No matches" in r.output

    def test_bad_regex(self, tools):
        r = tools.grep_search("([unclosed")
        assert not r.success
        assert "Invalid regex" in r.output


# ── run_shell ─────────────────────────────────────────────────────────────────
class TestRunShell:
    def test_echo(self, tools):
        r = tools.run_shell("echo emberforge-works")
        assert r.success
        assert "emberforge-works" in r.output
        assert "[exit code 0]" in r.output

    def test_nonzero_exit(self, tools):
        cmd = "exit 3" if sys.platform != "win32" else "cmd /c exit 3"
        r = tools.run_shell(cmd)
        assert not r.success
        assert "[exit code 3]" in r.output

    @pytest.mark.parametrize("cmd", [
        "rm -rf /",
        "git push origin main --force",
        "git reset --hard HEAD~5",
        "shutdown -h now",
        "dd if=/dev/zero of=/dev/sda",
    ])
    def test_dangerous_blocked(self, tools, cmd):
        r = tools.run_shell(cmd)
        assert not r.success
        assert "blocked" in r.output.lower()


# ── ToolExecutor: dispatch + approval ─────────────────────────────────────────
class TestExecutor:
    def test_dispatch_json_args(self, executor):
        r = executor.execute("read_file", '{"path": "app.py"}')
        assert r.success

    def test_dict_args(self, executor):
        r = executor.execute("list_dir", {"path": "."})
        assert r.success

    def test_invalid_json(self, executor):
        r = executor.execute("read_file", "{bad json")
        assert not r.success
        assert "Invalid JSON" in r.output

    def test_unknown_tool(self, executor):
        r = executor.execute("teleport", "{}")
        assert not r.success
        assert "Unknown tool" in r.output

    def test_bad_arguments(self, executor):
        r = executor.execute("read_file", '{"wrong_arg": true}')
        assert not r.success

    def test_mutating_denied_without_approver(self, tools):
        ex = ToolExecutor(tools, auto_approve=False, approver=None)
        r = ex.execute("write_file", '{"path": "x.py", "content": "1"}')
        assert not r.success
        assert "approval" in r.output.lower()

    def test_approver_denies(self, tools, repo):
        ex = ToolExecutor(tools, auto_approve=False, approver=lambda n, d, p: False)
        r = ex.execute("edit_file",
                       '{"path": "app.py", "old_string": "add", "new_string": "sum"}')
        assert not r.success
        assert "denied" in r.output.lower()
        assert "def add" in (repo / "app.py").read_text()   # unchanged

    def test_approver_allows_and_sees_diff(self, tools, repo):
        seen = {}
        def approver(name, desc, preview):
            seen["name"], seen["preview"] = name, preview
            return True
        ex = ToolExecutor(tools, auto_approve=False, approver=approver)
        r = ex.execute("edit_file",
                       '{"path": "app.py", "old_string": "return a + b", '
                       '"new_string": "return b + a"}')
        assert r.success
        assert seen["name"] == "edit_file"
        assert "-    return a + b" in seen["preview"]
        assert "+    return b + a" in seen["preview"]

    def test_readonly_never_needs_approval(self, tools):
        ex = ToolExecutor(tools, auto_approve=False, approver=lambda *a: False)
        assert ex.execute("read_file", '{"path": "app.py"}').success
        assert ex.execute("grep_search", '{"pattern": "def"}').success
        assert ex.execute("list_dir", "{}").success

    def test_files_changed_tracking(self, executor):
        executor.execute("write_file", '{"path": "a.py", "content": "1"}')
        executor.execute("write_file", '{"path": "b.py", "content": "2"}')
        executor.execute("write_file", '{"path": "a.py", "content": "3"}')
        assert executor.files_changed == ["a.py", "b.py"]

    def test_history_recorded(self, executor):
        executor.execute("read_file", '{"path": "app.py"}')
        executor.execute("teleport", "{}")
        assert len(executor.history) == 2
        assert executor.history[0].success
        assert not executor.history[1].success


# ── schemas ───────────────────────────────────────────────────────────────────
def test_schemas_cover_all_tools():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert names == {
        "read_file", "write_file", "edit_file",
        "list_dir", "grep_search", "run_shell",
    }
    assert MUTATING_TOOLS <= names
