"""Unit tests for EMBERFORGE compressor pipeline."""
import pytest
from emberforge.compressor import EmberCompressor
from emberforge.compressor.shell import ShellCompressor
from emberforge.compressor.ast_compress import ASTCompressor


# ── Shell compressor tests ─────────────────────────────────────────────────────
class TestShellCompressor:
    def setup_method(self):
        self.c = ShellCompressor()

    def test_git_status_clean(self):
        text = "On branch main\nnothing to commit, working tree clean"
        result = self.c.compress(text, hint="git")
        assert "git status: clean" in result.compressed
        assert result.reduction_pct > 0

    def test_git_hash_replacement(self):
        text = "commit a3f2b1c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0\nAuthor: test"
        result = self.c.compress(text, hint="git")
        assert "a3f2b1c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0" not in result.compressed
        assert "[sha]" in result.compressed

    def test_pip_already_satisfied(self):
        lines = "\n".join([f"Requirement already satisfied: pkg{i}" for i in range(10)])
        result = self.c.compress(lines, hint="pip")
        assert "Requirement already satisfied: pkg" not in result.compressed
        assert "9 packages already satisfied" in result.compressed or "10 packages already satisfied" in result.compressed
        assert result.reduction_pct > 50

    def test_blank_line_collapse(self):
        text = "line1\n\n\n\n\nline2"
        result = self.c.compress(text, hint="generic")
        assert "\n\n\n" not in result.compressed

    def test_auto_detect_git(self):
        text = "On branch main\nmodified: emberforge/cli.py"
        result = self.c.compress(text)  # auto detect
        assert result.compressed  # should not be empty


# ── AST compressor tests ───────────────────────────────────────────────────────
class TestASTCompressor:
    def setup_method(self):
        self.c = ASTCompressor()

    def test_simple_function_signature(self):
        source = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    result = a + b
    return result
'''
        result = self.c.compress(source, mode="signatures")
        assert "def add(a: int, b: int) -> int:" in result.compressed
        assert "result = a + b" not in result.compressed
        assert result.reduction_pct > 0

    def test_class_signature(self):
        source = '''
class MyRouter(BaseRouter):
    """Routes requests to providers."""

    def __init__(self, providers: list):
        self.providers = providers
        self._health = {}

    def route(self, prompt: str) -> str:
        """Route to best provider."""
        for p in self.providers:
            if p.healthy:
                return p.complete(prompt)
        return ""
'''
        result = self.c.compress(source, mode="signatures")
        assert "class MyRouter" in result.compressed
        assert "def __init__" in result.compressed
        assert "def route" in result.compressed
        assert "self._health = {}" not in result.compressed

    def test_syntax_error_passthrough(self):
        source = "def broken(:\n    pass"
        result = self.c.compress(source, mode="signatures")
        assert result.mode == "parse_failed"
        assert result.compressed == source

    def test_cache_key_stable(self):
        source = "def foo(): pass"
        r1 = self.c.compress(source)
        r2 = self.c.compress(source)
        assert r1.cache_key == r2.cache_key

    def test_reversible(self):
        source = "def foo(x: int) -> int:\n    return x * 2\n"
        result = self.c.compress(source, mode="signatures")
        retrieved = self.c.retrieve(result.cache_key)
        assert retrieved == source

    def test_non_python_passthrough(self):
        source = "function hello() { return 'world'; }"
        result = self.c.compress(source, mode="signatures", filename="code.js")
        assert result.compressed == source
        assert result.mode == "full"


# ── Pipeline tests ─────────────────────────────────────────────────────────────
class TestEmberCompressor:
    def setup_method(self):
        self.c = EmberCompressor()

    def test_code_pipeline(self):
        source = '''
def complex_function(data: list, threshold: float = 0.5) -> dict:
    """Process data with threshold."""
    results = {}
    for item in data:
        if item > threshold:
            results[item] = item * 2
    return results
'''
        result = self.c.compress(source, content_type="code", filename="process.py")
        assert result.final_tokens < result.original_tokens
        assert "def complex_function" in result.final_text

    def test_json_list_sampling(self):
        import json
        data = [{"id": i, "name": f"item_{i}", "value": i * 10} for i in range(50)]
        text = json.dumps(data)
        result = self.c.compress(text, content_type="json")
        assert result.final_tokens < result.original_tokens
        assert "__sampled__" in result.final_text

    def test_dedup_removes_repeated_chunks(self):
        chunk = "This is a repeated paragraph about the same topic with similar content."
        text = "\n\n".join([chunk] * 5)
        result = self.c.compress(text, content_type="text")
        # Should have fewer chunks after dedup
        assert result.final_tokens < result.original_tokens

    def test_reduction_pct_calculation(self):
        source = "x = 1\n" * 100
        result = self.c.compress(source, content_type="text")
        assert 0 <= result.reduction_pct <= 100

    def test_empty_input(self):
        result = self.c.compress("", content_type="text")
        assert result.final_text == ""
