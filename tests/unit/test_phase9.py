"""Phase 9 tests — repo map: tags, reference graph, PageRank, BM25, RRF,
rendering, and the rewired context engine."""
import pytest

from emberforge.context import EmberContext
from emberforge.repomap import RepoMap, _extract_python, _extract_generic


@pytest.fixture
def repo(tmp_path):
    """A repo with a clear dependency structure:
    utils.helper is used by app, models, and main -> utils is central.
    orphan is referenced by nobody."""
    (tmp_path / "utils.py").write_text(
        "def helper(x):\n    return x * 2\n\n"
        "def format_output(y):\n    return str(y)\n",
        encoding="utf-8",
    )
    (tmp_path / "models.py").write_text(
        "from utils import helper\n\n"
        "class User:\n"
        "    def score(self):\n"
        "        return helper(10)\n",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "from utils import helper, format_output\n"
        "from models import User\n\n"
        "def run_app():\n"
        "    u = User()\n"
        "    return format_output(helper(u.score()))\n",
        encoding="utf-8",
    )
    (tmp_path / "main.py").write_text(
        "from app import run_app\n\n"
        "if __name__ == '__main__':\n"
        "    run_app()\n",
        encoding="utf-8",
    )
    (tmp_path / "orphan.py").write_text(
        "def unused_thing():\n    return 'zebra_xylophone'\n",
        encoding="utf-8",
    )
    return tmp_path


# ── Tag extraction ────────────────────────────────────────────────────────────
class TestTags:
    def test_python_defs_and_refs(self):
        defs, refs = _extract_python(
            "from utils import helper\n\n"
            "class Foo:\n"
            "    def bar(self):\n"
            "        return helper(1)\n"
        )
        assert defs == {"Foo", "bar"}
        assert refs["helper"] >= 2   # import + call

    def test_python_syntax_error_safe(self):
        defs, refs = _extract_python("def broken(:\n")
        assert defs == set() and not refs

    def test_generic_extraction(self):
        defs, refs = _extract_generic(
            "export function loadUser(id) {\n  return db.find(id);\n}\n"
            "class OrderService {}\n"
        )
        assert "loadUser" in defs
        assert "OrderService" in defs

    def test_mtime_cache(self, repo):
        rm = RepoMap(repo)
        first = rm.index()["utils.py"]
        second = rm.index()["utils.py"]
        assert first is second                       # cached object
        (repo / "utils.py").write_text("def helper(x):\n    return x\n",
                                       encoding="utf-8")
        import os, time
        os.utime(repo / "utils.py", (time.time() + 5, time.time() + 5))
        third = rm.index()["utils.py"]
        assert third is not first                    # rebuilt after change


# ── Ranking ───────────────────────────────────────────────────────────────────
class TestRanking:
    def test_central_file_outranks_orphan(self, repo):
        ranks = dict(RepoMap(repo).pagerank())
        assert ranks["utils.py"] > ranks["orphan.py"]

    def test_mention_boost(self, repo):
        ranked = [f for f, _ in RepoMap(repo).pagerank("the User model score")]
        assert ranked[0] == "models.py"              # defines User + score

    def test_bm25_finds_rare_term(self, repo):
        top = RepoMap(repo).bm25("zebra_xylophone")[0][0]
        assert top == "orphan.py"

    def test_bm25_empty_query(self, repo):
        assert RepoMap(repo).bm25("") == []

    def test_fused_rank_combines_both(self, repo):
        # 'zebra_xylophone' only exists lexically in orphan.py — BM25 must
        # be able to surface it through the fusion despite zero centrality
        fused = RepoMap(repo).fused_rank("zebra_xylophone unused_thing")
        assert fused[0] == "orphan.py"

    def test_fused_rank_structural_default(self, repo):
        fused = RepoMap(repo).fused_rank("")
        assert fused.index("utils.py") < fused.index("orphan.py")

    def test_empty_repo(self, tmp_path):
        rm = RepoMap(tmp_path)
        assert rm.pagerank() == []
        assert rm.fused_rank("anything") == []
        assert rm.render_map("x") == ""


# ── Rendering ─────────────────────────────────────────────────────────────────
class TestRenderMap:
    def test_map_lists_files_and_symbols(self, repo):
        block = RepoMap(repo).render_map()
        assert "utils.py: format_output, helper" in block
        assert "models.py:" in block and "User" in block

    def test_budget_respected(self, repo):
        tiny = RepoMap(repo).render_map(token_budget=15)
        assert len(tiny.splitlines()) <= 2

    def test_query_reorders(self, repo):
        block = RepoMap(repo).render_map("User model score")
        assert block.splitlines()[0].startswith("models.py")


# ── Context engine on fused ranking ───────────────────────────────────────────
class TestContextEngine:
    def test_relevant_file_selected_and_compressed(self, repo):
        ctx = EmberContext(repo, max_tokens=2000)
        result = ctx.build_context("fix the User score method", mode="signatures")
        assert "models.py" in result.files_included
        assert "class User:" in result.context
        assert result.total_tokens <= 2000

    def test_full_mode_passthrough(self, repo):
        ctx = EmberContext(repo, max_tokens=4000)
        result = ctx.build_context("User score", mode="full")
        assert "return helper(10)" in result.context   # bodies included
        assert result.compressed is False

    def test_repo_map_block(self, repo):
        block = EmberContext(repo).repo_map_block("User score")
        assert "models.py" in block

    def test_no_dead_codelore_path(self):
        assert not hasattr(EmberContext, "_codelore_context")


# ── Agent integration ─────────────────────────────────────────────────────────
class TestAgentIntegration:
    async def test_agent_context_contains_repo_map(self, repo, monkeypatch):
        import emberforge.config.settings as settings
        monkeypatch.setattr(settings, "EMBERFORGE_HOME", repo / ".home")
        monkeypatch.setattr(settings, "SKILLS_DIR", repo / ".home" / "skills")
        monkeypatch.setattr(settings, "LOG_FILE", repo / ".home" / "log")
        from emberforge.config.settings import EmberConfig, MemoryConfig
        from emberforge.core import Ember
        from emberforge.agent import AgentResult

        config = EmberConfig(memory=MemoryConfig(path=str(repo / ".home" / "m.db")))
        ember = Ember(project="p", repo_path=repo, verbose=False, config=config)

        class StubAgent:
            async def run(self, task, context=""):
                self.context = context
                return AgentResult(content="ok", success=True, steps=1,
                                   tool_calls_made=0, provider="m", model="m")

        stub = StubAgent()
        await ember.run_agent("fix the User score method", agent=stub)
        assert "<repo-map>" in stub.context
        assert "models.py" in stub.context
