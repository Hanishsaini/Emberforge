"""Unit tests for FORGE memory layer."""
import pytest
import tempfile
from pathlib import Path
from forge.memory import ForgeMemory, SessionRecord, Skill


@pytest.fixture
def mem():
    """Create a temp memory DB for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    m = ForgeMemory(db_path)
    yield m
    # Windows fix: close connection before deleting
    m._conn.close()
    import gc
    gc.collect()
    try:
        Path(db_path).unlink(missing_ok=True)
    except PermissionError:
        pass  # Windows sometimes holds the file — acceptable in CI


class TestForgeMemory:
    def test_save_and_retrieve_session(self, mem):
        record = SessionRecord(
            project="test-proj",
            task_type="debug",
            prompt="fix the router bug",
            response="Here's the fix...",
            provider="groq",
            model="llama-3.3-70b",
            tokens_in=100,
            tokens_out=200,
        )
        sid = mem.save_session(record)
        assert sid > 0

        recent = mem.recent_sessions("test-proj", limit=5)
        assert len(recent) == 1
        assert recent[0]["prompt"] == "fix the router bug"
        assert recent[0]["provider"] == "groq"

    def test_session_count(self, mem):
        for i in range(3):
            mem.save_session(SessionRecord(
                project="proj",
                task_type="write",
                prompt=f"task {i}",
                response=f"response {i}",
                provider="gemini",
                model="gemini-flash",
            ))
        assert mem.session_count("proj") == 3
        assert mem.session_count("other-proj") == 0

    def test_upsert_project(self, mem):
        mem.upsert_project("myproject", description="Test project", stack='["python"]')
        proj = mem.get_project("myproject")
        assert proj is not None
        assert proj["description"] == "Test project"

        # Update
        mem.upsert_project("myproject", description="Updated description")
        proj = mem.get_project("myproject")
        assert proj["description"] == "Updated description"

    def test_log_decision(self, mem):
        mem.upsert_project("proj")
        mem.log_decision("proj", "Use SQLite for memory backend")
        mem.log_decision("proj", "Use BM25+RRF for context retrieval")
        proj = mem.get_project("proj")
        assert "SQLite" in proj["decisions"]
        assert "BM25+RRF" in proj["decisions"]

    def test_save_and_search_skill(self, mem):
        skill = Skill(
            title="Python AST Signature Extraction",
            task_type="refactor",
            content="## When to Use\nWhen compressing Python code for LLM context...",
            project="global",
        )
        sid = mem.save_skill(skill)
        assert sid > 0

        results = mem.search_skills("AST Python compression")
        assert len(results) > 0
        assert "AST" in results[0]["title"]

    def test_list_skills(self, mem):
        for i in range(3):
            mem.save_skill(Skill(
                title=f"Skill {i}",
                task_type="debug",
                content=f"Content {i}",
                project="global",
            ))
        skills = mem.list_skills(limit=10)
        assert len(skills) == 3

    def test_log_failure(self, mem):
        fid = mem.log_failure(
            project="proj",
            prompt="broken task",
            provider="groq",
            error="HTTP 429: rate limit",
            analysis="Provider rate limit hit, should fallback faster",
        )
        assert fid > 0

        failures = mem.recent_failures("proj")
        assert len(failures) == 1
        assert failures[0]["error"] == "HTTP 429: rate limit"

    def test_total_stats(self, mem):
        mem.save_session(SessionRecord(
            project="p", task_type="write", prompt="t", response="r",
            provider="groq", model="llama",
            tokens_in=100, tokens_out=200, tokens_saved=50,
        ))
        stats = mem.total_stats()
        assert stats["calls"] == 1
        assert stats["tokens_in"] == 100
        assert stats["tokens_out"] == 200
        assert stats["tokens_saved"] == 50
