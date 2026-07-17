"""
EMBERFORGE Memory — SQLite-backed persistent session memory.
Inspired by Hanish OS memory layer + Hermes post-task skill generation.

Stores:
  - Sessions: task → response → metadata
  - Projects: per-repo context, architecture decisions
  - Skills: auto-generated post-task skill files (Hermes-style)
  - Failures: GEPA-style failure trace log for self-improvement
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path


# ── Schema ────────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT    DEFAULT 'default',
    task_type   TEXT,
    prompt      TEXT,
    response    TEXT,
    provider    TEXT,
    model       TEXT,
    tokens_in   INTEGER DEFAULT 0,
    tokens_out  INTEGER DEFAULT 0,
    tokens_saved INTEGER DEFAULT 0,
    latency_ms  INTEGER DEFAULT 0,
    success     INTEGER DEFAULT 1,
    ts          REAL    DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS projects (
    name        TEXT PRIMARY KEY,
    description TEXT DEFAULT '',
    stack       TEXT DEFAULT '',        -- JSON list
    decisions   TEXT DEFAULT '',        -- Architecture decisions log
    agents_md   TEXT DEFAULT '',        -- AGENTS.md content
    last_active REAL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS skills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    task_type   TEXT,
    project     TEXT DEFAULT 'global',
    content     TEXT NOT NULL,          -- Markdown skill file content
    source_ids  TEXT DEFAULT '',        -- JSON list of session IDs that generated this
    use_count   INTEGER DEFAULT 0,
    ts          REAL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS failures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT DEFAULT 'default',
    prompt      TEXT,
    provider    TEXT,
    error       TEXT,
    analysis    TEXT DEFAULT '',        -- GEPA: why it failed
    fixed       INTEGER DEFAULT 0,
    ts          REAL DEFAULT (unixepoch('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
    title, content, task_type,
    content='skills',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS skills_ai AFTER INSERT ON skills BEGIN
    INSERT INTO skills_fts(rowid, title, content, task_type)
    VALUES (new.id, new.title, new.content, new.task_type);
END;
"""


@dataclass
class SessionRecord:
    project:      str
    task_type:    str
    prompt:       str
    response:     str
    provider:     str
    model:        str
    tokens_in:    int  = 0
    tokens_out:   int  = 0
    tokens_saved: int  = 0
    latency_ms:   int  = 0
    success:      bool = True


@dataclass
class Skill:
    title:     str
    task_type: str
    content:   str
    project:   str   = "global"


class EmberMemory:
    """
    Persistent memory backend for EMBERFORGE.
    SQLite with FTS5 for skill search.
    """

    def __init__(self, db_path: str | Path = "~/.emberforge/memory.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ── Sessions ──────────────────────────────────────────────────────────────
    def save_session(self, record: SessionRecord) -> int:
        cur = self._conn.execute(
            """INSERT INTO sessions
               (project, task_type, prompt, response, provider, model,
                tokens_in, tokens_out, tokens_saved, latency_ms, success)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record.project, record.task_type, record.prompt, record.response,
                record.provider, record.model, record.tokens_in, record.tokens_out,
                record.tokens_saved, record.latency_ms, int(record.success),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def recent_sessions(
        self, project: str = "default", limit: int = 10
    ) -> list[dict]:
        rows = self._conn.execute(
            """SELECT task_type, prompt, response, provider, model, success, ts
               FROM sessions WHERE project=? ORDER BY ts DESC LIMIT ?""",
            (project, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Recall (Phase 4): memory that reads, not just writes ─────────────────
    def get_context_brief(
        self, project: str, max_sessions: int = 5, max_chars: int = 1600
    ) -> str:
        """
        Compact project memory injected at the start of a run: recent
        architecture decisions + one-liners of recent successful sessions.
        Empty string when there is nothing worth recalling.
        """
        parts: list[str] = []

        proj = self.get_project(project)
        decisions = (proj or {}).get("decisions") or ""
        if decisions.strip():
            last = decisions.strip().splitlines()[-5:]
            parts.append("Recent decisions:\n" + "\n".join(f"- {d}" for d in last))

        sessions = [
            s for s in self.recent_sessions(project, limit=max_sessions * 2)
            if s.get("success")
        ][:max_sessions]
        if sessions:
            lines = [
                f"- [{s['task_type']}] {' '.join(s['prompt'].split())[:90]}"
                for s in sessions
            ]
            parts.append("Recent work in this project:\n" + "\n".join(lines))

        brief = "\n\n".join(parts)
        return brief[:max_chars]

    def similar_failures(
        self, prompt: str, project: str, limit: int = 3
    ) -> list[dict]:
        """
        Past unfixed failures whose prompts share keywords with the new task —
        so the agent is warned before repeating a known dead end.
        """
        keywords = {w.lower() for w in re.findall(r"[A-Za-z_]\w{2,}", prompt)}
        if not keywords:
            return []
        rows = self._conn.execute(
            """SELECT prompt, provider, error, analysis, ts
               FROM failures WHERE project=? AND fixed=0
               ORDER BY ts DESC LIMIT 50""",
            (project,),
        ).fetchall()
        scored = []
        for r in rows:
            words = {w.lower() for w in re.findall(r"[A-Za-z_]\w{2,}", r["prompt"])}
            overlap = len(keywords & words)
            if overlap >= max(2, len(keywords) // 4):
                scored.append((overlap, dict(r)))
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[:limit]]

    def session_count(self, project: str = "default") -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE project=?", (project,)
        ).fetchone()[0]

    # ── Projects ──────────────────────────────────────────────────────────────
    def upsert_project(self, name: str, **kwargs) -> None:
        existing = self._conn.execute(
            "SELECT name FROM projects WHERE name=?", (name,)
        ).fetchone()
        if existing:
            for key, val in kwargs.items():
                self._conn.execute(
                    f"UPDATE projects SET {key}=?, last_active=unixepoch('now') WHERE name=?",
                    (val, name),
                )
        else:
            self._conn.execute(
                "INSERT INTO projects (name) VALUES (?)", (name,)
            )
            for key, val in kwargs.items():
                self._conn.execute(
                    f"UPDATE projects SET {key}=? WHERE name=?", (val, name)
                )
        self._conn.commit()

    def get_project(self, name: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM projects WHERE name=?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def log_decision(self, project: str, decision: str) -> None:
        """Append an architecture decision to the project log."""
        existing = self.get_project(project)
        if not existing:
            self.upsert_project(project)
            existing = self.get_project(project)
        current = existing.get("decisions", "") or ""
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        updated = current + f"\n[{timestamp}] {decision}"
        self.upsert_project(project, decisions=updated.strip())

    # ── Skills (Hermes-style) ─────────────────────────────────────────────────
    def save_skill(self, skill: Skill, source_ids: list[int] | None = None) -> int:
        cur = self._conn.execute(
            """INSERT INTO skills (title, task_type, project, content, source_ids)
               VALUES (?,?,?,?,?)""",
            (
                skill.title, skill.task_type, skill.project,
                skill.content,
                json.dumps(source_ids or []),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    @staticmethod
    def sanitize_fts_query(query: str) -> str:
        """
        FTS5 MATCH chokes on punctuation ('?', quotes, parens). Keep only
        alphanumeric tokens, dedupe, cap at 8 terms, OR-join.
        """
        seen: list[str] = []
        for tok in re.findall(r"[A-Za-z0-9_]{3,}", query):
            low = tok.lower()
            if low not in (s.lower() for s in seen):
                seen.append(tok)
            if len(seen) >= 8:
                break
        return " OR ".join(seen)

    def search_skills(self, query: str, limit: int = 3) -> list[dict]:
        """FTS5 full-text search over skills. Uses OR semantics for partial matches."""
        fts_query = self.sanitize_fts_query(query)
        if not fts_query:
            return []

        try:
            rows = self._conn.execute(
                """SELECT s.id, s.title, s.task_type, s.content, s.use_count
                   FROM skills_fts f
                   JOIN skills s ON s.id = f.rowid
                   WHERE skills_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            # FTS query syntax error → fallback to LIKE
            rows = self._conn.execute(
                """SELECT id, title, task_type, content, use_count
                   FROM skills WHERE title LIKE ? OR content LIKE ?
                   LIMIT ?""",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def find_similar_skill(self, title: str) -> dict | None:
        """
        Dedupe check before saving a new skill: a skill whose title shares
        >= 60% of its words with `title` counts as the same skill.
        """
        new_words = {w.lower() for w in re.findall(r"[A-Za-z0-9_]{3,}", title)}
        if not new_words:
            return None
        rows = self._conn.execute(
            "SELECT id, title, task_type, content, use_count FROM skills"
        ).fetchall()
        for r in rows:
            words = {w.lower() for w in re.findall(r"[A-Za-z0-9_]{3,}", r["title"])}
            if not words:
                continue
            overlap = len(new_words & words) / max(len(new_words | words), 1)
            if overlap >= 0.6:
                return dict(r)
        return None

    def increment_skill_use(self, skill_id: int) -> None:
        self._conn.execute(
            "UPDATE skills SET use_count=use_count+1 WHERE id=?", (skill_id,)
        )
        self._conn.commit()

    def list_skills(self, project: str = "global", limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            """SELECT id, title, task_type, use_count, ts
               FROM skills WHERE project IN (?, 'global')
               ORDER BY use_count DESC, ts DESC LIMIT ?""",
            (project, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Failures (GEPA-style) ─────────────────────────────────────────────────
    def log_failure(
        self,
        project:  str,
        prompt:   str,
        provider: str,
        error:    str,
        analysis: str = "",
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO failures (project, prompt, provider, error, analysis)
               VALUES (?,?,?,?,?)""",
            (project, prompt, provider, error, analysis),
        )
        self._conn.commit()
        return cur.lastrowid

    def recent_failures(self, project: str = "default", limit: int = 5) -> list[dict]:
        rows = self._conn.execute(
            """SELECT prompt, provider, error, analysis, ts
               FROM failures WHERE project=? AND fixed=0
               ORDER BY ts DESC LIMIT ?""",
            (project, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Stats ─────────────────────────────────────────────────────────────────
    def total_stats(self) -> dict:
        row = self._conn.execute(
            """SELECT
               COUNT(*) as calls,
               SUM(tokens_in) as tokens_in,
               SUM(tokens_out) as tokens_out,
               SUM(tokens_saved) as tokens_saved,
               AVG(latency_ms) as avg_latency
               FROM sessions WHERE success=1"""
        ).fetchone()
        return dict(row) if row else {}

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
