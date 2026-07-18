"""
EmberForge Repo Map — Aider-inspired structural retrieval, zero new deps.

How it works:
  1. Tag extraction: every code file yields DEFINITIONS (functions/classes it
     declares — Python via stdlib ast, other languages via declaration regex)
     and REFERENCES (identifiers it uses). Cached by file mtime.
  2. Reference graph: file A gets an edge to file B for every symbol A uses
     that B defines. A function called from 20 files makes its home file
     structurally central.
  3. PageRank over that graph ranks files by centrality, with a personalization
     boost for files whose name or symbols match the current task (Aider's
     "mentioned identifiers weigh 10x").
  4. BM25 over file text catches what structure misses (comments, strings,
     config), and Reciprocal Rank Fusion merges the two rankings.
  5. render_map() emits a compact "path: symbol, symbol, ..." skeleton within
     a token budget — the agent's orientation before its first grep.
"""
from __future__ import annotations

import ast
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from emberforge.compressor.tokens import count_tokens

EXCLUDE_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
}
CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
    ".cpp", ".c", ".h", ".rb", ".php",
}
MAX_FILE_BYTES = 200_000

# generic declaration-name extractor for non-Python languages
_GENERIC_DEF_RE = re.compile(
    r"^\s*(?:export\s+)?(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:function|class|struct|interface|trait|impl|enum|type|def|func|fn)\s+"
    r"(\w+)", re.MULTILINE,
)
_IDENT_RE = re.compile(r"[A-Za-z_]\w{2,}")

_STOPWORDS = {
    "the", "and", "for", "not", "with", "this", "that", "from", "import",
    "return", "def", "class", "self", "None", "True", "False", "int", "str",
    "float", "bool", "list", "dict", "set", "tuple", "print", "len", "range",
    "function", "const", "var", "let", "new", "void", "public", "private",
}


@dataclass
class FileTags:
    mtime:  float
    defs:   set[str] = field(default_factory=set)
    refs:   Counter  = field(default_factory=Counter)
    words:  Counter  = field(default_factory=Counter)   # for BM25
    length: int      = 0


def _extract_python(source: str) -> tuple[set[str], Counter]:
    defs: set[str] = set()
    refs: Counter = Counter()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return defs, refs
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            refs[node.id] += 1
        elif isinstance(node, ast.Attribute):
            refs[node.attr] += 1
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                refs[alias.name] += 1
    return defs, refs


def _extract_generic(source: str) -> tuple[set[str], Counter]:
    defs = set(_GENERIC_DEF_RE.findall(source))
    refs = Counter(
        ident for ident in _IDENT_RE.findall(source)
        if ident not in _STOPWORDS
    )
    return defs, refs


class RepoMap:
    """Structural + lexical retrieval over one repository."""

    def __init__(self, repo_root: str | Path = "."):
        self.repo_root = Path(repo_root).resolve()
        self._cache: dict[str, FileTags] = {}

    # ── indexing ──────────────────────────────────────────────────────────────
    def _collect_files(self) -> list[Path]:
        found: list[Path] = []
        for path in sorted(self.repo_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in CODE_EXTENSIONS:
                continue
            if any(part in EXCLUDE_DIRS or part.startswith(".")
                   for part in path.relative_to(self.repo_root).parts[:-1]):
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            found.append(path)
        return found

    def _tags_for(self, path: Path) -> FileTags:
        rel = str(path.relative_to(self.repo_root))
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return FileTags(mtime=0)
        cached = self._cache.get(rel)
        if cached is not None and cached.mtime == mtime:
            return cached

        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            source = ""
        if path.suffix == ".py":
            defs, refs = _extract_python(source)
        else:
            defs, refs = _extract_generic(source)
        words = Counter(w.lower() for w in _IDENT_RE.findall(source))
        tags = FileTags(mtime=mtime, defs=defs, refs=refs,
                        words=words, length=sum(words.values()))
        self._cache[rel] = tags
        return tags

    def index(self) -> dict[str, FileTags]:
        return {
            str(p.relative_to(self.repo_root)): self._tags_for(p)
            for p in self._collect_files()
        }

    # ── ranking ───────────────────────────────────────────────────────────────
    @staticmethod
    def _keywords(query: str) -> set[str]:
        return {
            w.lower() for w in _IDENT_RE.findall(query or "")
            if w.lower() not in _STOPWORDS
        }

    def pagerank(self, query: str = "", iterations: int = 20,
                 damping: float = 0.85) -> list[tuple[str, float]]:
        idx = self.index()
        files = list(idx)
        if not files:
            return []

        # symbol -> defining files
        definers: dict[str, list[str]] = {}
        for fname, tags in idx.items():
            for d in tags.defs:
                definers.setdefault(d, []).append(fname)

        # edges: referencing file -> defining file (weight split among definers)
        out_edges: dict[str, Counter] = {f: Counter() for f in files}
        for fname, tags in idx.items():
            for symbol, n in tags.refs.items():
                for target in definers.get(symbol, []):
                    if target != fname:
                        out_edges[fname][target] += n / len(definers[symbol])

        # personalization: task-relevant files pull rank toward themselves
        keywords = self._keywords(query)
        weights: dict[str, float] = {}
        for fname, tags in idx.items():
            w = 1.0
            lower = fname.lower()
            if any(k in lower for k in keywords):
                w *= 3.0                       # filename mention
            if any(k in {d.lower() for d in tags.defs} for k in keywords):
                w *= 10.0                      # defines a mentioned symbol
            weights[fname] = w
        total_w = sum(weights.values()) or 1.0
        base = {f: weights[f] / total_w for f in files}

        rank = dict(base)
        for _ in range(iterations):
            new = {f: (1 - damping) * base[f] for f in files}
            for fname in files:
                edges = out_edges[fname]
                out_total = sum(edges.values())
                if out_total <= 0:
                    for f in files:            # dangling: spread by base
                        new[f] += damping * rank[fname] * base[f]
                else:
                    for target, w in edges.items():
                        new[target] += damping * rank[fname] * (w / out_total)
            rank = new
        return sorted(rank.items(), key=lambda kv: -kv[1])

    def bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> list[tuple[str, float]]:
        idx = self.index()
        if not idx:
            return []
        keywords = self._keywords(query)
        if not keywords:
            return []
        n_docs = len(idx)
        avg_len = (sum(t.length for t in idx.values()) / n_docs) or 1.0

        scores: dict[str, float] = {}
        for term in keywords:
            df = sum(1 for t in idx.values() if term in t.words)
            if df == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            for fname, tags in idx.items():
                tf = tags.words.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + k1 * (1 - b + b * (tags.length / avg_len))
                scores[fname] = scores.get(fname, 0.0) + idf * tf * (k1 + 1) / denom
        return sorted(scores.items(), key=lambda kv: -kv[1])

    def fused_rank(self, query: str = "", rrf_k: int = 60) -> list[str]:
        """Reciprocal Rank Fusion of structural (PageRank) + lexical (BM25)."""
        scores: dict[str, float] = {}
        for ranking in (self.pagerank(query), self.bm25(query)):
            for position, (fname, _) in enumerate(ranking):
                scores[fname] = scores.get(fname, 0.0) + 1.0 / (rrf_k + position + 1)
        return [f for f, _ in sorted(scores.items(), key=lambda kv: -kv[1])]

    # ── rendering ─────────────────────────────────────────────────────────────
    def render_map(self, query: str = "", token_budget: int = 600) -> str:
        """
        Compact repo orientation: 'path: defined symbols' lines, most relevant
        first, cut to the token budget.
        """
        idx = self.index()
        lines: list[str] = []
        used = 0
        for fname in self.fused_rank(query) or list(idx):
            tags = idx.get(fname)
            if tags is None:
                continue
            defs = ", ".join(sorted(tags.defs)[:8]) or "(no definitions)"
            line = f"{fname}: {defs}"
            cost = count_tokens(line)
            if used + cost > token_budget:
                break
            lines.append(line)
            used += cost
        return "\n".join(lines)
