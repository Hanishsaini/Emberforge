"""
FORGE Context Engine
Retrieves relevant codebase context before sending to LLM.
Signature-mode reads (LeanCTX-inspired): full file → signatures only → 13 tokens per re-read.
Falls back to simple file reading if CodeLore not installed.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from forge.compressor import ForgeCompressor, PipelineResult


@dataclass
class ContextResult:
    context:         str
    files_included:  list[str]
    total_tokens:    int
    compressed:      bool
    reduction_pct:   float


# Files/dirs to always exclude
EXCLUDE_PATTERNS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    "*.pyc", "*.pyo", "*.egg-info", "dist", "build",
    ".DS_Store", "*.lock", "*.min.js", "*.min.css",
}

# Extensions we'll read
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".go", ".rs", ".java",
    ".cpp", ".c", ".h", ".md", ".yaml", ".yml",
    ".json", ".toml", ".txt", ".sh",
}


class ForgeContext:
    """
    Codebase context engine. Two modes:
    1. CodeLore mode (if installed): BM25+RRF retrieval — gets RELEVANT files only
    2. Fallback mode: walk directory, score files by keyword match, compress
    """

    def __init__(
        self,
        repo_path:   str | Path = ".",
        max_tokens:  int        = 4000,
        compressor:  ForgeCompressor | None = None,
    ):
        self.repo_path  = Path(repo_path).resolve()
        self.max_tokens = max_tokens
        self._compressor = compressor or ForgeCompressor()
        self._codelore   = self._try_import_codelore()

        # Signature cache: path → (mtime, compressed_content)
        self._sig_cache: dict[str, tuple[float, str]] = {}

    def _try_import_codelore(self):
        try:
            import codelore
            return codelore
        except ImportError:
            return None

    def build_context(
        self,
        task:     str,
        top_k:    int  = 5,
        mode:     str  = "signatures",  # signatures | full | auto
    ) -> ContextResult:
        """
        Build compressed context relevant to the task.
        mode: signatures (default, fast), full (expensive), auto (smart pick)
        """
        if self._codelore:
            return self._codelore_context(task, top_k, mode)
        return self._fallback_context(task, top_k, mode)

    # ── CodeLore path ─────────────────────────────────────────────────────────
    def _codelore_context(
        self, task: str, top_k: int, mode: str
    ) -> ContextResult:
        try:
            results = self._codelore.retrieve(
                query=task,
                repo_path=str(self.repo_path),
                top_k=top_k,
            )
            chunks = []
            files  = []
            tokens = 0

            for result in results:
                file_path = result.get("file", "unknown")
                content   = result.get("content", "")
                files.append(file_path)

                compressed = self._compress_file(content, file_path, mode)
                chunk = f"### {file_path}\n{compressed.final_text}"
                chunk_tokens = compressed.final_tokens

                if tokens + chunk_tokens > self.max_tokens:
                    break

                chunks.append(chunk)
                tokens += chunk_tokens

            context = "\n\n".join(chunks)
            avg_reduction = sum(
                r.reduction_pct for r in
                [self._compress_file(r.get("content",""), r.get("file",""), mode)
                 for r in results[:len(chunks)]]
            ) / max(len(chunks), 1)

            return ContextResult(
                context=context,
                files_included=files[:len(chunks)],
                total_tokens=tokens,
                compressed=mode != "full",
                reduction_pct=avg_reduction,
            )
        except Exception as e:
            # CodeLore failed → fallback
            return self._fallback_context(task, top_k, mode)

    # ── Fallback path ─────────────────────────────────────────────────────────
    def _fallback_context(
        self, task: str, top_k: int, mode: str
    ) -> ContextResult:
        """
        Keyword-scored file selection + compression.
        Used when CodeLore not installed.
        """
        keywords = self._extract_keywords(task)
        candidates = self._score_files(keywords)

        chunks = []
        files  = []
        tokens = 0

        for file_path, _ in candidates[:top_k * 2]:
            try:
                source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            compressed = self._compress_file(source, file_path, mode)
            chunk = f"### {file_path}\n{compressed.final_text}"
            chunk_tokens = compressed.final_tokens

            if tokens + chunk_tokens > self.max_tokens:
                break

            chunks.append(chunk)
            files.append(file_path)
            tokens += chunk_tokens

            if len(files) >= top_k:
                break

        context = "\n\n".join(chunks)
        return ContextResult(
            context=context,
            files_included=files,
            total_tokens=tokens,
            compressed=mode != "full",
            reduction_pct=50.0,  # rough estimate for fallback
        )

    # ── File compression (LeanCTX signature mode) ─────────────────────────────
    def _compress_file(
        self, source: str, filename: str, mode: str
    ) -> PipelineResult:
        """
        Compress a single file.
        signature mode: AST extraction → function signatures only
        full mode: pass through unchanged
        """
        if mode == "full":
            tokens = len(source) // 4
            return PipelineResult(
                final_text=source,
                original_tokens=tokens,
                final_tokens=tokens,
                stages_applied=["passthrough"],
            )

        return self._compressor.compress(
            text=source,
            content_type="code",
            filename=filename,
            mode=mode,
        )

    def read_file_signature(self, file_path: str) -> str:
        """
        LeanCTX-style: read file in signature mode.
        Cached: re-reads cost ~13 tokens instead of 2000.
        """
        path = Path(file_path)
        if not path.exists():
            return f"# File not found: {file_path}"

        mtime = path.stat().st_mtime
        if file_path in self._sig_cache:
            cached_mtime, cached_sig = self._sig_cache[file_path]
            if cached_mtime == mtime:
                return cached_sig  # cache hit: ~13 tokens

        source    = path.read_text(encoding="utf-8", errors="ignore")
        compressed = self._compress_file(source, file_path, "signatures")
        sig       = compressed.final_text
        self._sig_cache[file_path] = (mtime, sig)
        return sig

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _extract_keywords(self, task: str) -> list[str]:
        """Extract searchable keywords from task description."""
        stop = {"the", "a", "an", "in", "is", "to", "and", "or", "for", "of", "with", "my"}
        words = re.findall(r'\b[a-zA-Z_]\w*\b', task)
        return [w for w in words if w.lower() not in stop and len(w) > 2]

    def _score_files(self, keywords: list[str]) -> list[tuple[str, float]]:
        """Score all files in repo by keyword relevance."""
        scored: list[tuple[str, float]] = []

        for root, dirs, files in os.walk(self.repo_path):
            # Prune excluded dirs in-place
            dirs[:] = [
                d for d in dirs
                if d not in EXCLUDE_PATTERNS and not d.startswith(".")
            ]

            for fname in files:
                fpath = os.path.join(root, fname)
                ext   = Path(fname).suffix.lower()
                if ext not in CODE_EXTENSIONS:
                    continue

                score = self._score_file(fpath, fname, keywords)
                if score > 0:
                    scored.append((fpath, score))

        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _score_file(
        self, fpath: str, fname: str, keywords: list[str]
    ) -> float:
        score = 0.0
        fname_lower = fname.lower()

        # Filename match is highest signal
        for kw in keywords:
            if kw.lower() in fname_lower:
                score += 3.0

        # Sample first 2000 chars of file
        try:
            sample = Path(fpath).read_text(encoding="utf-8", errors="ignore")[:2000]
            sample_lower = sample.lower()
            for kw in keywords:
                count = sample_lower.count(kw.lower())
                score += min(count * 0.5, 3.0)  # cap per-keyword contribution
        except Exception:
            pass

        return score
