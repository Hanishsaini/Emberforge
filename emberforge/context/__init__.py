"""
EMBERFORGE Context Engine — retrieves relevant codebase context before an LLM call.

Phase 9: file selection is powered by the repo map (emberforge/repomap.py) —
a PageRank over the symbol reference graph fused with BM25 via Reciprocal
Rank Fusion — instead of the old walk-everything keyword scan. Selected files
are then compressed (signatures by default) into the token budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from emberforge.compressor import EmberCompressor, PipelineResult
from emberforge.repomap import RepoMap


@dataclass
class ContextResult:
    context:         str
    files_included:  list[str]
    total_tokens:    int
    compressed:      bool
    reduction_pct:   float


class EmberContext:
    """
    Codebase context engine:
      build_context()  — compressed content of the most relevant files (run mode)
      repo_map_block() — compact structural orientation map (agent mode)
    """

    def __init__(
        self,
        repo_path:   str | Path = ".",
        max_tokens:  int        = 4000,
        compressor:  EmberCompressor | None = None,
    ):
        self.repo_path   = Path(repo_path).resolve()
        self.max_tokens  = max_tokens
        self._compressor = compressor or EmberCompressor()
        self._repomap    = RepoMap(self.repo_path)

    def build_context(
        self,
        task:     str,
        top_k:    int  = 5,
        mode:     str  = "signatures",  # signatures | full
    ) -> ContextResult:
        """Compressed context of the top-ranked files for this task."""
        ranked = self._repomap.fused_rank(task)

        chunks: list[str] = []
        files:  list[str] = []
        tokens = 0
        reductions: list[float] = []

        for rel_name in ranked[: top_k * 2]:
            path = self.repo_path / rel_name
            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            compressed = self._compress_file(source, str(path), mode)
            chunk = f"### {rel_name}\n{compressed.final_text}"
            chunk_tokens = compressed.final_tokens

            if tokens + chunk_tokens > self.max_tokens:
                break

            chunks.append(chunk)
            files.append(rel_name)
            tokens += chunk_tokens
            reductions.append(compressed.reduction_pct)

            if len(files) >= top_k:
                break

        return ContextResult(
            context="\n\n".join(chunks),
            files_included=files,
            total_tokens=tokens,
            compressed=mode != "full",
            reduction_pct=round(sum(reductions) / len(reductions), 1) if reductions else 0.0,
        )

    def repo_map_block(self, task: str = "", token_budget: int = 600) -> str:
        """Aider-style orientation map for the agent: ranked 'file: symbols' lines."""
        return self._repomap.render_map(task, token_budget=token_budget)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _compress_file(self, source: str, filename: str, mode: str) -> PipelineResult:
        if mode == "full":
            from emberforge.compressor.tokens import count_tokens
            tokens = count_tokens(source)
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
