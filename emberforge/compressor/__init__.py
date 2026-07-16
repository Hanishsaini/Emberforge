"""
EMBERFORGE Compressor — content-aware token compression pipeline.
Inspired by Headroom's CCR, Claw-Compactor's Fusion Pipeline, LeanCTX's read modes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from emberforge.compressor.shell import ShellCompressor, CompressResult
from emberforge.compressor.ast_compress import ASTCompressor, ASTResult
from emberforge.compressor.polyglot import PolyglotCompressor
from emberforge.compressor.tokens import count_tokens


@dataclass
class PipelineResult:
    final_text:       str
    original_tokens:  int   # estimated
    final_tokens:     int   # estimated
    stages_applied:   list[str]

    @property
    def reduction_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return round(100 * (1 - self.final_tokens / self.original_tokens), 1)

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.final_tokens


# Accurate token counting (tiktoken when available, chars/4 fallback)
def _est_tokens(text: str) -> int:
    return count_tokens(text)


class EmberCompressor:
    """
    Main compression pipeline. Chains stages in order:
      1. Shell output compression (git/pip/npm/pytest)
      2. JSON deduplication / sampling
      3. AST signature extraction (Python code)
      4. Simhash-based deduplication across chunks
      5. Generic blank-line collapse

    Usage:
        compressor = EmberCompressor()
        result = compressor.compress(text, content_type="auto")
        print(result.final_text)
        print(f"Saved {result.reduction_pct}% tokens")
    """

    def __init__(self) -> None:
        self._shell    = ShellCompressor()
        self._ast      = ASTCompressor()
        self._polyglot = PolyglotCompressor()

        # Simhash cache for deduplication
        self._seen_hashes: set[int] = set()

    def compress(
        self,
        text:         str,
        content_type: str = "auto",
        filename:     str = "",
        mode:         str = "signatures",
    ) -> PipelineResult:
        """
        Full compression pipeline. content_type: auto|code|shell|json|text
        """
        original_tokens = _est_tokens(text)
        stages: list[str] = []
        current = text

        detected = self._detect_content_type(text, content_type, filename)

        # Stage 1: Shell output compression
        if detected == "shell":
            result = self._shell.compress(current)
            current = result.compressed
            stages.append(f"shell({result.reduction_pct}%)")

        # Stage 2: Code → signatures (Python AST; JS/TS/Go/Rust/Java via polyglot)
        elif detected == "code" and mode != "full":
            lang = filename.split(".")[-1].lower() if filename else "py"
            if lang == "py":
                result = self._ast.compress(current, mode=mode, filename=filename or "code.py")
                current = result.compressed
                stages.append(f"ast({result.reduction_pct}%)")
            elif self._polyglot.supports(filename):
                p_result = self._polyglot.compress(current, filename)
                current = p_result.compressed
                stages.append(f"polyglot-{p_result.language}({p_result.reduction_pct}%)")

        # Stage 3: JSON compression
        elif detected == "json":
            current, saved = self._compress_json(current)
            stages.append(f"json({saved}%)")

        # Stage 4: Simhash deduplication (always runs)
        current, dedup_removed = self._simhash_dedup(current)
        if dedup_removed > 0:
            stages.append(f"dedup({dedup_removed} chunks)")

        # Stage 5: Generic cleanup (always runs)
        current = self._generic_cleanup(current)
        stages.append("cleanup")

        final_tokens = _est_tokens(current)

        return PipelineResult(
            final_text=current,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            stages_applied=stages,
        )

    def reset_dedup_cache(self) -> None:
        """Call between sessions to clear simhash dedup cache."""
        self._seen_hashes.clear()

    # ── Content type detection ────────────────────────────────────────────────
    def _detect_content_type(
        self, text: str, hint: str, filename: str
    ) -> str:
        if hint != "auto":
            return hint

        # Filename extension wins
        if filename:
            ext = filename.split(".")[-1].lower()
            if ext in ("py", "js", "ts", "go", "rs", "java", "cpp", "c"):
                return "code"
            if ext == "json":
                return "json"

        # Content sniffing
        sample = text[:500]
        if any(k in sample for k in ("def ", "class ", "import ", "function ")):
            return "code"
        if sample.strip().startswith(("{", "[")):
            return "json"
        if any(k in sample for k in ("On branch", "modified:", "npm install", "pip install", "PASSED", "FAILED")):
            return "shell"

        return "text"

    # ── JSON compressor ───────────────────────────────────────────────────────
    def _compress_json(self, text: str) -> tuple[str, float]:
        """Statistical sampling for large JSON — keep structure, sample values."""
        import json
        try:
            data = json.loads(text)
        except Exception:
            return text, 0.0

        original_len = len(text)

        # If it's a large list, sample first 5 + last 1
        if isinstance(data, list) and len(data) > 10:
            sampled = data[:5]
            compressed = json.dumps(
                {"__sampled__": f"{len(data)} items, showing 5", "items": sampled},
                indent=2,
            )
            pct = round(100 * (1 - len(compressed) / original_len), 1)
            return compressed, pct

        # If nested dict, collapse leaf values > 100 chars
        if isinstance(data, dict):
            data = self._truncate_long_values(data)
            compressed = json.dumps(data, indent=2)
            pct = round(100 * (1 - len(compressed) / original_len), 1)
            return compressed, max(0.0, pct)

        return text, 0.0

    def _truncate_long_values(self, obj: object, max_len: int = 80) -> object:
        if isinstance(obj, dict):
            return {k: self._truncate_long_values(v, max_len) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._truncate_long_values(i, max_len) for i in obj[:10]]
        if isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + f"...[{len(obj)-max_len} chars]"
        return obj

    # ── Simhash deduplication ─────────────────────────────────────────────────
    def _simhash(self, text: str, n: int = 64) -> int:
        """Lightweight simhash — detect near-duplicate chunks."""
        words = re.findall(r'\w+', text.lower())
        v = [0] * n
        for word in words:
            h = hash(word)
            for i in range(n):
                if h & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1
        return sum(1 << i for i in range(n) if v[i] > 0)

    def _hamming(self, a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    def _simhash_dedup(self, text: str, threshold: int = 5) -> tuple[str, int]:
        """Remove near-duplicate paragraphs (hamming distance < threshold)."""
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        kept, removed = [], 0

        for chunk in chunks:
            if len(chunk) < 50:   # short chunks → always keep
                kept.append(chunk)
                continue
            h = self._simhash(chunk)
            is_dup = any(self._hamming(h, seen) < threshold for seen in self._seen_hashes)
            if is_dup:
                removed += 1
            else:
                self._seen_hashes.add(h)
                kept.append(chunk)

        return "\n\n".join(kept), removed

    # ── Generic cleanup ───────────────────────────────────────────────────────
    def _generic_cleanup(self, text: str) -> str:
        # Collapse 3+ newlines → 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip trailing whitespace per line
        text = "\n".join(line.rstrip() for line in text.splitlines())
        return text
