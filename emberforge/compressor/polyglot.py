"""
EMBERFORGE Polyglot Signature Extractor — JS / TS / Go / Rust / Java.

Regex-based, deliberately: full parsing needs per-language tree-sitter grammars
(heavy install, platform wheels). Signatures only need declaration lines, which
regexes capture reliably. If a file yields too few signatures, we return the
original so the model never loses information silently.

Upgrade path: swap _extract() for tree-sitter queries per language, keeping the
same interface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PolyglotResult:
    compressed:    str
    original:      str
    language:      str
    signature_count: int

    @property
    def reduction_pct(self) -> float:
        if not self.original:
            return 0.0
        return round(100 * (1 - len(self.compressed) / len(self.original)), 1)


# Declaration-line patterns per language.
_LANG_PATTERNS: dict[str, list[str]] = {
    "javascript": [
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*\w*\s*\([^)]*\)",
        r"^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>",
        r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+\w+",
        r"^\s*(?:static\s+)?(?:async\s+)?[\w$]+\s*\([^)]*\)\s*\{",          # methods
        r"^\s*(?:export\s+)?(?:const|let|var)\s+[A-Z_][A-Z0-9_]*\s*=",       # constants
    ],
    "typescript": [
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*\w*\s*(?:<[^>]*>)?\s*\([^)]*\)",
        r"^\s*(?:export\s+)?(?:const|let|var)\s+\w+(?:\s*:\s*[^=]+)?\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>",
        r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+\w+",
        r"^\s*(?:export\s+)?interface\s+\w+",
        r"^\s*(?:export\s+)?type\s+\w+\s*=",
        r"^\s*(?:export\s+)?enum\s+\w+",
        r"^\s*(?:public|private|protected|static|readonly|async|\s)*[\w$]+\s*(?:<[^>]*>)?\s*\([^)]*\)\s*(?::\s*[\w<>\[\]., |&]+)?\s*\{",
    ],
    "go": [
        r"^\s*func\s+(?:\(\s*\w+\s+\*?\w+\s*\)\s*)?\w+\s*\([^)]*\)",
        r"^\s*type\s+\w+\s+(?:struct|interface)",
        r"^\s*type\s+\w+\s+",
        r"^\s*(?:var|const)\s+\w+",
        r"^\s*package\s+\w+",
    ],
    "rust": [
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+\w+",
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+\w+",
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?enum\s+\w+",
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?trait\s+\w+",
        r"^\s*impl(?:<[^>]*>)?\s+\w+",
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const|static)\s+\w+",
    ],
    "java": [
        r"^\s*(?:public|private|protected|static|final|abstract|\s)*(?:class|interface|enum|record)\s+\w+",
        r"^\s*(?:public|private|protected|static|final|abstract|synchronized|\s)*[\w<>\[\], ]+\s+\w+\s*\([^)]*\)\s*(?:throws\s+[\w, ]+)?\s*\{",
        r"^\s*(?:public|private|protected|static|final|\s)*[\w<>\[\], ]+\s+[A-Z_][A-Z0-9_]*\s*=",
    ],
}

_IMPORT_PATTERNS: dict[str, str] = {
    "javascript": r"^\s*(?:import\s.+|const\s+.+=\s*require\(.+\))",
    "typescript": r"^\s*import\s.+",
    "go":         r"^\s*import\s|^\s*\"[\w./-]+\"\s*$",
    "rust":       r"^\s*use\s.+",
    "java":       r"^\s*import\s.+",
}

EXT_TO_LANG = {
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}


class PolyglotCompressor:
    """Signature extraction for non-Python languages."""

    def supports(self, filename: str) -> bool:
        return self._lang(filename) is not None

    def _lang(self, filename: str) -> str | None:
        for ext, lang in EXT_TO_LANG.items():
            if filename.lower().endswith(ext):
                return lang
        return None

    def compress(self, source: str, filename: str) -> PolyglotResult:
        lang = self._lang(filename)
        if lang is None:
            return PolyglotResult(source, source, "unknown", 0)

        decl_rx   = [re.compile(p) for p in _LANG_PATTERNS[lang]]
        import_rx = re.compile(_IMPORT_PATTERNS[lang])

        signatures: list[str] = []
        import_count = 0
        comment_buffer: list[str] = []

        for line in source.splitlines():
            stripped = line.rstrip()
            if not stripped.strip():
                comment_buffer.clear()
                continue
            if import_rx.match(stripped):
                import_count += 1
                continue
            # keep doc comments immediately above a declaration
            if re.match(r"^\s*(///|//!|/\*\*|\*|//)", stripped):
                comment_buffer.append(stripped)
                if len(comment_buffer) > 3:
                    comment_buffer.pop(0)
                continue
            if any(rx.match(stripped) for rx in decl_rx):
                if comment_buffer:
                    signatures.append(comment_buffer[-1])       # closest doc line
                sig = re.sub(r"\s*\{\s*$", "", stripped)         # drop opening brace
                signatures.append(sig)
            comment_buffer.clear()

        # Not enough structure found → return original rather than losing info
        if len(signatures) < 2:
            return PolyglotResult(source, source, lang, len(signatures))

        header = f"// [{lang} signatures — {import_count} imports collapsed]"
        compressed = header + "\n" + "\n".join(signatures)
        return PolyglotResult(compressed, source, lang, len(signatures))
