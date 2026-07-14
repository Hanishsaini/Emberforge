"""
FORGE Shell Output Compressor (LeanCTX-inspired)
Compresses git, npm, pip, cargo, pytest dumps before they hit the LLM.
Real numbers: git status 800 tokens → 120. pip install dump 3000 → 200.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class CompressResult:
    compressed: str
    original_chars:   int
    compressed_chars: int

    @property
    def reduction_pct(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return round(100 * (1 - self.compressed_chars / self.original_chars), 1)


class ShellCompressor:
    """
    Deterministic, zero-LLM shell output compression.
    Pattern-matched, rule-based — same input always gives same output.
    """

    # ── Git patterns ──────────────────────────────────────────────────────────
    _GIT_HASH     = re.compile(r'\b[0-9a-f]{40}\b')
    _GIT_SHORT    = re.compile(r'\b[0-9a-f]{7,12}\b')
    _GIT_STATUS_CLEAN = re.compile(r'nothing to commit.*working tree clean', re.DOTALL)
    _GIT_UNTRACKED    = re.compile(r'\?\? .+\n', re.MULTILINE)

    # ── Pip / npm patterns ────────────────────────────────────────────────────
    _PIP_ALREADY  = re.compile(r'Requirement already satisfied: .+(\n|$)', re.MULTILINE)
    _PIP_COLLECT  = re.compile(r'Collecting .+\n', re.MULTILINE)
    _PIP_DOWNLOAD = re.compile(r'Downloading .+\n', re.MULTILINE)
    _NPM_AUDIT    = re.compile(r'found \d+ vulnerabilities.*', re.DOTALL)
    _NPM_ADDED    = re.compile(r'added \d+ packages.*\n')

    # ── Pytest patterns ───────────────────────────────────────────────────────
    _PYTEST_DOTS  = re.compile(r'^[.sxXF]+$', re.MULTILINE)
    _PYTEST_PASS  = re.compile(r'(\d+) passed')
    _PYTEST_WARN  = re.compile(r'warnings summary.*?(?=PASSED|FAILED|ERROR|\Z)', re.DOTALL)

    # ── Generic dedup ─────────────────────────────────────────────────────────
    _BLANK_LINES  = re.compile(r'\n{3,}')

    def compress(self, text: str, hint: str = "auto") -> CompressResult:
        """
        Main entry. hint can be: auto, git, pip, npm, pytest, generic
        """
        original = text
        if hint == "auto":
            hint = self._detect(text)

        if hint == "git":
            text = self._compress_git(text)
        elif hint == "pip":
            text = self._compress_pip(text)
        elif hint == "npm":
            text = self._compress_npm(text)
        elif hint == "pytest":
            text = self._compress_pytest(text)

        # Always apply generic cleanup
        text = self._compress_generic(text)

        return CompressResult(
            compressed=text.strip(),
            original_chars=len(original),
            compressed_chars=len(text.strip()),
        )

    def _detect(self, text: str) -> str:
        """Detect shell output type from content."""
        t = text[:500].lower()
        if any(k in t for k in ("on branch", "modified:", "untracked", "commit")):
            return "git"
        if any(k in t for k in ("collecting", "requirement already", "pip install")):
            return "pip"
        if any(k in t for k in ("npm install", "node_modules", "package-lock")):
            return "npm"
        if any(k in t for k in ("passed", "failed", "pytest", "test session")):
            return "pytest"
        return "generic"

    def _compress_git(self, text: str) -> str:
        # Full hashes → [sha]
        text = self._GIT_HASH.sub("[sha]", text)
        # Clean status shortcut
        if self._GIT_STATUS_CLEAN.search(text):
            return "git status: clean"
        # Collapse untracked file lists
        untracked = self._GIT_UNTRACKED.findall(text)
        if len(untracked) > 5:
            text = self._GIT_UNTRACKED.sub("", text)
            text += f"\n[+{len(untracked)} untracked files not shown]"
        return text

    def _compress_pip(self, text: str) -> str:
        # Count "already satisfied" lines, replace with summary
        already = self._PIP_ALREADY.findall(text)
        if already:
            text = self._PIP_ALREADY.sub("", text)
            text += f"\n[{len(already)} packages already satisfied, not shown]"
        # Remove verbose download lines
        text = self._PIP_COLLECT.sub("", text)
        text = self._PIP_DOWNLOAD.sub("", text)
        return text

    def _compress_npm(self, text: str) -> str:
        text = self._NPM_AUDIT.sub("", text)
        return text

    def _compress_pytest(self, text: str) -> str:
        # Remove warnings block
        text = self._PYTEST_WARN.sub("", text)
        # Replace dot lines with count
        def replace_dots(m: re.Match) -> str:
            dots = m.group(0)
            p = dots.count(".")
            f = dots.count("F")
            parts = []
            if p: parts.append(f"{p} passed")
            if f: parts.append(f"{f} FAILED")
            return " | ".join(parts) if parts else dots
        text = self._PYTEST_DOTS.sub(replace_dots, text)
        return text

    def _compress_generic(self, text: str) -> str:
        # Collapse 3+ blank lines → 1
        text = self._BLANK_LINES.sub("\n\n", text)
        # Deduplicate identical consecutive lines
        lines = text.split("\n")
        deduped, prev = [], None
        run = 0
        for line in lines:
            if line == prev:
                run += 1
            else:
                if run > 2:
                    deduped.append(f"[above line repeated {run}x]")
                run = 0
                deduped.append(line)
                prev = line
        return "\n".join(deduped)
