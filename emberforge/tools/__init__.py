"""
EMBERFORGE Tools — the agent's hands.
read_file / write_file / edit_file / list_dir / grep_search / run_shell

Safety model:
  - All file paths are confined to the repo root (no ../ escapes).
  - Mutating tools (write_file, edit_file, run_shell) require approval via a
    callback unless auto_approve is set. No approver configured → denied.
  - run_shell enforces a dangerous-command blocklist even with auto_approve,
    plus a timeout. Output is compressed through ShellCompressor.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from emberforge.compressor import EmberCompressor

# ── Result type ───────────────────────────────────────────────────────────────
@dataclass
class ToolResult:
    output:  str
    success: bool = True
    # set for mutating tools so the agent/CLI can report what changed
    changed_path: str = ""


# ── Constants ─────────────────────────────────────────────────────────────────
MAX_OUTPUT_CHARS   = 8_000     # hard cap on any tool output sent to the model
MAX_READ_LINES     = 400       # default read window
MAX_GREP_MATCHES   = 60

EXCLUDE_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".egg-info",
}

# Commands that are never allowed, even with auto_approve.
DANGEROUS_SHELL = [
    r"\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)\b.*(/|~|\*)",   # rm -rf / ~ *
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r">\s*/dev/sd[a-z]",
    r"\bshutdown\b|\breboot\b|\bhalt\b",
    r"\bformat\s+[a-z]:",
    r"\b(del|rd|rmdir)\s+/s\b.*[a-z]:\\\\?\s*$",
    r":\(\)\s*\{.*\};\s*:",                                     # fork bomb
    r"\bgit\s+push\s+.*--force\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bDROP\s+(TABLE|DATABASE)\b",
]

MUTATING_TOOLS = {"write_file", "edit_file", "run_shell"}


# ── OpenAI-format tool schemas ────────────────────────────────────────────────
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file from the repository. mode='signatures' returns only "
                "function/class signatures for Python files (cheap — use first); "
                "mode='full' returns numbered source lines. Use start_line to page "
                "through large files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":       {"type": "string", "description": "Repo-relative file path"},
                    "mode":       {"type": "string", "enum": ["full", "signatures"], "default": "full"},
                    "start_line": {"type": "integer", "default": 1},
                    "max_lines":  {"type": "integer", "default": MAX_READ_LINES},
                    "force":      {"type": "boolean", "default": False,
                                   "description": "Re-emit content even if unchanged since your last read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file or fully overwrite an existing one with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace an exact string in a file. old_string must appear exactly "
                "once (or set replace_all=true). Read the file first so old_string "
                "matches exactly, including whitespace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":        {"type": "string"},
                    "old_string":  {"type": "string"},
                    "new_string":  {"type": "string"},
                    "replace_all": {"type": "boolean", "default": False},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and subdirectories at a repo-relative path ('.' = repo root).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": (
                "Regex-search file contents across the repository. Returns "
                "file:line:text matches. Use to locate where something is defined or used."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python regex"},
                    "path":    {"type": "string", "default": ".", "description": "Subdirectory to search"},
                    "glob":    {"type": "string", "default": "", "description": "Filename filter, e.g. *.py"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command in the repo root (tests, git status, linters, "
                "builds). Output is compressed. Destructive commands are blocked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "default": 60, "description": "Seconds"},
                },
                "required": ["command"],
            },
        },
    },
]


# ── Tool implementations ──────────────────────────────────────────────────────
class EmberTools:
    """All tool implementations, confined to repo_root."""

    def __init__(
        self,
        repo_root:  str | Path = ".",
        compressor: EmberCompressor | None = None,
        compress:   bool = True,
    ):
        self.repo_root = Path(repo_root).resolve()
        self._compressor = compressor or EmberCompressor()
        # Master switch for the eval suite's with/without-compression axis:
        # False disables signature reads, shell compression, and the read cache.
        self.compress = compress
        # Progressive disclosure: remember what we already sent the model.
        # (path, mode, start_line, max_lines) -> sha1 of file content at send time
        self._read_cache: dict[tuple, str] = {}

    def invalidate_read(self, path: str) -> None:
        """
        Embert cached reads for `path`. Called automatically after write/edit,
        and by the agent when history compaction truncates a read's content
        (so the model can get the full content again instead of a cache marker).
        """
        self._read_cache = {k: v for k, v in self._read_cache.items() if k[0] != path}

    # — path safety —
    def _resolve(self, rel_path: str) -> Path:
        p = (self.repo_root / rel_path).resolve()
        if not str(p).startswith(str(self.repo_root)):
            raise PermissionError(f"Path escapes repository root: {rel_path}")
        return p

    # — read —
    def read_file(
        self,
        path:       str,
        mode:       str  = "full",
        start_line: int  = 1,
        max_lines:  int  = MAX_READ_LINES,
        force:      bool = False,
    ) -> ToolResult:
        try:
            p = self._resolve(path)
        except PermissionError as e:
            return ToolResult(str(e), success=False)
        if not p.exists():
            return ToolResult(f"File not found: {path}", success=False)
        if p.is_dir():
            return ToolResult(f"{path} is a directory — use list_dir.", success=False)

        source = p.read_text(encoding="utf-8", errors="ignore")

        if not self.compress:
            mode = "full"   # compression disabled: no signature reads, no cache
        else:
            # Progressive disclosure: identical re-read of an unchanged file costs
            # a one-line marker instead of the full content (it's already in context).
            cache_key    = (path, mode, start_line, max_lines)
            content_hash = hashlib.sha1(source.encode("utf-8", "ignore")).hexdigest()
            if not force and self._read_cache.get(cache_key) == content_hash:
                return ToolResult(
                    f"[cached] {path} is unchanged since you last read it — the "
                    "content is already in this conversation. Pass force=true to re-emit."
                )
            self._read_cache[cache_key] = content_hash

        if mode == "signatures" and p.suffix == ".py":
            result = self._compressor.compress(
                source, content_type="code", filename=str(p), mode="signatures"
            )
            out = f"[signatures of {path} — use mode='full' for bodies]\n{result.final_text}"
            return ToolResult(out[:MAX_OUTPUT_CHARS])

        lines = source.splitlines()
        start = max(1, start_line)
        window = lines[start - 1 : start - 1 + max_lines]
        numbered = "\n".join(f"{i:5d}| {line}" for i, line in enumerate(window, start))
        total = len(lines)
        if start - 1 + max_lines < total:
            numbered += (
                f"\n... [{total - (start - 1 + len(window))} more lines — "
                f"call again with start_line={start + len(window)}]"
            )
        return ToolResult(numbered[:MAX_OUTPUT_CHARS] or "[empty file]")

    # — write —
    def write_file(self, path: str, content: str) -> ToolResult:
        try:
            p = self._resolve(path)
        except PermissionError as e:
            return ToolResult(str(e), success=False)
        existed = p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        self.invalidate_read(path)
        action = "Overwrote" if existed else "Created"
        return ToolResult(
            f"{action} {path} ({len(content.splitlines())} lines).",
            changed_path=path,
        )

    # — edit —
    def edit_file(
        self, path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> ToolResult:
        try:
            p = self._resolve(path)
        except PermissionError as e:
            return ToolResult(str(e), success=False)
        if not p.exists():
            return ToolResult(f"File not found: {path}", success=False)

        source = p.read_text(encoding="utf-8", errors="ignore")
        count = source.count(old_string)
        if count == 0:
            return ToolResult(
                f"old_string not found in {path}. Read the file and match exactly "
                "(including whitespace).",
                success=False,
            )
        if count > 1 and not replace_all:
            return ToolResult(
                f"old_string appears {count} times in {path}. Add more surrounding "
                "context to make it unique, or set replace_all=true.",
                success=False,
            )

        updated = source.replace(old_string, new_string) if replace_all \
            else source.replace(old_string, new_string, 1)
        p.write_text(updated, encoding="utf-8")
        self.invalidate_read(path)
        n = count if replace_all else 1
        return ToolResult(f"Edited {path} ({n} replacement(s)).", changed_path=path)

    # — list —
    def list_dir(self, path: str = ".") -> ToolResult:
        try:
            p = self._resolve(path)
        except PermissionError as e:
            return ToolResult(str(e), success=False)
        if not p.exists():
            return ToolResult(f"Directory not found: {path}", success=False)
        if not p.is_dir():
            return ToolResult(f"{path} is a file — use read_file.", success=False)

        entries = []
        for child in sorted(p.iterdir(), key=lambda c: (c.is_file(), c.name.lower())):
            if child.name in EXCLUDE_DIRS or child.name.startswith("."):
                continue
            if child.is_dir():
                entries.append(f"{child.name}/")
            else:
                entries.append(f"{child.name}  ({child.stat().st_size:,} B)")
        return ToolResult("\n".join(entries) or "[empty directory]")

    # — grep —
    def grep_search(self, pattern: str, path: str = ".", glob: str = "") -> ToolResult:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return ToolResult(f"Invalid regex: {e}", success=False)
        try:
            root = self._resolve(path)
        except PermissionError as e:
            return ToolResult(str(e), success=False)

        glob_rx = None
        if glob:
            glob_rx = re.compile(
                "^" + re.escape(glob).replace(r"\*", ".*").replace(r"\?", ".") + "$"
            )

        matches: list[str] = []
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
            for fname in files:
                if glob_rx and not glob_rx.match(fname):
                    continue
                fpath = Path(dirpath) / fname
                if fpath.stat().st_size > 1_000_000:   # skip huge/binary
                    continue
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if "\x00" in text[:1024]:              # binary
                    continue
                rel = fpath.relative_to(self.repo_root)
                for lineno, line in enumerate(text.splitlines(), 1):
                    if rx.search(line):
                        matches.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                        if len(matches) >= MAX_GREP_MATCHES:
                            matches.append(f"... [capped at {MAX_GREP_MATCHES} matches]")
                            return ToolResult("\n".join(matches))
        return ToolResult("\n".join(matches) or f"No matches for /{pattern}/")

    # — shell —
    def run_shell(self, command: str, timeout: int = 60) -> ToolResult:
        for danger in DANGEROUS_SHELL:
            if re.search(danger, command, re.IGNORECASE):
                return ToolResult(
                    f"Command blocked by safety policy: matches /{danger}/",
                    success=False,
                )
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=min(timeout, 300),
                encoding="utf-8",
                errors="ignore",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(f"Command timed out after {timeout}s: {command}", success=False)

        raw = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        raw = raw.strip() or "[no output]"

        # Compress verbose shell output (git/pip/pytest dumps) before the model sees it
        if self.compress and len(raw) > 1_500:
            compressed = self._compressor.compress(raw, content_type="shell")
            raw = compressed.final_text

        out = f"[exit code {proc.returncode}]\n{raw}"
        return ToolResult(out[:MAX_OUTPUT_CHARS], success=(proc.returncode == 0))


# ── Executor: dispatch + approval gate ────────────────────────────────────────
# approval callback: (tool_name, description, preview) -> bool
ApprovalCallback = Callable[[str, str, str], bool]


@dataclass
class ExecutionRecord:
    tool:    str
    args:    dict
    success: bool
    output_preview: str = ""


class ToolExecutor:
    """
    Dispatches a ToolCall to the right EmberTools method, gating mutating tools
    behind an approval callback. Records everything for the transcript.
    """

    def __init__(
        self,
        tools:        EmberTools,
        auto_approve: bool = False,
        approver:     ApprovalCallback | None = None,
    ):
        self._tools       = tools
        self.auto_approve = auto_approve
        self._approver    = approver
        self.history:       list[ExecutionRecord] = []
        self.files_changed: list[str] = []

    @property
    def schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    def invalidate_read(self, path: str) -> None:
        """Forward cache invalidation to the tool layer (used by the agent)."""
        self._tools.invalidate_read(path)

    def execute(self, name: str, arguments: str | dict) -> ToolResult:
        # Parse args
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments or "{}")
            except json.JSONDecodeError as e:
                return self._record(name, {}, ToolResult(
                    f"Invalid JSON arguments: {e}", success=False))
        else:
            args = dict(arguments)

        method = getattr(self._tools, name, None)
        if method is None or name.startswith("_"):
            return self._record(name, args, ToolResult(
                f"Unknown tool: {name}. Available: read_file, write_file, "
                "edit_file, list_dir, grep_search, run_shell.", success=False))

        # Approval gate for mutating tools
        if name in MUTATING_TOOLS and not self.auto_approve:
            preview = self._preview(name, args)
            desc    = self._describe(name, args)
            if self._approver is None:
                return self._record(name, args, ToolResult(
                    f"'{name}' requires approval but no approver is configured. "
                    "Denied.", success=False))
            if not self._approver(name, desc, preview):
                return self._record(name, args, ToolResult(
                    f"User denied '{name}': {desc}", success=False))

        try:
            result: ToolResult = method(**args)
        except TypeError as e:
            result = ToolResult(f"Bad arguments for {name}: {e}", success=False)
        except Exception as e:
            result = ToolResult(f"Tool {name} crashed: {e}", success=False)

        if result.changed_path and result.changed_path not in self.files_changed:
            self.files_changed.append(result.changed_path)
        return self._record(name, args, result)

    # — helpers —
    def _record(self, name: str, args: dict, result: ToolResult) -> ToolResult:
        self.history.append(ExecutionRecord(
            tool=name, args=args, success=result.success,
            output_preview=result.output[:120],
        ))
        return result

    def _describe(self, name: str, args: dict) -> str:
        if name == "run_shell":
            return f"$ {args.get('command', '')}"
        if name == "write_file":
            return f"write {args.get('path', '?')}"
        if name == "edit_file":
            return f"edit {args.get('path', '?')}"
        return name

    def _preview(self, name: str, args: dict) -> str:
        """Unified diff preview for edits, content head for writes."""
        try:
            if name == "edit_file":
                p = self._tools._resolve(args.get("path", ""))
                if p.exists():
                    old = p.read_text(encoding="utf-8", errors="ignore")
                    new = old.replace(
                        args.get("old_string", ""), args.get("new_string", ""),
                        -1 if args.get("replace_all") else 1,
                    )
                    diff = difflib.unified_diff(
                        old.splitlines(), new.splitlines(),
                        fromfile=f"a/{args.get('path')}", tofile=f"b/{args.get('path')}",
                        lineterm="", n=3,
                    )
                    return "\n".join(list(diff)[:50])
            if name == "write_file":
                content = args.get("content", "")
                head = "\n".join(content.splitlines()[:20])
                return f"--- new content of {args.get('path')} (first 20 lines) ---\n{head}"
            if name == "run_shell":
                return args.get("command", "")
        except Exception:
            pass
        return ""
