"""
EmberForge MCP Server — expose EmberForge to any MCP host
(Claude Desktop, Cline, Continue, Goose, ...) over stdio.

Design for testability:
  - handle_message() is a pure request->response function: every protocol
    situation can be driven in tests without a subprocess.
  - The Ember orchestrator is built through an injectable factory, so tests
    exercise the full protocol without API keys.

Run: emberforge mcp-serve [--repo PATH]   (or python -m emberforge.mcp.server)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from emberforge import __version__

PROTOCOL_VERSION = "2025-06-18"

# JSON-RPC 2.0 error codes
PARSE_ERROR      = -32700
INVALID_REQUEST  = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS   = -32602
INTERNAL_ERROR   = -32603


class InvalidParams(Exception):
    pass


TOOL_DEFS: list[dict] = [
    {
        "name": "ember_agent",
        "description": (
            "Run EmberForge's autonomous coding agent on a repository: it "
            "explores the code, edits files, and runs tests in a loop until "
            "the task is done. Edits are auto-approved; destructive shell "
            "commands remain hard-blocked. Returns a summary and the list of "
            "changed files."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt":    {"type": "string", "description": "The coding task"},
                "repo":      {"type": "string", "description": "Path to the repository (default: server's repo)"},
                "max_steps": {"type": "integer", "default": 15},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "ember_ask",
        "description": (
            "One-shot question about a repository, answered with compressed "
            "codebase context (no files are modified)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "repo":   {"type": "string"},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "ember_repo_map",
        "description": (
            "Structural map of a repository: files ranked by PageRank over the "
            "symbol reference graph (fused with BM25 when a query is given), "
            "with the symbols each file defines."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo":         {"type": "string"},
                "query":        {"type": "string", "description": "Optional task to rank against"},
                "token_budget": {"type": "integer", "default": 600},
            },
            "required": [],
        },
    },
    {
        "name": "ember_compress",
        "description": (
            "Compress text before sending it to an LLM: Python/JS/TS/Go/Rust/"
            "Java code collapses to signatures, shell output keeps failures "
            "and drops noise, large JSON gets sampled. Returns compressed text "
            "plus measured token savings."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text":         {"type": "string"},
                "content_type": {"type": "string",
                                 "enum": ["auto", "code", "shell", "json", "text"],
                                 "default": "auto"},
                "filename":     {"type": "string", "description": "Helps language detection, e.g. app.py"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "ember_status",
        "description": "List EmberForge's configured LLM providers with tier, availability, and cooldown state.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
]

_TOOLS_BY_NAME = {t["name"]: t for t in TOOL_DEFS}


class MCPServer:
    def __init__(self, repo_default: str | Path = ".", ember_factory=None):
        self.repo_default = str(repo_default)
        self._ember_factory = ember_factory or self._default_ember_factory
        self.initialized = False

    @staticmethod
    def _default_ember_factory(repo: str):
        from emberforge.core import Ember
        return Ember(project=Path(repo).resolve().name, repo_path=repo, verbose=False)

    # ── JSON-RPC dispatch ─────────────────────────────────────────────────────
    async def handle_message(self, msg) -> dict | None:
        """One JSON-RPC message in, one response dict out (None for notifications)."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
            return self._error(None, INVALID_REQUEST, "Invalid JSON-RPC request")

        method       = msg["method"]
        msg_id       = msg.get("id")
        notification = "id" not in msg

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "emberforge", "version": __version__},
                }
            elif method == "notifications/initialized":
                self.initialized = True
                return None
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOL_DEFS}
            elif method == "tools/call":
                result = await self._tools_call(msg.get("params") or {})
            else:
                if notification:
                    return None            # unknown notifications are ignored
                return self._error(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        except InvalidParams as e:
            return self._error(msg_id, INVALID_PARAMS, str(e))
        except Exception as e:
            return self._error(msg_id, INTERNAL_ERROR, str(e)[:300])

        if notification:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": code, "message": message}}

    # ── tools/call ────────────────────────────────────────────────────────────
    async def _tools_call(self, params: dict) -> dict:
        name = params.get("name")
        args = params.get("arguments") or {}
        if not name or name not in _TOOLS_BY_NAME:
            raise InvalidParams(f"Unknown tool: {name!r}")
        for required in _TOOLS_BY_NAME[name]["inputSchema"].get("required", []):
            if required not in args:
                raise InvalidParams(f"Missing required argument: {required!r}")

        handler = getattr(self, f"_tool_{name}")
        try:
            text, is_error = await handler(args)
        except InvalidParams:
            raise
        except Exception as e:
            text, is_error = f"Tool execution failed: {str(e)[:300]}", True

        return {"content": [{"type": "text", "text": text}], "isError": is_error}

    # ── tool handlers ─────────────────────────────────────────────────────────
    async def _tool_ember_agent(self, args: dict) -> tuple[str, bool]:
        repo  = args.get("repo") or self.repo_default
        ember = self._ember_factory(repo)
        result = await ember.run_agent(
            prompt=args["prompt"],
            auto_approve=True,
            max_steps=int(args.get("max_steps", 15)),
        )
        text = result.content
        if result.files_changed:
            text += "\n\nFiles changed: " + ", ".join(result.files_changed)
        text += (f"\n[{result.provider} · {result.steps} steps · "
                 f"{result.tokens_in + result.tokens_out} tokens]")
        return text, not result.success

    async def _tool_ember_ask(self, args: dict) -> tuple[str, bool]:
        repo  = args.get("repo") or self.repo_default
        ember = self._ember_factory(repo)
        result = await ember.run(prompt=args["prompt"])
        return result.content, not result.success

    async def _tool_ember_repo_map(self, args: dict) -> tuple[str, bool]:
        from emberforge.repomap import RepoMap
        repo = args.get("repo") or self.repo_default
        if not Path(repo).is_dir():
            raise InvalidParams(f"Not a directory: {repo}")
        block = RepoMap(repo).render_map(
            args.get("query", ""),
            token_budget=int(args.get("token_budget", 600)),
        )
        return block or "(no code files found in repository)", False

    async def _tool_ember_compress(self, args: dict) -> tuple[str, bool]:
        from emberforge.compressor import EmberCompressor
        result = EmberCompressor().compress(
            args["text"],
            content_type=args.get("content_type", "auto"),
            filename=args.get("filename", ""),
        )
        text = (f"{result.final_text}\n\n"
                f"[{result.original_tokens} → {result.final_tokens} tokens, "
                f"{result.reduction_pct}% reduction]")
        return text, False

    async def _tool_ember_status(self, args: dict) -> tuple[str, bool]:
        from emberforge.config.settings import load_config
        from emberforge.providers import build_providers
        providers = build_providers(load_config())
        if not providers:
            return "No providers configured (run: emberforge init).", False
        lines = []
        for name, p in providers.items():
            if p.health.in_cooldown():
                state = f"cooldown {p.health.cooldown_remaining()}s"
            else:
                state = "available" if p.is_available() else "unavailable"
            lines.append(f"{name} [{p.tier}] — {state}")
        return "\n".join(lines), False


# ── stdio pump ────────────────────────────────────────────────────────────────
async def handle_raw_line(server: MCPServer, line: str) -> dict | None:
    """Parse one wire line and dispatch it (exposed separately for tests)."""
    line = line.strip()
    if not line:
        return None
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return MCPServer._error(None, PARSE_ERROR, "Parse error")
    return await server.handle_message(msg)


async def serve_stdio(server: MCPServer) -> None:
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:                      # EOF: host closed the pipe
            break
        response = await handle_raw_line(server, line)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="EmberForge MCP server (stdio)")
    parser.add_argument("--repo", default=".", help="Default repository path")
    parsed = parser.parse_args()
    asyncio.run(serve_stdio(MCPServer(parsed.repo)))


if __name__ == "__main__":
    main()
