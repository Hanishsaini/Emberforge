"""MCP server loop tests — every protocol situation, no subprocess needed.

Drives MCPServer.handle_message / handle_raw_line directly with every message
shape an MCP host can produce: good, weird, and hostile.
"""
import json

import pytest

from emberforge import __version__
from emberforge.agent import AgentResult
from emberforge.core import EmberResult
from emberforge.mcp.server import (
    MCPServer, handle_raw_line, TOOL_DEFS,
    PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR,
)


# ── helpers ───────────────────────────────────────────────────────────────────
def rpc(method, params=None, msg_id=1):
    msg = {"jsonrpc": "2.0", "method": method, "id": msg_id}
    if params is not None:
        msg["params"] = params
    return msg


class StubEmber:
    """Stands in for the real orchestrator — no API keys needed."""

    def __init__(self, agent_success=True, ask_success=True):
        self._agent_success = agent_success
        self._ask_success = ask_success

    async def run_agent(self, prompt, auto_approve=False, max_steps=15, **kw):
        return AgentResult(
            content=f"agent did: {prompt}",
            success=self._agent_success,
            steps=3, tool_calls_made=2,
            files_changed=["calc.py"] if self._agent_success else [],
            provider="stub", model="m", tokens_in=100, tokens_out=50,
            error="" if self._agent_success else "step_budget_exhausted",
        )

    async def run(self, prompt, **kw):
        return EmberResult(
            content=f"answer: {prompt}", provider="stub", model="m",
            task_type="explain", tokens_in=10, tokens_out=10, tokens_saved=0,
            latency_ms=5, attempts=1, success=self._ask_success,
            error="" if self._ask_success else "all providers failed",
        )


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "calc.py").write_text(
        "def divide(a, b):\n    return a / b\n", encoding="utf-8")
    (tmp_path / "app.py").write_text(
        "from calc import divide\n\ndef price(x):\n    return divide(x, 2)\n",
        encoding="utf-8")
    return tmp_path


@pytest.fixture
def server(repo):
    return MCPServer(repo, ember_factory=lambda r: StubEmber())


# ── Handshake & lifecycle ─────────────────────────────────────────────────────
class TestHandshake:
    async def test_initialize(self, server):
        resp = await server.handle_message(rpc("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "claude-desktop", "version": "1.0"},
        }))
        result = resp["result"]
        assert result["protocolVersion"]
        assert "tools" in result["capabilities"]
        assert result["serverInfo"] == {"name": "emberforge", "version": __version__}

    async def test_initialized_notification_silent(self, server):
        msg = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        assert await server.handle_message(msg) is None
        assert server.initialized

    async def test_ping(self, server):
        resp = await server.handle_message(rpc("ping"))
        assert resp["result"] == {}

    async def test_id_zero_is_a_request_not_notification(self, server):
        resp = await server.handle_message(rpc("ping", msg_id=0))
        assert resp is not None and resp["id"] == 0


# ── tools/list ────────────────────────────────────────────────────────────────
class TestToolsList:
    async def test_all_tools_listed_with_valid_schemas(self, server):
        resp = await server.handle_message(rpc("tools/list"))
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {"ember_agent", "ember_ask", "ember_repo_map",
                         "ember_compress", "ember_status"}
        for t in tools:
            assert t["description"]
            assert t["inputSchema"]["type"] == "object"
            for req in t["inputSchema"].get("required", []):
                assert req in t["inputSchema"]["properties"]


# ── tools/call: every tool ────────────────────────────────────────────────────
class TestToolCalls:
    async def _call(self, server, name, arguments=None, msg_id=7):
        resp = await server.handle_message(
            rpc("tools/call", {"name": name, "arguments": arguments or {}}, msg_id))
        return resp

    async def test_repo_map(self, server):
        resp = await self._call(server, "ember_repo_map", {"query": "divide"})
        result = resp["result"]
        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "calc.py" in text and "divide" in text

    async def test_repo_map_explicit_repo(self, server, repo):
        resp = await self._call(server, "ember_repo_map", {"repo": str(repo)})
        assert "calc.py" in resp["result"]["content"][0]["text"]

    async def test_repo_map_bad_dir_invalid_params(self, server):
        resp = await self._call(server, "ember_repo_map",
                                {"repo": "Z:/does/not/exist"})
        assert resp["error"]["code"] == INVALID_PARAMS

    async def test_compress(self, server):
        code = "def f():\n    return 1\n\n\ndef g():\n    return 2\n" * 20
        resp = await self._call(server, "ember_compress",
                                {"text": code, "content_type": "code",
                                 "filename": "big.py"})
        text = resp["result"]["content"][0]["text"]
        assert "% reduction]" in text
        assert resp["result"]["isError"] is False

    async def test_agent_success(self, server):
        resp = await self._call(server, "ember_agent", {"prompt": "fix the bug"})
        result = resp["result"]
        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "agent did: fix the bug" in text
        assert "Files changed: calc.py" in text

    async def test_agent_failure_flagged(self, repo):
        server = MCPServer(repo, ember_factory=lambda r: StubEmber(agent_success=False))
        resp = await self._call(server, "ember_agent", {"prompt": "impossible"})
        assert resp["result"]["isError"] is True

    async def test_ask(self, server):
        resp = await self._call(server, "ember_ask", {"prompt": "what is calc?"})
        assert "answer: what is calc?" in resp["result"]["content"][0]["text"]

    async def test_status_never_crashes(self, server):
        resp = await self._call(server, "ember_status")
        assert "result" in resp   # works with or without configured providers


# ── Hostile & malformed input ─────────────────────────────────────────────────
class TestHostileInput:
    async def test_unknown_method(self, server):
        resp = await server.handle_message(rpc("resources/list"))
        assert resp["error"]["code"] == METHOD_NOT_FOUND

    async def test_unknown_notification_ignored(self, server):
        msg = {"jsonrpc": "2.0", "method": "notifications/cancelled"}
        assert await server.handle_message(msg) is None

    async def test_unknown_tool(self, server):
        resp = await server.handle_message(
            rpc("tools/call", {"name": "ember_teleport", "arguments": {}}))
        assert resp["error"]["code"] == INVALID_PARAMS

    async def test_missing_required_argument(self, server):
        resp = await server.handle_message(
            rpc("tools/call", {"name": "ember_agent", "arguments": {}}))
        assert resp["error"]["code"] == INVALID_PARAMS
        assert "prompt" in resp["error"]["message"]

    async def test_call_without_params(self, server):
        resp = await server.handle_message(rpc("tools/call"))
        assert resp["error"]["code"] == INVALID_PARAMS

    async def test_not_jsonrpc(self, server):
        resp = await server.handle_message({"hello": "world"})
        assert resp["error"]["code"] == INVALID_REQUEST

    async def test_non_dict_message(self, server):
        resp = await server.handle_message(["not", "a", "dict"])
        assert resp["error"]["code"] == INVALID_REQUEST

    async def test_parse_error_on_wire(self, server):
        resp = await handle_raw_line(server, "{this is not json")
        assert resp["error"]["code"] == PARSE_ERROR

    async def test_blank_line_ignored(self, server):
        assert await handle_raw_line(server, "   \n") is None

    async def test_factory_crash_is_internal_error(self, repo):
        def bomb(r):
            raise RuntimeError("boom")
        server = MCPServer(repo, ember_factory=bomb)
        resp = await server.handle_message(
            rpc("tools/call", {"name": "ember_ask", "arguments": {"prompt": "x"}}))
        # tool execution failures are reported in-band, host stays alive
        assert resp["result"]["isError"] is True
        assert "boom" in resp["result"]["content"][0]["text"]

    async def test_request_id_echoed_back(self, server):
        resp = await server.handle_message(rpc("ping", msg_id="abc-123"))
        assert resp["id"] == "abc-123"

    async def test_sequential_calls_state_survives(self, server):
        """A whole session in order — the situational loop test."""
        assert (await server.handle_message(rpc("initialize", {}, 1)))["result"]
        assert await server.handle_message(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
        assert (await server.handle_message(rpc("tools/list", None, 2)))["result"]
        for i, (name, args) in enumerate([
            ("ember_repo_map", {"query": "divide"}),
            ("ember_compress", {"text": "x = 1\n" * 100}),
            ("ember_ask", {"prompt": "q"}),
            ("ember_agent", {"prompt": "fix"}),
            ("ember_status", {}),
        ], start=3):
            resp = await server.handle_message(
                rpc("tools/call", {"name": name, "arguments": args}, i))
            assert "result" in resp, f"{name} failed: {resp}"
        assert (await server.handle_message(rpc("ping", None, 99)))["result"] == {}
