"""MCP integration loop tests.

Part 1 — the self-hosting round trip: EmberForge's MCP CLIENT talks to
EmberForge's own MCP SERVER over a real subprocess stdio pipe. If this
passes, the wire protocol works end-to-end on this platform.

Part 2 — the agent loop with MCP tools: external tools join the loop next
to local tools, in both native tool-calling and ReAct protocols, and a
misbehaving external server never kills the run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from emberforge.mcp.client import MCPClient, MCPManager
from emberforge.tools import EmberTools, ToolExecutor
from emberforge.agent import EmberAgent

from tests.integration.test_agent_loop import MockProvider, tc, calls, answer

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "calc.py").write_text(
        "def divide(a, b):\n    return a / b\n", encoding="utf-8")
    (tmp_path / "app.py").write_text(
        "from calc import divide\n\ndef price(x):\n    return divide(x, 2)\n",
        encoding="utf-8")
    return tmp_path


def spawn_args(repo):
    return dict(
        command=sys.executable,
        args=["-m", "emberforge.mcp.server", "--repo", str(repo)],
        cwd=PROJECT_ROOT,
    )


# ══ Part 1: real subprocess round trip ════════════════════════════════════════
class TestRoundTrip:
    async def test_full_session_over_real_stdio(self, repo):
        client = MCPClient("ember", **spawn_args(repo))
        try:
            await client.start()

            # handshake produced the tool list
            names = {t["name"] for t in client.tools}
            assert names == {"ember_agent", "ember_ask", "ember_repo_map",
                             "ember_compress", "ember_status"}

            # call 1: repo map over the wire
            text, is_error = await client.call_tool(
                "ember_repo_map", {"query": "divide"})
            assert not is_error
            assert "calc.py" in text and "divide" in text

            # call 2: compression over the wire
            text, is_error = await client.call_tool(
                "ember_compress",
                {"text": "def f():\n    return 1\n" * 50,
                 "content_type": "code", "filename": "x.py"})
            assert not is_error
            assert "% reduction]" in text

            # call 3: status (works with or without providers configured)
            text, is_error = await client.call_tool("ember_status", {})
            assert not is_error
            assert text

            # hostile: unknown tool → JSON-RPC error raised client-side
            with pytest.raises(RuntimeError, match="-?32602|Unknown tool"):
                await client.call_tool("ember_teleport", {})

            # the session survived the error — pipe still works
            text, is_error = await client.call_tool(
                "ember_repo_map", {"query": "price"})
            assert not is_error and "app.py" in text
        finally:
            await client.close()

    async def test_manager_namespacing_over_real_stdio(self, repo):
        manager = MCPManager({"ember": spawn_args(repo)})
        try:
            errors = await manager.connect()
            assert errors == []
            assert manager.connected == ["ember"]

            # schemas are OpenAI-format and namespaced
            schema_names = {s["function"]["name"] for s in manager.schemas}
            assert "mcp__ember__ember_repo_map" in schema_names
            for s in manager.schemas:
                assert s["type"] == "function"
                assert s["function"]["parameters"]["type"] == "object"

            # routed call through the namespace, JSON-string args (agent style)
            output, ok = await manager.call(
                "mcp__ember__ember_repo_map", json.dumps({"query": "divide"}))
            assert ok and "calc.py" in output

            # bad JSON args → error string, no exception
            output, ok = await manager.call("mcp__ember__ember_repo_map", "{broken")
            assert not ok and "Invalid JSON" in output

            # unknown server → error string, no exception
            output, ok = await manager.call("mcp__ghost__tool", "{}")
            assert not ok and "Unknown MCP tool" in output
        finally:
            await manager.close()

    async def test_dead_server_reported_not_fatal(self):
        manager = MCPManager({
            "broken": {"command": sys.executable,
                       "args": ["-c", "import sys; sys.exit(1)"]},
        })
        errors = await manager.connect()
        assert len(errors) == 1 and "broken" in errors[0]
        assert manager.connected == []
        await manager.close()   # closing with no clients is safe


# ══ Part 2: MCP tools inside the agent loop ═══════════════════════════════════
class FakeMCP:
    """In-process MCP manager double — deterministic, no subprocess."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    @property
    def schemas(self):
        return [{
            "type": "function",
            "function": {
                "name": "mcp__docs__search_docs",
                "description": "Search external documentation. [external MCP tool from 'docs']",
                "parameters": {"type": "object",
                               "properties": {"query": {"type": "string"}},
                               "required": ["query"]},
            },
        }]

    def owns(self, name):
        return name.startswith("mcp__")

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        if self.fail:
            return "MCP call to 'mcp__docs__search_docs' failed: timeout", False
        args = json.loads(arguments or "{}")
        return f"DOCS RESULT for '{args.get('query', '')}'", True


def make_agent(repo, provider, mcp, **kwargs):
    executor = ToolExecutor(EmberTools(repo), auto_approve=True)
    return EmberAgent(providers={provider.name: provider}, executor=executor,
                      verbose=False, mcp=mcp, **kwargs)


class TestAgentWithMCP:
    async def test_mcp_schemas_advertised_to_provider(self, repo):
        provider = MockProvider([answer("done")])
        agent = make_agent(repo, provider, FakeMCP())
        await agent.run("anything")
        tool_names = {t["function"]["name"] for t in provider.requests[0]["tools"]}
        assert "mcp__docs__search_docs" in tool_names
        assert "read_file" in tool_names            # local tools still present

    async def test_mixed_local_and_mcp_workflow(self, repo):
        """The flagship loop: external lookup + local read + local edit."""
        provider = MockProvider([
            calls(tc("mcp__docs__search_docs", query="ValueError best practice")),
            calls(tc("read_file", path="calc.py")),
            calls(tc("edit_file", path="calc.py",
                     old_string="    return a / b",
                     new_string="    if b == 0:\n        raise ValueError('div0')\n    return a / b")),
            answer("Fixed using the documented pattern."),
        ])
        mcp = FakeMCP()
        agent = make_agent(repo, provider, mcp)
        result = await agent.run("fix divide using docs guidance")

        assert result.success
        assert result.tool_calls_made == 3
        assert mcp.calls[0][0] == "mcp__docs__search_docs"
        # MCP result reached the model
        tool_msgs = [m for m in provider.requests[1]["messages"] if m["role"] == "tool"]
        assert "DOCS RESULT for 'ValueError best practice'" in tool_msgs[0]["content"]
        # and the local edit really happened
        assert "raise ValueError" in (repo / "calc.py").read_text()
        # trace completeness: the MCP call is in the executor history
        assert agent._executor.history[0].tool == "mcp__docs__search_docs"

    async def test_mcp_failure_does_not_kill_loop(self, repo):
        provider = MockProvider([
            calls(tc("mcp__docs__search_docs", query="anything")),
            answer("Proceeded without docs."),
        ])
        agent = make_agent(repo, provider, FakeMCP(fail=True))
        result = await agent.run("try the docs tool")
        assert result.success                        # loop survived
        tool_msgs = [m for m in provider.requests[1]["messages"] if m["role"] == "tool"]
        assert "failed" in tool_msgs[0]["content"]   # model saw the failure

    async def test_react_protocol_with_mcp_tool(self, repo):
        react_call = (
            "```tool\n"
            '{"name": "mcp__docs__search_docs", "arguments": {"query": "zero division"}}\n'
            "```"
        )
        provider = MockProvider(
            [answer(react_call), answer("done, used the docs")],
            supports_tools=False,
        )
        mcp = FakeMCP()
        agent = make_agent(repo, provider, mcp)
        result = await agent.run("look it up")
        assert result.success
        assert mcp.calls   # external tool executed through the text protocol
        # ReAct system prompt advertises the external tool
        sys_prompt = provider.requests[0]["messages"][0]["content"]
        assert "mcp__docs__search_docs" in sys_prompt

    async def test_no_mcp_means_no_mcp_schemas(self, repo):
        provider = MockProvider([answer("done")])
        agent = make_agent(repo, provider, None)
        await agent.run("anything")
        tool_names = {t["function"]["name"] for t in provider.requests[0]["tools"]}
        assert not any(n.startswith("mcp__") for n in tool_names)
