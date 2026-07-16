"""
Agent-loop integration tests — every workflow shape the harness must handle.

A scripted MockProvider plays the LLM, so each test drives the REAL agent loop,
REAL tool executor, and REAL files on disk deterministically.
"""
from __future__ import annotations

import json

import pytest

from emberforge import TIER_SMART_FREE
from emberforge.agent import EmberAgent
from emberforge.providers.base import BaseProvider, EmberResponse, ToolCall
from emberforge.tools import EmberTools, ToolExecutor


# ── Scripted fake LLM ─────────────────────────────────────────────────────────
class MockProvider(BaseProvider):
    """Returns a scripted sequence of responses; records every request."""

    def __init__(self, script: list[EmberResponse], name: str = "mock",
                 tier: str = TIER_SMART_FREE, supports_tools: bool = True):
        super().__init__(
            name=name, api_key="test-key", tier=tier,
            models={"primary": "mock-1"}, base_url="http://mock",
            rpm_limit=10_000, supports_tools=supports_tools,
        )
        self.script = list(script)
        self.requests: list[dict] = []   # {"messages": [...], "tools": ...}

    async def complete(self, prompt, context="", system="",
                       max_tokens=4096, stream=False) -> EmberResponse:
        raise AssertionError("agent must use chat(), not complete()")

    async def chat(self, messages, tools=None, max_tokens=4096) -> EmberResponse:
        self.requests.append({
            "messages": [dict(m) for m in messages],
            "tools": tools,
        })
        if not self.script:
            return EmberResponse(content="", provider=self.name, model="mock-1",
                                 success=False, error="script exhausted")
        resp = self.script.pop(0)
        resp.provider = resp.provider or self.name
        resp.model = resp.model or "mock-1"
        return resp

    async def health_check(self) -> bool:
        return True


def tc(name: str, i: int = 0, **args) -> ToolCall:
    return ToolCall(id=f"call_{name}_{i}", name=name, arguments=json.dumps(args))


def answer(text: str) -> EmberResponse:
    return EmberResponse(content=text, provider="", model="", tokens_in=10, tokens_out=5)


def calls(*tool_calls: ToolCall) -> EmberResponse:
    return EmberResponse(content="", provider="", model="",
                         tokens_in=10, tokens_out=5, tool_calls=list(tool_calls))


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def repo(tmp_path):
    (tmp_path / "calc.py").write_text(
        "def divide(a, b):\n"
        "    return a / b\n",
        encoding="utf-8",
    )
    (tmp_path / "docs.md").write_text("# Calc\n", encoding="utf-8")
    return tmp_path


def make_agent(repo, providers, max_steps=10, auto_approve=True, approver=None,
               **kwargs) -> EmberAgent:
    tools = EmberTools(repo)
    executor = ToolExecutor(tools, auto_approve=auto_approve, approver=approver)
    if isinstance(providers, BaseProvider):
        providers = {providers.name: providers}
    return EmberAgent(providers=providers, executor=executor,
                      max_steps=max_steps, verbose=False, **kwargs)


# ══ Workflow 1: pure Q&A — no tools needed ════════════════════════════════════
async def test_direct_answer(repo):
    provider = MockProvider([answer("It divides a by b.")])
    agent = make_agent(repo, provider)
    result = await agent.run("what does divide do?")
    assert result.success
    assert result.content == "It divides a by b."
    assert result.steps == 1
    assert result.tool_calls_made == 0


# ══ Workflow 2: explore then answer (read) ════════════════════════════════════
async def test_read_then_answer(repo):
    provider = MockProvider([
        calls(tc("read_file", path="calc.py")),
        answer("divide(a, b) returns a / b."),
    ])
    agent = make_agent(repo, provider)
    result = await agent.run("explain calc.py")
    assert result.success
    assert result.tool_calls_made == 1
    # the tool result actually reached the model on the second request
    tool_msgs = [m for m in provider.requests[1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "def divide" in tool_msgs[0]["content"]


# ══ Workflow 3: the classic fix loop — read → edit → verify file changed ══════
async def test_read_edit_flow(repo):
    provider = MockProvider([
        calls(tc("read_file", path="calc.py")),
        calls(tc("edit_file", path="calc.py",
                 old_string="    return a / b",
                 new_string="    if b == 0:\n        raise ValueError('division by zero')\n    return a / b")),
        answer("Added a zero-division guard to divide()."),
    ])
    agent = make_agent(repo, provider)
    result = await agent.run("fix the zero division bug")
    assert result.success
    assert result.files_changed == ["calc.py"]
    assert "raise ValueError" in (repo / "calc.py").read_text()


# ══ Workflow 4: create a new file ═════════════════════════════════════════════
async def test_create_file_flow(repo):
    provider = MockProvider([
        calls(tc("write_file", path="tests/test_calc.py",
                 content="from calc import divide\n\ndef test_divide():\n    assert divide(6, 2) == 3\n")),
        answer("Created tests/test_calc.py."),
    ])
    agent = make_agent(repo, provider)
    result = await agent.run("write a test for divide")
    assert result.success
    assert (repo / "tests" / "test_calc.py").exists()
    assert result.files_changed == ["tests/test_calc.py"]


# ══ Workflow 5: run a shell command and use its output ════════════════════════
async def test_shell_flow(repo):
    provider = MockProvider([
        calls(tc("run_shell", command="echo all-tests-pass")),
        answer("Verified: all tests pass."),
    ])
    agent = make_agent(repo, provider)
    result = await agent.run("run the tests")
    assert result.success
    tool_msgs = [m for m in provider.requests[1]["messages"] if m["role"] == "tool"]
    assert "all-tests-pass" in tool_msgs[0]["content"]


# ══ Workflow 6: full multi-step chain — grep → read → edit → verify ═══════════
async def test_multi_step_chain(repo):
    provider = MockProvider([
        calls(tc("grep_search", pattern="divide")),
        calls(tc("read_file", path="calc.py")),
        calls(tc("edit_file", path="calc.py",
                 old_string="def divide(a, b):",
                 new_string="def divide(a: float, b: float) -> float:")),
        calls(tc("run_shell", command="echo typecheck-ok")),
        answer("Added type hints to divide() and verified."),
    ])
    agent = make_agent(repo, provider)
    result = await agent.run("add type hints to divide")
    assert result.success
    assert result.steps == 5
    assert result.tool_calls_made == 4
    assert "a: float" in (repo / "calc.py").read_text()


# ══ Workflow 7: parallel tool calls in one response ═══════════════════════════
async def test_parallel_tool_calls(repo):
    provider = MockProvider([
        calls(
            tc("read_file", 0, path="calc.py"),
            tc("read_file", 1, path="docs.md"),
        ),
        answer("Read both files."),
    ])
    agent = make_agent(repo, provider)
    result = await agent.run("compare calc.py and docs.md")
    assert result.success
    assert result.tool_calls_made == 2
    tool_msgs = [m for m in provider.requests[1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 2


# ══ Workflow 8: tool error → model sees it and recovers ═══════════════════════
async def test_error_recovery(repo):
    provider = MockProvider([
        calls(tc("read_file", path="wrong_name.py")),
        calls(tc("read_file", path="calc.py")),
        answer("Found it in calc.py."),
    ])
    agent = make_agent(repo, provider)
    result = await agent.run("read the calculator module")
    assert result.success
    # the error text reached the model
    tool_msgs = [m for m in provider.requests[1]["messages"] if m["role"] == "tool"]
    assert "not found" in tool_msgs[0]["content"].lower()


# ══ Workflow 9: unknown tool name → graceful error, loop continues ════════════
async def test_unknown_tool_handled(repo):
    provider = MockProvider([
        calls(tc("summon_demon", incantation="xyzzy")),
        answer("Sorry, used a real tool instead."),
    ])
    agent = make_agent(repo, provider)
    result = await agent.run("do something")
    assert result.success
    tool_msgs = [m for m in provider.requests[1]["messages"] if m["role"] == "tool"]
    assert "Unknown tool" in tool_msgs[0]["content"]


# ══ Workflow 10: step budget exhaustion → graceful stop, work reported ════════
async def test_step_budget(repo):
    provider = MockProvider(
        [calls(tc("read_file", i, path="calc.py")) for i in range(50)]
    )
    agent = make_agent(repo, provider, max_steps=4)
    result = await agent.run("loop forever")
    assert not result.success
    assert result.error == "step_budget_exhausted"
    assert result.steps == 4
    assert "budget" in result.content.lower()


# ══ Workflow 11: user denies an edit → file untouched, agent informed ═════════
async def test_approval_denied(repo):
    provider = MockProvider([
        calls(tc("edit_file", path="calc.py", old_string="a / b", new_string="a // b")),
        answer("Understood — leaving the file as is."),
    ])
    agent = make_agent(repo, provider, auto_approve=False,
                       approver=lambda name, desc, preview: False)
    result = await agent.run("change division to floor division")
    assert result.success
    assert "a / b" in (repo / "calc.py").read_text()          # unchanged
    assert result.files_changed == []
    tool_msgs = [m for m in provider.requests[1]["messages"] if m["role"] == "tool"]
    assert "denied" in tool_msgs[0]["content"].lower()


# ══ Workflow 12: provider dies mid-loop → conversation continues on next ══════
async def test_provider_rotation_mid_loop(repo):
    flaky = MockProvider([
        calls(tc("read_file", path="calc.py")),
        EmberResponse(content="", provider="flaky", model="mock-1",
                      success=False, error="HTTP 429: quota exceeded"),
    ], name="flaky")
    backup = MockProvider([answer("Finished on the backup provider.")], name="backup")

    agent = make_agent(repo, {"flaky": flaky, "backup": backup})
    result = await agent.run("explain calc.py")
    assert result.success
    assert result.content == "Finished on the backup provider."
    # backup received the full conversation, including the earlier tool result
    roles = [m["role"] for m in backup.requests[0]["messages"]]
    assert "tool" in roles


# ══ Workflow 13: all providers fail → clean failure result ════════════════════
async def test_all_providers_fail(repo):
    p = MockProvider([
        EmberResponse(content="", provider="mock", model="m",
                      success=False, error="HTTP 500"),
    ])
    agent = make_agent(repo, p)
    result = await agent.run("anything")
    assert not result.success
    assert "HTTP 500" in result.error


# ══ Workflow 14: no providers configured at all ═══════════════════════════════
async def test_no_providers(repo):
    agent = make_agent(repo, {})
    result = await agent.run("anything")
    assert not result.success
    assert result.error == "no_providers"


# ══ Workflow 15: ReAct fallback — provider without native tool calling ════════
async def test_react_protocol(repo):
    react_call = (
        "I'll read the file first.\n"
        "```tool\n"
        '{"name": "read_file", "arguments": {"path": "calc.py"}}\n'
        "```"
    )
    provider = MockProvider(
        [answer(react_call), answer("divide(a, b) returns a / b.")],
        supports_tools=False,
    )
    agent = make_agent(repo, provider)
    result = await agent.run("explain calc.py")
    assert result.success
    assert result.tool_calls_made == 1
    # ReAct providers must never receive native tool schemas
    assert provider.requests[0]["tools"] is None
    # system prompt carries the ReAct protocol
    assert "TOOL PROTOCOL" in provider.requests[0]["messages"][0]["content"]
    # tool result came back as a user message
    user_msgs = [m for m in provider.requests[1]["messages"] if m["role"] == "user"]
    assert any("TOOL RESULT" in m["content"] for m in user_msgs)


# ══ Workflow 16: native provider rejects tools at runtime → ReAct retry ═══════
async def test_tools_unsupported_runtime_switch(repo):
    class FlipProvider(MockProvider):
        async def chat(self, messages, tools=None, max_tokens=4096):
            self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
            if tools is not None:
                self.supports_tools = False   # mimics openai_compat 400-handling
                return EmberResponse(content="", provider=self.name, model="mock-1",
                                     success=False,
                                     error="tools_unsupported: HTTP 400")
            return self.script.pop(0)

    provider = FlipProvider([answer("Done without native tools.")])
    agent = make_agent(repo, provider)
    result = await agent.run("do a thing")
    assert result.success
    assert result.content == "Done without native tools."
    assert provider.requests[0]["tools"] is not None    # first try: native
    assert provider.requests[1]["tools"] is None        # retry: ReAct


# ══ Workflow 17: REPL continuity — second turn remembers the first ════════════
async def test_multi_turn_memory(repo):
    provider = MockProvider([
        answer("The file is calc.py."),
        answer("Yes — as I said, calc.py."),
    ])
    agent = make_agent(repo, provider)
    r1 = await agent.run("which file has divide?")
    r2 = await agent.run("are you sure?")
    assert r1.success and r2.success
    # second request contains the full first exchange
    msgs = provider.requests[1]["messages"]
    contents = [str(m.get("content")) for m in msgs]
    assert any("which file has divide?" in c for c in contents)
    assert any("The file is calc.py." in c for c in contents)

    agent.reset()
    assert agent.messages == []


# ══ Workflow 19 (Phase 2): re-read of unchanged file returns cache marker ═════
async def test_agent_read_cache_in_loop(repo):
    provider = MockProvider([
        calls(tc("read_file", 0, path="calc.py")),
        calls(tc("read_file", 1, path="calc.py")),   # identical re-read
        answer("done"),
    ])
    agent = make_agent(repo, provider)
    result = await agent.run("read calc twice")
    assert result.success
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert "def divide" in tool_msgs[0]["content"]      # first: full content
    assert "[cached]" in tool_msgs[1]["content"]        # second: marker only


# ══ Workflow 20 (Phase 2): edit busts the cache inside the loop ═══════════════
async def test_agent_cache_busted_by_edit(repo):
    provider = MockProvider([
        calls(tc("read_file", 0, path="calc.py")),
        calls(tc("edit_file", path="calc.py",
                 old_string="a / b", new_string="a / b  # checked")),
        calls(tc("read_file", 1, path="calc.py")),   # re-read after edit
        answer("done"),
    ])
    agent = make_agent(repo, provider)
    result = await agent.run("edit then verify")
    assert result.success
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert "[cached]" not in tool_msgs[2]["content"]    # fresh content
    assert "# checked" in tool_msgs[2]["content"]


# ══ Workflow 21 (Phase 2): compaction invalidates the read cache ══════════════
async def test_compaction_invalidates_read_cache(repo):
    (repo / "big.py").write_text(
        "\n".join(f"item_{i} = {i}  # padding" for i in range(300)),
        encoding="utf-8",
    )
    # read big.py, then 12 filler reads to push it out of the keep-tail window,
    # then read big.py again — its content was compacted away, so the agent
    # must get FULL content back, not a [cached] marker.
    script = (
        [calls(tc("read_file", 0, path="big.py"))]
        + [calls(tc("read_file", i, path="calc.py", force=True)) for i in range(1, 13)]
        + [calls(tc("read_file", 99, path="big.py"))]
        + [answer("done")]
    )
    provider = MockProvider(script)
    agent = make_agent(repo, provider, max_steps=20, compact_threshold_tokens=1_500)
    result = await agent.run("stress the compactor")
    assert result.success
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    # first big.py read was compacted
    assert "...[compacted:" in tool_msgs[0]["content"]
    # final big.py read is full content again — cache was invalidated
    assert "[cached]" not in tool_msgs[-1]["content"]
    assert "item_200" in tool_msgs[-1]["content"]


# ══ Workflow 22 (Phase 2): internal metadata never reaches the provider ═══════
async def test_emberforge_meta_stripped_from_wire(repo):
    provider = MockProvider([
        calls(tc("read_file", path="calc.py")),
        answer("done"),
    ])
    agent = make_agent(repo, provider)
    await agent.run("read it")
    for req in provider.requests:
        for m in req["messages"]:
            assert "_emberforge_meta" not in m


# ══ Workflow 18: history compaction keeps context bounded ═════════════════════
async def test_history_compaction(repo):
    # a file whose read output is ~1000 tokens, read 14 times
    (repo / "big.py").write_text(
        "\n".join(f"variable_{i} = {i}  # padding line" for i in range(300)),
        encoding="utf-8",
    )
    provider = MockProvider(
        [calls(tc("read_file", i, path="big.py")) for i in range(14)] + [answer("done")]
    )
    agent = make_agent(repo, provider, max_steps=20,
                       compact_threshold_tokens=2_000)
    result = await agent.run("read big.py many times")
    assert result.success
    compacted = [m for m in agent.messages
                 if m["role"] == "tool" and "compacted" in str(m.get("content", ""))]
    assert compacted, "old tool outputs should have been compacted"
    # the most recent tool messages stay intact
    last_tools = [m for m in agent.messages[-8:] if m["role"] == "tool"]
    assert all("compacted" not in str(m["content"]) for m in last_tools)
