"""
EMBERFORGE Agent — the harness loop.
explore → act → observe → repeat, until the task is done or the budget runs out.

Two protocols:
  1. Native tool calling (OpenAI function-calling format) — preferred.
  2. ReAct text fallback — for providers/models without native tool support:
     the model emits one ```tool {...}``` JSON block per turn.

Provider handling: candidates are tried in tier order; if one fails mid-loop
(quota, network), the conversation continues on the next candidate.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from emberforge import TIER_SMART_FREE
from emberforge.compressor import EmberCompressor
from emberforge.providers import get_providers_at_or_above_tier
from emberforge.providers.base import BaseProvider, EmberResponse, ToolCall
from emberforge.tools import ToolExecutor, ToolResult

AGENT_SYSTEM_PROMPT = """You are EMBERFORGE, an autonomous coding agent working inside the user's repository.

You have tools: read_file, write_file, edit_file, list_dir, grep_search, run_shell.

WORK LOOP:
1. EXPLORE — locate relevant code with list_dir / grep_search, then read_file.
   TOKEN DISCIPLINE: read mode='signatures' FIRST to see a file's shape, then
   mode='full' only for the parts you actually need (use start_line to page).
   Re-reading an unchanged file returns a short [cached] marker — the content
   is already in this conversation; pass force=true only if you truly lost it.
2. ACT — make minimal, precise changes with edit_file / write_file.
3. VERIFY — run tests or the program with run_shell to confirm the change works.
4. When done, reply with a short plain-text summary of what you changed and
   how you verified it. Do NOT call a tool in your final reply.

RULES:
- Always read a file before editing it; edit_file requires an exact match.
- Keep edits minimal. Do not reformat code you are not changing.
- Prefer the simplest change that works: reuse existing code, the standard
  library, or an installed dependency before writing something new.
- If a tool returns an error, adapt — do not repeat the identical call.
- Never invent file contents; if you have not read it, read it.
- Be terse. No preamble, no restating the task."""

REACT_SUFFIX = """

TOOL PROTOCOL (this model has no native tool calling):
To use a tool, reply with ONLY a fenced block, exactly one per turn:
```tool
{"name": "<tool_name>", "arguments": {<args>}}
```
The result will come back in the next user message.
When finished, reply with your plain-text summary and NO tool block."""

_REACT_RE = re.compile(r"```tool\s*\n(.*?)```", re.DOTALL)


@dataclass
class AgentResult:
    content:        str
    success:        bool
    steps:          int
    tool_calls_made: int
    files_changed:  list[str] = field(default_factory=list)
    provider:       str = ""
    model:          str = ""
    tokens_in:      int = 0
    tokens_out:     int = 0
    latency_ms:     int = 0
    error:          str = ""


class EmberAgent:
    """
    The agent loop. Keeps `self.messages` across run() calls, so a REPL can
    reuse one instance for a whole conversation.
    """

    def __init__(
        self,
        providers:   dict[str, BaseProvider],
        executor:    ToolExecutor,
        compressor:  EmberCompressor | None = None,
        max_steps:   int  = 25,
        max_tokens:  int  = 4096,
        verbose:     bool = True,
        min_tier:    str  = TIER_SMART_FREE,
        compact_threshold_tokens: int = 12_000,
        system_prompt: str | None = None,
        mcp=None,   # optional MCPManager: external tools joining the loop
    ):
        self._providers  = providers
        self._executor   = executor
        self._compressor = compressor or EmberCompressor()
        self.max_steps   = max_steps
        self.max_tokens  = max_tokens
        self.verbose     = verbose
        self.min_tier    = min_tier
        self.compact_threshold = compact_threshold_tokens
        self.system_prompt = system_prompt or AGENT_SYSTEM_PROMPT
        self._mcp = mcp

        self.messages: list[dict] = []
        self._console = None
        if verbose:
            from rich.console import Console
            self._console = Console()

    # ── public API ────────────────────────────────────────────────────────────
    async def run(self, task: str, context: str = "") -> AgentResult:
        """Run (or continue) the agent conversation with a new user task."""
        t_start = time.time()

        if not self.messages:
            self.messages.append({"role": "system", "content": self.system_prompt})

        user_content = f"<context>\n{context}\n</context>\n\n{task}" if context else task
        self.messages.append({"role": "user", "content": user_content})

        candidates = self._candidates()
        if not candidates:
            return AgentResult(
                content="No providers available. Check API keys in ~/.emberforge/config.yaml",
                success=False, steps=0, tool_calls_made=0,
                error="no_providers",
            )

        tokens_in = tokens_out = 0
        tool_calls_made = 0
        provider_idx = 0
        last_provider: BaseProvider | None = None
        last_error = ""

        for step in range(1, self.max_steps + 1):
            self._compact_history()

            # ── pick a live provider (rotate on failure) ──────────────────────
            response: EmberResponse | None = None
            while provider_idx < len(candidates):
                provider = candidates[provider_idx]
                last_provider = provider

                use_native = provider.supports_tools
                msgs = self._messages_for(provider, native=use_native)
                if use_native:
                    tools = self._executor.schemas + (
                        self._mcp.schemas if self._mcp is not None else [])
                else:
                    tools = None

                self._say(f"[dim]  step {step}: {provider.name} "
                          f"({'native tools' if use_native else 'ReAct'})[/dim]")
                try:
                    response = await provider.chat(
                        messages=msgs, tools=tools, max_tokens=self.max_tokens,
                    )
                except NotImplementedError:
                    response = EmberResponse(
                        content="", provider=provider.name, model="",
                        success=False, error="chat() not implemented",
                    )

                if response.success:
                    break

                last_error = response.error
                if response.error.startswith("tools_unsupported"):
                    # same provider, retry immediately in ReAct mode
                    provider.supports_tools = False   # ensure the switch sticks
                    self._say(f"[dim yellow]  {provider.name}: no native tools — "
                              f"switching to ReAct[/dim yellow]")
                    continue

                self._say(f"[dim red]  ✗ {provider.name}: {response.error[:80]} — "
                          f"rotating provider[/dim red]")
                provider_idx += 1
                response = None

            if response is None:
                return AgentResult(
                    content=f"All providers failed. Last error: {last_error}",
                    success=False, steps=step, tool_calls_made=tool_calls_made,
                    files_changed=list(self._executor.files_changed),
                    error=last_error or "all_providers_failed",
                    latency_ms=int((time.time() - t_start) * 1000),
                )

            tokens_in  += response.tokens_in
            tokens_out += response.tokens_out

            # ── native tool calls ─────────────────────────────────────────────
            if last_provider.supports_tools and response.has_tool_calls:
                self.messages.append(self._assistant_msg(response))
                for tc in response.tool_calls:
                    output = await self._dispatch_tool(tc.name, tc.arguments)
                    tool_calls_made += 1
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                        "_emberforge_meta": self._tool_meta(tc.name, tc.arguments),
                    })
                continue

            # ── ReAct text protocol ───────────────────────────────────────────
            if not last_provider.supports_tools:
                parsed = self._parse_react(response.content)
                if parsed is not None:
                    name, args = parsed
                    self.messages.append({"role": "assistant", "content": response.content})
                    output = await self._dispatch_tool(name, json.dumps(args))
                    tool_calls_made += 1
                    self.messages.append({
                        "role": "user",
                        "content": f"TOOL RESULT ({name}):\n{output}",
                        "_emberforge_meta": self._tool_meta(name, json.dumps(args)),
                    })
                    continue

            # ── no tool call → final answer ───────────────────────────────────
            self.messages.append({"role": "assistant", "content": response.content})
            return AgentResult(
                content=response.content,
                success=True,
                steps=step,
                tool_calls_made=tool_calls_made,
                files_changed=list(self._executor.files_changed),
                provider=response.provider,
                model=response.model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=int((time.time() - t_start) * 1000),
            )

        # ── budget exhausted ──────────────────────────────────────────────────
        return AgentResult(
            content=(
                "Step budget reached before the task finished. Work so far may be "
                "partial — review files changed: "
                + (", ".join(self._executor.files_changed) or "none")
            ),
            success=False,
            steps=self.max_steps,
            tool_calls_made=tool_calls_made,
            files_changed=list(self._executor.files_changed),
            provider=last_provider.name if last_provider else "",
            model="",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=int((time.time() - t_start) * 1000),
            error="step_budget_exhausted",
        )

    def reset(self) -> None:
        """Clear conversation history (start a fresh task)."""
        self.messages.clear()

    # ── internals ─────────────────────────────────────────────────────────────
    def _candidates(self) -> list[BaseProvider]:
        cands = get_providers_at_or_above_tier(self._providers, self.min_tier)
        if not cands:   # fall back to anything configured
            cands = [p for p in self._providers.values() if p.is_available()]
        return cands

    async def _dispatch_tool(self, name: str, arguments: str) -> str:
        """Route a tool call: external MCP tools vs local filesystem tools."""
        if self._mcp is not None and self._mcp.owns(name):
            self._say(f"[dim cyan]    ⚙ {name}({arguments[:80]}) [MCP][/dim cyan]")
            output, ok = await self._mcp.call(name, arguments)
            self._executor.record_external(name, arguments, output, ok)
            icon = "✓" if ok else "✗"
            self._say(f"[dim]    {icon} {(output.splitlines() or [''])[0][:100]}[/dim]")
            return output
        return self._run_tool(name, arguments).output

    def _run_tool(self, name: str, arguments: str) -> ToolResult:
        self._say(f"[dim cyan]    ⚙ {name}({arguments[:80]})[/dim cyan]")
        result = self._executor.execute(name, arguments)
        icon = "✓" if result.success else "✗"
        self._say(f"[dim]    {icon} {result.output.splitlines()[0][:100] if result.output else ''}[/dim]")
        return result

    def _assistant_msg(self, response: EmberResponse) -> dict:
        return {
            "role": "assistant",
            "content": response.content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in response.tool_calls
            ],
        }

    def _tool_meta(self, name: str, arguments: str) -> dict:
        """Internal bookkeeping attached to tool-result messages (never sent to APIs)."""
        path = ""
        try:
            path = json.loads(arguments or "{}").get("path", "")
        except json.JSONDecodeError:
            pass
        return {"tool": name, "path": path}

    def _messages_for(self, provider: BaseProvider, native: bool) -> list[dict]:
        """
        Build the wire-format message list: always a sanitized copy (internal
        _emberforge_meta stripped). ReAct providers additionally get the protocol
        appended to the system prompt and tool messages flattened to text.
        """
        if native:
            return [
                {k: v for k, v in m.items() if not k.startswith("_")}
                for m in self.messages
            ]

        react_suffix = REACT_SUFFIX
        if self._mcp is not None and self._mcp.schemas:
            ext = "\n".join(
                f"- {s['function']['name']}: {s['function']['description'][:100]}"
                for s in self._mcp.schemas
            )
            react_suffix += f"\n\nEXTERNAL TOOLS (same ```tool protocol):\n{ext}"

        msgs: list[dict] = []
        for m in self.messages:
            if m["role"] == "system":
                msgs.append({"role": "system", "content": m["content"] + react_suffix})
            elif m["role"] == "tool":
                msgs.append({"role": "user", "content": f"TOOL RESULT:\n{m['content']}"})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                calls = "\n".join(
                    f"```tool\n{json.dumps({'name': tc['function']['name'], 'arguments': json.loads(tc['function']['arguments'] or '{}')})}\n```"
                    for tc in m["tool_calls"]
                )
                msgs.append({"role": "assistant", "content": (m.get("content") or "") + "\n" + calls})
            else:
                msgs.append({"role": m["role"], "content": m.get("content") or ""})
        return msgs

    def _parse_react(self, content: str) -> tuple[str, dict] | None:
        match = _REACT_RE.search(content or "")
        if not match:
            return None
        try:
            data = json.loads(match.group(1).strip())
            return data["name"], data.get("arguments", {})
        except (json.JSONDecodeError, KeyError):
            return None

    def _compact_history(self) -> None:
        """
        Keep context small: when history exceeds the threshold, truncate old
        tool outputs (they were only needed at the step that requested them).
        The system prompt and the most recent 8 messages stay intact.

        When a read_file result is truncated, its content is no longer in
        context — so the tool layer's read cache is invalidated for that path,
        letting the model get full content again instead of a [cached] marker.
        """
        est_tokens = sum(len(str(m.get("content") or "")) for m in self.messages) // 4
        if est_tokens <= self.compact_threshold:
            return

        keep_tail = 8
        for m in self.messages[1:-keep_tail]:
            content = m.get("content") or ""
            is_tool_msg = (
                m["role"] == "tool"
                or (m["role"] == "user" and content.startswith("TOOL RESULT"))
            )
            if is_tool_msg and len(content) > 200 and "...[compacted:" not in content:
                m["content"] = content[:200] + "\n...[compacted: old tool output]"
                meta = m.get("_emberforge_meta") or {}
                if meta.get("tool") == "read_file" and meta.get("path"):
                    self._executor.invalidate_read(meta["path"])

    def _say(self, msg: str) -> None:
        if self._console:
            self._console.print(msg)
