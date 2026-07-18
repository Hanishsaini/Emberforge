"""
EmberForge MCP Client — let the agent use tools from external MCP servers.

MCPClient  : one server — spawn, handshake, tools/list, tools/call, close.
MCPManager : many servers — merged OpenAI-format schemas namespaced
             mcp__{server}__{tool}, routing, and trace-friendly results.

A dead, slow, or misbehaving external server must never kill the agent loop:
every failure path returns an error string instead of raising.
"""
from __future__ import annotations

import asyncio
import json

from emberforge import __version__
from emberforge.mcp.server import PROTOCOL_VERSION

NAMESPACE_PREFIX = "mcp__"
NAMESPACE_SEP    = "__"


class MCPClient:
    """One external MCP server over stdio (newline-delimited JSON-RPC)."""

    def __init__(
        self,
        name:    str,
        command: str,
        args:    list[str] | None = None,
        timeout: float = 30.0,
        cwd:     str | None = None,
    ):
        self.name    = name
        self.command = command
        self.args    = args or []
        self.timeout = timeout
        self.cwd     = cwd
        self.tools:  list[dict] = []
        self._proc = None
        self._id   = 0

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=self.cwd,
        )
        await self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "emberforge", "version": __version__},
        })
        await self._notify("notifications/initialized")
        listing = await self._request("tools/list", {})
        self.tools = listing.get("tools", [])

    async def _write(self, obj: dict) -> None:
        self._proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

    async def _notify(self, method: str, params: dict | None = None) -> None:
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params:
            msg["params"] = params
        await self._write(msg)

    async def _request(self, method: str, params: dict) -> dict:
        self._id += 1
        msg_id = self._id
        await self._write({"jsonrpc": "2.0", "id": msg_id,
                           "method": method, "params": params})
        while True:
            line = await asyncio.wait_for(
                self._proc.stdout.readline(), timeout=self.timeout)
            if not line:
                raise RuntimeError(f"MCP server '{self.name}' closed the pipe")
            try:
                msg = json.loads(line.decode("utf-8", "ignore"))
            except json.JSONDecodeError:
                continue                    # skip non-JSON noise on stdout
            if msg.get("id") != msg_id:
                continue                    # notification or stale response
            if "error" in msg:
                err = msg["error"]
                raise RuntimeError(
                    f"MCP error {err.get('code')}: {err.get('message', '')}")
            return msg.get("result", {})

    async def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        """Returns (text, is_error)."""
        result = await self._request("tools/call",
                                     {"name": name, "arguments": arguments})
        text = "\n".join(
            c.get("text", "") for c in result.get("content", [])
            if c.get("type") == "text"
        )
        return text, bool(result.get("isError"))

    async def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            await asyncio.wait_for(self._proc.wait(), timeout=5)
        except (ProcessLookupError, asyncio.TimeoutError):
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass


class MCPManager:
    """
    Fleet of MCP clients presented to the agent as extra tools.
    Tool names are namespaced mcp__{server}__{tool} to avoid collisions with
    local tools and between servers.
    """

    def __init__(self, servers: dict[str, dict]):
        """servers: name -> {"command": str, "args": [str], "cwd": str|None}"""
        self._clients: dict[str, MCPClient] = {
            name: MCPClient(
                name, cfg["command"], cfg.get("args") or [], cwd=cfg.get("cwd"),
            )
            for name, cfg in servers.items()
        }

    async def connect(self) -> list[str]:
        """Start every server; a failing server is reported, not fatal."""
        errors: list[str] = []
        for name, client in list(self._clients.items()):
            try:
                await asyncio.wait_for(client.start(), timeout=client.timeout)
            except Exception as e:
                errors.append(f"{name}: {str(e)[:120]}")
                del self._clients[name]
        return errors

    @property
    def connected(self) -> list[str]:
        return list(self._clients)

    @property
    def schemas(self) -> list[dict]:
        out: list[dict] = []
        for server_name, client in self._clients.items():
            for tool in client.tools:
                out.append({
                    "type": "function",
                    "function": {
                        "name": f"{NAMESPACE_PREFIX}{server_name}{NAMESPACE_SEP}{tool['name']}",
                        "description": (
                            f"{tool.get('description', '')} "
                            f"[external MCP tool from '{server_name}']"
                        ).strip()[:1024],
                        "parameters": tool.get("inputSchema")
                                      or {"type": "object", "properties": {}},
                    },
                })
        return out

    def owns(self, name: str) -> bool:
        return name.startswith(NAMESPACE_PREFIX)

    async def call(self, prefixed: str, arguments: str | dict) -> tuple[str, bool]:
        """Returns (output, success). Never raises — agent loops must survive
        any external server behavior."""
        rest = prefixed[len(NAMESPACE_PREFIX):]
        server_name, sep, tool_name = rest.partition(NAMESPACE_SEP)
        if not sep or server_name not in self._clients:
            return f"Unknown MCP tool: {prefixed}", False

        if isinstance(arguments, str):
            try:
                args = json.loads(arguments or "{}")
            except json.JSONDecodeError as e:
                return f"Invalid JSON arguments for {prefixed}: {e}", False
        else:
            args = dict(arguments)

        try:
            text, is_error = await self._clients[server_name].call_tool(tool_name, args)
            return text, not is_error
        except (RuntimeError, asyncio.TimeoutError, OSError) as e:
            return f"MCP call to '{prefixed}' failed: {str(e)[:200]}", False

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()
