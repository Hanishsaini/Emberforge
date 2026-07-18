"""EmberForge MCP — server (expose EmberForge to MCP hosts) and client
(let the agent use external MCP tools). Newline-delimited JSON-RPC 2.0
over stdio, hand-rolled: zero new dependencies, every path testable.

Imports are lazy so `python -m emberforge.mcp.server` starts cleanly.
"""


def __getattr__(name):
    if name in ("MCPServer", "serve_stdio", "handle_raw_line"):
        from emberforge.mcp import server
        return getattr(server, name)
    if name in ("MCPClient", "MCPManager"):
        from emberforge.mcp import client
        return getattr(client, name)
    raise AttributeError(f"module 'emberforge.mcp' has no attribute {name!r}")
