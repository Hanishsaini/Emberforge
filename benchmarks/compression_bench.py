"""
EMBERFORGE Compression Benchmark — measured numbers, not marketing numbers.

Runs the real EmberCompressor over:
  1. Every Python file in emberforge/ (signature mode)          — AST compression
  2. Live shell output (git log/status) + canned pip/pytest — shell compression
  3. A 50-item JSON payload                                 — JSON sampling
  4. Realistic TypeScript + Go sources                      — polyglot signatures
  5. The read-cache re-read path                            — progressive disclosure

Token counts use tiktoken (cl100k_base) when available; the report says which.

Run:  emberforge bench          (or: python -m benchmarks.compression_bench)
Output: printed table + benchmarks/RESULTS.md
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from emberforge.compressor import EmberCompressor
from emberforge.compressor.tokens import count_tokens, counting_is_exact
from emberforge.tools import EmberTools

REPO_ROOT = Path(__file__).parent.parent

# ── Canned samples (deterministic across machines) ────────────────────────────
PIP_INSTALL_OUTPUT = "\n".join(
    [f"Collecting package-{i}>=1.0.0" for i in range(15)]
    + [f"  Downloading package_{i}-1.2.{j}-py3-none-any.whl (125 kB)"
       for i in range(15) for j in range(3)]
    + ["Installing collected packages: " + ", ".join(f"package-{i}" for i in range(15))]
    + ["Successfully installed " + " ".join(f"package-{i}-1.2.0" for i in range(15))]
)

PYTEST_OUTPUT = "\n".join(
    [f"tests/unit/test_module_{i}.py::test_case_{j} PASSED         [ {(i*10+j)}%]"
     for i in range(8) for j in range(10)]
    + ["", "=" * 60, "80 passed, 0 failed in 4.21s", "=" * 60]
)

TYPESCRIPT_SAMPLE = """import { Router } from 'express';
import { db } from '../db';
import { validate } from '../middleware/validate';

/** Fetch a user by id with their orders joined. */
export async function getUserWithOrders(userId: string): Promise<UserWithOrders> {
    const user = await db.users.findUnique({ where: { id: userId } });
    if (!user) {
        throw new NotFoundError(`user ${userId} not found`);
    }
    const orders = await db.orders.findMany({ where: { userId } });
    return { ...user, orders };
}

export interface UserWithOrders {
    id: string;
    email: string;
    orders: Order[];
}

export class OrderService {
    private cache = new Map<string, Order>();

    /** Get one order, cached. */
    async getOrder(id: string): Promise<Order> {
        if (this.cache.has(id)) {
            return this.cache.get(id)!;
        }
        const order = await db.orders.findUnique({ where: { id } });
        this.cache.set(id, order);
        return order;
    }

    async cancelOrder(id: string): Promise<void> {
        const order = await this.getOrder(id);
        if (order.status === 'shipped') {
            throw new ConflictError('cannot cancel a shipped order');
        }
        await db.orders.update({ where: { id }, data: { status: 'cancelled' } });
        this.cache.delete(id);
    }
}

export const MAX_RETRIES = 3;
export const handler = async (req, res) => {
    const result = await getUserWithOrders(req.params.id);
    res.json(result);
};
""" * 3   # repeat to simulate a realistically sized module

GO_SAMPLE = """package server

import (
    "encoding/json"
    "net/http"
    "time"
)

// Server wraps the HTTP mux with graceful shutdown.
type Server struct {
    mux     *http.ServeMux
    timeout time.Duration
}

// NewServer builds a Server with sane defaults.
func NewServer(timeout time.Duration) *Server {
    s := &Server{mux: http.NewServeMux(), timeout: timeout}
    s.mux.HandleFunc("/health", s.handleHealth)
    s.mux.HandleFunc("/users", s.handleUsers)
    return s
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (s *Server) handleUsers(w http.ResponseWriter, r *http.Request) {
    users, err := s.loadUsers(r.Context())
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    json.NewEncoder(w).Encode(users)
}

const DefaultTimeout = 30 * time.Second
""" * 3


def _live_shell(cmd: str) -> str:
    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=str(REPO_ROOT), capture_output=True,
            text=True, timeout=30, encoding="utf-8", errors="ignore",
        )
        return proc.stdout or ""
    except Exception:
        return ""


def _row(name: str, category: str, original: str, final: str, note: str = "") -> dict:
    o, f = count_tokens(original), count_tokens(final)
    return {
        "name": name,
        "category": category,
        "original_tokens": o,
        "final_tokens": f,
        "reduction_pct": round(100 * (1 - f / o), 1) if o else 0.0,
        "note": note,
    }


def run_benchmark() -> list[dict]:
    compressor = EmberCompressor()
    rows: list[dict] = []

    # 1 ── Python signature compression over this very repo
    py_files = sorted((REPO_ROOT / "emberforge").rglob("*.py"))
    total_orig = total_final = 0
    for f in py_files:
        src = f.read_text(encoding="utf-8", errors="ignore")
        if not src.strip():
            continue
        result = compressor.compress(src, content_type="code",
                                     filename=str(f), mode="signatures")
        total_orig  += count_tokens(src)
        total_final += count_tokens(result.final_text)
    rows.append({
        "name": f"Python -> signatures ({len(py_files)} files, emberforge/)",
        "category": "code",
        "original_tokens": total_orig,
        "final_tokens": total_final,
        "reduction_pct": round(100 * (1 - total_final / total_orig), 1) if total_orig else 0.0,
        "note": "aggregate over this repo",
    })

    # 2 ── Shell output
    git_log = _live_shell("git log --stat -n 15")
    if git_log:
        c = compressor.compress(git_log, content_type="shell")
        rows.append(_row("git log --stat -n 15 (live)", "shell", git_log, c.final_text))
    git_status = _live_shell("git status")
    if git_status:
        c = compressor.compress(git_status, content_type="shell")
        rows.append(_row("git status (live)", "shell", git_status, c.final_text))
    c = compressor.compress(PIP_INSTALL_OUTPUT, content_type="shell")
    rows.append(_row("pip install dump (canned)", "shell", PIP_INSTALL_OUTPUT, c.final_text))
    c = compressor.compress(PYTEST_OUTPUT, content_type="shell")
    rows.append(_row("pytest output, 80 tests (canned)", "shell", PYTEST_OUTPUT, c.final_text))

    # 3 ── JSON sampling
    payload = json.dumps(
        [{"id": i, "name": f"user-{i}", "email": f"user{i}@example.com",
          "roles": ["viewer", "editor"], "active": i % 2 == 0}
         for i in range(50)],
        indent=2,
    )
    c = compressor.compress(payload, content_type="json")
    rows.append(_row("JSON array, 50 items", "json", payload, c.final_text))

    # 4 ── Polyglot signatures
    c = compressor.compress(TYPESCRIPT_SAMPLE, content_type="code",
                            filename="service.ts", mode="signatures")
    rows.append(_row("TypeScript -> signatures", "code", TYPESCRIPT_SAMPLE, c.final_text))
    c = compressor.compress(GO_SAMPLE, content_type="code",
                            filename="server.go", mode="signatures")
    rows.append(_row("Go -> signatures", "code", GO_SAMPLE, c.final_text))

    # 5 ── Read-cache: re-reading an unchanged file
    tools = EmberTools(REPO_ROOT)
    first  = tools.read_file("emberforge/core.py")
    second = tools.read_file("emberforge/core.py")   # unchanged → [cached] marker
    rows.append(_row("re-read unchanged file (read cache)", "agent",
                     first.output, second.output,
                     note="agent loop: 2nd read of emberforge/core.py"))

    return rows


def render_markdown(rows: list[dict]) -> str:
    exact = counting_is_exact()
    lines = [
        "# EMBERFORGE Compression Benchmark — Measured Results",
        "",
        f"- Date: {time.strftime('%Y-%m-%d')}",
        f"- Token counting: {'tiktoken cl100k_base (exact)' if exact else 'chars/4 (estimate — install tiktoken for exact)'}",
        f"- Reproduce: `emberforge bench` or `python -m benchmarks.compression_bench`",
        "",
        "| Content | Tokens before | Tokens after | Reduction |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['original_tokens']:,} | "
            f"{r['final_tokens']:,} | **{r['reduction_pct']}%** |"
        )
    lines += ["", "_Generated by benchmarks/compression_bench.py — numbers above are "
              "produced by the actual pipeline, not estimated._"]
    return "\n".join(lines)


def main() -> list[dict]:
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    console.print("[bold cyan]EMBERFORGE Compression Benchmark[/bold cyan]")
    console.print(f"[dim]token counting: "
                  f"{'tiktoken (exact)' if counting_is_exact() else 'chars/4 (estimate)'}[/dim]\n")

    rows = run_benchmark()

    table = Table(box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Content")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Reduction", justify="right", style="bold green")
    for r in rows:
        table.add_row(r["name"], f"{r['original_tokens']:,}",
                      f"{r['final_tokens']:,}", f"{r['reduction_pct']}%")
    console.print(table)

    out = REPO_ROOT / "benchmarks" / "RESULTS.md"
    out.write_text(render_markdown(rows), encoding="utf-8")
    console.print(f"\n[dim]Written to {out}[/dim]")
    return rows


if __name__ == "__main__":
    main()
