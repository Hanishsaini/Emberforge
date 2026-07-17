<div align="center">

```
███████╗███╗   ███╗██████╗ ███████╗██████╗
██╔════╝████╗ ████║██╔══██╗██╔════╝██╔══██╗
█████╗  ██╔████╔██║██████╔╝█████╗  ██████╔╝
██╔══╝  ██║╚██╔╝██║██╔══██╗██╔══╝  ██╔══██╗
███████╗██║ ╚═╝ ██║██████╔╝███████╗██║  ██║
╚══════╝╚═╝     ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝
        F  ·  O  ·  R  ·  G  ·  E
```

# EmberForge

**Free · Open-source · Routing & Generation Engine**

A self-improving agentic coding harness. Built to make Claude Code–level capability free for everyone.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-169%20passing-green.svg)](tests/)

</div>

---

## What Is EmberForge?

Claude Code costs $20+/month and hits rate limits constantly. EmberForge is the engineering answer — a terminal-native coding **agent harness** that:

- **Acts autonomously** — explores your repo, edits files, and runs tests in a loop; every edit and shell command is gated behind a diff-preview approval
- **Routes intelligently** across 7 free LLM providers (Groq, Gemini, NVIDIA NIM, OpenCode Zen, Mistral, OpenRouter, local Ollama)
- **Compresses measurably** — AST/signature code compression, shell output collapse, read caching. 63–98% fewer tokens, measured by `emberforge bench` (table below)
- **Falls back and recovers** — 429s honor Retry-After, failing providers cool down and come back automatically, refusals and truncated replies rotate to the next provider
- **Remembers and recalls** — decisions, sessions, and failure traces in SQLite are injected back into future runs, so the agent doesn't repeat known dead ends
- **Learns skills** — after repeated successful sessions of the same task type, a deduplicated SKILL.md is generated and searched into future context

Zero cost. No subscriptions. No rate limit anxiety.

---

## Install

Not on PyPI yet — install from source:

```bash
git clone https://github.com/Hanishsaini/forge.git emberforge
cd emberforge
pip install -e .
emberforge init
```

---

## 60-Second Setup

```bash
# 1. Install (see above)

# 2. Configure (interactive — paste your free API keys)
emberforge init

# 3. Check providers
emberforge status

# 4. Start building
emberforge agent "fix the failing tests in this repo"
emberforge agent "add type hints to the parser module"
emberforge run "why is my AST compressor failing on decorated functions"
emberforge chat     # interactive agent session
```

---

## Free API Keys (All Free Tier)

| Provider | Get Key | Tier | Best For |
|---|---|---|---|
| Groq | [console.groq.com/keys](https://console.groq.com/keys) | fast_free | Debug loops, quick fixes |
| Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | smart_free | General coding |
| NVIDIA NIM | [build.nvidia.com](https://build.nvidia.com/settings/api-keys) | best_free | Architecture, research |
| OpenCode Zen | [opencode.ai/auth](https://opencode.ai/auth) | smart_free | General coding |
| Mistral | [console.mistral.ai](https://console.mistral.ai) | smart_free | Code generation |
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) | smart_free | Auto-rotating free models |
| Ollama | (local, no key) | local | Autocomplete, simple tasks |

---

## How It Works

Two modes: `agent` (the harness — acts on your repo) and `run` (one-shot Q&A
with compressed codebase context). Both share the same router, memory, and
compression engine.

```
emberforge agent "fix the failing tests"
        ↓
┌─────────────────────────────────────────────────┐
│                  AGENT LOOP                     │
│                                                 │
│  recall  → skills (FTS5) + project decisions    │
│            + past-failure warnings              │
│  explore → grep_search / list_dir / read_file   │
│            (signatures first; re-reads cached)  │
│  act     → edit_file / write_file               │
│            (diff preview → y/n approval)        │
│  verify  → run_shell (output compressed:        │
│            passes collapsed, failures kept)     │
│  repeat  → until done, or step budget hit       │
│                                                 │
│  every LLM call goes through the router:        │
│   classify (regex heuristics → local judge      │
│   when unsure) → pick provider by tier /        │
│   health / latency → on 429/5xx/refusal/        │
│   truncation: cooldown + rotate provider,       │
│   conversation continues where it left off      │
└─────────────────────────────────────────────────┘
        ↓
summary + files changed + token stats
        ↓
memory: session saved, decisions logged,
skills generated from repeated successes
```

---

## Commands

```bash
emberforge agent "task"        # AGENT MODE: explores repo, edits files, runs tests in a loop
emberforge chat                # interactive agent REPL (conversation persists)
emberforge run "task"          # one-shot Q&A with codebase context
emberforge init                # setup wizard — configure API keys
emberforge status              # provider health check
emberforge providers           # list all providers + tiers
emberforge skills              # list all learned skills
emberforge skills --search "AST"  # search skills
emberforge learn               # force skill generation from recent sessions
emberforge stats               # lifetime token stats + savings
emberforge bench               # run the compression benchmark (measured numbers)
```

Agent mode gates every file edit and shell command behind a y/n approval with
a diff preview — pass `--yes` to auto-approve. Destructive commands
(`rm -rf /`, force-push, `git reset --hard`) are blocked outright.

### Flags

```bash
emberforge agent "task" --yes           # auto-approve edits & shell commands
emberforge agent "task" --max-steps 40  # raise the loop budget (default 25)
emberforge run "task" --project myapp   # specify project
emberforge run "task" --repo /path/to/repo
emberforge run "task" --no-context      # skip codebase context
emberforge run "task" --full            # full files (no compression)
emberforge run "task" --no-stream       # disable token streaming
emberforge run "task" --quiet           # hide routing logs
emberforge run "task" --max-tokens 8192
```

---

## Architecture

```
emberforge/
├── cli.py              # Typer CLI — all commands
├── core.py             # Ember orchestrator — wires everything
├── agent.py            # The harness loop: explore → act → verify, ReAct fallback
├── tools/
│   └── __init__.py     # read/write/edit/list/grep/shell + approval gates + read cache
├── providers/
│   ├── base.py         # BaseProvider + EmberResponse + cooldown-based health
│   └── openai_compat.py # One class for all OpenAI-compat APIs, SSE streaming
├── router/
│   ├── classifier.py   # Heuristic classification + router-as-judge prompts
│   └── router.py       # Tier routing, quality gates, silent fallback
├── compressor/
│   ├── __init__.py     # EmberCompressor pipeline
│   ├── shell.py        # Git/pip/npm/pytest output compression
│   ├── ast_compress.py # Python AST signature extraction
│   ├── polyglot.py     # JS/TS/Go/Rust/Java signature extraction
│   └── tokens.py       # Exact token counting (tiktoken, chars/4 fallback)
├── context/
│   └── __init__.py     # Codebase context engine (keyword-scored fallback)
├── memory/
│   └── __init__.py     # SQLite + FTS5: sessions, decisions, failures, recall
└── skills/
    └── __init__.py     # Gated + deduped post-task skill generation

benchmarks/             # emberforge bench — the measured-numbers pipeline
tests/                  # 169 tests: unit + 22 agent-loop workflow scenarios
```

---

## Token Savings (Measured, Not Estimated)

Produced by the real pipeline with exact tiktoken counts. Reproduce with `emberforge bench`.

| Content | Tokens before | Tokens after | Reduction |
|---|---:|---:|---:|
| Python → signatures (21 files, this repo) | 34,699 | 6,887 | **80.2%** |
| `git log --stat -n 15` (live) | 424 | 80 | **81.1%** |
| `pytest -v` output, 80 tests | 1,617 | 113 | **93.0%** |
| `pip install` dump | 1,416 | 201 | **85.8%** |
| JSON array, 50 items | 2,552 | 278 | **89.1%** |
| TypeScript → signatures | 1,017 | 292 | **71.3%** |
| Go → signatures | 684 | 252 | **63.2%** |
| Agent re-read of unchanged file (read cache) | 2,168 | 30 | **98.6%** |

Full report: [benchmarks/RESULTS.md](benchmarks/RESULTS.md)

---

## Inspiration & Credits

EmberForge is built on ideas from:
- **[Headroom](https://github.com/headroomlabs-ai/headroom)** — CacheAligner, ContentRouter, reversible CCR
- **[LeanCTX](https://github.com/leanctx/leanctx)** — Signature mode, read modes, shell pattern compression
- **[Claw-Compactor](https://github.com/openclaw/claw-compactor)** — Fusion Pipeline, simhash dedup
- **[Hermes Agent](https://github.com/NousResearch/hermes)** — Post-task skill generation loop
- **AHE (ICLR 2026)** — Agentic Harness Engineering, 7-component decomposition

---

## Roadmap

Done:
- [x] Agent mode — multi-file edit loop with approval gates (v0.2)
- [x] Router-as-judge — local model classifies tasks when heuristics are unsure (v0.3)
- [x] Memory recall — decisions, sessions, and failure traces injected into runs (v0.3)
- [x] Measured compression benchmark — `emberforge bench` (v0.2)

Next:
- [ ] PyPI release
- [ ] Task-success eval suite — measure answer quality with/without compression, not just tokens
- [ ] GEPA failure analysis — *why* did this fail, not just that it failed (failure recall shipped; analysis pending)
- [ ] AHE evolution loop — EmberForge improves its own system prompts from traces
- [ ] BM25+RRF retrieval for codebase context (current: keyword scoring)
- [ ] Cross-project persistent memory
- [ ] MCP server support — use EmberForge as an MCP provider

---

## License

MIT — built by [Honey Stark](https://github.com/Hanishsaini)
