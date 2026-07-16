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
[![Tests](https://img.shields.io/badge/tests-34%20passing-green.svg)](tests/)

</div>

---

## What Is EMBERFORGE?

Claude Code costs $20+/month and hits rate limits constantly. EMBERFORGE is the engineering answer — a terminal-native coding harness that:

- **Routes intelligently** across 10+ free LLM providers (Groq, Gemini, NVIDIA NIM, OpenCode, Mistral, OpenRouter, Ollama)
- **Compresses aggressively** — AST-aware code compression, shell output dedup, simhash deduplication. 60–92% fewer tokens per request
- **Learns automatically** — Hermes-style post-task skill generation. Every 5 complex tasks → new SKILL.md auto-written and searchable
- **Remembers everything** — SQLite-backed persistent memory across sessions. Architecture decisions, project context, failure traces
- **Falls back silently** — quota hit on Groq? Switches to Gemini. Gemini slow? Routes to NVIDIA NIM. You never notice

Zero cost. No subscriptions. No rate limit anxiety.

---

## Install

```bash
pip install emberforge
emberforge init
```

---

## 60-Second Setup

```bash
# 1. Install
pip install emberforge

# 2. Configure (interactive — paste your free API keys)
emberforge init

# 3. Check providers
emberforge status

# 4. Start building
emberforge "refactor the retrieval pipeline in codelore"
emberforge "why is my AST compressor failing on decorated functions"
emberforge "write pytest tests for the memory module"
emberforge "design the multi-tier routing architecture"
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

```
emberforge "your task"
        ↓
┌──────────────────────────────────────────┐
│              EMBERFORGE PIPELINE              │
│                                          │
│  1. Skill Lookup (FTS5)                  │
│     → load relevant past learnings       │
│                                          │
│  2. Codebase Context (CodeLore/BM25+RRF) │
│     → pull only relevant files           │
│                                          │
│  3. Compression Pipeline                 │
│     → AST signatures (60-70% reduction) │
│     → Shell output dedup (85% reduction)│
│     → Simhash deduplication             │
│                                          │
│  4. Task Classification                  │
│     → simple/debug/write/architecture   │
│     → maps to minimum provider tier     │
│                                          │
│  5. Smart Provider Routing               │
│     Local → Fast Free → Smart → Best    │
│     Auto-fallback on quota/failure      │
│                                          │
│  6. Memory + Skill Generation            │
│     → store session in SQLite           │
│     → auto-generate skill after 5 tasks │
└──────────────────────────────────────────┘
        ↓
   clean output + token stats
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
emberforge "task" --project myapp     # specify project
emberforge "task" --repo /path/to/repo # specify repo path
emberforge "task" --no-context        # skip codebase context
emberforge "task" --full              # use full files (no compression)
emberforge "task" --quiet             # hide routing logs
emberforge "task" --max-tokens 8192   # override token limit
```

---

## Architecture

```
emberforge/
├── cli.py              # Typer CLI — all commands
├── core.py             # Ember orchestrator — wires everything
├── providers/
│   ├── base.py         # BaseProvider + EmberResponse + ProviderHealth
│   └── openai_compat.py # Single class for all OpenAI-compat APIs
├── router/
│   ├── classifier.py   # Task type + tier classification
│   └── router.py       # Smart multi-provider routing with fallback
├── compressor/
│   ├── __init__.py     # EmberCompressor pipeline
│   ├── shell.py        # Git/pip/npm/pytest output compression
│   └── ast_compress.py # Python AST signature extraction
├── context/
│   └── __init__.py     # Codebase context engine (CodeLore/fallback)
├── memory/
│   └── __init__.py     # SQLite + FTS5 memory backend
└── skills/
    └── __init__.py     # Hermes-style post-task skill generation
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

EMBERFORGE is built on ideas from:
- **[Headroom](https://github.com/headroomlabs-ai/headroom)** — CacheAligner, ContentRouter, reversible CCR
- **[LeanCTX](https://github.com/leanctx/leanctx)** — Signature mode, 10 read modes, shell pattern compression
- **[Claw-Compactor](https://github.com/openclaw/claw-compactor)** — 14-stage Fusion Pipeline, simhash dedup
- **[Hermes Agent](https://github.com/NousResearch/hermes)** — Post-task skill generation loop
- **AHE (ICLR 2026)** — Agentic Harness Engineering, 7-component decomposition

---

## Roadmap

- [ ] `emberforge learn` — GEPA failure analysis (why did this fail, not just that it failed)
- [ ] Router-as-judge — Ollama qwen:7b classifies tasks instead of heuristics
- [ ] AHE evolution loop — EMBERFORGE improves its own system prompts from traces
- [ ] CodeLore deep integration — full BM25+RRF retrieval
- [ ] Hanish OS memory wiring — cross-project persistent context
- [ ] Multi-file edit mode — Claude Code-style file diffing
- [ ] MCP server support — use EMBERFORGE as an MCP provider

---

## License

MIT — built by [Honey Stark]([https://github.com/Hanishsaini])
