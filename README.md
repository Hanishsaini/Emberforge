<div align="center">

```
███████╗ ██████╗ ██████╗  ██████╗ ███████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
█████╗  ██║   ██║██████╔╝██║  ███╗█████╗  
██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  
██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
```

**Free · Open-source · Routing & Generation Engine**

A self-improving agentic coding harness. Built to replace Claude Code.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-34%20passing-green.svg)](tests/)

</div>

---

## What Is FORGE?

Claude Code costs $20+/month and hits rate limits constantly. FORGE is the engineering answer — a terminal-native coding harness that:

- **Routes intelligently** across 10+ free LLM providers (Groq, Gemini, NVIDIA NIM, OpenCode, Mistral, OpenRouter, Ollama)
- **Compresses aggressively** — AST-aware code compression, shell output dedup, simhash deduplication. 60–92% fewer tokens per request
- **Learns automatically** — Hermes-style post-task skill generation. Every 5 complex tasks → new SKILL.md auto-written and searchable
- **Remembers everything** — SQLite-backed persistent memory across sessions. Architecture decisions, project context, failure traces
- **Falls back silently** — quota hit on Groq? Switches to Gemini. Gemini slow? Routes to NVIDIA NIM. You never notice

Zero cost. No subscriptions. No rate limit anxiety.

---

## Install

```bash
pip install forge-ai
forge init
```

---

## 60-Second Setup

```bash
# 1. Install
pip install forge-ai

# 2. Configure (interactive — paste your free API keys)
forge init

# 3. Check providers
forge status

# 4. Start building
forge "refactor the retrieval pipeline in codelore"
forge "why is my AST compressor failing on decorated functions"
forge "write pytest tests for the memory module"
forge "design the multi-tier routing architecture"
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
forge "your task"
        ↓
┌──────────────────────────────────────────┐
│              FORGE PIPELINE              │
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
forge "task"              # run a task (auto-detects project)
forge init                # setup wizard — configure API keys
forge status              # provider health check
forge providers           # list all providers + tiers
forge skills              # list all learned skills
forge skills --search "AST"  # search skills
forge learn               # force skill generation from recent sessions
forge stats               # lifetime token stats + savings
```

### Flags

```bash
forge "task" --project myapp     # specify project
forge "task" --repo /path/to/repo # specify repo path
forge "task" --no-context        # skip codebase context
forge "task" --full              # use full files (no compression)
forge "task" --quiet             # hide routing logs
forge "task" --max-tokens 8192   # override token limit
```

---

## Architecture

```
forge/
├── cli.py              # Typer CLI — all commands
├── core.py             # Forge orchestrator — wires everything
├── providers/
│   ├── base.py         # BaseProvider + ForgeResponse + ProviderHealth
│   └── openai_compat.py # Single class for all OpenAI-compat APIs
├── router/
│   ├── classifier.py   # Task type + tier classification
│   └── router.py       # Smart multi-provider routing with fallback
├── compressor/
│   ├── __init__.py     # ForgeCompressor pipeline
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

## Token Savings (Real Numbers)

| Content Type | Before | After | Reduction |
|---|---|---|---|
| Python file (full → signatures) | 2,000 tokens | 200 tokens | **90%** |
| `git status` output | 800 tokens | 120 tokens | **85%** |
| `pip install` dump | 3,000 tokens | 200 tokens | **93%** |
| JSON array (50 items) | 1,500 tokens | 300 tokens | **80%** |
| Re-reading a cached file | 2,000 tokens | 13 tokens | **99%** |

---

## Inspiration & Credits

FORGE is built on ideas from:
- **[Headroom](https://github.com/headroomlabs-ai/headroom)** — CacheAligner, ContentRouter, reversible CCR
- **[LeanCTX](https://github.com/leanctx/leanctx)** — Signature mode, 10 read modes, shell pattern compression
- **[Claw-Compactor](https://github.com/openclaw/claw-compactor)** — 14-stage Fusion Pipeline, simhash dedup
- **[Hermes Agent](https://github.com/NousResearch/hermes)** — Post-task skill generation loop
- **AHE (ICLR 2026)** — Agentic Harness Engineering, 7-component decomposition

---

## Roadmap

- [ ] `forge learn` — GEPA failure analysis (why did this fail, not just that it failed)
- [ ] Router-as-judge — Ollama qwen:7b classifies tasks instead of heuristics
- [ ] AHE evolution loop — FORGE improves its own system prompts from traces
- [ ] CodeLore deep integration — full BM25+RRF retrieval
- [ ] Hanish OS memory wiring — cross-project persistent context
- [ ] Multi-file edit mode — Claude Code-style file diffing
- [ ] MCP server support — use FORGE as an MCP provider

---

## License

MIT — built by [Honey Stark]([https://github.com/Hanishsaini])
