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

### An open-source coding agent that runs on free LLM tiers. <br> Claude Code–style workflow. $0/month.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-169%20passing-brightgreen.svg)](tests/)
[![Version](https://img.shields.io/badge/version-0.3.0-orange.svg)](pyproject.toml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-ff69b4.svg)](#contributing)

**[Quickstart](#-quickstart)** · **[How it works](#-how-it-works)** · **[Measured savings](#-token-savings-measured-not-estimated)** · **[Roadmap](#-roadmap)**

</div>

---

Coding agents are the best thing to happen to developer productivity — and they cost $20+/month, which locks out students, hobbyists, and most of the world. **EmberForge is a terminal coding agent that reads your repo, edits files, and runs your tests in a loop — powered entirely by free LLM tiers** (Groq, Gemini, NVIDIA NIM, Mistral, OpenRouter, local Ollama). When one provider hits its quota, the conversation continues on the next. You never notice.

```console
$ emberforge agent "add input validation to the signup endpoint"

→ Task: write | Tier: smart_free
  step 1: groq (native tools)
    ⚙ grep_search("signup")
    ⚙ read_file("api/routes.py", mode="signatures")
  step 3: groq
    ⚙ edit_file("api/routes.py")        ← diff preview → you approve
    ⚙ run_shell("pytest tests/ -q")
    ✓ 12 passed

Added email + password validation to /signup, verified with tests.
⚡ groq · 5 steps · 6 tool calls · 8,214→612 tokens · 📝 api/routes.py
```

## 🆚 Why EmberForge?

| | Claude Code | EmberForge |
|---|---|---|
| 💰 Price | $20+/month | **$0** |
| 🧠 Models | Claude only | **7 free providers + local Ollama** |
| 🚧 Rate limits | hard caps, "try again later" | **auto-rotates across quotas, cooldowns recover** |
| 🔍 Source | closed | **MIT, ~4k lines, readable in an evening** |
| 🎯 Maturity | polished product | v0.3 — young, moving fast, honest about it |

EmberForge is not a Claude Code clone with a different API key — it's a bet that **aggressive context engineering + smart routing across many free tiers** can deliver real agentic coding without a subscription. The token math below is how the bet gets paid.

## ✨ What you get

- 🤖 **A real agent loop** — explore → edit → run tests → repeat, with a step budget. Every file edit shows you a diff and asks first (`--yes` to skip). Destructive commands (`rm -rf /`, force-push) are hard-blocked.
- 🔀 **Smart routing** — tasks are classified (regex heuristics, escalating to a local Ollama judge when unsure) and sent to the cheapest tier that can handle them. 429s honor `Retry-After`; failing providers cool down and auto-recover; refusals and truncated replies rotate silently.
- 🗜️ **Measured compression** — Python/TS/Go/Rust/Java files collapse to signatures, shell output keeps failures and drops noise, unchanged re-reads cost 30 tokens instead of 2,000. **63–98% fewer tokens, verified by `emberforge bench`.**
- 🧠 **Memory that recalls** — decisions, sessions, and failure traces persist in SQLite and get injected into future runs, so the agent doesn't repeat known dead ends.
- 📈 **Self-improving** — repeated successful sessions of the same task type generate searchable, deduplicated SKILL.md files that feed future context.
- 💬 **Two more modes** — `emberforge chat` (persistent REPL) and `emberforge run` (streamed one-shot Q&A with compressed repo context).

## 🚀 Quickstart

```bash
# install (PyPI release coming — from source for now)
git clone https://github.com/Hanishsaini/forge.git emberforge
cd emberforge && pip install -e .

# configure — paste any free API keys you have (one is enough to start)
emberforge init

# go
emberforge status
emberforge agent "fix the failing tests in this repo"
emberforge chat
```

### Free API keys (all $0 tiers)

| Provider | Get a key | Tier | Best for |
|---|---|---|---|
| Groq | [console.groq.com/keys](https://console.groq.com/keys) | fast_free | Debug loops, quick fixes |
| Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | smart_free | General coding |
| NVIDIA NIM | [build.nvidia.com](https://build.nvidia.com/settings/api-keys) | best_free | Architecture, research |
| OpenCode Zen | [opencode.ai/auth](https://opencode.ai/auth) | smart_free | General coding |
| Mistral | [console.mistral.ai](https://console.mistral.ai) | smart_free | Code generation |
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) | smart_free | Auto-rotating free models |
| Ollama | local — no key | local | Task classification, autocomplete |

## ⚙️ How it works

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

Models without native function calling fall back to a ReAct text protocol automatically — EmberForge detects the rejection at runtime and switches, mid-conversation.

## 📊 Token savings (measured, not estimated)

Every number below is produced by the real pipeline with exact tiktoken counts — no marketing math. Reproduce with `emberforge bench`.

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

Full report: [benchmarks/RESULTS.md](benchmarks/RESULTS.md). Fun fact: the first run of this benchmark caught our own shell compressor doing **0%** on pytest output. We fixed it, then published the numbers. That's the standard here.

## 🧰 Commands

```bash
emberforge agent "task"     # THE HARNESS: explore, edit, test in a loop
emberforge chat             # interactive agent REPL (context persists)
emberforge run "task"       # one-shot Q&A with compressed repo context
emberforge init             # setup wizard — configure API keys
emberforge status           # provider health + cooldowns
emberforge bench            # run the compression benchmark yourself
emberforge skills           # list learned skills (--search "AST")
emberforge stats            # lifetime token stats + savings
```

<details>
<summary><b>All flags</b></summary>

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
</details>

<details>
<summary><b>Architecture</b> — ~4k lines, built to be read</summary>

```
emberforge/
├── cli.py              # Typer CLI — all commands
├── core.py             # Ember orchestrator — wires everything
├── agent.py            # The harness loop: explore → act → verify, ReAct fallback
├── tools/              # read/write/edit/list/grep/shell + approval gates + read cache
├── providers/
│   ├── base.py         # BaseProvider + cooldown-based health
│   └── openai_compat.py # One class for all OpenAI-compat APIs, SSE streaming
├── router/
│   ├── classifier.py   # Heuristic classification + router-as-judge prompts
│   └── router.py       # Tier routing, quality gates, silent fallback
├── compressor/
│   ├── shell.py        # Git/pip/npm/pytest output compression
│   ├── ast_compress.py # Python AST signature extraction
│   ├── polyglot.py     # JS/TS/Go/Rust/Java signature extraction
│   └── tokens.py       # Exact token counting (tiktoken, chars/4 fallback)
├── context/            # Codebase context engine (keyword-scored)
├── memory/             # SQLite + FTS5: sessions, decisions, failures, recall
└── skills/             # Gated + deduped post-task skill generation

benchmarks/             # emberforge bench — the measured-numbers pipeline
tests/                  # 169 tests, incl. 22 agent-loop workflow scenarios
```
</details>

## 🗺️ Roadmap

**Shipped:** agent mode with approval gates (v0.2) · measured benchmark (v0.2) · cooldown routing + streaming + router-as-judge (v0.3) · memory recall + skill dedupe (v0.3)

**Next:**
- [ ] PyPI release (`pip install emberforge`)
- [ ] Task-success eval suite — answer quality with/without compression, not just tokens
- [ ] GEPA failure analysis — *why* did it fail, not just that it failed
- [ ] AHE evolution loop — EmberForge improves its own system prompts from traces
- [ ] BM25+RRF retrieval for codebase context
- [ ] MCP server support

## 🤝 Contributing

EmberForge exists so people who can't pay $20/month still get a real coding agent. If that mission speaks to you:

- ⭐ **Star the repo** — it's genuinely how people find this
- 🐛 [Open an issue](https://github.com/Hanishsaini/forge/issues) — bug reports with `emberforge status` output are gold
- 🔧 Pick anything on the roadmap — PRs welcome; every module has tests to copy from (`python -m pytest tests/ -q`)
- 📣 Tell one person who's priced out of AI tooling

## 🙏 Credits

Built on ideas from [Headroom](https://github.com/headroomlabs-ai/headroom) (reversible compression), [LeanCTX](https://github.com/leanctx/leanctx) (signature reads), [Claw-Compactor](https://github.com/openclaw/claw-compactor) (fusion pipeline, simhash), [Hermes](https://github.com/NousResearch/hermes) (post-task skill generation), and AHE (ICLR 2026, agentic harness engineering).

## 📄 License

[MIT](LICENSE) — free, forever, like the tools it runs on. Built by [Hanish Saini](https://github.com/Hanishsaini).
