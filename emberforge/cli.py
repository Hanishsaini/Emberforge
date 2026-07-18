"""
EMBERFORGE CLI — terminal interface, drop-in Claude Code substitute.

Commands:
  emberforge run "task"      → run a task
  emberforge init            → setup ~/.emberforge/config.yaml
  emberforge status          → show provider health
  emberforge skills          → list all learned skills
  emberforge learn           → force skill generation from recent sessions
  emberforge stats           → session + lifetime token stats
  emberforge providers       → list all providers + health
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich import box

app     = typer.Typer(
    name="emberforge",
    help="EMBERFORGE — Free, Open-source, Routing & Generation Engine",
    add_completion=False,
    invoke_without_command=False,
)
console = Console()

BANNER = """[bold red]
███████╗███╗   ███╗██████╗ ███████╗██████╗
██╔════╝████╗ ████║██╔══██╗██╔════╝██╔══██╗
█████╗  ██╔████╔██║██████╔╝█████╗  ██████╔╝
██╔══╝  ██║╚██╔╝██║██╔══██╗██╔══╝  ██╔══██╗
███████╗██║ ╚═╝ ██║██████╔╝███████╗██║  ██║
╚══════╝╚═╝     ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝[/bold red]
[bold yellow]        F  ·  O  ·  R  ·  G  ·  E[/bold yellow]
[dim]EmberForge — Free · Open-source · Routing & Generation Engine[/dim]
[dim]Built by Honey Stark — github.com/Hanishsaini/emberforge[/dim]
"""


def _get_emberforge(project: str, repo: str, verbose: bool):
    from emberforge.core import Ember
    return Ember(project=project, repo_path=repo, verbose=verbose)


def _detect_project(path: str) -> str:
    p = Path(path).resolve()
    for manifest in ("pyproject.toml", "package.json", "Cargo.toml"):
        if (p / manifest).exists():
            return p.name
    return p.name


def _print_result(result, show_stats: bool = True, show_content: bool = True) -> None:
    if not result.success:
        console.print(f"\n[bold red]❌ Error:[/bold red] {result.error}")
        return

    if show_content:
        console.print()
        if "```" in result.content:
            console.print(Markdown(result.content))
        else:
            console.print(result.content)

    if show_stats:
        console.print(
            f"\n[dim]"
            f"⚡ {result.provider} · {result.model} · "
            f"{result.tokens_in}→{result.tokens_out} tokens"
            f"{f' · saved {result.tokens_saved}' if result.tokens_saved else ''}"
            f" · {result.latency_ms}ms"
            f"{f' · {result.attempts} attempts' if result.attempts > 1 else ''}"
            f"[/dim]"
        )
        if result.skill_generated:
            console.print(f"[dim green]✨ Skill learned: '{result.skill_generated}'[/dim green]")


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command(name="run")
def run(
    prompt: str = typer.Argument(..., help="Task description"),
    project: str = typer.Option("", "--project", "-p", help="Project name"),
    repo: str = typer.Option(".", "--repo", "-r", help="Repo path"),
    no_context: bool = typer.Option(False, "--no-context", help="Skip codebase context"),
    full: bool = typer.Option(False, "--full", help="Use full file content (no compression)"),
    max_tokens: int = typer.Option(4096, "--max-tokens", "-m"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="No routing logs"),
    system: str = typer.Option("", "--system", "-s", help="Custom system prompt"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream tokens as they arrive"),
):
    """Run a coding task."""
    project = project or _detect_project(repo)
    mode    = "full" if full else "signatures"
    emberforge   = _get_emberforge(project, repo, verbose=not quiet)

    streamed = {"any": False}

    def on_token(tok: str) -> None:
        streamed["any"] = True
        print(tok, end="", flush=True)

    async def _run():
        result = await emberforge.run(
            prompt=prompt,
            use_context=not no_context,
            context_mode=mode,
            max_tokens=max_tokens,
            system=system,
            on_token=on_token if stream else None,
        )
        if streamed["any"] and result.success:
            print()  # close the streamed output
            _print_result(result, show_content=False)
        else:
            _print_result(result)

    asyncio.run(_run())


def _interactive_approver(name: str, desc: str, preview: str) -> bool:
    """CLI approval gate: show what the agent wants to do, ask y/n."""
    console.print(f"\n[bold yellow]⚠ Agent wants to run:[/bold yellow] [bold]{desc}[/bold]")
    if preview:
        console.print(f"[dim]{preview[:2000]}[/dim]")
    return typer.confirm("  Allow?", default=True)


@app.command(name="agent")
def agent(
    prompt: str = typer.Argument(..., help="Task for the agent to complete"),
    project: str = typer.Option("", "--project", "-p"),
    repo: str = typer.Option(".", "--repo", "-r"),
    max_steps: int = typer.Option(25, "--max-steps", help="Agent loop budget"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve edits and shell commands"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
):
    """Agentic run: EMBERFORGE explores the repo, edits files, runs commands in a loop."""
    project = project or _detect_project(repo)
    emberforge   = _get_emberforge(project, repo, verbose=not quiet)

    async def _run():
        result = await emberforge.run_agent(
            prompt=prompt,
            auto_approve=yes,
            approver=None if yes else _interactive_approver,
            max_steps=max_steps,
        )
        console.print()
        if "```" in result.content:
            console.print(Markdown(result.content))
        else:
            console.print(result.content)
        console.print(
            f"\n[dim]⚡ {result.provider} · {result.steps} steps · "
            f"{result.tool_calls_made} tool calls · "
            f"{result.tokens_in}→{result.tokens_out} tokens · {result.latency_ms}ms[/dim]"
        )
        if result.files_changed:
            console.print(f"[dim]📝 Changed: {', '.join(result.files_changed)}[/dim]")
        if not result.success:
            console.print(f"[bold red]⚠ {result.error}[/bold red]")

    asyncio.run(_run())


@app.command(name="chat")
def chat(
    project: str = typer.Option("", "--project", "-p"),
    repo: str = typer.Option(".", "--repo", "-r"),
    max_steps: int = typer.Option(25, "--max-steps"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve edits and shell commands"),
):
    """Interactive agent REPL — conversation persists across turns."""
    project = project or _detect_project(repo)
    emberforge   = _get_emberforge(project, repo, verbose=True)
    agent_instance = emberforge.create_agent(
        auto_approve=yes,
        approver=None if yes else _interactive_approver,
        max_steps=max_steps,
    )

    console.print(BANNER)
    console.print("[dim]Agent REPL — type a task; 'reset' clears history; 'exit' quits.[/dim]\n")

    async def _turn(task: str):
        result = await emberforge.run_agent(prompt=task, agent=agent_instance)
        console.print()
        if "```" in result.content:
            console.print(Markdown(result.content))
        else:
            console.print(result.content)
        console.print(
            f"[dim]⚡ {result.provider} · {result.steps} steps · "
            f"{result.tool_calls_made} tool calls[/dim]"
        )
        if result.files_changed:
            console.print(f"[dim]📝 Changed: {', '.join(result.files_changed)}[/dim]")

    while True:
        try:
            task = console.input("[bold cyan]emberforge>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not task:
            continue
        if task.lower() in ("exit", "quit"):
            break
        if task.lower() == "reset":
            agent_instance.reset()
            console.print("[dim]History cleared.[/dim]")
            continue
        asyncio.run(_turn(task))

    console.print("[dim]bye 👋[/dim]")


@app.command()
def init():
    """Setup EMBERFORGE — create ~/.emberforge/config.yaml interactively."""
    console.print(BANNER)

    emberforge_home  = Path.home() / ".emberforge"
    config_path = emberforge_home / "config.yaml"

    if config_path.exists():
        overwrite = typer.confirm(
            f"Config already exists at {config_path}. Overwrite?", default=False
        )
        if not overwrite:
            console.print("[yellow]Keeping existing config.[/yellow]")
            return

    emberforge_home.mkdir(exist_ok=True)
    (emberforge_home / "skills").mkdir(exist_ok=True)

    console.print("\n[bold]Let's configure your providers.[/bold]")
    console.print("[dim]Press Enter to skip any provider.[/dim]\n")

    keys = {}
    providers_info = [
        ("groq",       "Groq (fast, free)        → console.groq.com/keys"),
        ("gemini",     "Gemini (free tier)        → aistudio.google.com/apikey"),
        ("nvidia_nim", "NVIDIA NIM (free)         → build.nvidia.com/settings/api-keys"),
        ("opencode",   "OpenCode Zen (free)       → opencode.ai/auth"),
        ("mistral",    "Mistral (free experiment) → console.mistral.ai"),
        ("openrouter", "OpenRouter (free models)  → openrouter.ai/keys"),
    ]

    for key_name, label in providers_info:
        val = typer.prompt(f"  {label}", default="", show_default=False)
        if val.strip():
            keys[key_name] = val.strip()

    template_path = Path(__file__).parent.parent / ".emberforge" / "config.yaml"
    if template_path.exists():
        import yaml
        with open(template_path) as f:
            cfg = yaml.safe_load(f)
        for provider_name, api_key in keys.items():
            if provider_name in cfg.get("providers", {}):
                cfg["providers"][provider_name]["api_key"] = api_key
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    else:
        with open(config_path, "w") as f:
            import yaml
            yaml.dump({"providers": {k: {"api_key": v} for k, v in keys.items()}}, f)

    console.print(f"\n[bold green]✅ Config saved to {config_path}[/bold green]")
    console.print("[dim]Run [bold]emberforge status[/bold] to check provider health.[/dim]")


@app.command()
def status():
    """Show provider health + availability."""
    console.print(BANNER)

    async def _check():
        from emberforge.config.settings import load_config
        from emberforge.providers import build_providers

        try:
            config = load_config()
        except FileNotFoundError:
            console.print("[red]No config found. Run [bold]emberforge init[/bold] first.[/red]")
            raise typer.Exit(1)

        providers = build_providers(config)

        if not providers:
            console.print("[yellow]No providers configured. Run [bold]emberforge init[/bold].[/yellow]")
            return

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Provider",  style="bold")
        table.add_column("Tier",      style="yellow")
        table.add_column("Model",     style="dim")
        table.add_column("Status")
        table.add_column("RPM Limit", justify="right", style="dim")

        console.print("\n[dim]Checking provider health...[/dim]")

        checks = await asyncio.gather(
            *[p.health_check() for p in providers.values()],
            return_exceptions=True,
        )

        for (name, provider), healthy in zip(providers.items(), checks):
            if isinstance(healthy, Exception):
                healthy = False
            if provider.health.in_cooldown():
                status_str = (f"[yellow]⏳ Cooldown {provider.health.cooldown_remaining()}s"
                              f" ({provider.health.cooldown_reason[:30]})[/yellow]")
            elif healthy:
                status_str = "[green]✅ Online[/green]"
            else:
                status_str = "[red]❌ Offline[/red]"
            table.add_row(
                name,
                provider.tier,
                provider.primary_model,
                status_str,
                str(provider.rpm_limit),
            )

        console.print(table)

    asyncio.run(_check())


@app.command()
def skills(
    project: str = typer.Option("", "--project", "-p"),
    search:  str = typer.Option("", "--search", "-s", help="Search skills"),
):
    """List all learned skills."""
    from emberforge.config.settings import load_config
    from emberforge.memory import EmberMemory

    try:
        config = load_config()
    except FileNotFoundError:
        console.print("[red]Run emberforge init first.[/red]")
        raise typer.Exit(1)

    memory = EmberMemory(config.memory.path)

    if search:
        results = memory.search_skills(search, limit=10)
        console.print(f"\n[bold]Skills matching '{search}':[/bold]\n")
    else:
        results = memory.list_skills(project or "global", limit=20)
        console.print("\n[bold]All learned skills:[/bold]\n")

    if not results:
        console.print("[dim]No skills yet. Run some tasks and EMBERFORGE will learn.[/dim]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("#",     width=4)
    table.add_column("Title", style="bold cyan")
    table.add_column("Type",  style="yellow")
    table.add_column("Uses",  justify="right")

    for i, skill in enumerate(results, 1):
        table.add_row(
            str(i),
            skill["title"],
            skill.get("task_type", "—"),
            str(skill.get("use_count", 0)),
        )

    console.print(table)


@app.command()
def stats(
    project: str = typer.Option("", "--project", "-p"),
):
    """Show lifetime token stats and savings."""
    from emberforge.config.settings import load_config
    from emberforge.memory import EmberMemory

    try:
        config = load_config()
    except FileNotFoundError:
        console.print("[red]Run emberforge init first.[/red]")
        raise typer.Exit(1)

    memory = EmberMemory(config.memory.path)
    s = memory.total_stats()

    if not s or not s.get("calls"):
        console.print("[dim]No sessions recorded yet.[/dim]")
        return

    console.print("\n[bold cyan]EMBERFORGE Lifetime Stats[/bold cyan]\n")

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Metric", style="dim")
    table.add_column("Value",  style="bold")

    table.add_row("Total Calls",  str(s.get("calls", 0)))
    table.add_row("Tokens In",    f"{s.get('tokens_in', 0):,}")
    table.add_row("Tokens Out",   f"{s.get('tokens_out', 0):,}")
    table.add_row("Tokens Saved", f"[green]{s.get('tokens_saved', 0):,}[/green]")
    table.add_row("Avg Latency",  f"{int(s.get('avg_latency', 0))}ms")
    table.add_row("Est. Cost",    "[green]$0.00[/green]")

    console.print(table)


@app.command()
def learn(
    project: str = typer.Option("", "--project", "-p"),
    repo:    str = typer.Option(".", "--repo", "-r"),
):
    """Force skill generation from recent sessions."""
    project = project or _detect_project(repo)

    async def _learn():
        emberforge = _get_emberforge(project, repo, verbose=True)
        recent = emberforge._memory.recent_sessions(project, limit=10)

        if not recent:
            console.print("[yellow]No sessions to learn from yet.[/yellow]")
            return

        by_type: dict[str, list] = {}
        for s in recent:
            tt = s.get("task_type", "write")
            by_type.setdefault(tt, []).append(s)

        generated = 0
        for task_type, sessions in by_type.items():
            emberforge._skills._tool_call_count = 999
            skill = await emberforge._skills.maybe_generate(
                project=project,
                task_type=task_type,
                sessions=sessions,
                router=emberforge._router,
            )
            if skill:
                console.print(f"[green]✨ Skill: '{skill.title}'[/green]")
                generated += 1

        console.print(f"\n[bold green]{generated} skill(s) generated.[/bold green]")

    asyncio.run(_learn())


@app.command()
def trace(
    raw: bool = typer.Option(False, "--json", help="Print the raw JSON trace"),
):
    """Show exactly what the last agent run did — every tool call, every result."""
    import json as _json
    from emberforge.config import settings as _settings

    path = _settings.EMBERFORGE_HOME / "last_trace.json"
    if not path.exists():
        console.print("[yellow]No trace yet — run [bold]emberforge agent[/bold] first.[/yellow]")
        raise typer.Exit(1)

    data = _json.loads(path.read_text(encoding="utf-8"))
    if raw:
        console.print_json(data=data)
        return

    status = "[green]✅ success[/green]" if data["success"] else f"[red]❌ {data.get('error', '')}[/red]"
    console.print(f"\n[bold]{data['prompt'][:100]}[/bold]")
    console.print(f"{status} · {data['provider']} · {data['steps']} steps · "
                  f"{data['tokens_in']}→{data['tokens_out']} tokens · "
                  f"{data['latency_ms']}ms · {data['ts']}\n")

    t = Table(box=box.SIMPLE, header_style="bold cyan")
    t.add_column("#", width=3)
    t.add_column("Tool")
    t.add_column("Args", max_width=50)
    t.add_column("Result", max_width=60)
    for i, call in enumerate(data.get("tool_calls", []), 1):
        icon = "✓" if call["success"] else "[red]✗[/red]"
        t.add_row(str(i), f"{icon} {call['tool']}",
                  _json.dumps(call["args"])[:50], call["output_preview"][:60])
    console.print(t)
    if data.get("files_changed"):
        console.print(f"[dim]📝 Changed: {', '.join(data['files_changed'])}[/dim]")


@app.command(name="eval")
def eval_cmd(
    task: str = typer.Option("", "--task", "-t", help="Run a single task by name"),
    compare: bool = typer.Option(False, "--compare", help="Run twice: compressed vs full context"),
    max_steps: int = typer.Option(15, "--max-steps"),
    keep: bool = typer.Option(False, "--keep", help="Keep sandbox dirs for inspection"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Task-success evals: the agent must actually finish real tasks in sandbox repos."""
    from emberforge.config.settings import load_config
    from emberforge.providers import build_providers
    from emberforge.evals import EvalRunner, render_markdown
    from emberforge.evals.tasks import TASKS

    config = load_config()
    providers = build_providers(config)
    if not providers:
        console.print("[red]No providers configured — run [bold]emberforge init[/bold] first.[/red]")
        raise typer.Exit(1)

    async def _run():
        scenarios = {"compressed": True, "full-context": False} if compare else {"compressed": True}
        reports = {}
        for label, compress in scenarios.items():
            console.print(f"\n[bold cyan]Scenario: {label}[/bold cyan] "
                          f"({len(TASKS) if not task else 1} task(s))")
            runner = EvalRunner(providers, compress=compress, max_steps=max_steps,
                                verbose=verbose, keep_sandboxes=keep)
            report = await runner.run_all(only=task)
            reports[label] = report

            t = Table(box=box.ROUNDED, header_style="bold cyan")
            for col in ("Task", "Passed", "Steps", "Tokens", "Time", "Provider"):
                t.add_column(col)
            for r in report.results:
                t.add_row(
                    r.task,
                    "[green]✅[/green]" if r.passed else f"[red]❌ {r.error[:30]}[/red]",
                    str(r.steps), f"{r.tokens_in + r.tokens_out:,}",
                    f"{r.seconds}s", r.provider,
                )
            console.print(t)
            console.print(f"[bold]Pass rate: {report.pass_rate:.0%}[/bold] · "
                          f"tokens: {report.total_tokens:,}")

        out = Path("evals_RESULTS.md")
        out.write_text(render_markdown(reports), encoding="utf-8")
        console.print(f"\n[dim]Report written to {out}[/dim]")

    asyncio.run(_run())


@app.command()
def bench():
    """Run the compression benchmark — measured token savings, written to benchmarks/RESULTS.md."""
    try:
        from benchmarks.compression_bench import main as bench_main
    except ImportError:
        console.print("[red]Benchmark module not found — run from the EMBERFORGE repo root.[/red]")
        raise typer.Exit(1)
    bench_main()


@app.command()
def providers():
    """List all configured providers with their tier and models."""
    from emberforge.config.settings import load_config
    from emberforge.providers import build_providers

    try:
        config = load_config()
    except FileNotFoundError:
        console.print("[red]Run emberforge init first.[/red]")
        raise typer.Exit(1)

    all_providers = build_providers(config)
    unconfigured  = [
        name for name, cfg in config.providers.items()
        if not cfg.api_key and name != "ollama"
    ]

    table = Table(box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Provider")
    table.add_column("Tier",           style="yellow")
    table.add_column("Primary Model",  style="dim")
    table.add_column("Fallback Model", style="dim")
    table.add_column("Configured")

    for name, p in all_providers.items():
        table.add_row(
            f"[bold]{name}[/bold]",
            p.tier,
            p.primary_model,
            p.fallback_model,
            "[green]✅[/green]",
        )

    for name in unconfigured:
        cfg = config.providers[name]
        table.add_row(
            f"[dim]{name}[/dim]",
            cfg.tier,
            cfg.models.get("primary", ""),
            cfg.models.get("fallback", ""),
            "[red]❌ No key[/red]",
        )

    console.print(f"\n[bold]{len(all_providers)} configured · {len(unconfigured)} missing keys[/bold]\n")
    console.print(table)
    console.print(f"\n[dim]Edit keys: [bold]~/.emberforge/config.yaml[/bold][/dim]")


if __name__ == "__main__":
    app()