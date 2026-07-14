"""
FORGE CLI — terminal interface, drop-in Claude Code substitute.

Commands:
  forge run "task"      → run a task
  forge init            → setup ~/.forge/config.yaml
  forge status          → show provider health
  forge skills          → list all learned skills
  forge learn           → force skill generation from recent sessions
  forge stats           → session + lifetime token stats
  forge providers       → list all providers + health
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
    name="forge",
    help="FORGE — Free, Open-source, Routing & Generation Engine",
    add_completion=False,
    invoke_without_command=False,
)
console = Console()

BANNER = """[bold cyan]
███████╗ ██████╗ ██████╗  ██████╗ ███████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
█████╗  ██║   ██║██████╔╝██║  ███╗█████╗  
██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  
██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝[/bold cyan]
[dim]Free · Open-source · Routing & Generation Engine[/dim]
[dim]Built by Honey Stark — github.com/Hanishsaini/forge[/dim]
"""


def _get_forge(project: str, repo: str, verbose: bool):
    from forge.core import Forge
    return Forge(project=project, repo_path=repo, verbose=verbose)


def _detect_project(path: str) -> str:
    p = Path(path).resolve()
    for manifest in ("pyproject.toml", "package.json", "Cargo.toml"):
        if (p / manifest).exists():
            return p.name
    return p.name


def _print_result(result, show_stats: bool = True) -> None:
    if not result.success:
        console.print(f"\n[bold red]❌ Error:[/bold red] {result.error}")
        return

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
):
    """Run a coding task."""
    project = project or _detect_project(repo)
    mode    = "full" if full else "signatures"
    forge   = _get_forge(project, repo, verbose=not quiet)

    async def _run():
        result = await forge.run(
            prompt=prompt,
            use_context=not no_context,
            context_mode=mode,
            max_tokens=max_tokens,
            system=system,
        )
        _print_result(result)

    asyncio.run(_run())


@app.command()
def init():
    """Setup FORGE — create ~/.forge/config.yaml interactively."""
    console.print(BANNER)

    forge_home  = Path.home() / ".forge"
    config_path = forge_home / "config.yaml"

    if config_path.exists():
        overwrite = typer.confirm(
            f"Config already exists at {config_path}. Overwrite?", default=False
        )
        if not overwrite:
            console.print("[yellow]Keeping existing config.[/yellow]")
            return

    forge_home.mkdir(exist_ok=True)
    (forge_home / "skills").mkdir(exist_ok=True)

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

    template_path = Path(__file__).parent.parent / ".forge" / "config.yaml"
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
    console.print("[dim]Run [bold]forge status[/bold] to check provider health.[/dim]")


@app.command()
def status():
    """Show provider health + availability."""
    console.print(BANNER)

    async def _check():
        from forge.config.settings import load_config
        from forge.providers import build_providers

        try:
            config = load_config()
        except FileNotFoundError:
            console.print("[red]No config found. Run [bold]forge init[/bold] first.[/red]")
            raise typer.Exit(1)

        providers = build_providers(config)

        if not providers:
            console.print("[yellow]No providers configured. Run [bold]forge init[/bold].[/yellow]")
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
            status_str = "[green]✅ Online[/green]" if healthy else "[red]❌ Offline[/red]"
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
    from forge.config.settings import load_config
    from forge.memory import ForgeMemory

    try:
        config = load_config()
    except FileNotFoundError:
        console.print("[red]Run forge init first.[/red]")
        raise typer.Exit(1)

    memory = ForgeMemory(config.memory.path)

    if search:
        results = memory.search_skills(search, limit=10)
        console.print(f"\n[bold]Skills matching '{search}':[/bold]\n")
    else:
        results = memory.list_skills(project or "global", limit=20)
        console.print("\n[bold]All learned skills:[/bold]\n")

    if not results:
        console.print("[dim]No skills yet. Run some tasks and FORGE will learn.[/dim]")
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
    from forge.config.settings import load_config
    from forge.memory import ForgeMemory

    try:
        config = load_config()
    except FileNotFoundError:
        console.print("[red]Run forge init first.[/red]")
        raise typer.Exit(1)

    memory = ForgeMemory(config.memory.path)
    s = memory.total_stats()

    if not s or not s.get("calls"):
        console.print("[dim]No sessions recorded yet.[/dim]")
        return

    console.print("\n[bold cyan]FORGE Lifetime Stats[/bold cyan]\n")

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
        forge = _get_forge(project, repo, verbose=True)
        recent = forge._memory.recent_sessions(project, limit=10)

        if not recent:
            console.print("[yellow]No sessions to learn from yet.[/yellow]")
            return

        by_type: dict[str, list] = {}
        for s in recent:
            tt = s.get("task_type", "write")
            by_type.setdefault(tt, []).append(s)

        generated = 0
        for task_type, sessions in by_type.items():
            forge._skills._tool_call_count = 999
            skill = await forge._skills.maybe_generate(
                project=project,
                task_type=task_type,
                sessions=sessions,
                router=forge._router,
            )
            if skill:
                console.print(f"[green]✨ Skill: '{skill.title}'[/green]")
                generated += 1

        console.print(f"\n[bold green]{generated} skill(s) generated.[/bold green]")

    asyncio.run(_learn())


@app.command()
def providers():
    """List all configured providers with their tier and models."""
    from forge.config.settings import load_config
    from forge.providers import build_providers

    try:
        config = load_config()
    except FileNotFoundError:
        console.print("[red]Run forge init first.[/red]")
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
    console.print(f"\n[dim]Edit keys: [bold]~/.forge/config.yaml[/bold][/dim]")


if __name__ == "__main__":
    app()