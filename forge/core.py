"""
FORGE Core Orchestrator
Wires: Compressor → Context → Router → Memory → Skills
Single entry point for all FORGE operations.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from forge.compressor import ForgeCompressor
from forge.config.settings import load_config, ensure_forge_home, ForgeConfig
from forge.context import ForgeContext
from forge.memory import ForgeMemory, SessionRecord
from forge.providers import build_providers
from forge.providers.base import BaseProvider
from forge.router.router import ForgeRouter, RouterResult
from forge.skills import SkillGenerator


@dataclass
class ForgeResult:
    content:       str
    provider:      str
    model:         str
    task_type:     str
    tokens_in:     int
    tokens_out:    int
    tokens_saved:  int
    latency_ms:    int
    attempts:      int
    success:       bool
    error:         str = ""
    skill_generated: str | None = None


class Forge:
    """
    Main FORGE class. Instantiate once per session.

    Usage:
        forge = Forge(project="my-project")
        result = await forge.run("refactor the retrieval pipeline")
        print(result.content)
    """

    def __init__(
        self,
        project:   str          = "default",
        repo_path: str | Path   = ".",
        verbose:   bool         = True,
        config:    ForgeConfig | None = None,
    ):
        ensure_forge_home()
        self.project   = project
        self.repo_path = Path(repo_path).resolve()
        self.verbose   = verbose

        # Load config
        self._config = config or load_config()

        # Boot all components
        self._compressor = ForgeCompressor()
        self._providers  = build_providers(self._config)
        self._router     = ForgeRouter(self._providers, verbose=verbose)
        self._context    = ForgeContext(
            repo_path=self.repo_path,
            max_tokens=self._config.compressor.max_context_tokens,
            compressor=self._compressor,
        )
        self._memory = ForgeMemory(self._config.memory.path)
        self._skills = SkillGenerator(
            self._memory,
            threshold=self._config.memory.skill_threshold,
        )

        # Ensure project exists in memory
        self._memory.upsert_project(self.project)

        # Session tracking
        self._session_ids:   list[int]  = []
        self._session_start: float      = time.time()

    async def run(
        self,
        prompt:        str,
        use_context:   bool = True,
        context_mode:  str  = "signatures",
        max_tokens:    int  = 4096,
        system:        str  = "",
    ) -> ForgeResult:
        """
        Full pipeline:
        1. Search relevant skills → prepend to context
        2. Build codebase context (CodeLore / fallback)
        3. Route to best available provider
        4. Store session in memory
        5. Maybe generate skill (Hermes-style)
        """

        t_start = time.time()
        context_parts: list[str] = []

        # ── Step 1: Skill lookup ───────────────────────────────────────────────
        relevant_skills = self._skills.find_relevant_skills(prompt)
        if relevant_skills:
            skill_ctx = "\n\n".join(
                f"## Relevant Skill: {s['title']}\n{s['content'][:500]}"
                for s in relevant_skills
            )
            context_parts.append(f"<skills>\n{skill_ctx}\n</skills>")
            if self.verbose:
                from rich.console import Console
                Console().print(f"[dim cyan]  ↑ Loaded {len(relevant_skills)} relevant skill(s)[/dim cyan]")

        # ── Step 2: Codebase context ───────────────────────────────────────────
        if use_context and self._providers:
            ctx_result = self._context.build_context(prompt, mode=context_mode)
            if ctx_result.context:
                context_parts.append(
                    f"<codebase context='{self.project}' "
                    f"files='{len(ctx_result.files_included)}' "
                    f"tokens='{ctx_result.total_tokens}'>\n"
                    f"{ctx_result.context}\n</codebase>"
                )

        # ── Step 3: Compress final context ────────────────────────────────────
        full_context = "\n\n".join(context_parts)
        if full_context:
            compressed = self._compressor.compress(full_context, content_type="text")
            final_context = compressed.final_text
            tokens_saved_ctx = compressed.tokens_saved
        else:
            final_context    = ""
            tokens_saved_ctx = 0

        # ── Step 4: Route ──────────────────────────────────────────────────────
        router_result: RouterResult = await self._router.route(
            prompt=prompt,
            context=final_context,
            system=system,
            max_tokens=max_tokens,
        )
        self._skills.record_tool_call()

        response = router_result.response
        total_ms = int((time.time() - t_start) * 1000)

        # ── Step 5: Store session ─────────────────────────────────────────────
        if response.success:
            session_id = self._memory.save_session(SessionRecord(
                project=self.project,
                task_type=router_result.classification.task_type,
                prompt=prompt,
                response=response.content,
                provider=response.provider,
                model=response.model,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                tokens_saved=tokens_saved_ctx,
                latency_ms=total_ms,
                success=True,
            ))
            self._session_ids.append(session_id)
        else:
            self._memory.log_failure(
                project=self.project,
                prompt=prompt,
                provider=response.provider,
                error=response.error,
            )

        # ── Step 6: Maybe generate skill (Hermes-style) ───────────────────────
        skill_generated = None
        if (
            self._config.memory.skill_auto_generate
            and self._skills.should_generate()
        ):
            recent = self._memory.recent_sessions(self.project, limit=10)
            skill = await self._skills.maybe_generate(
                project=self.project,
                task_type=router_result.classification.task_type,
                sessions=recent,
                router=self._router,
            )
            if skill:
                skill_generated = skill.title
                if self.verbose:
                    from rich.console import Console
                    Console().print(f"[dim green]  ✨ Skill generated: '{skill.title}'[/dim green]")

        return ForgeResult(
            content=response.content,
            provider=response.provider,
            model=response.model,
            task_type=router_result.classification.task_type,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            tokens_saved=tokens_saved_ctx,
            latency_ms=total_ms,
            attempts=router_result.attempts,
            success=response.success,
            error=response.error,
            skill_generated=skill_generated,
        )

    def session_stats(self) -> dict:
        """Stats for this session."""
        router_stats = self._router.stats()
        mem_stats    = self._memory.total_stats()
        return {
            "session_calls":   len(self._session_ids),
            "session_duration": int(time.time() - self._session_start),
            **router_stats,
            "lifetime_tokens": mem_stats.get("tokens_in", 0) + mem_stats.get("tokens_out", 0),
            "lifetime_saved":  mem_stats.get("tokens_saved", 0),
        }

    def available_providers(self) -> list[str]:
        return [
            str(p)
            for p in self._providers.values()
        ]
