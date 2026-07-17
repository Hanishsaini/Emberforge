"""
EMBERFORGE Core Orchestrator
Wires: Compressor → Context → Router → Memory → Skills
Single entry point for all EMBERFORGE operations.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from emberforge.agent import AgentResult, EmberAgent
from emberforge.compressor import EmberCompressor
from emberforge.config.settings import load_config, ensure_emberforge_home, EmberConfig
from emberforge.context import EmberContext
from emberforge.memory import EmberMemory, SessionRecord
from emberforge.providers import build_providers
from emberforge.providers.base import BaseProvider
from emberforge.router.router import EmberRouter, RouterResult
from emberforge.skills import SkillGenerator
from emberforge.tools import ApprovalCallback, EmberTools, ToolExecutor


@dataclass
class EmberResult:
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


class Ember:
    """
    Main EMBERFORGE class. Instantiate once per session.

    Usage:
        emberforge = Ember(project="my-project")
        result = await emberforge.run("refactor the retrieval pipeline")
        print(result.content)
    """

    def __init__(
        self,
        project:   str          = "default",
        repo_path: str | Path   = ".",
        verbose:   bool         = True,
        config:    EmberConfig | None = None,
    ):
        ensure_emberforge_home()
        self.project   = project
        self.repo_path = Path(repo_path).resolve()
        self.verbose   = verbose

        # Load config
        self._config = config or load_config()

        # Boot all components
        self._compressor = EmberCompressor()
        self._providers  = build_providers(self._config)
        self._router     = EmberRouter(self._providers, verbose=verbose)
        self._context    = EmberContext(
            repo_path=self.repo_path,
            max_tokens=self._config.compressor.max_context_tokens,
            compressor=self._compressor,
        )
        self._memory = EmberMemory(self._config.memory.path)
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
        on_token=None,
    ) -> EmberResult:
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

        # ── Step 1b: Memory recall (Phase 4) ──────────────────────────────────
        brief = self._memory.get_context_brief(self.project)
        if brief:
            context_parts.append(f"<memory>\n{brief}\n</memory>")

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
            on_token=on_token,
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

        return EmberResult(
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

    # ── Agent mode (Phase 1 harness) ──────────────────────────────────────────
    def create_agent(
        self,
        auto_approve: bool = False,
        approver:     ApprovalCallback | None = None,
        max_steps:    int  = 25,
        max_tokens:   int  = 4096,
    ) -> EmberAgent:
        """
        Build a persistent agent (keeps conversation across .run() calls — used
        by the chat REPL). For one-shot tasks use run_agent().
        """
        tools    = EmberTools(self.repo_path, compressor=self._compressor)
        executor = ToolExecutor(tools, auto_approve=auto_approve, approver=approver)
        return EmberAgent(
            providers=self._providers,
            executor=executor,
            compressor=self._compressor,
            max_steps=max_steps,
            max_tokens=max_tokens,
            verbose=self.verbose,
        )

    async def run_agent(
        self,
        prompt:       str,
        auto_approve: bool = False,
        approver:     ApprovalCallback | None = None,
        max_steps:    int  = 25,
        max_tokens:   int  = 4096,
        agent:        EmberAgent | None = None,
    ) -> AgentResult:
        """
        Agentic run: the model explores the repo, edits files, and runs commands
        in a loop until the task is done. Skills are injected as context; the
        session and per-tool-call counts feed the memory/skill engine.
        """
        agent = agent or self.create_agent(
            auto_approve=auto_approve, approver=approver,
            max_steps=max_steps, max_tokens=max_tokens,
        )

        # Inject learned skills + project memory + failure warnings (Phase 4)
        context_parts: list[str] = []

        relevant_skills = self._skills.find_relevant_skills(prompt)
        if relevant_skills:
            context_parts.append("\n\n".join(
                f"## Relevant Skill: {s['title']}\n{s['content'][:500]}"
                for s in relevant_skills
            ))

        brief = self._memory.get_context_brief(self.project)
        if brief:
            context_parts.append(f"<memory>\n{brief}\n</memory>")

        past_failures = self._memory.similar_failures(prompt, self.project)
        if past_failures:
            warnings = "\n".join(
                f"- \"{' '.join(f['prompt'].split())[:80]}\" failed with: {f['error'][:120]}"
                for f in past_failures
            )
            context_parts.append(
                "<past-failures>\nSimilar tasks failed before — avoid repeating "
                f"these dead ends:\n{warnings}\n</past-failures>"
            )

        result = await agent.run(prompt, context="\n\n".join(context_parts))

        # Feed the skill engine with REAL tool-call counts (not run counts)
        for _ in range(max(result.tool_calls_made, 1)):
            self._skills.record_tool_call()

        # Persist session
        if result.success:
            session_id = self._memory.save_session(SessionRecord(
                project=self.project,
                task_type="agent",
                prompt=prompt,
                response=result.content,
                provider=result.provider,
                model=result.model,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                tokens_saved=0,
                latency_ms=result.latency_ms,
                success=True,
            ))
            self._session_ids.append(session_id)
            # Changed code = a decision worth remembering next session
            if result.files_changed:
                self._memory.log_decision(
                    self.project,
                    f"agent: {' '.join(prompt.split())[:100]} → changed "
                    + ", ".join(result.files_changed[:5]),
                )
        else:
            self._memory.log_failure(
                project=self.project,
                prompt=prompt,
                provider=result.provider or "agent",
                error=result.error,
            )
        return result

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
