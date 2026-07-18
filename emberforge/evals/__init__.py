"""
EmberForge Eval Suite — does the agent actually finish the job?

The token benchmark (emberforge bench) proves compression saves tokens.
This suite proves the thing the whole field skips measuring: whether the
agent still SUCCEEDS at real tasks — per provider, with and without
compression. It is also the fitness function for the AHE evolution loop.

Run: emberforge eval [--task NAME] [--compare] [--max-steps N]
"""
from __future__ import annotations

import time
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from emberforge.agent import EmberAgent
from emberforge.evals.tasks import TASKS, EvalTask, get_task
from emberforge.providers.base import BaseProvider
from emberforge.tools import EmberTools, ToolExecutor


@dataclass
class EvalResult:
    task:        str
    task_type:   str
    passed:      bool
    steps:       int
    tool_calls:  int
    tokens_in:   int
    tokens_out:  int
    seconds:     float
    provider:    str
    compress:    bool
    error:       str = ""
    sandbox:     str = ""


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed for r in self.results) / len(self.results)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_in + r.tokens_out for r in self.results)


class EvalRunner:
    """
    Runs eval tasks through the REAL agent loop in disposable sandbox repos.
    Sandboxes are auto-approved (they're throwaway dirs, not your code) but
    still protected by the shell blocklist and path confinement.
    """

    def __init__(
        self,
        providers: dict[str, BaseProvider],
        compress:  bool = True,
        max_steps: int  = 15,
        verbose:   bool = False,
        keep_sandboxes: bool = False,
    ):
        self._providers = providers
        self.compress   = compress
        self.max_steps  = max_steps
        self.verbose    = verbose
        self.keep_sandboxes = keep_sandboxes

    async def run_task(self, task: EvalTask) -> EvalResult:
        sandbox = Path(tempfile.mkdtemp(prefix=f"emberforge-eval-{task.name}-"))
        task.setup(sandbox)

        tools    = EmberTools(sandbox, compress=self.compress)
        executor = ToolExecutor(tools, auto_approve=True)
        agent    = EmberAgent(
            providers=self._providers, executor=executor,
            max_steps=self.max_steps, verbose=self.verbose,
        )

        t0 = time.time()
        agent_result = await agent.run(task.prompt)
        seconds = round(time.time() - t0, 1)

        passed = bool(task.verify(sandbox))

        result = EvalResult(
            task=task.name,
            task_type=task.task_type,
            passed=passed,
            steps=agent_result.steps,
            tool_calls=agent_result.tool_calls_made,
            tokens_in=agent_result.tokens_in,
            tokens_out=agent_result.tokens_out,
            seconds=seconds,
            provider=agent_result.provider,
            compress=self.compress,
            error=agent_result.error,
            sandbox=str(sandbox),
        )

        if passed and not self.keep_sandboxes:
            shutil.rmtree(sandbox, ignore_errors=True)
        return result

    async def run_all(self, only: str = "") -> EvalReport:
        tasks = [get_task(only)] if only else TASKS
        report = EvalReport()
        for task in tasks:
            if task is None:
                continue
            report.results.append(await self.run_task(task))
        return report


def render_markdown(reports: dict[str, EvalReport]) -> str:
    """reports: label -> report (e.g. {'compressed': ..., 'full-context': ...})"""
    lines = [
        "# EmberForge Eval Results — Task Success, Measured",
        "",
        f"- Date: {time.strftime('%Y-%m-%d')}",
        "- Scoring: agent runs the real loop in a sandbox repo; pass = the",
        "  task's verification (usually pytest) succeeds afterward.",
        "- Reproduce: `emberforge eval` (or `--compare` for both scenarios)",
        "",
    ]
    for label, report in reports.items():
        lines += [
            f"## Scenario: {label}",
            "",
            f"**Pass rate: {report.pass_rate:.0%}** · "
            f"total tokens: {report.total_tokens:,}",
            "",
            "| Task | Type | Passed | Steps | Tools | Tokens | Time | Provider |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
        for r in report.results:
            lines.append(
                f"| {r.task} | {r.task_type} | {'✅' if r.passed else '❌'} | "
                f"{r.steps} | {r.tool_calls} | {r.tokens_in + r.tokens_out:,} | "
                f"{r.seconds}s | {r.provider} |"
            )
        lines.append("")
    return "\n".join(lines)
