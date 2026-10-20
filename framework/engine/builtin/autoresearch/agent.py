"""AutoResearch's one exploration role, its Input, and its gate. One role, so
one module: the siblings nest ``agents/<role>/io.py`` for several of each."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import ConfigDict

from engine.capability import ExecutionCapability
from engine.durable_execution import ActionInvocation, action
from engine.execution_output import SuccessfulOutput, VerifierOutput
from engine.failure import Failure
from engine.work_unit.agent.base import Agent, AgentInput
from engine.work_unit.agent.policy import ALL_BUILTIN_TOOLS, RolePolicy, SystemDir
from engine.work_unit.program import Program
from engine.work_unit.program.attempt_verifier import (AttemptVerifierInput,
                                                       AttemptVerifierProgram)
from infra.fs.artifact_dir import prepare_dir


@dataclass(frozen=True)
class AutoResearchWorkerInput(AgentInput):
    """The few facts one step cannot read off the shared files themselves.

    ``previous_record`` is None for step 1 and for a predecessor that produced
    no artifacts; ``previous_failure`` is None unless it failed, whose Action a
    fresh Worker cannot see."""

    step: int
    max_steps: int | None
    source_dir: str
    records_dir: str
    progress_enabled: bool
    previous_record: str | None
    previous_failure: str | None


@dataclass(frozen=True)
class HandoffNoteInput:
    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    readme_path: str


class HandoffNoteProgram(Program[HandoffNoteInput]):
    """Require the note this step promised the step after it. What the note
    SAYS is a semantic review's question, and this package composes none."""

    name: ClassVar[str] = "autoresearch_handoff_note"

    @action
    def run(self) -> VerifierOutput | Failure:
        readme = Path(self.input.readme_path)
        if readme.is_file() and readme.read_text(encoding="utf-8").strip():
            return VerifierOutput(passed=True, reason="")
        return VerifierOutput(
            passed=False,
            reason=f"{readme} is missing or empty. It is the only prose the "
            "next step inherits: write your starting point, the change you "
            "made, the evidence behind it, the result, and what to try next, "
            "then call StructuredOutput again.",
        )


AUTORESEARCH_WORKER_POLICY = RolePolicy(
    tools=tuple(ALL_BUILTIN_TOOLS),
    system_read=(SystemDir.CONDA_PACKAGES,),
    blocks_long_sleep=True,
    task_environment=True,
)


class AutoResearchWorkerAgent(Agent[AutoResearchWorkerInput, SuccessfulOutput]):
    """Carry the shared post-training codebase forward by one research step.

    Its submission carries no fields: the record is the README it wrote and the
    Git history it left, and the product consumes its attempt."""

    name: ClassVar[str] = "worker"
    max_turns = 150
    prompt_template = "agent/worker.j2"
    policy = AUTORESEARCH_WORKER_POLICY

    async def _prepare_conversation(self) -> None:
        """Create this step's attempt directory before the session opens."""
        await self.ctx.step(prepare_dir, f"{self.artifacts_dir}/full")

    # Any: the base types the Input half of an invocation as Any because each
    # verifier in a chain has its own Input; only the Output half is common.
    async def verifier_invocations(
        self, candidate: SuccessfulOutput
    ) -> tuple[ActionInvocation[Any, VerifierOutput], ...]:
        """The two promises a normal step makes: the note first, then the attempt."""
        del candidate
        grant = ExecutionCapability(wall_clock_seconds=self._local_wall_clock)
        note = HandoffNoteProgram(
            input=HandoffNoteInput(f"{self.artifacts_dir}/README.md")
        )
        attempt = AttemptVerifierProgram(
            input=AttemptVerifierInput(f"{self.artifacts_dir}/full")
        )
        return ((note.run, grant), (attempt.run, grant))
