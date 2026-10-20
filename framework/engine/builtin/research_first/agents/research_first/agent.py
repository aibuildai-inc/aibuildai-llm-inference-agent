"""The four Research First roles: Planner, Committee, Worker, and Reviewer.

The Planner and the Worker each write and compile one whole document of their own — the proposal and the report — with the compiler their ``required_pdf`` declaration puts on their PATH, and the shared document gate asks only that the promised PDF is there. A scored project's attempt runs the same shared attempt gate every scored attempt runs. What is left, below, is the one check only this family can make: the Committee's one-decision-per-published-plan rule, read against the Board itself. The Reviewer's execution-versus-claim agreement needs no durable work at all and lives in its Output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

from pydantic import ConfigDict

from engine.capability import ExecutionCapability
from engine.durable_execution import ActionInvocation, action
from engine.execution_output import VerifierOutput
from engine.failure import Failure
from engine.work_unit.agent.base import Agent
from engine.work_unit.agent.policy import (
    ALL_BUILTIN_TOOLS,
    JUDGE_TOOLS_WITH_SCRATCH,
    RolePolicy,
    SystemDir,
)
from engine.work_unit.program.attempt_verifier import (
    AttemptVerifierInput,
    AttemptVerifierProgram,
)
from engine.work_unit.program.base import Program
from engine.builtin.research_first.agents.research_first.io import (
    ResearchFirstCommitteeInput,
    ResearchFirstCommitteeOutput,
    ResearchFirstPlannerInput,
    ResearchFirstProposalOutput,
    ResearchFirstProposalRecord,
    ResearchFirstReviewerInput,
    ResearchFirstReviewerOutput,
    ResearchFirstWorkerInput,
    ResearchFirstWorkerOutput,
)
from aibuildai_exec.files import atomic_write_text

# Where each author keeps its own document, relative to its artifacts. The
# Board publishes these directories; nothing rewrites what is inside them.
PROPOSAL_DIR = "proposal"
REPORT_DIR = "report"


# ── Planner ──────────────────────────────────────────────────────────────────

RESEARCH_FIRST_PLANNER_POLICY = RolePolicy(
    tools=(
        *JUDGE_TOOLS_WITH_SCRATCH,
        "BashOutput",
        "KillBash",
        "StructuredOutput",
    ),
    system_read=(SystemDir.CONDA_PACKAGES,),
)


class ResearchFirstPlannerAgent(
    Agent[ResearchFirstPlannerInput, ResearchFirstProposalOutput]
):
    """Write one research proposal as ``proposal/proposal.pdf`` and its facts."""

    name: ClassVar[str] = "planner"
    prompt_template = "agent/research_first_planner.j2"
    policy = RESEARCH_FIRST_PLANNER_POLICY
    required_pdf: ClassVar[str] = f"{PROPOSAL_DIR}/proposal.pdf"

    async def verifier_invocations(
        self, candidate: ResearchFirstProposalOutput
    ) -> tuple[ActionInvocation[Any, VerifierOutput], ...]:
        """The proposal document this role promised, and nothing else."""
        del candidate
        return self.document_verifier()

    def _finalize_output(
        self,
        output: ResearchFirstProposalOutput | Failure,
    ) -> ResearchFirstProposalOutput | Failure:
        """Record the machine manifest beside the accepted proposal.

        The proposal itself is already written and compiled by its author and
        is not touched here. The manifest is the Board's own index of the
        facts the Committee and the Search read; it is not another copy of the
        document."""
        if output.failed is True:
            return output
        manifest = ResearchFirstProposalRecord(
            plan_uid=self.uid,
            title=output.title,
            abstract=output.abstract,
            metadata=output.metadata,
            expected_result=output.expected_result,
            requested_gpus=output.requested_gpus,
            scored=output.scored,
        )
        atomic_write_text(
            Path(self.artifacts_dir) / PROPOSAL_DIR / "manifest.json",
            manifest.model_dump_json(indent=2),
        )
        return output


# ── Committee ────────────────────────────────────────────────────────────────

RESEARCH_FIRST_COMMITTEE_POLICY = RolePolicy(
    tools=(
        *JUDGE_TOOLS_WITH_SCRATCH,
        "StructuredOutput",
    ),
)


class ResearchFirstCommitteeAgent(
    Agent[ResearchFirstCommitteeInput, ResearchFirstCommitteeOutput]
):
    """Judge every waitlist proposal in one round and assign the free slots."""

    name: ClassVar[str] = "committee"
    prompt_template = "agent/research_first_committee.j2"
    policy = RESEARCH_FIRST_COMMITTEE_POLICY

    async def verifier_invocations(
        self, candidate: ResearchFirstCommitteeOutput
    ) -> tuple[ActionInvocation[Any, VerifierOutput], ...]:
        from engine.builtin.research_first.search import ResearchFirstSearch

        parent = self.parent
        if not isinstance(parent, ResearchFirstSearch):
            raise AssertionError(
                "the Committee must be spawned directly by ResearchFirstSearch"
            )
        committee_input = cast(ResearchFirstCommitteeInput, self.input)
        return (
            (
                CommitteeVerifierProgram(
                    input=CommitteeVerifierInput(
                        board_dir=committee_input.board_dir,
                        free_slots=await parent.committee_free_slots(on=self.ctx),
                    ),
                ).run,
                ExecutionCapability(
                    wall_clock_seconds=self._local_wall_clock,
                    gpus=0
                ),
                candidate,
            ),
        )


# ── Worker ───────────────────────────────────────────────────────────────────

# The Research First Worker owns no design: it codes in its own scratch and
# writes the scored attempt under its own artifacts, so its policy binds no
# design directory.
RESEARCH_FIRST_WORKER_POLICY = RolePolicy(
    tools=tuple(ALL_BUILTIN_TOOLS),
    system_read=(SystemDir.CONDA_PACKAGES,),
    blocks_long_sleep=True,
    task_environment=True,
)


class ResearchFirstWorkerAgent(
    Agent[ResearchFirstWorkerInput, ResearchFirstWorkerOutput]
):
    """Conduct one whole research project and close it with ``report.pdf``.

    One LLM execution per accepted proposal: it reads the proposal, writes
    and runs the code, inspects the artifacts, analyzes what happened, and
    writes and compiles the final report — all in the same session."""

    name: ClassVar[str] = "worker"
    max_turns = 150
    prompt_template = "agent/research_first_worker.j2"
    policy = RESEARCH_FIRST_WORKER_POLICY
    required_pdf: ClassVar[str] = f"{REPORT_DIR}/report.pdf"

    async def verifier_invocations(
        self, candidate: ResearchFirstWorkerOutput
    ) -> tuple[ActionInvocation[Any, VerifierOutput], ...]:
        # The report the role promised first, then the attempt: a project that
        # never compiled its report has nothing to score either way. A proposal
        # the Board did not mark scored produces no attempt at all, so there is
        # nothing for the attempt gate to look at.
        del candidate
        chain: list[ActionInvocation[Any, VerifierOutput]] = [
            *self.document_verifier()
        ]
        if not cast(ResearchFirstWorkerInput, self.input).scored:
            return tuple(chain)
        chain.append(
            (
                AttemptVerifierProgram(
                    input=AttemptVerifierInput(
                        attempt_dir=f"{self.artifacts_dir}/full",
                    ),
                ).run,
                ExecutionCapability(
                    wall_clock_seconds=self._local_wall_clock,
                    gpus=0
                ),
            )
        )
        return tuple(chain)

    async def _prepare_conversation(self) -> None:
        """Create the attempt output directory before the session."""
        from infra.fs.artifact_dir import prepare_dir

        await self.ctx.step(prepare_dir, f"{self.artifacts_dir}/full")


# ── Reviewer ─────────────────────────────────────────────────────────────────

RESEARCH_FIRST_REVIEWER_POLICY = RolePolicy(
    tools=(
        *JUDGE_TOOLS_WITH_SCRATCH,
        "BashOutput",
        "KillBash",
        "StructuredOutput",
    ),
)


class ResearchFirstReviewerAgent(
    Agent[ResearchFirstReviewerInput, ResearchFirstReviewerOutput]
):
    """Judge one finished project against its proposal, artifacts, and score."""

    name: ClassVar[str] = "reviewer"
    prompt_template = "agent/research_first_reviewer.j2"
    policy = RESEARCH_FIRST_REVIEWER_POLICY


@dataclass(frozen=True)
class CommitteeVerifierInput:
    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    # The board the round's waitlist is read from, and the slots this round
    # may fill. Both are the run's state at this moment, not the Committee's
    # answer, so they identify the verifier rather than travel in its request.
    board_dir: str
    free_slots: int


class CommitteeVerifierProgram(Program[CommitteeVerifierInput]):
    """Require one legal decision per waitlisted plan, within the free slots.

    The waitlist is read from the board here, in the verifier, so the round is
    judged against the plans that are really published and undecided rather
    than against a copy the Committee was handed earlier."""

    name: ClassVar[str] = "committee_verifier"

    @action
    def run(self, candidate: ResearchFirstCommitteeOutput) -> VerifierOutput | Failure:
        from engine.builtin.research_first.search import _read_waitlist

        expected = tuple(_read_waitlist(self.input.board_dir))
        decided = tuple(decision.plan_uid for decision in candidate.decisions)
        problems: list[str] = []
        for uid in expected:
            count = decided.count(uid)
            if count != 1:
                problems.append(f"plan {uid} has {count} decisions, needs 1")
        for uid in decided:
            if uid not in expected:
                problems.append(f"decision names unknown plan {uid}")
        for decision in candidate.decisions:
            if not decision.reason.strip():
                problems.append(f"decision on {decision.plan_uid} has no reason")
        announcement = candidate.announcement
        if announcement is not None and not announcement.strip():
            problems.append("announcement is blank; use null or Markdown text")
        accepted = sum(
            decision.verdict == "accept" for decision in candidate.decisions
        )
        if accepted > self.input.free_slots:
            problems.append(
                f"{accepted} proposals accepted but only "
                f"{self.input.free_slots} slot(s) are free"
            )
        if problems:
            return VerifierOutput(
                passed=False,
                reason=(
                    "; ".join(problems)
                    + ". Give exactly one decision per waitlisted plan, each with "
                    "a non-empty reason, accepting at most the free slot count, "
                    "with announcement either null or non-blank Markdown, then "
                    "call StructuredOutput again."
                ),
            )
        return VerifierOutput(passed=True, reason="")
