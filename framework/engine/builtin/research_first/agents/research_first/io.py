"""Input and Output records for the Research First Agent family.

Research First models research as the primary activity: a Planner writes a
proposal, a Committee assigns execution slots, one Worker session conducts the
whole project and writes its report, and a Reviewer judges the finished
project. Research history lives in the Research First Board — a real
filesystem directory with one folder per plan — so every Input here is
path-based: it names where to read, never a copied snapshot of the content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, model_validator

from engine.base import WorkflowBaseModel
from engine.builtin.aibuildai.programs.score import ScoreOutput
from engine.execution_output import SuccessfulOutput
from engine.failure import Failure
from engine.metric_contract import MetricContractInput
from engine.work_unit.agent.base import AgentInput


class ResearchFirstProposalRecord(WorkflowBaseModel):
    """One plan's ``proposal/manifest.json`` on the Research First Board.

    The system writes every field here from the Planner execution that
    produced the proposal; the model invents none of them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_uid: str
    title: str
    abstract: str
    metadata: dict[str, JsonValue]
    expected_result: str | None
    requested_gpus: int
    scored: bool


class ResearchFirstDecisionRecord(WorkflowBaseModel):
    """One Committee verdict, appended as one immutable Board decision file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_uid: str
    committee_round: int
    verdict: Literal["accept", "waitlist", "reject"]
    reason: str


class ResearchFirstScoreRecord(WorkflowBaseModel):
    """One plan's ``score/result.json`` on the Research First Board.

    ``scoring`` is the framework score program's own Output. A score program
    failure stays as its own structured fact. The Search that scored this
    plan keeps the live Handle that produced it in its own local bookkeeping;
    this record is the Board's durable copy of the score facts, not a path
    from which that Handle could be rebuilt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_uid: str
    scoring: ScoreOutput | None
    score_failure: Failure | None
    output_dir: str


@dataclass(frozen=True)
class ResearchFirstPlannerInput(AgentInput):
    """Input for one Research First proposal."""

    board_dir: str
    metric_contract: MetricContractInput | None
    remaining_minutes: float
    gpu_count: int


class ResearchFirstProposalOutput(SuccessfulOutput):
    """The machine-readable facts of one proposal the Planner wrote.

    The proposal document itself is not here: the Planner writes and compiles
    ``proposal/proposal.pdf`` in its own artifacts. These are the facts the
    Committee, the Search, and the Board index actually read."""

    title: str = Field(description="proposal title as PLAIN text, no LaTeX markup")
    abstract: str = Field(
        description="plain-text abstract of the proposal for the Board index"
    )
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="small machine-readable facts about the proposal, e.g. "
        '{"type": "exploration", "direction": "learning-rate"}',
    )
    expected_result: str | None = Field(
        default=None,
        description="what the Planner predicts will happen, or null when the "
        "proposal is exploratory and makes no prediction",
    )
    requested_gpus: int = Field(
        ge=0,
        description="cards the project holds for its whole life, Worker and "
        "score program together; 0 when it needs no card",
    )
    scored: bool = Field(
        description="true when the Worker writes a checkpoint to attempt_dir "
        "for the framework score program; false for a study that trains "
        "nothing, submits nothing, and is never scored. Independent of "
        "requested_gpus",
    )


@dataclass(frozen=True)
class ResearchFirstCommitteeInput(AgentInput):
    """Stable identity facts for the Research First Committee."""

    board_dir: str
    metric_contract: MetricContractInput | None


class ResearchFirstCommitteeDecision(WorkflowBaseModel):
    """One Committee verdict on one waitlisted plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_uid: str
    verdict: Literal["accept", "waitlist", "reject"]
    reason: str


class ResearchFirstCommitteeOutput(SuccessfulOutput):
    """The Committee's verdict on every waitlisted plan in one round."""

    decisions: tuple[ResearchFirstCommitteeDecision, ...] = Field(
        description="exactly one decision per waitlisted plan: accept starts "
        "the project now, waitlist keeps it for a later round, reject removes "
        "it with the recorded reason"
    )
    announcement: str | None = Field(
        default=None,
        min_length=1,
        description="optional Markdown notice to every future Planner, "
        "published on the Board as committee/announcement_NNNN.md: the state "
        "of the research so far, what past plans showed works or does not, "
        "and what to propose next and why; null when the last announcement "
        "still holds",
    )


@dataclass(frozen=True)
class ResearchFirstWorkerInput(AgentInput):
    """One accepted plan's Board folder for the Research First Worker."""

    plan_dir: str
    task_name: str
    gpu_count: int
    scored: bool
    progress_enabled: bool


class ResearchFirstWorkerOutput(SuccessfulOutput):
    """The Worker's submission carries no fields of its own.

    Everything a reader wants is in the report the Worker wrote and compiled
    itself, and everything the product consumes is on disk: the scored attempt,
    the framework score, and the Reviewer's verdict. Calling StructuredOutput
    is the Worker saying its project is closed."""


@dataclass(frozen=True)
class ResearchFirstReviewerInput(AgentInput):
    """One finished plan's Board folder for the Research First Reviewer."""

    plan_dir: str
    scored: bool
    metric_contract: MetricContractInput | None


class ResearchFirstReviewerOutput(SuccessfulOutput):
    """One peer review of a finished Research First project."""

    completion_status: Literal["complete", "partial", "failed", "invalid"] = Field(
        description="how much of the proposed project actually ran: complete "
        "= the proposed work ran and produced interpretable evidence; partial "
        "= some of it ran; failed = the execution broke before producing "
        "evidence; invalid = what ran does not answer the proposal's question"
    )
    completion_assessment: str = Field(
        description="what was actually completed, judged from artifacts, "
        "never from the report's polish"
    )
    hypothesis_outcome: Literal["supported", "refuted", "inconclusive"] = Field(
        description="what the evidence says about the proposal's hypothesis; "
        "an invalid or failed execution is inconclusive, not refuted"
    )
    scientific_assessment: str = Field(
        description="whether the report's claims are supported by the "
        "artifacts and, for a scored project, the framework score, and where "
        "they overreach"
    )
    supported_findings: tuple[str, ...] = Field(
        description="claims the artifacts genuinely support, phrased so a "
        "future Planner can build on them"
    )
    unsupported_or_overbroad_claims: tuple[str, ...] = Field(
        description="claims the report makes that the artifacts do not "
        "support at the stated scope"
    )
    limitations: tuple[str, ...] = Field(
        description="boundaries of what this project can conclude"
    )
    recommended_next_work: tuple[str, ...] = Field(
        description="follow-up directions this project's evidence motivates"
    )

    @model_validator(mode="after")
    def _execution_and_science_agree(self) -> "ResearchFirstReviewerOutput":
        """Refuse a review that contradicts itself.

        A failed or invalid execution produced no evidence, so it can neither
        support nor refute the hypothesis; and an assessment field that is
        blank has not been written. Both are properties of this answer alone,
        so nothing outside it decides them."""
        if self.completion_status in {"failed", "invalid"} and (
            self.hypothesis_outcome != "inconclusive"
        ):
            raise ValueError(
                "a failed or invalid execution cannot support or refute the "
                "hypothesis. Separate execution validity from the scientific "
                "result: set hypothesis_outcome to inconclusive when "
                "completion_status is failed or invalid."
            )
        blank = [
            field
            for field, value in (
                ("completion_assessment", self.completion_assessment),
                ("scientific_assessment", self.scientific_assessment),
            )
            if not value.strip()
        ]
        if blank:
            raise ValueError(
                f"review field(s) {', '.join(blank)} must not be blank. Fill "
                "every assessment field."
            )
        return self
