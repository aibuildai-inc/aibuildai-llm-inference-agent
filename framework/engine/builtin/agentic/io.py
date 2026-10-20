"""The agentic package's own business records.

Every record here is local to this package. Another package may spell a field
the same way with its own meaning, or declare none of them."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from engine.base import WorkflowBaseModel
from engine.builtin.aibuildai.agents.submitter.io import ExternalScore
from engine.builtin.aibuildai.programs.score import ScoreLaunch, ScoreOutput
from engine.builtin.agentic.agents.aggregator.io import EnsembleProposal
from engine.builtin.agentic.agents.designer.io import WorkerDesignPlan
from engine.capability import ExecutionCapability
from engine.durable_execution import FileRef
from engine.metric_contract import MetricContractInput


class CandidateGrant(WorkflowBaseModel):
    """What one agentic Worker candidate or Ensemble needs to run.

    Frozen at the spawn site out of the Search's own immutable Input, so a
    Composite this Search spawns never reads configuration, never asks a
    resolver for a capability, and never queries the run to rediscover a fact
    its owner already had."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_name: str
    progress_enabled: bool
    worker: ExecutionCapability
    score: ScoreLaunch


class WorkerCandidateInput(WorkflowBaseModel):
    """One fresh Worker attempt: a single plan run end to end by one Worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan: WorkerDesignPlan
    # Advisory external results the Search had when it admitted this Worker;
    # a live read taken at the spawn site, never re-derived inside the
    # Composite itself.
    external_scores: tuple[ExternalScore, ...]
    grant: CandidateGrant

    @property
    def requested_gpus(self) -> int:
        return self.plan.requested_gpus


class WorkerCandidateOutput(ScoreOutput):
    """The terminal success Output of one scored agentic Worker candidate.

    It IS the Score Program's own Output plus the Worker's own feedback, so
    the score, the attempt directory, and the components have one owner and
    are stored once."""

    feedback: str = ""


class EnsembleMember(WorkflowBaseModel):
    """One scored Worker candidate an ensemble was told to combine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uid: str
    directory: str
    # This member's own tree, so an ensemble over N members is granted the N
    # members the Manager named and nothing else.
    files: FileRef
    score: float


class EnsembleInput(WorkflowBaseModel):
    """One bounded combination of already scored Worker candidates.

    Its members are also its direct ``upstream``; their facts are here because
    the Search that selected them had them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal: EnsembleProposal
    members: tuple[EnsembleMember, ...] = Field(min_length=1)
    metric_contract: MetricContractInput
    aggregator: ExecutionCapability
    grant: CandidateGrant
