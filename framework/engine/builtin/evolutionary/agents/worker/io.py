"""Input and Output records for the evolutionary package's Worker."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ConfigDict, Field

from engine.base import WorkflowBaseModel
from engine.builtin.aibuildai.agents.submitter.io import ExternalScore
from engine.builtin.evolutionary.agents.designer.io import DesignPlan
from engine.execution_output import SuccessfulOutput
from engine.metric_contract import MetricContractInput
from engine.work_unit.agent.base import AgentInput


class ParentCandidate(WorkflowBaseModel):
    """One completed population member this Worker breeds from.

    Every field is a fact the Search already held when it chose this parent:
    the parent's own formal score and its parts, what its Worker reported, the
    design it started from when it had one, and the exact directories to read.
    Nothing here is recovered by walking the run workspace, and two parents are
    two of these -- never one merged history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    # The seed design this parent started from, or None when the parent was
    # itself bred: an offspring's design lives in its source, not in a plan.
    plan: DesignPlan | None
    # The parent's own source tree, which this Worker reads and copies from,
    # and the attempt directory its score below was measured on. Both are
    # addresses to read; a completed parent is never written to.
    source_dir: str
    output_dir: str
    # The formal score the run's own score program measured, which the parent
    # Worker never saw during its own session.
    score: float
    components: tuple[tuple[str, float], ...]
    feedback: str


@dataclass(frozen=True)
class WorkerInput(AgentInput):
    """One population member: a seed design, a mutation, or a crossover.

    Exactly one of the two sources is filled. ``plan`` alone seeds a fresh
    member. One parent is a mutation and two parents are a recombination, and
    this Worker performs that mutation or recombination itself -- there is no
    separate breeding role whose proposal it carries out."""

    plan: DesignPlan | None
    parents: tuple[ParentCandidate, ...]
    # The run's own metric and its direction. A parent's formal score is
    # meaningless without it: the Worker is told to refine against that
    # number, so it must know which way is better.
    metric_contract: MetricContractInput
    task_name: str
    gpu_count: int
    progress_enabled: bool
    # Advisory external results recorded so far, newest last. Empty when
    # sharing is off, when no submission has been made yet, or when none has
    # returned -- three states the prompt must not conflate.
    external_scores: tuple[ExternalScore, ...]


class WorkerOutput(SuccessfulOutput):
    feedback: str = Field(
        default="",
        description=(
            "What this member actually did and what it left behind: which "
            "change you made to your parents' recipe and why, the "
            "measurements you took, the instability you saw, and any caveats "
            "about the output you wrote. A later member bred from this one "
            "reads this beside its formal score."
        ),
    )
