"""Input and Output records for the agentic package's own Aggregator role."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ConfigDict, Field

from engine.base import WorkflowBaseModel
from engine.execution_output import SuccessfulOutput
from engine.metric_contract import MetricContractInput
from engine.work_unit.agent.base import AgentInput


class EnsembleProposal(WorkflowBaseModel):
    """The creation data for the search's ensemble design.

    The Aggregator builds one more design out of the graded ones, so the ensemble is a design like any other: it carries its members' design UIDs as its creation data, the Aggregator agent fills its attempt dir, and the run's score program grades that dir on the same scale as every plain design. It wins the run only by scoring better.
    """

    model_config = ConfigDict(frozen=True)

    member_uids: tuple[str, ...] = Field(
        description="Graded Worker results this ensemble was built from."
    )


@dataclass(frozen=True)
class AggregatorInput(AgentInput):
    mem_cap: str
    # Only the members this ensemble was told to combine. Its own upstream IS
    # those members, so nothing here carries the rest of the population.
    member_uids: tuple[str, ...]
    member_directories: tuple[str, ...]
    member_metrics: tuple[float | None, ...]
    metric_contract: MetricContractInput | None


class AggregatorOutput(SuccessfulOutput):
    feedback: str = Field(
        default="",
        description=(
            "Ensemble decision summary: which method was chosen and why "
            "(probability averaging / weighted blend / rank averaging / hard "
            "voting / etc.), which designs participated and which "
            "were excluded with reasons, key blending weights, "
            "and any notes about the "
            "output it wrote (calibration applied, members it had to skip)."
        ),
    )
    soft_blended_member_uids: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Composite UIDs whose continuous outputs your ensemble SOFT-AVERAGES into "
            "its own output (probability / logit / rank blending). Record it so a "
            "reader of the run can see which designs the blend was built from. Leave "
            "EMPTY when the output is not a soft average of member outputs: a "
            "single-model pick, hard voting, or a trained stacker / meta-learner."
        ),
    )
