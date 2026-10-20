"""The evolutionary package's own business records, and how a run's configuration becomes them.

Every record here is local to this package. Another package may spell a field
the same way with its own meaning, or declare none of them. ``build_input``
below is this package's one construction entry: it resolves the user's
settings into the immutable values every child is handed, once, at startup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field

from engine.base import WorkflowBaseModel
from engine.builtin.aibuildai.agents.submitter.io import ExternalScore
from engine.builtin.aibuildai.capability import score_launch
from engine.builtin.aibuildai.programs.score import ScoreLaunch, ScoreOutput
from engine.builtin.evolutionary.agents.aggregator.agent import AggregatorAgent
from engine.builtin.evolutionary.agents.designer.agent import DesignerAgent
from engine.builtin.evolutionary.agents.designer.io import DesignPlan
from engine.builtin.evolutionary.agents.worker.agent import WorkerAgent
from engine.builtin.evolutionary.agents.worker.io import ParentCandidate
from engine.capability import ExecutionCapability
from engine.durable_execution import FileRef
from engine.metric_contract import MetricContractInput
from engine.paths import RunPaths

if TYPE_CHECKING:
    from config import AgentConfig
    from engine.work_unit.agent.base import Agent
    from engine.work_unit.base import WorkUnit


# --- What this package's own children may spend --------------------------
# This package resolves its own roles, from the run's configuration, once at
# startup. It is deliberately a copy of what every other package does: no
# shared layer answers for an arbitrary role name, so one package's children
# can never be named, reached, or re-priced from another. The question is
# always asked of a Definition class this package owns, never of a string.
def _capability(
    configs: "AgentConfig", unit_type: "type[WorkUnit]"
) -> ExecutionCapability:
    """What one child of this package may spend."""
    kind, name = unit_type.kind, unit_type.name
    if name is None:
        raise AssertionError(f"{unit_type.__name__} declares no configured name")
    time_config = configs.work_unit_time_for(kind, name)
    limit = configs.resources.work_unit.limit_for(kind, name)
    return ExecutionCapability(
        wall_clock_seconds=time_config.cap_seconds(
            expected_minutes=None,
            pipeline_minutes=configs.run.budget.wall_clock_minutes,
        ),
        cpu_max_cores=limit.cpu_max_cores,
        memory_max_gb=float(limit.memory_max_gb),
        gpus=0,
        # File isolation is physical, so it is declared by the unit that does
        # the work, not by a container above it. None where the run did not ask
        # for it, so a unit never argues with an owner that did.
        retry=time_config.retry,
    )


def _reviewer(
    configs: "AgentConfig", producer: "type[Agent]"
) -> ExecutionCapability | None:
    """What this role's own semantic review may spend, when this run composes one.

    None when the role composes no reviewer for this configuration. That None
    is the whole answer: it travels to the role as its frozen
    ``reviewer_capability``, so nothing asks configuration again at run time."""
    reviewer = producer.semantic_reviewer()
    if reviewer is None or not producer.composes_reviewer(configs):
        return None
    return _capability(configs, reviewer)


# Unset crossover_prob resolves to this (user-set default): half of all
# breeds recombine two parents, half mutate one.
_DEFAULT_CROSSOVER_PROB = 0.5


class EvolutionarySearchParameters(WorkflowBaseModel):
    """EvolutionarySearch's own business parameters: the half a user writes.

    One offspring per breed is fixed (``num_revisions``); the population shape
    and breeding odds are free. Local to this kind: another Search may spell a
    field the same way with its own meaning, or not declare it at all."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parallel: int = Field(
        gt=0,
        description="max concurrently running population members.",
        json_schema_extra={"placeholder": 4},
    )
    num_designs: int | None = Field(
        default=None,
        ge=1,
        description="initial population size the designer seeds. null = the "
        "designer decides the count from task complexity.",
    )
    num_revisions: Literal[1] = 1
    crossover_prob: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="per breed, the Bernoulli probability of recombining TWO "
        "fitness-selected parents (crossover) instead of mutating one; "
        "degrades to mutation when the population has fewer than 2 scored "
        "individuals. null defaults to 0.5; 0.0 = pure mutation, 1.0 = always "
        "crossover (when 2+ scored parents exist).",
    )
    max_evaluations: int | None = Field(
        default=None,
        ge=0,
        description="attempted-evaluation ceiling (failures occupy a slot); 0 "
        "runs no evaluations at all. null = no count cap beyond the run-level "
        "wall-clock + cost budget.",
    )
    early_stopping: int = Field(
        default=3,
        ge=0,
        description="stop the search once this many completed evaluations in a "
        "row have all failed the same way (the same failure.kind). 0 turns "
        "this off.",
    )


class CandidateGrant(WorkflowBaseModel):
    """What one population member needs to run and score its own Worker.

    Frozen at the spawn site out of the Search's own immutable Input, so a
    member never reads configuration, never asks a resolver for a capability,
    and never queries the run to rediscover a fact its owner already had."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_name: str
    progress_enabled: bool
    worker: ExecutionCapability
    # None when this run composes no Worker reviewer; resolved once by startup
    # either way, so the Worker is constructed with its answer instead of
    # asking a resolver at the moment it runs.
    worker_reviewer: ExecutionCapability | None
    score: ScoreLaunch


class EvolutionarySearchInput(EvolutionarySearchParameters):
    """EvolutionarySearch's complete immutable Input.

    The business parameters above are the user's. Everything below is resolved
    once by ``build_input``, so nothing this Search creates ever reads
    configuration or asks for a capability by role name."""

    # The user may leave the breeding odds unset; this Search never sees that
    # null, because the construction entry below resolves it once.
    crossover_prob: float = _DEFAULT_CROSSOVER_PROB
    designer: ExecutionCapability
    designer_reviewer: ExecutionCapability | None
    aggregator: ExecutionCapability
    grant: CandidateGrant


def build_input(
    parameters: dict[str, Any], configs: "AgentConfig", run_paths: RunPaths
) -> EvolutionarySearchInput:
    """Complete this package's Input from the user's validated parameters."""
    resolved = dict(parameters)
    if resolved.get("crossover_prob") is None:
        resolved["crossover_prob"] = _DEFAULT_CROSSOVER_PROB
    return EvolutionarySearchInput(
        **resolved,
        designer=_capability(configs, DesignerAgent),
        designer_reviewer=_reviewer(configs, DesignerAgent),
        aggregator=_capability(configs, AggregatorAgent),
        grant=CandidateGrant(
            task_name=configs.run.task_name,
            progress_enabled=configs.progress.enable,
            worker=_capability(configs, WorkerAgent),
            worker_reviewer=_reviewer(configs, WorkerAgent),
            score=score_launch(configs, run_paths),
        ),
    )


class WorkerCandidateInput(WorkflowBaseModel):
    """One population member: a seed design, a mutation, or a crossover.

    The three differ by the business facts below, not by three Composite
    classes: a seed states its plan and no parent, a mutation states one
    parent, and a crossover states two. The two constructors below are the
    only ways this record is built, so the Search states WHICH member it is
    starting and this record works out what that member is handed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan: DesignPlan | None
    parents: tuple[ParentCandidate, ...] = Field(max_length=2)
    # Each parent's own tree. ``parents`` says what those members scored and
    # which directories to read; this is what lets this member read them.
    parent_files: tuple[FileRef, ...] = Field(max_length=2)
    # What this member's Worker may claim. A seed claims what its own plan
    # asked for; a bred member claims the most any of its parents claimed.
    # That ceiling is a search policy the owner accepted, not a
    # refactor-equivalent substitution for what BASE did. One Worker invents
    # the mutation or recombination and runs it in the same session, so
    # nothing states an offspring's resource need before its WorkUnit is
    # spawned, and its parents' claim is the only bound knowable then. The
    # accepted cost: a CPU-only lineage never breeds a GPU offspring, and a
    # GPU parent's descendants keep that claim even when the recipe they
    # invent does not need it.
    requested_gpus: int = Field(ge=0)
    # The run's own metric and its direction, so the Worker that is told to
    # refine against its parent's formal score knows which way is better.
    metric_contract: MetricContractInput
    # Advisory external results the Search had when it admitted this member; a
    # live read taken at the spawn site, never re-derived inside the Composite.
    external_scores: tuple[ExternalScore, ...]
    grant: CandidateGrant

    @classmethod
    def seed(
        cls,
        plan: DesignPlan,
        *,
        metric_contract: MetricContractInput,
        external_scores: tuple[ExternalScore, ...],
        grant: CandidateGrant,
    ) -> "WorkerCandidateInput":
        """One fresh member, from one of the Designer's own starting plans."""
        return cls(
            plan=plan,
            parents=(),
            parent_files=(),
            requested_gpus=plan.requested_gpus,
            metric_contract=metric_contract,
            external_scores=external_scores,
            grant=grant,
        )

    @classmethod
    def bred(
        cls,
        parents: tuple[tuple[str, FileRef, "WorkerCandidateOutput"], ...],
        *,
        metric_contract: MetricContractInput,
        external_scores: tuple[ExternalScore, ...],
        grant: CandidateGrant,
    ) -> "WorkerCandidateInput":
        """One member bred from one parent (mutation) or two (crossover).

        Each parent arrives as its durable path beside the Output it settled
        with, so every fact the offspring is handed is one the Search already
        held. Two parents stay two records: their lineages are never merged
        into one history here or in the prompt that renders them."""
        return cls(
            plan=None,
            parents=tuple(
                ParentCandidate(
                    path=path,
                    plan=output.plan,
                    source_dir=output.source_dir,
                    output_dir=output.output_dir,
                    score=output.score,
                    components=output.components,
                    feedback=output.feedback,
                )
                for path, _, output in parents
            ),
            parent_files=tuple(files for _, files, _ in parents),
            requested_gpus=max(
                (output.requested_gpus for _, _, output in parents), default=0
            ),
            metric_contract=metric_contract,
            external_scores=external_scores,
            grant=grant,
        )


class EnsembleMember(WorkflowBaseModel):
    """One scored population member an ensemble was told to combine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uid: str
    # This member's own tree, so an ensemble over N members is granted the N
    # members the Search chose and nothing else.
    files: FileRef
    # The exact directory this member's own score was measured on, so the
    # Aggregator opens the scored output instead of guessing which directory
    # under a Composite holds it.
    output_dir: str
    score: float


class EnsembleInput(WorkflowBaseModel):
    """One bounded combination of already scored population members.

    Its members are also its direct ``upstream``; their facts are here because
    the Search that selected them had them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    members: tuple[EnsembleMember, ...] = Field(min_length=1)
    metric_contract: MetricContractInput
    memory_cap: str
    aggregator: ExecutionCapability
    score: ScoreLaunch

    @classmethod
    def over(
        cls,
        members: "tuple[tuple[str, FileRef, ScoreOutput], ...]",
        *,
        metric_contract: MetricContractInput,
        aggregator: ExecutionCapability,
        score: ScoreLaunch,
    ) -> "EnsembleInput":
        """The combination of the members the Search chose, as its own record.

        Each member arrives as its uid beside the Output it settled with, so
        every fact here is one the Search already held; the Aggregator is
        never asked to find a member's scored directory for itself."""
        chosen = tuple(
            EnsembleMember(
                uid=uid, files=files, output_dir=output.output_dir, score=output.score
            )
            for uid, files, output in members
        )
        return cls(
            members=chosen,
            metric_contract=metric_contract,
            memory_cap=f"{int(aggregator.memory_max_gb or 0)}G",
            aggregator=aggregator,
            score=score,
        )


class WorkerCandidateOutput(ScoreOutput):
    """The terminal success Output of one scored population member.

    It IS the Score Program's own Output plus this package's extra business
    facts, so the score, the attempt directory and the components have one
    owner and are stored once. Everything an offspring bred from this member
    needs is here, which is why that offspring is handed values rather than a
    directory to walk."""

    # This member's own directory: where its Worker kept the source, and what
    # an offspring reads and copies from.
    source_dir: str
    # The seed design this member started from, or None when the member was
    # itself bred: an offspring's design lives in its source, not in a plan.
    plan: DesignPlan | None
    feedback: str
    # What this member's Worker actually claimed, so an offspring bred from it
    # can inherit a claim large enough to run the recipe it inherits.
    requested_gpus: int
