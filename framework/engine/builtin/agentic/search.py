"""AgenticSearch: one Manager Agent decides the whole search.

The one built-in package that hands its whole exploration to a single model
turn: ``ManagerAgent`` gets three tools -- designer, worker, aggregator -- and
drives the search itself, calling each tool as many times as it judges
worthwhile before ending with an empty Structured Output. Everything else --
which Worker candidates were started, which combination the Manager finally
built -- is work this Search delegated to that Manager, and it asks the SDK
for it. The tools run on the MANAGER's Action workflow, and every workflow
restores its own execution objects, so what one of them writes on a Python
object no other one has; ``self.delegated`` is how a Search that did not spawn
its own children still receives them, their Outputs and the Handles that name
their exact Actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

from pydantic import ConfigDict, Field

from engine.base import WorkflowBaseModel
from engine.builtin.aibuildai.search import SEARCH_TEMPLATE, AIBuildAISearch
from engine.builtin.aibuildai.capability import score_launch
from engine.builtin.aibuildai.io import SearchOutput, task_data
from engine.builtin.aibuildai.programs.score import ScoreOutput
from engine.builtin.agentic.agents.aggregator.agent import AggregatorAgent
from engine.builtin.agentic.agents.aggregator.io import EnsembleProposal
from engine.builtin.agentic.agents.designer.agent import WorkerDesignerAgent
from engine.builtin.agentic.agents.designer.io import DesignerInput, WorkerDesignerOutput, WorkerDesignPlan
from engine.builtin.agentic.agents.worker.agent import WorkerAgent
from engine.builtin.agentic.candidate import CompleteWorker, Ensemble
from engine.builtin.agentic.io import (
    CandidateGrant,
    EnsembleInput,
    EnsembleMember,
    WorkerCandidateInput,
    WorkerCandidateOutput,
)
from engine.capability import ExecutionCapability
from engine.durable_execution import (
    ExecutionContext,
    Handle,
    own_files,
    run_host,
)
from engine.execution_output import ExecutionOutput, SuccessfulOutput
from engine.failure import Failure, FailureKind
from engine.metric_contract import MetricContractInput
from engine.paths import RunPaths
from engine.search import RecordedAction
from engine.work_unit.agent.base import Agent, AgentInput, tool
from engine.work_unit.agent.policy import RolePolicy

if TYPE_CHECKING:
    from config import AgentConfig
    from engine.work_unit.base import WorkUnit

_AnyHandle = Handle[ExecutionOutput[bool]]


# --- What this package's own children may spend --------------------------
# This package resolves its own roles, from the run's configuration, once at
# startup. It is deliberately a copy of what every other package does: no
# shared layer answers for an arbitrary role name, so one package's children
# can never be named, reached, or re-priced from another. The question is
# always asked of a Definition class this package owns, never of a string.
def _capability(
    configs: "AgentConfig",
    unit_type: "type[WorkUnit]",
    *,
    expected_minutes: float | None = None,
    gpus: int | tuple[int, ...] = 0,
) -> ExecutionCapability:
    """What one child of this package may spend."""
    kind, name = unit_type.kind, unit_type.name
    if name is None:
        raise AssertionError(f"{unit_type.__name__} declares no configured name")
    time_config = configs.work_unit_time_for(kind, name)
    limit = configs.resources.work_unit.limit_for(kind, name)
    return ExecutionCapability(
        wall_clock_seconds=time_config.cap_seconds(
            expected_minutes=expected_minutes,
            pipeline_minutes=configs.run.budget.wall_clock_minutes,
        ),
        cpu_max_cores=limit.cpu_max_cores,
        memory_max_gb=float(limit.memory_max_gb),
        gpus=gpus,
        # File isolation is physical, so it is declared by the unit that does
        # the work, not by a container above it. None where the run did not ask
        # for it, so a unit never argues with an owner that did.
        retry=time_config.retry,
    )



@dataclass(frozen=True)
class ManagerInput(AgentInput):
    """What the Manager needs that only the algorithm knows.

    The grading contract is settled before the exploration opens, and the
    ``aggregator`` tool is where this package uses it. It travels HERE because
    an execution is recorded as its Input plus its Capability and rebuilt from
    exactly those two: a fact carried on any other attribute is simply absent
    on the instance that runs the Action, which is a different object on the
    Manager's own workflow. The prompt renders none of it."""

    metric_contract: MetricContractInput


class ManagerAgent(Agent[ManagerInput, SuccessfulOutput]):
    """The AgenticSearch scheduler: it designs, evaluates, and aggregates by calling its own tools, and ends by submitting its empty Structured Output.

    It belongs to AgenticSearch alone, which is why it is defined here rather
    than in the Agent family: it reaches into this module's own admission and
    ensemble policy, and a general Agent module that reached back into Search
    would close a Search -> Agent -> Search cycle. Its three tools are ordinary
    business methods that run on this Agent's own Action Workflow, outside its
    opaque conversation Step, so each one starts a real child Action through
    ``self.ctx`` and hands the model back the typed result it settled with.

    The Search it works for is its own PARENT in the durable graph, never
    ``self.search``: that answers with the running ROOT Search service, so a
    Manager under a nested AgenticSearch reaches ``self.parent`` -- ITS OWN
    owner -- and calls the ``run_designer`` / ``run_worker`` / ``run_ensemble``
    methods that owner declares, never a root-scoped population.

    It starts work and reads what that work left. Its read grant is its OWN
    subtree, because every Designer, Worker and Ensemble it causes is a durable
    descendant of it; sibling and root work belongs to somebody else. It writes
    nothing but its own scratch, and it holds no command tool, so every execution
    it causes is one of its three declared tools. It grants no ``mcp__*``,
    ``Skill``, ``Agent`` or ``Task``, which is how that last sentence stays true:
    the operator's configured servers default to every active role and an
    arbitrary one of them may start work, and a default-scoped plugin agent would
    otherwise have the spec builder hand this role ``Agent`` and ``Task`` as well.
    Its own tools and the read-only ``resources`` server are the product's own
    mounts and do not come from that grant.
    """

    name: ClassVar[str] = "manager"
    prompt_template = "agent/manager.j2"
    policy = RolePolicy(tools=("Read", "Glob", "Grep"))

    @tool
    async def designer(self) -> WorkerDesignerOutput | Failure:
        """Run one Designer and return the design plans it proposes. Call it again whenever more designs are worth having; each call is a fresh Designer that sees what the search has done so far."""
        return await cast("AgenticSearch", self.parent).run_designer(self.ctx, self)

    @tool
    async def worker(self, plan: WorkerDesignPlan) -> WorkerCandidateOutput | Failure:
        """Run one complete Worker candidate on one design plan and return its scored result. Pass one plan's own fields directly as this tool's arguments, exactly as `designer` returned them; do not wrap them in a `plan` object. The Search decides whether the candidate is admitted; a refusal comes back as this tool's failure and means no more evaluations may start."""
        return await cast("AgenticSearch", self.parent).run_worker(self.ctx, self, plan)

    @tool
    async def aggregator(self, proposal: EnsembleProposal) -> ScoreOutput | Failure:
        """Combine the candidates you name into one final scored result. Name the uid of every successful `worker` result you want combined. A successful combination becomes this search's result; without one, the best single `worker` result is delivered instead."""
        return await cast("AgenticSearch", self.parent).run_ensemble(
            self.ctx, self, proposal, self.input.metric_contract
        )


class AgenticSearchParameters(WorkflowBaseModel):
    """AgenticSearch's own business parameters: the half a user writes.

    Local to this kind: another Search may spell a field the same way with its
    own meaning, or not declare it at all."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_designs: int | None = Field(
        default=None,
        ge=1,
        description="starting complete-Worker evaluations. null = the "
        "Designer decides the count from task complexity.",
    )
    max_evaluations: int | None = Field(
        default=None,
        ge=0,
        description="attempted-evaluation ceiling (failures occupy a slot); 0 "
        "runs no evaluations at all. null = no count cap beyond the run-level "
        "wall-clock + cost budget.",
    )


class AgenticSearchInput(AgenticSearchParameters):
    """AgenticSearch's complete immutable Input.

    The business parameters above are the user's. Everything below is resolved
    once by startup, so nothing this Search creates ever reads configuration or
    asks for a capability by role name."""

    manager: ExecutionCapability
    designer: ExecutionCapability
    worker: ExecutionCapability
    aggregator: ExecutionCapability
    grant: CandidateGrant


class AgenticSearch(AIBuildAISearch[AgenticSearchInput]):
    description = (
        "One Manager Agent runs the search: it calls a Designer, runs the "
        "Worker candidates it chooses, and combines the results it keeps."
    )
    __doc__ = description


    prompt_template = SEARCH_TEMPLATE
    agent_types = (ManagerAgent, WorkerDesignerAgent, WorkerAgent, AggregatorAgent)
    parameters_type = AgenticSearchParameters

    @classmethod
    def build_input(cls, configs: "AgentConfig", run_paths: RunPaths) -> AgenticSearchInput:
        """Validate the user's ``search.input`` and complete this package's Input."""
        return AgenticSearchInput(
            **cls.parameters(configs),
            manager=_capability(configs, ManagerAgent),
            designer=_capability(configs, WorkerDesignerAgent),
            worker=_capability(configs, WorkerAgent),
            aggregator=_capability(configs, AggregatorAgent),
            grant=CandidateGrant(
                task_name=configs.run.task_name,
                progress_enabled=configs.progress.enable,
                worker=_capability(configs, WorkerAgent),
                score=score_launch(configs, run_paths),
            ),
        )

    # --- What the Manager's tools call on their own owner -------------------
    # Each of these answers about work this Search delegated to that Manager,
    # so each asks ``self.delegated`` rather than an attribute: the candidates
    # are that Manager's own children, and what each settled with is the
    # Output of the Action that ran it. That is the same answer on the
    # workflow that started it, on this Search's own workflow, and after a
    # resume, and none of it is a fact this package keeps.
    async def _candidates(
        self, ctx: ExecutionContext, manager: ManagerAgent
    ) -> "tuple[RecordedAction[CompleteWorker], ...]":
        """Every Worker candidate one Manager started, in the order it did."""
        return await self.delegated(manager, CompleteWorker, on=ctx)

    async def _done_count(self, ctx: ExecutionContext, manager: ManagerAgent) -> int:
        return sum(
            1
            for candidate in await self._candidates(ctx, manager)
            if isinstance(candidate.output, WorkerCandidateOutput)
        )

    async def _admission_refusal(
        self, ctx: ExecutionContext, manager: ManagerAgent
    ) -> "str | None":
        """Why this Search refuses another Worker candidate, if it does.

        Its own count, from the candidates its Manager started. A candidate
        recorded before its Action opened still occupies a slot, which is why
        this counts the candidates and not the Actions they ran."""
        cap = self.input.max_evaluations
        attempted = len(
            {candidate.execution.uid for candidate in await self._candidates(ctx, manager)}
        )
        if cap is not None and attempted >= cap:
            return (
                f"{attempted} evaluations were attempted, which is the "
                f"max_evaluations={cap} ceiling this search was given"
            )
        return None

    async def run_designer(
        self, ctx: ExecutionContext, manager: ManagerAgent
    ) -> WorkerDesignerOutput | Failure:
        """Run one Designer for this Search's own Manager and return its plans.

        Everything durable here is journaled on ``ctx``, the MANAGER's running
        Action, because that is the Action this call is part of: the Manager
        holds it open while its tool runs, and this Search's own ``run`` Action
        is not the one executing. That is the whole reason the budget
        observation names ``ctx``; asking ``self.budget`` to record it where
        this Search is parked is where an ``AgenticSearch is not running``
        assertion came from."""
        designer = WorkerDesignerAgent(
            input=DesignerInput(
                num_proposals=self.input.num_designs,
                remaining_minutes=(
                    await self.budget.snapshot(on=ctx)
                ).wall_clock_remaining_s
                / 60.0,
                done_count=await self._done_count(ctx, manager),
                gpu_ceiling=run_host().host_visible_gpu_count(),
            ),
        )
        return await (
            await ctx.spawn(
                designer.run,
                "Start the assigned work.",
                read=(task_data(),),
                capture_failure=True,
                capability=self.input.designer,
            )
        ).result()

    async def run_worker(
        self, ctx: ExecutionContext, manager: ManagerAgent, plan: WorkerDesignPlan
    ) -> WorkerCandidateOutput | Failure:
        """Admit and run one Worker candidate for this Search's own Manager."""
        refusal = await self._admission_refusal(ctx, manager)
        if refusal is not None:
            return Failure(kind=FailureKind.PERMANENT, reason=refusal)
        candidate = CompleteWorker(
            input=WorkerCandidateInput(
                plan=plan,
                external_scores=self.external_scores(),
                grant=self.input.grant,
            )
        )
        result = await (await ctx.spawn(candidate.run, capture_failure=True)).result()
        if isinstance(result, WorkerCandidateOutput) and not any(
            isinstance(ensemble.output, ScoreOutput)
            for ensemble in await self.delegated(manager, Ensemble, on=ctx)
        ):
            best = self.selected_output
            if best is None or (
                result.score < best.score
                if manager.input.metric_contract.metric_direction == "min"
                else result.score > best.score
            ):
                self.select(SearchOutput(
                    output_dir=result.output_dir, score=result.score,
                    components=result.components,
                ))
        return result

    async def run_ensemble(
        self,
        ctx: ExecutionContext,
        manager: ManagerAgent,
        proposal: EnsembleProposal,
        metric_contract: MetricContractInput,
    ) -> ScoreOutput | Failure:
        """Combine the Worker results this Search's own Manager names.

        Same failed-Worker rule as the rest of this product world: a uid this
        Search never scored -- because it never started it, or because it
        started it and it failed -- is refused rather than salvaged into the
        combination."""
        scored = {
            candidate.execution.uid: (candidate, output)
            for candidate in await self._candidates(ctx, manager)
            for output in (candidate.output,)
            if isinstance(output, WorkerCandidateOutput)
        }
        members: list[EnsembleMember] = []
        sources: list[_AnyHandle] = []
        for uid in proposal.member_uids:
            found = scored.get(uid)
            if found is None:
                return Failure(
                    kind=FailureKind.PERMANENT,
                    reason=f"{uid!r} is not one of this search's own scored "
                    f"{WorkerAgent.name} results",
                )
            candidate, output = found
            members.append(
                EnsembleMember(
                    uid=uid,
                    directory=candidate.execution.directory,
                    files=candidate.execution.files(),
                    score=output.score,
                )
            )
            sources.append(candidate.require_handle())
        if not members:
            return Failure(
                kind=FailureKind.NO_OUTPUT,
                reason=f"{AggregatorAgent.name} has no scored candidate to combine",
            )
        if metric_contract.metric_direction == "max" and all(
            abs(member.score) <= 1e-12 for member in members
        ):
            return Failure(
                kind=FailureKind.NO_OUTPUT,
                reason=f"{AggregatorAgent.name} cannot improve an all-zero candidate set",
            )
        ensemble = Ensemble(
            input=EnsembleInput(
                proposal=proposal,
                members=tuple(members),
                metric_contract=metric_contract,
                aggregator=self.input.aggregator,
                grant=self.input.grant,
            )
        )
        combined = await (
            await ctx.spawn(
                ensemble.run, upstream=tuple(sources), capture_failure=True
            )
        ).result()
        if isinstance(combined, ScoreOutput):
            self.select(SearchOutput(
                output_dir=combined.output_dir, score=combined.score,
                components=combined.components,
            ))
        return combined

    # --- The algorithm --------------------------------------------------
    async def explore(self, metric_contract: MetricContractInput) -> "SearchOutput | Failure | None":
        # The Manager declares exactly what its own role was configured, with
        # nothing stripped out of it. It used to be spawned with its CPU,
        # memory and GPU numbers blanked, because a ceiling on this role was
        # also the ceiling every candidate it started could declare, and an
        # agent-sized number would have refused the first candidate that asked
        # for more. That is no longer how a limit reaches a descendant: the
        # limits belong to the Action that holds a process open, and this
        # Search's own Action claims its level without ceilings, so the
        # Manager's numbers bound the Manager and nothing under it.
        manager = ManagerAgent(input=ManagerInput(metric_contract=metric_contract))
        await (
            await self.ctx.spawn(
                manager.run,
                "Manage this search.",
                upstream=(self.setup_handle,),
                # The one live-subtree case. Its tools hand this Manager's own
                # ExecutionContext back to this Search, so every candidate and
                # ensemble it starts is a child of the MANAGER, under the
                # Manager's own tree. It reads that tree and nothing wider:
                # one reference, recorded once, covering descendants that do
                # not exist yet when the grant is written.
                read=(task_data(), own_files()),
                capability=self.input.manager,
            )
        ).result()
        return self.selected_output


SEARCH_TYPE = AgenticSearch
