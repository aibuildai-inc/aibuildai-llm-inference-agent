"""EvolutionarySearch: single-population async steady-state evolution.

The Designer drafts ``num_designs`` initial individuals as the population.
Whenever a free slot opens -- the same ``parallel`` ceiling for the seeds and
for everything bred afterwards -- the method draws one or two fitness-selected
parents and starts ONE complete Worker that performs the mutation or the
crossover itself, implements it, runs it, and exports the result. There is no
Coder, Training Program, Reviser or Crossover role, and no breeding Action
queued in front of a member: a slot holds a member, never a proposal for one.
No LLM judge or selector is used, and the Aggregator ensembles the whole
scored population at the end.

``select_evolutionary_parents`` is the one random step of the algorithm and is
replayed, not re-derived, on resume. The population it draws from is this
Search's OWN record of the members it started -- never rediscovered."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from engine.builtin.aibuildai.io import SearchOutput, task_data
from engine.builtin.aibuildai.search import SEARCH_TEMPLATE, AIBuildAISearch
from engine.builtin.evolutionary.agents.aggregator.agent import AggregatorAgent
from engine.builtin.evolutionary.agents.designer.agent import DesignerAgent
from engine.builtin.evolutionary.agents.designer.io import DesignerInput
from engine.builtin.evolutionary.agents.worker.agent import WorkerAgent
from engine.builtin.evolutionary.candidate import Ensemble, WorkerCandidate
from engine.builtin.evolutionary.io import (
    EnsembleInput,
    EvolutionarySearchInput,
    EvolutionarySearchParameters,
    WorkerCandidateInput,
    WorkerCandidateOutput,
    build_input,
)
from engine.durable_execution import run_host
from engine.durable_execution import Handle
from engine.execution_output import ExecutionOutput
from engine.failure import Failure, FailureKind
from engine.metric_contract import MetricContractInput
from engine.paths import RunPaths

if TYPE_CHECKING:
    from config import AgentConfig

_AnyHandle = Handle[ExecutionOutput[bool]]
_CandidateHandle = Handle[WorkerCandidateOutput | Failure]


def select_evolutionary_parents(
    population: "tuple[tuple[str, float], ...]",
    crossover_prob: float,
    direction: str,
) -> "tuple[str, ...]":
    """Choose one mutation parent or two crossover parents once.

    ``population`` is every SCORED member, as (uid, score): a failed member has
    no fitness and is never bred from. Fitness maps the metric onto [0, 1] with
    higher = fitter for the run's own direction (no spread means uniformly fit),
    and a fixed-temperature softmax turns that into relative ``random.choices``
    weights, so every member keeps a non-zero chance. Called through
    ``ctx.step``, so a resumed run reads back the parents it picked."""
    if not population:
        return ()
    metrics = [score for _, score in population]
    lo, hi = min(metrics), max(metrics)
    span = hi - lo
    fitness = [
        1.0
        if span <= 0
        else ((hi - score) / span if direction == "min" else (score - lo) / span)
        for _, score in population
    ]
    top = max(fitness)
    weights = [math.exp(value - top) for value in fitness]
    if len(population) < 2 or random.random() >= crossover_prob:
        return (random.choices(population, weights=weights, k=1)[0][0],)
    picked: list[str] = []
    choices = list(population)
    for _ in range(2):
        choice = random.choices(choices, weights=weights, k=1)[0]
        index = choices.index(choice)
        picked.append(choice[0])
        choices.pop(index)
        weights.pop(index)
    return tuple(picked)


@dataclass
class _Started:
    """One population member this Search launched, and what it produced."""

    composite: WorkerCandidate
    handle: _CandidateHandle
    output: "WorkerCandidateOutput | Failure | None" = None


class EvolutionarySearch(AIBuildAISearch[EvolutionarySearchInput]):
    description = """Single-population async steady-state evolution: the designer seeds the
    population, each free slot is refilled by one complete Worker that mutates one
    fitness-selected parent or recombines two, and the aggregator ensembles the scored population."""
    __doc__ = description

    prompt_template = SEARCH_TEMPLATE
    # designer (seeds) + worker (breeds, implements, runs and exports every
    # member) + aggregator (ensemble). No judge/selector/router here.
    agent_types = (DesignerAgent, WorkerAgent, AggregatorAgent)
    parameters_type = EvolutionarySearchParameters

    @classmethod
    def build_input(
        cls, configs: "AgentConfig", run_paths: RunPaths
    ) -> EvolutionarySearchInput:
        """Validate the user's ``search.input`` and complete this package's Input."""
        return build_input(cls.parameters(configs), configs, run_paths)

    # --- What this Search asks its own children -----------------------------
    def _scored(self) -> "list[tuple[_Started, WorkerCandidateOutput]]":
        """Every member that reached a usable, scored state, oldest first."""
        return [
            (started, started.output)
            for started in self._by_uid.values()
            if isinstance(started.output, WorkerCandidateOutput)
        ]

    async def _member(
        self, payload: WorkerCandidateInput, upstream: "tuple[_AnyHandle, ...]"
    ) -> _Started:
        """Launch one population member and record it as this Search's own."""
        candidate = WorkerCandidate(input=payload)
        handle = cast(
            _CandidateHandle,
            await self.ctx.spawn(candidate.run, upstream=upstream, capture_failure=True),
        )
        started = _Started(composite=candidate, handle=handle)
        self._by_uid[candidate.uid] = started
        return started

    async def _breed(self, metric_contract: MetricContractInput) -> "_Started | None":
        """Draw this breed's parents and start the member that breeds them."""
        population = self._scored()
        if not population:
            return None
        # The choice is random, so it runs as a step: a replay reads back the
        # parents this run picked instead of picking a different pair.
        parent_uids = await self.ctx.step(
            select_evolutionary_parents,
            tuple((started.composite.uid, output.score) for started, output in population),
            self.input.crossover_prob,
            metric_contract.metric_direction,
        )
        if not parent_uids:
            return None
        by_uid = {item[0].composite.uid: item for item in population}
        parents = tuple(by_uid[uid] for uid in parent_uids)
        offspring = WorkerCandidateInput.bred(
            tuple(
                (started.composite.path, started.composite.files(), output)
                for started, output in parents
            ),
            metric_contract=metric_contract,
            external_scores=self.external_scores(),
            grant=self.input.grant,
        )
        return await self._member(
            offspring, tuple(started.handle for started, _ in parents)
        )

    def _settle(
        self, started: _Started, output: "WorkerCandidateOutput | Failure"
    ) -> None:
        """Record one member's result in the order it actually settled.

        The same-failure streak counts settlements, not launches: the order
        members finish in is not the order they started in. A success clears
        it; a different failure kind restarts it at one."""
        started.output = output
        if not isinstance(output, Failure):
            self._streak_kind, self._streak = None, 0
        elif output.kind is self._streak_kind:
            self._streak += 1
        else:
            self._streak_kind, self._streak = output.kind, 1

    # --- The algorithm ------------------------------------------------------
    async def explore(
        self, metric_contract: MetricContractInput
    ) -> "SearchOutput | Failure | None":
        self._by_uid: dict[str, _Started] = {}
        self._streak_kind: FailureKind | None = None
        self._streak = 0
        designer_handle = await self.ctx.spawn(
            DesignerAgent(
                input=DesignerInput(
                    num_proposals=self.input.num_designs,
                    remaining_minutes=(await self.budget.snapshot()).wall_clock_remaining_s
                    / 60.0,
                    done_count=0,
                    gpu_ceiling=run_host().host_visible_gpu_count(),
                    reviewer_capability=self.input.designer_reviewer,
                ),
            ).run,
            "Start the assigned work.",
            upstream=(self.setup_handle,),
            read=(task_data(),),
            capability=self.input.designer,
        )
        designer_output = await designer_handle.result()

        # Seeds this Search has not started yet, oldest first. They obey the
        # same slot ceiling everything bred later obeys.
        unseeded = list(designer_output.plans)
        pending: dict[_CandidateHandle, _Started] = {}

        async def fill_slots() -> None:
            """Fill every free slot: the remaining seeds first, then a breed."""
            while len(pending) < self.input.parallel and not self._admission_closed():
                if unseeded:
                    seed = WorkerCandidateInput.seed(
                        unseeded.pop(0),
                        metric_contract=metric_contract,
                        external_scores=self.external_scores(),
                        grant=self.input.grant,
                    )
                    started = await self._member(seed, (designer_handle,))
                else:
                    bred = await self._breed(metric_contract)
                    if bred is None:
                        return
                    started = bred
                pending[started.handle] = started

        await fill_slots()
        while pending:
            handle = cast(
                _CandidateHandle,
                (await self.ctx.wait(set(pending), min_completed=1)).completed[0],
            )
            started = pending.pop(handle)
            self._settle(started, await handle.result())
            member = started.output
            best = self.selected_output
            if isinstance(member, WorkerCandidateOutput) and (
                best is None
                or (
                    member.score < best.score
                    if metric_contract.metric_direction == "min"
                    else member.score > best.score
                )
            ):
                self.select(SearchOutput(
                    output_dir=member.output_dir, score=member.score,
                    components=member.components,
                ))
            await fill_slots()

        scored = self._scored()
        if not self.worth_ensembling(
            tuple(output.score for _, output in scored), metric_contract
        ):
            return self.selected_output
        ensemble = Ensemble(
            input=EnsembleInput.over(
                tuple(
                    (started.composite.uid, started.composite.files(), out)
                    for started, out in scored
                ),
                metric_contract=metric_contract,
                aggregator=self.input.aggregator,
                score=self.input.grant.score,
            )
        )
        handle = await self.ctx.spawn(
            ensemble.run,
            upstream=tuple(started.handle for started, _ in scored),
            capture_failure=True,
        )
        output = await handle.result()
        if output.failed is True:
            return self.selected_output
        return SearchOutput(
            score=output.score,
            output_dir=output.output_dir,
            components=output.components,
        )

    # --- This Search's own admission and stopping policy -------------------
    def _admission_closed(self) -> bool:
        """Whether this Search has stopped launching evaluations.

        Only STARTED members count -- no breeding Action is left to reserve a
        slot for a member that may never exist."""
        cap = self.input.max_evaluations
        if cap is not None and len(self._by_uid) >= cap:
            return True
        streak = self.input.early_stopping
        return streak > 0 and self._streak >= streak

SEARCH_TYPE = EvolutionarySearch
