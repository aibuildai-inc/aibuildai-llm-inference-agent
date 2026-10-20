"""The agentic package's own Composites.

One Worker candidate is Worker -> Score. The Ensemble is Aggregator -> Score.
Everything a Composite here needs arrives on its own frozen Input: it reads no
configuration, resolves no capability by role name, and asks the run nothing
about work it did not start itself."""

from __future__ import annotations

from engine.builtin.agentic.agents.aggregator.agent import AggregatorAgent
from engine.builtin.agentic.agents.aggregator.io import AggregatorInput
from engine.builtin.agentic.agents.worker.agent import WorkerAgent
from engine.builtin.agentic.agents.worker.io import WorkerInput
from engine.builtin.agentic.io import (
    EnsembleInput,
    WorkerCandidateInput,
    WorkerCandidateOutput,
)
from engine.builtin.aibuildai.programs.score import ScoreOutput
from engine.composite import Composite
from engine.builtin.aibuildai.io import task_data
from engine.durable_execution import action
from engine.failure import Failure

class CompleteWorker(Composite[WorkerCandidateInput]):
    """One fresh Worker attempt, scored after it finishes.

    AgenticSearch always forces zero revisions (its Manager has no tool that
    continues one), so this Composite never runs a revision and produces no
    source tree of its own for a later candidate to inherit."""

    @action
    async def run(self) -> WorkerCandidateOutput | Failure:
        payload = self.input
        grant = payload.grant
        worker = WorkerAgent(
            input=WorkerInput(
                plan=payload.plan,
                revision=None,
                parent_path=None,
                parent_directory=None,
                parent_metric=None,
                task_name=grant.task_name,
                min_revisions=0,
                num_revisions=0,
                gpu_count=payload.requested_gpus,
                progress_enabled=grant.progress_enabled,
                external_scores=payload.external_scores,
            ),
        )
        handle = await self.ctx.spawn(
            worker.run,
            "Start the assigned work.",
            read=(task_data(),),
            capture_failure=True,
            capability=grant.worker.model_copy(
                update={"gpus": payload.requested_gpus}
            ),
        )
        worker_output = await handle.result()
        if worker_output.failed is True:
            return worker_output
        scored_handle = await grant.score.measure(
            self.ctx,
            handle,
            output_dir=f"{worker.artifacts_dir}/full",
            attempt=worker.files(),
        )
        scored = await scored_handle.result()
        if scored.failed is True:
            return scored
        return WorkerCandidateOutput(
            score=scored.score,
            output_dir=scored.output_dir,
            components=scored.components,
            feedback=worker_output.feedback,
        )


class Ensemble(Composite[EnsembleInput]):
    """One bounded combination of already scored Worker candidates."""

    @action
    async def run(self) -> ScoreOutput | Failure:
        payload = self.input
        grant = payload.grant
        aggregator = AggregatorAgent(
            input=AggregatorInput(
                mem_cap=f"{int(payload.aggregator.memory_max_gb or 0)}G",
                member_uids=tuple(member.uid for member in payload.members),
                member_directories=tuple(
                    member.directory for member in payload.members
                ),
                member_metrics=tuple(member.score for member in payload.members),
                metric_contract=payload.metric_contract,
            ),
        )
        # The members are this Composite's own upstream; the Aggregator reads
        # them through its Input.
        handle = await self.ctx.spawn(
            aggregator.run,
            "Start the assigned work.",
            # Derived from the member collection: N members are N references,
            # never N hand-written paths.
            read=(task_data(), *(member.files for member in payload.members)),
            capture_failure=True,
            capability=payload.aggregator,
        )
        output = await handle.result()
        if output.failed is True:
            return output
        scored_handle = await grant.score.measure(
            self.ctx,
            handle,
            output_dir=f"{aggregator.artifacts_dir}/full",
            attempt=aggregator.files(),
        )
        scored = await scored_handle.result()
        return scored
