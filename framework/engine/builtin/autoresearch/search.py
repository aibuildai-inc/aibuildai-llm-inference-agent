"""AutoResearch: one shared post-training codebase, improved one step at a time.

The state is a directory, not a candidate list, yet each step is still an
independent result: it is scored over the immutable attempt it exported. The
Search prompt states the run shape and the Worker prompt owns the step
contract; nothing here runs a Git command or reads Git state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic import ConfigDict, Field

from engine.base import WorkflowBaseModel
from engine.builtin.aibuildai.capability import score_launch
from engine.builtin.aibuildai.io import SearchOutput, task_data
from engine.builtin.aibuildai.programs.score import ScoreLaunch, ScoreOutput
from engine.builtin.aibuildai.search import SEARCH_TEMPLATE, AIBuildAISearch
from engine.builtin.autoresearch.agent import (AutoResearchWorkerAgent,
                                               AutoResearchWorkerInput)
from engine.builtin.autoresearch.template import POST_TRAIN_FILES
from engine.capability import ExecutionCapability
from engine.durable_execution import Handle
from engine.execution_output import ExecutionOutput
from engine.failure import Failure
from engine.metric_contract import MetricContractInput
from engine.paths import RunPaths
from infra.fs.artifact_dir import publish_link

if TYPE_CHECKING:
    from config import AgentConfig

logger = logging.getLogger(__name__)

# Post-training is one card's work and v1 runs one Worker at a time, so the
# call asks for one. Nothing above it names a card: each Action declares the
# cards it needs, and the run's allowed device set bounds what it may get.
WORKER_GPUS = 1

_AnyHandle = Handle[ExecutionOutput[bool]]

# The two resources this Search owns under its own directory. The names live
# here because what "the codebase" and "the records" mean is AutoResearch's
# business, and nothing outside the package has to learn either one.
_CODE_DIRNAME = "code"
_RECORDS_DIRNAME = "records"


def _initialize_source(source_dir: str) -> None:
    """Write the shipped project into the shared source tree, exactly once.

    Staged beside the tree and published by rename, so a half-written tree
    cannot exist and a died-before-publishing attempt fails the next one
    loudly. An existing tree is left alone: after step 1 every file belongs to
    the Workers, deletions included."""
    tree = Path(source_dir)
    if tree.is_dir():
        return
    staging = Path(f"{source_dir}.staging")
    staging.mkdir(parents=True)
    for name, body in POST_TRAIN_FILES.items():
        (staging / name).write_text(body, encoding="utf-8")
    staging.rename(tree)


class AutoResearchSearchParameters(WorkflowBaseModel):
    """AutoResearch's own business parameters: the half a user writes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_steps: int | None = Field(
        default=None,
        gt=0,
        description="maximum number of fresh Worker identities to start; a "
        "retry, a commit, or a training run inside one step is not another "
        "step. null = no count cap beyond the run-level wall-clock + cost "
        "budget.",
    )


class AutoResearchSearchInput(AutoResearchSearchParameters):
    """The user's parameter above, plus what startup resolves once so nothing
    this Search creates reads configuration."""

    progress_enabled: bool
    worker: ExecutionCapability
    score: ScoreLaunch


class AutoResearchSearch(AIBuildAISearch[AutoResearchSearchInput]):
    #: Text shown for this method in the starter config.
    description = (
        "AutoResearch: one shared post-training codebase that every step "
        "improves in place. The run writes a canonical prepare/train/eval/"
        "export project once, then gives it to one fresh Worker at a time; "
        "each reads the earlier steps' records and Git history, makes one "
        "bounded change, runs it, and exports an attempt the framework scores. "
        "The best framework score wins."
    )
    __doc__ = description

    prompt_template = SEARCH_TEMPLATE
    agent_types = (AutoResearchWorkerAgent,)
    parameters_type = AutoResearchSearchParameters

    @classmethod
    def build_input(
        cls, configs: "AgentConfig", run_paths: RunPaths
    ) -> AutoResearchSearchInput:
        """Validate the user's ``search.input`` and complete this package's Input."""
        # Asked of this package's own Definition class, never of a role name a
        # sibling could also spell.
        role = AutoResearchWorkerAgent
        time_config = configs.work_unit_time_for(role.kind, role.name)
        limit = configs.resources.work_unit.limit_for(role.kind, role.name)
        return AutoResearchSearchInput(
            **cls.parameters(configs),
            progress_enabled=configs.progress.enable,
            worker=ExecutionCapability(
                wall_clock_seconds=time_config.cap_seconds(
                    expected_minutes=None,
                    pipeline_minutes=configs.run.budget.wall_clock_minutes,
                ),
                cpu_max_cores=limit.cpu_max_cores,
                memory_max_gb=float(limit.memory_max_gb),
                gpus=WORKER_GPUS,
                retry=time_config.retry,
            ),
            score=score_launch(configs, run_paths),
        )

    # --- The algorithm ------------------------------------------------------
    async def explore(
        self, metric_contract: MetricContractInput
    ) -> "SearchOutput | Failure | None":
        # This Search's own two resources, named here and nowhere else: the
        # shared project every step writes, and the record index only this
        # Search writes -- an index of links, so a reader of it reads the step
        # artifacts it names. Nothing outside this package knows either name.
        code = self.files(f"{_CODE_DIRNAME}/post_train")
        records = self.files(_RECORDS_DIRNAME, links=True)
        source_dir, records_dir = code.path(), records.path()
        await self.ctx.step(_initialize_source, source_dir)
        started = 0
        # What the next step waits for, so two Workers never hold the tree.
        upstream: _AnyHandle = self.setup_handle
        record: str | None = None
        failure: str | None = None
        while (refusal := self._admission_refusal(started)) is None:
            started += 1
            worker = AutoResearchWorkerAgent(
                input=AutoResearchWorkerInput(
                    step=started,
                    max_steps=self.input.max_steps,
                    source_dir=source_dir,
                    records_dir=records_dir,
                    progress_enabled=self.input.progress_enabled,
                    previous_record=record,
                    previous_failure=failure,
                ),
            )
            step = await self.ctx.spawn(
                worker.run,
                "Carry the shared post-training project forward by one step.",
                upstream=(upstream,),
                read=(task_data(), records),
                # Several steps may hold this one tree at once; that one
                # Worker runs at a time is this Search's choice, made by the
                # ``upstream`` above, and nothing in the runtime enforces it.
                write=(code,),
                capture_failure=True,
                capability=self.input.worker,
            )
            upstream = step
            output = await step.result()
            # A failed step is published too: its artifacts are the only
            # account of what went wrong in the tree the next step inherits.
            record_path = f"{records_dir}/{worker.uid}"
            published = await self.ctx.step(publish_link, worker.artifacts_dir, record_path)
            record = record_path if published else None
            failure = None
            if output.failed is True:
                # A failed Action stays failed: its half-finished attempt is
                # never scored, and one failure does not end the search.
                failure = cast(Failure, output).reason
                logger.warning("AutoResearch step %d failed: %s", started, failure)
                continue
            # The score serves the model this step trained, so it runs on the
            # exact cards that step's Action was placed on, read back from the
            # Action rather than asked for again.
            upstream = await self.input.score.measure(
                self.ctx,
                step,
                output_dir=f"{worker.artifacts_dir}/full",
                attempt=worker.files(),
                gpus=worker.record.actions[-1].capability.gpu_indices,
            )
            result = await upstream.result()
            if result.failed is True:
                logger.warning("AutoResearch step %d scored nothing: %s",
                               started, cast(Failure, result).reason)
                continue
            scoring = cast(ScoreOutput, result)
            best = self.selected_output
            if best is None or (
                scoring.score < best.score
                if metric_contract.metric_direction == "min"
                else scoring.score > best.score
            ):
                self.select(SearchOutput(
                    output_dir=scoring.output_dir, score=scoring.score,
                    components=scoring.components,
                ))
        logger.info("AutoResearch launched no further step: %s", refusal)
        return self.selected_output

    def _admission_refusal(self, started: int) -> "str | None":
        """Why no further step may start, or None while one still may: this
        Search's own step ceiling, then the run's admission. No third."""
        cap = self.input.max_steps
        if cap is not None and started >= cap:
            return (f"{started} step(s) ran, which is the max_steps={cap} "
                    "ceiling this search was given")
        return self.cap_stop_reason()


SEARCH_TYPE = AutoResearchSearch
