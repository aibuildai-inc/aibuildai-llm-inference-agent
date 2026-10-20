"""Research First Search: research is the primary activity (issue: Research First).

```text
Research First Planner    -> writes proposal.pdf
    -> Research First Committee -> assigns free project slots
    -> Research First Worker    -> one session: code, run, analyze, report.pdf
    -> framework scoring         -> the existing score program
    -> Research First Reviewer  -> judges the finished project
    -> Research First Board      -> feeds later Planner and Committee calls
```

The Research First Board is a real filesystem directory, one folder per plan
plus the Board-wide Committee announcements under ``committee/``, and it is
the single source of truth for research history: proposals, every Committee
decision (repeated Waitlist included), announcements, worker artifacts,
reports, framework scores, and reviews. The Search keeps its own resolved
capabilities, running child Handles, and loop-local counts; every research
status view is derived from the Board's immutable files.

The loop is event-driven: a Committee round runs only after the Board changed
— one or more cold-start or later Planners published proposals, a Worker's
score landed, or a Reviewer published a review. A Committee call finishing is
never a reason to call the Committee again on the unchanged Board.
Planner completion is separate: success or failure can start replacement
Planner work, but only a published proposal can start a Committee round.

Research First adds no automatic stop from failures, rejections, empty
Committee rounds, or a stalling score. The Committee can deliberately end the
Search through the funded-nothing path when the remaining time cannot fit more
work or every direction is exhausted. Budget exhaustion and the explicitly
configured ``max_projects`` cap remain execution limits.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic import ConfigDict, Field

from engine.base import WorkflowBaseModel
from engine.builtin.aibuildai.capability import score_launch
from engine.builtin.aibuildai.io import SearchOutput, task_data
from engine.builtin.aibuildai.programs.score import ScoreLaunch
from engine.builtin.aibuildai.search import SEARCH_TEMPLATE, AIBuildAISearch
from engine.builtin.research_first.agents.research_first.agent import (
    ResearchFirstCommitteeAgent,
    ResearchFirstPlannerAgent,
    ResearchFirstReviewerAgent,
    ResearchFirstWorkerAgent,
)
from engine.builtin.research_first.agents.research_first.io import (
    ResearchFirstCommitteeInput,
    ResearchFirstCommitteeOutput,
    ResearchFirstDecisionRecord,
    ResearchFirstPlannerInput,
    ResearchFirstProposalOutput,
    ResearchFirstProposalRecord,
    ResearchFirstReviewerInput,
    ResearchFirstReviewerOutput,
    ResearchFirstScoreRecord,
    ResearchFirstWorkerInput,
    ResearchFirstWorkerOutput,
)
from engine.capability import ExecutionCapability
from engine.durable_execution import run_host
from engine.durable_execution import ExecutionContext, Handle
from engine.execution_output import ExecutionOutput
from engine.failure import Failure
from engine.metric_contract import MetricContractInput
from engine.paths import RunPaths
from infra.fs.artifact_dir import publish_link
from aibuildai_exec.files import atomic_write_text

if TYPE_CHECKING:
    from config import AgentConfig
    from engine.work_unit.base import WorkUnit

logger = logging.getLogger(__name__)

BOARD_DIRNAME = "research_first_board"

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
    gpu_count: int = 0,
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
        gpus=gpu_count,
        # File isolation is physical, so it is declared by the unit that does
        # the work, not by a container above it. None where the run did not ask
        # for it, so a unit never argues with an owner that did.
        retry=time_config.retry,
    )



def _plan_ordinal(plan_uid: str) -> int:
    return int(plan_uid.rsplit("_", 1)[-1])


def _publish_proposal(board_dir: str, plan_uid: str, proposal_dir: str) -> bool:
    """Create one plan's Board folder and point it at the Planner's proposal."""
    return publish_link(proposal_dir, f"{board_dir}/{plan_uid}/proposal")


def _read_waitlist(board_dir: str) -> list[str]:
    """Plans whose proposal is published and not yet accepted or rejected."""
    board = Path(board_dir)
    waitlist: list[str] = []
    if not board.is_dir():
        return waitlist
    for plan_dir in board.iterdir():
        if not (plan_dir / "proposal" / "manifest.json").is_file():
            continue
        committee_dir = plan_dir / "committee"
        decided = False
        if committee_dir.is_dir():
            for decision_file in committee_dir.iterdir():
                record = ResearchFirstDecisionRecord.model_validate_json(
                    decision_file.read_text(encoding="utf-8")
                )
                if record.verdict in ("accept", "reject"):
                    decided = True
                    break
        if not decided:
            waitlist.append(plan_dir.name)
    return sorted(waitlist, key=_plan_ordinal)


def _append_decisions(
    board_dir: str,
    decisions: list[dict[str, object]],
    announcement: str | None,
) -> None:
    """Append one immutable decision file per verdict, Waitlist included, and
    the round's announcement when the Committee wrote one.

    A plan receives at most one decision per Committee round, so the round
    number names every file of the round. A retry accepts the same saved
    facts and rejects different ones: a round that saved any decision or a
    notice is committed, and its announcement -- text or null -- is the one
    on the Board."""
    records = [ResearchFirstDecisionRecord.model_validate(p) for p in decisions]
    rounds = {record.committee_round for record in records}
    if len(rounds) != 1:
        raise AssertionError(f"one Committee round expected, got {sorted(rounds)}")
    committee_round = rounds.pop()
    board = Path(board_dir)
    notice_path = board / "committee" / f"announcement_{committee_round:04d}.md"
    saved = notice_path.read_text(encoding="utf-8") if notice_path.is_file() else None
    committed = saved is not None or any(
        (
            board / r.plan_uid / "committee" / f"decision_{committee_round:04d}.json"
        ).is_file()
        for r in records
    )
    if committed:
        if saved != announcement:
            raise AssertionError(
                f"Committee round {committee_round} already saved "
                f"{'an announcement' if saved is not None else 'no announcement'}; "
                f"the retry brought {'a different one' if announcement is not None else 'none'}"
            )
    elif announcement is not None:
        notice_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(notice_path, announcement)
    for payload in decisions:
        record = ResearchFirstDecisionRecord.model_validate(payload)
        committee_dir = Path(board_dir) / record.plan_uid / "committee"
        committee_dir.mkdir(parents=True, exist_ok=True)
        decision_path = committee_dir / f"decision_{record.committee_round:04d}.json"
        if decision_path.is_file():
            saved = ResearchFirstDecisionRecord.model_validate_json(
                decision_path.read_text(encoding="utf-8")
            )
            if saved != record:
                raise AssertionError(
                    f"Committee round {record.committee_round} already saved a "
                    f"different decision for {record.plan_uid}"
                )
            continue
        atomic_write_text(decision_path, record.model_dump_json(indent=2))


def _next_committee_round(board_dir: str) -> int:
    """The next round number, read from the decisions and announcements on the Board."""
    rounds = [
        int(path.stem.rsplit("_", 1)[-1])
        for pattern in ("*/committee/decision_*.json", "committee/announcement_*.md")
        for path in Path(board_dir).glob(pattern)
    ]
    return max(rounds, default=0) + 1


def _read_manifest(board_dir: str, plan_uid: str) -> dict[str, object]:
    """One plan's proposal manifest, as the JSON object the Planner wrote."""
    manifest_path = Path(board_dir) / plan_uid / "proposal" / "manifest.json"
    return cast(
        "dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8"))
    )


def _publish_worker(
    board_dir: str,
    plan_uid: str,
    scratch_dir: str,
    artifacts_dir: str,
    worker_path: str,
    worker_failed: bool,
    scored: bool,
) -> None:
    """Point the Board at one finished Worker's own directories, behind its
    manifest. Only a scored project must publish an attempt; a study submits
    nothing."""
    plan_dir = Path(board_dir) / plan_uid
    worker_dir = plan_dir / "worker"
    manifest_path = worker_dir / "manifest.json"
    if manifest_path.is_file():
        return
    worker_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "source": (scratch_dir, worker_dir / "source"),
        "attempt": (f"{artifacts_dir}/full", worker_dir / "attempt"),
        "report": (f"{artifacts_dir}/report", plan_dir / "report"),
    }
    artifacts = Path(artifacts_dir)
    for name in (
        entry.name
        for entry in (artifacts.iterdir() if artifacts.is_dir() else ())
        if entry.is_dir() and entry.name not in ("full", "report")
    ):
        targets[name] = (str(artifacts / name), worker_dir / name)
    published = {
        name
        for name, (source, target) in targets.items()
        if publish_link(source, str(target))
    }
    if not worker_failed:
        required = {"source", "report"} | ({"attempt"} if scored else set())
        missing = required - published
        if missing:
            raise AssertionError(
                f"successful Worker did not publish: {sorted(missing)}"
            )
    # The manifest is the Board's own record and is written last, so its
    # presence is what says this plan's links are complete. ``worker_path`` is
    # a diagnostic fact for a human reading the Board, never a handle to be
    # revived: the Search that ran this project keeps that live Handle itself.
    atomic_write_text(
        manifest_path,
        json.dumps(
            {
                "plan_uid": plan_uid,
                "worker_path": worker_path,
                "published": sorted(published),
            },
            indent=2,
        ),
    )


def _publish_score(board_dir: str, plan_uid: str, record: dict[str, object]) -> bool:
    """Write one plan's framework score record."""
    score_dir = Path(board_dir) / plan_uid / "score"
    score_dir.mkdir(parents=True, exist_ok=True)
    result_path = score_dir / "result.json"
    if result_path.is_file():
        return True
    atomic_write_text(result_path, json.dumps(record, indent=2, sort_keys=True))
    return True


def _publish_review(board_dir: str, plan_uid: str, review: dict[str, object]) -> bool:
    """Write one plan's structured review."""
    review_dir = Path(board_dir) / plan_uid / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_path = review_dir / "review.json"
    if review_path.is_file():
        return True
    atomic_write_text(review_path, json.dumps(review, indent=2, sort_keys=True))
    return True


class ResearchFirstSearchParameters(WorkflowBaseModel):
    """Research First Search's own business parameters: the half a user writes.

    Local to this kind: another Search may spell a field the same way with its
    own meaning, or not declare it at all."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parallel: int = Field(
        gt=0,
        description="maximum number of accepted Research First projects "
        "executing concurrently.",
        json_schema_extra={"placeholder": 4},
    )
    waitlist_size: int | None = Field(
        default=None,
        ge=1,
        description="target size of the waitlist: the Planner supply "
        "tops the undecided proposals back up to this size and does nothing "
        "else. null = same as parallel.",
    )
    max_projects: int | None = Field(
        default=None,
        gt=0,
        description="maximum number of accepted projects (failures occupy a "
        "slot); Planner, Committee, and Reviewer calls do not count. null = "
        "no count cap beyond the run-level wall-clock + cost budget. This is "
        "an execution limit, not an early-stopping policy.",
    )


class ResearchFirstSearchInput(ResearchFirstSearchParameters):
    """Research First Search's complete immutable Input.

    The business parameters above are the user's. Everything below is
    resolved once by startup, so nothing this Search creates ever reads
    configuration or asks for a capability by role name."""

    task_name: str
    progress_enabled: bool
    planner: ExecutionCapability
    committee: ExecutionCapability
    worker: ExecutionCapability
    reviewer: ExecutionCapability
    score: ScoreLaunch


@dataclass
class _Worker:
    """One accepted Research First project this Search itself launched.

    The Search's own bookkeeping of what it started: ``output`` settles once
    the Worker finishes."""

    plan_uid: str
    agent: ResearchFirstWorkerAgent
    output: "ResearchFirstWorkerOutput | Failure | None" = None


class ResearchFirstSearch(AIBuildAISearch[ResearchFirstSearchInput]):
    #: Text shown for this method in the starter config.
    description = (
        "Research First Search: a Planner writes research proposals, a "
        "Committee assigns the free project slots, one Worker session "
        "conducts each accepted project and writes its report, and a "
        "Reviewer judges the finished project. The accumulated Research "
        "First Board — a filesystem folder per plan plus the Committee's "
        "announcements — feeds every later Planner and Committee call."
    )
    __doc__ = description

    prompt_template = SEARCH_TEMPLATE
    agent_types = (
        ResearchFirstPlannerAgent,
        ResearchFirstCommitteeAgent,
        ResearchFirstWorkerAgent,
        ResearchFirstReviewerAgent,
    )
    unit_types = ()
    parameters_type = ResearchFirstSearchParameters

    @classmethod
    def build_input(
        cls, configs: "AgentConfig", run_paths: RunPaths
    ) -> ResearchFirstSearchInput:
        """Validate the user's ``search.input`` and complete this package's Input."""
        return ResearchFirstSearchInput(
            **cls.parameters(configs),
            task_name=configs.run.task_name,
            progress_enabled=configs.progress.enable,
            planner=_capability(configs, ResearchFirstPlannerAgent),
            committee=_capability(configs, ResearchFirstCommitteeAgent),
            worker=_capability(configs, ResearchFirstWorkerAgent),
            reviewer=_capability(configs, ResearchFirstReviewerAgent),
            score=score_launch(configs, run_paths),
        )

    async def committee_free_slots(self, *, on: "ExecutionContext | None" = None) -> int:
        """Free project slots right now, counted from the projects this Search
        started.

        Its own Committee asks this from the COMMITTEE's Action, which is a
        different workflow, and every workflow restores its own execution
        objects: a dict this Search filled in ``explore`` is simply absent on
        the object that workflow holds. So the count comes from the work this
        Search delegated, which every workflow reads the same, and ``on``
        names the Action the observation is journaled under -- the Committee's
        when the Committee asks, this Search's own while ``explore`` asks."""
        started = await self.delegated(self, ResearchFirstWorkerAgent, on=on)
        accepted = {project.execution.uid for project in started}
        running = {
            project.execution.uid for project in started if project.output is None
        }
        free = self.input.parallel - len(running)
        if self.input.max_projects is not None:
            free = min(free, self.input.max_projects - len(accepted))
        return max(0, free)

    async def explore(self, metric_contract: MetricContractInput) -> "SearchOutput | Failure | None":
        self._workers: dict[str, _Worker] = {}
        parallel = self.input.parallel
        waitlist_target = (
            self.input.waitlist_size
            if self.input.waitlist_size is not None
            else parallel
        )
        max_projects = self.input.max_projects
        # The one Research First resource: the shared Board this Search owns
        # and writes, and every role reads. It is an index as much as a
        # directory -- each plan folder points at the proposal, attempt and
        # report their producers wrote -- so reading the Board means reading
        # what it names. That rule is the Board's, stated here by its owner.
        board_files = self.files(BOARD_DIRNAME, links=True)
        board = board_files.path()
        # Operational runtime state only: the identities and Handles of
        # running children and counters this process owns. Research history
        # and plan status come from the Board, read through checkpointed
        # steps so a replay sees the same Board the original pass saw.
        planner_pending: dict[_AnyHandle, ResearchFirstPlannerAgent] = {}
        planner_handles: dict[str, _AnyHandle] = {}
        worker_pending: dict[_AnyHandle, str] = {}
        reviewer_pending: dict[_AnyHandle, str] = {}
        stop_reason: str | None = None
        board_changes: list[_AnyHandle] = []
        committee_rounds: list[_AnyHandle] = []
        # One Committee identity answers every round; each round is one call.
        committee = ResearchFirstCommitteeAgent(
            input=ResearchFirstCommitteeInput(
                board_dir=board,
                metric_contract=metric_contract,
            ),
        )

        def cap_left() -> int | None:
            if max_projects is None:
                return None
            return max_projects - len(self._workers)

        async def spawn_planner() -> None:
            planner = ResearchFirstPlannerAgent(
                input=ResearchFirstPlannerInput(
                    board_dir=board,
                    metric_contract=metric_contract,
                    remaining_minutes=(
                        await self.budget.snapshot()
                    ).wall_clock_remaining_s
                    / 60.0,
                    gpu_count=run_host().host_visible_gpu_count(),
                ),
            )
            handle = await self.ctx.spawn(
                planner.run,
                "Write one research proposal for the Board.",
                upstream=(self.setup_handle,),
                read=(task_data(), board_files),
                capture_failure=True,
                capability=self.input.planner,
            )
            planner_pending[handle] = planner
            planner_handles[planner.uid] = handle

        async def refill_waitlist(undecided: int) -> None:
            """Top the waitlist back up to its target size — the one
            supply rule. Skipped once the explicit project cap is
            exhausted — more proposals could never launch."""
            if cap_left() == 0:
                return
            needed = waitlist_target - (len(planner_pending) + undecided)
            for _ in range(needed):
                await spawn_planner()

        async def publish_planner(handle: _AnyHandle) -> bool:
            planner = planner_pending.pop(handle)
            output = cast(
                "ResearchFirstProposalOutput | Failure", await handle.result()
            )
            if output.failed is True:
                logger.warning(
                    "Research First planner %s failed: %s",
                    planner.uid,
                    output.reason,
                )
                return False
            published = bool(
                await self.ctx.step(
                    _publish_proposal,
                    board,
                    planner.uid,
                    f"{planner.artifacts_dir}/proposal",
                )
            )
            if not published:
                raise AssertionError(
                    f"successful Planner {planner.uid} has no proposal"
                )
            board_changes.append(handle)
            return True

        async def launch_worker(plan_uid: str) -> None:
            nonlocal stop_reason
            manifest = ResearchFirstProposalRecord.model_validate(
                await self.ctx.step(_read_manifest, board, plan_uid)
            )
            gpus = manifest.requested_gpus
            worker = ResearchFirstWorkerAgent(
                input=ResearchFirstWorkerInput(
                    plan_dir=f"{board}/{plan_uid}",
                    task_name=self.input.task_name,
                    gpu_count=gpus,
                    scored=manifest.scored,
                    progress_enabled=self.input.progress_enabled,
                ),
            )
            # A plan folder is named by the Planner that published it, and the
            # round that accepted it is the Committee's latest call.
            handle = await self.ctx.spawn(
                worker.run,
                "Conduct the accepted project and write its report.",
                upstream=(planner_handles[plan_uid], committee_rounds[-1]),
                read=(task_data(), board_files),
                capture_failure=True,
                capability=self.input.worker.model_copy(update={"gpus": gpus}),
            )
            worker_pending[handle] = plan_uid
            self._workers[plan_uid] = _Worker(plan_uid=plan_uid, agent=worker)
            if cap_left() == 0:
                stop_reason = f"max_projects={max_projects} accepted plans"

        async def fill_free_slots() -> None:
            free = await self.committee_free_slots()
            waitlist = list(await self.ctx.step(_read_waitlist, board))
            if free <= 0:
                await refill_waitlist(len(waitlist))
                return
            if not waitlist:
                await refill_waitlist(0)
                return
            committee_round = int(await self.ctx.step(_next_committee_round, board))
            running = list(worker_pending.values())
            message = (
                "The Board changed. Read the current Board and allocate the free slots. "
                f"Free slots: {free}. Waitlist: {', '.join(waitlist)}. "
                f"Running: {', '.join(running) or 'none'}. "
                f"Remaining time: "
                f"{(await self.budget.snapshot()).wall_clock_remaining_s / 60:.0f} minutes."
            )
            round_handle = await self.ctx.spawn(
                committee.run,
                message,
                upstream=tuple(board_changes),
                read=(task_data(), board_files),
                capture_failure=True,
                capability=self.input.committee,
            )
            committee_rounds.append(round_handle)
            committee_output = cast(
                "ResearchFirstCommitteeOutput | Failure", await round_handle.result()
            )
            board_changes.clear()
            if committee_output.failed is True:
                # A failed round is a failed child execution, nothing more:
                # commission new alternatives so their completion is the next
                # trigger; the search never stops because rounds failed.
                logger.warning(
                    "Research First committee round %d failed: %s",
                    committee_round,
                    committee_output.reason,
                )
                await refill_waitlist(len(waitlist))
                return
            await self.ctx.step(
                _append_decisions,
                board,
                [
                    ResearchFirstDecisionRecord(
                        plan_uid=decision.plan_uid,
                        committee_round=committee_round,
                        verdict=decision.verdict,
                        reason=decision.reason,
                    ).model_dump(mode="json")
                    for decision in committee_output.decisions
                ],
                committee_output.announcement,
            )
            tally = {"accept": 0, "waitlist": 0, "reject": 0}
            for decision in committee_output.decisions:
                tally[decision.verdict] += 1
                if decision.verdict == "accept":
                    await launch_worker(decision.plan_uid)
            board_changes.append(round_handle)
            # Only a rejection frees a waitlist place: the supply refills the
            # waitlist to its target, so fresh alternatives arrive exactly when
            # the Committee rejects, and an all-Waitlist round goes quiet.
            await refill_waitlist(tally["waitlist"])
            logger.info(
                "Research First committee round %d: %d accepted, %d waitlisted, "
                "%d rejected; waitlist holds %d, %d project slot(s) free",
                committee_round,
                tally["accept"],
                tally["waitlist"],
                tally["reject"],
                tally["waitlist"],
                parallel - len(worker_pending),
            )

        async def finish_worker(handle: _AnyHandle) -> bool:
            # The project keeps its slot until its score is in: the score is
            # awaited here, so no Committee round runs before this returns.
            plan_uid = worker_pending.pop(handle)
            started = self._workers[plan_uid]
            worker = started.agent
            scored = ResearchFirstProposalRecord.model_validate(
                await self.ctx.step(_read_manifest, board, plan_uid)
            ).scored
            worker_output = cast(
                "ResearchFirstWorkerOutput | Failure", await handle.result()
            )
            started.output = worker_output
            await self.ctx.step(
                _publish_worker,
                board,
                plan_uid,
                worker.scratch_dir,
                worker.artifacts_dir,
                worker.path,
                worker_output.failed is True,
                scored,
            )
            attempt_dir = f"{board}/{plan_uid}/worker/attempt"
            score_handle: _AnyHandle | None = None
            published = True
            # A study (proposal scored false) runs no score program and gets
            # no score record; readers learn that from the proposal. A failed
            # Worker is not scored either: a failed Action stays failed, and
            # what it left behind is preserved for inspection only.
            if scored and worker_output.failed is not True:
                # The score serves the checkpoint the Worker trained, so it
                # runs on the exact cards that Worker Action was placed on.
                # They are read back from the Action rather than re-asked for,
                # so the checkpoint and its measurement meet on one device.
                score_handle = await self.input.score.measure(
                    self.ctx,
                    handle,
                    output_dir=attempt_dir,
                    attempt=board_files,
                    gpus=worker.record.actions[-1].capability.gpu_indices,
                )
                score_output = await score_handle.result()
                if score_output.failed is True:
                    scoring = None
                    score_failure = score_output
                else:
                    scoring = score_output
                    score_failure = None
                    best = self.selected_output
                    if best is None or (
                        scoring.score < best.score
                        if metric_contract.metric_direction == "min"
                        else scoring.score > best.score
                    ):
                        self.select(SearchOutput(
                            output_dir=attempt_dir, score=scoring.score,
                            components=scoring.components,
                        ))
                record = ResearchFirstScoreRecord(
                    plan_uid=plan_uid,
                    scoring=scoring,
                    score_failure=score_failure,
                    output_dir=attempt_dir,
                )
                published = bool(
                    await self.ctx.step(
                        _publish_score,
                        board,
                        plan_uid,
                        record.model_dump(mode="json"),
                    )
                )
            reviewer_pending[
                await self.ctx.spawn(
                    ResearchFirstReviewerAgent(
                        input=ResearchFirstReviewerInput(
                            plan_dir=f"{board}/{plan_uid}",
                            scored=scored,
                            metric_contract=metric_contract,
                        ),
                    ).run,
                    "Judge the finished project on the Board.",
                    upstream=(
                        (handle,) if score_handle is None else (handle, score_handle)
                    ),
                    read=(task_data(), board_files),
                    capture_failure=True,
                    capability=self.input.reviewer,
                )
            ] = plan_uid
            board_changes.append(handle)
            if score_handle is not None:
                board_changes.append(score_handle)
            return published

        async def finish_reviewer(handle: _AnyHandle) -> bool:
            plan_uid = reviewer_pending.pop(handle)
            review_output = cast(
                "ResearchFirstReviewerOutput | Failure", await handle.result()
            )
            if review_output.failed is True:
                logger.warning(
                    "Research First reviewer for %s failed: %s",
                    plan_uid,
                    review_output.reason,
                )
                return False
            published = bool(
                await self.ctx.step(
                    _publish_review,
                    board,
                    plan_uid,
                    review_output.model_dump(mode="json"),
                )
            )
            if published:
                board_changes.append(handle)
            return published

        # Cold start: the initial Planner batch, settled and published as one
        # event before the first Committee round sees any waitlisted plan. The one
        # supply rule applies here too: fill the waitlist to its target.
        if cap_left() != 0:
            for _ in range(waitlist_target):
                await spawn_planner()
        cold_start_publications = 0
        while planner_pending:
            settled = await self.ctx.wait(set(planner_pending), min_completed=1)
            for handle in settled.completed:
                cold_start_publications += await publish_planner(handle)
            await refill_waitlist(len(await self.ctx.step(_read_waitlist, board)))
        if cold_start_publications:
            await fill_free_slots()

        while planner_pending or worker_pending or reviewer_pending:
            pending: set[_AnyHandle] = (
                set(planner_pending) | set(worker_pending) | set(reviewer_pending)
            )
            done = await self.ctx.wait(pending, min_completed=1)
            publication_count = 0
            planner_completed = False
            for handle in done.completed:
                if handle in planner_pending:
                    planner_completed = True
                    publication_count += await publish_planner(handle)
                elif handle in worker_pending:
                    publication_count += await finish_worker(handle)
                else:
                    publication_count += await finish_reviewer(handle)
            if planner_completed:
                await refill_waitlist(len(await self.ctx.step(_read_waitlist, board)))
            if publication_count:
                await fill_free_slots()
        if stop_reason is None:
            leftover = list(await self.ctx.step(_read_waitlist, board))
            if leftover:
                stop_reason = (
                    "the Committee funded nothing and nothing is running; "
                    f"{len(leftover)} proposal(s) stay waitlisted"
                )
                logger.info("Research First stopped: %s", stop_reason)

        return self.selected_output


SEARCH_TYPE = ResearchFirstSearch
