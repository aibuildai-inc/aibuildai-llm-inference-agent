"""CLI entry point for the RunState beam-search ML workflow.

Argument parsing and YAML loading live in :mod:`startup.config_loader`. This module is the composition root: a thin shell that wires the concrete services to engine, wraps the whole process in :class:`infra.process.process_lifecycle.ProcessLifecycle` (re-exec into a systemd service and process cleanup), and dispatches the parsed command to a :class:`Command`. Commands return a :class:`CommandOutcome`; the root finalizes it (rendering any summary) after the ``ProcessLifecycle`` block exits.
"""

import abc
import asyncio
import json
import logging
import os
import signal
import sys
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from pydantic import ValidationError

import features
from config import (
    StartupUserInputError,
    WorkUnitTimeConfigError,
)
from engine.durable_execution import latest_run_state, run_store
from engine.event.events import RunEpochOpened, RunOpened
# RunPinMismatch lives in engine.event.store (not bootstrap, which is
# import-heavy) precisely so it is cheap to import at module top here (#1269).
from engine.event.store import (
    SCHEMA_VERSION,
    RunPinMismatch,
    open_search_store,
    read_events,
)
from engine.failure import Failure
from engine.mcp import (
    McpStartupError,
    RunMcpClient,
    resolve_mcp_definitions,
)
from engine.paths import (
    RunPaths,
    RunTimestamp,
)
from engine.run_state import RunState
from engine.run_query import read_run_head
from aibuildai_conversation.environment import scrub_claude_feature_env
from infra.process.process_lifecycle import ProcessLifecycle
from infra.postgresql import PostgreSQLUnavailable
from infra.process.transient_unit import unit_active
from infra.host_resource.cgroup import (
    HostCapabilityUnavailable,
    kubernetes_start_mode,
    run_unit_name,
)
from infra.kb_service import KbRequestPayload
from memory.contribute import (
    contribution_text,
    upload_run,
)
from output.report import RunReport
from startup import config_loader, run_config_store, run_launch
from startup.config_loader import format_validation_error, load_config
from startup.example_yaml import example_yaml_text
from startup.logging_config import configure_logging
from startup.model_endpoint import model_endpoint
from startup.resource_materializer import materialize_to_tmpfs
from aibuildai_version import APP_VERSION, is_frozen

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from config import AgentConfig
    from engine.event.store import SearchApplication
    from engine.builtin.aibuildai.search import AIBuildAISearch
    from engine.work_unit.agent.policy import ModelRouter
    from aibuildai_conversation.model_endpoint import OpenAIProxy


logger = logging.getLogger(__name__)


def _check_task_folder_exists(config: "AgentConfig") -> None:
    """Reject configs whose ``run.data_root`` is not a real directory.

    ``run.data_root`` IS the task folder now: the run reads it whole and expects no layout inside it, so the only thing to check here is that the folder is there. Without this sub-second filesystem check, a typo in the path burns the whole SETUP budget on a folder that does not exist (``conda create``, ``pip install torch``, a large download) and then crashes with a generic ``RuntimeError: Setup failed: failed(timeout)`` that never names the underlying cause. We catch it here before any LLM call, so the user learns of the typo for free rather than for a paid 2-minute SETUP timeout.
    """
    if os.path.isdir(config.task_folder):
        return

    raise ValueError(
        "\n".join(
            [
                # No `config error:` prefix here: the cli() boundary already prepends a
                # single `error: ` when it renders this message, so embedding another
                # prefix produced a doubled `error: config error: ...`.
                f"no task folder found at {config.task_folder!r}",
                "  run.data_root must name the task folder itself: the directory "
                "holding what the task means",
                "  plus every material it needs. The run reads that folder whole and "
                "expects no layout inside it.",
                f"  run.data_root:   {config.run.data_root}",
            ]
        )
    )


def run_memorize_command(config: "AgentConfig", model_env: dict[str, str]) -> None:
    """Composition root for the memorize command.

    Resolves the memory store here. Does NOT touch resource limits, dataset checks, startup checks, router, resume, run-dir-lock, or the loop — those are run-command concerns only.
    """
    from memory.memorize import run_memorize
    from memory.store import MemoryStore

    # The runs to summarize live under run.playground_root, which every config
    # already declares as this run's output root. No second field names it.
    store = MemoryStore.user(config.memory.user_dir)
    result = asyncio.run(
        run_memorize(
            task_name=config.run.task_name,
            playground_root=config.run.playground_root,
            store=store,
            model=config.llm.default.model,
            model_env=model_env,
        )
    )
    print(
        f"[memorize] read {result['runs_read']} run(s) for "
        f"'{config.run.task_name}' -> {result['task_memory_path']}"
    )


def prepare_run(
    config: "AgentConfig",
    *,
    resume_run_timestamp: "str | None" = None,
) -> tuple[
    "type[AIBuildAISearch]",
    RunPaths,
    "ModelRouter | None",
    "RunMcpClient",
]:
    """Run-command startup sequence: everything between config-load and the loop.

    Validates required fields + the dataset dir, runs fail-closed startup checks, builds the Router policy, resolves MCP and plugin inputs, creates the run home, and installs logging. Returns the checked AIBuildAISearch class and the live owners it will receive.
    """
    # Required fields are enforced at AgentConfig() construction (run.* /
    # llm.default.model have no defaults). This loop guards against an
    # explicitly-empty value.
    if not config.run.task_name:
        raise ValueError("YAML missing required field: run.task_name")
    if not config.run.data_root:
        raise ValueError("YAML missing required field: run.data_root")
    if not config.run.playground_root:
        raise ValueError("YAML missing required field: run.playground_root")
    if not config.llm.default.model:
        raise ValueError("YAML missing required field: llm.default.model")

    _check_task_folder_exists(config)

    # data_root and playground_root are canonicalized by
    # RunConfig._normalize_root_path at the pydantic boundary.

    from infra.fs.pathing import _assert_posix_path_semantics
    from aibuildai_exec.sandbox import probe_unprivileged_userns
    from infra.host_resource.gpu_alloc import configure_visible_devices

    _assert_posix_path_semantics(
        Path(config.output_base_dir),
        purpose="aibuildai run outputs",
    )
    configure_visible_devices(
        config.resources.cuda_visible_devices,
    )
    if config.resources.sandbox.enable:
        # The host prerequisites of bwrap, and nothing else: whether the
        # launcher the product generates is well-formed is a property of the
        # source, proved once per change, not a check every run repeats.
        probe_unprivileged_userns()

    from engine.builtin import search_type_for_kind
    from engine.work_unit.agent import prompt
    from engine.work_unit.agent.plugins import (
        resolve_enable_config,
        validate_agent_roles,
        validate_plugin_config,
    )
    from startup.extensions import import_extension_classes

    import_extension_classes(config.extensions.dir)
    if "extensions" not in features.hidden_config_fields():
        try:
            prompt.register_extension_templates(
                str(Path(config.extensions.dir) / "templates")
            )
        except Exception as exc:  # noqa: BLE001 -- user templates execute here
            raise StartupUserInputError(
                f"invalid extension prompt template: {type(exc).__name__}: {exc}"
            ) from exc
    search_method_type = search_type_for_kind(config.search.kind)
    active_agent_types = search_method_type.active_agent_types(config)
    try:
        prompt.search_template_for(search_method_type)
        for agent_type in active_agent_types.values():
            agent_template = agent_type.prompt_template
            assert agent_template is not None
            prompt.agent_template_for(agent_type)
    except Exception as exc:  # noqa: BLE001 -- user templates execute here
        raise StartupUserInputError(
            f"invalid selected prompt template: {type(exc).__name__}: {exc}"
        ) from exc
    known_roles = tuple(sorted(active_agent_types))
    validate_agent_roles(config, known_roles)
    search_method_type.validate_config(config)

    enable = resolve_enable_config(config, active_agent_types)
    # Routing is one package's own concrete role. A package that declares no
    # Router answers None, and a configuration that asks for routing under it
    # is refused here instead of being quietly ignored.
    router = search_method_type.prepare_router(config, enable=enable)
    if config.llm.router_on and router is None:
        raise StartupUserInputError(
            f"llm.auto asks this run to route models, but search.kind="
            f"{config.search.kind!r} declares no Router. Remove llm.auto or "
            "select a search kind that has one."
        )
    mcp_definitions = resolve_mcp_definitions(config, known_roles)
    validate_plugin_config(
        config,
        enable,
        mcp_definitions.keys(),
        known_roles,
    )

    # Resolve the run directory only after startup has checked the complete
    # selected composition. Invalid configuration must not create a run home.
    # The Run ID itself was allocated at the outer launch boundary (cli()),
    # before this process re-execed into its own systemd unit: the directory
    # suffix, the archived config, the root DBOS workflow, and the unit all
    # carry that one identity. The unit name is the single-writer boundary;
    # there is no directory lock.
    if resume_run_timestamp is None:
        # Consume the marker: it names exactly one launch, and a child
        # process or a later fresh run must never inherit this identity.
        run_id = os.environ.pop("AIBUILDAI_RUN_ID", None)
        if not run_id:
            raise AssertionError("a fresh run has no allocated Run ID")
        timestamp = RunTimestamp.new(run_id)
    else:
        timestamp = RunTimestamp.parse(resume_run_timestamp)
    run_paths = RunPaths.from_output_base(
        config.output_base_dir,
        timestamp,
    )

    # The manager-owned siblings of the run workspace, plus the
    # candidate-visible public dir. All are bound writable into a role's sandbox
    # (setup fills public/ and private/, the finalizer writes the deliverable),
    # and a bwrap write bind needs its dir to already exist, so the run home
    # creates them here with the rest of its own layout.
    Path(run_paths.public_dir).mkdir(parents=True, exist_ok=True)
    Path(run_paths.private_dir).mkdir(parents=True, exist_ok=True)
    Path(run_paths.deliverable_dir).mkdir(parents=True, exist_ok=True)

    # The final config owns the fixed global identity. User YAML cannot set it.
    if resume_run_timestamp is None:
        config.run = type(config.run).model_validate(
            {
                **config.run.model_dump(),
                "run_id": timestamp.run_id,
                "kubernetes_start_mode": kubernetes_start_mode(),
            }
        )
    elif config.run.require_run_id() != timestamp.run_id:
        raise StartupUserInputError(
            f"run {config.run.require_run_id()} predates the one-identity "
            "run directory shape and cannot be resumed by this build; open "
            "it in the local Web Workspace"
        )


    # Archive the resolved config before birth. The bootstrap publishes the Run
    # to the XDG index only after RunOpened commits. A failed pre-birth launch is
    # therefore not visible to Workspace.
    if resume_run_timestamp is None:
        run_config_store.save_run_config(config, str(run_paths.run_home))

    # Stage 2: the run's own stderr handler replaces the early bootstrap one.
    configure_logging()

    return (
        search_method_type,
        run_paths,
        router,
        RunMcpClient(
            mcp_definitions,
            request_payload=_kb_run_payload(config),
        ),
    )


# The blame-neutral exit-summary headline for an unexpected internal bug.
_UNEXPECTED_BUG_MESSAGE = (
    "an unexpected error occurred (this is a bug in aibuildai, not your code)."
)

# Issue #1069 boundary case 1: a crash in composition-root setup. The one-liner
# carries no file pointer.
_PRE_RUN_CRASH_MESSAGE = (
    "aibuildai: an unexpected error occurred before the run could start — "
    "this is a bug in aibuildai, not your code or data. Please file an issue "
    "at https://github.com/aibuildai/AI-Build-AI/issues"
)

# Issue #2829: the same shape for an auth command, which runs before any run
# exists and so cannot point at a run home.
_AUTH_CRASH_MESSAGE = (
    "aibuildai: an unexpected error occurred during the auth command — "
    "this is a bug in aibuildai, not your account. Please file an issue "
    "at https://github.com/aibuildai/AI-Build-AI/issues"
)

# The exit codes every boundary in this module answers with, and the one
# question that picks between them: whose move is it next?
#
#   0    the command did what was asked.
#   1    the work ran and did not complete -- a run that failed on its own terms.
#   2    the USER can fix it: a bad config value, an unusable resume, a sign-in
#        that needs starting again. Printed as `error: <sentence>`, never a
#        traceback, because the sentence IS the fix.
#   70   AIBUILDAI is broken: nothing the user did explains it. Printed as the
#        blame-neutral one-liner, with the raw traceback only on a non-frozen
#        developer run.
#   130  the user pressed Ctrl-C.
#
# The distinction that matters is 2 against 70, and it is about the READER, not
# about severity: 70 tells a person to file an issue and wait, so spending it on
# something they could have fixed in one command is a real cost to them.
#
# BSD sysexits.h EX_SOFTWARE: "an internal software error has been detected".
# The dedicated exit code for "the tool itself broke", distinct from the
# expected-non-completion code (1) — issue #1069.
_EX_SOFTWARE = 70


class CommandOutcome(abc.ABC):
    """A command's result, finalized by the composition root after ProcessLifecycle teardown. It renders any user-facing summary and returns the process exit code."""

    @abc.abstractmethod
    def finalize(self) -> int: ...


class RunOutcome(CommandOutcome):
    """run command's result: holds the data RunReport needs; renders in finalize()."""

    def __init__(
        self,
        config: "AgentConfig",
        run_paths: RunPaths,
        run_state: "RunState | None",
        *,
        status: str,
        error: "str | run_launch.WebSafeDiagnostic | None" = None,
    ):
        self.config = config
        self.run_paths = run_paths
        self.run_state = run_state
        self.status = status
        self.error = error

    def finalize(self) -> int:
        return RunReport(
            self.config,
            self.run_paths,
            self.run_state,
            status=self.status,
            error=None if self.error is None else str(self.error),
        ).write()


class ExitCodeOutcome(CommandOutcome):
    """No-render result carrying only an exit code (memorize command)."""

    def __init__(self, code: int):
        self.code = code

    def finalize(self) -> int:
        return self.code


class Command(abc.ABC):
    """A CLI command. Receives the loaded config and returns a CommandOutcome; the composition root finalizes it into the process exit code after lifecycle teardown. Commands perform no process-level effects and never call sys.exit."""

    @abc.abstractmethod
    def execute(
        self,
        config: "AgentConfig",
        *,
        openai_proxy: "OpenAIProxy | None" = None,
        model_env: dict[str, str],
    ) -> CommandOutcome: ...


@asynccontextmanager
async def _interrupt_signals(
    request: "Callable[[str], None]",
) -> "AsyncIterator[None]":
    """Map SIGINT and SIGTERM onto one interrupt request callback.

    asyncio's ``loop.add_signal_handler`` delivers the callback at a safe loop-iteration boundary (never mid-bytecode, unlike a synchronous ``signal.signal`` handler that can fire between any two ops, including inside asyncio internals). Both signals converge on the same request path (REQUEST); the caller stays alive to JOIN the root workflow, and the durable runtime CLOSES only after that join. The handlers are removed in finally so they never leak past this scope.
    """
    running = asyncio.get_running_loop()
    running.add_signal_handler(signal.SIGINT, request, "ctrl-c")
    running.add_signal_handler(signal.SIGTERM, request, "sigterm")
    try:
        yield
    finally:
        running.remove_signal_handler(signal.SIGINT)
        running.remove_signal_handler(signal.SIGTERM)


class RunCommand(Command):
    def execute(
        self,
        config: "AgentConfig",
        *,
        resume_run_timestamp: "str | None" = None,
        launch_environment_file: "Path | None" = None,
        openai_proxy: "OpenAIProxy | None" = None,
        model_env: dict[str, str],
    ) -> CommandOutcome:
        (
            search_method_type,
            run_paths,
            router,
            mcp_client,
        ) = prepare_run(
            config,
            resume_run_timestamp=resume_run_timestamp,
        )
        if openai_proxy is not None:
            openai_proxy.set_run_home(run_paths.run_home)

        from engine.status import TerminalStatus

        search: "AIBuildAISearch | None" = None
        # What assembly built for this run. The CLI owns the run, so it keeps
        # the journal state and the event store it needs at teardown rather
        # than reaching into the Search for them.
        run_state: "RunState | None" = None
        store: "SearchApplication | None" = None
        signal_reason: str | None = None
        try:
            from bootstrap import build_search_for_production
            from engine.durable_execution import run_host

            async def build_and_run() -> None:
                nonlocal search, run_state, store, signal_reason
                supervisor = asyncio.current_task()
                assert supervisor is not None

                def request(reason: str) -> None:
                    # Cooperative suspend is for ONE case: a root workflow task
                    # is still running and nothing has been asked yet. Every
                    # other interrupt forces teardown by cancelling the joining
                    # supervisor -- a repeat of any earlier interrupt, a signal
                    # before the root workflow starts, and a signal after the
                    # root has returned, where only the epoch JOIN is left and
                    # a suspend request could never be recorded. The saved
                    # suspend request covers a Ctrl-C key that suspended
                    # through the display, so its next press escalates exactly
                    # like a signal does.
                    nonlocal signal_reason
                    repeated = signal_reason is not None
                    if signal_reason is None:
                        signal_reason = reason
                    pipeline = None if search is None else search._pipeline_task
                    if (
                        repeated
                        or search is None
                        or pipeline is None
                        or pipeline.done()
                        or run_state is None
                        or run_state.aibuildai_search.run_suspend_requested
                        is not None
                    ):
                        logger.warning("forced teardown: %s", reason)
                        supervisor.cancel()
                        return
                    search.suspend(reason)

                async with _interrupt_signals(request):
                    search = await build_search_for_production(
                        config,
                        search_method_type,
                        run_paths,
                        router,
                        mcp_client,
                        launch_environment_file=launch_environment_file,
                        model_env=model_env,
                    )
                    # run_search closes the durable runtime, which unbinds the
                    # run services, so the host is resolved before it starts.
                    host = run_host()
                    try:
                        run_state, store = latest_run_state(), run_store()
                        if (
                            resume_run_timestamp is not None
                            and search._run_output() is not None
                        ):
                            # The launch validates: a run with its final result
                            # has nothing left to resume.
                            raise StartupUserInputError(
                                f"run {config.run.require_run_id()} has finished; "
                                "open it in the local Web Workspace"
                            )
                        from engine.search.base import run_search

                        await run_search(search)
                    finally:
                        await host.release()

            asyncio.run(build_and_run())
            assert search is not None
            terminal = run_state.aibuildai_search.terminal_status()
        except (asyncio.CancelledError, KeyboardInterrupt):
            # Every interrupt lands here after the JOIN. The saved suspend
            # request names the reason whenever the run reached the point of
            # recording one; before that, the signal that actually arrived
            # does, so a SIGTERM during startup is never reported as a
            # Ctrl-C. Process cleanup is owned by ProcessLifecycle.
            request = (
                run_state.aibuildai_search.run_suspend_requested
                if search is not None
                else None
            )
            failed: "tuple[str, Failure] | None" = (
                run_state.latest_recoverable_failure()
                if search is not None
                else None
            )
            if request is None and signal_reason is None and failed is not None:
                # A recoverable Failure ended this epoch: the run stays
                # unfinished and a later Web Resume continues it in place.
                path, failure = failed
                return RunOutcome(
                    config,
                    run_paths,
                    run_state,
                    status="incomplete",
                    error=(
                        f"{path} failed recoverably ({failure.reason}); "
                        "the run is unfinished and can be resumed from the "
                        "Web Workspace"
                    ),
                )
            reason = (signal_reason or "ctrl-c") if request is None else request.reason
            return RunOutcome(
                config,
                run_paths,
                run_state,
                status="sigterm" if reason == "sigterm" else "interrupted",
                # A pause asked for from the Web Workspace says so in the run's
                # end state, so the page and the terminal agree on why it stopped.
                error=(
                    "paused from the Web Workspace"
                    if reason == "web"
                    else None
                    if reason in ("ctrl-c", "sigterm")
                    else reason
                ),
            )
        except HostCapabilityUnavailable:
            # A capability this run REQUIRES that the host does not provide, decided
            # by root outside this process: a USER-ENVIRONMENT condition, not an
            # internal bug. Re-raise so ProcessLifecycle.run renders the plain,
            # actionable message + EX_UNAVAILABLE, instead of the generic except
            # below swallowing it into a status="crashed" "bug in aibuildai" banner.
            #
            # The member that actually arrives here is CgroupWriteRejected from the
            # resource-tree bootstrap (#1824). The family's other members --
            # WorkerCgroupUnavailable (#1137) and ProcessNotKillable (#1298) -- are
            # raised inside ProcessLifecycle.__enter__, so they never enter this try
            # and reach the same renderer by the shorter road. The clause names the
            # family rather than one member because the rule is the family's.
            raise
        except PostgreSQLUnavailable as error:
            logger.error("PostgreSQL unavailable: %s", error)  # noqa: TRY400 — expected refusal, not an unexpected traceback.
            raise
        except RunPinMismatch:
            # A resume whose journaled environment pins (app_version,
            # gpu_visibility, feature_set_hash, cwd, ...) no longer match the
            # CURRENT process (#1269) is a user-caused refusal, not an internal
            # bug: re-raise so the cli() boundary renders the pin-mismatch
            # message naming every changed pin, instead of the generic except
            # below folding it into a status="crashed" "bug in aibuildai" banner.
            raise
        except WorkUnitTimeConfigError:
            # A missing exact WorkUnit time entry is a user configuration error,
            # including when Search construction reads one before Search.run can
            # persist a terminal. Let the CLI's ValueError boundary print the
            # exact identity and exit 2 without an internal-bug banner.
            raise
        except McpStartupError:
            details = "; ".join(
                f"{name}: {status.status}"
                for name, status in mcp_client.validated_mcps.items()
                if status.status != "ok"
            )
            return RunOutcome(
                config,
                run_paths,
                run_state,
                status="incomplete",
                error=run_launch.WebSafeDiagnostic(
                    f"configured MCP server startup failed: {details}"
                ),
            )
        except Exception:  # noqa: BLE001
            # Reached only when an exception propagates out of Search.run(), or
            # when setup fails before Search.run() starts. An ordinary
            # in-pipeline bug is recorded as a Failure and lands in the `else`
            # branch below. The traceback reaches stderr through the normal
            # Python logger.
            logger.exception("Run crashed")
            return RunOutcome(
                config,
                run_paths,
                run_state,
                status="crashed",
                error=_UNEXPECTED_BUG_MESSAGE,
            )
        else:
            if terminal == TerminalStatus.COMPLETED:
                # A finished run may share itself with the Kb. It is a
                # convenience after the result, not a stage of the run: the
                # upload answers instead of raising, and what it answers is
                # printed and forgotten.
                if _auto_contribute_enabled():
                    try:
                        print(
                            contribution_text(
                                upload_run(
                                    Path(run_paths.run_home), _kb_run_payload(config)
                                )
                            )
                        )
                    except KeyboardInterrupt:
                        # The result is saved and the run IS completed. A key
                        # pressed during the optional upload stops the upload.
                        print("aibuildai: contribution interrupted", file=sys.stderr)
                return RunOutcome(
                    config,
                    run_paths,
                    run_state,
                    status="completed",
                )
            if terminal == TerminalStatus.FAILED_UNEXPECTED:
                # Search.run() logged the traceback and persisted the
                # FAILED/UNEXPECTED state, then returned cleanly. Map it to
                # status="crashed" (exit 70) under the blame-neutral banner.
                return RunOutcome(
                    config,
                    run_paths,
                    run_state,
                    status="crashed",
                    error=_UNEXPECTED_BUG_MESSAGE,
                )
            # A non-completed terminal includes expected execution dead ends.
            # Search saved their reason and returned cleanly, so they land here
            # rather than in `except`. The root Action's own Failure carries
            # that reason ("Aggregator skipped: ...") — a plain sentence, no
            # traceback — so the summary headlines the persisted result and
            # falls back to the terminal name only when none was recorded.
            output = search._run_output()
            return RunOutcome(
                config,
                run_paths,
                run_state,
                status="incomplete",
                error=(
                    output.reason
                    if isinstance(output, Failure)
                    else f"terminal status: {terminal.value}"
                ),
            )
        finally:
            if search is not None:
                try:
                    if (
                        not config.run.require_kubernetes_start_mode()
                        and read_run_head(
                            config.run.require_run_id(), run_paths.run_home
                        ).result
                        is not None
                    ):
                        run_launch.remove_run_launch_context(
                            config.run.require_run_id()
                        )
                except Exception as error:  # noqa: BLE001 — terminal result already persisted; cleanup must not rewrite it.
                    logger.exception(
                        "final Run launch-context cleanup failed: %s", error
                    )
                # Every record already committed at save. Closing only frees the
                # connection pool and remains independent from result cleanup.
                if store is not None:
                    store.close()


class WritePaperCommand(Command):
    """`aibuildai write-paper <run-dir>`: write a NeurIPS paper describing a FINISHED run and compile it to a PDF, without re-running the pipeline.

    Reuses the production assembly on the existing run. WriterAgent is a standalone DurableExecution, so the finished AIBuildAISearch stays DONE."""

    def __init__(self, run_dir: "Path") -> None:
        self._run_dir = run_dir

    def execute(
        self,
        config: "AgentConfig",
        *,
        openai_proxy: "OpenAIProxy | None" = None,
        model_env: dict[str, str],
    ) -> CommandOutcome:
        run_timestamp = self._run_dir.resolve().name
        (
            search_method_type,
            run_paths,
            router,
            mcp_client,
        ) = prepare_run(config, resume_run_timestamp=run_timestamp)
        if openai_proxy is not None:
            openai_proxy.set_run_home(run_paths.run_home)
        search: "AIBuildAISearch | None" = None
        store: "SearchApplication | None" = None
        paper: str | None = None
        try:
            from bootstrap import build_search_for_production
            from engine.durable_execution import run_host

            async def build_and_write() -> None:
                nonlocal search, store, paper
                task = asyncio.current_task()
                assert task is not None

                def request(_reason: str) -> None:
                    task.cancel()

                async with _interrupt_signals(request):
                    search = await build_search_for_production(
                        config,
                        search_method_type,
                        run_paths,
                        router,
                        mcp_client,
                        launch_environment_file=None,
                        model_env=model_env,
                    )
                    try:
                        store = run_store()
                        paper = await search.write_paper()
                    finally:
                        await run_host().release()

            asyncio.run(build_and_write())
        except (asyncio.CancelledError, KeyboardInterrupt):
            print("aibuildai: write-paper interrupted", file=sys.stderr)
            return ExitCodeOutcome(130)
        except Exception as e:  # noqa: BLE001 — surface a clean CLI error, no traceback to the user
            print(
                f"aibuildai: write-paper failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return ExitCodeOutcome(1)
        finally:
            if search is not None:
                if store is not None:
                    store.close()
        # The Writer's own artifact is the paper: the command reports the path
        # that execution owns, because no run-global copy of it exists.
        pdf = None if paper is None else Path(paper)
        if pdf is not None and pdf.is_file() and pdf.read_bytes()[:4] == b"%PDF":
            print(f"paper written: {pdf}")
            return ExitCodeOutcome(0)
        print("aibuildai: write-paper produced no PDF", file=sys.stderr)
        return ExitCodeOutcome(1)


class MemorizeCommand(Command):
    def execute(
        self,
        config: "AgentConfig",
        *,
        openai_proxy: "OpenAIProxy | None" = None,
        model_env: dict[str, str],
    ) -> CommandOutcome:
        run_memorize_command(config, model_env)
        return ExitCodeOutcome(0)


class AuditCommand:
    """Audit a completed run or recorded Failure for trust and integrity red flags. This is not a config-driven Command: it takes a resolved run dir, not an AgentConfig, and is routed outside the run and memorize config, endpoint, and resource path in cli() before ProcessLifecycle. It is read-only, with no workers, systemd service, or subscription check. It runs engine.audit.run_audit, prints the findings, and leaves audit.json in the run dir."""

    def __init__(
        self, run_dir: str, reference: "str | None", model: "str | None"
    ) -> None:
        self.run_dir = run_dir
        self.reference = reference
        self.model = model

    def execute(self) -> None:
        # Function-local: run_audit opens its own Conversation only for audit
        # invocations; every non-audit command avoids importing the adapter here.
        from engine.audit import run_audit

        out = asyncio.run(run_audit(self.run_dir, self.reference, self.model))
        print(f"Audit of {self.run_dir}")
        print(f"  trustworthy: {out.trustworthy}")
        print(f"  {out.summary}")
        for f in out.findings:
            severity = "severe" if f.severe else "minor"
            print(f"  [{f.verdict}] {f.violation_id} ({severity}): {f.label}")
            if f.verdict != "absent":
                if f.evidence:
                    print(f"      evidence: {f.evidence}")
                print(f"      {f.explanation}")
        print(f"  report written to {self.run_dir}/audit.json")


class ContributeCommand:
    """`aibuildai contribute <run-dir>`: pack a finished run home's shareable text and upload it to the kb service's `/contribute` endpoint (manual). Routed before ProcessLifecycle like audit; no config, no workers, no subscription check -- so this command runs no run gate of its own, and the upload carries whatever run grant is already cached, or none."""

    def __init__(self, run_dir: str) -> None:
        self._run_dir = Path(run_dir)

    def execute(self) -> None:
        run_home = self._run_dir

        # Contribution reads identity, not a configuration it will execute.
        # Old runs can carry retired runtime fields without changing that identity.
        try:
            archive = json.loads(
                (run_home / run_config_store.RUN_CONFIG_FILENAME).read_text(encoding="utf-8")
            )
            payload = KbRequestPayload(
                run_id=archive["run"]["run_id"],
                task_name=archive["run"]["task_name"],
                caller=None,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(
                f"aibuildai: cannot read the archived run ID and task name in {run_home}: {exc}",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        result = upload_run(run_home, payload)
        text = contribution_text(result)
        if result.status == "failed":
            raise SystemExit(f"aibuildai: {text}")
        print(text)


def _auto_contribute_enabled() -> bool:
    """Resolve the strict auto-Contribution toggle for this process.

    Unset means on in a source checkout and off in the shipped binary. Sending a run's content to the shared memory is the CUSTOMER's decision, so their build asks first; a source checkout is ours, and it is what the shared memory is meant to learn from. An explicit `0` or `1` still decides either build.

    The launching shell's choice reaches every epoch of the run through the private launch EnvironmentFile that ``ProcessLifecycle`` captures and Resume hands back, so a resumed process reads the same explicit value here. Unset stays stable across a resume too, because a resume requires the run's own product version and the default follows only that.
    """
    raw = os.environ.get("AIBUILDAI_CONTRIBUTE")
    if raw is None:
        return not is_frozen()
    if raw == "0":
        return False
    if raw == "1":
        return True
    raise AssertionError(
        f"AIBUILDAI_CONTRIBUTE must be exactly '0' or '1' when set, got {raw!r}"
    )


def _kb_run_payload(config: "AgentConfig") -> KbRequestPayload:
    return KbRequestPayload(
        run_id=config.run.require_run_id(),
        task_name=config.run.task_name,
        caller=None,
    )


def _prepare_stored_state_types(run_home: str) -> None:
    """Load the classes named by one run before its state is restored."""
    from engine.builtin import load_run_definitions
    from startup.extensions import import_extension_classes

    config = run_config_store.load_run_config(run_home)
    import_extension_classes(config.extensions.dir)
    load_run_definitions(config.search.kind)


COMMANDS: dict[str, type[Command]] = {
    "run": RunCommand,
    "memorize": MemorizeCommand,
}


def _wire_frozen_ca_bundle() -> None:
    """Point the standard CA-bundle env vars at certifi's bundled cert in the frozen binary, so every outbound HTTPS can verify.

    The PyInstaller binary ships no system CA store, so OpenSSL's default verify paths resolve to nothing: the stdlib default SSL context used by ``urllib`` (the MCP startup check in ``engine/mcp.py``), ``requests``, and the bundled ``claude`` subprocess all fail HTTP-type MCP connections with ``CERTIFICATE_VERIFY_FAILED``. Exporting ``SSL_CERT_FILE`` (read by OpenSSL, hence stdlib ``ssl`` / ``urllib``) and ``REQUESTS_CA_BUNDLE`` (read by ``requests``) makes all of them -- and the ``claude`` subprocess that inherits this env -- trust certifi's bundle. ``setdefault`` so an operator-provided CA override still wins. Inert in source mode (system certs already present); the Clerk session verifier is unaffected (it builds its own certifi-rooted context).
    """
    if not is_frozen():
        return
    import certifi  # bundled CA; imported in-branch as in subscription.session

    ca_bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)


def _check_release_artifact(run_home: str) -> None:
    """Write and restore the journal with the built binary before release."""
    if not is_frozen():
        raise AssertionError("the release check requires a frozen binary")

    from infra.postgresql import ensure_postgresql
    from config import AgentConfig
    from engine.builtin.tree.io import build_input
    import yaml

    ensure_postgresql()

    run_id = uuid.uuid4().hex
    timestamp = datetime.now(UTC)
    config = AgentConfig.model_validate(yaml.safe_load(example_yaml_text()))
    search_input = build_input(
        {"parallel": 1}, config,
        RunPaths(Path(run_home), RunTimestamp.new(run_id)),
    )
    state = RunState(
        run_id=run_id,
        run_home=run_home,
        app_version=APP_VERSION,
        feature_set_hash=features.hidden_feature_set_hash(),
        gpu_visibility=None,
        archive_location=str(Path(run_home) / "run_config.json"),
        schema_version=SCHEMA_VERSION,
        search_kind="tree",
        search_input=search_input.model_dump(mode="json"),
        run_started_at_unix_s=timestamp.timestamp(),
        ts=0.0,
        timestamp=timestamp,
    )
    store = open_search_store(run_id)
    store.save_pending(
        lambda emit: None,
        state=state,
        ts=0.0,
        live=False,
        workflow_path=None,
        workflow_step=None,
    )
    store.save_pending(
        lambda emit: emit(RunEpochOpened, None, cwd=os.getcwd()),
        state=state,
        ts=0.0,
        live=False,
        workflow_path=None,
        workflow_step=None,
    )
    store.close()
    store = open_search_store(run_id, read_only=True)
    events = read_events(store)
    restored = store.restore()
    store.close()
    if [type(event) for event in events] != [RunOpened, RunEpochOpened]:
        raise AssertionError("the release journal must contain RunOpened and RunEpochOpened")
    if events[0].originator_version >= events[1].originator_version:
        raise AssertionError("the release journal sequence must increase")
    if not isinstance(restored, RunState):
        raise AssertionError("the release journal must restore RunState")
    if restored.run_home != run_home:
        raise AssertionError("the restored release journal does not match its events")


def _execute_configured(
    config: "AgentConfig",
    action: "Callable[[OpenAIProxy | None, dict[str, str]], CommandOutcome]",
) -> CommandOutcome:
    """Run the shared subscription, tool, endpoint, and model startup gates."""
    ignore_verdict = (
        not is_frozen() and os.environ.get("AIBUILDAI_SKIP_SUBSCRIPTION_CHECK") == "1"
    )
    from subscription.run_grant import authorize_run
    from subscription.session import SubscriptionError

    try:
        authorize_run()
    except SubscriptionError as error:
        if not ignore_verdict:
            raise run_launch.WebSafeDiagnostic(str(error)) from error

    from infra.owned_tools import setup_owned_tools
    from infra.postgresql import ensure_postgresql

    setup_owned_tools()
    ensure_postgresql()
    with model_endpoint(config.llm, Path(config.output_base_dir)) as (model_env, openai_proxy):
        return action(openai_proxy, model_env)


def _record_existing_run_failure(
    result_file: "Path | None", diagnostic: run_launch.WebSafeDiagnostic
) -> None:
    """Write one safe Resume failure without changing the Run result."""
    if result_file is None:
        return
    try:
        run_launch.write_resume_failure(result_file, diagnostic)
    except Exception:  # noqa: BLE001 — diagnostic failure must not replace the original failure.
        logger.exception("private Resume result could not be written")


def run_existing_process(run_id: str) -> None:
    """Enter one existing Run process that its startup launcher already created."""
    scrub_claude_feature_env()
    _wire_frozen_ca_bundle()
    result_file: Path | None = None
    try:
        from startup.run_target_resolution import resolve_run_target

        run = resolve_run_target(run_id)
        result_file = run_launch.claim_resume_result(run_id)

        def execute() -> CommandOutcome:
            materialize_to_tmpfs()
            config = run_config_store.load_run_config(run.run_home)

            def resume(openai_proxy: "OpenAIProxy | None", model_env: dict[str, str]) -> CommandOutcome:
                return RunCommand().execute(
                    config,
                    resume_run_timestamp=Path(run.run_home).name,
                    openai_proxy=openai_proxy,
                    model_env=model_env,
                )

            return _execute_configured(config, resume)

        outcome = ProcessLifecycle.run_already_in_service(execute)
        exit_code = outcome.finalize()
        if exit_code != 0:
            diagnostic = (
                outcome.error
                if isinstance(outcome, RunOutcome)
                and isinstance(outcome.error, run_launch.WebSafeDiagnostic)
                else run_launch.WebSafeDiagnostic(
                    "the resumed Run stopped before its live member became ready"
                )
            )
            _record_existing_run_failure(result_file, diagnostic)
        raise SystemExit(exit_code)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        diagnostic = run_launch.WebSafeDiagnostic(
            "Resume was interrupted before the Run became live"
        )
        _record_existing_run_failure(result_file, diagnostic)
        print(f"aibuildai: {diagnostic}", file=sys.stderr)
        raise SystemExit(130) from None
    except (
        HostCapabilityUnavailable,
        PostgreSQLUnavailable,
        RunPinMismatch,
        run_launch.WebSafeDiagnostic,
    ) as error:
        diagnostic = error if isinstance(error, run_launch.WebSafeDiagnostic) else run_launch.WebSafeDiagnostic(str(error) or "Resume startup was refused")
        _record_existing_run_failure(result_file, diagnostic)
        print(f"aibuildai: {error}", file=sys.stderr)
        raise SystemExit(69 if isinstance(error, HostCapabilityUnavailable) else 2) from None
    except Exception as error:  # noqa: BLE001 — private process boundary.
        known = isinstance(error, (OSError, StartupUserInputError, ValueError))
        diagnostic = run_launch.WebSafeDiagnostic(
            "Resume startup failed; inspect the Run log"
            if known
            else "Resume startup failed unexpectedly"
        )
        if not known and not is_frozen():
            traceback.print_exc()
        _record_existing_run_failure(result_file, diagnostic)
        print(f"aibuildai: {error}" if known else _PRE_RUN_CRASH_MESSAGE, file=sys.stderr)
        raise SystemExit(2 if known else _EX_SOFTWARE) from None


def cli() -> None:
    scrub_claude_feature_env()
    # Frozen binary ships no system CA store: aim SSL_CERT_FILE /
    # REQUESTS_CA_BUNDLE at certifi's bundle BEFORE any HTTPS client, the MCP
    # proxy, or the bundled claude subprocess starts, so outbound HTTPS (e.g.
    # HTTP-type MCP servers) can verify. Inert in source mode.
    _wire_frozen_ca_bundle()

    if sys.argv[1:2] == ["__release_check"]:
        if len(sys.argv) != 3:
            raise SystemExit("usage: aibuildai __release_check CODE_DIR")
        _check_release_artifact(sys.argv[2])
        return

    # Auth commands (login/logout/whoami/account) carry no config, skip the
    # systemd service, and skip the subscription check. Route them BEFORE
    # ProcessLifecycle so an interactive browser login never re-execs into a
    # systemd service. Parsing has no YAML or other side effect. Each process
    # keeps this one result. A systemd re-exec is a new process and parses its
    # own command line once.
    args = config_loader.parse_args()
    command = args.command
    if command in config_loader.AUTH_COMMANDS:
        from subscription import auth

        try:
            {
                "login": lambda: auth.login(no_browser=bool(args.no_browser)),
                "logout": auth.logout,
                "whoami": auth.whoami,
                "account": auth.open_account_page,
            }[command]()
        except KeyboardInterrupt:
            # Ctrl-C during an auth command (e.g. mid network read) is an expected
            # user action, not a bug. Stop cleanly with the standard interrupt exit
            # code instead of letting this BaseException reach the frozen PyInstaller
            # bootloader, which would print a raw traceback + "Failed to execute
            # script". The run path has its own interrupt handling; the auth dispatch
            # is routed before that machinery, so it carries its own boundary here.
            print("Interrupted.", file=sys.stderr)
            sys.exit(130)
        except auth.LoginError as error:
            # An expected, actionable sign-in failure: the callback port is
            # taken, the redirect never arrived, the browser tab was closed. The
            # message already names what the user must do next, so report it the
            # way every other user-fixable CLI error here is reported -- the
            # sentence plainly on stderr, exit 2. Unhandled, it left the process
            # as a raw traceback in source mode and, in the frozen binary a paying
            # user actually runs, the PyInstaller bootloader's "Failed to execute
            # script" noise, which buries the one line that says what to do.
            print(f"error: {error}", file=sys.stderr)
            sys.exit(2)
        except (
            requests.ConnectionError,
            requests.HTTPError,
            requests.Timeout,
            requests.TooManyRedirects,
        ) as error:
            # Every network leg of the auth flow -- the authorize redirect, the
            # code-for-tokens exchange, the userinfo lookup, the run-grant
            # pre-check -- surfaces here: DNS, TLS, timeout, redirect exhaustion,
            # or a rejecting status. The user's machine or the sign-in service is
            # the thing to act on, not aibuildai, so this is a plain error too.
            # Request-shape failures such as MissingSchema and InvalidURL are not
            # in this arm: product code owns every endpoint, so a malformed one is
            # an internal defect and reaches the exit-70 branch below.
            print(
                f"error: could not reach the sign-in service: {error}", file=sys.stderr
            )
            sys.exit(2)
        except Exception:  # noqa: BLE001 — auth boundary, mirrors the run boundary below.
            # Anything else is an internal defect. Same shape as the run-startup
            # boundary at the end of this function: the blame-neutral one-liner
            # on stderr and exit 70, with the raw traceback ONLY on a non-frozen
            # dev run. Keeping the traceback off a shipped user's screen is
            # deliberate and stays that way -- a login traceback's frames sit on
            # the code-for-tokens exchange and the userinfo call, and a future
            # change to how Python renders a frame must not be the thing that
            # decides whether a credential reaches a terminal.
            if not is_frozen():
                traceback.print_exc()
            print(_AUTH_CRASH_MESSAGE, file=sys.stderr)
            sys.exit(_EX_SOFTWARE)
        return

    # setup is a thin pre-warm shortcut around the SAME owned-tools path every
    # run-like command executes automatically; it owns no logic of its own and
    # run never depends on it having been called. Routed before ProcessLifecycle
    # like the other config-free commands.
    if command == "setup":
        from infra.owned_tools import setup_owned_tools

        try:
            setup_owned_tools()
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
        print("aibuildai: owned tools are ready")
        return

    # Audit is a read-only command over a completed run or recorded Failure. It
    # needs the owned PostgreSQL service, but no config, worker, model, or
    # subscription check. ProcessLifecycle is not needed. It accepts a Run ID,
    # a Run home, or a Run dir through the
    # shared target resolver, so
    # bad input is one sentence naming
    # what the user typed, never a traceback naming the current directory (#2201).
    if command == "audit":
        from infra.owned_tools import setup_owned_tools
        from infra.postgresql import ensure_postgresql
        from startup.run_catalog import run_facts
        from startup.run_target_resolution import (
            RunTargetResolutionError,
            resolve_run_target,
        )

        setup_owned_tools()
        ensure_postgresql()
        try:
            target = resolve_run_target(args.target)
        except RunTargetResolutionError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        target_config = run_config_store.load_run_config(target.run_home)
        if not target_config.run.require_kubernetes_start_mode() and unit_active(
            run_unit_name(target.run_id)
        ):
            print(
                f"error: run {target.run_id} is still running in another process",
                file=sys.stderr,
            )
            sys.exit(2)
        if run_facts(target.run_id, target.run_home).result is None:
            print(
                f"error: audit needs a completed run or a recorded final "
                f"Failure; run {target.run_id} has neither",
                file=sys.stderr,
            )
            sys.exit(1)
        AuditCommand(
            target.run_home,
            args.reference,
            args.model,
        ).execute()
        return

    # Contribute is an opt-in upload of a finished run dir to the shared
    # experience memory (no config, no workers, no subscription check). Route it
    # before ProcessLifecycle like audit; it returns without sys.exit.
    if command == "contribute":
        ContributeCommand(args.run_dir).execute()
        return

    # `config` prints a starter YAML (every field + its default) to stdout: pure
    # introspection, no YAML load / subscription check / systemd service. Routed
    # before ProcessLifecycle and the auth commands.
    if command == "config":
        print(example_yaml_text(), end="")
        return

    # write-paper carries a run-dir (not a YAML); capture its args
    # here and build the config from the run dir inside ProcessLifecycle below.
    write_paper_run_dir = None
    if command == "write-paper":
        write_paper_run_dir = Path(args.run_dir)
        write_paper_review = bool(args.review)

    # One Run identity at the outer launch boundary, before the run process
    # exists. The deterministic systemd unit is named after it, so the
    # supervisor itself refuses a second process for the same active run. A
    # fresh run allocates the identity here and carries it across the service
    # re-exec in the env marker; write-paper reuses its existing identity.
    # Commands without a Run ID (memorize) keep a
    # process-instance unit name.
    run_unit = None
    launch_run_id = None
    if command == "run":
        launch_run_id = os.environ.get("AIBUILDAI_RUN_ID") or uuid.uuid4().hex
        os.environ["AIBUILDAI_RUN_ID"] = launch_run_id
        run_unit = run_unit_name(launch_run_id)
    elif write_paper_run_dir is not None:
        paper_home = Path(os.path.normpath(str(Path.cwd() / write_paper_run_dir)))
        run_unit = run_unit_name(RunTimestamp.parse(paper_home.name).run_id)

    try:
        # Auto-update (frozen binary only): check the public repo for a newer stable
        # release and re-exec into it BEFORE any heavy component (systemd scope,
        # worker slice, run-dir lock) is built, so the os.execv leaks nothing.
        # Deferred heavy import: only paid when the gate passes. Inside this try so
        # the pre-lifecycle KeyboardInterrupt boundary below also covers its blocking
        # network check; maybe_auto_update already swallows Exception, so this does
        # not change the #1069 except-Exception path.
        from infra.update.gate import should_check

        if should_check(command):
            from infra.update.runner import maybe_auto_update

            maybe_auto_update(command)

        def execute_command(
            fresh_environment_file: "Path | None",
        ) -> CommandOutcome:
            if command == "run":
                if launch_run_id is None:
                    raise AssertionError("a fresh Run has no allocated Run ID")
                launch_environment_file = fresh_environment_file
            else:
                launch_environment_file = None
            # Resource freezing belongs to process launch, not to either config
            # loader. Do it after ProcessLifecycle has re-execed into the run's
            # systemd service and before fresh, resumed, or write-paper config
            # construction can resolve any shipped filesystem-shaped resource.
            materialize_to_tmpfs()

            if write_paper_run_dir is not None:
                # write-paper: the archived run_config.json is the config SSOT
                # (the same archive `resume` reloads). recover_config_from_rundir
                # reads it back; only the run's LOCATION (task_name +
                # playground_root) follows the passed dir, so a copied run dir
                # writes its paper in place while every setting stays verbatim.
                from startup.run_target_resolution import recover_config_from_rundir

                try:
                    config = recover_config_from_rundir(
                        write_paper_run_dir, review_writer=write_paper_review
                    )
                except (FileNotFoundError, ValueError) as e:
                    print(f"error: {e}", file=sys.stderr)
                    sys.exit(2)
            else:
                try:
                    loaded_command, config = load_config(args)
                except ValidationError as e:
                    print(f"error: {format_validation_error(e)}", file=sys.stderr)
                    sys.exit(2)
                if loaded_command != command:
                    raise AssertionError("the parsed command changed while loading config")

            def dispatch(openai_proxy: "OpenAIProxy | None", model_env: dict[str, str]) -> CommandOutcome:
                if write_paper_run_dir is not None:
                    return WritePaperCommand(write_paper_run_dir).execute(
                        config, openai_proxy=openai_proxy, model_env=model_env
                    )
                if command == "run":
                    return RunCommand().execute(
                        config,
                        launch_environment_file=launch_environment_file,
                        model_env=model_env,
                        openai_proxy=openai_proxy,
                    )
                return COMMANDS[command]().execute(config, openai_proxy=openai_proxy, model_env=model_env)

            try:
                return _execute_configured(config, dispatch)
            except ValueError as error:
                print(f"error: {error}", file=sys.stderr)
                raise SystemExit(2) from None

        outcome = ProcessLifecycle.run(execute_command, unit_name=run_unit)
    except KeyboardInterrupt:
        # Ctrl-C anywhere in the pre-lifecycle run-startup region -- the auto-update
        # check and the subscription gate (authorize_run) are both blocking network
        # calls, and config load / endpoint-env / resource-limit setup precede the
        # command's own run loop. A KeyboardInterrupt here is an
        # expected user action, not a bug: stop cleanly with the standard interrupt
        # exit code instead of letting this BaseException reach the frozen PyInstaller
        # bootloader (raw traceback + "Failed to execute script"). The
        # ProcessLifecycle service owns the child processes, so its teardown
        # cleans them up as the interrupt leaves the with-block. The run loop's
        # own SIGINT handling (RunCommand.execute maps Ctrl-C to an "interrupted"
        # outcome) runs deeper and returns normally, so it never reaches here -- this
        # boundary does not disturb it.
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except (RunPinMismatch, PostgreSQLUnavailable) as exc:
        # A resume whose journaled environment pins no longer match the CURRENT
        # process (#1269) is reported like a config error (the actionable
        # message plainly on stderr + a clean exit code), never the
        # blame-neutral "bug in aibuildai" banner nor a raw traceback. Arrives
        # here via RunCommand.execute's re-raise past its own generic except.
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    except StartupUserInputError as exc:
        # A startup failure caused by USER INPUT -- a broken user extension file
        # (#1714) or an unregistered config value such as search.kind (#1720) --
        # not an internal bug. Report it like a config error: the actionable
        # message naming the user's mistake plainly on stderr + a clean exit code
        # (2), never the blame-neutral "bug in aibuildai" banner nor a raw
        # traceback. These arise in prepare_run (extension
        # import / search-kind selection), which runs before RunCommand.execute's
        # own try block, so they propagate directly. This clause precedes the
        # generic `except Exception` below so the user-input class wins over the
        # bug-banner path.
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception:  # noqa: BLE001
        # Issue #1069 boundary case 1: an UNEXPECTED crash in composition-root
        # setup. Print the blame-neutral one-liner (+ the raw traceback only on a
        # non-frozen dev run) and exit 70. Every sys.exit above
        # raises SystemExit, which is a BaseException, not an Exception, so it
        # propagates unchanged (clean CLI errors keep their codes); KeyboardInterrupt
        # (also a BaseException) is handled by the boundary just above.
        if not is_frozen():
            traceback.print_exc()
        print(_PRE_RUN_CRASH_MESSAGE, file=sys.stderr)
        sys.exit(_EX_SOFTWARE)
    # ProcessLifecycle.__exit__ has returned after the run's child processes were
    # handled by its service. Render the result on the ordinary terminal.
    sys.exit(outcome.finalize())
