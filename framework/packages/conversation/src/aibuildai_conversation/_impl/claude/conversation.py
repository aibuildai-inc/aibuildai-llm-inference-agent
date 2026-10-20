"""Conversation: one live model conversation lineage over the Claude SDK.

A Conversation owns transport, provider request boundaries, interruption recovery, session continuity, and live notifications (#3047). Its whole surface is ``send``, ``notify``, ``interrupt``, and ``close``: a caller hands it one message per interaction and never sees a provider request, a retry, or an SDK type. Every fact of a request reaches the ``ConversationObserver`` that interaction's ``send`` was given, as it happens, as a named semantic observation carrying the fact's own values, so the Agent journals what the provider did while the Conversation alone decides what the model hears next. Every fact is reported inside the send that caused it: nothing is observed between interactions or at close.

The one invariant: a provider request with no recorded completion is never resent. The next request opens with the recorded lifecycle facts, and the caller's message rides along only as labelled context.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import anyio
from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, ClaudeSDKError, ConversationResetMessage
from claude_agent_sdk import ProcessError, RateLimitEvent, ResultError, ResultMessage, SystemMessage, ToolUseBlock

from aibuildai_conversation.errors import AgentError
from aibuildai_conversation._impl.claude import config_paths
from aibuildai_conversation._impl.claude.agent_launcher import _create_memfd, build_agent_launcher
from aibuildai_conversation._impl.claude.errors import (
    BackendError,
    CANCELLED_REASONS,
    PermanentError,
    _RetryableError,
    classify_assistant_error,
    classify_result_error,
)
from aibuildai_conversation._impl.claude.event_adapter import RequestStream
from aibuildai_conversation._impl.observer import ConversationObserver
from aibuildai_conversation._impl.claude.spec_to_options import spec_to_options
from aibuildai_conversation._impl.spec import AgentSpec, Image, Message
from aibuildai_conversation.environment import scrub_claude_feature_env

logger = logging.getLogger(__name__)

# Flat poll interval between requests while the provider is unavailable: the
# three-outcome contract asks "is it recovered yet", never "when" (#1253).
_RETRY_POLL_INTERVAL_S = 30.0
# Enough of one failed claude-cli attempt to diagnose it, bounded both ways.
_STDERR_TAIL_MAX_LINES = 32
_STDERR_LINE_MAX_CHARS = 2048

# The recorded lifecycle facts a request opens with, and the label the caller's
# message carries when it follows them: context for the model, not a replay.
# Every line states something this Conversation observed about its own
# provider requests, or was told about them by the caller that owns the fact;
# nothing here speaks for the caller's identity, workspace, Run, or process.
_LIFECYCLE_HEADER = "[AIBuildAI recorded lifecycle facts]"
_CONTEXT_HEADER = "[Current caller request, repeated as task context]"
_INTERRUPTED = (
    "- The previous provider request did not reach a recorded completion.",
    "- Some model output or tool effects may already have occurred.",
    "- AIBuildAI did not replay that request or roll back the workspace.",
)


def _content(message: Message) -> tuple[str | Image, ...]:
    if not isinstance(message, (str, Sequence)):
        raise TypeError("a message must contain text or Image values")
    parts = (message,) if isinstance(message, str) else tuple(message)
    if not parts or any(not isinstance(part, (str, Image)) for part in parts):
        raise ValueError("a message requires text or Image values")
    return parts


def _join_content(groups: Sequence[tuple[str | Image, ...]]) -> tuple[str | Image, ...]:
    """Join framework groups with the existing blank-line separator; keep image order."""
    parts: list[str | Image] = []
    for group in groups:
        if parts:
            last = parts[-1]
            separator = (last if isinstance(last, str) else "") + "\n\n"
            if isinstance(last, str):
                parts.pop()
            first, *rest = group
            if isinstance(first, str):
                parts.append(separator + first)
                parts.extend(rest)
                continue
            parts.append(separator)
        parts.extend(group)
    return tuple(parts)


async def _native_prompt(parts: tuple[str | Image, ...]) -> AsyncIterator[dict]:
    """Emit one SDK user message for one provider request."""
    yield {
        "type": "user",
        "message": {"role": "user", "content": [
            {"type": "text", "text": part} if isinstance(part, str) else
            {"type": "image", "source": {"type": "base64", "media_type": part.mime_type,
             "data": base64.b64encode(part.data).decode("ascii")}}
            for part in parts
        ]},
        "parent_tool_use_id": None,
    }


def _facts(*lines: str) -> str:
    """One lifecycle-facts block: facts only, no instruction."""
    return "\n".join((_LIFECYCLE_HEADER, *lines))


def _interrupted_facts(diagnostic: str | None) -> str:
    return _facts(*_INTERRUPTED, *([f"- Provider diagnostic: {diagnostic}"] if diagnostic else []))


def _replaced_facts(old_session_id: str, new_session_id: str) -> str:
    return _facts(
        f"- The provider conversation changed from {old_session_id} to {new_session_id}.",
        "- The new provider conversation may not contain the earlier discussion.",
        "- The current caller request remains open.",
    )


def _cli_failure(exc: ClaudeSDKError, captured_stderr: str, oom_before: int | None, oom_after: int | None) -> AgentError:
    """What the SDK said about the CLI, with this attempt's own stderr beside it. A ``ProcessError`` is the CLI dying without a result: its only structured fact is the exit code (the SDK's ``stderr`` field there is a placeholder), so the report is the exit code plus the captured tail. Every other SDK error reports itself."""
    said = f"Claude CLI exited with code {exc.exit_code}" if isinstance(exc, ProcessError) else f"{type(exc).__name__}: {exc}"
    return AgentError(f"{said}\nCLI stderr tail:\n{captured_stderr}" if captured_stderr else said,
        oom_killed=oom_before is not None and oom_after is not None and oom_after > oom_before)


def _is_genuine_recovery(sdk_ev: object) -> bool:
    """True iff this event is evidence the provider is productive again, for the sole purpose of closing an open rate-limit/retry pause. A ``SystemMessage`` (a fresh reconnect) is not recovery by itself, and an ``AssistantMessage`` carrying ``.error`` is the same ongoing failure; excluding both lets the outage start accumulate while the provider stays down (#1438) yet closes promptly on real content (#1336)."""
    if isinstance(sdk_ev, (ResultMessage, SystemMessage, ToolUseBlock)):
        return False
    return not (isinstance(sdk_ev, AssistantMessage) and sdk_ev.error)


@contextmanager
def _preserve_failure(failure: BaseException | None) -> Iterator[None]:
    """Keep a request or caller failure when later cleanup also fails."""
    try:
        yield
    except BaseException as cleanup_error:
        if failure is not None and cleanup_error is not failure:
            raise failure from cleanup_error
        raise


@dataclass(frozen=True)
class Response:
    """What one ``send`` produced: the provider's terminal structured candidate, if any, and the completed turn's top-level text. The caller's own gates decide whether the candidate becomes an Output."""

    output: dict[str, Any] | None  # The caller's JSON schema determines each value.
    text: str


class Conversation:
    """One live model conversation over ClaudeSDKClient.

    Lifecycle:
      async with Conversation(spec) as conversation:
          response = await conversation.send(message, observers=(business_observer,))
          await conversation.notify(text)
          await conversation.notify(text, interrupt=True)
          await conversation.interrupt()

    One ``send`` at a time; ``notify`` and ``interrupt`` may be called from another task while it runs. An interrupt reaches only a provider request that is connected and in flight; with none, a queued notification simply belongs to the next request, and a request never sends while an earlier control request is still unsettled.

    The SDK owns its reader task, control requests, in-process MCP bridge, receive stream, subprocess, and cancellation-safe disconnect. This class owns the session id, the native result classification, the transport retry policy, and the input of every provider request: one request performs at most one ``client.query``; a request the provider does not complete ends the request, never the send, and the next request waits out the flat poll interval and opens with the recorded facts, so no input is ever replayed (#3047).

    ``resume_session_id`` continues the recorded provider conversation under ``spec.cwd``; one the SDK no longer has fails the first send with ``PermanentError(SESSION_LOST, resume_unavailable)``, so no caller mistakes a fresh conversation for continuity. ``unfinished_request`` is the caller's folded journal fact that its last provider request has no recorded completion: the first request then opens with the interruption facts and carries the caller's message as context; a reopened conversation whose last request completed gets the next message plainly. ``pause_open_for_s`` is the caller's folded fact that a provider outage pause is already open and how long it has lasted: this Conversation then owns that pause from the start and closes it, once, on the first productive event. ``retry_ceiling_s`` bounds the outage patience; None waits forever (#2502).
    """

    def __init__(
        self,
        spec: AgentSpec,
        *,
        resume_session_id: str | None = None,
        unfinished_request: bool = False,
        pause_open_for_s: float | None = None,
        retry_ceiling_s: float | None = None,
    ) -> None:
        self._spec = spec
        self._last_result: ResultMessage | None = None
        self._tool_denials: deque[dict] = deque()
        self._observed_hooks: set[str] = set()
        self._dispatch_inputs: dict[str, asyncio.Future[dict]] = {}
        self._pending_commands: set[str] = set()
        self._disconnect_lock = asyncio.Lock()
        self._env = dict(os.environ)
        scrub_claude_feature_env(self._env)
        self._env["CLAUDE_CODE_ENTRYPOINT"] = "sdk-py"
        self._env.update(spec.env)
        if not self._env.get("CLAUDE_CONFIG_DIR"):
            self._env.pop("CLAUDE_CONFIG_DIR", None)
        self._retry_ceiling_s = retry_ceiling_s
        self._client: ClaudeSDKClient | None = None
        self._launcher_path: str | None = None
        self._launcher_fd: int | None = None
        prompt_fd = _create_memfd("aibuildai-claude-prompt")
        try:
            with os.fdopen(os.dup(prompt_fd), "wb") as stream:
                stream.write(self._spec.instructions.encode("utf-8"))
        except BaseException:
            os.close(prompt_fd)
            raise
        self._prompt_fd: int | None = prompt_fd
        self._last_sid: str | None = resume_session_id
        self._resume_unchecked: str | None = resume_session_id
        self._closed = False
        self._terminated = False  # set after PermanentError; guards send
        self._sending = False
        # The connected client whose request is in flight: the only thing an
        # interrupt can reach. Control requests in flight are counted so the
        # next request sends only once every one of them has settled.
        self._active: ClaudeSDKClient | None = None
        self._controls = 0
        self._control_idle = asyncio.Event()
        self._control_idle.set()
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_MAX_LINES)
        # Outage state across requests: when the current outage began (until
        # the first productive event, so the ceiling spans every interrupted
        # request) and when the last interrupted request failed (the next one
        # waits out the remaining flat interval from there).
        self._wait_started_at: float | None = (
            None if pause_open_for_s is None else time.time() - pause_open_for_s
        )
        self._interrupted_at: float | None = None
        # The recorded facts the next request opens with, if any.
        self._facts: str | None = _interrupted_facts(None) if unfinished_request else None
        self._notifications: list[tuple[str | Image, ...]] = []  # live-only, for the next request

    async def __aenter__(self) -> "Conversation":
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        with _preserve_failure(exc):
            await self.close()

    # ------------------------------------------------------------------
    # notify / interrupt / close
    # ------------------------------------------------------------------

    async def notify(self, message: Message, *, interrupt: bool = False) -> None:
        """Queue live-only extra context for the next provider request, like typing another message while the model works. A send in flight delivers it before returning; ``interrupt=True`` stops the provider request in flight first, so it is the next input at once, and with no request in flight it is simply part of the next one. Not durable: process death or a conversation that already ended loses it."""
        self._notifications.append(_content(message))
        if interrupt:
            await self.interrupt()

    async def interrupt(self) -> None:
        """Stop the provider request in flight. No-op while none is: a request is in flight from its ``client.query`` to its end, so an interrupt during a connect, a retry wait, or between requests touches nothing and a queued notification then belongs to the next request.

        The SDK's interrupt is one control request answered by the CLI, not tied to a request id, so it is counted as in flight until answered and the next request sends only after that. The CLI answers the interrupted request with a result whose ``terminal_reason`` names the abort; the request ends on it as a completed, cancelled turn, never as a provider failure. An interrupt whose request ended under it, or that lost a race with ``close()``, has nothing left to interrupt: the SDK reports that as a connection error or an unanswered control request, and both are moot once that request is over.
        """
        client = self._active
        if client is None:
            return
        self._controls += 1
        self._control_idle.clear()
        try:
            await client.interrupt()
        except Exception:
            if self._active is not client or self._client is None:
                return
            raise
        finally:
            self._controls -= 1
            if self._controls == 0:
                self._control_idle.set()

    async def close(self) -> None:
        """Release the conversation's resources. Idempotent."""
        if self._closed and not self._pending_commands:
            return
        self._closed = True
        try:
            await self._disconnect()
        finally:
            if self._launcher_fd is not None:
                os.close(self._launcher_fd)
                self._launcher_fd = None
            if self._prompt_fd is not None:
                os.close(self._prompt_fd)
                self._prompt_fd = None

    async def _disconnect(self) -> None:
        """Stop an active request before closing its tool connections."""
        async with self._disconnect_lock:
            client, self._client = self._client, None
            try:
                if client is not None:
                    try:
                        if self._active is client:
                            await self.interrupt()
                    finally:
                        self._active = None
                        await client.disconnect()
            finally:
                await self._terminate_commands(tuple(self._pending_commands))

    async def _terminate_commands(self, identities: tuple[str, ...]) -> None:
        confinement = self._spec.confinement
        if confinement is not None:
            for identity in identities:
                if identity not in self._pending_commands:
                    continue
                self._pending_commands.remove(identity)
                try:
                    await confinement.terminate_command(identity)
                except BaseException:
                    self._pending_commands.add(identity)
                    raise

    # ------------------------------------------------------------------
    # send
    # ------------------------------------------------------------------

    async def send(self, message: Message, *, observers: tuple[ConversationObserver, ...] = ()) -> Response:
        """One caller interaction with the model conversation. Returns once a provider request completes with no live notification waiting, and raises only a terminal failure: a recoverable provider interruption is recovered here, with the recorded facts and this message as context, and never reaches the caller. ``observers`` hear every fact of this interaction's provider requests; a caller with nothing to record passes none."""
        if self._sending:
            raise AssertionError("a Conversation sends one message at a time")
        carried = list(_content(message))
        self._sending = True
        try:
            while True:
                response = await self._request(carried, observers)
                if response is None:
                    # Not completed: the next request opens with the facts and
                    # carries the message as context, never as a resend.
                    continue
                carried = []
                if not self._notifications:
                    return response
                # A notification accepted meanwhile is the next input, whether
                # an interrupt or the natural completion ended that request.
        finally:
            self._sending = False

    def _report_tool_denials(self, observers: tuple[ConversationObserver, ...]) -> None:
        while self._tool_denials:
            denial = self._tool_denials.popleft()
            for observer in observers:
                observer.tool_denied(**denial)

    def _close_pause(self, observers: tuple[ConversationObserver, ...]) -> None:
        """Balance an open pause bracket so this Agent's Local Time is not left frozen."""
        if self._wait_started_at is not None:
            self._wait_started_at = None
            for observer in observers:
                observer.rate_limit_resumed()

    async def _request(self, message: list[str | Image], observers: tuple[ConversationObserver, ...]) -> Response | None:
        commands_before = frozenset(self._pending_commands)
        """One provider request: at most one ``client.query``. Returns the completed turn's Response, or None when the provider did not complete it and the facts for the next request are recorded."""
        if self._terminated or self._closed:
            raise BackendError("conversation terminated")
        interrupted_at, self._interrupted_at = self._interrupted_at, None
        if interrupted_at is not None:
            # Flat-interval poll (#1253) counted from the interrupted request.
            remaining = _RETRY_POLL_INTERVAL_S - (time.time() - interrupted_at)
            if remaining > 0:
                await asyncio.sleep(remaining)
        # The stderr tail and the OOM-kill counter both belong to this launch
        # attempt alone: the cgroup outlives the attempt, so only the change
        # across it attributes a kill, and a stale tail would name the wrong
        # subprocess.
        stderr_tail = self._stderr_tail
        stderr_tail.clear()
        confinement = self._spec.confinement
        oom_before = 0 if confinement is None else confinement.memory_kills()

        client = self._client
        connected = client is not None
        if client is None:
            resume, self._resume_unchecked = self._resume_unchecked, None
            if resume is not None and not config_paths.session_available(resume, self._spec.cwd, self._env):
                # This launch's native config root has no matching transcript;
                # refuse here rather than open a fresh conversation (#3047).
                self._terminated = True
                raise PermanentError(
                    PermanentError.Reason.SESSION_LOST,
                    provider_code="resume_unavailable",
                    message=f"Claude session {resume} is not available in {config_paths.config_dir(self._env)} for {self._spec.cwd}",
                )
            opts = await spec_to_options(
                self._spec,
                environment=self._env,
                denials=self._tool_denials,
                observed_hooks=self._observed_hooks,
                dispatch_inputs=self._dispatch_inputs,
                pending_commands=self._pending_commands,
                resume_session_id=self._last_sid,
                stderr_callback=self._make_stderr_callback(stderr_tail),
            )
            prompt_fd = self._prompt_fd
            if prompt_fd is None:
                raise BackendError("conversation prompt is closed")
            launcher_path, launcher_fd = build_agent_launcher(
                self._spec.confinement,
                prompt_source=f"/proc/{os.getpid()}/fd/{prompt_fd}",
                env=opts.env,
            )
            old_launcher_fd, self._launcher_fd = self._launcher_fd, launcher_fd
            if old_launcher_fd is not None:
                os.close(old_launcher_fd)
            opts.cli_path = launcher_path
            client = ClaudeSDKClient(options=opts)
            self._client = client
            self._last_result = None
            self._launcher_path = launcher_path
        # Outcome of this request. A completed turn and a native conversation
        # replacement keep the client; every failed request disconnects first.
        outcome: dict[str, Any] = {"kind": "stream_exhausted"}  # Native outcomes carry different event payloads.
        keep_client = False
        assistant_error: BackendError | None = None
        # After a native reset: the outgoing id, then the successor's.
        replaced: tuple[str, str | None] | None = None
        stream = RequestStream(
            observers, launcher_path=self._launcher_path, requested_model=self._spec.model,
            previous_result=self._last_result, observed_hooks=self._observed_hooks,
        )
        @asynccontextmanager
        async def finish_request() -> AsyncIterator[None]:
            failure: BaseException | None = None
            try:
                yield
            except BaseException as error:
                failure = error
                raise
            finally:
                if failure is None:
                    failure = outcome.get("exc")
                with _preserve_failure(failure):
                    if stream.result is not None:
                        self._last_result = stream.result
                        if stream.result.terminal_reason in CANCELLED_REASONS:
                            async with self._disconnect_lock:
                                await self._terminate_commands(tuple(self._pending_commands - commands_before))
                    if stream.completed:
                        self._active = None
                    # Take the client before the await: close() may run at the same
                    # time (Ctrl+C), and whoever takes it first disconnects; the
                    # other finds None and leaves the SDK's own disconnect alone.
                    if not keep_client and self._client is client:
                        await self._disconnect()
                    self._active = None
                    self._dispatch_inputs.clear()
                    self._observed_hooks.clear()
                    if not keep_client:
                        self._report_tool_denials(observers)

        # Reported before the send, so a request with no recorded completion
        # leaves its durable evidence: the still-open invocation (#3047).
        async with finish_request():
            try:
                if not connected:
                    await client.connect()
                # A control request still in flight (an interrupt aimed at the
                # previous request) settles before this one sends, so it can
                # never reach this request.
                await self._control_idle.wait()
                # Composed after the connect and the settle: a notification
                # accepted during either is part of this request, never
                # delayed behind it.
                prompt = self._compose(message)
                for observer in observers:
                    observer.invocation_started(message=prompt)
                # The one native send of this request (#3047).
                await client.query(_native_prompt(prompt))
                self._active = client

                async for sdk_ev in client.receive_response():
                    # Capture the sid before translation: the next request resumes it.
                    if isinstance(sdk_ev, SystemMessage) and sdk_ev.subtype == "init":
                        sid = (sdk_ev.data or {}).get("session_id")
                        if sid is not None:
                            self._last_sid = sid

                    # The stream-level rate-limit signal opens the pause on the
                    # first rejection; nothing predicts recovery, so no reset
                    # time is kept. It carries no product content.
                    if isinstance(sdk_ev, RateLimitEvent):
                        if sdk_ev.rate_limit_info.status == "rejected" and self._wait_started_at is None:
                            self._wait_started_at = time.time()
                            for observer in observers:
                                observer.rate_limit_paused()
                        continue

                    # Primary close of the pause: the first genuinely recovered
                    # event of any request, before the StreamEvent routing so a
                    # mid-pause stream event resumes promptly (#1327, #1438).
                    if self._wait_started_at is not None and _is_genuine_recovery(sdk_ev):
                        self._close_pause(observers)

                    # The CLI replaced this conversation on the same connection:
                    # earlier facts stay real, later messages belong to the
                    # successor, and this request ends without completing.
                    if isinstance(sdk_ev, ConversationResetMessage):
                        replaced = (sdk_ev.session_id, None)
                        outcome = {"kind": "replaced"}
                        continue
                    if replaced is not None and replaced[1] is None:
                        # The first message carrying a different id (an init
                        # system message or stream event included) names the
                        # successor, before any successor fact reaches the caller.
                        new_sid = ((sdk_ev.data or {}).get("session_id") if isinstance(sdk_ev, SystemMessage)
                                   else getattr(sdk_ev, "session_id", None))
                        if isinstance(new_sid, str) and new_sid and new_sid != replaced[0]:
                            replaced = (replaced[0], new_sid)
                            self._last_sid = new_sid
                            for observer in observers:
                                observer.session_replaced(session_id=new_sid)
                    if replaced is not None and isinstance(sdk_ev, ResultMessage):
                        break  # the reset cycle's own result: consumed, not reported

                    # A failed result still carries real usage. The reader excludes
                    # its candidate before the failure selects recovery or exit.
                    if isinstance(sdk_ev, AssistantMessage) and sdk_ev.error:
                        assistant_error = classify_assistant_error(sdk_ev)
                        continue  # Read the terminal billing facts before raising.
                    if (
                        isinstance(sdk_ev, ResultMessage)
                        and sdk_ev.is_error
                        and sdk_ev.terminal_reason not in CANCELLED_REASONS
                    ):
                        stream.read(sdk_ev)
                        result_error = classify_result_error(sdk_ev)
                        raise assistant_error if isinstance(assistant_error, PermanentError) else result_error

                    # Every remaining message, the partial-message stream
                    # included (#1327), is read into observations here.
                    if isinstance(sdk_ev, AssistantMessage):
                        for block in sdk_ev.content:
                            if isinstance(block, ToolUseBlock) and block.name in ("Agent", "Task"):
                                pending = self._dispatch_inputs.setdefault(block.id, asyncio.get_running_loop().create_future())
                                if not pending.done():
                                    pending.set_result(dict(block.input))
                    stream.read(sdk_ev)
                    if isinstance(sdk_ev, ResultMessage) and assistant_error is not None:
                        raise assistant_error
                    if stream.completed:
                        outcome = {"kind": "turn_complete"}
                        break
                if assistant_error is not None:
                    raise assistant_error
                # Exhausted without a completed turn: a clean turn boundary
                # all the same.

            except _RetryableError as e:
                # The pause stays OPEN: the request ends while the provider is
                # still down, so the ceiling keeps accumulating.
                outcome = {"kind": "retryable", "last_error": str(e)}
            except PermanentError as e:
                outcome = {"kind": "permanent", "exc": e}
            except ResultError as exc:
                # The CLI reported an error result and exited before this
                # request read it (a refused resume during initialize, say);
                # its structured fields classify exactly as the ResultMessage
                # would, and an exit on a cancelled turn is that turn's end.
                if exc.terminal_reason in CANCELLED_REASONS:
                    outcome = {"kind": "cancelled"}
                elif isinstance(err := classify_result_error(exc), _RetryableError):
                    outcome = {"kind": "retryable", "last_error": str(err)}
                else:
                    outcome = {"kind": "permanent", "exc": err}
            except anyio.ClosedResourceError:
                # close() disconnected the client under this request, which
                # closes the SDK's receive stream beneath the consumer.
                if not self._closed:
                    raise
                outcome = {"kind": "closed"}
            except ClaudeSDKError as exc:
                # The CLI subprocess itself failed: an expected infrastructure
                # failure whose diagnostic is the stderr tail this attempt
                # captured (#1604, #2202).
                oom_after = confinement.memory_kills() if confinement is not None else None
                captured_stderr = "\n".join(stderr_tail)
                if captured_stderr:
                    logger.log(logging.ERROR, "[claude-stderr fatal]\n%s", captured_stderr)
                outcome = {
                    "kind": "infra",
                    "exc": _cli_failure(exc, captured_stderr, oom_before, oom_after),
                }
            except Exception:
                # Our own code inside the loop above, the caller's callback
                # included: a raw internal bug reaching the crash banner.
                self._close_pause(observers)
                raise
            keep_client = outcome["kind"] in ("turn_complete", "stream_exhausted", "replaced")
            if keep_client:
                try:
                    self._report_tool_denials(observers)
                except BaseException:
                    keep_client = False
                    raise


        # Finish the selected outcome.
        kind = outcome["kind"]
        if kind in ("turn_complete", "stream_exhausted", "cancelled"):
            self._close_pause(observers)
            if not stream.completed:
                # The input went unanswered: an infrastructure fault, named as
                # one rather than mistaken for a model that declined (#1693).
                raise BackendError("the provider ended the request without a completed turn")
            return Response(output=stream.output, text=stream.text)
        if kind == "replaced":
            if replaced is None or replaced[1] is None:
                # No message followed the reset: the successor id is unknown.
                self._terminated = True
                raise PermanentError(
                    PermanentError.Reason.SESSION_LOST,
                    provider_code="conversation_reset",
                    message="Claude reset the conversation and reported no successor",
                )
            self._facts = _replaced_facts(replaced[0], replaced[1])
            return None
        if kind == "retryable":
            if stream.completed and stream.result is not None and not stream.result.is_error:
                # Only a successful terminal result makes a later transport
                # error noise. A failed result still requires recovery.
                self._close_pause(observers)
                return Response(output=stream.output, text=stream.text)
            # Every non-conclusive error lands here (#1253): open the pause
            # once for the outage (the one latch the in-stream reject shares),
            # then terminate honestly at the configured ceiling or end this
            # request and open the next one with the recorded facts (#3047).
            if self._wait_started_at is None:
                self._wait_started_at = time.time()
                for observer in observers:
                    observer.rate_limit_paused()
            # Elapsed is real time from the outage's first wait, never this
            # Agent's pause-frozen Local Time; the ceiling is
            # ``llm.retry_ceiling_s`` and None waits forever, because a paused
            # run holding its GPU is recoverable and a terminated one is not
            # (owner decision, #2502). A pause freezes Local Time ONLY: Run
            # Time, and with it the exploration Run Budget, keeps advancing, so
            # inside the exploration window the Run Budget still expires the
            # paused Agent, and outside it, with the ceiling unset, the wait is
            # genuinely unbounded. Issue #2502's body says the wait costs no
            # run budget, and that part is not true.
            elapsed = time.time() - self._wait_started_at
            ceiling = self._retry_ceiling_s
            if ceiling is not None and elapsed >= ceiling:
                # The ceiling is the pause's own end: the resume closes the
                # bracket and the PermanentError below names the reason, this
                # Conversation's own, with the provider's last word as text.
                self._close_pause(observers)
                self._terminated = True
                raise PermanentError(
                    PermanentError.Reason.RETRY_CEILING,
                    message=outcome["last_error"],
                )
            self._interrupted_at = time.time()
            self._facts = _interrupted_facts(outcome["last_error"])
            return None
        # A conclusive provider answer or a dead CLI ends the pause too.
        self._close_pause(observers)
        if kind == "permanent":
            self._terminated = True
        if kind in ("permanent", "infra"):
            raise outcome["exc"]
        if kind == "closed":
            raise BackendError("conversation closed during the request")
        raise AssertionError(f"unreachable outcome kind={kind!r}")

    def _compose(self, message: list[str | Image]) -> tuple[str | Image, ...]:
        """Label prior input as context after interruption, retaining its text and images."""
        facts, self._facts = self._facts, None
        notes, self._notifications = self._notifications, []
        original = tuple(message)
        groups: list[tuple[str | Image, ...]] = [(facts,)] if facts else []
        groups.extend(notes)
        if original:
            if facts:
                first, *rest = original
                labelled = (_CONTEXT_HEADER + "\n" + first, *rest) if isinstance(first, str) else (_CONTEXT_HEADER + "\n", *original)
                groups.append(labelled)
            else:
                groups.append(original)
        if not groups:
            raise AssertionError("a provider request needs some input")
        # A later attempt carries the input as labelled context, not as a replay.
        message[:] = _join_content([*notes, *((original,) if original else ())])
        return _join_content(groups)

    def _make_stderr_callback(self, stderr_tail: deque[str]) -> Callable[[str], None]:
        """Retain a bounded per-attempt tail without publishing live stderr: it stays engineering-only (#51) and reaches the AgentError only when the attempt fails (#1604)."""
        def _cb(line: str) -> None:
            logger.debug("[claude-stderr] %s", line)
            stderr_tail.append(line[-_STDERR_LINE_MAX_CHARS:])
        return _cb
