"""Upload the shareable record of one finished run to the Kb.

Transport plus one generic archive rule: ordinary UTF-8 text under the run's
workspace and deliverable, and nothing else. That rule knows no product
feature -- no Search, Agent, Board, attempt, or package layout -- so a new role
or artifact needs no edit here, and a new kind of large binary is excluded
because it is not text, not because someone listed it.

The Kb stores those bytes as a black box and unpacks them for the summarizer
Agent, which inspects the directory itself. No manifest, schema, or hash
travels with the archive.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import io
import logging
import os
import signal
import tarfile
import time
from pathlib import Path
from typing import Literal, NoReturn

import requests

from engine.base import WorkflowBaseModel
from infra.kb_service import (
    KbRequestPayload,
    KbServiceConfigError,
    attach_request_payload_url,
    kb_contribute_url,
)

logger = logging.getLogger(__name__)

_CONTRIBUTION_DEADLINE_SECONDS = 300

# What a finished run may share: the executions it materialized and what it
# handed back. The run's private grading state, its Submitter records, its task
# data and its library cache are simply not walked.
_SHARED_ROOTS = ("workspace", "deliverable")
# Only text leaves the machine. A file with a NUL byte, one that fails UTF-8
# decode, or one over this bound is binary -- a weight in any format, an image,
# a compiled object -- and is not packed. There is no suffix blacklist, so the
# next weight format needs no edit.
_MAX_TEXT_BYTES = 32 * 1024 * 1024


def _raise_walk_error(error: OSError) -> NoReturn:
    raise error


def _shared_text_files(run_home: Path) -> "Iterator[tuple[str, bytes]]":
    """Every shareable text file of one run as ``(run-relative path, bytes)``.

    Symlinks are never followed, for directories or for files: the producer
    originals already live under ``workspace/``, so the Forum and Board links
    that point at them would only pack the same bytes twice, and a link is the
    one way a walk of these two roots could otherwise leave them.
    """
    run_home = Path(run_home).resolve()
    for name in _SHARED_ROOTS:
        root = run_home / name
        if root.is_symlink() or not root.is_dir():
            continue
        for current, dir_names, file_names in os.walk(
            root, onerror=_raise_walk_error
        ):
            dir_names[:] = sorted(
                child
                for child in dir_names
                if not (Path(current) / child).is_symlink()
            )
            for file_name in sorted(file_names):
                path = Path(current) / file_name
                if path.is_symlink() or not path.is_file():
                    continue
                if path.stat().st_size > _MAX_TEXT_BYTES:
                    continue
                raw = path.read_bytes()
                if b"\x00" in raw:
                    continue
                try:
                    raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                yield path.relative_to(run_home).as_posix(), raw


def pack_run(run_home: Path) -> bytes:
    """One run's shareable text as an ordinary gzip tar, at run-relative paths."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, raw in _shared_text_files(run_home):
            info = tarfile.TarInfo(name=name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    return buf.getvalue()


class ContributeError(RuntimeError):
    """The run could not be packaged, delivered, or acknowledged."""


def post_contribution_bytes(
    url: str,
    body: bytes,
    payload: KbRequestPayload,
) -> None:
    """POST gzip-tar bytes and their run payload to a /contribute URL.

    Returning is what "the service took it" means. Every other answer -- a transport error, a failed receipt, an unreadable body -- raises ``ContributeError`` carrying the reason, and ``upload_run`` turns that into this module's one answer.
    """
    try:
        resp = requests.post(
            attach_request_payload_url(url, payload),
            data=body,
            headers={"Content-Type": "application/gzip"},
            timeout=_CONTRIBUTION_DEADLINE_SECONDS,
        )
    except requests.RequestException as exc:
        raise ContributeError(
            f"Could not reach the contribution service at {url}: {exc}"
        ) from exc
    try:
        response = resp.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise ContributeError(
            "contribution service receipt must be valid JSON"
        ) from exc
    if response == {"status": "received"} and resp.ok:
        return
    if (
        isinstance(response, dict)
        and set(response) == {"status", "reason"}
        and response.get("status") == "failed"
        and isinstance(response.get("reason"), str)
        and response["reason"]
        and not resp.ok
    ):
        raise ContributeError(response["reason"])
    if not resp.ok:
        raise ContributeError(
            f"contribution service returned HTTP {resp.status_code}: {response!r}"
        )
    raise ContributeError(
        f"contribution service returned an invalid result: {response!r}"
    )


def _deadline_expired(_signum: int, _frame: object) -> None:
    raise ContributeError(
        f"Contribution did not finish within {_CONTRIBUTION_DEADLINE_SECONDS} seconds"
    )


@contextmanager
def _contribution_deadline() -> Iterator[None]:
    """Apply one wall-clock deadline to package, send, and receive."""
    old_handler = signal.signal(signal.SIGALRM, _deadline_expired)
    old_delay, old_interval = signal.setitimer(
        signal.ITIMER_REAL,
        _CONTRIBUTION_DEADLINE_SECONDS,
    )
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, old_delay, old_interval)
        signal.signal(signal.SIGALRM, old_handler)


class ContributionResult(WorkflowBaseModel):
    """What the Kb answered for one run's archive.

    ``received`` carries no reason; ``failed`` carries the one-line reason the caller shows. It is the answer to one upload and nothing else keeps it: a contribution is a convenience after a finished run, not a fact the run records."""

    status: Literal["received", "failed"]
    reason: str | None = None


def contribution_text(result: ContributionResult) -> str:
    """The one wording every caller shows a person for this outcome."""
    if result.status == "received":
        return "received"
    if result.reason is None:
        raise AssertionError("failed Contribution has no reason")
    return f"failed: {result.reason}"


def upload_run(run_home: Path, payload: KbRequestPayload) -> ContributionResult:
    """Pack and send one run within one wall-clock deadline, and say what happened.

    Contribution is a side effect of a finished run. Whatever goes wrong here -- packing, the service, or a product bug -- becomes a ``failed`` result with its reason, and the full traceback goes to the log. The automatic upload carries on and the run stays completed; the manual command reports the failure.
    """
    try:
        with _contribution_deadline():
            url = kb_contribute_url()
            started = time.monotonic()
            logger.info("Packing Contribution for %s", payload.run_id)
            body = pack_run(Path(run_home))
            logger.info(
                "Uploading Contribution: %d bytes, packed in %.2f seconds",
                len(body), time.monotonic() - started,
            )
            post_contribution_bytes(url, body, payload)
            logger.info("Contribution finished in %.2f seconds", time.monotonic() - started)
    except (ContributeError, KbServiceConfigError) as exc:
        result = ContributionResult(status="failed", reason=str(exc))
    except Exception as exc:  # noqa: BLE001 -- owner rule: Contribution never fails a completed run
        logger.exception("Contribution failed with a product error")
        result = ContributionResult(
            status="failed", reason=f"{type(exc).__name__}: {exc}"
        )
    else:
        result = ContributionResult(status="received")
    logger.info("Contribution %s", contribution_text(result))
    return result
