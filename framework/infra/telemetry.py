"""Opt-out telemetry for tracking CLI usage metrics via PostHog.

Consent model:
- Default: enabled.
- Disable: ``export AIBUILDAI_TELEMETRY=0`` before running aibuildai. Any other value (unset, empty, ``1``, ``true``, anything else) leaves telemetry enabled.

No interactive prompt, no stored consent state, no CLI flag. The env var is the single switch; this makes the model trivially auditable ("did the user disable it?" → "is ``$AIBUILDAI_TELEMETRY`` set to ``0``?") and prevents the pre-2026-05-15 footgun where a startup call silently overwrote any persisted preference.

The bundled PostHog project key is a public ingest key intended to ship. It accepts events; it does not grant account control.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import time
import uuid
from pathlib import Path

import posthog

from aibuildai_version import APP_VERSION

logger = logging.getLogger(__name__)


def _config_dir() -> Path:
    return Path.home() / ".aibuildai"


def _identity_file() -> Path:
    return _config_dir() / "identity.json"

_POSTHOG_API_KEY = "phc_QmEXL6EnDFAwolxY5gz9Csr82zDJqgjRwXX3XSmx3iq"

# Configure PostHog SDK -- minimal data only, no device/OS/IP fingerprinting.
# Server-side Property Filter transformation strips $os, $ip, etc. at ingestion.
posthog.api_key = _POSTHOG_API_KEY
posthog.host = "https://us.i.posthog.com"
posthog.debug = False
posthog.disable_geoip = True


def _shutdown_posthog():
    try:
        posthog.shutdown()
    except Exception:
        logger.exception("telemetry: posthog.shutdown failed")


atexit.register(_shutdown_posthog)


def _load_identity() -> dict:
    identity_file = _identity_file()
    if identity_file.exists():
        return json.loads(identity_file.read_text())
    return {}


def _save_identity(data: dict) -> None:
    config_dir = _config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    _identity_file().write_text(json.dumps(data, indent=2))


def _get_or_create_user_id() -> str:
    """Get or create a persistent anonymous user ID. Sends user_registered on first creation."""
    data = _load_identity()
    uid = data.get("user_id")
    if uid:
        return uid
    uid = str(uuid.uuid4())
    data["user_id"] = uid
    data.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _save_identity(data)
    # Fire user_registered event directly (can't use track() here to avoid recursion)
    posthog.capture(distinct_id=uid, event="user_registered", properties={"cli_version": APP_VERSION}, )
    return uid


def is_enabled() -> bool:
    """Telemetry is enabled unless ``AIBUILDAI_TELEMETRY=0`` is set."""
    return os.environ.get("AIBUILDAI_TELEMETRY") != "0"


def track(event: str, *, flush: bool = False, **properties: object) -> None:
    """Track an event via PostHog. No-op if telemetry is disabled.

    Args:
        flush: If True, flush the PostHog queue immediately (use for events
               sent right before process exit).
    """
    if not is_enabled():
        return
    posthog.capture(
        distinct_id=_get_or_create_user_id(),
        event=event,
        properties={
            "cli_version": APP_VERSION,
            **properties,
        },
        disable_geoip=True,
    )
    if flush:
        posthog.flush()
