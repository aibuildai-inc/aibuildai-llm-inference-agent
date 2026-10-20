"""Select the product's endpoint and report startup diagnostics."""

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from config import LlmConfig, endpoint_disallowed_tools
from aibuildai_conversation.model_catalog import endpoint_is_anthropic
import aibuildai_conversation.model_endpoint as endpoints
from aibuildai_conversation.model_endpoint import OpenAIProxy

_USER_VAR_TO_INTERNAL = {
    "AIBUILDAI_SMALL_FAST_MODEL": "ANTHROPIC_SMALL_FAST_MODEL",
    "AIBUILDAI_HAIKU_MODEL": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "AIBUILDAI_SONNET_MODEL": "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "AIBUILDAI_OPUS_MODEL": "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "AIBUILDAI_SUBAGENT_MODEL": "CLAUDE_CODE_SUBAGENT_MODEL",
}


@contextmanager
def model_endpoint(llm: LlmConfig, log_home: Path) -> Iterator[tuple[dict[str, str], OpenAIProxy | None]]:
    """Select one Run endpoint; the shared scope owns any adapter process."""
    provider = llm.endpoint_provider
    base = os.environ.get("AIBUILDAI_BASE_URL", "").strip()
    if not base and provider != "openai":
        base = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    token = (os.environ.get("AIBUILDAI_API_KEY", "").strip()
             or os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
             or os.environ.get("ANTHROPIC_API_KEY", "").strip() or None)
    if not base and provider == "openrouter":
        base = "https://openrouter.ai/api"
    if not base and provider == "deepseek":
        base = "https://api.deepseek.com/anthropic"
    if provider != "anthropic" and provider != "openai" and not base:
        raise ValueError(f"provider {provider!r} requires AIBUILDAI_BASE_URL")
    if base and not token:
        raise ValueError("custom endpoint requires an auth token")
    protocol = "openai" if provider == "openai" and not base else "anthropic"
    if protocol == "openai":
        token = os.environ.get("AIBUILDAI_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip() or None
        authentication = "api_key" if token else "subscription"
    else:
        authentication = ("api_key" if endpoint_is_anthropic(base) else "bearer") if token else "login"
    extra = tuple(llm.auto.allowed_models or ()) if llm.auto is not None else ()
    with endpoints.open_model_endpoint(protocol=protocol, authentication=authentication,
            model=llm.default.model, log_home=log_home, base_url=base or None,
            credential=token, extra_models=extra) as (env, proxy):
        for user, internal in _USER_VAR_TO_INTERNAL.items():
            if value := os.environ.get(user):
                env[internal] = value
            elif internal in os.environ:
                env[internal] = os.environ[internal]
        if provider == "openai":
            for internal in _USER_VAR_TO_INTERNAL.values():
                env.setdefault(internal, llm.default.model)
        if provider == "deepseek" and not env.get("ANTHROPIC_SMALL_FAST_MODEL"):
            env["ANTHROPIC_SMALL_FAST_MODEL"] = "deepseek-v4-flash"
        host = (urlparse(env["ANTHROPIC_BASE_URL"]).hostname or "").lower()
        if provider == "anthropic" and any(host == known or host.endswith("." + known) for known in ("deepseek.com", "openrouter.ai")):
            print(f"[endpoint] warning: Claude model ids will be sent to {host!r}; check the configured URL.", file=sys.stderr)
        if denied := endpoint_disallowed_tools(llm, env):
            print(f"[endpoint] warning: this endpoint does not serve {sorted(denied)}; agents use Bash for web access.", file=sys.stderr)
        yield env, proxy
