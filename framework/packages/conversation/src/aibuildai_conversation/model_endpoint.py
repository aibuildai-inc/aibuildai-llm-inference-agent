"""Open one explicitly selected model endpoint without changing process environment."""

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from aibuildai_conversation.model_catalog import endpoint_is_anthropic
from aibuildai_conversation._impl import openai_proxy as local
from aibuildai_conversation._impl.openai_proxy import OpenAIProxy

_PROVIDER_ENV = (
    "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK",
)
_SAMPLING_FIELDS = frozenset({
    "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty",
    "thinking_token_budget", "max_tokens", "chat_template_kwargs.enable_thinking", "guided_decoding", "structured_outputs.structural_tag",
})


@contextmanager
def open_model_endpoint(
    *, protocol: Literal["anthropic", "openai"],
    authentication: Literal["login", "api_key", "bearer", "subscription"],
    model: str, log_home: Path, base_url: str | None = None, credential: str | None = None,
    extra_models: Sequence[str] = (), request_overrides: Mapping[str, object] | None = None,
) -> Iterator[tuple[dict[str, str], OpenAIProxy | None]]:
    """Keep one selected endpoint alive for the caller's scope and yield its environment and owned proxy."""
    if not model:
        raise ValueError("the endpoint requires a model")
    if base_url is not None:
        url = urlsplit(base_url)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) URL without credentials, query, or fragment")
        base_url = base_url.rstrip("/")
    if protocol not in ("anthropic", "openai"):
        raise ValueError(f"unsupported endpoint protocol {protocol!r}")
    allowed_auth = ("login", "api_key", "bearer") if protocol == "anthropic" else ("api_key", "subscription")
    if authentication not in allowed_auth:
        raise ValueError(f"{authentication!r} is not supported for {protocol}")
    if authentication in ("login", "subscription") and (credential is not None or base_url is not None):
        raise ValueError("native login cannot be combined with an explicit credential or URL")
    if authentication in ("api_key", "bearer") and (not isinstance(credential, str) or not credential.strip()):
        raise ValueError("the selected authentication requires a credential")
    overrides = dict(request_overrides or {})
    if overrides and (protocol != "openai" or base_url is None):
        raise ValueError("sampling overrides require a custom OpenAI-compatible endpoint")
    if unknown := overrides.keys() - _SAMPLING_FIELDS:
        raise ValueError(f"unsupported sampling overrides: {sorted(unknown)}")
    numeric = overrides.keys() - {"chat_template_kwargs.enable_thinking", "guided_decoding", "structured_outputs.structural_tag"}
    for name in numeric:
        value = overrides[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    for name in ("top_k", "thinking_token_budget", "max_tokens"):
        if name in overrides and type(overrides[name]) is not int:
            raise ValueError(f"{name} must be an integer")
    max_tokens = overrides.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if "chat_template_kwargs.enable_thinking" in overrides and type(overrides["chat_template_kwargs.enable_thinking"]) is not bool:
        raise ValueError("enable_thinking must be a boolean")
    if "guided_decoding" in overrides and "structured_outputs.structural_tag" in overrides:
        raise ValueError("specify one structural-tag control, not both provider forms")
    if "structured_outputs.structural_tag" in overrides:
        tag = overrides["structured_outputs.structural_tag"]
        if not isinstance(tag, str) or not isinstance(json.loads(tag), dict):
            raise ValueError("structured_outputs.structural_tag requires a JSON object string")
    if "guided_decoding" in overrides:
        guided = overrides["guided_decoding"]
        if not isinstance(guided, dict) or set(guided) != {"structural_tag"} or not isinstance(guided["structural_tag"], str):
            raise ValueError("guided_decoding requires a structural_tag JSON string")
        if not isinstance(json.loads(guided["structural_tag"]), dict):
            raise ValueError("structural_tag must describe a JSON object")
    env: dict[str, str] = dict.fromkeys(_PROVIDER_ENV, "")
    proxy = None
    try:
        if protocol == "openai":
            if authentication == "subscription":
                proxy = local.build_openai_subscription_proxy(model, log_home, extra_models)
            else:
                proxy = local.OpenAIProxy(model=model, log_home=log_home, api_key=credential,
                    api_base_url=base_url, extra_models=extra_models, request_overrides=overrides)
            env["ANTHROPIC_BASE_URL"] = proxy.start()
            env["ANTHROPIC_AUTH_TOKEN"] = local.anthropic_compat_token()
            for name in ("ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_SMALL_FAST_MODEL"):
                env[name] = model
            if "max_tokens" in overrides:
                env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(overrides["max_tokens"])
        else:
            env["ANTHROPIC_BASE_URL"] = base_url or ""
            if authentication == "login":
                env.pop("CLAUDE_CODE_OAUTH_TOKEN")
            else:
                env["ANTHROPIC_API_KEY" if authentication == "api_key" else "ANTHROPIC_AUTH_TOKEN"] = credential or ""
        if env["ANTHROPIC_BASE_URL"] and not endpoint_is_anthropic(env["ANTHROPIC_BASE_URL"]):
            env["CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK"] = "1"
        yield env, proxy
    finally:
        if proxy is not None:
            proxy.stop()


__all__ = ('open_model_endpoint', 'OpenAIProxy')
