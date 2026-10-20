"""Local Anthropic-compatible CLIProxyAPI process for GPT routing."""

from __future__ import annotations

import json
import atexit
import contextlib
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from aibuildai_conversation.assets import proxy_path

_READY_TIMEOUT_S = 10.0
_READY_POLL_S = 0.05
_ANTHROPIC_COMPAT_TOKEN = "aibuildai-openai-proxy"
_PROCESS_LOG = "process.log"
_PROCESS_OUTPUT_TAIL_BYTES = 4096
_PASSTHROUGH_ENV = (
    "PATH",
    "HOME",
    "XDG_CONFIG_HOME",
    "XDG_STATE_HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "SSL_CERT_FILE",
    "NODE_EXTRA_CA_CERTS",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)


def _codex_auth() -> dict[str, str]:
    path = Path.home() / ".codex" / "auth.json"
    try:
        text = path.read_text()
    except FileNotFoundError as exc:
        raise ValueError(f"OpenAI subscription login was not found at {path}") from exc
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"OpenAI subscription login at {path} could not be read"
        ) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"OpenAI subscription login at {path} is not valid JSON"
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError(f"OpenAI subscription login at {path} is not an object")
    tokens = raw.get("tokens")
    if not isinstance(tokens, dict):
        raise ValueError(f"OpenAI subscription login at {path} has no `tokens` object")
    auth = {
        key: value.strip()
        for key in ("access_token", "refresh_token", "account_id", "id_token")
        if isinstance((value := tokens.get(key)), str) and value.strip()
    }
    missing = [key for key in ("access_token", "refresh_token") if key not in auth]
    if missing:
        raise ValueError(
            f"OpenAI subscription login at {path} is missing "
            f"{', '.join(missing)} in `tokens`"
        )
    return {"type": "codex", **auth}


def _passthrough_env() -> dict[str, str]:
    return {
        key: value
        for key in _PASSTHROUGH_ENV
        if (value := os.environ.get(key)) is not None
    }


def _write_cliproxyapi_auth(auth_dir: Path) -> None:
    auth_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    (auth_dir / "codex.json").write_text(json.dumps(_codex_auth()))


def _write_cliproxyapi_config(
    *, port: int, model: str, config_root: Path, api_key: str | None = None,
    api_base_url: str | None = None, extra_models: Sequence[str] = (),
    request_overrides: Mapping[str, object] | None = None,
) -> Path:
    config_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    auth_dir = config_root / "auths"
    models = [{"name": name, "alias": name, "force-mapping": True}
              for name in dict.fromkeys((model, *extra_models))]
    config: dict[str, object] = {
        "host": "127.0.0.1", "port": port, "auth-dir": str(auth_dir),
        "api-keys": [_ANTHROPIC_COMPAT_TOKEN], "debug": False,
        "logging-to-file": False, "disable-image-generation": True,
        "disable-cooling": True,
        "remote-management": {"secret-key": "", "disable-control-panel": True,
                              "disable-auto-update-panel": True},
    }
    if api_key is None:
        _write_cliproxyapi_auth(auth_dir)
        config["oauth-model-alias"] = {"codex": models}
    else:
        auth_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        if api_base_url is None:
            config["codex-api-key"] = [{"api-key": api_key,
                "base-url": "https://api.openai.com/v1", "models": models}]
        else:
            config["openai-compatibility"] = [{"name": "custom", "base-url": api_base_url,
                "disable-cooling": True, "api-key-entries": [{"api-key": api_key}], "models": models}]
    if request_overrides:
        if api_base_url is None:
            raise ValueError("request overrides require a custom OpenAI-compatible URL")
        config["payload"] = {"override": [{
            "models": [{"name": entry["name"], "protocol": "openai", "from-protocol": "claude"} for entry in models],
            "params": dict(request_overrides),
        }]}
    config_path = config_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def _output_tail(output_path: Path | None) -> str:
    if output_path is None or not output_path.exists():
        return ""
    raw = output_path.read_bytes()
    if not raw:
        return ""
    return raw[-_PROCESS_OUTPUT_TAIL_BYTES:].decode(errors="replace").strip()


def _with_output_tail(message: str, output_path: Path | None) -> str:
    tail = _output_tail(output_path)
    if not tail:
        return message
    return f"{message}; output tail:\n{tail}"


def _wait_until_ready(
    base_url: str,
    process: subprocess.Popen[str],
    *,
    expected_model: str | None = None,
    output_path: Path | None = None,
) -> None:
    deadline = time.monotonic() + _READY_TIMEOUT_S
    health_url = f"{base_url}/healthz"
    models_url = f"{base_url}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ValueError(
                _with_output_tail(
                    "CLIProxyAPI exited before the OpenAI proxy became ready",
                    output_path,
                )
            )
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if response.status != 200:
                    time.sleep(_READY_POLL_S)
                    continue
            if expected_model is None:
                return
            request = urllib.request.Request(
                models_url,
                headers={"Authorization": f"Bearer {_ANTHROPIC_COMPAT_TOKEN}"},
            )
            with urllib.request.urlopen(request, timeout=1.0) as response:
                body = json.loads(response.read().decode())
                model_ids = {str(item.get("id")) for item in body.get("data", [])}
                if expected_model in model_ids:
                    return
        except urllib.error.URLError:
            time.sleep(_READY_POLL_S)
            continue
        time.sleep(_READY_POLL_S)
    raise ValueError(
        _with_output_tail(
            f"CLIProxyAPI did not register {expected_model or 'healthz'} at {base_url} "
            f"within {_READY_TIMEOUT_S:.0f}s",
            output_path,
        )
    )


class OpenAIProxy:
    """Own one local CLIProxyAPI process."""

    def __init__(
        self,
        *,
        model: str,
        log_home: "str | os.PathLike[str]",
        api_key: str | None = None,
        extra_models: Sequence[str] = (),
        api_base_url: str | None = None,
        request_overrides: Mapping[str, object] | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.request_overrides = dict(request_overrides or {})
        self.extra_models = tuple(extra_models)
        self._process: subprocess.Popen | None = None
        self._scratch_dir: Path | None = None
        self._log_home = Path(log_home)
        self._stopped = False

    def set_run_home(self, run_home: "str | os.PathLike[str]") -> None:
        self._log_home = Path(run_home)

    def start(self) -> str:
        """Start the proxy, wait until it serves the model, and return its URL."""
        if self._process is not None:
            raise AssertionError("OpenAI proxy was already started")
        scratch_dir = Path(tempfile.mkdtemp(prefix="aibuildai-openai-proxy-"))
        self._scratch_dir = scratch_dir
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                port = int(sock.getsockname()[1])
            config_path = _write_cliproxyapi_config(
                port=port,
                model=self.model,
                api_key=self.api_key,
                config_root=scratch_dir,
                extra_models=self.extra_models,
                api_base_url=self.api_base_url,
                request_overrides=self.request_overrides,
            )
            log_path = scratch_dir / _PROCESS_LOG
            with log_path.open("w") as out:
                self._process = subprocess.Popen(
                    [
                        str(proxy_path()),
                        "-config",
                        str(config_path),
                    ],
                    env=_passthrough_env(),
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    start_new_session=True,
                )
            atexit.register(self.stop)
            base_url = f"http://127.0.0.1:{port}"
            _wait_until_ready(
                base_url,
                self._process,
                expected_model=self.model,
                output_path=log_path,
            )
            return base_url
        except BaseException:
            self.stop()
            raise

    def _preserve_log(self) -> None:
        """Move process.log to a numbered immutable file under <log_home>/openai-proxy/ (#2282).

        The scratch dir starts in /tmp because the adapter starts before any run directory exists. log_home is required at construction and starts at the task output base, so the log survives EVERY stop -- including a memorize session and a failure before the run home is born (#2362); run/resume refine it to the run home through set_run_home once that directory exists.
        """
        scratch = self._scratch_dir
        if scratch is None:
            return
        src = scratch / _PROCESS_LOG
        if not src.exists():
            return
        log_dir = self._log_home / "openai-proxy"
        log_dir.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while True:
            dst = log_dir / f"openai-proxy-{attempt:04d}.log"
            try:
                fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                attempt += 1
                continue
            os.close(fd)
            break
        shutil.move(str(src), str(dst))

    def stop(self) -> None:
        """Stop the local proxy, preserve the log, and remove credentials."""
        if self._stopped:
            return
        process = self._process
        if process is not None and process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        self._preserve_log()
        if self._scratch_dir is not None:
            shutil.rmtree(self._scratch_dir)
            self._scratch_dir = None
        self._stopped = True


# Model ids the ChatGPT/Codex SUBSCRIPTION route can serve -- its static plan
# catalog. The gpt-5.4/5.5 tier mirrors
# submodules/cli-proxy-api/internal/registry/models/models.json (codex-pro/plus);
# the gpt-5.6 tier is registered via WithCodexBuiltins in that submodule's
# model_definitions.go (a models.json data entry would be clobbered by the
# remote model updater). The api-key route is NOT constrained by this: the
# check lives inside the subscription builder, so registering an unservable
# alias on that route cannot be written.
_SUBSCRIPTION_SERVABLE_MODELS: frozenset[str] = frozenset(
    {
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
    }
)


def build_openai_subscription_proxy(
    model: str,
    log_home: "str | os.PathLike[str]",
    extra_models: Sequence[str] = (),
) -> OpenAIProxy:
    unservable = [
        m for m in (model, *extra_models) if m not in _SUBSCRIPTION_SERVABLE_MODELS
    ]
    if unservable:
        raise ValueError(
            f"OpenAI model id(s) {unservable} are not servable by the "
            f"ChatGPT/Codex subscription, which serves only "
            f"{sorted(_SUBSCRIPTION_SERVABLE_MODELS)}. Use the "
            "OpenAI API-key route (export "
            "AIBUILDAI_API_KEY/OPENAI_API_KEY) for other ids."
        )
    return OpenAIProxy(
        model=model, log_home=log_home, extra_models=extra_models
    )


def anthropic_compat_token() -> str:
    return _ANTHROPIC_COMPAT_TOKEN
