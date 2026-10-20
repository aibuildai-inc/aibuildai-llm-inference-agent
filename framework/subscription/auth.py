"""Clerk login / logout / account-management commands.

Login flow: OAuth 2.0 Authorization Code with PKCE.
- CLI generates a per-login PKCE verifier+challenge pair (RFC 7636)
- Spawns a one-shot HTTP listener on a random loopback port
- Opens the system browser to Clerk's hosted sign-in page
- Captures the authorization code on the redirect, exchanges for access_token / refresh_token / id_token
- Persists id_token + refresh_token to ~/.aibuildai/auth.json (0600)

`aibuildai login --no-browser` is the headless variant for a machine with no browser: instead of the loopback listener, it prints the authorize URL for the user to open on any device and reads back the `<code>:<state>` string the redirect page shows, then runs the same code-for-tokens exchange. This is NOT OAuth Device Flow (Clerk does not support it) -- it reuses the same authorization_code+PKCE flow and only swaps how the code returns to the CLI.

The session module verifies the id_token locally with Clerk's public key. The access_token is unused after the immediate /oauth/userinfo lookup at login time. It carries no plan info and cannot authorize a run.

Endpoints, client_id (publishable key), and the redirect URL whitelist must be configured in the Clerk dashboard before this works.
"""
from __future__ import annotations

import base64
import errno
import hashlib
import http.server
import os
import secrets
import shutil
import subprocess
import threading
import urllib.parse
import webbrowser
from datetime import datetime, timezone
from typing import Optional

import jwt
import requests

from aibuildai_conversation.account import account_email
from subscription import run_grant, session, store
from subscription.clerk_config import (
    CLERK_ACCOUNT_PORTAL,
    CLERK_CLI_PASTE_REDIRECT,
    CLERK_FRONTEND_API,
    CLERK_LOGIN_REDIRECT_PORT,
    CLERK_OAUTH_CLIENT_ID,
)


_AUTHORIZE_PATH = "/oauth/authorize"
_TOKEN_PATH = "/oauth/token"

_LOGIN_TIMEOUT_SECONDS = 300
_HTTP_TIMEOUT_SECONDS = 10

# How long to watch a browser launcher before deciding it (likely) opened. A
# failing launcher exits well within this window (WSL2's `gio open` errors in
# milliseconds); a real GUI open stays running past it.
_BROWSER_LAUNCH_PROBE_SECONDS = 1.5


class LoginError(Exception):
    """Raised when the browser-based login flow fails to complete."""


# What a returning sign-in whose state does not match this request is told.
#
# The CHECK is unchanged and still refuses the code: only the report changed.
# This used to be an AssertionError, which the command line reads as an internal
# defect and answers with "this is a bug in aibuildai" and exit 70. That is the
# wrong thing to say to someone whose fix is to run one command again. The
# message names no cause: the same mismatch is produced by a tab left open too
# long, a second sign-in started in another window, and a replayed redirect, and
# guessing between them in the message would either alarm a user who did nothing
# unusual or reassure one who should look closer. It also blames no one -- the
# user did nothing wrong in the ordinary case, which is nearly every case.
_STATE_MISMATCH_MESSAGE = (
    "Login did not complete: the sign-in that came back does not match the one "
    "this command started. Run `aibuildai login` again to start a new sign-in."
)


def _can_open_gui_browser() -> bool:
    """Report whether the host has a graphical browser worth launching.

    On a headless / SSH host with no DISPLAY, the stdlib `webbrowser` module still registers text-mode browsers (lynx / w3m / links / www-browser, and the elinks controller) that open in the FOREGROUND and block on the terminal. Launching one hijacks the TTY, and a text browser cannot render the JavaScript-only OAuth sign-in page anyway -- so opening it is not a degraded path, it is a dead end that also buries the fallback URL we just printed. The distinguishing property is whether the resolved browser opens in the background (non-blocking):

    - `BackgroundBrowser` (the graphical Unix browsers) -> background.
    - A bare `GenericBrowser` (lynx / w3m / links / www-browser and any `BROWSER=` override) -> foreground text browser.
    - The `UnixBrowser` family declares a `background` flag: the graphical ones (Mozilla / Chrome / ...) set it True; the text `Elinks` sets it False.
    - Platform controllers (`MacOSXOSAScript` / `WindowsDefault`) open a GUI and expose no `background` flag -> treated as graphical.

    A host with no browser at all raises `webbrowser.Error`, reported here as "cannot open".
    """
    try:
        controller = webbrowser.get()
    except webbrowser.Error:
        return False
    # `BackgroundBrowser` is itself a `GenericBrowser` subclass, so check it
    # before the bare-`GenericBrowser` (foreground) case.
    if isinstance(controller, webbrowser.BackgroundBrowser):
        return True
    if isinstance(controller, webbrowser.GenericBrowser):
        return False
    return getattr(controller, "background", True) is not False


def _is_wsl() -> bool:
    """True on WSL (Windows Subsystem for Linux).

    The WSL kernel writes a "microsoft" marker into /proc/version; matching it case-insensitively is the standard WSL probe. A missing / unreadable file (a non-Linux host, or a locked-down environment) simply means "not WSL".
    """
    try:
        # The WSL kernel writes a "microsoft" marker into /proc/version.
        with open("/proc/version", encoding="utf-8") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


def _browser_opener_argv(url: str) -> Optional[list[str]]:
    """Resolve the command that opens `url` in the user's default GUI browser, or None when no suitable launcher is on PATH.

    On WSL2 the Linux openers (`gio open` / `xdg-open`) start successfully then fail milliseconds later ("Operation not supported") because there is no Linux desktop portal -- so the Windows-interop launchers are preferred there, in order: `wslview`, then `powershell.exe Start-Process`. Everywhere else the standard Linux openers are used (`xdg-open`, then `gio open`). Only launchers actually present on PATH are returned; the caller shows the URL regardless.
    """
    if _is_wsl():
        wslview = shutil.which("wslview")
        if wslview:
            return [wslview, url]
        powershell = shutil.which("powershell.exe")
        if powershell:
            # Single-quote the URL so PowerShell reads it literally (the query
            # string carries `&` / `=` / `?`); Clerk URLs never contain a quote.
            return [powershell, "-NoProfile", "-Command", f"Start-Process '{url}'"]
        # No Windows interop available -- fall through to the Linux openers.
    xdg_open = shutil.which("xdg-open")
    if xdg_open:
        return [xdg_open, url]
    gio = shutil.which("gio")
    if gio:
        return [gio, "open", url]
    return None


def _try_open_browser(url: str) -> bool:
    """Best-effort launch of a GUI browser at `url`; report whether it (likely) opened.

    THE URL IS THE PRIMARY ARTIFACT -- every caller prints it first, and this is a silent best-effort attempt. Two properties make it safe and honest:

    - No child output can reach the user's terminal: the launcher runs with stdout / stderr redirected to DEVNULL and in its own session, so a launcher that fails on WSL2 (`gio open` prints "Operation not supported" and exits nonzero milliseconds later) can never garble what the user is reading.
    - The success report is verified, not assumed: a failing launcher exits promptly, so a nonzero exit within a short probe window means "did not open"; a launcher still running past the window is a real GUI open (treated as launched). We never claim success on an unverified API return.

    `_can_open_gui_browser` still gates this so a foreground text browser (lynx / w3m) is never spawned to hijack the terminal, and a host with no browser at all short-circuits before any launch.
    """
    if not _can_open_gui_browser():
        return False
    argv = _browser_opener_argv(url)
    if argv is None:
        return False
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False
    try:
        return proc.wait(timeout=_BROWSER_LAUNCH_PROBE_SECONDS) == 0
    except subprocess.TimeoutExpired:
        return True


def _generate_pkce_pair() -> tuple[str, str]:
    """Return (verifier, challenge). Verifier stays in CLI, challenge goes to Clerk in the authorize request. After Clerk redirects back with a code, the CLI sends the original verifier to prove it owns this flow.
    """
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


class _OneShotCallbackHandler(http.server.BaseHTTPRequestHandler):
    """Captures the ?code=... query param from Clerk's redirect, then closes. The captured code is stored on the class for the main thread to read.
    """

    captured_code: Optional[str] = None
    captured_state: Optional[str] = None
    captured_error: Optional[str] = None
    captured_error_description: Optional[str] = None

    def do_GET(self) -> None:
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _OneShotCallbackHandler.captured_code = params.get("code", [None])[0]
        _OneShotCallbackHandler.captured_state = params.get("state", [None])[0]
        _OneShotCallbackHandler.captured_error = params.get("error", [None])[0]
        _OneShotCallbackHandler.captured_error_description = params.get(
            "error_description", [None]
        )[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        if _OneShotCallbackHandler.captured_code:
            body = (
                "<h1>Logged in.</h1>"
                "<p>You can close this tab and return to your terminal.</p>"
            )
        else:
            err = _OneShotCallbackHandler.captured_error or "Unknown error"
            desc = _OneShotCallbackHandler.captured_error_description
            detail = f"{err}: {desc}" if desc else err
            body = (
                "<h1>Login failed.</h1>"
                f"<p>{detail}</p>"
                "<p>Return to your terminal for the same message; "
                "it will help diagnose the issue.</p>"
            )
        self.wfile.write(body.encode())

    def log_message(self, *_: object) -> None:
        pass


def _reset_callback_capture() -> None:
    _OneShotCallbackHandler.captured_code = None
    _OneShotCallbackHandler.captured_state = None
    _OneShotCallbackHandler.captured_error = None
    _OneShotCallbackHandler.captured_error_description = None


def _start_callback_server() -> tuple[int, http.server.HTTPServer]:
    try:
        server = http.server.HTTPServer(
            ("127.0.0.1", CLERK_LOGIN_REDIRECT_PORT),
            _OneShotCallbackHandler,
        )
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        raise LoginError(
            "Could not start the local login callback server because "
            f"127.0.0.1:{CLERK_LOGIN_REDIRECT_PORT} is already in use. "
            "Unset AIBUILDAI_CLERK_LOGIN_REDIRECT_PORT to let the OS choose "
            "a free loopback port, or set it to another allowed free port."
        ) from exc
    return int(server.server_address[1]), server


def _exchange_code_for_tokens(code: str, verifier: str, redirect_uri: str) -> dict:
    """Run the OAuth code-for-tokens exchange with PKCE.

    Public client: no client_secret, just code + verifier. Confidential client (dev escape hatch): AIBUILDAI_OAUTH_CLIENT_SECRET env var sends a Basic auth header instead.
    """
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
    }
    secret = os.environ.get("AIBUILDAI_OAUTH_CLIENT_SECRET")
    if secret:
        token_resp = requests.post(
            f"{CLERK_FRONTEND_API}{_TOKEN_PATH}",
            data=form,
            auth=(CLERK_OAUTH_CLIENT_ID, secret),
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
    else:
        form["client_id"] = CLERK_OAUTH_CLIENT_ID
        token_resp = requests.post(
            f"{CLERK_FRONTEND_API}{_TOKEN_PATH}",
            data=form,
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
    token_resp.raise_for_status()
    return token_resp.json()


def _authorize_query(redirect_uri: str, challenge: str, state: str) -> str:
    """Build the `/oauth/authorize` query string for the PKCE flow.

    Shared by both the loopback and the no-browser paste path -- the ONLY difference between them is the redirect_uri, so the rest of the authorize request (scopes, PKCE challenge, CSRF state) lives here once.
    """
    return urllib.parse.urlencode(
        {
            "client_id": CLERK_OAUTH_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            # `openid` triggers OIDC and makes Clerk return an id_token JWT
            # alongside the access_token. `public_metadata` is required
            # for Clerk to include user.public_metadata as a claim in the
            # id_token -- that is how `whoami` sees the plan written by the
            # Cloudflare webhook. `offline_access` is the
            # standard OIDC scope for receiving a refresh_token. All five
            # scopes must ALSO be enabled on the OAuth Application in the
            # Clerk dashboard, otherwise Clerk responds with
            # `invalid_scope`.
            "scope": "openid profile email public_metadata offline_access",
        }
    )


def _login_by_loopback(challenge: str, csrf_state: str) -> tuple[str, str]:
    """Capture the authorization code via a one-shot loopback HTTP listener.

    Spawns a localhost server, opens the system browser to Clerk, waits for the redirect, validates the CSRF state, and returns (code, redirect_uri). The default login path for a machine that has a browser.
    """
    _reset_callback_capture()

    port, server = _start_callback_server()
    redirect_uri = f"http://127.0.0.1:{port}/callback"

    server_thread = threading.Thread(
        target=server.handle_request, name="clerk-callback", daemon=True
    )
    server_thread.start()

    authorize_url = (
        f"{CLERK_FRONTEND_API}{_AUTHORIZE_PATH}"
        f"?{_authorize_query(redirect_uri, challenge, csrf_state)}"
    )

    # Print the URL first and on its own line so it is visible no matter what
    # follows -- then attempt the graphical browser. On a headless host
    # `_try_open_browser` opens nothing (it never launches a text browser), so
    # the fallback line points the user at the reliable headless paths instead.
    print(f"To sign in, open this URL in a browser:\n\n  {authorize_url}\n")
    if _try_open_browser(authorize_url):
        print("Opening it in your browser now...")
    else:
        print(
            "Could not open a browser on this machine -- open the URL above on "
            "any device to sign in. This terminal is waiting for the sign-in to "
            "complete; over SSH, either forward the callback port "
            f"(ssh -L {port}:127.0.0.1:{port} ...) or, more simply, cancel and "
            "run `aibuildai login --no-browser` to paste the code back instead."
        )

    server_thread.join(timeout=_LOGIN_TIMEOUT_SECONDS)
    server.server_close()

    code = _OneShotCallbackHandler.captured_code
    received_state = _OneShotCallbackHandler.captured_state

    if not code:
        err = _OneShotCallbackHandler.captured_error or "timeout or cancelled"
        desc = _OneShotCallbackHandler.captured_error_description
        full = f"{err} ({desc})" if desc else err
        raise LoginError(f"Login did not complete: {full}")

    if received_state != csrf_state:
        raise LoginError(_STATE_MISMATCH_MESSAGE)
    return code, redirect_uri


def _login_by_paste(challenge: str, csrf_state: str) -> tuple[str, str]:
    """Headless login: print the authorize URL, read back the pasted code.

    For a machine with no browser. The redirect goes to a static Cloudflare Worker page (CLERK_CLI_PASTE_REDIRECT) that renders the code as `<code>:<state>` for the user to copy. The PKCE verifier NEVER leaves this process -- only the public challenge is in the URL -- and the CSRF state is validated here, exactly as the loopback path does. Returns (code, redirect_uri).
    """
    redirect_uri = CLERK_CLI_PASTE_REDIRECT
    authorize_url = (
        f"{CLERK_FRONTEND_API}{_AUTHORIZE_PATH}"
        f"?{_authorize_query(redirect_uri, challenge, csrf_state)}"
    )

    print("Open this URL in any browser to sign in:")
    print(f"  {authorize_url}\n")
    print("After signing in, the page shows a code. Paste it here.")
    pasted = input("Code: ").strip()

    # The page renders `<code>:<state>`. Split from the RIGHT: the CSRF state
    # is a token_urlsafe string that never contains a colon, while the
    # authorization code may, so rpartition pins the state unambiguously.
    code, _, received_state = pasted.rpartition(":")
    if received_state != csrf_state:
        raise LoginError(_STATE_MISMATCH_MESSAGE)
    return code, redirect_uri


def _persist_login(tokens: dict) -> store.AuthState:
    """Validate the token response, read identity, persist and return AuthState.

    Shared tail of both login paths: the code-for-tokens exchange is identical once a code is in hand, so the id_token check + /oauth/userinfo lookup + store.save live here once.
    """
    id_token = tokens.get("id_token")
    refresh_token = tokens.get("refresh_token")
    access_token = tokens.get("access_token", "")
    if not id_token:
        raise AssertionError(
            "Clerk did not return id_token. Confirm the OAuth Application's "
            "scopes include 'openid' in the Clerk dashboard."
        )
    # A session without a refresh_token cannot survive its first gated command:
    # every one of them enters session._ensure_fresh_token, which refreshes
    # unconditionally and raises before any network call when the token is empty.
    # Defaulting the field and persisting it anyway turned a failed login into a
    # login that REPORTS success and is then read back as "not signed in", with
    # no way out -- each retry writes the same unusable state. Refuse it here, in
    # the same breath as the missing id_token, so the failure names its cause at
    # the moment it happens.
    if not refresh_token:
        raise AssertionError(
            "Clerk did not return refresh_token. Confirm the OAuth "
            "Application's scopes include 'offline_access' in the Clerk "
            "dashboard."
        )

    # Read user identity from /oauth/userinfo while we still have the
    # access_token. From here on the access_token is unused -- the check
    # uses id_token + JWKS exclusively.
    user_resp = requests.get(
        f"{CLERK_FRONTEND_API}/oauth/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    user_resp.raise_for_status()
    user = user_resp.json()

    state = store.AuthState(
        user_id=user["sub"],
        id_token=id_token,
        refresh_token=refresh_token,
    )
    store.save(state)
    # The run-grant is session state: it is signed for a specific Clerk `sub`.
    # Persisting a login (including an account switch without a prior logout)
    # invalidates any previously-cached grant -- it was issued for the user who
    # just got replaced. Drop it so the next run refetches a grant for this
    # user, instead of authorize_run hard-blocking on the sub mismatch for the
    # whole grant exp.
    store.clear_grant()
    print(f"Logged in as {user.get('email', '')}")
    return state


def login(*, no_browser: bool = False) -> store.AuthState:
    """Run the PKCE login. Persists and returns AuthState.

    Two ways to get the authorization code back to the CLI:
    - default (no_browser=False): loopback redirect on a localhost port. A graphical browser is opened automatically when one is available; on a headless host no text browser is launched -- the authorize URL is printed for the user to open elsewhere (the loopback callback still completes over SSH port forwarding), or the user can switch to --no-browser.
    - no_browser=True: print the authorize URL for the user to open on any device, then read back the `<code>:<state>` string the redirect page shows. For headless servers with no browser. NOT OAuth Device Flow -- Clerk does not support it; this reuses the same authorization_code+PKCE flow and only swaps how the code returns.

    Both paths keep the PKCE verifier CLI-local (never in any URL) and validate the CSRF state.
    """
    if not CLERK_OAUTH_CLIENT_ID or CLERK_OAUTH_CLIENT_ID.startswith("PLACEHOLDER"):
        raise AssertionError(
            "CLERK_OAUTH_CLIENT_ID in subscription/clerk_config.py is a "
            "placeholder. Provision a Clerk OAuth Application and paste the "
            "publishable key."
        )
    if not CLERK_FRONTEND_API or CLERK_FRONTEND_API.startswith("PLACEHOLDER"):
        raise AssertionError(
            "CLERK_FRONTEND_API in subscription/clerk_config.py is a "
            "placeholder. Paste the Frontend API URL from the Clerk dashboard."
        )

    verifier, challenge = _generate_pkce_pair()
    csrf_state = secrets.token_urlsafe(16)

    if no_browser:
        code, redirect_uri = _login_by_paste(challenge, csrf_state)
    else:
        code, redirect_uri = _login_by_loopback(challenge, csrf_state)

    tokens = _exchange_code_for_tokens(code, verifier, redirect_uri)
    state = _persist_login(tokens)
    _verify_run_authorization()
    return state


def _verify_run_authorization() -> None:
    """Pre-check run entitlement right after login, fail-soft.

    Fetch a server-signed run-grant so the user learns AT LOGIN whether their plan can start runs, instead of only discovering a denial on their first `aibuildai run`. The denial explanation is composed server-side and rendered verbatim by run_grant, so a refused user sees the real reason (e.g. the tier their client line requires) right here.

    This never fails login: the user IS signed in; a denial only means they cannot start runs yet. Only SubscriptionError is caught -- a broader failure (a build defect, an unexpected bug) is a real problem and must surface. The The Clerk session and run authorization modules do not import this command module, so the dependency points one way.
    """
    try:
        run_grant.authorize_run()
    except session.SubscriptionError as exc:
        print(f"Warning: {exc}")
        return
    print("Run authorization is ready. You can start a run with `aibuildai run`.")


def logout() -> None:
    store.clear()
    print("Logged out.")


def _print_model_credential() -> None:
    """Print the model-access identity (the Claude OAuth account) and the state file it was read from, derived from the operator environment via the shared ``config_paths`` rule ($CLAUDE_CONFIG_DIR/.claude.json when set, else the top-level ~/.claude.json)."""
    env = dict(os.environ)
    print(
        f"Model credential: {account_email(env)} "
    )


def _print_run_authorization(state: store.AuthState, *, from_cache: bool) -> None:
    """Print whoami's run-eligibility line under the Plan/Status lines.

    Same architecture as the run gate everywhere else: the client renders the server's explanation verbatim and never composes policy text. On the online path (from_cache=False) it reuses the FULL run gate -- a valid cached grant answers offline, otherwise authorize_run fetches a fresh verdict -- and prints `Run authorization: ok (grant valid until <date>)`, or the server's SubscriptionError message on denial. On the offline path (from_cache=True, where whoami has just reported the account service unreachable) it NEVER fetches: it reports a still-valid cached grant if one exists, else a one-line "could not check" note. Fail-soft: a SubscriptionError becomes the line's text, never a crash. The command module imports both owners at module load; neither owner imports the command module.
    """
    def _ok(claims: dict) -> str:
        valid_until = datetime.fromtimestamp(
            int(claims["exp"]), tz=timezone.utc
        ).strftime("%Y-%m-%d")
        return f"Run authorization: ok (grant valid until {valid_until})"

    if from_cache:
        # The account service is unreachable on this path, so do NOT fetch. Only
        # a still-valid cached grant (verified offline) can answer here.
        try:
            cached = run_grant._verified_cached_grant(state.user_id)
        except session.SubscriptionError:
            cached = None
        print(
            _ok(cached)
            if cached is not None
            else "Run authorization: could not check (offline)"
        )
        return

    try:
        print(_ok(run_grant.authorize_run()))
    except session.SubscriptionError as exc:
        print(f"Run authorization: {exc}")


def whoami() -> None:
    # The model credential is the Claude OAuth account a run authenticates and
    # bills against -- a DIFFERENT identity from the Clerk subscription below.
    # Derived via the same state_file rule the credential + launcher use (#1110,
    # closing the PR #1062 follow-up), so `whoami` finally answers "which
    # model-access identity will my runs use".
    _print_model_credential()

    state = store.load()
    if state is None:
        print("Not signed in. Run `aibuildai login` to continue.")
        return

    # Branch on exception type so each failure mode gets a precise message:
    # each branch maps one transport- or crypto-layer failure to its own
    # user-facing instruction.
    try:
        claims, from_cache = session.current_claims()
    except session.SessionRevokedError:
        # Clerk rejected the refresh request. The local "Signed in as"
        # label would contradict reality -- drop it. This is the only
        # branch where the user must re-login.
        print(
            "Not signed in. Your previous session has expired or "
            "been revoked.\n"
            "Run `aibuildai login` to view your current plan."
        )
        return
    except session.CachedSessionTooOldError:
        print("Signed in.")
        print(
            "Your cached session is older than 30 days. "
            "Run `aibuildai login` to refresh."
        )
        return
    except session.InvalidTokenFromServiceError:
        print("Signed in.")
        print(
            "aibuildai's account service returned an invalid token. "
            "Please try again in a moment."
        )
        return
    except session.SubscriptionError as exc:
        # Any other SubscriptionError the refresh path now raises -- notably the
        # clock-skew message from _ensure_fresh_token when this host's clock is
        # off beyond the leeway. It is a complete, actionable sentence, so render
        # it verbatim rather than let it escape as a traceback. Placed AFTER the
        # specific subclasses above (which subclass SubscriptionError) so each
        # keeps its own bespoke instruction.
        print("Signed in.")
        print(str(exc))
        return
    except requests.HTTPError:
        print("Signed in.")
        print(
            "aibuildai's account service is unavailable right now. "
            "Please try again in a few minutes."
        )
        return
    except (requests.ConnectionError, requests.Timeout):
        # Network unreachable AND cached id_token is past exp -- the
        # offline-grace fallback inside current_claims only
        # re-raises these when cached is also dead.
        print("Signed in.")
        print(
            "Could not reach aibuildai's account service, and your "
            "cached session has expired. Please try again when you "
            "are online."
        )
        return
    except jwt.PyJWKClientError:
        print("Signed in.")
        print(
            "Could not reach aibuildai's account service to verify "
            "your session. Please try again in a moment."
        )
        return
    except jwt.PyJWTError:
        print("Signed in.")
        print(
            "Your cached session token can't be verified. Run "
            "`aibuildai logout` then `aibuildai login`."
        )
        return

    print(f"Signed in as: {claims['email']}")

    metadata = claims.get("public_metadata") or {}
    plan_label = metadata.get("plan_name") or metadata.get("plan") or "Unknown"
    status_label = metadata.get("status") or "Unknown"
    # Surface the grace window so a user who clicked Cancel sees their
    # remaining entitlement instead of "Pro / canceled" with no end date.
    if metadata.get("cancel_pending") and metadata.get("plan_exp"):
        ends = datetime.fromtimestamp(
            int(metadata["plan_exp"]), tz=timezone.utc
        ).strftime("%Y-%m-%d")
        plan_label = f"{plan_label} (canceled, ends {ends})"

    if from_cache:
        # Refresh failed because Clerk was unreachable, but the cached
        # id_token was still within exp and within the 30-day staleness
        # ceiling. Mark the values as cached so the user knows they may
        # not reflect a very recent change made on Clerk's side.
        print(f"Plan:         {plan_label} (offline cache)")
        print(f"Status:       {status_label} (offline cache)")
        print(
            "Note:         Could not reach aibuildai's account service. "
            "Values shown are from the last successful sync."
        )
    else:
        print(f"Plan:         {plan_label}")
        print(f"Status:       {status_label}")

    # Whether THIS machine can currently start a run -- the same server-authored
    # verdict the run gate and login pre-check surface, rendered verbatim.
    _print_run_authorization(state, from_cache=from_cache)


def open_account_page() -> None:
    # Clerk hosted user portal: one SPA at `/user` with Profile, Security,
    # Plans, and Billing tabs. We expose a single CLI command that lands
    # the user on the portal home; from there they pick what they came for.
    url = f"{CLERK_ACCOUNT_PORTAL}/user"
    # The URL is the primary artifact: print it FIRST, on its own line with blank
    # lines around it, BEFORE any (silent, best-effort) launch attempt -- so it
    # stays readable no matter what the launch does.
    print(f"\nYour aibuildai account page:\n\n  {url}\n")
    if _try_open_browser(url):
        lead = "Opened your aibuildai account page in the browser."
    else:
        lead = (
            "Could not open a browser automatically -- open the URL above to "
            "reach your account page."
        )
    print(
        f"{lead}\n"
        "  - Plans tab: subscribe / change / cancel a plan\n"
        "  - Billing tab: update payment method, view invoices\n"
        "  - Profile / Security tabs: account settings"
    )
