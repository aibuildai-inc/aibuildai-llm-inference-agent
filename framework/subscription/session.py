"""Refresh and verify the user's Clerk session.

The single run check lives in `subscription/run_grant.py:authorize_run`. This module owns the Clerk session it consumes: refresh the cached id_token when needed, verify every id_token, and return verified claims to `whoami`.

How the id_token machinery works (no Clerk Backend API, no secret key in the binary):

1. Read the cached id_token from ~/.aibuildai/auth.json.
2. Verify the id_token's RS256 signature with Clerk's public key from the JWKS endpoint. Also verify iss, aud, azp (when present), and the presence of sub/email/exp/iat/iss/aud -- this proves Clerk really issued the token for this client_id and tenant.
3. If the token is expired (or close to it), use the refresh_token to mint a new id_token, then re-verify.

The id_token's public_metadata field is populated by the Cloudflare worker that listens to Clerk webhook events about subscription state changes. The worker writes plan info into Clerk user.public_metadata, and Clerk embeds that field into every id_token it signs for the user. `whoami` surfaces that metadata; entitlement itself is decided server-side by the Worker when it signs the run-grant.

Exception model (typed, not prose-encoded):

- SubscriptionError                  base for account service failures
- SessionRevokedError                Clerk rejected refresh_token (4xx invalid_grant) or auth.json has no refresh_token. User must re-login.
- InvalidTokenFromServiceError       refresh returned a token already past exp -- Clerk-side anomaly.
- CachedSessionTooOldError           offline + cached iat older than the 30-day staleness ceiling. Display path only; never raised by the run check.

Outside SubscriptionError, callers see typed exceptions where appropriate: requests.ConnectionError / Timeout for network unreachable, requests.HTTPError for Clerk 5xx / 429, jwt.PyJWTError (and subclasses) for crypto / claim failures, jwt.PyJWKClientError for JWKS outages.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import jwt
import requests
from jwt import PyJWKClient

from subscription import store
from subscription.clerk_config import (
    CLERK_FRONTEND_API,
    CLERK_JWKS_URL,
    CLERK_OAUTH_CLIENT_ID,
)


class SubscriptionError(Exception):
    """Base for account service failures."""


class SessionRevokedError(SubscriptionError):
    """Clerk rejected the refresh_token, or auth.json has no refresh_token. The local session is dead; the user must re-login."""


class InvalidTokenFromServiceError(SubscriptionError):
    """Refresh succeeded but the new id_token is already past exp. A Clerk-side anomaly; caller should retry later, not re-login."""


class CachedSessionTooOldError(SubscriptionError):
    """Offline + cached id_token's iat is older than the staleness ceiling (30 days). Reached only via the offline display path -- means the cached plan is too old to surface as "current" with confidence."""


class _RefreshRejectedError(Exception):
    """Clerk rejected the refresh token, or no refresh token exists."""


_HTTP_TIMEOUT_SECONDS = 10

# Margin for clock skew + "expired by the time we POST" risk.
_TOKEN_EXPIRY_GRACE_SECONDS = 30

# Wall-clock tolerance when verifying a credential the server signed moments
# ago (a fresh id_token here, a fresh run-grant in run_grant.py, which imports
# this constant). NTP-corrected hosts drift < 1s; this absorbs benign drift
# (NTP jitter, a brief suspend) so small offsets never surface, while a larger
# skew -- the canonical case is a WSL2 host resuming after the Windows host
# slept, whose clock lags until it resyncs -- is reported honestly as a clock
# problem rather than a bad credential or a network fault. 300s extends the
# 3-day offline grant window by 0.12% (negligible), and clock-ROLLBACK abuse
# stays out of scope: the offline window is already bounded by the grant's exp.
CLOCK_LEEWAY_SECONDS = 300


def clock_skew_error() -> SubscriptionError:
    """A SubscriptionError blaming THIS host's clock, for when a credential the server signed milliseconds ago still fails exp/iat verification beyond the leeway -- near-certainly local clock skew, not a bad credential. Raised for both a just-refreshed id_token (here) and a just-fetched run-grant (run_grant.py)."""
    return SubscriptionError(
        "aibuildai could not authorize this run because this computer's clock "
        "appears to be out of sync. The account service rejected a just-issued "
        "authorization as expired or not-yet-valid, which almost always means "
        "the system clock is wrong. Turn on automatic date and time (or sync "
        "the clock) and try again."
    )

# Cached id_token's iat must be within this window (in seconds) to be
# shown as offline cache. Beyond this, refresh has been broken so long
# that the plan info almost certainly does not reflect Clerk's truth.
_CACHED_STALENESS_CEILING_SECONDS = 30 * 86400

# Verified live: Clerk's id_token iss == CLERK_FRONTEND_API exactly
# ("https://clerk.aibuildai.io", no trailing slash). Verifying iss
# rejects tokens from a different Clerk tenant that happens to share
# the publishable client_id prefix.
_CLERK_ISSUER = CLERK_FRONTEND_API


# Cache one JWKS client per process. PyJWKClient handles fetching and
# caching the public keys internally; first verify pays the network cost,
# subsequent verifies are local.
_jwks_client: Optional[PyJWKClient] = None


def _get_jwks_client() -> PyJWKClient:
    """Return a process-global PyJWKClient bound to Clerk's JWKS endpoint.

    Builds the client with an explicit `ssl_context` rooted at certifi's CA bundle. PyJWKClient uses stdlib `urllib.request` (not `requests`), so it falls back to `ssl.create_default_context()` for trust decisions when no context is provided. Inside the PyInstaller-frozen binary, the default context cannot locate the OS CA store and every JWKS fetch fails with PyJWKClientConnectionError -- even though `requests` (used by login) works fine via its own bundled certifi. Passing certifi here makes the check robust in both source-mode (conda env, system certs available) and frozen-binary mode (no system certs, certifi bundled).
"""
    global _jwks_client
    if _jwks_client is None:
        import ssl
        import certifi
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        _jwks_client = PyJWKClient(
            CLERK_JWKS_URL,
            cache_keys=True,
            lifespan=24 * 3600,
            ssl_context=ssl_ctx,
        )
    return _jwks_client


def _verify_id_token(id_token: str) -> dict:
    """Verify signature + iss + aud + required claims. Returns payload.

    Does NOT check exp -- callers do that explicitly so they can distinguish "token expired, refresh and retry" from "token signed by someone other than Clerk" (the latter is a hard failure).

    If the token carries an `azp` claim, it must equal CLERK_OAUTH_CLIENT_ID (OIDC 3.1.3.7). Tokens without `azp` are accepted (Clerk's id_tokens observed live do not include it).

    May raise jwt.PyJWTError (signature / claim mismatch) or jwt.PyJWKClientError (JWKS endpoint unreachable, kid not found, etc).
"""
    signing_key = _get_jwks_client().get_signing_key_from_jwt(id_token).key
    claims = jwt.decode(
        id_token,
        signing_key,
        algorithms=["RS256"],
        audience=CLERK_OAUTH_CLIENT_ID,
        issuer=_CLERK_ISSUER,
        leeway=CLOCK_LEEWAY_SECONDS,
        options={
            "verify_exp": False,
            "require": ["sub", "email", "exp", "iat", "iss", "aud"],
        },
    )
    azp = claims.get("azp")
    if azp is not None and azp != CLERK_OAUTH_CLIENT_ID:
        raise jwt.InvalidTokenError(
            f"azp claim {azp!r} does not match expected client_id."
        )
    return claims


def _is_expired_or_near(claims: dict) -> bool:
    return (claims.get("exp", 0) - _TOKEN_EXPIRY_GRACE_SECONDS) < time.time()


def _is_cached_too_old(claims: dict) -> bool:
    iat = claims.get("iat", 0)
    return (time.time() - iat) > _CACHED_STALENESS_CEILING_SECONDS


def _refresh_id_token(state: store.AuthState) -> store.AuthState:
    """Refresh the Clerk id_token and save any rotated refresh_token."""
    if not state.refresh_token:
        raise _RefreshRejectedError(
            "Your session has no refresh token. Please run `aibuildai login`."
        )

    form = {
        "grant_type": "refresh_token",
        "refresh_token": state.refresh_token,
        "client_id": CLERK_OAUTH_CLIENT_ID,
        "scope": "openid profile email public_metadata offline_access",
    }
    client_secret = os.environ.get("AIBUILDAI_OAUTH_CLIENT_SECRET")
    if client_secret:
        del form["client_id"]
        response = requests.post(
            f"{CLERK_FRONTEND_API}/oauth/token",
            data=form,
            auth=(CLERK_OAUTH_CLIENT_ID, client_secret),
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
    else:
        response = requests.post(
            f"{CLERK_FRONTEND_API}/oauth/token",
            data=form,
            timeout=_HTTP_TIMEOUT_SECONDS,
        )

    if response.status_code == 429 or 500 <= response.status_code < 600:
        response.raise_for_status()
    if response.status_code != 200:
        raise _RefreshRejectedError(
            f"Clerk rejected the refresh request "
            f"(HTTP {response.status_code}). Please run `aibuildai login` again."
        )

    tokens = response.json()
    new_id_token = tokens.get("id_token")
    if not new_id_token:
        raise AssertionError("Clerk refresh did not return a new id_token")

    state.id_token = new_id_token
    if tokens.get("refresh_token"):
        state.refresh_token = tokens["refresh_token"]
    store.save(state)
    return state


def _ensure_fresh_token(
    state: store.AuthState,
) -> tuple[store.AuthState, dict, bool]:
    """Return (state, verified_claims, from_cache).

    from_cache=False -> claims come from a fresh refresh from Clerk.
    from_cache=True  -> network was unreachable; claims come from the
                        cached id_token (offline grace).

    Exceptions (typed, not prose-encoded):
      SessionRevokedError              Clerk rejected refresh_token (4xx)
                                        or auth.json has no refresh_token
      InvalidTokenFromServiceError     refresh succeeded but the new token
                                        is (near) expired while minted within
                                        the leeway -- a service-side anomaly,
                                        not clock skew
      SubscriptionError (base)         a just-refreshed token failed exp/iat
                                        verification BEYOND the clock leeway
                                        (iat too far future, or our clock too
                                        far past its iat) -- near-certain local
                                        clock skew, raised via clock_skew_error
      CachedSessionTooOldError         offline AND cached iat older than
                                        the 30-day staleness ceiling
      requests.HTTPError               Clerk 5xx / 429
      requests.ConnectionError /
        requests.Timeout               offline AND cached token also past
                                        exp; otherwise we return cached
                                        silently with from_cache=True
      jwt.PyJWTError                   signature / claim verify failure
      jwt.PyJWKClientError             JWKS endpoint unreachable or kid
                                        missing
    """
    try:
        state = _refresh_id_token(state)
    except (requests.ConnectionError, requests.Timeout) as exc:
        # Network unreachable. Offline grace: return cached claims iff
        # they are still within exp AND younger than the staleness
        # ceiling. Otherwise let the caller see the original failure
        # (the network error, or the staleness error).
        claims = _verify_id_token(state.id_token)
        if _is_expired_or_near(claims):
            raise  # re-raise the network error; cached is also dead
        if _is_cached_too_old(claims):
            raise CachedSessionTooOldError(
                "The cached session is older than 30 days and may be "
                "stale. Run `aibuildai login` to refresh."
            ) from exc
        return state, claims, True
    except _RefreshRejectedError as exc:
        raise SessionRevokedError(
            "Your aibuildai session has expired or been revoked. "
            "Run `aibuildai login` to continue."
        ) from exc
    try:
        claims = _verify_id_token(state.id_token)
    except jwt.ImmatureSignatureError as exc:
        # Clerk minted this id_token milliseconds ago, yet its iat is still in
        # the future by MORE than the leeway: THIS host's clock lags the server
        # (the canonical case: WSL2 after the Windows host slept). Point at the
        # clock, not the session -- logout/login would not help.
        raise clock_skew_error() from exc
    if _is_expired_or_near(claims):
        if (time.time() - claims["iat"]) > CLOCK_LEEWAY_SECONDS:
            # Clerk minted this token milliseconds ago, yet OUR clock already
            # sits more than the leeway PAST its iat: this host's clock LEADS the
            # server. A refresh/JWKS round trip cannot span the leeway, so a
            # now-iat this large can only be clock skew -- the opposite direction
            # of the iat case above. Point at the clock, not the service.
            raise clock_skew_error()
        # The token was minted within the leeway, yet our clock already sees it
        # as (near) expired -- NOT clock skew (a large offset would have tripped
        # the branch above). Either the server minted a too-short/expired token
        # or the round trip ate its remaining life; both are honestly a "try
        # again" service condition, so keep the existing attribution.
        raise InvalidTokenFromServiceError(
            "The account service returned an already-expired session "
            "token. Please try again in a moment."
        )
    return state, claims, False


def current_claims() -> tuple[dict, bool]:
    """Return the verified Clerk claims and whether they came from offline cache.

    Propagates the typed exceptions described in `_ensure_fresh_token`.

    The id_token refresh rotates the single-use Clerk refresh_token, so it runs under the same cross-process auth lock the run path uses (load-fresh -> refresh -> save). Otherwise a whoami concurrent with a run could present a consumed token and revoke the session.
"""
    with store.auth_refresh_lock():
        state = store.load()
        if state is None:
            raise SessionRevokedError("Not signed in. Run `aibuildai login`.")
        state, claims, from_cache = _ensure_fresh_token(state)
    return claims, from_cache


def fresh_id_token_for_grant(state: store.AuthState) -> store.AuthState:
    """Refresh (or offline-validate) the id_token before fetching a run-grant.

    Wraps `_ensure_fresh_token` and maps every transport / crypto failure to a user-facing SubscriptionError, so the single run gate (`run_grant.authorize_run`) has exactly one exception type to catch. The refreshed id_token is what the Worker re-verifies when it signs a fresh grant; the returned AuthState carries it.

    SessionRevokedError / InvalidTokenFromServiceError / CachedSessionTooOldError are already SubscriptionError subclasses and propagate unchanged, as does the base SubscriptionError _ensure_fresh_token raises for a just-refreshed token whose exp/iat fails verification beyond the clock leeway (local clock skew).
"""
    try:
        state, _claims, _from_cache = _ensure_fresh_token(state)
    except (requests.ConnectionError, requests.Timeout) as exc:
        # _ensure_fresh_token only re-raises these when the cached token was
        # also past exp. Surface a clear user-facing message.
        raise SubscriptionError(
            "Could not reach aibuildai's account service to verify "
            "your subscription, and your cached session has expired. "
            "Please check your internet connection and try again."
        ) from exc
    except requests.HTTPError as exc:
        raise SubscriptionError(
            "aibuildai's account service is unavailable right now. "
            "Please try again in a few minutes."
        ) from exc
    except jwt.PyJWKClientError as exc:
        # JWKS unreachable / kid missing -- distinct from PyJWTError below,
        # which signals an actually-bad token.
        raise SubscriptionError(
            "Could not reach aibuildai's account service to verify "
            "your session. Please try again in a moment."
        ) from exc
    except jwt.PyJWTError as exc:
        raise SubscriptionError(
            f"Your session token failed verification "
            f"({exc.__class__.__name__}). Please run `aibuildai logout` "
            f"then `aibuildai login`."
        ) from exc
    return state
