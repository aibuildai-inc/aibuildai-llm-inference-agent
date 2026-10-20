"""The single run gate for the aibuildai workflow: signed lease + offline grace.

Entitlement is decided server-side: our Cloudflare Worker re-reads the authoritative plan from the Clerk Backend API and -- only if entitled -- signs a short-lived run-grant with an RS256 PRIVATE key the binary never holds. The CLI verifies that grant with the PUBLIC key compiled into the binary (clerk_config.RUN_GRANT_PUBLIC_KEY), so forging a grant requires the server-side private key, not a one-bit binary patch.

`authorize_run()` is the only public entry. Per run, in order:

1. Load the cached grant from ~/.aibuildai/run_grant.jwt and verify it OFFLINE against the compiled-in public key (signature + exp + iss + aud + sub == this session's user + client_version == this build). A still-valid cached grant for THIS build authorizes the run with NO network call. The grant's own `exp` IS the offline-grace window -- this module is TTL-agnostic, it never hardcodes a duration; the Worker owns the TTL. The grant carries the client_version it was minted for; a build upgrade makes the cached grant stop matching, so the next run refetches a fresh verdict instead of coasting on the pre-upgrade grant until its exp.
2. If there is no valid cached grant (none cached, or the cached one expired), refresh the id_token, fetch a fresh grant from the Worker (obtain_run_grant), cache it, and allow.
3. If the Worker is unreachable AND there is no valid cached grant, FAIL CLOSED (raise SubscriptionError). There is NO "network unreachable -> allow" fallback that ignores the lease: such a fallback would be trivially triggered by an attacker firewalling the Worker engine. Offline tolerance comes only from a previously-issued, still-unexpired grant.

Flow (obtain_run_grant, the fetch half):

1. Generate a per-run nonce.
2. POST {id_token, run_nonce, client_version} to the Worker's /cli/run-grant endpoint. The id_token is the one just refreshed -- at most milliseconds old, so the Worker's own exp check passes. client_version is this build's derived product version (aibuildai_version.APP_VERSION, never hardcoded); the Worker reads it to apply its version-based tier policy (>=2.5 -> Max) and fails closed to the strictest tier if it is missing -- so every client sends it explicitly rather than relying on that fallback.
3. The Worker verifies the id_token against Clerk's JWKS, re-reads the authoritative plan via the Clerk Backend API, and returns a compact-JWS run-grant (its claims echo back the client_version we POSTed). It signs for every authenticated caller; entitlement shows up in `aud`, not in the status code. When the caller is NOT entitled to run, the 200 also carries a display-ready `user_message` sentence explaining why (e.g. the tier this line requires); the client raises that string verbatim and never composes policy text itself (RFC 9457 "detail" idiom). A NEUTRAL fallback sentence (never claiming the subscription is inactive) is used ONLY when `user_message` is absent. A Worker that still answers 403 is handled by the same path as any other non-200.
4. Verify the grant locally: RS256 signature against the compiled-in public key, iss, exp (not expired), sub matches this session's user, and the nonce is echoed back unchanged. Cache the verified grant string for the offline-grace path above.
5. THEN read the run decision off `aud`. The Worker signs every authenticated caller and states in `aud` which services they may reach; only a caller whose live plan entitles them gets `RUN_GRANT_AUDIENCE`. A grant without it is still genuine and is still cached at step 4 -- it just does not authorize a run, so this module raises the Worker's own `user_message`. The Kb needs no grant: it serves every caller openly.
"""
from __future__ import annotations

import secrets

import jwt
import requests

from subscription import session, store
from subscription.session import SubscriptionError
from subscription.clerk_config import (
    RUN_GRANT_AUDIENCE,
    RUN_GRANT_ENDPOINT,
    RUN_GRANT_ISSUER,
    RUN_GRANT_PUBLIC_KEY,
)
from aibuildai_version import APP_VERSION, display_version

# The grant round-trip is a single short request; keep the timeout tight so a
# hung Worker fails the run quickly rather than stalling startup.
_HTTP_TIMEOUT_SECONDS = 10

def _loaded_state_or_raise() -> store.AuthState:
    state = store.load()
    if state is None:
        raise SubscriptionError(
            "You are not signed in. Run `aibuildai login` to continue."
        )
    return state


def _verified_cached_grant(user_id: str) -> dict | None:
    """Verified claims from the cached grant for `user_id`, or None when none is cached.

    Wraps load_grant + _verify_cached_grant so the cached-grant check is one call at both the lock-free fast path and the in-lock re-check.
    """
    cached = store.load_grant()
    if cached is None:
        return None
    return _verify_cached_grant(cached, user_id)


def authorize_run() -> dict:
    """Authorize a run. Returns the verified grant claims, or raises.

    Offline-first: a cached, still-valid grant authorizes with no network. On a cache miss / expiry, refresh the id_token, fetch a fresh grant, cache it. Fails closed (SubscriptionError) when there is no valid cached grant and the Worker cannot be reached or refuses.

    Concurrent cold-cache runs are serialized by store.auth_refresh_lock so the single-use Clerk refresh_token is rotated by exactly one process (see below); the warm-cache fast path takes no lock and runs fully in parallel.

    Clock-rollback is an accepted limitation, not defended here: the offline window is already bounded by the grant's own `exp`, and an attacker who can set the system clock back can more cheaply patch the frozen binary. Adding monotonic-clock protection would not raise the bypass cost, so it is out.
    """
    # Fast path: a warm cached grant authorizes with no lock and no refresh, so
    # warm-cache concurrent runs never contend.
    state = _loaded_state_or_raise()
    verified = _verified_cached_grant(state.user_id)
    if verified is not None:
        return verified

    # Cold/expired grant cache: serialize the refresh+fetch across processes. Clerk
    # refresh_tokens are single-use and rotate on every refresh; two concurrent runs that
    # both refreshed would make the second present a consumed token and Clerk would revoke
    # the whole session. The lock makes the read-modify-write atomic across processes.
    with store.auth_refresh_lock():
        # Re-load INSIDE the lock: while we waited, a concurrent run may have rotated the
        # refresh_token AND cached a fresh grant.
        state = _loaded_state_or_raise()
        verified = _verified_cached_grant(state.user_id)
        if verified is not None:
            # The grant a concurrent run just wrote: reuse it, skip the refresh entirely.
            return verified
        state = session.fresh_id_token_for_grant(state)
        return obtain_run_grant(state)


def _verify_cached_grant(grant: str, user_id: str) -> dict | None:
    """Verify a cached grant OFFLINE. Returns claims, or None to re-fetch.

    Enforces signature + exp + iss + aud against the compiled-in public key (no network), and that the grant was issued for this signed-in user. The mint-time `nonce` is NOT retained across runs, so it is not checked here -- the signature + exp already make the grant unforgeable and time-bounded.

    `aud` IS enforced here, unlike in `obtain_run_grant`. A cached grant that admits its holder to the Kb but not to a run returns None, so the caller re-fetches and raises the Worker's current `user_message` -- which only a fresh fetch carries. The re-fetch also refreshes the cached Kb credential, so the run this call refuses still leaves the Kb reachable.

    Returns None when the grant is expired or otherwise unverifiable, so the caller falls through to a fresh fetch. Also returns None when the grant was minted for a DIFFERENT client_version than this build (a binary upgrade): the grant carries the client_version the Worker signed it for, and a stale one no longer reflects this build's tier policy, so the next run refetches a fresh verdict (and a fresh denial message) instead of coasting on the pre-upgrade grant until its exp. A pre-contract grant (no client_version claim -> None != version) refetches the same way, which is correct and self-healing. Raises SubscriptionError when the grant is validly signed but issued for a DIFFERENT user -- that is a real account mismatch, not a stale lease, so it is rejected rather than silently refetched.
    """
    try:
        claims = jwt.decode(
            grant,
            RUN_GRANT_PUBLIC_KEY,
            algorithms=["RS256"],
            audience=RUN_GRANT_AUDIENCE,
            issuer=RUN_GRANT_ISSUER,
            leeway=session.CLOCK_LEEWAY_SECONDS,
            options={"require": ["sub", "exp", "iat", "iss", "aud"]},
        )
    except jwt.PyJWTError:
        # ANY unverifiable cached grant -- expired, bad signature, wrong aud/iss,
        # or structurally malformed (a missing required claim, a truncated disk
        # write) -- is treated as "no valid cache": return None so the caller
        # falls through to a fresh fetch. This self-heals a corrupt cache on the
        # next online run and never opens the gate (the fetch itself fails closed
        # when the Worker is unreachable).
        return None

    if claims["sub"] != user_id:
        raise SubscriptionError(
            "aibuildai's cached run authorization was issued for a different "
            "account. Run `aibuildai logout` then `aibuildai login`."
        )
    if display_version(claims.get("client_version") or "") != display_version(APP_VERSION):
        # Version-bind the offline grant cache. The Worker echoes the POSTed
        # client_version into the grant, so after a binary upgrade the cached
        # grant no longer matches this build. Compare at DISPLAY granularity
        # (strip the PEP 440 +local build id) so a per-commit source build does
        # NOT refetch on every commit; a release build's display id is the plain
        # semver, and an absent claim -> "" -> naturally mismatches. Return None
        # to refetch a fresh verdict rather than reuse a stale one until its exp.
        return None
    return claims


def obtain_run_grant(state: store.AuthState) -> dict:
    """Fetch, verify, and cache a server-signed run-grant. Returns the claims.

    On success the raw grant string is cached to disk (store.save_grant) so a later run can re-verify it offline. Raises SubscriptionError on any failure (no fallback). Raises AssertionError if the compiled-in public key is still the placeholder (a build defect, not a user-facing condition).
    """
    if "PLACEHOLDER_RUN_GRANT_PUBLIC_KEY" in RUN_GRANT_PUBLIC_KEY:
        raise AssertionError(
            "RUN_GRANT_PUBLIC_KEY in subscription/clerk_config.py is a "
            "placeholder. Generate the run-grant RSA key pair, set the private "
            "key as the Worker's RUN_GRANT_PRIVATE_KEY secret, and paste the "
            "SPKI public key PEM here before building."
        )

    run_nonce = secrets.token_urlsafe(32)

    try:
        resp = requests.post(
            RUN_GRANT_ENDPOINT,
            json={
                "id_token": state.id_token,
                "run_nonce": run_nonce,
                "client_version": APP_VERSION,
            },
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise SubscriptionError(
            "Could not reach aibuildai's account service to authorize this "
            "run. Please check your internet connection and try again."
        ) from exc

    if resp.status_code == 403:
        # The Worker re-checked the authoritative plan and refused. It composes a
        # complete, display-ready denial sentence in the 403 body's `user_message`
        # field (RFC 9457 "detail" idiom: the client renders it verbatim and never
        # re-derives policy text -- the reason, e.g. the required tier, is the
        # server's to word). Falling back when `user_message` is absent or the body
        # is unparseable is a deliberate cross-deploy compatibility contract -- an
        # older/rolled-back Worker, or a CDN/WAF error page in front of it -- not
        # defensive programming. The fallback stays NEUTRAL: it never asserts the
        # subscription is inactive, which is false for exactly the active-plan
        # population this branch serves (the wrong claim this change removes).
        try:
            denial = resp.json()
        except ValueError:
            denial = None
        message = denial.get("user_message") if isinstance(denial, dict) else None
        if isinstance(message, str) and message.strip():
            raise SubscriptionError(message)
        raise SubscriptionError(
            "aibuildai could not authorize this run for your current plan. "
            "Run `aibuildai account` to review your plan."
        )
    if resp.status_code == 401:
        raise SubscriptionError(
            "aibuildai could not verify your session. Run `aibuildai logout` "
            "then `aibuildai login`."
        )
    if resp.status_code != 200:
        raise SubscriptionError(
            "aibuildai's account service is unavailable right now. "
            "Please try again in a few minutes."
        )

    # A 200 from a CDN / WAF in front of the Worker can carry an HTML error
    # page; resp.json() then raises a JSONDecodeError (a ValueError subclass)
    # that is NOT a SubscriptionError. Convert it so the run is blocked cleanly
    # rather than crashing with a raw traceback (D1: every failure raises).
    try:
        body = resp.json()
    except ValueError as exc:
        raise SubscriptionError(
            "aibuildai's account service returned an unreadable response. "
            "Please try again in a moment."
        ) from exc

    grant = (body or {}).get("grant")
    if not grant:
        raise SubscriptionError(
            "aibuildai's account service returned an empty authorization. "
            "Please try again in a moment."
        )

    try:
        # `verify_aud` is off HERE ONLY, and the audience is checked by hand
        # below instead. The Worker signs every authenticated caller now and
        # says what they may reach in `aud`; a grant without the run audience
        # is still genuine and is CACHED, so the offline-grace path holds the
        # Worker's latest answer for this user. Leaving the library check here
        # would raise before `save_grant`.
        claims = jwt.decode(
            grant,
            RUN_GRANT_PUBLIC_KEY,
            algorithms=["RS256"],
            issuer=RUN_GRANT_ISSUER,
            leeway=session.CLOCK_LEEWAY_SECONDS,
            options={
                "require": ["sub", "exp", "iat", "iss", "aud", "nonce"],
                "verify_aud": False,
            },
        )
    except (jwt.ExpiredSignatureError, jwt.ImmatureSignatureError) as exc:
        # The Worker signed this grant milliseconds ago, yet its exp is already
        # past (this host's clock leads the server) or its iat is still in the
        # future (this host's clock lags it) by MORE than the leeway. A grant
        # this fresh can only look expired or not-yet-valid when the LOCAL clock
        # disagrees with the server -- point at the clock, not the account
        # service. (Clock-rollback abuse stays out of scope, as documented on
        # authorize_run: the offline window is bounded by the grant's exp.)
        raise session.clock_skew_error() from exc
    except jwt.PyJWTError as exc:
        raise SubscriptionError(
            f"aibuildai could not verify the run authorization "
            f"({exc.__class__.__name__}). Please try again in a moment."
        ) from exc

    if claims.get("nonce") != run_nonce:
        raise SubscriptionError(
            "aibuildai's run authorization did not match this request. "
            "Please try again in a moment."
        )
    if claims.get("sub") != state.user_id:
        raise SubscriptionError(
            "aibuildai's run authorization was issued for a different "
            "account. Run `aibuildai logout` then `aibuildai login`."
        )

    # Cache the verified grant string for the offline-grace path: later runs
    # re-verify it locally (against `exp`) with no network round-trip. This runs
    # BEFORE the run-audience check on purpose -- see the decode above.
    store.save_grant(grant)

    # The run decision, read off signed material rather than computed here: the
    # Worker put `RUN_GRANT_AUDIENCE` in `aud` only for a caller whose live plan
    # entitles them. A grant without it still reaches the Kb, and it is already
    # cached above, so the Kb keeps working for a caller who may not run.
    aud = claims["aud"]
    values = aud if isinstance(aud, list) else [aud]
    # Refuse a non-string member the way pyjwt does, so this check and the
    # cached-grant check downstream cannot disagree about the same grant.
    if any(not isinstance(v, str) for v in values) or RUN_GRANT_AUDIENCE not in values:
        raise SubscriptionError(
            body.get("user_message")
            or "Your plan does not cover this version of aibuildai."
        )
    return claims
