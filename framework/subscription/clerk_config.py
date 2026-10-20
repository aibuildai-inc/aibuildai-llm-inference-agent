"""Clerk authentication / subscription configuration.

Production values for the aibuildai.io Clerk instance. ALL values here are PUBLIC and safe to ship in the binary. The Clerk secret key (sk_live_*) is NEVER read here -- it only flows through the Cloudflare Worker's encrypted secret store.
"""
import os

CLERK_FRONTEND_API = "https://clerk.aibuildai.io"

CLERK_ACCOUNT_PORTAL = "https://accounts.aibuildai.io"

CLERK_OAUTH_CLIENT_ID = "PRsVKGUtj54SKyGX"

# Browser login callback port. 0 asks the OS for a free loopback port.
# Set AIBUILDAI_CLERK_LOGIN_REDIRECT_PORT to force a fixed port for debugging.
CLERK_LOGIN_REDIRECT_PORT = int(os.environ.get(
    "AIBUILDAI_CLERK_LOGIN_REDIRECT_PORT", "0",
))

# CLI no-browser paste-login redirect; must be whitelisted in the Clerk OAuth
# Application. `aibuildai login --no-browser` points Clerk's redirect here
# instead of the loopback port; the page shows the user a `<code>:<state>`
# string to paste back into the terminal (for machines with no browser).
CLERK_CLI_PASTE_REDIRECT = "https://aibuildai-clerk-sync.subscription-af1.workers.dev/cli/callback"

# JWKS endpoint Clerk publishes its RSA public keys to. The CLI fetches and
# caches these to verify id_token signatures locally, without the secret key.
CLERK_JWKS_URL = f"{CLERK_FRONTEND_API}/.well-known/jwks.json"

# --- Server-signed run-grant (#671 Tier-A) ---------------------------------
# The single run gate: the CLI asks our Cloudflare Worker for a short-lived
# run-grant the Worker signs with an RS256 PRIVATE key the binary never holds.
# The CLI verifies that grant with the PUBLIC key compiled in below, so forging
# a grant requires the server-side private key, not a one-bit binary patch. The
# verified grant is cached on disk and re-verified OFFLINE on later runs (its
# own `exp` is the offline-grace window), so a transient network blip does not
# block a paying user. The Worker itself is the existing Clerk sync worker.
RUN_GRANT_ISSUER = "https://aibuildai-clerk-sync.subscription-af1.workers.dev"

# Endpoint the CLI POSTs {id_token, run_nonce} to.
RUN_GRANT_ENDPOINT = f"{RUN_GRANT_ISSUER}/cli/run-grant"

# Audience the signed grant must carry, asserted client-side.
RUN_GRANT_AUDIENCE = "aibuildai-cli"

# SPKI public-key PEM matching the Worker's RUN_GRANT_PRIVATE_KEY secret. This
# is PUBLIC and safe to ship. obtain_run_grant() fails fast if this is ever
# reset to a placeholder (same idiom as auth.login() rejecting a placeholder
# client_id).
RUN_GRANT_PUBLIC_KEY = (
    "-----BEGIN PUBLIC KEY-----\n"
    "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAsFkIfpNBNgRY1DrtxdEz\n"
    "lb72MlTOo0eY9d9/2jzFpX2vmWgZEOMP0GBj0QonY3pLNftOr4TWrN3OdRiOKVoD\n"
    "bnDZPvG/k2L8urKGrr5zPK+1/omjr8551SO1hqy75yq9fcdT8OelO3zV1GJN3lUc\n"
    "3sLrZx4I+7VyXKiXwYtjum9PdCttln1JnfmPxAmlMyNH0zV0yvM2tShZC3moycv9\n"
    "3mm+fZVEtkKOwGtY8k1rMB5KyMYKfTVJpzJ71xBMeGXbiw2mcVLVnVwomDkjIJ3R\n"
    "IZPJBuYftekCqrO+PkQSQAdXepRvHNzd5ohEDxgiAkrCYV17PtFsApcTJ+Tq22aQ\n"
    "XwIDAQAB\n"
    "-----END PUBLIC KEY-----\n"
)
