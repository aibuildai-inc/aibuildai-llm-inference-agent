"""Read the account label for the effective launch environment."""
import json
from collections.abc import Mapping
from ._impl.claude.config_paths import state_file

__all__ = ("account_email",)

def account_email(env: Mapping[str, str]) -> str:
    """The host account email from ``state_file(env)``'s ``oauthAccount.emailAddress``, or the literal ``'local'`` when the state file is absent / unparseable / carries no email.

    The absent-or-empty case is a genuine "we do not know the email yet" (a host that has never completed an interactive ``claude`` login), not an error to hide; the label is cosmetic, so a corrupt state file degrades to ``'local'`` rather than crashing the surface that reads it."""
    sf = state_file(env)
    if sf.exists():
        try:
            data = json.loads(sf.read_text())
        except (json.JSONDecodeError, OSError):
            return "local"
        email = (data.get("oauthAccount") or {}).get("emailAddress")
        if email:
            return email
    return "local"
