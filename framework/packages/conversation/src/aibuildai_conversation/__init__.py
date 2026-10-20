"""Caller-owned declarations and opaque model conversations."""
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._impl.spec import AgentSpec, Message, Image, SubAgent, HookSpec
    from ._impl.observer import ConversationObserver
    from ._impl.claude.conversation import Conversation, Response

__all__ = ("AgentSpec", "Conversation", "ConversationObserver", "Response", "Message", "Image", "SubAgent", "HookSpec")
_EXPORT_MODULES = {
    "AgentSpec": "._impl.spec", "Message": "._impl.spec", "Image": "._impl.spec",
    "SubAgent": "._impl.spec", "HookSpec": "._impl.spec",
    "Conversation": "._impl.claude.conversation", "Response": "._impl.claude.conversation",
    "ConversationObserver": "._impl.observer",
}


def __getattr__(name: str) -> Any:  # Any: the eight exports have distinct types.
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value
