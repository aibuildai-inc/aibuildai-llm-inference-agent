"""Physical operations borrowed from the caller."""
from collections.abc import Collection, Mapping
from typing import Protocol

__all__ = ("Confinement",)

class Confinement(Protocol):
    """The launch operations a conversation borrows from its physical owner."""

    @property
    def apply_bwrap(self) -> bool: ...
    def memory_kills(self) -> int | None: ...
    def delegates_commands(self) -> bool: ...
    def wrap_command(self, command: str, tool_use_id: str) -> str: ...
    def reap_command(self, tool_use_id: str) -> None: ...
    async def terminate_command(self, tool_use_id: str) -> None:
        """Stop and remove one command and its descendant processes."""
        ...
    def as_script(
        self, inner_argv: list[str], *, setenv: dict[str, str] | None = None,
        reopen_fds: Mapping[int, str] | None = None,
        environment_keys: Collection[str] | None = None, prelude: str = "",
        read_paths: Collection[str] = (), write_paths: Collection[str] = (),
    ) -> str: ...
