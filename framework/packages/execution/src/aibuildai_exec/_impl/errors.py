"""Public errors for unavailable execution environments and failed launches."""

class ExecutionError(RuntimeError):
    """The host cannot provide the requested execution or its cleanup failed."""
