"""Pure aggregation of reported usage and optional costs."""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict



class Usage(BaseModel):
    """Canonical persisted token usage for one model call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    web_search_requests: int = 0
    speed: str = ""

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            web_search_requests=(self.web_search_requests + other.web_search_requests),
            speed=self.speed or other.speed,
        )

    @classmethod
    def from_raw(cls, raw: dict | None) -> "Usage":
        """Parse a raw SDK or flattened transcript usage payload."""
        raw = raw or {}
        cache_creation = raw.get("cache_creation_input_tokens")
        if cache_creation is None:
            nested = raw.get("cache_creation") or {}
            cache_creation = (nested.get("ephemeral_5m_input_tokens", 0) or 0) + (
                nested.get("ephemeral_1h_input_tokens", 0) or 0
            )
        web_search = raw.get("web_search_requests")
        if web_search is None:
            server_tool_use = raw.get("server_tool_use") or {}
            web_search = server_tool_use.get("web_search_requests", 0)
        return cls(
            input_tokens=int(raw.get("input_tokens", 0) or 0),
            output_tokens=int(raw.get("output_tokens", 0) or 0),
            cache_read_input_tokens=int(raw.get("cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=int(cache_creation or 0),
            web_search_requests=int(web_search or 0),
            speed=str(raw.get("speed", "") or ""),
        )


def sum_costs(*costs: float | None) -> float | None:
    """An unknown component makes the total unknown, not zero."""
    return None if None in costs else sum(cost for cost in costs if cost is not None)


def sum_usage(*values: Mapping[str, float | None]) -> dict[str, float | None]:
    """Sum reported counters by name and preserve unknown values."""
    return {name: sum_costs(*(value.get(name, 0) for value in values))
            for name in dict.fromkeys(name for value in values for name in value)}


def sum_model_usage(*values: Mapping[str, Mapping[str, float | None]] | None) -> dict[str, dict[str, float | None]] | None:
    """Sum per-model reports; an absent report leaves the total unknown."""
    if any(value is None for value in values):
        return None
    reported = [value for value in values if value is not None]
    return {model: sum_usage(*(value[model] for value in reported if model in value))
            for model in dict.fromkeys(model for value in reported for model in value)}


__all__ = ('Usage', 'sum_usage', 'sum_costs', 'sum_model_usage')
