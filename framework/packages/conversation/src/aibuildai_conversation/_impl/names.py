"""Canonical model identifiers shared by callers and providers."""

import re

_DATE_SUFFIX_RE = re.compile(r"-\d{8}(?=(\[[^\]]+\])?$)")


def canonical_model_name(model: str) -> str:
    """One canonical model id: strip the trailing -YYYYMMDD date stamp (keeping any [variant] suffix). Pricing needles in aibuildai_conversation/model_catalog.json are substrings, so canonical names still resolve to the same tier.

    'claude-haiku-4-5-20251001' -> 'claude-haiku-4-5'; 'claude-opus-4-8-20260514[1m]' -> 'claude-opus-4-8[1m]'. Keeps distinct versions distinct (opus-4-7 vs opus-4-8). Use this when keying per-model aggregates so an aliased request id and the dated snapshot the SDK reports back don't split into two rows.
    """
    return _DATE_SUFFIX_RE.sub("", model)
