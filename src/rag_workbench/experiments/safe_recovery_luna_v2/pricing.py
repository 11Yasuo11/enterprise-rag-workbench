"""Configurable OpenAI short-context list prices for SAFE_RECOVERY_LUNA_V2.

Source: official OpenAI GPT-5.6 model docs consulted 2026-08-20.
Cache-write rate is 1.25x uncached input when the API exposes write tokens.
Do not scatter these numbers elsewhere.
"""

from __future__ import annotations

from typing import Any

PRICING: dict[str, Any] = {
    "source": {
        "luna": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
        "sol": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
        "pricing": "https://developers.openai.com/api/docs/pricing",
        "consulted_at": "2026-08-20",
    },
    "context": "short-context <=272K list prices",
    "models": {
        "gpt-5.6-luna": {
            "input_per_million": 0.20,
            "cached_input_per_million": 0.02,
            "cache_write_per_million": 0.25,
            "output_per_million": 1.20,
        },
        "gpt-5.6-sol": {
            "input_per_million": 5.00,
            "cached_input_per_million": 0.50,
            "cache_write_per_million": 6.25,
            "output_per_million": 30.00,
        },
    },
}


def estimate_cost_usd(
    *,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    prices = PRICING["models"].get(model)
    if prices is None:
        return 0.0
    uncached = max(int(input_tokens) - int(cached_input_tokens) - int(cache_write_tokens), 0)
    return (
        uncached * prices["input_per_million"]
        + int(cached_input_tokens) * prices["cached_input_per_million"]
        + int(cache_write_tokens) * prices["cache_write_per_million"]
        + int(output_tokens) * prices["output_per_million"]
    ) / 1_000_000
