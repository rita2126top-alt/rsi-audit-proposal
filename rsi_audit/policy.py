"""A strict declarative mutation space; no executable plugins are accepted."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Callable

from .common import digest

FEEDBACK_STYLES = {"compact", "detailed"}
REPAIR_ORDERS = {"syntax_then_examples", "examples_then_syntax"}


@dataclass(frozen=True)
class Policy:
    strategy_prompt: str
    feedback_style: str = "compact"
    repair_order: str = "syntax_then_examples"

    @property
    def sha256(self) -> str:
        return digest(asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: dict, count_tokens: Callable[[str], int], max_tokens: int = 1024) -> "Policy":
        keys = {"strategy_prompt", "feedback_style", "repair_order"}
        if not isinstance(obj, dict) or set(obj) != keys:
            raise ValueError("Policy must contain exactly strategy_prompt, feedback_style, repair_order")
        text = obj["strategy_prompt"]
        if not isinstance(text, str) or not text.strip() or len(text) > 32768:
            raise ValueError("Policy prompt must be nonempty bounded text")
        if count_tokens(text) > max_tokens:
            raise ValueError("Policy exceeds the frozen tokenizer budget")
        if (not isinstance(obj["feedback_style"], str) or not isinstance(obj["repair_order"], str)
                or obj["feedback_style"] not in FEEDBACK_STYLES or obj["repair_order"] not in REPAIR_ORDERS):
            raise ValueError("Unsupported policy enum")
        return cls(**obj)

    @classmethod
    def from_response(cls, text: str, count_tokens: Callable[[str], int], max_tokens: int = 1024) -> "Policy":
        text = text.strip()
        if text.startswith("```json\n") and text.endswith("```"):
            text = text[8:-3].strip()
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate policy key")
                result[key] = value
            return result
        try:
            parsed = json.loads(text, object_pairs_hook=unique)
        except RecursionError as exc:
            raise ValueError("Policy JSON is nested too deeply") from exc
        return cls.from_dict(parsed, count_tokens, max_tokens)


BASELINE = Policy(
    "Read the task contract carefully. Implement a complete, simple Python standard-library solution. "
    "Check boundary cases and return types. Return only the complete function and any required imports."
)
