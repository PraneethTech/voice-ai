"""Brain seam: the only place reasoning plugs in.

The stub implementation streams a hardcoded string. The real model (LFM2.5 or
Qwen3 via llama.cpp on localhost) implements the same interface later — nothing
else in the pipeline changes.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    role: str  # "user" | "assistant"
    text: str


class Brain:
    async def respond(self, user_text: str, history: list[Turn]) -> AsyncIterator[str]:
        raise NotImplementedError
        yield  # pragma: no cover — makes this an async generator signature

    def reset(self) -> None:  # noqa: B027 — optional hook
        pass
