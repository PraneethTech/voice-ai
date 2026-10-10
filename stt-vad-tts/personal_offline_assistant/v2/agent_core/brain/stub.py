"""StubBrain — hardcoded streaming responses + keyword tool routing.

Proves the whole loop (streaming text -> sentence TTS -> barge-in) and the tool
seam before the real SLM lands. Later: brain/llm_openai.py pointing at a
localhost llama.cpp server implements the same interface; function calling
replaces the keyword router; the tools themselves stay unchanged.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator

from agent_core.brain.base import Brain

_REPLIES = [
    "I hear you. I am a hardcoded stub right now, but the whole voice loop around me is live.",
    "Copy that. Everything you just heard ran on this device, fully offline.",
    "Understood. Swap me for a real language model and I will actually think about that.",
]

_MEMORY_RE = re.compile(
    r"\bwhat did we (?:talk|discuss|say)\b|\b(?:earlier|last time|before)\b.*\?"
    r"|\bsummar\w*\b.*\b(?:conversation|discussion|meeting|chat)\b"
    r"|\btranscript\b|\bremember\b.*\babout\b", re.I)


class StubBrain(Brain):
    def __init__(self, pace_ms: float = 25.0, *, router=None, store=None, embedder=None) -> None:
        self.pace_ms = pace_ms
        self._i = 0
        self.router = router
        self.store = store
        self.embedder = embedder

    async def respond(self, user_text: str, history: list) -> AsyncIterator[str]:
        reply = self._compose(user_text)
        for word in reply.split(" "):
            yield word + " "
            await asyncio.sleep(self.pace_ms / 1000)

    def _compose(self, user_text: str) -> str:
        if not user_text.strip():
            return "I did not catch that. Say it again?"

        # Memory / transcript questions -> hybrid retrieval over stored transcripts.
        if self.store is not None and _MEMORY_RE.search(user_text):
            emb = self.embedder.embed(user_text) if self.embedder is not None else None
            hits = self.store.hybrid(user_text, emb, k=4)
            if not hits:
                return "My memory is empty so far. Talk to me first, then ask again."
            lines = [f"{h['role']}: {h['text'][:90]}" for h in hits]
            return "Here is what I remember: " + " | ".join(lines)

        # Tool intents (weather / search / telegram). The only online path.
        if self.router is not None and self.router.is_tool_query(user_text):
            body = None
            if re.search(r"\bsummary\b|\btranscript\b", user_text, re.I) and self.store is not None:
                body = "\n".join(f"{u['role']}: {u['text']}" for u in self.store.recent(limit=40))
            try:
                return self.router.run(user_text, transcript_for_sending=body)
            except Exception as exc:  # noqa: BLE001 — a dead tool must not kill the turn
                return f"The tool call failed: {exc}"

        reply = _REPLIES[self._i % len(_REPLIES)]
        self._i += 1
        if user_text.strip():
            reply += f" You said: {user_text.strip()[:80]}."
        return reply
