"""GroqBrain — DEV-TIME testing bridge over Groq's OpenAI-compatible API.

NOT sovereign: this exists so the full loop (memory + retrieval + summarization
+ tool phrasing) can be exercised with a real LLM before LFM2.5 lands on the
Pi. The product path stays StubBrain -> LlmBrain(llama.cpp, localhost).

Same seam as every other brain: respond() streams text chunks.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator

import httpx

from agent_core.brain.base import Brain

_API = "https://api.groq.com/openai/v1/chat/completions"

_SYSTEM = (
    "You are tobi, an offline voice assistant running on a Raspberry Pi. "
    "Reply in at most 3 short sentences — your words are spoken aloud. "
    "Use the CONTEXT block when it is relevant, but never mention the block, "
    "the labels, or where the information came from — just answer naturally."
)

_MEMORY_RE = re.compile(
    r"\bwhat did we (?:talk|discuss|say)\b|\bsummar\w*\b|\btranscript\b|\bremember\b|\brecap\b", re.I)


class GroqBrain(Brain):
    def __init__(self, *, api_key_env: str = "GROQ_API_KEY", model: str = "llama-3.3-70b-versatile",
                 store=None, embedder=None, router=None) -> None:
        self.api_key_env = api_key_env
        self.model = model
        self.store = store
        self.embedder = embedder
        self.router = router

    def has_key(self) -> bool:
        return bool(os.environ.get(self.api_key_env, "").strip())

    def _context(self, user_text: str) -> str:
        """RAG context: hybrid retrieval + full recent transcript for summaries."""
        parts: list[str] = []
        if self.store is not None:
            emb = self.embedder.embed(user_text) if self.embedder is not None else None
            hits = self.store.hybrid(user_text, emb, k=5)
            if hits:
                parts.append("RELEVANT PAST UTTERANCES:\n" + "\n".join(
                    f"[{h['role']}] {h['text']}" for h in hits))
            if _MEMORY_RE.search(user_text):
                recent = self.store.recent(limit=60)
                if recent:
                    parts.append("RECENT TRANSCRIPT:\n" + "\n".join(
                        f"[{u['role']}] {u['text']}" for u in recent))
        if self.router is not None and self.router.is_tool_query(user_text):
            body = None
            if re.search(r"\bsummary\b|\btranscript\b", user_text, re.I) and self.store is not None:
                body = "\n".join(f"[{u['role']}] {u['text']}" for u in self.store.recent(limit=60))
            try:
                parts.append("TOOL RESULT (already executed):\n" +
                             self.router.run(user_text, transcript_for_sending=body))
            except Exception as exc:  # noqa: BLE001 — spoken failure, not a crash
                parts.append(f"TOOL RESULT: the tool call failed ({exc}).")
        return "\n\n".join(parts)

    def _messages(self, user_text: str, history: list) -> list[dict]:
        msgs = [{"role": "system", "content": _SYSTEM}]
        ctx = self._context(user_text)
        if ctx:
            msgs.append({"role": "system", "content": f"CONTEXT:\n{ctx}"})
        for turn in history[-8:]:
            msgs.append({"role": turn.role, "content": turn.text})
        msgs.append({"role": "user", "content": user_text})
        return msgs

    async def respond(self, user_text: str, history: list) -> AsyncIterator[str]:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            yield f"Groq key {self.api_key_env} is not set, so I cannot think right now."
            return
        payload = {
            "model": self.model,
            "messages": self._messages(user_text, history),
            "stream": True,
            "temperature": 0.5,
            "max_tokens": 300,
        }
        if self.model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = "low"  # voice loop: TTFT beats deliberation
        headers = {"Authorization": f"Bearer {key}"}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                async with client.stream("POST", _API, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
                        if delta:
                            yield delta
        except Exception as exc:  # noqa: BLE001 — a dead cloud must not kill the turn
            yield f"The Groq call failed: {exc}"
