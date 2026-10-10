"""Store (BM25 + embeddings + hybrid RRF), tool router, tool-aware stub brain,
and the meeting diarizer smoke test."""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from agent_core.brain.stub import StubBrain
from agent_core.config import load
from agent_core.embedder import Embedder
from agent_core.meeting import diarize_segments, summarize_stub
from agent_core.store import Store
from agent_core.tools.router import ToolRouter, parse_intent
from tests.test_vad_turn import ROOT

CFG = load(ROOT / "configs" / "desktop.yaml")


@pytest.fixture()
def store(tmp_path) -> Store:
    s = Store(tmp_path / "test.db")
    sid = s.start_session("assistant")
    s.add_utterance(sid, 0, "user", "we planned the hackathon demo for tomorrow morning")
    s.add_utterance(sid, 1000, "assistant", "great, the speech pipeline is working offline")
    s.add_utterance(sid, 2000, "user", "remember to buy milk on the way home")
    return s


def test_store_bm25(store: Store) -> None:
    hits = store.search_bm25("hackathon demo", k=2)
    assert hits and "hackathon" in hits[0]["text"]


def test_store_hybrid_rrf(store: Store) -> None:
    emb = Embedder(str(ROOT / "models" / "embeddings"))
    hits = store.hybrid("buy milk shopping", emb.embed("buy milk shopping"), k=2)
    assert hits and "milk" in hits[0]["text"]


def test_intent_parsing() -> None:
    assert parse_intent("what is the weather in Hyderabad").tool == "weather"
    assert parse_intent("search for AI models released this month").tool == "search"
    assert parse_intent("send me the summary on my phone").tool == "telegram"
    assert parse_intent("tell me a joke") is None


def test_stub_brain_routes_tools(store: Store) -> None:
    brain = StubBrain(pace_ms=0, router=ToolRouter(CFG), store=store,
                      embedder=Embedder(str(ROOT / "models" / "embeddings")))

    async def get_reply(text: str) -> str:
        return "".join([c async for c in brain.respond(text, [])])

    # Memory question must NOT hit the canned reply.
    mem = asyncio.run(get_reply("what did we talk about earlier?"))
    assert "remember" in mem.lower() and "stub" not in mem.lower()

    # Unknown tool query stays offline (search disabled flag keeps it deterministic? enabled).
    # Weather with a bogus place returns a spoken failure, not an exception.
    wx = asyncio.run(get_reply("what is the weather in NotARealPlace123"))
    assert "could not find" in wx or "weather" in wx


def test_groq_brain_messages_and_fallback(store: Store) -> None:
    from agent_core.brain.groq import GroqBrain

    brain = GroqBrain(api_key_env="GROQ_API_KEY_TEST_SHOULD_NOT_EXIST", store=store,
                      embedder=Embedder(str(ROOT / "models" / "embeddings")),
                      router=ToolRouter(CFG))
    assert not brain.has_key()

    # Message building: RAG context included, user turn last.
    msgs = brain._messages("what did we discuss about the demo?", [])
    assert msgs[-1]["role"] == "user"
    assert any("RELEVANT PAST UTTERANCES" in m["content"] for m in msgs)
    assert any("hackathon" in m["content"] for m in msgs)

    # Keyless: respond() degrades to a spoken message, never an exception.
    async def get_reply() -> str:
        return "".join([c async for c in brain.respond("hello", [])])

    reply = asyncio.run(get_reply())
    assert "cannot think" in reply


def test_meeting_diarizer_runs(tmp_path) -> None:
    """Smoke: the torch-free embedding extractor labels every segment.

    Real 2-3 speaker quality is verified manually (see docs/TESTING.md).
    """
    rng = np.random.default_rng(7)
    segs = [(0, (rng.standard_normal(16000) * 0.1).astype(np.float32)),
            (2000, (rng.standard_normal(16000) * 0.1).astype(np.float32))]
    labels = diarize_segments(segs, str(ROOT / CFG.meeting.diarize.embedding))
    assert len(labels) == 2 and all(l.startswith("SPEAKER_") or l == "SPEAKER_?" for l in labels)


def test_meeting_summary_stub() -> None:
    entries = [{"speaker": "SPEAKER_1", "text": "we should finish the demo"},
               {"speaker": "SPEAKER_2", "text": "the pipeline is ready"}]
    s = summarize_stub(entries, duration_ms=120000)
    assert "2 spoken segments" in s and "speaker(s)" in s
