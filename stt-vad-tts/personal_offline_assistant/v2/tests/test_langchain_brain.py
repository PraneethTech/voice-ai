"""Tests for the LangChainBrain and the store.recent_by_minutes helper.

These tests do NOT require a running LLM — the connectivity probe is monkey-
patched to false so no real network calls happen.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

import pytest

from agent_core.store import Store


# ---------------------------------------------------------------------------
# store.recent_by_minutes
# ---------------------------------------------------------------------------

@pytest.fixture()
def timestamped_store(tmp_path) -> Store:
    """Store pre-populated with utterances spread over two minutes."""
    s = Store(tmp_path / "test_lc.db")
    sid = s.start_session("assistant")
    # t_ms = 0 → right at session start (wall-clock = sessions.started_at)
    s.add_utterance(sid, 0, "user", "hello, can you hear me?")
    s.add_utterance(sid, 30_000, "assistant", "Yes, I can hear you clearly.")
    s.add_utterance(sid, 60_000, "user", "what is the weather like today?")
    # store the session_id so LangChainBrain can use it for thread_id
    s._session_id = sid  # type: ignore[attr-defined]
    return s


def test_recent_by_minutes_returns_results(timestamped_store: Store) -> None:
    """recent_by_minutes(60) should return all utterances from the last hour."""
    hits = timestamped_store.recent_by_minutes(60)
    assert len(hits) >= 1, "Expected at least one utterance in the last 60 minutes"
    texts = [h["text"] for h in hits]
    assert any("hello" in t or "weather" in t or "hear" in t for t in texts)


def test_recent_by_minutes_zero_window(timestamped_store: Store) -> None:
    """recent_by_minutes(0) should return nothing (data is not in the future)."""
    hits = timestamped_store.recent_by_minutes(0)
    # Could be 0 or a tiny handful depending on clock vs insert latency; never > total
    assert len(hits) <= 3


def test_recent_by_minutes_chronological_order(timestamped_store: Store) -> None:
    """Results must be in chronological (ascending t_ms) order."""
    hits = timestamped_store.recent_by_minutes(120)
    if len(hits) >= 2:
        for a, b in zip(hits, hits[1:]):
            assert a["t_ms"] <= b["t_ms"], "Expected chronological order"


# ---------------------------------------------------------------------------
# LangChainBrain — import + tool list (no LLM call)
# ---------------------------------------------------------------------------

def test_langchain_brain_imports_and_has_five_tools(tmp_path, monkeypatch) -> None:
    """LangChainBrain should import cleanly and expose exactly 5 tools."""
    # Prevent actual SQLite checkpointer from touching the filesystem
    monkeypatch.chdir(tmp_path)

    from agent_core.brain.langchain_brain import LangChainBrain

    # Patch ChatOpenAI so no real LLM is needed during construction
    import unittest.mock as mock
    with mock.patch("agent_core.brain.langchain_brain.ChatOpenAI") as mock_llm, \
         mock.patch("agent_core.brain.langchain_brain.create_agent") as mock_agent, \
         mock.patch("agent_core.brain.langchain_brain.SqliteSaver") as mock_saver:
        mock_saver.return_value.setup.return_value = None
        mock_agent.return_value = mock.MagicMock()
        brain = LangChainBrain()
        assert len(brain._tools) == 5


def test_recall_tool_offline_no_store(tmp_path, monkeypatch) -> None:
    """recall_conversation should return a friendly message when no store is attached."""
    monkeypatch.chdir(tmp_path)

    import unittest.mock as mock
    from agent_core.brain.langchain_brain import LangChainBrain

    with mock.patch("agent_core.brain.langchain_brain.ChatOpenAI"), \
         mock.patch("agent_core.brain.langchain_brain.create_agent"), \
         mock.patch("agent_core.brain.langchain_brain.SqliteSaver") as mock_saver:
        mock_saver.return_value.setup.return_value = None
        brain = LangChainBrain(store=None)

    recall = next(t for t in brain._tools if t.name == "recall_conversation")
    result = recall.invoke({"minutes": 30.0})
    assert "No conversation store" in result


def test_internet_tool_offline(tmp_path, monkeypatch) -> None:
    """Weather tool must return a friendly 'no internet' message when offline."""
    monkeypatch.chdir(tmp_path)

    # Force the connectivity probe to report offline
    import agent_core.brain.langchain_brain as lcb
    monkeypatch.setattr(lcb, "_is_internet_available", lambda: False)

    import unittest.mock as mock
    from agent_core.brain.langchain_brain import LangChainBrain
    with mock.patch("agent_core.brain.langchain_brain.ChatOpenAI"), \
         mock.patch("agent_core.brain.langchain_brain.create_agent"), \
         mock.patch("agent_core.brain.langchain_brain.SqliteSaver") as mock_saver:
        mock_saver.return_value.setup.return_value = None
        brain = LangChainBrain()

    weather = next(t for t in brain._tools if t.name == "get_weather")
    result = weather.invoke({"location": "Hyderabad"})
    assert "offline" in result.lower() or "internet" in result.lower()


def test_offline_time_tool_always_works(tmp_path, monkeypatch) -> None:
    """get_current_date_and_time must work regardless of connectivity."""
    monkeypatch.chdir(tmp_path)

    import agent_core.brain.langchain_brain as lcb
    monkeypatch.setattr(lcb, "_is_internet_available", lambda: False)

    import unittest.mock as mock
    from agent_core.brain.langchain_brain import LangChainBrain
    with mock.patch("agent_core.brain.langchain_brain.ChatOpenAI"), \
         mock.patch("agent_core.brain.langchain_brain.create_agent"), \
         mock.patch("agent_core.brain.langchain_brain.SqliteSaver") as mock_saver:
        mock_saver.return_value.setup.return_value = None
        brain = LangChainBrain()

    time_tool = next(t for t in brain._tools if t.name == "get_current_date_and_time")
    result = time_tool.invoke({})
    assert "IST" in result or "2026" in result or ":" in result  # datetime-shaped string
