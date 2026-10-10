"""LangChainBrain — ReAct agent via LangChain/LangGraph.

Replaces the keyword ToolRouter with a proper function-calling agent.

Tool implementations stay in agent_core/tools/* (sovereign, offline-first).
Network-bound tools (weather, search, telegram) are guarded by a deterministic
connectivity probe that rejects them gracefully when the device is offline,
without affecting offline-only tools (time, recall_conversation).

Same seam as every other brain: respond() is an async generator of text chunks.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import time
from collections.abc import AsyncIterator
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

# LangGraph ReAct agent + SQLite memory
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from agent_core.brain.base import Brain
from agent_core.tools.weather import get_weather as _get_weather_impl
from agent_core.tools.search import web_search as _web_search_impl
from agent_core.tools.telegram import send_telegram as _send_telegram_impl

# ---------------------------------------------------------------------------
# Internet connectivity probe
# ---------------------------------------------------------------------------

_PROBE_HOST = "1.1.1.1"   # Cloudflare DNS — fast, always up when online
_PROBE_PORT = 443
_PROBE_TIMEOUT = 2.0       # seconds


def _is_internet_available() -> bool:
    """Deterministic TCP probe. Returns True only when internet is reachable.

    Uses a raw socket connect — no DNS, no HTTP overhead — so it never throws
    and completes in < _PROBE_TIMEOUT seconds.
    """
    import socket
    try:
        with socket.create_connection((_PROBE_HOST, _PROBE_PORT), timeout=_PROBE_TIMEOUT):
            return True
    except OSError:
        return False


def _require_internet(tool_name: str) -> str | None:
    """Return an error string if internet is not available, else None."""
    if not _is_internet_available():
        return (
            f"Internet is required for {tool_name} but the device is currently offline. "
            "Please retry after an internet connection is established."
        )
    return None


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are Tobi, a helpful offline voice assistant running on a Raspberry Pi.
Your words are spoken aloud — keep answers concise (≤ 3 sentences) unless detail is requested.

Tools available:
- get_current_date_and_time: Always use for date/time questions (works offline).
- recall_conversation: Retrieve what was said in the last N minutes (works offline).
- get_weather: Needs internet. Get current weather for a location.
- web_search: Needs internet. Search the web for facts and recent news.
- send_telegram_message: Needs internet. Send a message to the user's phone.

Rules:
- Use tools when the request requires real data — never invent facts.
- For tool failures caused by no internet, relay the exact error message to the user.
- Keep spoken responses natural and brief.
"""


# ---------------------------------------------------------------------------
# Brain class
# ---------------------------------------------------------------------------

class LangChainBrain(Brain):
    """LangChain ReAct agent wired to the voice pipeline.

    Args:
        model: LLM model name (e.g. "lfm2.5", "llama-3.3-70b-versatile").
        base_url: OpenAI-compatible base URL.  Defaults to LLM_BASE_URL env var
                  or http://127.0.0.1:8080/v1 (llama.cpp default).
        api_key_env: Env var holding the API key (default "LLM_API_KEY").
        memory_db: Path to the SQLite file used for LangGraph checkpointing.
        store: agent_core.store.Store instance for recall_conversation tool.
        telegram_token_env: env var name holding the Telegram bot token.
        telegram_chat_id_env: env var name holding the Telegram chat ID.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "LLM_API_KEY",
        memory_db: str = "agent_memory.db",
        store=None,
        telegram_token_env: str = "TOBI_TG_TOKEN",
        telegram_chat_id_env: str = "TOBI_TG_CHAT",
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL", "lfm2.5")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
        self.api_key = os.getenv(api_key_env, "no-key-required")
        self.store = store
        self._tg_token_env = telegram_token_env
        self._tg_chat_env = telegram_chat_id_env

        self._llm = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=0,
        )

        # SQLite checkpointer — persistent cross-session memory
        self._conn = sqlite3.connect(memory_db, check_same_thread=False)
        self._checkpointer = SqliteSaver(self._conn)
        self._checkpointer.setup()

        self._tools = self._build_tools()
        self._agent = create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=_SYSTEM_PROMPT,
            checkpointer=self._checkpointer,
        )

    # ------------------------------------------------------------------
    # Tool construction
    # ------------------------------------------------------------------

    def _build_tools(self) -> list:
        store = self.store
        tg_token_env = self._tg_token_env
        tg_chat_env = self._tg_chat_env

        # ---- offline tools -----------------------------------------------

        @tool
        def get_current_date_and_time() -> str:
            """Get the current date and time (works fully offline)."""
            now = datetime.now(ZoneInfo("Asia/Kolkata"))
            return now.strftime("%Y-%m-%d %H:%M:%S %Z")

        @tool
        def recall_conversation(minutes: float = 30.0) -> str:
            """Retrieve what was said in the last `minutes` minutes of conversation.

            Works fully offline — reads from the local SQLite transcript store.
            Use minutes=30 for recent context, minutes=120 for a longer look-back.
            """
            if store is None:
                return "No conversation store is available yet."
            utterances = store.recent_by_minutes(minutes)
            if not utterances:
                return f"No conversation found in the last {minutes:.0f} minutes."
            lines = []
            for u in utterances:
                who = u.get("speaker") or u["role"]
                wall = u.get("wall_time", "")
                lines.append(f"[{wall}] {who}: {u['text']}")
            return "\n".join(lines)

        # ---- internet-required tools -------------------------------------

        @tool
        def get_weather(location: str) -> str:
            """Get current weather for a location. Requires internet access."""
            err = _require_internet("get_weather")
            if err:
                return err
            try:
                return _get_weather_impl(location)
            except Exception as exc:  # noqa: BLE001
                return f"Weather lookup failed: {exc}"

        @tool
        def web_search(query: str) -> str:
            """Search the web for information, facts, and recent news. Requires internet."""
            err = _require_internet("web_search")
            if err:
                return err
            try:
                return _web_search_impl(query, max_results=3)
            except Exception as exc:  # noqa: BLE001
                return f"Web search failed: {exc}"

        @tool
        def send_telegram_message(message: str) -> str:
            """Send a Telegram message to the user's phone. Requires internet."""
            err = _require_internet("send_telegram_message")
            if err:
                return err
            try:
                return _send_telegram_impl(tg_token_env, tg_chat_env, message)
            except Exception as exc:  # noqa: BLE001
                return f"Telegram send failed: {exc}"

        return [
            get_current_date_and_time,
            recall_conversation,
            get_weather,
            web_search,
            send_telegram_message,
        ]

    # ------------------------------------------------------------------
    # Brain interface
    # ------------------------------------------------------------------

    async def respond(self, user_text: str, history: list) -> AsyncIterator[str]:
        """Run the ReAct agent and stream the final reply word-by-word.

        The agent runs synchronously in a thread executor so the async voice
        pipeline is not blocked. The final text is then yielded word-by-word
        so TTS can begin streaming immediately.
        """
        # Each pipeline session gets a stable thread_id from the store session_id
        # so LangGraph memory is scoped per session.
        thread_id = (
            f"voice-{self.store._session_id}"  # type: ignore[union-attr]
            if self.store is not None and hasattr(self.store, "_session_id")
            else "voice-default"
        )
        config = {"configurable": {"thread_id": thread_id}}

        def _invoke() -> str:
            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": user_text}]},
                config=config,
            )
            return result["messages"][-1].content

        try:
            reply = await asyncio.get_event_loop().run_in_executor(None, _invoke)
        except Exception as exc:  # noqa: BLE001
            reply = f"The agent encountered an error: {exc}"

        # Stream word-by-word so TTS sentences can form while we speak
        for word in reply.split(" "):
            yield word + " "
            await asyncio.sleep(0)

    def reset(self) -> None:
        """No-op: LangGraph checkpointer retains memory across resets."""
