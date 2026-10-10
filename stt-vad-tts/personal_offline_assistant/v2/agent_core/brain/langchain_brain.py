"""LangChainBrain — tool-calling agent via LangChain `create_agent` (port of agent.py).

Same model configuration, summarization middleware, SQLite memory, tools and
tool-call trace as agent.py, behind the Brain seam the voice pipeline uses:
the speech transcript of each user turn is the agent's input.

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
import sqlite3
import sys
from collections.abc import AsyncIterator
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from agent_core.brain.base import Brain
from agent_core.tools import search, telegram, weather

load_dotenv()

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "lfm2.5")
LLM_API_KEY = os.getenv("LLM_API_KEY", "no-key-required")

SUMMARY_PROMPT = """
Summarize the conversation accurately for future use.
Preserve user facts, preferences, decisions, important tool
results, and completed actions. Never invent facts.
Mark unknown information as unknown.
Use concise sections for USER PROFILE, CONVERSATION CONTEXT,
COMPLETED ACTIONS, and UNCERTAINTIES.
Always prefix the summary with 'Previous Convo:'.

Conversation:
{messages}
"""


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
# Tools
# ---------------------------------------------------------------------------

@tool
def get_current_date_and_time() -> str:
    """Get the current date and time in India (works fully offline)."""
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


@tool
def send_telegram_message(message: str) -> str:
    """Send a message to the default Telegram recipient. Requires internet."""
    err = _require_internet("send_telegram_message")
    if err:
        return err
    try:
        return telegram.send_telegram_message(message)
    except Exception as exc:  # noqa: BLE001
        return f"Telegram send failed: {exc}"


@tool
def get_weather(location: str) -> str:
    """Get current weather and a 3-day forecast for a location. Requires internet."""
    err = _require_internet("get_weather")
    if err:
        return err
    try:
        return weather.get_weather(location)
    except Exception as exc:  # noqa: BLE001
        return f"Weather lookup failed: {exc}"


@tool
def web_search(query: str) -> str:
    """Search the web for information, facts, and recent news. Requires internet."""
    err = _require_internet("web_search")
    if err:
        return err
    try:
        return search.web_search(query)
    except Exception as exc:  # noqa: BLE001
        return f"Web search failed: {exc}"


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are Tobi, a helpful voice assistant powered by LFM2.5, running on a Raspberry Pi.
Your words are spoken aloud — keep answers concise (≤ 3 sentences) unless detail is requested.

Tools:
- get_current_date_and_time: Get current India time. Works offline.
- recall_conversation: Retrieve what was said in the last N minutes. Works offline.
- send_telegram_message: Send a Telegram message. Needs internet.
- get_weather: Get current weather and forecast for a location. Needs internet.
- web_search: Search the web for information. Needs internet.

Rules:
- Use tools when the request requires them.
- Always call get_current_date_and_time for current date/time questions.
- Always call send_telegram_message for each new send request.
- Use get_weather for weather questions.
- Use web_search for online research and recent information.
- Use recall_conversation for questions about what was said earlier.
- For weather requests that also ask to send a Telegram message,
  call get_weather first and then send_telegram_message.
- Use actual tool results. Never invent results or claim success
  when a tool fails.
- If a tool fails because the device is offline, relay that to the user.
- The default Telegram recipient is the configured username.
- Keep answers concise.
"""


# ---------------------------------------------------------------------------
# Brain class
# ---------------------------------------------------------------------------

class LangChainBrain(Brain):
    """LangChain tool-calling agent wired to the voice pipeline.

    Args:
        memory_db: Path to the SQLite file used for LangGraph checkpointing.
        store: agent_core.store.Store instance for recall_conversation tool.
    """

    def __init__(self, *, memory_db: str = "agent_memory.db", store=None) -> None:
        self.store = store

        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=0,
        )

        self.summary_llm = ChatOpenAI(
            model=os.getenv("SUMMARY_MODEL", LLM_MODEL),
            base_url=os.getenv("SUMMARY_BASE_URL", LLM_BASE_URL),
            api_key=os.getenv("SUMMARY_API_KEY", LLM_API_KEY),
            temperature=0,
        )

        # SQLite checkpointer — persistent cross-session memory
        self.connection = sqlite3.connect(memory_db, check_same_thread=False)
        self.checkpointer = SqliteSaver(self.connection)
        self.checkpointer.setup()

        self.summarizer = SummarizationMiddleware(
            model=self.summary_llm,
            trigger=("tokens", 2000),
            keep=("messages", 6),
            summary_prompt=SUMMARY_PROMPT,
        )

        self._tools = self._build_tools()
        self.agent = create_agent(
            model=self.llm,
            tools=self._tools,
            system_prompt=SYSTEM_PROMPT,
            middleware=[self.summarizer],
            checkpointer=self.checkpointer,
        )

    # ------------------------------------------------------------------
    # Tool construction
    # ------------------------------------------------------------------

    def _build_tools(self) -> list:
        store = self.store

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

        return [
            get_current_date_and_time,
            recall_conversation,
            send_telegram_message,
            get_weather,
            web_search,
        ]

    # ------------------------------------------------------------------
    # Agent run with tool trace
    # ------------------------------------------------------------------

    def run_agent(self, query: str, thread_id: str = "voice-default") -> dict:
        """Invoke the agent once; return the reply plus the tool calls made in this run."""
        config = {"configurable": {"thread_id": thread_id}}

        # Tool calls already in the checkpointed thread belong to earlier runs.
        previous_state = self.agent.get_state(config)
        previous_call_ids = set()
        if previous_state and previous_state.values:
            for old_message in previous_state.values.get("messages", []):
                for call in getattr(old_message, "tool_calls", None) or []:
                    call_id = call.get("id")
                    if call_id:
                        previous_call_ids.add(call_id)

        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
        )

        trace = {
            "query": query,
            "thread_id": thread_id,
            "tool_calls": [],
            "final_response": "",
        }
        current_calls = {}
        for message in result["messages"]:
            for call in getattr(message, "tool_calls", None) or []:
                call_id = call.get("id")
                if not call_id or call_id in previous_call_ids:
                    continue
                entry = {
                    "tool_call_id": call_id,
                    "tool_name": call.get("name"),
                    "arguments": call.get("args"),
                    "output": None,
                }
                trace["tool_calls"].append(entry)
                current_calls[call_id] = entry
            if isinstance(message, ToolMessage):
                entry = current_calls.get(message.tool_call_id)
                if entry is not None:
                    entry["output"] = message.content

        trace["final_response"] = result["messages"][-1].content
        return trace

    # ------------------------------------------------------------------
    # Brain interface
    # ------------------------------------------------------------------

    async def respond(self, user_text: str, history: list) -> AsyncIterator[str]:
        """Run the agent on the speech transcript and stream the reply word-by-word.

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

        try:
            trace = await asyncio.get_running_loop().run_in_executor(
                None, self.run_agent, user_text, thread_id)
            reply = trace["final_response"]
            if trace["tool_calls"]:
                print("[brain] tool calls: "
                      + json.dumps(trace["tool_calls"], ensure_ascii=False, default=str),
                      file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001
            reply = f"The agent encountered an error: {exc}"

        # Stream word-by-word so TTS sentences can form while we speak
        for word in reply.split(" "):
            yield word + " "
            await asyncio.sleep(0)

    def reset(self) -> None:
        """No-op: LangGraph checkpointer retains memory across resets."""
