"""Intent router — the tool-calling seam.

Tonight the StubBrain routes with keyword patterns. When the real LLM lands it
replaces this with native function calling; the tool functions stay unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from agent_core.config import Config
from agent_core.tools.search import web_search
from agent_core.tools.telegram import send_telegram_message
from agent_core.tools.weather import get_weather


@dataclass(frozen=True)
class Intent:
    tool: str
    arg: str


_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("weather", re.compile(r"\bweather\b|\btemperature\b.*\b(?:in|at)\b|\brain\b.*\btoday\b", re.I)),
    ("search", re.compile(r"\b(?:search|look ?up|google|find|latest|news|released?)\b|\bwho is\b|\bwhat happened\b", re.I)),
    ("telegram", re.compile(r"\b(?:send|share|forward)\b.*\b(?:telegram|phone|my (?:mobile|cell))\b|\btelegram\b", re.I)),
]


def parse_intent(text: str) -> Intent | None:
    for tool, pat in _PATTERNS:
        if pat.search(text):
            return Intent(tool, text)
    return None


import time
from collections.abc import Callable


class ToolRouter:
    def __init__(self, cfg: Config, on_event: Callable[[dict], None] | None = None) -> None:
        self.cfg = cfg
        self.on_event = on_event

    def is_tool_query(self, text: str) -> bool:
        return parse_intent(text) is not None

    def run(self, text: str, *, transcript_for_sending: str | None = None) -> str:
        """Execute the tool the query implies. Only this method may use the network."""
        intent = parse_intent(text)
        if intent is None:
            return "I do not have a tool for that yet."
        tool = intent.tool
        tool_id = f"t_{int(time.time() * 1000) % 10000}"
        if self.on_event:
            self.on_event({"type": "tool", "id": tool_id, "name": tool, "status": "start"})

        try:
            if tool == "weather":
                if not self.cfg.tools.weather.enabled:
                    res = "The weather tool is disabled."
                else:
                    place = _place_from(text)
                    res = get_weather(place)
            elif tool == "search":
                if not self.cfg.tools.search.enabled:
                    res = "Search is disabled."
                else:
                    res = web_search(text)
            elif tool == "telegram":
                if not self.cfg.tools.telegram.enabled:
                    res = "Telegram is disabled."
                else:
                    body = transcript_for_sending or text
                    res = send_telegram_message(body)
            else:
                res = "Unknown tool."

            if self.on_event:
                detail = res.split("\n")[0][:30] if res else "done"
                self.on_event({"type": "tool", "id": tool_id, "name": tool, "status": "ok", "detail": detail})
            return res
        except Exception as exc:
            if self.on_event:
                self.on_event({"type": "tool", "id": tool_id, "name": tool, "status": "error", "detail": str(exc)[:30]})
            raise


def _place_from(text: str) -> str:
    m = re.search(r"\b(?:in|at|for)\s+([A-Za-z][A-Za-z ]{2,30})", text, re.I)
    return (m.group(1).strip() if m else "Hyderabad").rstrip("?.!")
