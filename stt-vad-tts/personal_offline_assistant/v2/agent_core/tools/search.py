"""Web search via Tavily — needs TAVILY_API_KEY. Online lookup tool."""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


def web_search(query: str) -> str:
    """Search the web for information, facts, and recent news."""

    if tavily_client is None:
        return "Set the TAVILY_API_KEY environment variable."

    try:
        response = tavily_client.search(
            query=query,
            max_results=5,
            search_depth="basic",
        )

        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
            for item in response.get("results", [])
        ]

        return json.dumps(results, ensure_ascii=False)

    except Exception as exc:  # noqa: BLE001 — a failed lookup is reported, not raised
        return f"Web search error: {exc}"
