"""Web search via DuckDuckGo Lite — no API key. Online lookup tool."""
from __future__ import annotations

import re

import httpx

_LITE = "https://lite.duckduckgo.com/lite/"
_TAG = re.compile(r"<[^>]+>")


def web_search(query: str, max_results: int = 3) -> str:
    with httpx.Client(timeout=10, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) tobi-offline-demo"}) as client:
        resp = client.post(_LITE, data={"q": query})
        resp.raise_for_status()
    rows = re.findall(r'<a rel="nofollow" href="([^"]+)"[^>]*>(.*?)</a>', resp.text)
    out = []
    for url, title in rows[:max_results]:
        title = _TAG.sub("", title).strip()
        if title:
            out.append(f"- {title} ({url})")
    return ("Top results for that:\n" + "\n".join(out)) if out else \
        "I searched but found nothing useful for that."
