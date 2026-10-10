"""Send messages to the user's phone via Telegram Bot API — free, no card.

One-time setup (5 minutes): message @BotFather -> /newbot -> token; send the bot
any message, then GET getUpdates to read your chat_id. Put both in the env vars
named by the config (TOBI_TG_TOKEN / TOBI_TG_CHAT).
"""
from __future__ import annotations

import os

import httpx


def send_telegram(token_env: str, chat_env: str, text: str) -> str:
    token = os.environ.get(token_env, "").strip()
    chat_id = os.environ.get(chat_env, "").strip()
    if not token or not chat_id:
        return ("I could not send that: Telegram is not configured. "
                f"Set {token_env} and {chat_id} in the environment first.")
    resp = httpx.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text[:4000]},
        timeout=10,
    )
    if resp.status_code == 200:
        return "Sent to your Telegram."
    return f"Telegram refused the message (HTTP {resp.status_code})."
