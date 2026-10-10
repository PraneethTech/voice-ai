"""Send messages to the user's phone via Telegram Bot API — free, no card.

One-time setup (5 minutes): message @BotFather -> /newbot -> token; then send the
bot any message from the recipient account. Set TELEGRAM_BOT_TOKEN and
TELEGRAM_USERNAME in .env.
"""
from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Default recipient. Change this to your intended Telegram username.
DEFAULT_TELEGRAM_USERNAME = os.getenv(
    "TELEGRAM_USERNAME", "tobi_reddy__"
)


def send_telegram_message(message: str) -> str:
    """Send a message to the default Telegram recipient."""

    if not BOT_TOKEN:
        return "Set the TELEGRAM_BOT_TOKEN environment variable."

    response = requests.get(
        f"{BASE_URL}/getUpdates",
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    if not data.get("ok"):
        return f"Telegram API error: {data}"

    updates = data.get("result", [])
    target_username = DEFAULT_TELEGRAM_USERNAME.lstrip("@").lower()

    first_chat_id = None
    matched_chat_id = None

    for update in updates:
        message_data = update.get("message", {})
        chat = message_data.get("chat", {})
        sender = message_data.get("from", {})
        chat_id = chat.get("id")

        if first_chat_id is None and chat_id is not None:
            first_chat_id = chat_id

        sender_username = (
            sender.get("username") or ""
        ).lower()

        chat_username = (
            chat.get("username") or ""
        ).lower()

        if (
            sender_username == target_username
            or chat_username == target_username
        ):
            matched_chat_id = chat_id

    if matched_chat_id is None:
        return (
            f"Could not find Telegram user @{target_username} "
            "in recent bot updates. Ask the recipient to message "
            "the bot first and check the configured username."
        )

    response = requests.post(
        f"{BASE_URL}/sendMessage",
        json={
            "chat_id": matched_chat_id,
            "text": message,
        },
        timeout=10,
    )
    response.raise_for_status()
    result = response.json()

    if not result.get("ok"):
        return f"Telegram send failed: {result}"

    return f"Message sent successfully: {message}"
