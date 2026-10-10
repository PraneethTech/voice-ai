
import os
import json
import sqlite3
import requests

from datetime import datetime
from zoneinfo import ZoneInfo
from tavily import TavilyClient

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================


os.environ["TELEGRAM_BOT_TOKEN"] = TELEGRAM_BOT_TOKEN
os.environ["TELEGRAM_USERNAME"] = TELEGRAM_USERNAME
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "lfm2.5")
LLM_API_KEY = os.getenv("LLM_API_KEY", "no-key-required")

llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    temperature=0,
)


# ============================================================
# 2. SUMMARIZER MODEL
# ============================================================

summary_llm = ChatOpenAI(
    model=os.getenv("SUMMARY_MODEL", LLM_MODEL),
    base_url=os.getenv("SUMMARY_BASE_URL", LLM_BASE_URL),
    api_key=os.getenv("SUMMARY_API_KEY", LLM_API_KEY),
    temperature=0,
)

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


# ============================================================
# 3. TELEGRAM CONFIGURATION
# ============================================================

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not BOT_TOKEN:
    raise RuntimeError("Set the TELEGRAM_BOT_TOKEN environment variable.")

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Default recipient. Change this to your intended Telegram username.
DEFAULT_TELEGRAM_USERNAME = os.getenv(
    "TELEGRAM_USERNAME", "tobi_reddy__"
)


# ============================================================
# 4. TELEGRAM MESSAGE TOOL
# ============================================================

@tool
def send_telegram_message(message: str) -> str:
    """Send a message to the default Telegram recipient."""

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


# ============================================================
# 5. CURRENT TIME TOOL
# ============================================================

@tool
def get_current_date_and_time() -> str:
    """Get the current date and time in India."""

    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


# ============================================================
# 6. WEATHER TOOL — OPEN-METEO
# ============================================================

@tool
def get_weather(location: str) -> str:
    """Get current weather and a 3-day forecast for a location."""

    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            timeout=10,
        )
        geo_response.raise_for_status()

        places = geo_response.json().get("results", [])

        if not places:
            return f"Location not found: {location}"

        place = places[0]
        latitude = place["latitude"]
        longitude = place["longitude"]
        place_name = (
            f"{place['name']}, {place.get('country', '')}"
        )

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,relative_humidity_2m,"
                    "apparent_temperature,precipitation,"
                    "weather_code,wind_speed_10m"
                ),
                "daily": (
                    "weather_code,temperature_2m_max,"
                    "temperature_2m_min,precipitation_probability_max"
                ),
                "forecast_days": 3,
                "timezone": "auto",
            },
            timeout=10,
        )
        weather_response.raise_for_status()
        data = weather_response.json()

        current = data["current"]
        daily = data["daily"]

        # Convert WMO weather codes to readable descriptions.
        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            80: "Rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
            96: "Thunderstorm with hail",
            99: "Severe thunderstorm with hail",
        }

        current["condition"] = weather_codes.get(
            current.get("weather_code"), "Unknown"
        )

        forecast = []

        for i, date in enumerate(daily["time"]):
            code = daily["weather_code"][i]

            forecast.append({
                "date": date,
                "condition": weather_codes.get(code, "Unknown"),
                "max_temp_c": daily["temperature_2m_max"][i],
                "min_temp_c": daily["temperature_2m_min"][i],
                "precipitation_probability_percent": (
                    daily["precipitation_probability_max"][i]
                ),
            })

        return json.dumps({
            "location": place_name,
            "current": current,
            "forecast": forecast,
        }, ensure_ascii=False)

    except requests.RequestException as exc:
        return f"Weather API error: {exc}"
    except (KeyError, IndexError, ValueError) as exc:
        return f"Could not process weather data: {exc}"


# ============================================================
# 7. WEB SEARCH TOOL — TAVILY
# ============================================================

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise RuntimeError("Set the TAVILY_API_KEY environment variable.")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


@tool
def web_search(query: str) -> str:
    """Search the web for information, facts, and recent news."""

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

    except Exception as exc:
        return f"Web search error: {exc}"


# ============================================================
# 8. SQLITE CHECKPOINTER
# ============================================================

connection = sqlite3.connect(
    "agent_memory.db",
    check_same_thread=False,
)

checkpointer = SqliteSaver(connection)
checkpointer.setup()


# ============================================================
# 9. SUMMARIZATION MIDDLEWARE
# ============================================================

summarizer = SummarizationMiddleware(
    model=summary_llm,
    max_tokens_before_summary=2000,
    messages_to_keep=6,
    summary_prompt=SUMMARY_PROMPT,
)


# ============================================================
# 10. CREATE THE AGENT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful assistant powered by LFM2.5.

Tools:
- get_current_date_and_time: Get current India time.
- send_telegram_message: Send a Telegram message.
- get_weather: Get current weather and forecast for a location.
- web_search: Search the web for information.

Rules:
- Use tools when the request requires them.
- Always call get_current_date_and_time for current date/time questions.
- Always call send_telegram_message for each new send request.
- Use get_weather for weather questions.
- Use web_search for online research and recent information.
- For weather requests that also ask to send a Telegram message,
  call get_weather first and then send_telegram_message.
- Use actual tool results. Never invent results or claim success
  when a tool fails.
- The default Telegram recipient is the configured username.
- Keep answers concise.
"""

agent = create_agent(
    model=llm,
    tools=[
        get_current_date_and_time,
        send_telegram_message,
        get_weather,
        web_search,
    ],
    system_prompt=SYSTEM_PROMPT,
    middleware=[summarizer],
    checkpointer=checkpointer,
)


# ============================================================
# 11. RUN AGENT WITH PERSISTENT MEMORY AND TOOL TRACE
# ============================================================

def run_agent(query: str, thread_id: str = "test103") -> dict:
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    previous_state = agent.get_state(config)
    previous_call_ids = set()

    if previous_state and previous_state.values:
        old_messages = previous_state.values.get("messages", [])

        for old_message in old_messages:
            for call in (
                getattr(old_message, "tool_calls", None) or []
            ):
                call_id = call.get("id")
                if call_id:
                    previous_call_ids.add(call_id)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        },
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
        for call in (
            getattr(message, "tool_calls", None) or []
        ):
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


# ============================================================
# 12. MAIN INTERACTIVE LOOP
# ============================================================

if __name__ == "__main__":
    THREAD_ID = "test105"

    try:
        print("Agent started. Type 'exit' to stop.")
        print(f"Conversation thread: {THREAD_ID}")

        while True:
            query = input("\nYou: ").strip()

            if query.lower() in {"exit", "quit"}:
                break

            if not query:
                continue

            try:
                result = run_agent(
                    query=query,
                    thread_id=THREAD_ID,
                )

                print("\nFinal response:")
                print(result["final_response"])

                print("\nTool calls in this run:")
                print(json.dumps(
                    result["tool_calls"],
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ))

            except Exception as exc:
                print(f"\nAgent error: {exc}")

    finally:
        connection.close()
