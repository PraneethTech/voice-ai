"""Weather via open-meteo — no API key, free. Online lookup tool."""
from __future__ import annotations

import httpx

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_CODES = {0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
          45: "fog", 51: "light drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
          71: "light snow", 80: "rain showers", 95: "thunderstorm"}


def get_weather(place: str) -> str:
    with httpx.Client(timeout=10) as client:
        geo = client.get(GEO_URL, params={"name": place, "count": 1, "language": "en", "format": "json"})
        geo.raise_for_status()
        results = geo.json().get("results")
        if not results:
            return f"I could not find a place called {place!r}."
        hit = results[0]
        lat, lon, name = hit["latitude"], hit["longitude"], hit["name"]
        res = client.get(FORECAST_URL, params={
            "latitude": lat, "longitude": lon, "current_weather": "true",
        })
        res.raise_for_status()
        cur = res.json()["current_weather"]
    desc = _CODES.get(cur.get("weathercode"), "changing conditions")
    return (f"Current weather in {name}: {cur.get('temperature')}°C, "
            f"wind {cur.get('windspeed')} km/h, {desc}.")
