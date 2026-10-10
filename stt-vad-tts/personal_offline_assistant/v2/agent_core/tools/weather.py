"""Weather via open-meteo — no API key, free. Online lookup tool.

Returns current conditions plus a 3-day forecast as JSON.
"""
from __future__ import annotations

import json

import httpx

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes -> readable descriptions.
_CODES = {
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


def get_weather(location: str) -> str:
    """Get current weather and a 3-day forecast for a location."""
    try:
        with httpx.Client(timeout=10) as client:
            geo_response = client.get(GEO_URL, params={
                "name": location, "count": 1, "language": "en", "format": "json",
            })
            geo_response.raise_for_status()
            places = geo_response.json().get("results", [])
            if not places:
                return f"Location not found: {location}"

            place = places[0]
            latitude = place["latitude"]
            longitude = place["longitude"]
            place_name = f"{place['name']}, {place.get('country', '')}"

            weather_response = client.get(FORECAST_URL, params={
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
            })
            weather_response.raise_for_status()
            data = weather_response.json()

        current = data["current"]
        daily = data["daily"]
        current["condition"] = _CODES.get(current.get("weather_code"), "Unknown")

        forecast = []
        for i, date in enumerate(daily["time"]):
            forecast.append({
                "date": date,
                "condition": _CODES.get(daily["weather_code"][i], "Unknown"),
                "max_temp_c": daily["temperature_2m_max"][i],
                "min_temp_c": daily["temperature_2m_min"][i],
                "precipitation_probability_percent": daily["precipitation_probability_max"][i],
            })

        return json.dumps({
            "location": place_name,
            "current": current,
            "forecast": forecast,
        }, ensure_ascii=False)

    except httpx.HTTPError as exc:
        return f"Weather API error: {exc}"
    except (KeyError, IndexError, ValueError) as exc:
        return f"Could not process weather data: {exc}"
