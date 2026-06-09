from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
import re
from typing import Dict, Optional


class ParseRequestError(RuntimeError):
    """Raised when request parsing cannot complete successfully."""


class WeatherLookupError(RuntimeError):
    """Raised when the OpenWeatherMap lookup cannot complete successfully."""


@dataclass(frozen=True)
class QueryIntent:
    location: str
    wants_weather: bool
    wants_time: bool
    needs_clarification: bool
    clarification_prompt: str


@dataclass(frozen=True)
class CurrentWeather:
    location: str
    description: str
    temperature_c: float
    timezone_offset_seconds: int


LOCATION_ALIASES: Dict[str, str] = {
    "台北": "Taipei",
    "臺北": "Taipei",
    "台中": "Taichung",
    "臺中": "Taichung",
    "台南": "Tainan",
    "臺南": "Tainan",
    "高雄": "Kaohsiung",
}

WEATHER_KEYWORDS = ("天氣", "weather", "氣溫", "溫度", "下雨", "降雨")
TIME_KEYWORDS = ("幾點", "時間", "幾點鐘", "time", "what time")


def _coerce_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _normalize_intent(result) -> QueryIntent:
    if not isinstance(result, dict):
        result = {}

    location = str(result.get("location", "")).strip()
    wants_weather = _coerce_bool(result.get("wants_weather"), False)
    wants_time = _coerce_bool(result.get("wants_time"), False)
    needs_clarification = _coerce_bool(
        result.get("needs_clarification"),
        (wants_weather or wants_time) and not location and wants_weather,
    )
    clarification_prompt = str(result.get("clarification_prompt", "")).strip()

    if needs_clarification and not clarification_prompt:
        clarification_prompt = "請提供要查詢的地點，例如台北、Tokyo、New York。"

    return QueryIntent(
        location=location,
        wants_weather=wants_weather,
        wants_time=wants_time,
        needs_clarification=needs_clarification,
        clarification_prompt=clarification_prompt,
    )


def _extract_json_object(text: str) -> Optional[dict]:
    text = text.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

    return None


def _has_any(text: str, keywords) -> bool:
    lowered = text.lower()
    return any(k in text or k in lowered for k in keywords)


def _extract_location_from_question(question: str) -> Optional[str]:
    # English pattern first: weather in Taipei / time in Tokyo
    match = re.search(r"(?:weather|time)\s+in\s+([A-Za-z\s]{2,32})", question, re.IGNORECASE)
    if match:
        candidate = re.sub(r"[^A-Za-z\s]", "", match.group(1)).strip()
        if candidate:
            return candidate.title()

    # Chinese/English token before weather/time keywords
    match = re.search(r"([A-Za-z\u4e00-\u9fff]{1,24})\s*(?:天氣|weather|時間|幾點)", question, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        candidate = re.sub(r"^(現在|今天|請問|幫我查|幫我看)", "", candidate).strip()
        if candidate and candidate.lower() not in {"the", "a", "an"}:
            return candidate

    for alias in sorted(LOCATION_ALIASES.keys(), key=len, reverse=True):
        if alias in question:
            return alias

    return None


def _heuristic_parse(question: str) -> Optional[dict]:
    wants_weather = _has_any(question, WEATHER_KEYWORDS)
    wants_time = _has_any(question, TIME_KEYWORDS)

    if not wants_weather and not wants_time:
        return {
            "location": "",
            "wants_weather": False,
            "wants_time": False,
            "needs_clarification": True,
            "clarification_prompt": "我目前支援天氣與時間查詢。請提供問題，例如：台北現在幾點？台北天氣如何？",
        }

    location = _extract_location_from_question(question) or ""

    if wants_weather and not location:
        return {
            "location": "",
            "wants_weather": wants_weather,
            "wants_time": wants_time,
            "needs_clarification": True,
            "clarification_prompt": "請提供要查詢天氣的地點，例如台北天氣如何？",
        }

    return {
        "location": location,
        "wants_weather": wants_weather,
        "wants_time": wants_time,
        "needs_clarification": False,
        "clarification_prompt": "",
    }


def _parse_with_openai(question: str, client) -> dict:
    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "Classify the user request and return JSON with keys: "
                    "location, wants_weather, wants_time, needs_clarification, clarification_prompt."
                ),
            },
            {"role": "user", "content": question},
        ],
    )

    output_text = getattr(response, "output_text", "")
    if not output_text:
        raise ParseRequestError("OpenAI returned an empty response.")

    extracted = _extract_json_object(output_text)
    if extracted is not None:
        return extracted

    return {
        "location": "",
        "wants_weather": False,
        "wants_time": False,
        "needs_clarification": True,
        "clarification_prompt": "請改用天氣或時間問題，並提供地點。",
    }


def parse_query_intent(question: str, client=None) -> QueryIntent:
    heuristic = _heuristic_parse(question)
    if heuristic is not None:
        return _normalize_intent(heuristic)

    if client is None:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise ParseRequestError("OpenAI client is unavailable.") from exc
        parsed = _parse_with_openai(question, OpenAI())
    elif callable(client):
        parsed = client(question)
    else:
        parsed = _parse_with_openai(question, client)

    return _normalize_intent(parsed)


def lookup_current_weather(
    location: str,
    *,
    api_key: Optional[str] = None,
    http_get=None,
    base_url: str = "https://api.openweathermap.org",
    units: str = "metric",
    lang: str = "zh_tw",
) -> CurrentWeather:
    location = location.strip()
    if not location:
        raise WeatherLookupError("Location is required.")

    if api_key is None:
        api_key = os.getenv("OPENWEATHERMAP_API_KEY", "").strip()
    if not api_key:
        raise WeatherLookupError("OPENWEATHERMAP_API_KEY is missing.")

    if http_get is None:
        try:
            import requests
        except Exception as exc:
            raise WeatherLookupError("requests is unavailable.") from exc
        http_get = requests.get

    query = LOCATION_ALIASES.get(location, location)

    geo_resp = http_get(
        f"{base_url.rstrip('/')}/geo/1.0/direct",
        params={"q": query, "limit": 1, "appid": api_key},
        timeout=10,
    )
    if getattr(geo_resp, "status_code", None) != 200:
        raise WeatherLookupError("Failed to resolve location.")

    try:
        geo_payload = geo_resp.json()
        first = geo_payload[0]
        lat = float(first["lat"])
        lon = float(first["lon"])
        resolved = str(first.get("name") or query)
    except Exception as exc:
        raise WeatherLookupError("city not found") from exc

    weather_resp = http_get(
        f"{base_url.rstrip('/')}/data/2.5/weather",
        params={
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": units,
            "lang": lang,
        },
        timeout=10,
    )
    if getattr(weather_resp, "status_code", None) != 200:
        try:
            msg = weather_resp.json().get("message", "OpenWeatherMap request failed.")
        except Exception:
            msg = "OpenWeatherMap request failed."
        raise WeatherLookupError(msg)

    try:
        payload = weather_resp.json()
        description = str(payload["weather"][0]["description"]).strip()
        temperature_c = float(payload["main"]["temp"])
        timezone_offset = int(payload.get("timezone", 0))
    except Exception as exc:
        raise WeatherLookupError("OpenWeatherMap response is missing required fields.") from exc

    return CurrentWeather(
        location=resolved,
        description=description,
        temperature_c=temperature_c,
        timezone_offset_seconds=timezone_offset,
    )


def lookup_current_time(location: str = "", *, timezone_offset_seconds: Optional[int] = None) -> str:
    if timezone_offset_seconds is None:
        now = datetime.now()
    else:
        now = datetime.now(timezone.utc) + timedelta(seconds=timezone_offset_seconds)
    return now.strftime("%Y-%m-%d %H:%M:%S")


def format_weather_reply(weather: CurrentWeather) -> str:
    return f"{weather.location} 目前天氣：{weather.description}\n溫度：{round(weather.temperature_c)}°C"


def format_time_reply(time_text: str, location: str = "") -> str:
    prefix = f"{location} 目前時間：" if location else "目前時間："
    return f"{prefix}{time_text}"


def main(
    input_func=None,
    output_func=print,
    parse_func=parse_query_intent,
    weather_lookup_func=lookup_current_weather,
    time_lookup_func=lookup_current_time,
) -> int:
    if input_func is None:
        input_func = input

    while True:
        try:
            user_input = input_func("你：")
        except EOFError:
            break

        if user_input.strip().lower() == "exit":
            break

        try:
            intent = parse_func(user_input)
        except ParseRequestError as exc:
            output_func(f"無法解析查詢：{exc}")
            continue

        if intent.needs_clarification:
            output_func(intent.clarification_prompt)
            continue

        weather = None
        if intent.wants_weather:
            try:
                weather = weather_lookup_func(intent.location)
                output_func(format_weather_reply(weather))
            except WeatherLookupError as exc:
                output_func(f"查詢天氣失敗：{exc}")
                continue

        if intent.wants_time:
            offset = weather.timezone_offset_seconds if weather is not None else None
            time_text = time_lookup_func(intent.location, timezone_offset_seconds=offset)
            output_func(format_time_reply(time_text, intent.location))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())