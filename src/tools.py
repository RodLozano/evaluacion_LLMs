# src/tools.py
from __future__ import annotations

import os
import json
import random
import logging
from datetime import date, datetime
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field, ValidationError


# ---------------------------
# Logging setup
# ---------------------------

def _get_logger() -> logging.Logger:
    logger = logging.getLogger("tool_calls")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    os.makedirs("logs", exist_ok=True)

    fh = logging.FileHandler("logs/tool_calls.log", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


LOGGER = _get_logger()


# ---------------------------
# Tool schema (Pydantic)
# ---------------------------

class GetWeatherArgs(BaseModel):
    """
    Arguments expected by get_weather(fecha).
    fecha should be an ISO date string: YYYY-MM-DD.
    """
    fecha: str = Field(..., description="Fecha en formato ISO: YYYY-MM-DD")


# ---------------------------
# Tool implementation
# ---------------------------

def _parse_iso_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError("Fecha inválida. Usa el formato YYYY-MM-DD.") from e


def get_weather(fecha: str) -> Dict[str, Any]:
    """
    Simulated weather forecast for a given date.
    Returns a structured dict that is easy to render in the assistant.

    This function intentionally includes basic error handling scenarios:
    - Invalid date format
    - Date too far in the past/future (configurable)
    - Random transient failure (optional, for testing)
    """
    dt = _parse_iso_date(fecha)

    today = date.today()

    # Basic sanity constraints (you can tune these):
    # - allow forecasts from 7 days in the past up to 30 days in the future
    min_dt = today.replace(day=today.day)  # no-op, just explicit
    # For "past window", do it as a simple delta in days:
    past_days_allowed = int(os.getenv("WEATHER_PAST_DAYS", "7"))
    future_days_allowed = int(os.getenv("WEATHER_FUTURE_DAYS", "30"))

    delta_days = (dt - today).days
    if delta_days < -past_days_allowed:
        raise ValueError(f"No hay datos para {fecha}: demasiado en el pasado.")
    if delta_days > future_days_allowed:
        raise ValueError(f"No hay predicción para {fecha}: demasiado lejos en el futuro.")

    # Optional transient failures for testing your error handling:
    fail_rate = float(os.getenv("WEATHER_FAIL_RATE", "0.0"))  # set 0.1 to test failures
    if fail_rate > 0 and random.random() < fail_rate:
        raise RuntimeError("Servicio meteorológico no disponible temporalmente.")

    # Deterministic-ish pseudo forecast based on date
    seed = int(dt.strftime("%Y%m%d"))
    rng = random.Random(seed)

    temp_min = rng.randint(4, 18)
    temp_max = rng.randint(temp_min + 3, temp_min + 12)
    conditions = rng.choice(["soleado", "parcialmente nublado", "nublado", "lluvia", "tormenta"])
    precip_prob = rng.choice([0, 10, 20, 30, 40, 60, 80])

    return {
        "fecha": fecha,
        "condicion": conditions,
        "temp_min_c": temp_min,
        "temp_max_c": temp_max,
        "prob_precipitacion": precip_prob,
        "fuente": "simulada",
    }


# ---------------------------
# OpenAI tool schema builder
# ---------------------------

def get_tools_schema():
    """
    Tools schema for OpenAI Responses API.
    """
    return [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Devuelve una predicción del tiempo para una fecha dada (YYYY-MM-DD).",
            "parameters": {
                "type": "object",
                "properties": {
                    "fecha": {
                        "type": "string",
                        "description": "Fecha en formato ISO (YYYY-MM-DD).",
                    }
                },
                "required": ["fecha"],
                "additionalProperties": False,
            },
        }
    ]


# ---------------------------
# Dispatcher for tool calls
# ---------------------------

def run_tool(tool_name: str, arguments: Any) -> Dict[str, Any]:
    """
    Executes a tool by name given raw arguments coming from the model.
    Logs every attempt (success or failure) to logs/tool_calls.log.

    arguments may be:
    - dict
    - JSON string
    """
    # Parse arguments
    raw_args = arguments
    parsed: Optional[Dict[str, Any]] = None

    try:
        if isinstance(arguments, str):
            parsed = json.loads(arguments)
        elif isinstance(arguments, dict):
            parsed = arguments
        else:
            raise TypeError("Arguments must be a dict or a JSON string.")

        # Validate with Pydantic
        if tool_name == "get_weather":
            args = GetWeatherArgs(**parsed)
            result = get_weather(args.fecha)

            LOGGER.info(
                "tool=%s status=success args=%s result=%s",
                tool_name,
                json.dumps(parsed, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
            )
            return {"ok": True, "tool": tool_name, "result": result}

        raise ValueError(f"Unknown tool: {tool_name}")

    except (ValidationError, ValueError, TypeError) as e:
        err = str(e)
        LOGGER.error(
            "tool=%s status=error args=%s error=%s",
            tool_name,
            json.dumps(raw_args, ensure_ascii=False) if not isinstance(raw_args, str) else raw_args,
            err,
        )
        return {"ok": False, "tool": tool_name, "error": err}

    except Exception as e:
        # Catch-all for unexpected errors
        err = f"Unexpected error: {e}"
        LOGGER.error(
            "tool=%s status=error args=%s error=%s",
            tool_name,
            json.dumps(raw_args, ensure_ascii=False) if not isinstance(raw_args, str) else raw_args,
            err,
        )
        return {"ok": False, "tool": tool_name, "error": err}


# ---------------------------
# Helper to render tool result to text (optional)
# ---------------------------

def weather_result_to_text(payload: Dict[str, Any]) -> str:
    """
    Convert the structured tool output to a short natural-language summary.
    """
    if not payload.get("ok"):
        return f"No he podido obtener el tiempo: {payload.get('error','error desconocido')}"

    w = payload["result"]
    return (
        f"Tiempo para {w['fecha']}: {w['condicion']}, "
        f"mín {w['temp_min_c']}°C / máx {w['temp_max_c']}°C, "
        f"prob. precipitación {w['prob_precipitacion']}% (fuente: {w['fuente']})."
    )



'''
COMO USARLO EN EL NOTEBOOK

from src.tools import get_tools_schema, run_tool, weather_result_to_text

print(get_tools_schema())

out = run_tool("get_weather", {"fecha": "2026-01-06"})
print(out)
print(weather_result_to_text(out))

# Caso de error (para demostrar manejo de fallos + logging)
out2 = run_tool("get_weather", {"fecha": "06-01-2026"})
print(weather_result_to_text(out2))
'''