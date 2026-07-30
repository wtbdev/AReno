"""Reward function for the calendar-scheduling agentic RL example.

Scores each trajectory based on constraint satisfaction (was the confirmed
meeting valid?) and tool-call efficiency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import importlib.util as _iu
_game_spec = _iu.spec_from_file_location("_calendar_game", str(Path(__file__).resolve().parent / "game.py"))
game = _iu.module_from_spec(_game_spec)  # type: ignore[assignment]
_game_spec.loader.exec_module(game)  # type: ignore[union-attr]


def _parse_tool_calls(record: Any) -> list[dict[str, Any]]:
    """Normalise tool calls from a reward record into a uniform list."""
    raw = getattr(record, "tool_calls", None) or []
    parsed: list[dict[str, Any]] = []
    for call in raw:
        name = call.get("name") if isinstance(call, dict) else None
        arguments = call.get("arguments") if isinstance(call, dict) else None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        parsed.append({"name": name, "arguments": arguments or {}})
    return parsed


def reward_fn(record: Any) -> float:
    """Score one scheduling trajectory with partial credit for progress.

    Returns
    -------
    float
        +1.0   confirmed a fully valid slot
        +0.8   proposed a valid slot (but didn't confirm)
        +0.5   queried required participants
        +0.2   bonus — ≤ 3 availability queries used
        -0.5   confirmed a conflict slot
        -1.0   no useful tool calls at all
    """
    source = record.source_record
    tool_calls = _parse_tool_calls(record)
    participants = source.get("participants", [])
    duration_min = source.get("duration_min", 60)
    required = source.get("required", [])

    queries = [c for c in tool_calls if c["name"] == "query_availability"]
    confirm_calls = [c for c in tool_calls if c["name"] == "confirm"]
    propose_calls = [c for c in tool_calls if c["name"] == "propose_slot"]

    if not tool_calls:
        return -1.0

    # Check confirmed slot.
    if confirm_calls:
        confirmed_time = confirm_calls[-1]["arguments"].get("utc_time")
        if not isinstance(confirmed_time, int):
            return -0.5
        parts = [game.Participant(**p) for p in participants]
        result = game.validate_slot(confirmed_time, parts, duration_min, required=required)
        score = 1.0 if result["valid"] else -0.5
        if queries and len(queries) <= 3:
            score += 0.2
        return max(-1.0, min(1.0, score))

    # Check proposed slots — give partial credit for valid proposals.
    if propose_calls:
        for pc in reversed(propose_calls):
            utc_time = pc["arguments"].get("utc_time")
            if not isinstance(utc_time, int):
                continue
            parts = [game.Participant(**p) for p in participants]
            result = game.validate_slot(utc_time, parts, duration_min, required=required)
            if result["valid"]:
                score = 0.8
                if queries and len(queries) <= 3:
                    score += 0.2
                return min(1.0, score)
        return -0.5  # proposed but none were valid

    # Only queried — give partial credit for gathering information.
    queried_required = len(set(c["arguments"].get("name", "") for c in queries if c["arguments"].get("name") in required))
    if queried_required > 0:
        return 0.5 * min(1.0, queried_required / len(required))

    return -1.0