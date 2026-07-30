"""Multi-turn agent function for calendar scheduling.

The agent drives a conversation loop: query a participant's availability,
propose a UTC time slot, and confirm the meeting.  Each model response is
recorded as an ``AgentTrajectoryTurn`` so the trainer can compute
policy-gradient updates over the full multi-step reasoning trace.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

import importlib.util as _iu
_game_spec = _iu.spec_from_file_location("_calendar_game", str(Path(__file__).resolve().parent / "game.py"))
game = _iu.module_from_spec(_game_spec)  # type: ignore[assignment]
_game_spec.loader.exec_module(game)  # type: ignore[union-attr]

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Schedule a meeting across time zones. Steps: query_availability for required people, convert to UTC, propose_slot to verify, confirm to finalize. Use UTC HHMM (e.g. 1430). Do not confirm without propose_slot."""

MAX_TURNS = 6

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_availability",
            "description": "Get a participant's available time blocks in their local time zone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The participant's name.",
                    }
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_slot",
            "description": "Check whether a proposed UTC time works for the listed participants.",
            "parameters": {
                "type": "object",
                "properties": {
                    "utc_time": {
                        "type": "integer",
                        "description": "Proposed meeting start time in UTC HHMM format (e.g. 1430).",
                    },
                    "participants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Participants to check.",
                    },
                },
                "required": ["utc_time", "participants"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm",
            "description": "Finalize the meeting at the specified UTC time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "utc_time": {
                        "type": "integer",
                        "description": "Confirmed meeting start time in UTC HHMM format.",
                    }
                },
                "required": ["utc_time"],
                "additionalProperties": False,
            },
        },
    },
]


def _lookup_participant(task: dict, name: str) -> dict | None:
    for p in task.get("participants", []):
        if p["name"] == name:
            return p
    return None


def _execute_tool(task: dict, tool_name: str, args: dict) -> str:
    """Execute a tool locally and return a JSON string result."""
    if tool_name == "query_availability":
        name = args.get("name", "")
        p = _lookup_participant(task, name)
        if p is None:
            return json.dumps({"error": f"'{name}' not found"})
        offset = p["utc_offset_hours"]
        sign = "+" if offset >= 0 else ""
        blocks = [f"{s:04d}-{e:04d}" for s, e in p["available_blocks"]]
        return json.dumps({"name": name, "tz": f"UTC{sign}{offset}", "free": blocks})

    if tool_name == "propose_slot":
        utc_time = args.get("utc_time")
        names = args.get("participants", [])
        parts = [game.Participant(**p) for p in task["participants"] if p["name"] in names]
        result = game.validate_slot(utc_time, parts, task["duration_min"], required=task.get("required"))
        return json.dumps(result)

    if tool_name == "confirm":
        utc_time = args.get("utc_time")
        return json.dumps({"success": True, "utc_time": utc_time, "message": "Meeting confirmed."})

    return json.dumps({"error": f"unknown tool: {tool_name}"})


async def run_agent(ctx, batch):
    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The calendar agentic example requires `openai` and `httpx`. "
            "Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info("calendar agent start requests=%d max_running=%d", len(items), ctx.max_running_prompts)
    max_conn = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_conn, max_keepalive_connections=max_conn),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)

    async def run_one(item):
        task = item.record
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        turn_records = []

        for _ in range(MAX_TURNS):
            response = await client.chat.completions.create(
                model="policy",
                messages=messages,
                tools=TOOLS,
                tool_choice="required",
                stream=False,
            )
            choice = response.choices[0] if response.choices else None
            if choice is None:
                break

            # Record this turn regardless of whether there is a tool call.
            turn_records.append(
                AgentTrajectoryTurn(
                    item=item,
                    messages=[dict(m) for m in messages],
                    response=response,
                    tools=TOOLS,
                    tool_choice="required",
                )
            )

            tool_calls = getattr(choice.message, "tool_calls", None) or []
            if not tool_calls:
                assistant_content = choice.message.content or ""
                logger.warning("no tool call in response: finish_reason=%s content_preview=%s",
                               getattr(choice, "finish_reason", "?"), assistant_content[:200])
                messages.append({"role": "assistant", "content": assistant_content})
                break

            # Process the first tool call.
            tc = tool_calls[0]
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
            result_text = _execute_tool(task, tool_name, tool_args)

            # Append assistant + tool messages.
            messages.append(
                {
                    "role": "assistant",
                    "content": choice.message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tc.function.arguments},
                        }
                    ],
                }
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

            if tool_name == "confirm":
                break

        return turn_records

    try:
        all_turns = list(await asyncio.gather(*(run_one(item) for item in items)))
        flat_turns = [t for turns in all_turns for t in turns]
        return AgentTrajectory(turns=flat_turns)
    finally:
        await client.close()