"""Web UI for the calendar-scheduling agentic example.

Run from the repository root::

    python examples/agentic/calendar/web_ui.py --base-url http://127.0.0.1:8000/v1 --api-key token
"""

from __future__ import annotations

import argparse
import importlib.util as _iu
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Isolated import — avoids polluting sys.modules["game"]
# ---------------------------------------------------------------------------
_game_spec = _iu.spec_from_file_location("_calendar_game_web", str(Path(__file__).resolve().parent / "game.py"))
game = _iu.module_from_spec(_game_spec)
_game_spec.loader.exec_module(game)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8768

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_availability",
            "description": "Get a participant's available time blocks (local time).",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_slot",
            "description": "Check whether a UTC time works for the listed participants.",
            "parameters": {
                "type": "object",
                "properties": {
                    "utc_time": {"type": "integer"},
                    "participants": {"type": "array", "items": {"type": "string"}},
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
                "properties": {"utc_time": {"type": "integer"}},
                "required": ["utc_time"],
                "additionalProperties": False,
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are a calendar scheduling assistant. Work step-by-step:
1. Use query_availability to check each required participant.
2. Convert their local times to UTC.
3. Find a UTC window where everyone is free for the required duration.
4. Use propose_slot to verify.
5. Use confirm to finalize.
All times in propose_slot and confirm must be UTC HHMM format (e.g. 1430)."""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class CalendarServer(ThreadingHTTPServer):
    def __init__(self, server_address, request_handler, *, args):
        super().__init__(server_address, request_handler)
        self.args = args
        self.task: dict[str, Any] | None = None
        self.messages: list[dict[str, Any]] = []
        self.events: list[str] = ["Ready. Press 'New Task' or 'Step' to begin."]
        self.openai_client = None


class CalendarHandler(BaseHTTPRequestHandler):
    server: CalendarServer

    def do_GET(self) -> None:
        route = _route_path(self.path)
        if route == "index":
            self._send_html(INDEX_HTML)
        elif route == "state":
            self._send_json(_payload(self.server))
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        route = _route_path(self.path)
        body = self._read_json() or {}
        if route == "new":
            _new_task(self.server)
            self._send_json(_payload(self.server))
        elif route == "step":
            agent_first = body.get("agent_first", True)
            _agent_step(self.server, agent_first=agent_first)
            self._send_json(_payload(self.server))
        elif route == "reset":
            _reset(self.server)
            self._send_json(_payload(self.server))
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("calendar-web: " + fmt % args + "\n")

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: Any) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_path(raw_path: str) -> str:
    path = urlparse(raw_path).path.rstrip("/") or "/"
    if path.endswith("/api/state"):
        return "state"
    if path.endswith("/api/new"):
        return "new"
    if path.endswith("/api/step"):
        return "step"
    if path.endswith("/api/reset"):
        return "reset"
    if path == "/" or "." not in path.rsplit("/", 1)[-1]:
        return "index"
    return "missing"


# ---------------------------------------------------------------------------
# Game logic
# ---------------------------------------------------------------------------

_SAMPLE_TASKS = [
    {
        "participants": [
            {"name": "Alice", "utc_offset_hours": +8, "available_blocks": [(900, 1700)]},
            {"name": "Bob", "utc_offset_hours": -5, "available_blocks": [(900, 1700)]},
            {"name": "Carol", "utc_offset_hours": 0, "available_blocks": [(800, 1200), (1300, 1800)]},
        ],
        "duration_min": 60,
        "required": ["Alice", "Bob"],
    },
    {
        "participants": [
            {"name": "Dave", "utc_offset_hours": +1, "available_blocks": [(1000, 1600)]},
            {"name": "Eve", "utc_offset_hours": -8, "available_blocks": [(700, 1200)]},
        ],
        "duration_min": 30,
        "required": ["Dave", "Eve"],
    },
]


def _new_task(server: CalendarServer) -> None:
    import random
    task = random.choice(_SAMPLE_TASKS)
    server.task = task
    parts = task["participants"]
    req = task["required"]
    dur = task["duration_min"]
    names = ", ".join(p["name"] for p in parts)
    server.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Schedule a {dur}-minute meeting. Required: {', '.join(req)}.\n\n"
            + "\n".join(
                f"- {p['name']} (UTC{'+' if p['utc_offset_hours']>=0 else ''}{p['utc_offset_hours']}): "
                + ", ".join(f"{s:04d}–{e:04d}" for s, e in p["available_blocks"])
                for p in parts
            ),
        },
    ]
    server.events = [f"New task: {len(parts)} participants, {dur}min meeting, {len(req)} required."]


def _reset(server: CalendarServer) -> None:
    server.task = None
    server.messages = []
    server.events = ["Ready."]


def _agent_step(server: CalendarServer, *, agent_first: bool = True) -> None:
    if server.task is None:
        server.events.insert(0, "No active task. Press 'New Task' first.")
        server.events = server.events[:10]
        return

    if not server.args.base_url:
        server.events.insert(0, "No --base-url configured. Use Best Mode instead.")
        server.events = server.events[:10]
        return

    if server.openai_client is None:
        try:
            from openai import OpenAI
        except ImportError:
            server.events.insert(0, "openai not installed. Use 'pip install openai'.")
            server.events = server.events[:10]
            return
        server.openai_client = OpenAI(base_url=server.args.base_url, api_key=server.args.api_key, max_retries=0)

    try:
        response = server.openai_client.chat.completions.create(
            model=server.args.model,
            messages=server.messages,
            tools=TOOLS,
            tool_choice="auto",
        )
    except Exception as exc:
        server.events.insert(0, f"LLM error: {exc}")
        server.events = server.events[:10]
        return

    choice = response.choices[0] if response.choices else None
    if choice is None:
        server.events.insert(0, "LLM returned no choices.")
        server.events = server.events[:10]
        return

    tool_calls = getattr(choice.message, "tool_calls", None) or []
    if not tool_calls:
        text = choice.message.content or "(no response)"
        server.events.insert(0, f"LLM: {text}")
        server.events = server.events[:10]
        return

    tc = tool_calls[0]
    tool_name = tc.function.name
    try:
        tool_args = json.loads(tc.function.arguments)
    except json.JSONDecodeError:
        tool_args = {}

    # Execute tool locally.
    if tool_name == "query_availability":
        name = tool_args.get("name", "?")
        p = next((p for p in server.task["participants"] if p["name"] == name), None)
        if p:
            offset = p["utc_offset_hours"]
            sign = "+" if offset >= 0 else ""
            blocks = ", ".join(f"{s:04d}–{e:04d} local (UTC{sign}{offset})" for s, e in p["available_blocks"])
            result_msg = f"{name}: {blocks}"
        else:
            result_msg = f"Participant '{name}' not found."
    elif tool_name == "propose_slot":
        utc_time = tool_args.get("utc_time")
        names = tool_args.get("participants", [])
        parts = [game.Participant(**p) for p in server.task["participants"] if p["name"] in names]
        result = game.validate_slot(utc_time, parts, server.task["duration_min"], required=server.task.get("required"))
        if result["valid"]:
            result_msg = f"✓ Valid at {utc_time:04d} UTC"
        else:
            conflicts = "; ".join(c["message"] for c in result["conflicts"])
            result_msg = f"✗ Conflict at {utc_time:04d} UTC: {conflicts}"
    elif tool_name == "confirm":
        utc_time = tool_args.get("utc_time")
        # Final validation.
        parts = [game.Participant(**p) for p in server.task["participants"]]
        result = game.validate_slot(utc_time, parts, server.task["duration_min"], required=server.task.get("required"))
        if result["valid"]:
            result_msg = f"✓ Meeting confirmed at {utc_time:04d} UTC!"
        else:
            result_msg = f"✗ Confirmed {utc_time:04d} UTC but it has conflicts!"
    else:
        result_msg = f"Unknown tool: {tool_name}"

    server.events.insert(0, f"[{tool_name}] {result_msg}")
    server.events = server.events[:10]

    # Append assistant + tool messages to conversation.
    server.messages.append({
        "role": "assistant",
        "content": choice.message.content or "",
        "tool_calls": [{
            "id": tc.id,
            "type": "function",
            "function": {"name": tool_name, "arguments": tc.function.arguments},
        }],
    })
    server.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_msg})


def _payload(server: CalendarServer) -> dict[str, Any]:
    return {
        "task": server.task,
        "events": server.events,
        "has_task": server.task is not None,
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Calendar Scheduler</title>
<style>
:root{font-family:Inter,ui-rounded,system-ui,sans-serif;color:#1a2a3a;background:#e8f0fe}
body{margin:0;min-height:100vh;background:linear-gradient(135deg,#dbeafe,#bfdbfe 40%,#a5d8ff 80%);display:grid;place-items:center}
.app{width:min(860px,94vw);display:grid;grid-template-columns:minmax(300px,400px) 1fr;gap:20px;align-items:start}
.panel{background:#fff;border:4px solid #1e3a5f;border-radius:22px;box-shadow:8px 8px 0 #1e3a5f;padding:18px}
h1{font-size:32px;line-height:1;margin:0 0 6px;color:#2563eb;text-shadow:2px 2px 0 #bfdbfe}
.subtitle{font-weight:700;color:#475569;margin-bottom:14px;font-size:14px}
.taskCard{background:#f0f9ff;border:2px solid #93c5fd;border-radius:12px;padding:14px;margin-top:12px;display:none}
.taskCard.on{display:block}
.participant{margin:6px 0;padding:8px 10px;background:#fff;border:2px solid #cbd5e1;border-radius:8px;font-size:13px}
.participant .name{font-weight:700;color:#1e40af}
.participant .tz{color:#64748b;font-size:11px;margin-left:6px}
.participant .blocks{color:#334155;font-size:12px;margin-top:2px}
.participant.req{border-color:#2563eb;background:#eff6ff}
.actions{margin-top:14px;display:flex;gap:10px;flex-wrap:wrap}
button{border:3px solid #1e3a5f;border-radius:14px;background:#ffd166;box-shadow:4px 4px 0 #1e3a5f;color:#1e3a5f;font-weight:800;padding:10px 14px;cursor:pointer;font-size:14px}
button:hover{transform:translateY(-1px)}button:disabled{filter:grayscale(.75);opacity:.55;cursor:not-allowed}
button.primary{background:#3b82f6;color:#fff;border-color:#1e40af}
.events{display:grid;gap:8px;margin-top:14px}
.event{background:#fff;border:3px solid #1e3a5f;border-radius:12px;padding:10px;font-weight:700;font-size:13px}
.event:first-child{background:#eff6ff;border-color:#3b82f6}
.empty{text-align:center;color:#94a3b8;font-weight:700;padding:20px}
.badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:11px;font-weight:800;margin-right:4px}
.badge-query{background:#dbeafe;color:#1e40af}
.badge-propose{background:#fef3c7;color:#92400e}
.badge-confirm{background:#dcfce7;color:#166534}
@media(max-width:700px){.app{grid-template-columns:1fr}h1{font-size:28px}}
</style>
</head>
<body>
<main class="app">
  <section class="panel">
    <h1>Calendar Scheduler</h1>
    <div class="subtitle">Multi-turn LLM meeting scheduling across time zones.</div>
    <div id="taskCard" class="taskCard">
      <div id="taskInfo"></div>
    </div>
    <div id="emptyTask" class="empty">No active task. Press <b>New Task</b> to generate one, then press <b>Step</b> to let the LLM take one action.</div>
    <div class="actions">
      <button class="primary" id="btnNew">New Task</button>
      <button id="btnStep">Step</button>
      <button id="btnReset">Reset</button>
    </div>
  </section>
  <aside class="panel">
    <h1 style="font-size:24px;color:#1e40af">Activity Log</h1>
    <div id="events" class="events"></div>
    <div style="margin-top:14px;font-size:12px;color:#64748b;font-weight:600">
      Tools: <span class="badge badge-query">query</span> <span class="badge badge-propose">propose</span> <span class="badge badge-confirm">confirm</span>
    </div>
  </aside>
</main>
<script>
const api = (path) => new URL(path, window.location.href).toString();
let state = null;
async function request(path, body){
  const opts = body ? {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)} : {};
  const res = await fetch(api(path), opts);
  state = await res.json();
  render();
}
function render(){
  const card = document.getElementById("taskCard");
  const empty = document.getElementById("emptyTask");
  if(state.has_task && state.task){
    card.classList.add("on"); empty.style.display="none";
    const t = state.task;
    const parts = t.participants.map(p => `<div class="participant ${t.required.includes(p.name)?'req':''}"><span class="name">${escapeHtml(p.name)}</span><span class="tz">UTC${p.utc_offset_hours>=0?'+':''}${p.utc_offset_hours}</span><div class="blocks">${(p.available_blocks||[]).map(b=>`${String(b[0])}–${String(b[1])}`).join(", ")}</div></div>`).join("");
    document.getElementById("taskInfo").innerHTML = `<div style="font-weight:800;margin-bottom:8px">${t.duration_min}min · Required: ${t.required.join(", ")}</div>${parts}`;
  } else {
    card.classList.remove("on"); empty.style.display="";
  }
  document.getElementById("events").innerHTML = (state.events||[]).map(e => `<div class="event">${formatEvent(e)}</div>`).join("");
}
function formatEvent(text){
  if(text.startsWith("[query_availability]")) return `<span class="badge badge-query">query</span>${escapeHtml(text.slice(20))}`;
  if(text.startsWith("[propose_slot]")) return `<span class="badge badge-propose">propose</span>${escapeHtml(text.slice(14))}`;
  if(text.startsWith("[confirm]")) return `<span class="badge badge-confirm">confirm</span>${escapeHtml(text.slice(10))}`;
  return escapeHtml(text);
}
function escapeHtml(t){return t.replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[c]));}
document.getElementById("btnNew").onclick = () => request("api/new", {});
document.getElementById("btnStep").onclick = () => request("api/step", {agent_first: true});
document.getElementById("btnReset").onclick = () => request("api/reset", {});
request("api/state");
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Calendar Scheduling web UI.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--model", default="policy")
    args = parser.parse_args()

    server = CalendarServer((args.host, args.port), CalendarHandler, args=args)
    url = f"http://{args.host}:{args.port}"
    print(f"Calendar scheduling web UI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()