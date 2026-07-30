"""Dataset loader for the calendar-scheduling agentic RL example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import importlib.util as _iu
_game_spec = _iu.spec_from_file_location("_calendar_game", str(Path(__file__).resolve().parent / "game.py"))
game = _iu.module_from_spec(_game_spec)  # type: ignore[assignment]
_game_spec.loader.exec_module(game)  # type: ignore[union-attr]


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "calendar-tasks.jsonl"
    if not path.exists():
        import dataset_generator  # noqa: F811

        return dataset_generator.generate_records(128, seed=2026)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def load_training_dataset(
    dataset_path: str, *, default_loader=None, **_: object
) -> list[dict]:
    """Return a list of task dicts, each with a ``"prompt"`` key."""
    del default_loader
    raw_records = _load_records(dataset_path)
    result: list[dict] = []
    for raw in raw_records:
        result.append(
            {
                "id": raw.get("id", f"task-{len(result):05d}"),
                "prompt": game.format_task(raw),
                "participants": raw.get("participants", []),
                "duration_min": raw.get("duration_min", 60),
                "required": raw.get("required", []),
                "scenario": raw.get("scenario", "unknown"),
            }
        )
    return result