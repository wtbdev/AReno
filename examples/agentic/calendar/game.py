"""Calendar scheduling logic.

Pure functions with no AReno imports.  All time arithmetic uses UTC
minutes-since-midnight internally; time zones are fixed offsets (no IANA,
no DST) so that every test fixture is deterministic.
"""

# NOTE: do NOT add ``from __future__ import annotations`` here.
# The test harness loads this module via importlib.util.spec_from_file_location
# which does not register a __module__ name, causing dataclass string-annotation
# resolution to fail on Python 3.14+.

import dataclasses
from typing import Any

MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 24 * MINUTES_PER_HOUR  # 1440


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Participant:
    """One calendar participant with a fixed UTC offset and availability."""

    name: str
    utc_offset_hours: float  # e.g. +8, -5, +5.5
    available_blocks: list  # list of (start_hhmm, end_hhmm) local time


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _hhmm_to_minutes(hhmm: int) -> int:
    """Convert an ``HHMM`` integer (e.g. 930, 1430) to minutes since midnight."""
    return (hhmm // 100) * MINUTES_PER_HOUR + (hhmm % 100)


def _minutes_to_hhmm(minutes: int) -> int:
    """Inverse of ``_hhmm_to_minutes``, wrapping across midnight."""
    m = minutes % MINUTES_PER_DAY
    return (m // MINUTES_PER_HOUR) * 100 + (m % MINUTES_PER_HOUR)


def local_to_utc(local_hhmm: int, utc_offset_hours: float) -> int:
    """Convert a local ``HHMM`` time to UTC minutes since midnight.

    Examples
    --------
    >>> local_to_utc(900, +8)   # 09:00 UTC+8 → 01:00 UTC → 60
    >>> local_to_utc(1400, -5)  # 14:00 UTC-5 → 19:00 UTC → 1140
    """
    local_min = _hhmm_to_minutes(local_hhmm)
    offset_min = int(round(utc_offset_hours * MINUTES_PER_HOUR))
    return local_min - offset_min


def utc_to_local(utc_minutes: int, utc_offset_hours: float) -> int:
    """Convert UTC minutes to a local ``HHMM`` time."""
    offset_min = int(round(utc_offset_hours * MINUTES_PER_HOUR))
    return _minutes_to_hhmm(utc_minutes + offset_min + MINUTES_PER_DAY)


# ---------------------------------------------------------------------------
# Slot finding
# ---------------------------------------------------------------------------


def find_overlap(
    participants: list[Participant],
    duration_min: int,
    *,
    required: list[str] | None = None,
) -> list[int]:
    """Return every UTC start-minute where *all* required participants are free.

    Each candidate must fit *duration_min* minutes entirely within every
    required participant's available blocks (converted to UTC).  Optional
    participants are ignored for validity.

    Returns a (possibly empty) list of UTC minute offsets, **sorted**
    ascending.
    """
    required_names = set(required or [p.name for p in participants])
    required_parts = [p for p in participants if p.name in required_names]
    if not required_parts:
        return []

    # Build the UTC availability timeline for each required participant
    # as a set of covered minutes.
    timelines: list[set[int]] = []
    for p in required_parts:
        free: set[int] = set()
        for start_hhmm, end_hhmm in p.available_blocks:
            utc_start = local_to_utc(start_hhmm, p.utc_offset_hours)
            utc_end = local_to_utc(end_hhmm, p.utc_offset_hours)
            if utc_end < utc_start:
                utc_end += MINUTES_PER_DAY
            for minute in range(utc_start, utc_end):
                free.add(minute % MINUTES_PER_DAY)
        timelines.append(free)

    intersection = timelines[0].copy()
    for tl in timelines[1:]:
        intersection &= tl

    candidates: list[int] = []
    for start_min in sorted(intersection):
        # Check that the full *duration_min* window fits.
        window = {m % MINUTES_PER_DAY for m in range(start_min, start_min + duration_min)}
        if window.issubset(intersection):
            candidates.append(start_min)
    return candidates


# ---------------------------------------------------------------------------
# Slot validation
# ---------------------------------------------------------------------------


def validate_slot(
    utc_start_hhmm: int,
    participants: list[Participant],
    duration_min: int,
    *,
    required: list[str] | None = None,
) -> dict[str, Any]:
    """Check whether a proposed UTC ``HHMM`` slot is valid.

    Returns
    -------
    dict
        ``{"valid": bool, "conflicts": list[dict], "reason": str}``
        Each conflict dict has ``name``, ``unavailable_block``, ``message``.
    """
    utc_start = _hhmm_to_minutes(utc_start_hhmm)
    utc_end = utc_start + duration_min

    conflicts: list[dict[str, Any]] = []
    for p in participants:
        is_required = p.name in (required or [])
        block_free = False
        for start_hhmm, end_hhmm in p.available_blocks:
            utc_block_start = local_to_utc(start_hhmm, p.utc_offset_hours)
            utc_block_end = local_to_utc(end_hhmm, p.utc_offset_hours)
            if utc_block_end < utc_block_start:
                utc_block_end += MINUTES_PER_DAY
            if utc_start >= utc_block_start and utc_end <= utc_block_end:
                block_free = True
                break
        if not block_free:
            conflicts.append(
                {
                    "name": p.name,
                    "required": is_required,
                    "unavailable_block": (utc_start_hhmm, _minutes_to_hhmm(utc_end)),
                    "message": (
                        f"{p.name} ({'required' if is_required else 'optional'}) "
                        f"is not available at {utc_start_hhmm:04d}–{_minutes_to_hhmm(utc_end):04d} UTC"
                    ),
                }
            )

    if any(c["required"] for c in conflicts):
        return {"valid": False, "conflicts": conflicts, "reason": "required participant(s) unavailable"}
    return {"valid": True, "conflicts": conflicts, "reason": "ok"}


# ---------------------------------------------------------------------------
# Task description
# ---------------------------------------------------------------------------


def format_task(task: dict[str, Any]) -> str:
    """Build a compact scheduling prompt from a task dict."""
    parts = task.get("participants", [])
    duration = task.get("duration_min", 60)
    required = task.get("required", [p["name"] for p in parts])

    lines = [f"Schedule {duration}min. Required: {', '.join(required)}."]
    for p in parts:
        offset = p["utc_offset_hours"]
        sign = "+" if offset >= 0 else ""
        blocks = ", ".join(f"{s:04d}-{e:04d}" for s, e in p["available_blocks"])
        lines.append(f"{p['name']} (UTC{sign}{offset}): {blocks}")
    return "\n".join(lines)