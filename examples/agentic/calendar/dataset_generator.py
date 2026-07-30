"""Generate synthetic calendar-scheduling tasks for agentic RL training.

Usage::

    python dataset_generator.py --count 2048 --seed 2026 --output /tmp/calendar-tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, TextIO

# Fixed UTC offsets (no IANA names, no DST — deterministic).
_UTC_OFFSETS = [-12.0, -8.0, -5.0, 0.0, +1.0, +3.0, +5.5, +8.0, +10.0]

_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
]

_DURATIONS = [30, 60, 90]


def _random_block(rng: random.Random, *, start: int = 700, end: int = 2000) -> tuple[int, int]:
    """Return a random HHMM availability block.

    Block length is uniformly 60–300 minutes; wrap-around is NOT generated
    so blocks are simple to reason about.
    """
    a = rng.randint(start // 100, (end // 100) - 1) * 100
    length = rng.choice([60, 90, 120, 180, 240, 300])
    b = a + (length // 60) * 100
    # Keep within [start, end] and avoid crossing hour boundaries oddly.
    if b > end:
        b = end - (end % 100)
        if b <= a:
            b = a + 100
    return (a, b)


def _generate_participant(rng: random.Random, name: str) -> dict[str, Any]:
    offset = rng.choice(_UTC_OFFSETS)
    num_blocks = rng.randint(1, 3)
    blocks = [_random_block(rng) for _ in range(num_blocks)]
    blocks.sort()
    return {"name": name, "utc_offset_hours": offset, "available_blocks": blocks}


def generate_records(
    count: int, *, seed: int = 2026
) -> list[dict[str, Any]]:
    """Return *count* calendar-scheduling task records."""
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for idx in range(count):
        rng2 = random.Random(rng.randint(0, 2**31 - 1))
        num = rng2.randint(2, 5)
        chosen = rng2.sample(_NAMES, num)
        participants = [_generate_participant(rng2, name) for name in chosen]
        duration = rng2.choice(_DURATIONS)

        # Determine required set: at least 2, at most all.
        min_req = min(2, len(chosen))
        num_required = rng2.randint(min_req, len(chosen))
        required = rng2.sample(chosen, num_required)

        # Assign scenario weight
        roll = rng.random()
        if roll < 0.10:
            scenario = "unsolvable"
        elif roll < 0.30:
            scenario = "tight"
        elif roll < 0.40:
            scenario = "multi_query"
        else:
            scenario = "solvable"

        # Modify participants to match the scenario.
        if scenario == "unsolvable":
            # Give every required participant only morning blocks in their
            # own time zone so that in UTC they never overlap.
            for p in participants:
                if p["name"] in required:
                    p["available_blocks"] = [(_local_morning_block(p["utc_offset_hours"]))]
        elif scenario == "tight":
            # Shrink blocks so only a narrow window remains.
            for p in participants:
                p["available_blocks"] = _tight_blocks(p["available_blocks"], rng2)
        # "solvable" and "multi_query" keep random blocks as-is.

        records.append(
            {
                "id": f"cal-{idx:05d}",
                "participants": participants,
                "duration_min": duration,
                "required": required,
                "scenario": scenario,
            }
        )
    return records


def _local_morning_block(offset: float) -> tuple[int, int]:
    """Return a morning block local to *offset* that maps to non-overlapping UTC."""
    # Pick a local 3-hour window that, after offset, will be isolated.
    base = 700  # 07:00 local
    return (base, base + 300)  # 07:00–12:00


def _tight_blocks(
    blocks: list[tuple[int, int]], rng: random.Random
) -> list[tuple[int, int]]:
    """Shrink blocks so only a narrow window remains."""
    if not blocks:
        return blocks
    chosen = rng.choice(blocks)
    start, end = chosen
    mid = (start + end) // 2
    # Keep only a 30–60 minute window around the middle.
    window = rng.choice([30, 60])
    half = (window // 2) // 100 * 100
    return [(mid - half, mid + half)]


def write_jsonl(records: list[dict[str, Any]], output: TextIO) -> None:
    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate calendar-scheduling tasks.")
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=128, help="Number of tasks to generate.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")

    records = generate_records(args.count, seed=args.seed)
    if args.output == "-":
        write_jsonl(records, sys.stdout)
    else:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            write_jsonl(records, handle)


if __name__ == "__main__":
    main()