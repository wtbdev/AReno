"""CPU tests for the calendar-scheduling agentic RL example."""

from __future__ import annotations

import importlib.util
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_CALENDAR_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "calendar"


def _load_module(name: str):
    path = _CALENDAR_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"calendar_{name}_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CalendarGameTest(unittest.TestCase):
    def setUp(self):
        self.game = _load_module("game")

    def test_timezone_conversion(self):
        """local_to_utc and utc_to_local round-trip."""
        g = self.game
        self.assertEqual(g.local_to_utc(900, 8), g._hhmm_to_minutes(100))
        self.assertEqual(g.local_to_utc(1400, -5), g._hhmm_to_minutes(1900))
        # Round trip
        utc_m = g.local_to_utc(1430, 5.5)
        local = g.utc_to_local(utc_m, 5.5)
        self.assertEqual(local, 1430)

    def test_overlap_found_for_solvable(self):
        g = self.game
        # Alice UTC+8:  9:00-12:00 local → 01:00-04:00 UTC
        # Bob   UTC-5: 10:00-16:00 local → 15:00-21:00 UTC  ← won't overlap with Alice
        # We need blocks that DO overlap in UTC:
        # Alice UTC+8: 14:00-22:00 local → 06:00-14:00 UTC
        # Bob   UTC-5:  2:00-10:00 local → 07:00-15:00 UTC
        # Carol UTC+0:  7:00-12:00 local → 07:00-12:00 UTC
        # Overlap: 07:00-12:00 UTC
        parts = [
            g.Participant("Alice", +8, [(1400, 2200)]),
            g.Participant("Bob", -5, [(200, 1000)]),
            g.Participant("Carol", 0, [(700, 1200)]),
        ]
        candidates = g.find_overlap(parts, 60, required=["Alice", "Bob", "Carol"])
        self.assertGreater(len(candidates), 0)
        for start in candidates:
            result = g.validate_slot(
                g._minutes_to_hhmm(start), parts, 60, required=["Alice", "Bob", "Carol"]
            )
            self.assertTrue(result["valid"], f"Expected valid at {g._minutes_to_hhmm(start)}")

    def test_no_overlap_for_unsolvable(self):
        g = self.game
        # Alice morning UTC+14, Bob evening UTC-12 — zero UTC overlap.
        parts = [
            g.Participant("Alice", +14, [(800, 1000)]),
            g.Participant("Bob", -12, [(1800, 2000)]),
        ]
        candidates = g.find_overlap(parts, 60, required=["Alice", "Bob"])
        self.assertEqual(len(candidates), 0)

    def test_validate_slot_detects_conflict(self):
        g = self.game
        parts = [
            g.Participant("Alice", +8, [(900, 1200)]),
            g.Participant("Bob", 0, [(300, 500)]),  # 03:00–05:00 UTC
        ]
        # 09:00 UTC+8 = 01:00 UTC.  09:00-12:00 UTC+8 = 01:00-04:00 UTC.  Bob is free 03:00-05:00 UTC.
        # A 60-min meeting at 03:00 UTC should work.
        result = g.validate_slot(300, parts, 60, required=["Alice", "Bob"])
        self.assertTrue(result["valid"])
        # A 60-min meeting at 05:00 UTC should fail (Alice's block ends at 04:00 UTC).
        result = g.validate_slot(500, parts, 60, required=["Alice", "Bob"])
        self.assertFalse(result["valid"])

    def test_optional_participant_conflict_does_not_invalidate(self):
        g = self.game
        parts = [
            g.Participant("Alice", +8, [(900, 1200)]),
            g.Participant("Bob", 0, [(300, 500)]),
            g.Participant("Carol", +1, [(1700, 1900)]),  # far away, optional
        ]
        # Only Alice required.  Bob is optional but IS free; Carol is optional and busy.
        # 03:00 UTC = Alice 11:00 local (in 9:00-12:00 ✓), Bob 03:00 local (in 03:00-05:00 ✓)
        result = g.validate_slot(300, parts, 60, required=["Alice"])
        self.assertTrue(result["valid"])
        # Carol appears in conflicts but doesn't invalidate because she's optional.
        conflict_names = [c["name"] for c in result["conflicts"]]
        self.assertIn("Carol", conflict_names)
        # Bob is NOT in conflicts because he IS free at 03:00 UTC.
        self.assertNotIn("Bob", conflict_names)

    def test_format_task_produces_readable_prompt(self):
        g = self.game
        task = {
            "participants": [
                {"name": "Alice", "utc_offset_hours": +8, "available_blocks": [(900, 1700)]},
                {"name": "Bob", "utc_offset_hours": -5, "available_blocks": [(1000, 1500)]},
            ],
            "duration_min": 60,
            "required": ["Alice"],
        }
        prompt = g.format_task(task)
        self.assertIn("Schedule 60min", prompt)
        self.assertIn("Alice", prompt)
        self.assertIn("Bob", prompt)
        self.assertIn("Required: Alice", prompt)


class CalendarGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.generator = _load_module("dataset_generator")

    def test_generator_produces_valid_records(self):
        records = self.generator.generate_records(32, seed=7)
        self.assertEqual(len(records), 32)
        for rec in records:
            self.assertIn("id", rec)
            self.assertIsInstance(rec["participants"], list)
            self.assertGreaterEqual(len(rec["participants"]), 2)
            self.assertIn("duration_min", rec)
            self.assertIn("required", rec)
            self.assertGreaterEqual(len(rec["required"]), 1)
            self.assertIn("scenario", rec)

    def test_unsolvable_scenarios_exist(self):
        records = self.generator.generate_records(200, seed=42)
        unsolvable = [r for r in records if r["scenario"] == "unsolvable"]
        self.assertGreater(len(unsolvable), 0)

    def test_seed_reproducibility(self):
        a = self.generator.generate_records(16, seed=123)
        b = self.generator.generate_records(16, seed=123)
        self.assertEqual(len(a), len(b))
        for ra, rb in zip(a, b):
            self.assertEqual(ra["id"], rb["id"])
            self.assertEqual(ra["participants"], rb["participants"])


class CalendarRewardTest(unittest.TestCase):
    def setUp(self):
        self.reward = _load_module("reward")
        self.game = _load_module("game")

    def _record(self, source, tool_calls):
        return SimpleNamespace(source_record=source, tool_calls=tool_calls)

    def test_reward_scores_valid_confirm(self):
        source = {
            "participants": [
                {"name": "Alice", "utc_offset_hours": +8, "available_blocks": [(900, 1700)]},
                {"name": "Bob", "utc_offset_hours": 0, "available_blocks": [(100, 500)]},
            ],
            "duration_min": 60,
            "required": ["Alice", "Bob"],
        }
        # 03:00 UTC = Alice 11:00, Bob 03:00 → both in range
        record = self._record(
            source,
            [
                {"name": "query_availability", "arguments": {"name": "Alice"}},
                {"name": "query_availability", "arguments": {"name": "Bob"}},
                {"name": "confirm", "arguments": {"utc_time": 300}},
            ],
        )
        score = self.reward.reward_fn(record)
        self.assertGreater(score, 0.5)  # valid + efficiency bonus

    def test_reward_penalizes_conflict(self):
        source = {
            "participants": [
                {"name": "Alice", "utc_offset_hours": +8, "available_blocks": [(900, 1200)]},
                {"name": "Bob", "utc_offset_hours": 0, "available_blocks": [(100, 300)]},
            ],
            "duration_min": 60,
            "required": ["Alice", "Bob"],
        }
        # 05:00 UTC = Alice 13:00 (outside 9-12), Bob 05:00 (outside 1-3) → conflict
        record = self._record(
            source,
            [{"name": "confirm", "arguments": {"utc_time": 500}}],
        )
        score = self.reward.reward_fn(record)
        self.assertLess(score, 0)  # conflict → negative

    def test_reward_penalizes_no_confirm(self):
        source = {
            "participants": [
                {"name": "Alice", "utc_offset_hours": +8, "available_blocks": [(900, 1700)]},
            ],
            "duration_min": 60,
            "required": ["Alice"],
        }
        # Only queried, never proposed or confirmed.
        record = self._record(
            source,
            [
                {"name": "query_availability", "arguments": {"name": "Alice"}},
                {"name": "query_availability", "arguments": {"name": "Alice"}},
            ],
        )
        score = self.reward.reward_fn(record)
        self.assertLess(score, 0)

    def test_no_tool_calls(self):
        record = SimpleNamespace(source_record={}, tool_calls=[])
        score = self.reward.reward_fn(record)
        self.assertEqual(score, -1.0)

    def test_proposed_but_not_confirmed(self):
        source = {
            "participants": [
                {"name": "Alice", "utc_offset_hours": +8, "available_blocks": [(900, 1700)]},
            ],
            "duration_min": 60,
            "required": ["Alice"],
        }
        record = self._record(
            source,
            [{"name": "propose_slot", "arguments": {"utc_time": 1000, "participants": ["Alice"]}}],
        )
        score = self.reward.reward_fn(record)
        self.assertEqual(score, -0.75)


class CalendarDatasetLoaderTest(unittest.TestCase):
    def test_loader_returns_prompt_dicts(self):
        loader = _load_module("dataset_loader")
        import tempfile, json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({
                "id": "cal-00000",
                "participants": [
                    {"name": "Alice", "utc_offset_hours": +8, "available_blocks": [(900, 1700)]},
                    {"name": "Bob", "utc_offset_hours": 0, "available_blocks": [(800, 1600)]},
                ],
                "duration_min": 60,
                "required": ["Alice", "Bob"],
                "scenario": "solvable",
            }) + "\n")
            f.write(json.dumps({
                "id": "cal-00001",
                "participants": [
                    {"name": "Carol", "utc_offset_hours": -5, "available_blocks": [(1000, 1500)]},
                ],
                "duration_min": 30,
                "required": ["Carol"],
                "scenario": "solvable",
            }) + "\n")
            tmp_path = f.name
        try:
            result = loader.load_training_dataset(tmp_path)
            self.assertEqual(len(result), 2)
            self.assertIn("prompt", result[0])
            self.assertIn("Schedule 60min", result[0]["prompt"])
            self.assertIn("required", result[0])
            self.assertEqual(result[0]["required"], ["Alice", "Bob"])
        finally:
            import os
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()