"""
Legacy Dedup Strategy Unit Tests
================================

This file covers dedup strategy behavior from the higher-level unit-test layer.
It complements the narrower tests under `tests/unit/core/dedup`.
"""

import unittest

from src.core.dedup.dedup_strategies import apply_keep_first, DedupDecision
from src.core.dedup.dedup_engine import DuplicateGroup

class TestDedupStrategies(unittest.TestCase):
    def test_apply_keep_first_empty_group(self):
        group = DuplicateGroup(items=[], similarity_scores={}, hash_type="phash")
        decision = apply_keep_first(group)
        self.assertIsNone(decision.keep_item)
        self.assertEqual(decision.remove_items, [])
        self.assertEqual(decision.reason, "Empty group")

    def test_apply_keep_first_single_item(self):
        group = DuplicateGroup(items=["item1"], similarity_scores={"item1": 100.0}, hash_type="phash")
        decision = apply_keep_first(group)
        self.assertEqual(decision.keep_item, "item1")
        self.assertEqual(decision.remove_items, [])
        self.assertEqual(decision.reason, "Keep first item")

    def test_apply_keep_first_multiple_items(self):
        group = DuplicateGroup(
            items=["item1", "item2", "item3"],
            similarity_scores={"item1": 100.0, "item2": 98.0, "item3": 95.0},
            hash_type="phash"
        )
        decision = apply_keep_first(group)
        self.assertEqual(decision.keep_item, "item1")
        self.assertEqual(decision.remove_items, ["item2", "item3"])
        self.assertEqual(decision.reason, "Keep first item")

if __name__ == "__main__":
    unittest.main()
