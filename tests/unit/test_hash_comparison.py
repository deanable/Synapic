"""
Compatibility Tests for Hash Comparison Behavior
================================================

This module verifies hash-comparison logic from the broader unit-test suite and
acts as a guard against regressions in duplicate detection scoring.
"""

import pytest
from src.core.dedup.hash_comparison import are_hashes_similar

class TestAreHashesSimilar:
    @pytest.mark.parametrize(
        "hash1, hash2, expected",
        [
            # Exact matches (100% similarity)
            ("1234567890abcdef", "1234567890abcdef", True),
            ("0000000000000000", "0000000000000000", True),
            ("ffffffffffffffff", "ffffffffffffffff", True),

            # Very similar, above default threshold (95.0)
            # length 16 hex chars = 64 bits.
            # 1 bit difference = 1/64 = 1.5625% difference = 98.4375% similarity > 95.0
            ("1234567890abcdef", "1234567890abcdee", True), # 'f' (1111) vs 'e' (1110) - 1 bit

            # Different lengths (should return False)
            ("123", "1234", False),
            ("", "a", False),

            # Invalid hex strings (should be caught by except ValueError and return False)
            ("invalid123456789", "invalid123456789", False), # Not valid hex
            ("1234567890abcdex", "1234567890abcdef", False), # 'x' is not hex
        ]
    )
    def test_default_threshold(self, hash1, hash2, expected):
        """Test with default threshold (95.0%)."""
        assert are_hashes_similar(hash1, hash2) is expected

    @pytest.mark.parametrize(
        "hash1, hash2, threshold, expected",
        [
            # Exact matches should always be True regardless of valid threshold
            ("1234567890abcdef", "1234567890abcdef", 100.0, True),
            ("1234567890abcdef", "1234567890abcdef", 99.9, True),
            ("1234567890abcdef", "1234567890abcdef", 50.0, True),

            # Below threshold
            # 'f' (1111) vs '0' (0000) - 4 bits diff. 4/64 = 6.25% diff = 93.75% similarity.
            ("1234567890abcdef", "1234567890abcde0", 95.0, False), # 93.75 < 95.0

            # Above/At custom threshold
            ("1234567890abcdef", "1234567890abcde0", 90.0, True), # 93.75 > 90.0
            ("1234567890abcdef", "1234567890abcde0", 93.75, True), # 93.75 >= 93.75
        ]
    )
    def test_custom_threshold(self, hash1, hash2, threshold, expected):
        """Test with custom thresholds."""
        assert are_hashes_similar(hash1, hash2, threshold) is expected

    def test_empty_hashes(self):
        """Test empty hashes (should handle ValueError nicely)."""
        # Empty string hex conversion fails or raises ValueError based on calculation logic
        assert are_hashes_similar("", "") is False
