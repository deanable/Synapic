import pytest
import sys
from unittest.mock import MagicMock

import unittest.mock
# Mock image processing dependencies to allow tests to run in environments lacking them
unittest.mock.patch.dict('sys.modules', {'PIL': MagicMock(), 'PIL.Image': MagicMock(), 'imagehash': MagicMock()}).start()

from src.core.dedup.hash_comparison import (
    calculate_hamming_distance,
    calculate_similarity_percentage,
    are_hashes_similar,
    are_hashes_exact_match,
)

class TestCalculateHammingDistance:
    def test_identical_hashes(self):
        hash1 = "1a2b3c4d"
        assert calculate_hamming_distance(hash1, hash1) == 0

    def test_different_hashes(self):
        # 1 = 0001, 2 = 0010 -> XOR is 0011 -> 2 bits difference
        hash1 = "1"
        hash2 = "2"
        assert calculate_hamming_distance(hash1, hash2) == 2

        # 0 = 0000, f = 1111 -> XOR is 1111 -> 4 bits difference
        assert calculate_hamming_distance("0", "f") == 4

        # Multiple hex digits
        # 1a = 0001 1010
        # 2b = 0010 1011
        # XOR: 0011 0001 -> 3 bits difference
        assert calculate_hamming_distance("1a", "2b") == 3

    def test_case_insensitivity(self):
        # f = 1111, F = 1111 -> XOR is 0000 -> 0 bits
        assert calculate_hamming_distance("f", "F") == 0
        assert calculate_hamming_distance("1a2b", "1A2B") == 0

    def test_different_length_raises_value_error(self):
        with pytest.raises(ValueError, match="Hashes must be of the same length to calculate Hamming distance."):
            calculate_hamming_distance("123", "12")

    def test_invalid_hex_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid hex string provided."):
            calculate_hamming_distance("xyz", "123")

class TestCalculateSimilarityPercentage:
    def test_identical_hashes(self):
        # 0 distance means 100% similarity
        assert calculate_similarity_percentage(0, 64) == 100.0

    def test_partial_similarity(self):
        # 32 distance out of 64 bits = 50%
        assert calculate_similarity_percentage(32, 64) == 50.0

        # 16 distance out of 64 bits = 75%
        assert calculate_similarity_percentage(16, 64) == 75.0

    def test_completely_different(self):
        # 64 distance out of 64 bits = 0%
        assert calculate_similarity_percentage(64, 64) == 0.0

    def test_bounding_min(self):
        # If distance > bit_length, we should cap at 0.0
        assert calculate_similarity_percentage(70, 64) == 0.0

    def test_zero_or_negative_length_raises_value_error(self):
        with pytest.raises(ValueError, match="Bit length must be positive."):
            calculate_similarity_percentage(0, 0)
        with pytest.raises(ValueError, match="Bit length must be positive."):
            calculate_similarity_percentage(0, -1)

    def test_negative_distance_raises_value_error(self):
        with pytest.raises(ValueError, match="Hamming distance cannot be negative."):
            calculate_similarity_percentage(-1, 64)

class TestAreHashesSimilar:
    def test_identical_hashes(self):
        hash1 = "1a2b3c4d5e6f7089"
        assert are_hashes_similar(hash1, hash1) is True

    def test_similar_hashes_above_threshold(self):
        # 16 hex chars = 64 bits.
        # threshold is 95.0.
        # 95% of 64 bits is 60.8 bits.
        # So up to 3 bits of difference is allowed (61 / 64 = 95.3125%).
        # Difference of 4 bits: 60 / 64 = 93.75% < 95.0%.

        hash1 = "0000000000000000"
        # 1 bit difference
        hash2 = "0000000000000001"
        assert are_hashes_similar(hash1, hash2, threshold=95.0) is True

        # 3 bits difference
        hash3 = "0000000000000007"
        assert are_hashes_similar(hash1, hash3, threshold=95.0) is True

    def test_similar_hashes_below_threshold(self):
        hash1 = "0000000000000000"
        # 4 bits difference = 93.75% similarity
        hash2 = "000000000000000f"
        assert are_hashes_similar(hash1, hash2, threshold=95.0) is False

        # If threshold is lowered, it should pass
        assert are_hashes_similar(hash1, hash2, threshold=90.0) is True

    def test_different_lengths_returns_false(self):
        assert are_hashes_similar("123", "12") is False

    def test_invalid_hex_string_returns_false(self):
        # The function catches ValueError and returns False
        assert are_hashes_similar("xyz", "123") is False

class TestAreHashesExactMatch:
    def test_identical_hashes(self):
        assert are_hashes_exact_match("123", "123") is True

    def test_different_hashes(self):
        assert are_hashes_exact_match("123", "124") is False
        assert are_hashes_exact_match("123", "12") is False
