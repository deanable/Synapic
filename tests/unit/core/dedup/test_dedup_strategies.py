import pytest
from unittest.mock import patch, MagicMock
import os
import sys
import time

# Mock out dependencies to prevent import errors in unit test environment
sys.modules['PIL'] = MagicMock()
sys.modules['imagehash'] = MagicMock()

from src.core.dedup.dedup_strategies import get_file_metadata

def test_get_file_metadata_success():
    """Test get_file_metadata returns correct values when os.stat succeeds."""
    class MockStatResult:
        st_size = 12345
        st_mtime = 1678886400.0

    with patch('os.stat', return_value=MockStatResult()):
        result = get_file_metadata('dummy_file.txt')

    assert result['size'] == 12345
    assert result['mtime'] == 1678886400.0

def test_get_file_metadata_exception():
    """Test get_file_metadata returns fallback values when os.stat raises an exception."""
    with patch('os.stat', side_effect=Exception("File not found")):
        result = get_file_metadata('missing_file.txt')

    assert result['size'] == 0
    assert result['mtime'] > 0
