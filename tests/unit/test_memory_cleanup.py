"""
Tests for Memory Cleanup Behavior
=================================

These tests target code paths where large images, model outputs, or payloads
must be released promptly to keep the desktop application stable over long
batch-processing runs.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.processing import ProcessingManager
from src.core.session import Session
from collections import deque


class TestMemoryCleanup(unittest.TestCase):
    """Tests for memory cleanup in the processing pipeline."""

    @patch('src.core.processing.gc.collect')
    def test_cleanup_after_local_job(self, mock_gc_collect):
        """Model is unloaded and gc.collect called after local job."""
        session = Session()
        session.engine.provider = "local"
        session.engine.device = "cpu"  # Avoid CUDA path which imports torch locally

        log_cb = MagicMock()
        prog_cb = MagicMock()

        manager = ProcessingManager(session, log_cb, prog_cb)
        manager.model = MagicMock()  # Simulate loaded model

        with patch.object(ProcessingManager, '_fetch_items', return_value=[]), \
             patch.object(ProcessingManager, '_init_local_model'):
            manager._run_job()

        # Verify model unloaded
        self.assertIsNone(manager.model)
        # Verify gc.collect called at end of job
        mock_gc_collect.assert_called()

    @patch('src.core.processing.gc.collect')
    def test_periodic_gc_collect_per_item(self, mock_gc_collect):
        """gc.collect is called periodically during item processing."""
        session = Session()
        session.engine.provider = "local"
        session.engine.device = "cpu"

        log_cb = MagicMock()
        prog_cb = MagicMock()

        manager = ProcessingManager(session, log_cb, prog_cb)

        # Create 6 fake items (triggers gc.collect at items 3 and 6 with new interval)
        from pathlib import Path
        fake_items = [Path(f"fake_{i}.jpg") for i in range(6)]

        with patch.object(ProcessingManager, '_fetch_items', return_value=fake_items), \
             patch.object(ProcessingManager, '_init_local_model'), \
             patch.object(ProcessingManager, '_process_single_item'):
            manager._run_job()

        # gc.collect should have been called multiple times:
        # at items 3 and 6 (every 3 items), plus once at end of job
        self.assertGreaterEqual(mock_gc_collect.call_count, 3)

    def test_session_results_bounded(self):
        """Session results list is bounded to prevent unbounded growth."""
        session = Session()
        
        # Verify results is a deque with maxlen
        self.assertIsInstance(session.results, deque)
        self.assertEqual(session.results.maxlen, 500)
        
        # Fill beyond capacity
        for i in range(600):
            session.results.append({"filename": f"test_{i}.jpg", "status": "Success", "tags": "test"})
        
        # Should be capped at 500
        self.assertEqual(len(session.results), 500)
        # Oldest items should have been dropped
        self.assertEqual(session.results[0]["filename"], "test_100.jpg")

    def test_session_results_reset_preserves_bound(self):
        """reset_stats creates a fresh bounded deque."""
        session = Session()
        session.results.append({"filename": "test.jpg", "status": "Success", "tags": "test"})
        
        session.reset_stats()
        
        # Results should be empty but still bounded
        self.assertEqual(len(session.results), 0)
        self.assertIsInstance(session.results, deque)
        self.assertEqual(session.results.maxlen, 500)


if __name__ == '__main__':
    unittest.main()