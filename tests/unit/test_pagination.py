"""
Unit tests for Daminion auto-pagination in ProcessingManager.

Verifies that:
1. When auto_paginate=True and the first fetch returns exactly 500 items,
   ProcessingManager reloads the search (always at offset 0) to get the next batch.
2. When auto_paginate=True and a subsequent reload returns fewer than 500 items,
   the loop terminates correctly.
3. When auto_paginate=False, get_items_filtered is called exactly once regardless
   of page size.
4. The infinite-loop guard stops processing if the same item IDs are returned twice.
5. get_items_filtered itself always returns at most one batch (single_page=True fix).
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.processing import ProcessingManager


def _make_dummy_items(count: int, id_offset: int = 0):
    """Return a list of minimal Daminion item dicts with unique IDs."""
    return [{"id": id_offset + i, "fileName": f"img_{id_offset + i}.jpg"} for i in range(count)]


def _make_manager(auto_paginate: bool):
    """Build a ProcessingManager wired to a fake Daminion session."""
    session = MagicMock()
    session.datasource.type = "daminion"
    session.datasource.daminion_scope = "all"
    session.datasource.daminion_saved_search_id = None
    session.datasource.daminion_saved_search = None
    session.datasource.daminion_collection_id = None
    session.datasource.daminion_catalog_id = None
    session.datasource.daminion_search_term = None
    session.datasource.daminion_untagged_keywords = False
    session.datasource.daminion_untagged_categories = False
    session.datasource.daminion_untagged_description = False
    session.datasource.status_filter = "all"
    session.datasource.max_items = 0

    session.engine.provider = "local"  # Avoid real model loading
    session.engine.model_id = "test-model"
    session.engine.task = "image-classification"
    session.engine.device = "cpu"

    session.daminion_client = MagicMock()
    session.processed_items = 0
    session.failed_items = 0
    session.total_items = 0

    def _reset_stats():
        session.processed_items = 0
        session.failed_items = 0
        session.total_items = 0

    session.reset_stats.side_effect = _reset_stats

    log_cb = MagicMock()
    progress_cb = MagicMock()

    manager = ProcessingManager(
        session=session,
        log_callback=log_cb,
        progress_callback=progress_cb,
        auto_paginate=auto_paginate,
    )
    return manager, session


class TestAutoPaginateManagerLoop(unittest.TestCase):
    """Verify ProcessingManager drives pagination with the reload-search strategy."""

    def _run_fetch_only(self, auto_paginate: bool, page_sizes: list):
        """
        Simulate _run_job up to the fetch loop only (no real item processing).

        Returns the number of times _fetch_items was called.  In the new
        reload-search strategy every call uses offset=0, so what matters is
        the call *count*, not the offset sequence.

        Each successive page returns items with distinct IDs so the
        infinite-loop guard does not trigger prematurely.
        """
        manager, session = _make_manager(auto_paginate)

        call_count = [0]

        def fake_get_items_filtered(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            size = page_sizes[idx] if idx < len(page_sizes) else 0
            # Use a distinct ID range per page so the duplicate-ID guard
            # (which only fires when identical IDs are returned) stays silent.
            return _make_dummy_items(size, id_offset=idx * 10000)

        session.daminion_client.get_items_filtered.side_effect = fake_get_items_filtered

        # Patch _process_single_item so we don't need a real model
        with patch.object(manager, '_process_single_item', return_value=None):
            with patch.object(manager, '_init_local_model', return_value=None):
                manager._run_job()

        return call_count[0]

    def test_auto_paginate_calls_second_fetch_when_first_is_full(self):
        """With auto_paginate=True a full first page should trigger a second reload."""
        fetches = self._run_fetch_only(
            auto_paginate=True,
            page_sizes=[500, 200],  # page 1 full → reload → page 2 partial → done
        )
        self.assertEqual(fetches, 2,
                         "Expected exactly two fetches: initial load + one reload")

    def test_auto_paginate_three_full_pages_then_empty(self):
        """Three full reloads followed by an empty response terminates cleanly."""
        fetches = self._run_fetch_only(
            auto_paginate=True,
            page_sizes=[500, 500, 500, 0],
        )
        self.assertEqual(fetches, 4,
                         "Expected 4 fetch calls: 3 full batches + 1 empty that stops the loop")

    def test_no_auto_paginate_stops_after_one_full_page(self):
        """With auto_paginate=False the loop should stop after the first page."""
        fetches = self._run_fetch_only(
            auto_paginate=False,
            page_sizes=[500, 200],  # second fetch should never happen
        )
        self.assertEqual(fetches, 1,
                         "Expected only one fetch when auto_paginate=False")

    def test_no_items_on_first_page_logs_and_exits(self):
        """Empty first page should exit cleanly without calling fetch again."""
        fetches = self._run_fetch_only(
            auto_paginate=True,
            page_sizes=[0],
        )
        self.assertEqual(fetches, 1,
                         "Expected exactly one fetch attempt even when empty")

    def test_infinite_loop_guard_fires_on_duplicate_ids(self):
        """Reload returning identical IDs should trigger the guard and stop."""
        manager, session = _make_manager(auto_paginate=True)

        # Both calls return identical item IDs – simulates a server that does
        # not filter out tagged items, which would loop forever without the guard.
        fixed_items = _make_dummy_items(500, id_offset=0)
        call_count = [0]

        def fake_get_items_filtered(**kwargs):
            call_count[0] += 1
            return list(fixed_items)  # same IDs every time

        session.daminion_client.get_items_filtered.side_effect = fake_get_items_filtered

        with patch.object(manager, '_process_single_item', return_value=None):
            with patch.object(manager, '_init_local_model', return_value=None):
                manager._run_job()

        # Should have fetched exactly twice: first call (processes items),
        # second call (same IDs detected → guard fires → stops).
        self.assertEqual(call_count[0], 2,
                         "Infinite-loop guard should stop after two fetches with identical IDs")


class TestGetItemsFilteredSinglePage(unittest.TestCase):
    """Verify get_items_filtered itself never returns more than one batch."""

    def setUp(self):
        from src.core.daminion_client import DaminionClient
        self.client = DaminionClient.__new__(DaminionClient)
        self.client._tag_name_to_id = {}
        self.client._tag_id_to_name = {}
        self.client._tag_schema = []

        # Mock the underlying API layer
        self.mock_api = MagicMock()
        self.client._api = self.mock_api

    def _setup_search_returning(self, batch_sizes: list):
        """Configure the mock search to return given batch sizes in sequence."""
        results = iter([_make_dummy_items(n) for n in batch_sizes])

        def _search(*args, **kwargs):
            try:
                return next(results)
            except StopIteration:
                return []

        self.mock_api.media_items.search.side_effect = _search

    def test_always_returns_single_batch_on_first_call(self):
        """Even when the underlying API could return more, only one batch."""
        # Simulate API returning 500 items for every call
        self._setup_search_returning([500, 500, 500])

        items = self.client.get_items_filtered(scope="all", start_index=0)
        # Should have stopped after the first batch of 500
        self.assertEqual(len(items), 500)
        self.assertEqual(self.mock_api.media_items.search.call_count, 1,
                         "get_items_filtered should issue exactly one API call per invocation")

    def test_partial_batch_returned_correctly(self):
        """A partial batch (< 500) is returned as-is."""
        self._setup_search_returning([237])
        items = self.client.get_items_filtered(scope="all", start_index=0)
        self.assertEqual(len(items), 237)

    def test_second_page_offset_respected(self):
        """start_index=500 is forwarded to the API correctly."""
        self._setup_search_returning([300])
        self.client.get_items_filtered(scope="all", start_index=500)
        call_kwargs = self.mock_api.media_items.search.call_args
        # The API should have been told to start at index 500
        self.assertEqual(call_kwargs.kwargs.get("index", None), 500)



if __name__ == "__main__":
    unittest.main()


class TestProgressCompletionSignal(unittest.TestCase):
    """
    Verify that progress_callback receives more_pages=False only on the very
    last item of the very last page, so the UI can safely switch button states.

    This guards against the regression where pct=1.0 fired after every full
    500-item page (because processed_items / total_items = 500/500 = 1.0)
    causing the Start button to re-enable mid-run.
    """

    def _collect_progress_calls(self, auto_paginate: bool, page_sizes: list):
        """
        Run _run_job with fake items and collect every progress_callback call.
        Returns a list of (pct, current, total, more_pages) tuples.
        """
        manager, session = _make_manager(auto_paginate)

        call_log = []  # page-fetch index tracker for fake_get_items_filtered
        call_counter = [0]

        def fake_get_items_filtered(**kwargs):
            idx = call_counter[0]
            call_counter[0] += 1
            size = page_sizes[idx] if idx < len(page_sizes) else 0
            return _make_dummy_items(size)

        session.daminion_client.get_items_filtered.side_effect = fake_get_items_filtered

        progress_calls = []

        def capture_progress(pct, current, total, more_pages=False):
            progress_calls.append((pct, current, total, more_pages))

        manager.progress = capture_progress

        with patch.object(manager, '_process_single_item', return_value=None):
            with patch.object(manager, '_init_local_model', return_value=None):
                manager._run_job()

        return progress_calls

    def test_multi_page_completion_signal_on_last_item_only(self):
        """
        With auto_paginate=True and pages [500, 200], more_pages=False should
        appear exactly once — on the final item of the second (partial) page.
        """
        calls = self._collect_progress_calls(
            auto_paginate=True,
            page_sizes=[500, 200],
        )
        # Filter out the page-boundary "start of page" calls (current==0 at start)
        item_calls = [c for c in calls if c[1] > 0]  # current > 0 → per-item calls

        # The very last call must have more_pages=False
        self.assertFalse(
            item_calls[-1][3],
            "Last progress call should have more_pages=False"
        )
        # All calls before the last must have more_pages=True
        for pct, current, total, more_pages in item_calls[:-1]:
            self.assertTrue(
                more_pages,
                f"Expected more_pages=True but got False at current={current}/{total}"
            )

    def test_single_page_no_paginate_final_call_is_done(self):
        """
        With auto_paginate=False and a single page of 300 items, the last
        progress call must signal more_pages=False.
        """
        calls = self._collect_progress_calls(
            auto_paginate=False,
            page_sizes=[300],
        )
        item_calls = [c for c in calls if c[1] > 0]
        self.assertFalse(
            item_calls[-1][3],
            "Last progress call for single-page run should have more_pages=False"
        )

    def test_full_first_page_does_not_prematurely_signal_done(self):
        """
        With auto_paginate=True and pages [500, 50], the 500th item's progress
        call must have more_pages=True (job not done — second page is coming).
        """
        calls = self._collect_progress_calls(
            auto_paginate=True,
            page_sizes=[500, 50],
        )
        item_calls = [c for c in calls if c[1] > 0]
        # The call where current==500 is the last item of the first page
        first_page_last = next(
            (c for c in item_calls if c[1] == 500), None
        )
        self.assertIsNotNone(first_page_last, "Should have a progress call for item 500")
        self.assertTrue(
            first_page_last[3],
            "Item 500/500 should still have more_pages=True (second page not fetched yet)"
        )
