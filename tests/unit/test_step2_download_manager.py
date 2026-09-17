"""
Regression Tests: DownloadManagerDialog Size Filter
===================================================

These tests reproduce the bug where a Hub search showed results briefly
("Loading...") and then filtered every model out once real sizes arrived:

Pass 1 rendered all size_bytes=0, pinning the size-filter max to 0. Pass 2
(refresh_size_filter=False) never re-derived the bounds, so every model with
a real size was dropped by _get_size_filtered_results().

The dialog is driven through a headless double (no real CustomTkinter), so
these tests run in CI without a display.
"""

import unittest
from unittest.mock import MagicMock, patch

# -------------------------------------------------------------------------
# Headless widget doubles
# -------------------------------------------------------------------------
# We deliberately do NOT swap sys.modules["customtkinter"]: import order must
# not affect other tests. Instead, setUpClass patches the `ctk` attribute on
# the already-imported step2_tagging module, so every widget construction
# inside the dialog methods uses these doubles, deterministically.
mock_ctk = MagicMock(name="customtkinter_mock")


def _make_dialog():
    """Build a DownloadManagerDialog with __init__ stubbed out (headless)."""
    from src.ui.steps import step2_tagging

    dialog = step2_tagging.DownloadManagerDialog.__new__(step2_tagging.DownloadManagerDialog)

    # State normally initialized in __init__
    dialog._search_results_cache = []
    dialog._size_filter_min = 0
    dialog._size_filter_max = 0
    dialog._size_filter_active_max = 0
    dialog.search_filter_var = MagicMock()
    dialog.search_filter_var.get.return_value = "multimodal"

    # UI doubles
    dialog.results_frame = MagicMock()
    dialog.lbl_status = MagicMock()
    dialog.size_slider = MagicMock()
    dialog.size_label = MagicMock()
    dialog.local_tab = MagicMock()

    return dialog


class TestSizeFilterStateMachine(unittest.TestCase):
    """Two-pass rendering: placeholder sizes first, real sizes later."""

    @classmethod
    def setUpClass(cls):
        # Redirect the dialog module's widget layer to headless doubles.
        # Scoped to the module attribute, so real customtkinter (if already
        # imported by another test) is untouched and import order is moot.
        from src.ui.steps import step2_tagging

        cls._ctk_patcher = patch.object(step2_tagging, "ctk", mock_ctk)
        cls._ctk_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls._ctk_patcher.stop()

    def setUp(self):
        self.dialog = _make_dialog()

    def test_placeholder_sizes_do_not_enable_filter(self):
        """Pass 1: all-zero sizes keep the filter pinned at 0."""
        results = [{"id": "m/a", "size_bytes": 0, "size_str": "Loading...", "capability": "x"}]
        self.dialog.show_search_results(results)

        self.assertEqual(self.dialog._size_filter_max, 0)
        self.assertEqual(self.dialog._size_filter_active_max, 0)

    def test_real_sizes_pass_not_filtered_out(self):
        """
        Pass 2 (refresh_size_filter=False) must re-derive bounds when the
        filter is still pinned at 0, so sized models survive the filter.
        Regression: this returned an empty list and the UI said
        "No models match the current size filter."
        """
        # Pass 1: placeholders
        self.dialog.show_search_results(
            [{"id": "m/a", "size_bytes": 0, "size_str": "Loading...", "capability": "x"}]
        )
        # Pass 2: real sizes arrive, slider refresh intentionally skipped
        results = [
            {"id": "m/a", "size_bytes": 3 * 1024**3, "size_str": "3.0 GB", "capability": "x"},
            {"id": "m/b", "size_bytes": 500 * 1024**2, "size_str": "500.0 MB", "capability": "y"},
        ]
        self.dialog.show_search_results(results, refresh_size_filter=False)

        self.assertEqual(self.dialog._size_filter_max, 3 * 1024**3)
        self.assertEqual(len(self.dialog._get_size_filtered_results()), 2)

    def test_user_slider_position_preserved_after_sizes_arrive(self):
        """Once real bounds exist, pass 2 must not reset the user's slider."""
        results = [
            {"id": "m/a", "size_bytes": 3 * 1024**3, "size_str": "3.0 GB", "capability": "x"},
            {"id": "m/b", "size_bytes": 500 * 1024**2, "size_str": "500.0 MB", "capability": "y"},
        ]
        self.dialog.show_search_results(results, refresh_size_filter=False)

        # User drags the slider down to 600 MB
        self.dialog._on_size_slider_change(600 * 1024**2)
        self.assertEqual(self.dialog._size_filter_active_max, 600 * 1024**2)

        # A later pass with the same data must not resurrect dropped models
        self.dialog.show_search_results(results, refresh_size_filter=False)
        self.assertEqual(self.dialog._size_filter_active_max, 600 * 1024**2)
        self.assertEqual(len(self.dialog._get_size_filtered_results()), 1)

    def test_zero_byte_models_still_visible_with_real_sizes(self):
        """Models with unknown size (0) should never be filtered out."""
        results = [
            {"id": "m/a", "size_bytes": 3 * 1024**3, "size_str": "3.0 GB", "capability": "x"},
            {"id": "m/unknown", "size_bytes": 0, "size_str": "Unknown", "capability": "y"},
        ]
        self.dialog.show_search_results(results, refresh_size_filter=False)

        filtered = self.dialog._get_size_filtered_results()
        self.assertIn(
            "m/unknown", [item["id"] for item in filtered],
            "size_bytes=0 models must pass the <= max filter"
        )


if __name__ == "__main__":
    unittest.main()
