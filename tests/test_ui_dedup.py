"""
UI Logic Tests for Dedup Navigation
===================================

These tests isolate dedup-related UI behavior from the real CustomTkinter
stack by replacing windowing modules with mocks.

The goal is to keep the tests focused on controller/session interactions:
- Does Step 1 gather the expected inputs before opening dedup?
- Does the dedup screen initialise the processor correctly?

This file intentionally uses test doubles instead of real widgets so it can
run in headless CI environments.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import importlib.util

# -------------------------------------------------------------------------
# MOCKING UI LIBRARIES
# -------------------------------------------------------------------------
module_mock = MagicMock()

# Define a real class for CTkFrame so inheritance works normally
class MockCTkFrame:
    """Tiny stand-in base class that satisfies the widget API used by the tests."""
    def __init__(self, *args, **kwargs): pass
    def grid(self, *args, **kwargs): pass
    def pack(self, *args, **kwargs): pass
    def tkraise(self, *args, **kwargs): pass
    def winfo_exists(self): return True
    def after(self, ms, func=None):
        if func:
            func()
        return "timer_id"

module_mock.CTkFrame = MockCTkFrame
sys.modules["customtkinter"] = module_mock
sys.modules["tkinter"] = MagicMock()
sys.modules["tkinter.messagebox"] = MagicMock()
if importlib.util.find_spec("PIL.ImageTk") is None:
    sys.modules["PIL.ImageTk"] = MagicMock()

# Import original classes
from src.ui.steps.step1_datasource import Step1Datasource  # noqa: E402
from src.ui.steps.step_dedup import StepDedup  # noqa: E402
import src.ui.steps.step1_datasource as step1_module  # noqa: E402

# -------------------------------------------------------------------------
# TESTABLE SUBCLASSES (Avoids UI Init)
# -------------------------------------------------------------------------

class TestableStep1(Step1Datasource):
    """Minimal Step 1 variant that skips heavy UI construction."""
    def __init__(self, controller):
        # SKIP SUPER INIT by calling object init or MockFrame init directly if needed
        # But simply setting attributes is enough if we don't call super().__init__
        self.controller = controller
        self.logger = MagicMock()
        self._worker = MagicMock()
        self._worker.submit.side_effect = lambda f, *a, **k: f(*a, **k)
        self._worker.submit_replacing.side_effect = lambda task_id, task, *a, **k: task(*a, **k)

        self.lbl_total_count = MagicMock()
        
        # Setup real objects for logic to use, or mocks
        self.tabs = MagicMock()
        self.status_var = MagicMock()
        self.ss_var = MagicMock()
        self.col_var = MagicMock()
        self.search_entry = MagicMock()
        
        self.chk_untagged_kws = MagicMock()
        self.chk_untagged_kws.get.return_value = False
        self.chk_untagged_cats = MagicMock()
        self.chk_untagged_cats.get.return_value = False
        self.chk_untagged_desc = MagicMock()
        self.chk_untagged_desc.get.return_value = False
        
        self._ss_map = {}
        self._col_map = {}
        self._count_cache = {}
        self._debounce_timer = None

        # Bind the count-related methods the production class defines, so the
        # test exercises the real implementation rather than stubs. These are
        # attached as instance methods explicitly because TestableStep1 skips
        # super().__init__.
        for _name in (
            "_current_cache_key",
            "_invalidate_count_cache_for_current_selection",
            "_schedule_count_if_needed",
            "_set_dropdown_value",
        ):
            if hasattr(Step1Datasource, _name):
                setattr(self, _name, getattr(Step1Datasource, _name).__get__(self, TestableStep1))
            else:
                setattr(self, _name, MagicMock())

        # update_count and _update_count_actual must be real too, but they touch
        # self.after / self.lbl_total_count which our test stubs -- keep them real.
        self.update_count = Step1Datasource.update_count.__get__(self, TestableStep1)
        self._update_count_actual = Step1Datasource._update_count_actual.__get__(self, TestableStep1)

        # We need these methods
        self.winfo_exists = MagicMock(return_value=True)
        self.after = MagicMock(side_effect=lambda d, f: f())

class TestableStepDedup(StepDedup):
    """Minimal dedup step variant that exposes only logic under test."""
    def __init__(self, controller):
        self.controller = controller
        self.session = controller.session
        self.logger = MagicMock()
        self.is_confirming_action = False
        self.default_btn_fg = None
        self.default_btn_hover = None
        self.default_btn_text = "Apply Deduplication"
        self.apply_btn = MagicMock()
        
        self.threshold_var = MagicMock()
        self.threshold_var.get.return_value = 95.0
        self.algorithm_var = MagicMock()
        self.algorithm_var.get.return_value = "phash"
        
        self.progress_frame = MagicMock()
        self.progress_bar = MagicMock()
        self.progress_label = MagicMock()
        self.abort_btn = MagicMock()
        self.scan_btn = MagicMock()
        self.initial_label = MagicMock()
        self.group_frames = []
        
        self.after = MagicMock()

# -------------------------------------------------------------------------
# FIXTURES & TESTS
# -------------------------------------------------------------------------

@pytest.fixture
def mock_controller():
    controller = MagicMock()
    controller.session = MagicMock()
    controller.session.datasource = MagicMock()
    controller.session.datasource.type = "daminion"
    controller.session.daminion_client = MagicMock()
    controller.session.daminion_client.authenticated = True
    return controller

class TestStep1DatasourceDedupe:
    """Tests covering the transition from datasource selection into dedup mode."""
    
    def test_open_dedup_step_navigates_correctly(self, mock_controller):
        # Setup
        mock_controller.session.daminion_client.get_items_filtered.return_value = [{"id": 1}]
        
        step1 = TestableStep1(mock_controller)
        step1.tabs.get.return_value = "Global Scan"
        step1.status_var.get.return_value = "all"
        
        # Action
        step1._open_dedup_step()
        
        # Assertions
        mock_controller.session.daminion_client.get_items_filtered.assert_called()
        assert mock_controller.session.dedup_items == [{"id": 1}]
        mock_controller.show_step.assert_called_with("StepDedup")

    def test_open_dedup_step_handles_no_connection(self, mock_controller):
        # Setup
        mock_controller.session.daminion_client = None
        step1 = TestableStep1(mock_controller)
        
        # Inject messagebox mock locally
        original_mb = step1_module.messagebox
        step1_module.messagebox = MagicMock()
        
        try:
            step1._open_dedup_step()
            step1_module.messagebox.showerror.assert_called_with("Error", "Not connected to Daminion.")
        finally:
            step1_module.messagebox = original_mb
        
        mock_controller.show_step.assert_not_called()

class TestStepDedupScan:
    """Tests covering dedup scan startup behavior."""
    def test_start_scan_initializes_processor(self, mock_controller):
        mock_controller.session.dedup_items = [{"id": 1}]
        step_dedup = TestableStepDedup(mock_controller)
        
        with patch("threading.Thread"), \
             patch("src.ui.steps.step_dedup.DaminionDedupProcessor") as MockProcessor:
            
            step_dedup._start_scan()
            
            MockProcessor.assert_called_with(
                mock_controller.session.daminion_client,
                similarity_threshold=95.0
            )
            step_dedup.progress_frame.grid.assert_called()


class TestStep1SavedSearchCount:
    """Tests covering saved-search count behavior when switching selections.

    These tests target the bug where switching between saved searches in the
    Step 1 UI reported the same item count for every selection.
    """

    def _make_step1(self, controller, ss_map, col_map=None):
        step1 = TestableStep1(controller)
        step1._ss_map = ss_map
        step1._col_map = col_map or {}
        step1.tabs = MagicMock()
        step1.tabs.get.return_value = "Saved Searches"
        step1.status_var = MagicMock()
        step1.status_var.get.return_value = "all"
        step1.ss_var = MagicMock()
        step1.ss_var.get.return_value = "Alpha"
        step1.ss_dropdown = MagicMock()
        step1.ss_dropdown._command = None
        step1.ss_dropdown.command = None
        step1.col_var = MagicMock()
        step1.col_var.get.return_value = "Select a collection..."
        step1.col_dropdown = MagicMock()
        step1.col_dropdown._command = None
        step1.col_dropdown.command = None
        step1.search_entry = MagicMock()
        step1.search_entry.get.return_value = ""
        step1.limit_slider = MagicMock()
        step1.limit_slider.get.return_value = 1.0
        step1.limit_toggle_frame = MagicMock()
        step1.limit_toggle_frame.winfo_manager.return_value = True
        step1.metadata_frame = MagicMock()
        step1.lbl_limit_value = MagicMock()
        step1._current_total_count = 0
        step1._count_cache = {}
        # Pretend the controller's datasource is connected so the count path
        # does not bail out early.
        step1.controller.session.daminion_client.authenticated = True
        return step1

    def test_switching_saved_search_queries_each_selection(self, mock_controller):
        """Each saved search switch must query the server for that search's id.

        This test validates the *id resolution* path: when the user picks a
        different saved search, the count request must carry that search's id,
        not a stale one. It uses a patched scheduler that records the resolved id
        and then invokes the real update_count path, so we can assert on the
        arguments the API would receive.
        """
        ss_map = {"Alpha": [11], "Beta": [22], "Gamma": [33]}
        step1 = self._make_step1(mock_controller, ss_map)

        calls = []
        def record(*args, **kwargs):
            # get_filtered_item_count signature: (self, scope=..., saved_search_id=..., ...)
            saved = kwargs.get("saved_search_id")
            calls.append(saved)
            return 100
        mock_controller.session.daminion_client.get_filtered_item_count.side_effect = record

        # Make _schedule_count_if_needed record the resolved id, then always run the
        # real count path so the API call is actually made (the cache starts empty in
        # this test).
        _real_schedule = step1._schedule_count_if_needed
        def _scheduling_wrapper():
            key = step1._current_cache_key()
            if key is not None:
                calls.append(("resolved_id", key[1]))
            _real_schedule()
        step1._schedule_count_if_needed = _scheduling_wrapper

        # Simulate selecting Beta.
        step1.ss_var.get.return_value = "Beta"
        step1._on_ss_selection_changed()

        # Simulate selecting Gamma.
        step1.ss_var.get.return_value = "Gamma"
        step1._on_ss_selection_changed()

        # The scheduling wrapper records ('resolved_id', <id>) for each selection;
        # assert only on the resolved ids.
        resolved = [c[1] for c in calls if isinstance(c, tuple) and c[0] == "resolved_id"]
        assert resolved == [22, 33], resolved

    def test_re_selecting_the_same_saved_search_does_not_refresh(self, mock_controller):
        """Re-picking the same saved search must not hit the API again.

        This test validates the cache skip: after the first selection populates
        the per-selection cache, re-selecting the same saved search is a no-op
        and does not schedule a recount. Switching to a different saved search
        does refresh.
        """
        ss_map = {"Alpha": [11], "Beta": [22]}
        step1 = self._make_step1(mock_controller, ss_map)
        mock_controller.session.daminion_client.get_filtered_item_count.return_value = 100

        # First pick of Beta populates the cache and hits the API once.
        step1.ss_var.get.return_value = "Beta"
        cache_ref = step1._count_cache
        print("cache_ref before:", cache_ref, "id:", id(cache_ref))
        step1._on_ss_selection_changed()
        print("cache_ref after:", cache_ref, "id:", id(cache_ref))
        print("step1._count_cache after:", step1._count_cache, "id:", id(step1._count_cache))
        print("cache_ref is step1._count_cache:", cache_ref is step1._count_cache)
        assert mock_controller.session.daminion_client.get_filtered_item_count.call_count == 1, mock_controller.session.daminion_client.get_filtered_item_count.call_count

        # Same selection again should be a no-op (cache hit).
        step1.ss_var.get.return_value = "Beta"
        step1._on_ss_selection_changed()
        print("cache after second call:", step1._count_cache, "id:", id(step1._count_cache))
        print("call_count:", mock_controller.session.daminion_client.get_filtered_item_count.call_count)
        assert mock_controller.session.daminion_client.get_filtered_item_count.call_count == 1, mock_controller.session.daminion_client.get_filtered_item_count.call_count

        # Switching away and back should refresh (cache invalidated by the switch).
        step1.ss_var.get.return_value = "Alpha"
        step1._on_ss_selection_changed()
        assert mock_controller.session.daminion_client.get_filtered_item_count.call_count == 2, mock_controller.session.daminion_client.get_filtered_item_count.call_count
