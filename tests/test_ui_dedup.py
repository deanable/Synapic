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
from unittest.mock import MagicMock, patch, ANY
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
        if func: func()
        return "timer_id"

module_mock.CTkFrame = MockCTkFrame
sys.modules["customtkinter"] = module_mock
sys.modules["tkinter"] = MagicMock()
sys.modules["tkinter.messagebox"] = MagicMock()
if importlib.util.find_spec("PIL.ImageTk") is None:
    sys.modules["PIL.ImageTk"] = MagicMock()

# Import original classes
from src.ui.steps.step1_datasource import Step1Datasource
from src.ui.steps.step_dedup import StepDedup
import src.ui.steps.step1_datasource as step1_module

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
        
        # We need these methods
        self.winfo_exists = MagicMock(return_value=True)
        self.after = MagicMock(side_effect=lambda d, f: f())

class TestableStepDedup(StepDedup):
    """Minimal dedup step variant that exposes only logic under test."""
    def __init__(self, controller):
        self.controller = controller
        self.session = controller.session
        self.logger = MagicMock()
        
        self.threshold_var = MagicMock()
        self.threshold_var.get.return_value = 95.0
        self.algorithm_var = MagicMock()
        self.algorithm_var.get.return_value = "phash"
        
        self.progress_frame = MagicMock()
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
        
        with patch("threading.Thread") as MockThread, \
             patch("src.ui.steps.step_dedup.DaminionDedupProcessor") as MockProcessor:
            
            step_dedup._start_scan()
            
            MockProcessor.assert_called_with(
                mock_controller.session.daminion_client,
                similarity_threshold=95.0
            )
            step_dedup.progress_frame.grid.assert_called()
