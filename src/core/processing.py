"""
Processing Pipeline Module
===========================

This module implements the core processing pipeline that orchestrates the entire
image tagging workflow. It manages the multi-threaded execution of AI model inference
and metadata writing operations.

Key Components:
- ProcessingManager: Main orchestrator class that runs in a background thread
- Item fetching: Retrieves images from local filesystem or Daminion
- Model initialization: Loads AI models for local inference
- Processing loop: Iterates through items, runs inference, writes metadata
- Progress tracking: Reports status to UI via callbacks

Threading Model:
- Main thread: UI event loop
- Background thread: Processing pipeline (created by ProcessingManager.start())
- The background thread can be interrupted via stop_event

Workflow Stages:
1. Fetch items (local folder scan or Daminion query)
2. Initialize model (if using local inference)
3. Process each item:
   a. Load image
   b. Run AI inference
   c. Extract tags from results
   d. Write metadata (EXIF/IPTC or Daminion)
   e. Verify metadata (optional)
4. Update statistics and progress

Author: Dean
"""

import gc
import logging
import threading
import time
from concurrent.futures import as_completed

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False
from pathlib import Path
from typing import Callable
from PIL import Image

# Internal modules
from .session import Session
from . import huggingface_utils
from . import image_processing
from . import config
from src.utils.concurrency import DaemonThreadPoolExecutor
from . import keyword_scoring
from . import keyword_scoring_adapters

# Optional metadata verification (for testing/debugging)
# This module may not be available in packaged distributions
try:
    import tests.verify_metadata as verifier
except ImportError:
    # Fallback if tests is not in path (e.g. when packaged)
    verifier = None


# ============================================================================
# PROCESSING MANAGER
# ============================================================================


class ProcessingManager:
    """
    Main processing orchestrator that runs the AI tagging pipeline.

    This class manages the entire processing workflow in a background thread,
    allowing the UI to remain responsive. It coordinates between:
    - Data source (local files or Daminion)
    - AI engine (local LFM models)
    - Metadata writing (EXIF/IPTC or Daminion API)

    The processing runs asynchronously and can be aborted by the user at any time.
    Progress and log messages are sent to the UI via callback functions.

    Attributes:
        session: Session object containing all configuration and state
        log: Callback function for sending log messages to UI
        progress: Callback function for updating progress bar (percentage, current, total)
        stop_event: Threading event used to signal abortion
        thread: Background thread running the processing job
        logger: Python logger for file-based logging
        model: Loaded AI model (only for local inference)

    Example:
        >>> manager = ProcessingManager(session, log_callback, progress_callback)
        >>> manager.start()  # Starts background thread
        >>> # ... user can abort ...
        >>> manager.abort()  # Signals thread to stop
    """

    def __init__(
        self,
        session: Session,
        log_callback: Callable[[str], None],
        progress_callback: Callable[[float, int, int], None],
        auto_paginate: bool = False,
    ):
        """
        Initialize the processing manager.

        Args:
            session: Session object with datasource and engine configuration
            log_callback: Function to call with log messages for UI display
            progress_callback: Function to call with progress updates (percentage, current, total)
            auto_paginate: When True the Daminion fetch loop repeats in 500-record
                           pages until the server has no more items to return.
        """
        self.session = session
        self.log = log_callback  # UI log callback
        self.progress = progress_callback  # UI progress callback
        self.stop_event = threading.Event()  # Signal for aborting
        self._start_time = None  # Job start time for ETA calculation
        self.thread = None  # Background processing thread
        self.logger = logging.getLogger(__name__)  # File logger
        self.auto_paginate = (
            auto_paginate  # Whether to page through all 500-record batches
        )
        # Guards session counters mutated from parallel worker threads.
        self._stats_lock = threading.Lock()
        # Model cache to avoid reloading the same model multiple times
        self._model_cache = {}  # (model_id, task, device) -> model


    def start(self):
        """
        Start the processing job in a background thread.

        This method creates and starts a daemon thread that runs the entire
        processing pipeline. The thread will automatically terminate when the
        main program exits.

        The processing workflow is:
        1. Reset statistics
        2. Fetch items from datasource
        3. Initialize model (if local)
        4. Process each item
        5. Report completion
        """
        self.logger.info("Starting processing job")
        self.logger.info(
            f"Datasource: {self.session.datasource.type}, Engine: {self.session.engine.provider}"
        )
        self.logger.info(
            f"Model: {self.session.engine.model_id}, Task: {self.session.engine.task}"
        )

        # Clear any previous abort signal
        self.stop_event.clear()
        self._start_time = None  # Will be set on first progress update

        # Create and start background thread
        # daemon=True ensures thread terminates when main program exits
        self.thread = threading.Thread(target=self._run_job, daemon=True)
        self.thread.start()

    def _emit_progress(
        self,
        pct: float,
        current: int,
        total: int,
        *,
        more_pages: bool,
        elapsed_seconds: float,
        etc_seconds: float,
    ) -> None:
        """
        Emit progress updates with backward compatibility for older callbacks.
        """
        try:
            self.progress(
                pct,
                current,
                total,
                more_pages=more_pages,
                elapsed_seconds=elapsed_seconds,
                etc_seconds=etc_seconds,
            )
            return
        except TypeError:
            pass

        try:
            self.progress(pct, current, total, more_pages)
        except TypeError:
            self.progress(pct, current, total)

    def abort(self):
        """
        Request abortion of the current processing job.

        This method sets a flag that the background thread checks between
        each item. The thread will stop processing new items but will
        complete the current item before exiting.

        Note: This is a graceful shutdown - the current item will finish processing.
        """
        if self.stop_event.is_set():
            return

        self.logger.warning("Processing job abort requested")
        self.stop_event.set()  # Signal the background thread to stop
        self.log("Stopping job... please wait.")

    def shutdown(self, timeout=2.0):
        """
        Ensure the processing manager shuts down completely.
        Called during application exit.

        Args:
            timeout: Maximum time to wait for the thread to join
        """
        if self.thread and self.thread.is_alive():
            self.logger.info("ProcessingManager shutdown initiated")
            self.abort()
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                self.logger.warning(
                    f"Processing thread did not terminate within {timeout}s - proceeding anyway"
                )

    def _run_job(self):
        """
        Main processing loop (runs in background thread).

        Orchestrates the entire workflow:
        1. Fetch a page of items from the datasource
        2. Initialize the AI model (local only, once)
        3. Process each item in the page
        4. If auto_paginate is enabled and the page was full (500 items),
           fetch the next page and repeat until exhausted
        5. Cleanup on completion
        """
        DAMINION_PAGE_SIZE = 500  # Hard limit imposed by Daminion API

        try:
            self.log("Job started.")
            self.session.reset_stats()  # Clear previous run statistics
            self.session.is_processing = True
            process_limit = (
                self.session.datasource.max_items
                if self.session.datasource.type == "daminion"
                and self.session.datasource.max_items > 0
                else None
            )

            # ================================================================
            # STAGE 1: INITIALIZE MODEL — done once before loop
            # ================================================================
            engine = self.session.engine

            if engine.provider != "local":
                # Synapic tags images with local models (LFM) only.
                self.logger.warning(
                    f"Engine provider '{engine.provider}' is not supported; "
                    "forcing local inference."
                )
                engine.provider = "local"
                engine.model_id = ""

            self._init_local_model()

            # ================================================================
            # STAGE 2: PAGINATED FETCH + PROCESS LOOP
            # ================================================================
            # For Daminion sources the API caps every response at 500 records.
            # Two pagination strategies are available:
            #
            # 1. RELOAD-SEARCH (offset always 0):
            #    Used when untagged filters are active. After each batch is
            #    processed, items are excluded by the server-side untagged
            #    filter, so re-fetching from offset 0 naturally returns the
            #    next set of untagged records without re-processing.
            #
            # 2. OFFSET-BASED (advancing offset):
            #    Used when no untagged filters are active (keyword search,
            #    saved search, collection, or global scan without untagged).
            #    The result set is stable (items don't disappear when tagged),
            #    so we advance the start_index by the number of items received
            #    to walk through the full set.
            #
            # For local sources offset is ignored and only one pass is made.
            page_num = 0
            grand_total_processed = 0
            last_page_ids: set = set()  # Guard against infinite loops
            items = None  # Current page of items (None until the first fetch)

            # ================================================================
            # PRE-FLIGHT COUNT — log the server-side total before fetching
            # ================================================================
            # For Daminion sources: ask the server how many records match the
            # current filters so we can confirm every page is retrieved.
            ds = self.session.datasource
            expected_total = 0  # Default for non-Daminion sources
            if process_limit is not None:
                self.log(f"Process limit active: up to {process_limit} item(s).")

            # Build untagged_fields list early — needed both for pre-flight
            # count AND for determining the pagination strategy below.
            untagged_fields = []
            if ds.daminion_untagged_keywords:
                untagged_fields.append("Keywords")
            if ds.daminion_untagged_categories:
                untagged_fields.append("Category")
            if ds.daminion_untagged_description:
                untagged_fields.append("Description")

            # Determine which pagination strategy to use.
            # Reload-search relies on the server-side untagged filter to
            # exclude processed items from subsequent fetches. Without it,
            # the same items are returned every time and the infinite-loop
            # guard would stop after ~500 items.
            use_reload_search = bool(untagged_fields)
            current_offset = 0

            if ds.type == "daminion" and self.session.daminion_client:
                try:
                    raw_expected_total = (
                        self.session.daminion_client.get_filtered_item_count(
                            scope=ds.daminion_scope,
                            saved_search_id=ds.daminion_saved_search_id
                            or ds.daminion_saved_search,
                            collection_id=ds.daminion_collection_id
                            or ds.daminion_catalog_id,
                            search_term=ds.daminion_search_term,
                            untagged_fields=untagged_fields,
                            status_filter=ds.status_filter,
                            force_refresh=True,
                        )
                    )
                    if isinstance(raw_expected_total, (int, float)):
                        expected_total = int(raw_expected_total)
                    else:
                        self.logger.warning(
                            "Pre-flight count returned non-numeric value (%r); "
                            "falling back to dynamic totals.",
                            raw_expected_total,
                        )
                        expected_total = 0
                    self.logger.info(
                        f"PRE-FLIGHT COUNT: server reports {expected_total} record(s) "
                        f"matching current filters (scope={ds.daminion_scope})"
                    )
                    if process_limit is not None:
                        expected_total = min(expected_total, process_limit)
                    self.log(
                        f"Server record count: {expected_total} item(s) "
                        f"matching filters before processing starts."
                    )
                except Exception as e:
                    self.logger.warning(f"Pre-flight count failed (non-fatal): {e}")

            while True:
                if (
                    process_limit is not None
                    and self.session.processed_items >= process_limit
                ):
                    self.log(
                        f"Process limit reached ({process_limit} item(s)) - stopping."
                    )
                    break

                page_num += 1
                if page_num == 1:
                    self.log("Fetching items...")
                else:
                    self.log("Reloading search for next batch...")

                # ============================================================
                # FETCH ONE PAGE
                #   reload-search (offset=0)  – when untagged filter is active
                #   offset-based               – when dataset is stable
                # ============================================================
                items = self._fetch_items(offset=current_offset)

                if not items:
                    if page_num == 1:
                        self.log("No items found to process.")
                    else:
                        self.log("No more items — all pages processed.")
                    break

                page_count = len(items)

                # ── Infinite-loop guard ──────────────────────────────────────
                # If the server returns the same IDs as the previous batch
                # (e.g. the untagged filter is not applied server-side), stop
                # rather than re-processing the same records forever.
                current_ids = {
                    item.get("id") if isinstance(item, dict) else str(item)
                    for item in items
                }
                if current_ids and current_ids == last_page_ids:
                    self.logger.warning(
                        "Reload-search returned identical items as the previous batch — "
                        "stopping to avoid infinite loop. "
                        "The server-side filter may not be filtering tagged items."
                    )
                    self.log(
                        "Warning: same items returned after reload — pagination stopped. "
                        "Check that the untagged filter is applied server-side."
                    )
                    del items
                    break
                last_page_ids = current_ids
                # ─────────────────────────────────────────────────────────────

                self.session.total_items += page_count
                strat = "reload-search" if use_reload_search else "offset-based"
                self.logger.info(
                    f"Page {page_num}: {page_count} items fetched "
                    f"({strat}, offset={current_offset}, auto_paginate={self.auto_paginate})"
                )
                self.log(f"Page {page_num}: {page_count} item(s) to process.")

                # Reset per-page counter; overall progress uses session counters
                # more_pages=True: this is a page boundary, job is not done yet
                # Record start time on first progress call for ETA calculation
                if self._start_time is None:
                    self._start_time = time.monotonic()
                elapsed = time.monotonic() - self._start_time
                processed = self.session.processed_items
                # Use expected_total for ETA when auto-pagination is enabled
                if self.auto_paginate and expected_total > 0:
                    effective_total = expected_total
                else:
                    effective_total = self.session.total_items
                if process_limit is not None:
                    effective_total = min(effective_total, process_limit)
                remaining = max(effective_total - processed, 0)
                etc = (elapsed / processed * remaining) if processed > 0 else 0
                self._emit_progress(
                    self.session.processed_items / max(effective_total, 1),
                    self.session.processed_items,
                    effective_total,
                    more_pages=True,
                    elapsed_seconds=elapsed,
                    etc_seconds=etc,
                )

                # ============================================================
                # PROCESS EACH ITEM IN THIS PAGE
                # ============================================================
                # Local GPU/CPU inference is sequential (max_workers=1) — the
                # model lives in the worker process and would thrash under
                # concurrent calls from multiple threads.
                max_workers = 1
                executor = (
                    DaemonThreadPoolExecutor(max_workers=max_workers)
                    if max_workers > 1
                    else None
                )
                try:
                    if executor is not None:
                        # Submit all items up front; as_completed yields results
                        # as each future finishes so progress doesn't stall on a
                        # slow early item. Workers skip remaining work via
                        # _process_item_guarded once the stop_event is set, so
                        # the final shutdown(wait=True) below drains quickly.
                        futures = [
                            executor.submit(self._process_item_guarded, item)
                            for item in items
                        ]
                        results_iter = as_completed(futures)
                    else:
                        results_iter = (
                            self._process_item_guarded(it) for it in items
                        )

                    for i, _ in enumerate(results_iter):
                        if self.stop_event.is_set():
                            self.logger.info(
                                f"Job aborted by user after processing "
                                f"{grand_total_processed} items total"
                            )
                            self.log("Job aborted by user.")
                            # items will be freed by the outer stop_event guard
                            # below; do NOT del here to avoid UnboundLocalError.
                            break

                        self.session.processed_items += 1
                        grand_total_processed += 1

                        # Periodic garbage collection (deterministic here, on
                        # the consuming thread) frees residual base64 strings
                        # and API response objects every 3 items.
                        if self.session.processed_items % 3 == 0:
                            gc.collect()

                        # Log memory consumption after each image for debugging
                        if _PSUTIL_AVAILABLE:
                            mem_mb = (
                                psutil.Process().memory_info().rss / (1024 * 1024)
                            )
                            self.logger.debug(
                                f"Memory usage after image "
                                f"{self.session.processed_items}/"
                                f"{self.session.total_items}: {mem_mb:.2f} MB"
                            )

                        # Determine whether more pages will follow this one.
                        # A page is "definitely the last" if:
                        #   - auto_paginate is off (never fetches more), OR
                        #   - this page is partial (< 500 items, server exhausted)
                        # Otherwise we conservatively keep more_pages=True even on
                        # the last item of a full page — the empty fetch that
                        # follows will simply exit the loop without emitting a
                        # misleading pct=1.0.
                        _is_last_page = (not self.auto_paginate) or (
                            page_count < DAMINION_PAGE_SIZE
                        )
                        _last_item_on_page = i == page_count - 1
                        _job_truly_done = _is_last_page and _last_item_on_page
                        elapsed = (
                            time.monotonic() - self._start_time
                            if self._start_time
                            else 0
                        )
                        processed = self.session.processed_items
                        if self.auto_paginate and expected_total > 0:
                            effective_total = expected_total
                        else:
                            effective_total = self.session.total_items
                        if process_limit is not None:
                            effective_total = min(effective_total, process_limit)
                        remaining = max(effective_total - processed, 0)
                        etc = (
                            (elapsed / processed * remaining) if processed > 0 else 0
                        )
                        self._emit_progress(
                            self.session.processed_items / max(effective_total, 1),
                            self.session.processed_items,
                            effective_total,
                            more_pages=not _job_truly_done,
                            elapsed_seconds=elapsed,
                            etc_seconds=etc,
                        )

                        if (
                            process_limit is not None
                            and self.session.processed_items >= process_limit
                        ):
                            self.logger.info(
                                f"Processing stopped at configured limit of "
                                f"{process_limit} items"
                            )
                            break
                finally:
                    if executor is not None:
                        # wait=True is safe on abort too: queued items skip work
                        # via _process_item_guarded, so draining is fast and no
                        # worker is left running during the cleanup below.
                        executor.shutdown(wait=True)

                # Stop pagination if abort was requested
                if self.stop_event.is_set():
                    if items is not None:
                        del items
                    break

                # Stop if auto-pagination is off OR this was a partial page
                # (partial page = server has no more untagged records)
                if not self.auto_paginate or page_count < DAMINION_PAGE_SIZE:
                    if self.auto_paginate and page_count < DAMINION_PAGE_SIZE:
                        self.log(
                            f"Last batch received ({page_count} items) — "
                            "all items processed."
                        )
                    del items
                    break

                # Advance the pagination offset for the next fetch.
                # - reload-search: reset to 0 (untagged filter excludes
                #   already-tagged items server-side)
                # - offset-based: advance by received count to walk through
                #   the stable result set
                if use_reload_search:
                    current_offset = 0
                else:
                    current_offset += page_count

                del items  # Free before fetching next batch

            # ================================================================
            # STAGE 3: FORCE-REFRESH DAMINION SEARCH CACHE
            # ================================================================
            # After tagging completes, Daminion's search index may not immediately
            # reflect the changes. A force-refreshed GetCount call tells the server
            # to re-evaluate the search results, making newly-tagged items properly
            # excluded from untagged filters.
            if ds.type == "daminion" and self.session.daminion_client:
                try:
                    self.log("Refreshing server search cache...")
                    remaining_count = (
                        self.session.daminion_client.get_filtered_item_count(
                            scope=ds.daminion_scope,
                            saved_search_id=ds.daminion_saved_search_id
                            or ds.daminion_saved_search,
                            collection_id=ds.daminion_collection_id
                            or ds.daminion_catalog_id,
                            search_term=ds.daminion_search_term,
                            untagged_fields=untagged_fields,
                            status_filter=ds.status_filter,
                            force_refresh=True,
                        )
                    )
                    completed = self.session.processed_items
                    self.logger.info(
                        f"Post-job cache refresh: server now reports "
                        f"{remaining_count} item(s) remaining (was ~{expected_total}). "
                        f"{completed} item(s) tagged in this session."
                    )
                    if isinstance(remaining_count, (int, float)):
                        if remaining_count >= 0:
                            self.log(
                                f"Server search cache refreshed: "
                                f"{int(remaining_count)} untagged item(s) remaining."
                            )
                        else:
                            self.log(
                                "Server search cache refreshed, but count is "
                                "unavailable. Tags were applied successfully."
                            )
                except Exception as e:
                    self.logger.warning(
                        f"Post-job cache refresh failed (non-fatal): {e}"
                    )

            # ================================================================
            # STAGE 4: COMPLETION & CLEANUP
            # ================================================================
            self.logger.debug("Hit end of processing loop")
            self.logger.info(
                f"Processing job completed — Processed: {self.session.processed_items}, "
                f"Failed: {self.session.failed_items}, Pages: {page_num}"
            )
            self.log(
                f"Job finished. "
                f"Processed {self.session.processed_items} item(s) across {page_num} page(s)."
            )

            # Explicitly unload model to free memory/VRAM
            if hasattr(self, "model") and self.model:
                self.logger.info("Unloading local model and performing memory cleanup")
                self.log(
                    "Cleaning up: unloading model from memory (this may take a moment)..."
                )
                self.model = None  # Release reference

                # Clear CUDA cache if GPU was used
                if self.session.engine.device == "cuda":
                    try:
                        import torch

                        if torch.cuda.is_available():
                            self.log("Cleaning up: flushing GPU VRAM cache...")
                            torch.cuda.empty_cache()
                            self.logger.info("CUDA cache cleared")
                    except ImportError:
                        pass

            # Force garbage collection after all cleanup
            gc.collect()
            self.log("Memory cleanup completed.")

        except Exception as e:
            # Catch any unexpected errors in the processing pipeline
            self.logger.exception("Processing job failed with exception")
            logging.exception("Processing failed")
            self.log(f"Error: {e}")
            self.session.failed_items += 1

            # Ensure cleanup even on failure
            if hasattr(self, "model") and self.model:
                self.model = None
            gc.collect()
        finally:
            self.session.is_processing = False

    def _fetch_items(self, offset: int = 0):
        """
        Fetch items to process from the configured datasource.

        For Daminion sources the ``offset`` parameter controls which page of
        500 records is returned.  Pass ``offset=0`` for the first page,
        ``offset=500`` for the second, etc.  For local filesystem sources
        this parameter is ignored — all matching files are returned in one
        call.

        Args:
            offset: Starting index for Daminion pagination (default 0).

        Returns:
            list: Items to process.  For local sources: list of Path objects.
                  For Daminion sources: list of item dicts.

        Raises:
            FileNotFoundError: If local path doesn't exist.
            ValueError: If Daminion client is not connected.
        """
        ds = self.session.datasource

        # ================================================================
        # LOCAL FILESYSTEM SOURCE
        # ================================================================
        if ds.type == "local":
            path = Path(ds.local_path)
            if not path.exists():
                raise FileNotFoundError(f"Folder not found: {path}")

            # Define supported image file extensions
            exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

            # Scan directory (recursive or shallow)
            if ds.local_recursive:
                self.logger.info(f"Performing recursive scan of {path}")
                # rglob("*") recursively finds all files in subdirectories
                files = [p for p in path.rglob("*") if p.suffix.lower() in exts]
            else:
                self.logger.info(f"Performing shallow scan of {path}")
                # iterdir() only scans the immediate directory
                files = [p for p in path.iterdir() if p.suffix.lower() in exts]

            self.logger.info(
                f"Found {len(files)} image files in local folder: {path} (recursive={ds.local_recursive})"
            )
            return files

        # ================================================================
        # DAMINION DAM SOURCE
        # ================================================================
        elif ds.type == "daminion":
            # Ensure Daminion client is connected
            if not self.session.daminion_client:
                raise ValueError("Daminion client not connected")

            self.logger.info(
                f"Fetching items from Daminion — Scope: {ds.daminion_scope}, "
                f"Status: {ds.status_filter}, Offset: {offset}"
            )
            self.log("Fetching items from Daminion...")

            # Build list of fields to filter for untagged items
            untagged_fields = []
            if ds.daminion_untagged_keywords:
                untagged_fields.append("Keywords")
            if ds.daminion_untagged_categories:
                untagged_fields.append("Category")
            if ds.daminion_untagged_description:
                untagged_fields.append("Description")

            # Determine maximum items to fetch (0 = unlimited). For limited runs,
            # only fetch the remaining quota so reload-pagination cannot exceed it.
            if ds.max_items > 0:
                remaining_limit = ds.max_items - self.session.processed_items
                if remaining_limit <= 0:
                    self.logger.info(
                        "Process limit already satisfied; no more items fetched."
                    )
                    return []
                max_to_fetch = remaining_limit
            else:
                max_to_fetch = None

            # Query Daminion. When offset > 0 we are in a pagination pass
            # and only want the single next page of 500 records.
            items = self.session.daminion_client.get_items_filtered(
                scope=ds.daminion_scope,
                saved_search_id=ds.daminion_saved_search_id or ds.daminion_saved_search,
                collection_id=ds.daminion_collection_id or ds.daminion_catalog_id,
                search_term=ds.daminion_search_term,
                untagged_fields=untagged_fields,
                status_filter=ds.status_filter,
                max_items=max_to_fetch,
                start_index=offset,
            )

            self.logger.info(
                f"Retrieved {len(items)} items from Daminion (offset={offset})"
            )
            self.log(f"Retrieved {len(items)} items from Daminion.")
            return items

        # Unknown datasource type
        return []

    def _init_local_model(self):
        """
        Initialize and load the AI model for local inference.

        This method is only called when using local inference (not API-based).
        It loads the model from Hugging Face's cache into memory and prepares
        it for inference on the selected device (CPU or GPU).

        The method:
        1. Checks model compatibility (rejects GPTQ, AWQ, etc.)
        2. Converts device string ('cpu'/'cuda') to integer format for pipeline
        3. Loads the model using huggingface_utils (with caching)
        4. Auto-detects and corrects the task if needed
        5. Stores the model in self.model for reuse across all items

        Device mapping:
        - 'cpu' -> -1 (use CPU for inference)
        - 'cuda' -> 0 (use GPU device 0 for inference)

        Raises:
            RuntimeError: If model loading fails or model is incompatible

        Note:
            The model is loaded once and reused for all items in the batch,
            which is much more efficient than loading per-item.
        """
        engine = self.session.engine

        # Check model compatibility before attempting to load
        reason = huggingface_utils.get_local_inference_incompatibility_reason(
            engine.model_id,
            task=engine.task,
        )
        if reason is not None:
            error_msg = (
                f"Cannot load model '{engine.model_id}': {reason}\n\n"
                "This model is not suitable for Synapic's local inference runtime.\n"
                "Please select a different model from the local cache."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Create cache key for model lookup
        cache_key = (engine.model_id, engine.task, engine.device)

        # Check if model is already cached
        if cache_key in self._model_cache:
            self.logger.info(f"Using cached model: {engine.model_id}")
            self.model = self._model_cache[cache_key]
            self.log(f"Using cached model: {engine.model_id}...")
            return

        self.logger.info(f"Initializing local model: {engine.model_id}")
        self.log(f"Loading local model: {engine.model_id}...")

        # Convert device string to integer format expected by transformers pipeline
        # -1 = CPU, 0 = CUDA device 0 (first GPU)
        device_int = -1 if engine.device == "cpu" else 0
        self.logger.info(f"Using device: {engine.device} (device_int={device_int})")

        try:
            # Load model from Hugging Face cache
            # This may download the model if not already cached
            self.model = huggingface_utils.load_model(
                model_id=engine.model_id,
                task=engine.task,
                progress_queue=None,  # No progress tracking for batch load
                device=device_int,
            )

            # Auto-detect actual task from loaded model
            # Some models may have a different task than configured
            # (e.g., VLMs use 'image-text-to-text' instead of 'image-to-text')
            actual_task = getattr(self.model, "task", None)
            if actual_task and actual_task != engine.task:
                self.logger.info(
                    f"Syncing session task from '{engine.task}' to actual pipeline task '{actual_task}'"
                )
                engine.task = actual_task

            self.logger.info(
                f"Local model loaded successfully: {engine.model_id} (Task: {engine.task}, Device: {engine.device})"
            )
            self.log(
                f"Model loaded successfully (Task: {engine.task}, Device: {engine.device})."
            )

            # Cache the model for future use
            self._model_cache[cache_key] = self.model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _process_item_guarded(self, item):
        """
        Process one item, skipping immediately once abort is requested.

        Workers pick items off the executor queue even after the main loop has
        broken out on abort; this guard keeps post-abort work bounded to the
        items already in flight (at most ``max_workers``).
        """
        if self.stop_event.is_set():
            return
        self._process_single_item(item)

    def _process_single_item(self, item):
        """
        Process a single image item through the complete AI tagging pipeline.

        This method orchestrates the four-stage processing workflow:
        1. **Image Loading**: Load from local file or download Daminion thumbnail
        2. **AI Inference**: Run the image through the configured AI model
        3. **Tag Extraction**: Parse model output and filter by confidence threshold
        4. **Metadata Writing**: Write tags to EXIF/IPTC or update Daminion

        Args:
            item: Either a Path object (local file) or dict (Daminion item with 'id', 'fileName')

        Processing Flow:
            - Detects item type (local vs Daminion) and loads image accordingly
            - Routes to appropriate inference method (local model vs API)
            - Handles different model types (VLM, captioning, classification, zero-shot)
            - Applies confidence threshold filtering to extracted tags
            - Writes metadata to destination (file or Daminion)
            - Optionally verifies Daminion metadata updates
            - Updates session statistics and results

        Error Handling:
            - Logs detailed error information for debugging
            - Increments failed_items counter
            - Continues processing remaining items (doesn't abort job)
            - Cleans up temporary files even on failure

        Note:
            For Daminion items, thumbnails are downloaded to temp files and
            cleaned up after processing to avoid disk space issues.
        """
        path = None
        is_daminion = isinstance(
            item, dict
        )  # Daminion items are dicts, local items are Path objects
        daminion_client = self.session.daminion_client
        temp_thumb = None  # Track temporary thumbnail file for cleanup

        try:
            engine = self.session.engine

            # ===============================================================
            # STAGE 1: IMAGE LOADING
            # ===============================================================
            # Load the image from either local filesystem or Daminion server
            if is_daminion:
                item_id = item.get("id")
                filename = item.get("fileName") or f"Item {item_id}"
                self.logger.debug(f"Processing Daminion item {item_id}: {filename}")
                self.log(f"Processing Daminion Item: {filename}...")

                # Download image (server-side resized for faster AI inference)
                # Use original at 100%, proportionally scaled preview at lower scales,
                # or a fixed 200px thumbnail when override is enabled
                ds = self.session.datasource
                if getattr(ds, "use_thumbnail_override", False):
                    # Fixed 200px thumbnail — fast, consistent, minimal bandwidth
                    path = daminion_client.download_thumbnail(
                        item_id, width=200, height=200
                    )
                    temp_thumb = path  # Cleaned up in the finally block below
                    if not path or not path.exists():
                        raise RuntimeError(
                            f"Could not download thumbnail for item {item_id}"
                        )
                else:
                    scale = getattr(ds, "resize_scale", 100)
                    if scale >= 100:
                        path = daminion_client.download_original(item_id)
                        temp_thumb = path  # Cleaned up in the finally block below
                        if not path or not path.exists():
                            raise RuntimeError(
                                f"Could not download original for item {item_id}"
                            )
                    else:
                        # Fetch dimensions once and compute both target width and
                        # height so download_preview() doesn't re-fetch them.
                        dims = daminion_client.get_item_dimensions(item_id)
                        if dims:
                            orig_w, orig_h = dims
                            target_w = max(75, int(orig_w * scale / 100))
                            target_h = max(75, int(orig_h * scale / 100))
                        else:
                            # Fallback: use scale of a base 2000px size
                            target_w = max(75, int(2000 * scale / 100))
                            target_h = None
                        path = daminion_client.download_preview(
                            item_id, width=target_w, height=target_h
                        )
                        temp_thumb = path  # Cleaned up in the finally block below
                        if not path or not path.exists():
                            raise RuntimeError(
                                f"Could not download preview for item {item_id}"
                            )
            else:
                path = item
                self.logger.debug(f"Processing local file: {path}")
                self.log(f"Processing: {path.name}...")

            # ===============================================================
            # STAGE 2: AI INFERENCE
            # ===============================================================
            # Run the image through the locally loaded AI model (LFM-only).

            # Tagging mode: 'llm', 'probability', or 'both'.
            # Legacy configs only persist probability_enabled -> map True to 'both'.
            mode = str(getattr(engine, "probability_mode", "") or "").lower()
            if mode not in ("llm", "probability", "both"):
                mode = "both" if getattr(engine, "probability_enabled", False) else "llm"
            elif mode == "llm" and getattr(engine, "probability_enabled", False):
                # Legacy sessions enable probabilities without a mode field
                mode = "both"

            # Handle probability scoring for local models
            # score_result carries the tier-annotated contract; prob_dict is
            # the legacy post-threshold {candidate: score} map kept for the
            # results export (and existing tests) below.
            score_result = None
            prob_dict = {}
            if engine.provider == "local" and mode != "llm":
                try:
                    score_result = keyword_scoring_adapters.score_keywords(
                        engine,
                        str(path),
                        mode=mode,
                        local_pipeline=self.model,
                    )
                    prob_dict = score_result.score_map
                    if (
                        score_result.tier == keyword_scoring_adapters.SCORING_TIER.UNAVAILABLE
                        and all(score == 0.0 for score in prob_dict.values())
                    ):
                        # Scoring did not run (or failed): surface the reason and
                        # continue with an empty map so the legacy fallbacks apply.
                        for note in score_result.notes:
                            self.logger.warning(f"Probability scoring unavailable: {note}")
                        score_result = None
                        prob_dict = {}
                    else:
                        # Log each candidate score and whether it passes the
                        # threshold (requirement: display per-option probabilities
                        # in the log)
                        threshold = engine.probability_threshold
                        for candidate, score in prob_dict.items():
                            passed = threshold <= 0.0 or score >= threshold
                            self.log(
                                f"  {candidate}: {score:.3f} "
                                f"{'PASS' if passed else 'FAIL'}"
                            )
                        # Apply optional threshold filter
                        if threshold > 0.0:
                            prob_dict = {
                                k: v for k, v in prob_dict.items()
                                if v >= threshold
                            }
                            score_result = (
                                keyword_scoring.build_thresholded_view(
                                    score_result, threshold
                                )
                            )
                except Exception as exc:
                    # Requirement: hide on failure - continue with normal flow
                    self.logger.warning(
                        f"Local probability inference failed ({type(exc).__name__}): {exc}"
                    )
                    score_result = None
                    prob_dict = {}

            result = None

            probability_only = (
                mode == "probability" and engine.provider == "local"
            )

            if probability_only and prob_dict:
                # ---------------------------------------------------------------
                # PROBABILITY-ONLY TAGGING (No LLM inference)
                # ---------------------------------------------------------------
                # Tags are derived directly from the candidate probability scores:
                # the top-scoring candidate becomes the category and every
                # candidate that passed the threshold becomes a keyword.
                cat = max(prob_dict, key=prob_dict.get)
                kws = list(prob_dict.keys())
                desc = ""
                self.log(f"Probability tagging: category={cat}, keywords={kws}")
            elif probability_only and not prob_dict:
                # Probability scoring failed (e.g. non-classification model).
                # Fall through to LLM instead of producing empty tags.
                self.log(
                    "Probability scoring returned no results — "
                    "falling back to LLM tagging"
                )
                probability_only = False
                # Deliberately fall through to the LLM path below

            if not probability_only:
                # ---------------------------------------------------------------
                # LOCAL INFERENCE (Model loaded in memory)
                # ---------------------------------------------------------------
                # The model was loaded in _init_local_model() and is reused
                # for all items in the batch for efficiency

                if engine.task in [
                    config.MODEL_TASK_IMAGE_TO_TEXT,
                    "image-text-to-text",
                ]:
                    # Image Captioning / Vision-Language Models (VLMs)
                    # Handles both standard captioning (BLIP, GIT) and modern VLMs (Qwen2-VL)
                    with Image.open(path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")

                        # Check if the pipeline is modern image-text-to-text (e.g. Qwen2-VL)
                        # These models expect chat-style messages with structured prompts
                        if (
                            hasattr(self.model, "task")
                            and self.model.task == "image-text-to-text"
                        ):
                            # Build the text instruction (include system prompt if set)
                            system_instruction = (
                                engine.system_prompt.strip()
                                if engine.system_prompt
                                else ""
                            )
                            user_text = (
                                "Analyze the image and return a JSON object with keys: "
                                "'description' (detailed caption), 'category' (single broad category), "
                                "and 'keywords' (list of 5-10 tags). Return ONLY the raw JSON string."
                            )
                            if system_instruction:
                                messages = [
                                    {"role": "system", "content": system_instruction},
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "image", "image": img},
                                            {"type": "text", "text": user_text},
                                        ],
                                    },
                                ]
                            else:
                                messages = [
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "image", "image": img},
                                            {"type": "text", "text": user_text},
                                        ],
                                    }
                                ]
                            try:
                                # For image-text-to-text pipelines, pass the formatted messages
                                result = self.model(
                                    text=messages,
                                    generate_kwargs={"max_new_tokens": 512},
                                )
                            except Exception as e:
                                self.logger.error(f"VLM inference failed: {e}")
                                raise
                        else:
                            # Standard image-to-text models (BLIP, GIT, etc.)
                            try:
                                result = self.model(
                                    img,
                                    prompt="Describe the image.",
                                    generate_kwargs={"max_new_tokens": 512},
                                )
                            except Exception as e:
                                self.logger.debug(
                                    f"Prompted inference failed ({e}), falling back to simple call."
                                )
                                result = self.model(img)

                elif engine.task == config.MODEL_TASK_ZERO_SHOT:
                    # Zero-Shot Image Classification
                    # Classifies image into one of the provided candidate labels
                    # without requiring training on those specific categories
                    with Image.open(path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        result = self.model(
                            img, candidate_labels=config.DEFAULT_CANDIDATE_LABELS
                        )

                else:
                    # Standard Image Classification
                    # Uses pre-trained categories from the model's training
                    with Image.open(path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        result = self.model(img)

            # ===============================================================
            # STAGE 3: TAG EXTRACTION
            # ===============================================================
            # Parse the model's output and extract structured metadata
            # The extraction logic handles different output formats:
            # - JSON objects (from VLMs)
            # - Classification results with scores
            # - Plain text descriptions

            # Convert threshold from UI scale (1-100) to model scale (0.0-1.0)
            # Tags with confidence scores below this threshold are filtered out
            threshold = engine.confidence_threshold / 100.0

            # Extract category, keywords, and description from model result
            # The extract_tags_with_semantics function handles:
            # - Parsing JSON from VLM responses
            # - Filtering classification results by threshold
            # - Extracting top predictions as keywords
            # - Adding semantic enhancement (if taxonomy available)
            if not probability_only:
                try:
                    cat, kws, desc, semantic_data = image_processing.extract_tags_with_semantics(
                        result, engine.task, threshold=threshold, taxonomy=None
                    )
                    self.logger.debug(f"Semantic data: {semantic_data}")
                except (AttributeError, ImportError):
                    # Fallback to original function if new one not available
                    cat, kws, desc, _probabilities = image_processing.extract_tags_from_result(
                        result, engine.task, threshold=threshold
                    )
            self.logger.debug(
                f"Extracted tags - Category: {cat}, Keywords: {len(kws)}, Description length: {len(desc) if desc else 0}"
            )

            # Free the (potentially large) model result now that tags are extracted
            del result

            # If extraction returned no useful data, write a placeholder so the item
            # is marked as processed and won't be reprocessed in subsequent runs
            if not cat and not kws and not desc:
                desc = "[AI: No Result]"
                self.logger.info(
                    f"No tags extracted for item, using placeholder: {desc}"
                )
                self.log("No results - marking with placeholder")

            # ===============================================================
            # STAGE 4: METADATA WRITING
            # ===============================================================
            # Write the extracted tags to the appropriate destination:
            # - Daminion: Update item metadata via API
            # - Local: Write to EXIF/IPTC metadata in image file

            if is_daminion:
                # Update Daminion item metadata via API
                # This sends the tags to the Daminion server for storage
                success = daminion_client.update_item_metadata(
                    item_id=item_id, category=cat, keywords=kws, description=desc
                )

                # Optional: Verify that the metadata was actually written
                # This is useful for debugging API issues or data corruption
                if success and verifier:
                    self.logger.info(
                        f"Verifying metadata for Daminion item {item_id}..."
                    )
                    verified = verifier.verify_metadata_update(
                        client=daminion_client,
                        item_id=item_id,
                        expected_cat=cat,
                        expected_kws=kws,
                        expected_desc=desc,
                    )
                    if verified:
                        self.logger.info(
                            f"Metadata verification successful for item {item_id}"
                        )
                        self.log("Verification: Passed")
                    else:
                        self.logger.warning(
                            f"Metadata verification failed for item {item_id}"
                        )
                        self.log("Verification: FAILED (Check details in log file)")
                        # We don't fail the whole item if verification fails,
                        # just log it as a warning for manual review

            else:
                # Write metadata to local image file (EXIF/IPTC)
                # This embeds the tags directly in the image file
                success = image_processing.write_metadata(
                    image_path=path, category=cat, keywords=kws, description=desc
                )

            # ===============================================================
            # RESULT TRACKING
            # ===============================================================
            # Log the processing result and add to session results
            status = "Success" if success else "Write Failed"
            tags_str = f"Cat: {cat}, Kws: {len(kws)}, Desc: {desc[:20]}..."
            self.logger.info(
                f"Item processed successfully - Status: {status}, Tags: {tags_str}"
            )
            self.log(f"Result: {tags_str}")

            # Store result for export/review in Step 4. The legacy
            # "probabilities" key (post-threshold score map) is preserved for
            # the export and existing consumers; "scoring" adds the tier-
            # annotated contract when a scoring pass ran.
            result_entry = {
                "filename": filename if is_daminion else path.name,
                "status": status,
                "tags": tags_str,
                "probabilities": prob_dict,
            }
            if score_result is not None:
                result_entry["scoring"] = score_result.to_plain_dict()
            self.session.results.append(result_entry)

        except Exception as e:
            # ===============================================================
            # ERROR HANDLING
            # ===============================================================
            # Log detailed error information for debugging
            # The job continues processing remaining items even if one fails
            name = (
                item.get("fileName")
                if is_daminion
                else (item.name if isinstance(item, Path) else str(item))
            )
            self.logger.error(
                f"Failed to process item '{name}': {type(e).__name__}: {str(e)}"
            )
            self.logger.exception("Full traceback:")
            logging.error(f"Failed to process {name}: {e}")

            # Update failure statistics (thread-safe under parallel workers)
            with self._stats_lock:
                self.session.failed_items += 1
            self.log(f"Failed: {e}")

        finally:
            # ===============================================================
            # CLEANUP
            # ===============================================================
            # Always clean up temporary files, even if processing failed
            # This prevents disk space issues when processing large batches
            if temp_thumb and temp_thumb.exists():
                try:
                    import os

                    os.remove(temp_thumb)
                    self.logger.debug(f"Cleaned up temporary thumbnail: {temp_thumb}")
                except Exception:
                    # Ignore cleanup errors - not critical
                    pass
            # Periodic garbage collection to free any residual base64 strings,
            # API response objects, and other short-lived allocations.
            # Every 10 items balances GC overhead with memory pressure.
            if hasattr(self, "session") and self.session.processed_items % 10 == 0:
                gc.collect()

                # Clear CUDA cache if GPU was used and we've processed a significant number of items
                if self.session.engine.device == "cuda" and self.session.processed_items % 50 == 0:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            self.logger.debug("CUDA cache cleared")
                    except ImportError:
                        pass

