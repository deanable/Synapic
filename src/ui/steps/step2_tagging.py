"""
Step 2: Tagging Engine Configuration UI (LFM)
==============================================

This module defines the UI for configuring the LFM (Local Foundation Models)
tagging engine. Synapic only runs image tagging with models downloaded to the
local Hugging Face cache — there are no cloud provider selections.

The UI keeps the existing global inference parameters (confidence threshold,
device selection) and embeds the local-model workflow directly:

- "+ Find & Download Models": browse the Hugging Face Hub and cache compatible
  models for offline tagging.
- Downloaded Models list: pick a cached model for local inference.

Key Components:
---------------
- LFM Engine Section: embedded local-model list with Hub download manager.
- Global Settings: confidence threshold slider and device toggle (CPU/CUDA).
- Download Manager: integrated downloader with real-time progress.

Author: Synapic Project
"""

import queue
import customtkinter as ctk
import logging
import tkinter.messagebox as messagebox
from src.utils.background_worker import BackgroundWorker
from src.utils.registry_config import load_ui_preferences, save_ui_preferences
from src.core import config

from .provider_tab_local import create_local_tab

logger = logging.getLogger(__name__)


class Step2Tagging(ctk.CTkFrame):
    """
    UI component for the second step of the tagging wizard (LFM / local-only).

    This frame coordinates local model selection and global inference settings,
    ensuring the 'EngineConfig' is fully populated before moving to the
    execution phase.

    Attributes:
        controller: The main App instance managing the wizard flow.
        session: Global application state and configuration.
    """
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.session = self.controller.session

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main container
        self.container = ctk.CTkScrollableFrame(self)
        self.container.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.container.grid_columnconfigure(0, weight=1)

        # Title
        title = ctk.CTkLabel(self.container, text="Step 2: Tagging Engine (LFM)", font=("Roboto", 24, "bold"))
        title.grid(row=0, column=0, pady=(20, 20))

        # LFM engine section — the local provider tab IS the engine config.
        # Synapic tags images locally only: no provider radio groups.
        self._worker = BackgroundWorker(name="Step2Worker")
        self._load_registry_ui_preferences()
        self.local_tab = create_local_tab(
            self.container, self.session, self._worker,
            self._persist_image_filter_preference, self._filter_image_models
        )
        self.local_tab.grid(row=1, column=0, pady=5, sticky="ew")

        # === Model Info Section ===
        model_info_frame = ctk.CTkFrame(self.container, fg_color="#2B2B2B", corner_radius=10)
        model_info_frame.grid(row=2, column=0, pady=10, padx=40, sticky="ew")
        model_info_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            model_info_frame,
            text="Selected Model:",
            font=("Roboto", 12, "bold")
        ).grid(row=0, column=0, padx=15, pady=10, sticky="w")

        self.model_info_label = ctk.CTkLabel(
            model_info_frame,
            text=self._get_model_display_text(),
            font=("Roboto", 12),
            text_color="#2FA572",
            anchor="w"
        )
        self.model_info_label.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        # === Global Settings Section ===
        settings_frame = ctk.CTkFrame(self.container, fg_color="#2B2B2B", corner_radius=10)
        settings_frame.grid(row=3, column=0, pady=10, padx=40, sticky="ew")

        ctk.CTkLabel(
            settings_frame,
            text="Global Settings",
            font=("Roboto", 14, "bold")
        ).pack(pady=(15, 10))

        # Device Toggle
        device_container = ctk.CTkFrame(settings_frame, fg_color="transparent")
        device_container.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            device_container,
            text="Inference Device:",
            font=("Roboto", 12)
        ).pack(side="left", padx=(0, 10))

        self.device_var = ctk.StringVar(value=self.controller.session.engine.device)
        self.device_switch = ctk.CTkSegmentedButton(
            device_container,
            values=["cpu", "cuda"],
            variable=self.device_var,
            command=self.on_device_change,
            width=140
        )
        self.device_switch.pack(side="left")

        # Confidence Threshold
        threshold_label_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        threshold_label_frame.pack(fill="x", padx=20, pady=(10, 5))

        ctk.CTkLabel(
            threshold_label_frame,
            text="Confidence Threshold:",
            font=("Roboto", 12, "bold")
        ).pack(side="left", padx=(0, 5))

        self.threshold_value_label = ctk.CTkLabel(
            threshold_label_frame,
            text=f"{self.controller.session.engine.confidence_threshold}%",
            font=("Roboto", 12),
            text_color="#2FA572"
        )
        self.threshold_value_label.pack(side="left", padx=5)

        ctk.CTkLabel(
            threshold_label_frame,
            text="(Filters out low-probability matches)",
            font=("Roboto", 9),
            text_color="gray"
        ).pack(side="left", padx=10)

        # Slider with precision level labels
        slider_container = ctk.CTkFrame(settings_frame, fg_color="transparent")
        slider_container.pack(fill="x", padx=20, pady=(0, 15))

        ctk.CTkLabel(
            slider_container,
            text="Free",
            font=("Roboto", 10),
            text_color="gray"
        ).pack(side="left", padx=(0, 10))

        self.threshold_slider = ctk.CTkSlider(
            slider_container,
            from_=1,
            to=100,
            number_of_steps=99,
            command=self.on_threshold_change
        )
        self.threshold_slider.set(self.controller.session.engine.confidence_threshold)
        self.threshold_slider.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            slider_container,
            text="Strict",
            font=("Roboto", 10),
            text_color="gray"
        ).pack(side="left", padx=(10, 0))

        # Navigation Buttons
        nav_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        nav_frame.grid(row=4, column=0, pady=20, sticky="ew")

        ctk.CTkButton(nav_frame, text="Previous", command=lambda: self.controller.show_step("Step1Datasource"), width=150, fg_color="gray").pack(side="left", padx=20)
        ctk.CTkButton(nav_frame, text="Next Step", command=self.next_step, width=200, height=40).pack(side="right", padx=20)

    def _get_model_display_text(self):
        """Generate display text for selected model with capability info."""
        session = self.controller.session
        model_id = session.engine.model_id or "None"
        task = session.engine.task or "unknown"

        # Map task to capability description
        capability_map = {
            "image-classification": "Keywords",
            "zero-shot-image-classification": "Categories",
            "image-to-text": "Description",
            "image-text-to-text": "Multi-modal (Keywords, Categories, Description)"
        }

        capability = capability_map.get(task, "Unknown capability")

        if model_id == "None" or not model_id:
            return "No model selected"

        return f"{model_id} • {capability}"

    def on_threshold_change(self, value):
        """Update threshold value label and session when slider changes."""
        threshold_int = int(value)
        self.threshold_value_label.configure(text=f"{threshold_int}%")
        self.controller.session.engine.confidence_threshold = threshold_int

    def on_device_change(self, value):
        """Update session device setting when toggle changes."""
        self.controller.session.engine.device = value
        logger.debug(f"Device changed to: {value}")

    def update_model_info(self):
        """Update the model info label after configuration changes."""
        self.model_info_label.configure(text=self._get_model_display_text())

    def next_step(self):
        # Flush the local tab's UI state into the session before validation,
        # so the session reflects what the user just configured.
        try:
            self.local_tab.save_to_session()
        except Exception as e:
            logger.warning(f"Could not flush tab state to session: {e}")

        # Validate before proceeding
        is_valid, error_msg = self.controller.session.validate_workflow_state("Step3Process")
        if not is_valid:
            messagebox.showwarning("Validation Error", error_msg)
            return

        logger.debug(f"Selected Engine: {self.controller.session.engine.provider}")
        self.controller.show_step("Step3Process")

    def refresh_stats(self):
        """
        Synchronize the UI elements with the current Session state.

        Called by the App coordinator whenever the user navigates to this step
        to ensure all inputs accurately reflect the persisted configuration.
        """
        # Sync the local tab (model, probability controls) with the session
        if hasattr(self.local_tab, 'refresh'):
            self.local_tab.refresh()
        self.update_model_info()
        # Update device and threshold from session
        self.device_var.set(self.controller.session.engine.device)
        self.threshold_slider.set(self.controller.session.engine.confidence_threshold)
        self.threshold_value_label.configure(text=f"{self.controller.session.engine.confidence_threshold}%")

    def _schedule_ui_update(self, callback):
        """Schedule a callback on the UI thread only while the dialog exists.

        ``after()`` is the one Tkinter call that is safe to invoke from a
        background thread, so we always marshal through it and check widget
        existence inside the callback (which runs on the main thread).
        """
        try:
            self.after(0, lambda: callback() if self.winfo_exists() else None)
        except Exception:
            pass  # Widget already destroyed; drop the update.

    def _load_registry_ui_preferences(self):
        """Overlay UI preference defaults from the Windows Registry."""
        try:
            prefs = load_ui_preferences()
        except Exception:
            prefs = {}

        for key, value in prefs.items():
            if hasattr(self.session.engine, key):
                setattr(self.session.engine, key, bool(value))

    def _persist_image_filter_preference(self, attr_name: str, value: bool):
        """Persist a single model-filter checkbox state to session + registry."""
        setattr(self.session.engine, attr_name, bool(value))
        save_ui_preferences({attr_name: bool(value)})

    def _model_supports_image(self, model) -> bool:
        """Heuristic check for image-capable models."""
        markers = (
            "vision", "image", "visual", "multimodal", "multi-modal",
            "image-to-text", "image-text-to-text", "visual-question-answering",
            "llava", "phi-3-vision", "vl"
        )

        texts = []
        if isinstance(model, dict):
            texts.extend([
                model.get("id", ""),
                model.get("model_id", ""),
                model.get("capability", ""),
                model.get("task", ""),
                model.get("family", ""),
            ])
        else:
            texts.append(str(model))

        for text in texts:
            text_lower = str(text).lower()
            if any(marker in text_lower for marker in markers):
                return True
        return False

    def _filter_image_models(self, models, enabled: bool):
        """Return only image-capable models when the checkbox is enabled."""
        model_list = list(models or [])
        if not enabled:
            return model_list
        return [model for model in model_list if self._model_supports_image(model)]


class DownloadManagerDialog(ctk.CTkToplevel):
    """
    Dedicated modal for browsing and downloading models for local use.

    This dialog interfaces with both the Hugging Face Hub (search) and the
    local filesystem (model caching). It provides real-time download progress
    via a background worker.
    """

    def __init__(self, parent, session, local_tab=None):
        super().__init__(parent)
        self.parent = parent
        self.session = session
        self.local_tab = local_tab
        self._search_results_cache = []
        self._size_filter_min = 0
        self._size_filter_max = 0
        self._size_filter_active_max = 0
        self.search_filter_var = ctk.StringVar(value="multimodal")
        self.title("Download Models from Hugging Face Hub")
        self.geometry("800x600")

        # Background worker for thread management (single persistent thread)
        self._worker = BackgroundWorker(name="DownloadManagerWorker")

        # Make the dialog modal or at least ensuring it stays on top
        self.transient(parent)
        self.grab_set()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # Search header
        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.search_entry = ctk.CTkEntry(header, placeholder_text="Search multi-modal models (e.g. 'blip', 'vit', 'qwen')...", width=350)
        self.search_entry.pack(side="left", padx=5, fill="x", expand=True)
        self.search_entry.bind("<Return>", lambda e: self.start_search())

        ctk.CTkButton(header, text="Search Hub", command=self.start_search, width=120).pack(side="left", padx=5)

        filter_row = ctk.CTkFrame(self)
        filter_row.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
        ctk.CTkLabel(filter_row, text="Filter:").pack(side="left", padx=(8, 10), pady=8)
        for value, label in [
            ("keyword", "Keyword"),
            ("category", "Category"),
            ("description", "Description"),
            ("multimodal", "Multimodal"),
        ]:
            ctk.CTkRadioButton(
                filter_row,
                text=label,
                value=value,
                variable=self.search_filter_var,
                command=self._on_filter_changed,
            ).pack(side="left", padx=8, pady=8)

        size_filter = ctk.CTkFrame(self)
        size_filter.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 5))
        size_filter.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(size_filter, text="Max size:").grid(row=0, column=0, padx=(8, 10), pady=8, sticky="w")
        self.size_slider = ctk.CTkSlider(
            size_filter,
            from_=0,
            to=1,
            number_of_steps=100,
            command=self._on_size_slider_change,
        )
        self.size_slider.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=8)
        self.size_slider.set(1)
        self.size_slider.configure(state="disabled")

        self.size_label = ctk.CTkLabel(size_filter, text="No size data yet", width=220, anchor="e")
        self.size_label.grid(row=0, column=2, padx=(0, 8), pady=8, sticky="e")

        # Results area
        self.results_frame = ctk.CTkScrollableFrame(self, label_text="Hugging Face Hub Results")
        self.results_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)

        # Add a header label
        header_text = f"{'Model ID':<40} | {'Capability':^15} | {'Size':>10}"
        self.results_header = ctk.CTkLabel(
            self.results_frame,
            text=header_text,
            font=("Courier New", 12, "bold"),
            text_color="gray",
            anchor="w"
        )
        self.results_header.pack(fill="x", pady=(5, 10), padx=5)

        # Status footer
        self.footer = ctk.CTkFrame(self)
        self.footer.grid(row=4, column=0, sticky="ew", padx=10, pady=10)

        self.lbl_status = ctk.CTkLabel(self.footer, text="Enter a query and click Search", text_color="gray")
        self.lbl_status.pack(side="left", padx=5)

        self.progress = ctk.CTkProgressBar(self.footer)
        self.progress.pack(side="right", padx=10, fill="x", expand=True)
        self.progress.set(0)

        # Auto-fetch models for the default filter on dialog open.
        self._prefetch_per_page = 20
        self._prefetch_categories = [
            ("keyword", [config.MODEL_TASK_IMAGE_CLASSIFICATION]),
            ("category", [config.MODEL_TASK_ZERO_SHOT]),
            ("description", [config.MODEL_TASK_IMAGE_TO_TEXT]),
            ("multimodal", ["image-text-to-text"]),
        ]
        self._prefetch_all_results = []
        self.after(200, self._on_filter_changed)

    def _show_prefetch_results(self, results):
        """Display prefetched models and set up infinite scroll."""
        self._prefetch_all_results = results
        self._search_results_cache = list(results)
        self._configure_size_filter(results)
        filtered = self._get_size_filtered_results()
        self.lbl_status.configure(
            text=f"Showing {len(filtered)} popular models. Use Search for specific queries.",
            text_color="gray",
        )
        self.show_search_results(results)

    def _on_filter_changed(self):
        """Fetch top models for the newly selected filter category."""
        search_filter = self.search_filter_var.get()
        self.lbl_status.configure(text=f"Loading top {search_filter} models from Hub...", text_color="gray")
        self._search_results_cache = []
        self._reset_size_filter()
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self._worker.submit_replacing("prefetch", self._filter_worker, search_filter)

    def _filter_worker(self, category):
        """Fetch top models for a single category from the Hub."""
        try:
            from huggingface_hub import list_models
            from src.core import huggingface_utils, config

            filter_tasks = {
                "keyword": [config.MODEL_TASK_IMAGE_CLASSIFICATION],
                "category": [config.MODEL_TASK_ZERO_SHOT],
                "description": [config.MODEL_TASK_IMAGE_TO_TEXT],
                "multimodal": ["image-text-to-text", "visual-question-answering", config.MODEL_TASK_IMAGE_TO_TEXT],
            }
            tasks = filter_tasks.get(category, filter_tasks["multimodal"])

            all_results = []
            for t in tasks:
                models = list_models(
                    filter=t,
                    limit=self._prefetch_per_page,
                    sort="downloads",
                )
                for m in models:
                    all_results.append({
                        "id": m.id,
                        "task": t,
                        "capability": huggingface_utils.get_model_capability(t),
                    })

            seen = set()
            unique = []
            for r in all_results:
                if r["id"] not in seen:
                    unique.append(r)
                    seen.add(r["id"])

            # Show models immediately with "Loading..." size, then fetch sizes
            # in background to avoid blocking the UI.
            for item in unique:
                item["size_bytes"] = 0
                item["size_str"] = "Loading..."
            self.after(0, lambda: self._show_prefetch_results(list(unique)) if self.winfo_exists() else None)

            from src.utils.concurrency import DaemonThreadPoolExecutor as ThreadPoolExecutor

            def fetch_size(item):
                try:
                    sz = huggingface_utils.get_remote_model_size(item["id"])
                    item["size_bytes"] = sz
                    item["size_str"] = huggingface_utils.format_size(sz)
                except Exception:
                    item["size_bytes"] = 0
                    item["size_str"] = "Unknown"
                return item

            with ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(fetch_size, unique))
            self.after(0, lambda: self._show_prefetch_results(results) if self.winfo_exists() else None)
        except Exception as exc:
            self.after(0, lambda e=exc: self.lbl_status.configure(
                text=f"Could not load models: {e}", text_color="red"
            ) if self.winfo_exists() else None)

    def start_search(self):
        query = self.search_entry.get()
        search_filter = self.search_filter_var.get()
        self.lbl_status.configure(text=f"Searching Hub for {search_filter} models...")
        self._search_results_cache = []
        self._reset_size_filter()

        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Use submit_replacing so rapid searches only execute the final one
        self._worker.submit_replacing("search", self._search_worker, query, search_filter)

    def _search_worker(self, query, search_filter):
        try:
            from huggingface_hub import list_models
            from src.core import huggingface_utils, config

            filter_tasks = {
                "keyword": [config.MODEL_TASK_IMAGE_CLASSIFICATION],
                "category": [config.MODEL_TASK_ZERO_SHOT],
                "description": [config.MODEL_TASK_IMAGE_TO_TEXT],
                "multimodal": ["image-text-to-text", "visual-question-answering", config.MODEL_TASK_IMAGE_TO_TEXT],
            }
            tasks = filter_tasks.get(search_filter, filter_tasks["multimodal"])

            all_results = []
            for t in tasks:
                models = list_models(
                    filter=t,
                    search=query,
                    limit=10,
                    sort="downloads",
                )
                for m in models:
                    all_results.append({
                        'id': m.id,
                        'task': t,
                        'capability': huggingface_utils.get_model_capability(t)
                    })

            # Deduplicate by ID, keeping the first task found
            seen = set()
            unique_results = []
            for r in all_results:
                if r['id'] not in seen:
                    unique_results.append(r)
                    seen.add(r['id'])

            # Show models immediately with placeholder sizes, then fetch sizes
            # in background so the user can see results right away.
            for item in unique_results:
                item['size_bytes'] = 0
                item['size_str'] = "Loading..."
            self.after(0, lambda: self.show_search_results(list(unique_results)) if self.winfo_exists() else None)

            from src.utils.concurrency import DaemonThreadPoolExecutor as ThreadPoolExecutor

            def fetch_size(item):
                try:
                    sz = huggingface_utils.get_remote_model_size(item['id'])
                    item['size_bytes'] = sz
                    item['size_str'] = huggingface_utils.format_size(sz)
                except Exception:
                    item['size_bytes'] = 0
                    item['size_str'] = "Unknown"
                return item

            with ThreadPoolExecutor(max_workers=10) as executor:
                results_with_details = list(executor.map(fetch_size, unique_results))
            self.after(0, lambda: self.show_search_results(results_with_details, refresh_size_filter=False) if self.winfo_exists() else None)
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: self.lbl_status.configure(text=f"Error: {error_msg}", text_color="red") if self.winfo_exists() else None)

    def show_search_results(self, results, refresh_size_filter=True):
        self._search_results_cache = list(results or [])
        if refresh_size_filter:
            self._configure_size_filter(self._search_results_cache)
        filtered_results = self._get_size_filtered_results()
        self.lbl_status.configure(
            text=f"Found {len(filtered_results)} of {len(self._search_results_cache)} models.",
            text_color="gray"
        )
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Re-add header
        header_text = f"{'Model ID':<40} | {'Capability':^15} | {'Size':>10}"
        ctk.CTkLabel(self.results_frame, text=header_text, font=("Courier New", 12, "bold"), text_color="gray", anchor="w").pack(fill="x", pady=(5, 10), padx=5)

        if not self._search_results_cache:
            ctk.CTkLabel(self.results_frame, text="No models found matching your query.", text_color="gray").pack(pady=20)
            return
        if not filtered_results:
            ctk.CTkLabel(self.results_frame, text="No models match the current size filter.", text_color="gray").pack(pady=20)
            return

        for item in filtered_results:
            self.add_result_item(item['id'], item['size_str'], item['capability'])

    def _reset_size_filter(self):
        self._size_filter_min = 0
        self._size_filter_max = 0
        self._size_filter_active_max = 0
        self.size_slider.configure(from_=0, to=1, state="disabled")
        self.size_slider.set(1)
        self.size_label.configure(text="No size data yet")

    def _configure_size_filter(self, results):
        if not results:
            self._reset_size_filter()
            return

        sizes = [max(int(item.get("size_bytes", 0) or 0), 0) for item in results]
        self._size_filter_min = min(sizes)
        self._size_filter_max = max(sizes)
        self._size_filter_active_max = self._size_filter_max

        if self._size_filter_min == self._size_filter_max:
            self.size_slider.configure(from_=self._size_filter_min, to=self._size_filter_max + 1, state="disabled")
            self.size_slider.set(self._size_filter_max + 1)
        else:
            self.size_slider.configure(from_=self._size_filter_min, to=self._size_filter_max, state="normal")
            self.size_slider.set(self._size_filter_max)

        self._update_size_label()

    def _get_size_filtered_results(self):
        return [
            item for item in self._search_results_cache
            if max(int(item.get("size_bytes", 0) or 0), 0) <= self._size_filter_active_max
        ]

    def _update_size_label(self):
        from src.core import huggingface_utils

        if not self._search_results_cache:
            self.size_label.configure(text="No size data yet")
            return

        current = huggingface_utils.format_size(self._size_filter_active_max)
        minimum = huggingface_utils.format_size(self._size_filter_min)
        maximum = huggingface_utils.format_size(self._size_filter_max)
        self.size_label.configure(text=f"{minimum} to {current} of {maximum}")

    def _on_size_slider_change(self, value):
        if not self._search_results_cache:
            return

        self._size_filter_active_max = int(round(value))
        self._update_size_label()
        self.show_search_results(self._search_results_cache, refresh_size_filter=False)

    def add_result_item(self, model_id, size_str, capability):
        frame = ctk.CTkFrame(self.results_frame)
        frame.pack(fill="x", pady=2, padx=5)

        # Consistent column-like look
        display_text = f"{model_id:<40} | {capability:^15} | {size_str:>10}"

        ctk.CTkLabel(
            frame,
            text=display_text,
            font=("Courier New", 12),
            anchor="w"
        ).pack(side="left", padx=10, fill="x", expand=True)

        # Buttons
        btn_select = ctk.CTkButton(frame, text="Select", width=100, fg_color="#3B8ED0",
                                   command=lambda m=model_id: self.select_remote_model(m))
        btn_select.pack(side="right", padx=5)

        btn_download = ctk.CTkButton(frame, text="Download", width=100, fg_color="#2FA572",
                                     command=lambda m=model_id: self.start_download(m))
        btn_download.pack(side="right", padx=5)

    def select_remote_model(self, model_id):
        """Select a Hub model for local inference without downloading it."""
        local_tab = self.local_tab
        if local_tab is None or not hasattr(local_tab, 'local_model_var'):
            self.lbl_status.configure(text="Cannot select model from this context.", text_color="red")
            return
        local_tab.local_model_var.set(model_id)
        if hasattr(local_tab, 'select_local_model'):
            local_tab.select_local_model(model_id)
        self.lbl_status.configure(text=f"Selected {model_id} for local inference.", text_color="green")

    def start_download(self, model_id):
        self.lbl_status.configure(text=f"Preparing download for {model_id}...", text_color="gray")
        self.progress.set(0)

        self.download_queue = queue.Queue()
        self._worker.submit(
            self._prepare_and_download_model,
            model_id,
            self.download_queue
        )

        self.poll_download_queue()

    def _prepare_and_download_model(self, model_id, download_queue):
        import logging
        from src.core import huggingface_utils

        logger = logging.getLogger(__name__)
        logger.info(f"[DownloadManager] Preparing download for {model_id}")

        download_queue.put(("status_update", f"Checking compatibility for {model_id}..."))
        logger.info(f"[DownloadManager] Running compatibility check for {model_id}")
        reason = huggingface_utils.get_local_inference_incompatibility_reason(model_id)
        if reason is not None:
            logger.warning(f"[DownloadManager] Model {model_id} incompatible: {reason}")
            download_queue.put(("incompatible_model", (model_id, reason)))
            return
        logger.info(f"[DownloadManager] Compatibility check passed for {model_id}")

        logger.info(f"[DownloadManager] Starting download_model_worker for {model_id}")
        huggingface_utils.download_model_worker(model_id, download_queue)

    def poll_download_queue(self):
        import logging
        _logger = logging.getLogger(__name__)
        try:
            while True:
                msg_type, data = self.download_queue.get_nowait()
                _logger.info(f"[DownloadManager] Queue message: {msg_type} (data={data!r:.200})" if isinstance(data, str) else f"[DownloadManager] Queue message: {msg_type}")

                if msg_type == "model_download_progress":
                    downloaded, total = data
                    if total > 0:
                        self.progress.set(downloaded / total)

                elif msg_type == "status_update":
                    self.lbl_status.configure(text=data, text_color="gray")

                elif msg_type == "download_complete":
                    _logger.info(f"[DownloadManager] Download complete: {data}")
                    self.on_download_complete(data)
                    return

                elif msg_type == "incompatible_model":
                    model_id, reason = data
                    self.lbl_status.configure(text=f"Cannot download {model_id}: {reason}", text_color="red")
                    import tkinter.messagebox as mb
                    mb.showerror(
                        "Incompatible Model",
                        f"The model '{model_id}' cannot be used with Synapic.\n\n"
                        f"Reason: {reason}\n\n"
                        "This model is not suitable for Synapic's local inference runtime.\n"
                        "Please choose a different model."
                    )
                    return

                elif msg_type == "error":
                    self.lbl_status.configure(text=f"Download failed: {data}", text_color="red")
                    return  # Stop polling

        except queue.Empty:
            # Continue polling if not closed
            if self.winfo_exists():
                self.after(100, self.poll_download_queue)

    def on_download_complete(self, model_id):
        self.lbl_status.configure(text=f"Download complete: {model_id}!", text_color="green")
        self.progress.set(1.0)

        # Auto-select the downloaded model for local inference
        self.session.engine.provider = "local"
        self.session.engine.model_id = model_id

        # Try to set the appropriate task based on model info
        try:
            from src.core import huggingface_utils
            local_models = huggingface_utils.find_local_models()
            model_info = local_models.get(model_id)
            if model_info:
                self.session.engine.task = model_info.get('suggested_task', "image-to-text")
                logger.debug(f"Auto-selected model {model_id} with task {self.session.engine.task}")
        except Exception as e:
            logger.warning(f"Could not determine task for {model_id}: {e}")
            self.session.engine.task = "image-to-text"  # Default fallback

        # Refresh local tab cache and update selection
        if self.local_tab is not None:
            if hasattr(self.local_tab, 'refresh_local_cache'):
                self.local_tab.refresh_local_cache()
            if hasattr(self.local_tab, 'local_model_var'):
                self.local_tab.local_model_var.set(model_id)

    def destroy(self):
        """Override destroy to clean up worker thread."""
        if hasattr(self, '_worker'):
            self._worker.shutdown()
        super().destroy()