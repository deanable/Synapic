"""
Local provider configuration tab for Step 2.
"""

import logging
import os
import re
import shutil
import tkinter.messagebox as mb

import customtkinter as ctk

from .provider_tab_base import ProviderTabBase

logger = logging.getLogger(__name__)


class LocalProviderTab(ProviderTabBase):
    """Local provider configuration tab."""

    def __init__(self, parent, session, worker, persist_preference_callback, filter_image_models_func):
        super().__init__(parent, session, worker, persist_preference_callback, filter_image_models_func)
        # Cache for models (not used for local, but keeping for interface compatibility)
        self._models_cache = []
        # Set up the local-specific UI
        self._setup_local_ui()

    def _get_provider_name(self) -> str:
        return "local"

    def _get_provider_display_name(self) -> str:
        return "LFM"

    def _get_default_model(self) -> str:
        # This will be set when a model is selected
        return self.session.engine.model_id or ""

    def _get_default_task(self) -> str:
        # Task will be determined from the selected model
        return self.session.engine.task or "image-classification"

    def _get_models_header_text(self) -> str:
        return f"{'Model ID':<40} | {'Capability':^15} | {'Size':>10}"

    def _setup_local_ui(self):
        """Set up the local provider specific UI."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # List area grows

        # Header with cache info
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text="📦 Downloaded Models",
                     font=("Roboto", 16, "bold")).pack(side="left", padx=5)

        self.cache_count_label = ctk.CTkLabel(header, text="(0 models)",
                                              text_color="gray")
        self.cache_count_label.pack(side="left", padx=5)

        # Tracks (button, model_id, is_classification) for filtering
        self._model_buttons: list[tuple] = []

        ctk.CTkButton(header, text="+ Find & Download Models",
                      command=self.open_download_manager,
                      width=180).pack(side="right", padx=5)

        # List of cached models ONLY
        self.local_list_frame = ctk.CTkScrollableFrame(
            self,
            label_text="Ready for Local Inference"
        )
        self.local_list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Add a header label for clarity
        header_text = self._get_models_header_text()
        self.list_header = ctk.CTkLabel(
            self.local_list_frame,
            text=header_text,
            font=("Courier New", 12, "bold"),
            text_color="gray",
            anchor="w"
        )
        self.list_header.pack(fill="x", pady=(5, 10), padx=5)

        # Selection and action
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        footer.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(footer, text="Selected:").pack(side="left")
        self.local_model_var = ctk.StringVar(value=self.session.engine.model_id or "")
        ctk.CTkLabel(footer, textvariable=self.local_model_var,
                     font=("Roboto", 12, "bold")).pack(side="left", padx=10)

        self.model_status_label = ctk.CTkLabel(
            footer, text="", text_color="gray", font=("Roboto", 10)
        )
        self.model_status_label.pack(side="left", padx=10)

        ctk.CTkButton(footer, text="Use for Local Inference",
                      command=self.save_local).pack(side="right")

        # Load cached models
        self.refresh_local_cache()

        # Probability Scoring Section (Local only)
        self._setup_probability_scoring_section()

    def _setup_api_key_section(self):
        """Local provider doesn't use API keys."""
        pass

    def _setup_tools_section(self):
        """Local provider doesn't need the standard tools section."""
        pass

    def _setup_models_list(self):
        """Model list is already set up in _setup_local_ui."""
        pass

    def _setup_selection_area(self):
        """Selection area is already set up in _setup_local_ui."""
        pass

    def open_download_manager(self):
        """Opens a separate dialog for browsing and downloading models."""
        from .step2_tagging import DownloadManagerDialog
        # CTkToplevel needs the root window, not a CTkFrame parent
        root = self.winfo_toplevel()
        DownloadManagerDialog(root, self.session, local_tab=self)

    def refresh_local_cache(self):
        """Refresh the list of locally cached models."""
        for widget in self.local_list_frame.winfo_children():
            widget.destroy()
        self._model_buttons.clear()

        try:
            from src.core import huggingface_utils
            # Get ALL local models, not filtered by task yet
            all_local = huggingface_utils.find_local_models()

            if not all_local:
                ctk.CTkLabel(
                    self.local_list_frame,
                    text="No models downloaded yet.\nClick '+ Find & Download Models' to browse the Hub.",
                    text_color="gray",
                    justify="center"
                ).pack(pady=20)
                self.cache_count_label.configure(text="(0 models, 0 B)")
            else:
                total_bytes = sum(m.get('size_bytes', 0) for m in all_local.values())
                total_str = huggingface_utils.format_size(total_bytes)
                self.cache_count_label.configure(text=f"({len(all_local)} models, {total_str})")

                # Sort models by size descending
                sorted_models = sorted(all_local.items(), key=lambda x: x[1].get('size_bytes', 0), reverse=True)

                for model_id, info in sorted_models:
                    task = info.get('suggested_task', '')
                    is_class = (task == 'image-classification')
                    self.add_cached_model_item(
                        model_id,
                        info.get('size_str', 'Unknown size'),
                        info.get('capability', 'Unknown'),
                        is_classification=is_class,
                    )

            # Apply current probability-mode filter
            self._update_model_list_filter()

        except Exception as e:
            ctk.CTkLabel(
                self.local_list_frame,
                text=f"Error scanning cache: {e}",
                text_color="red"
            ).pack()

    def _setup_probability_scoring_section(self):
        """Set up the probability scoring controls for local models only."""
        # Probability scoring frame
        self.probability_frame = ctk.CTkFrame(self)
        self.probability_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))
        self.probability_frame.grid_columnconfigure(0, weight=1)

        # Section header with tagging-mode radio group
        self.probability_header = ctk.CTkFrame(self.probability_frame, fg_color="transparent")
        self.probability_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        self.probability_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.probability_header,
            text="Tagging Mode:",
            font=("Roboto", 12, "bold"),
        ).pack(side="left", padx=(5, 10))

        self.probability_mode_var = ctk.StringVar(value="llm")

        self.mode_llm = ctk.CTkRadioButton(
            self.probability_header,
            text="LLM only",
            variable=self.probability_mode_var,
            value="llm",
            command=self._toggle_probability_section,
        )
        self.mode_llm.pack(side="left", padx=5)

        self.mode_probability = ctk.CTkRadioButton(
            self.probability_header,
            text="Probability only",
            variable=self.probability_mode_var,
            value="probability",
            command=self._toggle_probability_section,
        )
        self.mode_probability.pack(side="left", padx=5)

        self.mode_both = ctk.CTkRadioButton(
            self.probability_header,
            text="Both",
            variable=self.probability_mode_var,
            value="both",
            command=self._toggle_probability_section,
        )
        self.mode_both.pack(side="left", padx=5)

        # Probability scoring content (initially visible)
        self.probability_content = ctk.CTkFrame(self.probability_frame, fg_color="transparent")
        self.probability_content.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.probability_content.grid_columnconfigure(1, weight=1)

        # Candidate tokens input — autocomplete entry
        ctk.CTkLabel(self.probability_content, text="Candidate Tokens:").grid(row=0, column=0, sticky="w", padx=(0, 10))

        candidate_frame = ctk.CTkFrame(self.probability_content, fg_color="transparent")
        candidate_frame.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(5, 5))
        candidate_frame.grid_columnconfigure(0, weight=1)

        self.candidate_entry = ctk.CTkEntry(
            candidate_frame,
            placeholder_text="Type to search model labels, e.g. 'forest'...",
            height=32,
        )
        self.candidate_entry.grid(row=0, column=0, sticky="ew")
        self.candidate_entry.bind("<KeyRelease>", self._on_candidate_key_release)
        self.candidate_entry.bind("<FocusOut>", self._on_candidate_focus_out)
        self.candidate_entry.bind("<Return>", self._on_candidate_focus_out)

        # Hidden text box backing store for comma-separated candidates
        self.candidate_box = ctk.CTkTextbox(candidate_frame, height=1)
        self.candidate_box.grid(row=1, column=0, sticky="ew")
        self.candidate_box.grid_remove()  # hidden — used only for persistence
        self.candidate_box.insert("1.0", "A,B,C,D")

        # Autocomplete dropdown (hidden by default)
        self._autocomplete_frame = None
        self._autocomplete_buttons: list[ctk.CTkButton] = []
        self._current_label_tokens: list[str] = []

        # Probability threshold — filters the *candidate* probability map before it is
        # used for tagging. This is separate from the global confidence threshold on the
        # main tagging step (which filters extracted tags; it uses a 1-100 scale).
        ctk.CTkLabel(self.probability_content, text="Candidate Probability Threshold:").grid(
            row=1, column=0, sticky="w", padx=(0, 10)
        )
        self.threshold_value_label = ctk.CTkLabel(
            self.probability_content,
            text="0%",
            width=50,
        )
        self.threshold_value_label.grid(row=1, column=1, sticky="e", padx=(0, 10), pady=(0, 5))

        slider_row = ctk.CTkFrame(self.probability_content, fg_color="transparent")
        slider_row.grid(row=2, column=0, columnspan=2, sticky="ew", padx=(0, 10), pady=(0, 5))
        slider_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            slider_row,
            text="Pass more candidates",
            font=("Roboto", 10),
            text_color="gray",
        ).grid(row=0, column=0, padx=(0, 10))

        self.threshold_slider = ctk.CTkSlider(
            slider_row,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            command=self._on_threshold_change,
        )
        self.threshold_slider.grid(row=0, column=1, sticky="ew")

        ctk.CTkLabel(
            slider_row,
            text="Pass fewer candidates",
            font=("Roboto", 10),
            text_color="gray",
        ).grid(row=0, column=2, padx=(10, 0))

        # CLIP embedding rescue (opt-in). When the model's label space cannot
        # score the candidates at all, the run can fall back to CLIP
        # image-text similarity — but CLIP is a ~600MB download on first use,
        # so this never happens unless the user asks for it here.
        self.embedding_rescue_var = ctk.BooleanVar(value=False)
        self.embedding_rescue_checkbox = ctk.CTkCheckBox(
            self.probability_content,
            text="Rescue unmatched candidates with CLIP similarity (downloads ~600MB model on first use; scores are not calibrated)",
            variable=self.embedding_rescue_var,
            font=("Roboto", 11),
        )
        self.embedding_rescue_checkbox.grid(
            row=3, column=0, columnspan=2, sticky="w", padx=(0, 10), pady=(5, 0)
        )

        # Populate from any previously saved session values
        self._sync_probability_controls_from_session()

    def _toggle_probability_section(self):
        """Show the probability controls unless the mode is LLM-only."""
        if self.probability_mode_var.get() == "llm":
            self.probability_content.grid_remove()
        else:
            self.probability_content.grid()
        self._update_model_list_filter()

    def _parse_probability_candidates(self):
        """Parse candidate tokens from the entry (comma separated)."""
        text = self.candidate_entry.get().strip()
        if not text:
            return []
        return [c.strip() for c in re.split(r"[,;\n]+", text) if c.strip()]

    def _parse_probability_threshold(self):
        """Read and clamp the probability threshold from the slider (0.0-1.0)."""
        try:
            value = float(self.threshold_slider.get())
        except (TypeError, ValueError):
            logger.warning("Invalid probability threshold, defaulting to 0.0")
            return 0.0
        return max(0.0, min(1.0, value))

    def _on_threshold_change(self, value):
        """Update the threshold value label when the slider moves."""
        try:
            pct = round(float(value) * 100)
        except (TypeError, ValueError):
            pct = 0
        self.threshold_value_label.configure(text=f"{pct}%")

    # ------------------------------------------------------------------
    # Model list filtering (probability mode)
    # ------------------------------------------------------------------

    def _update_model_list_filter(self):
        """Dim non-classification models when probability mode is active."""
        if not hasattr(self, "probability_mode_var"):
            return
        mode = self.probability_mode_var.get()
        needs_classifier = mode in ("probability", "both")
        selectable = 0
        total = len(self._model_buttons)

        for btn, _mid, is_class in self._model_buttons:
            try:
                if needs_classifier and not is_class:
                    btn.configure(
                        fg_color="#3B3B3B",
                        text_color="#666666",
                        state="disabled",
                    )
                else:
                    btn.configure(
                        fg_color="transparent",
                        text_color="#2FA572",
                        state="normal",
                    )
                    selectable += 1
            except Exception:
                pass

        # Update the list header with a compatibility note
        if not hasattr(self, "list_header"):
            return
        try:
            if needs_classifier and total > 0:
                self.list_header.configure(
                    text=(
                        f"{self._get_models_header_text()}"
                        f"    [{selectable}/{total} models support probability scoring]"
                    ),
                )
            else:
                self.list_header.configure(text=self._get_models_header_text())
        except Exception:
            pass  # widget may not be fully realised yet during init

    # ------------------------------------------------------------------
    # Candidate token autocomplete
    # ------------------------------------------------------------------

    def _on_candidate_key_release(self, event):
        """Show autocomplete suggestions as the user types."""
        # Ignore modifier-only keys
        if event.keysym in ("Shift_L", "Shift_R", "Control_L", "Control_R",
                            "Alt_L", "Alt_R", "Caps_Lock", "Tab"):
            return

        typed = self.candidate_entry.get().strip().rstrip(",;")
        if len(typed) < 1:
            self._hide_autocomplete()
            return

        # Load label tokens from the selected model (cached)
        if not self._current_label_tokens:
            model_id = self.local_model_var.get()
            if not model_id:
                return
            try:
                from src.core import huggingface_utils
                self._current_label_tokens = huggingface_utils.get_model_label_tokens(model_id)
            except Exception:
                return
            if not self._current_label_tokens:
                return

        typed_lower = typed.lower()
        matches = [tok for tok in self._current_label_tokens if typed_lower in tok.lower()]
        matches = matches[:8]  # cap dropdown

        if not matches:
            self._hide_autocomplete()
            return

        self._show_autocomplete(matches, typed)

    def _show_autocomplete(self, matches, typed):
        """Display an autocomplete dropdown below the entry."""
        self._hide_autocomplete()

        self._autocomplete_frame = ctk.CTkToplevel(self)
        self._autocomplete_frame.overrideredirect(True)
        self._autocomplete_frame.attributes("-topmost", True)
        self._autocomplete_frame.configure(fg_color="#2B2B2B")

        # Position below the entry widget
        x = self.candidate_entry.winfo_rootx()
        y = self.candidate_entry.winfo_rooty() + self.candidate_entry.winfo_height()
        w = self.candidate_entry.winfo_width()
        h = min(len(matches) * 28 + 4, 250)
        self._autocomplete_frame.geometry(f"{w}x{h}+{x}+{y}")

        self._autocomplete_buttons.clear()
        for label in matches:
            btn = ctk.CTkButton(
                self._autocomplete_frame,
                text=label,
                anchor="w",
                fg_color="transparent",
                hover_color="#3B5998",
                height=26,
                font=("Roboto", 11),
                command=lambda lbl=label: self._select_autocomplete(lbl),
            )
            btn.pack(fill="x", padx=2, pady=1)
            self._autocomplete_buttons.append(btn)

    def _select_autocomplete(self, label):
        """Insert the selected label into the entry, replacing the partial text."""
        current = self.candidate_entry.get().strip()
        # Build the new candidate list: replace partial with the full label
        parts = [p.strip() for p in re.split(r"[,;]+", current) if p.strip()]
        if parts:
            parts[-1] = label  # replace the last (incomplete) token
        else:
            parts = [label]
        new_text = ", ".join(parts) + ", "
        self.candidate_entry.delete(0, "end")
        self.candidate_entry.insert(0, new_text)

        # Also update the hidden backing store
        self.candidate_box.delete("1.0", "end")
        self.candidate_box.insert("1.0", new_text)

        self._hide_autocomplete()
        self.candidate_entry.focus_set()

    def _hide_autocomplete(self, _event=None):
        """Close the autocomplete dropdown if open."""
        if self._autocomplete_frame is not None:
            try:
                self._autocomplete_frame.destroy()
            except Exception:
                pass
            self._autocomplete_frame = None
            self._autocomplete_buttons.clear()

    # ------------------------------------------------------------------
    # Candidate input formatting
    # ------------------------------------------------------------------

    def _on_candidate_focus_out(self, _event=None):
        """Format the candidate text when the user leaves the entry."""
        self._hide_autocomplete()
        self._format_candidate_text()

    def _format_candidate_text(self):
        """Normalize the comma-separated candidate string.

        - Strips leading/trailing whitespace from each token
        - Removes empty tokens (double commas, trailing comma)
        - Deduplicates tokens (case-insensitive, keeps first occurrence)
        - Joins with ', ' (consistent single space after comma)
        """
        raw = self.candidate_entry.get().strip()
        if not raw:
            return
        parts = [p.strip() for p in re.split(r"[,;\n]+", raw) if p.strip()]
        # Deduplicate preserving order (case-insensitive)
        seen: set[str] = set()
        unique: list[str] = []
        for part in parts:
            key = part.lower()
            if key not in seen:
                seen.add(key)
                unique.append(part)
        formatted = ", ".join(unique)
        self.candidate_entry.delete(0, "end")
        self.candidate_entry.insert(0, formatted)
        # Sync hidden backing store
        self.candidate_box.delete("1.0", "end")
        self.candidate_box.insert("1.0", formatted)

    def _sync_probability_controls_from_session(self):
        """Populate the probability scoring controls from the session engine."""
        engine = self.session.engine
        mode = str(getattr(engine, "probability_mode", "") or "").lower()
        if mode not in ("llm", "probability", "both"):
            # Legacy configs only persisted probability_enabled
            mode = "both" if getattr(engine, "probability_enabled", False) else "llm"
        elif mode == "llm" and getattr(engine, "probability_enabled", False):
            # Legacy sessions enable probabilities without a mode field
            mode = "both"
        self.probability_mode_var.set(mode)

        candidates = getattr(engine, "probability_candidates", None) or []
        if candidates:
            text = ",".join(str(c) for c in candidates)
            self.candidate_entry.delete(0, "end")
            self.candidate_entry.insert(0, text)
            self.candidate_box.delete("1.0", "end")
            self.candidate_box.insert("1.0", text)
        # Clear cached labels so they reload for the new model
        self._current_label_tokens = []

        threshold = float(getattr(engine, "probability_threshold", 0.0) or 0.0)
        threshold = max(0.0, min(1.0, threshold))
        self.threshold_slider.set(threshold)
        self.threshold_value_label.configure(text=f"{round(threshold * 100)}%")

        self.embedding_rescue_var.set(
            bool(getattr(engine, "embedding_rescue_enabled", False))
        )

        self._toggle_probability_section()

    def add_cached_model_item(self, model_id, size_str, capability, is_classification=True):
        """Add a cached model to the list."""
        frame = ctk.CTkFrame(self.local_list_frame)
        frame.pack(fill="x", pady=2)

        # Format the text with "columns" using padding/fixed width font if possible,
        # but for now a nice formatted string.
        display_text = f"✓ {model_id:<40} | {capability:^15} | {size_str:>10}"

        btn = ctk.CTkButton(
            frame,
            text=display_text,
            font=("Courier New", 12),
            fg_color="transparent",
            border_width=1,
            text_color="#2FA572",
            anchor="w",
            command=lambda m=model_id: self.select_local_model(m)
        )
        btn.pack(side="left", fill="x", expand=True)

        # Track for filtering
        self._model_buttons.append((btn, model_id, is_classification))

        # Delete button
        ctk.CTkButton(
            frame,
            text="🗑️",
            width=30,
            fg_color="transparent",
            hover_color="red",
            command=lambda m=model_id: self.delete_cached_model(m)
        ).pack(side="right", padx=2)

    def select_local_model(self, model_id):
        """Handle local model selection."""
        self.local_model_var.set(model_id)
        self._preload_candidate_tokens(model_id)
        self._update_probability_compatibility_warning()
        self.model_status_label.configure(
            text="✓ Model selected — click 'Use for Local Inference' to confirm",
            text_color="#2FA572",
        )

    def _update_probability_compatibility_warning(self):
        """Show an early, explicit warning when the candidate set is incompatible
        with the selected local classification model.

        This is a pre-flight check only. It mirrors the logic in
        :func:`~src.core.huggingface_utils.run_local_logprob_inference` so users
        see problems at config time instead of mid-batch.
        """
        if not hasattr(self, "probability_mode_var"):
            return

        mode = self.probability_mode_var.get()
        if mode == "llm":
            self._clear_probability_warning()
            return

        model_id = self.local_model_var.get()
        if not model_id:
            self._clear_probability_warning()
            return

        try:
            from src.core import huggingface_utils

            candidates = self._parse_probability_candidates()
            labels = huggingface_utils.get_model_label_tokens(model_id)
            summary = huggingface_utils.summarize_candidate_compatibility(candidates, labels)

            if summary["total"] == 0:
                self._clear_probability_warning()
                return

            if summary["unmatched"] and not summary["exact"] and not summary["fuzzy"]:
                self.model_status_label.configure(
                    text=(
                        "⚠ Probability scoring will likely fail: none of the candidate tokens "
                        f"match {model_id}'s labels.\n"
                        f"Details: {summary['reason']}"
                    ),
                    text_color="#FF8C00",
                )
                return

            if not summary["exact"] and (summary["fuzzy"] or summary["unmatched"]):
                self.model_status_label.configure(
                    text=(
                        "⚠ Probability scoring is usable, but not exact: "
                        f"{summary['reason']}\n"
                        "Candidates that do not match a model label score 0.0, "
                        "and fuzzy matches score the nearest model label, not the candidate concept."
                    ),
                    text_color="#FF8C00",
                )
                return

            self._clear_probability_warning()
        except Exception as e:
            logger.debug(f"Could not check probability candidate compatibility for {model_id}: {e}")
            self._clear_probability_warning()

    def _clear_probability_warning(self):
        """Restore the normal selection status text after a compatibility check."""
        if hasattr(self, "probability_mode_var") and self.probability_mode_var.get() != "llm":
            if not self.local_model_var.get():
                return
            self.model_status_label.configure(
                text="✓ Model selected — click 'Use for Local Inference' to confirm",
                text_color="#2FA572",
            )

    def _preload_candidate_tokens(self, model_id):
        """Preload the candidate tokens from the model's label set.

        Fills the entry with the model's classification labels (from the local
        cache) so the user doesn't have to type candidates by hand. Models
        without a label set (e.g. captioning VLMs) leave the entry unchanged.
        Also caches the labels for autocomplete.
        """
        try:
            from src.core import huggingface_utils
            labels = huggingface_utils.get_model_label_tokens(model_id)
        except Exception as e:
            logger.debug(f"Could not preload candidate tokens for {model_id}: {e}")
            return
        # Cache for autocomplete
        self._current_label_tokens = labels or []
        if labels:
            text = ",".join(labels)
            self.candidate_entry.delete(0, "end")
            self.candidate_entry.insert(0, text)
            self.candidate_box.delete("1.0", "end")
            self.candidate_box.insert("1.0", text)
            logger.info(
                f"Preloaded {len(labels)} candidate tokens for {model_id}"
            )

    def delete_cached_model(self, model_id):
        """Delete a cached model from disk."""
        # Simple confirmation using tkinter.messagebox if available, or just delete for now
        if mb.askyesno("Confirm Delete", f"Are you sure you want to delete {model_id} from local cache?\nThis will free up disk space."):
            try:
                from src.core import huggingface_utils
                path = huggingface_utils.get_model_cache_dir(model_id)
                if os.path.exists(path):
                    shutil.rmtree(path)
                    logger.info(f"Deleted model directory: {path}")

                self.refresh_local_cache()
            except Exception as e:
                mb.showerror("Error", f"Failed to delete model: {e}")

    def save_to_session(self):
        """Write the local tab's UI state (model + probability scoring) into the session engine."""
        self.session.engine.provider = "local"
        self.session.engine.model_id = self.local_model_var.get()
        mode = self.probability_mode_var.get()
        self.session.engine.probability_mode = mode
        self.session.engine.probability_enabled = mode != "llm"
        self.session.engine.probability_candidates = self._parse_probability_candidates()
        self.session.engine.probability_threshold = self._parse_probability_threshold()
        self.session.engine.embedding_rescue_enabled = bool(
            self.embedding_rescue_var.get()
        )

    def save_local(self):
        """Save the local provider configuration."""
        self.save_to_session()

        # Find task from cache
        try:
            from src.core import huggingface_utils
            local_models = huggingface_utils.find_local_models()
            model_info = local_models.get(self.session.engine.model_id)
            if model_info:
                # Use the newly added suggested_task from hf_utils
                self.session.engine.task = model_info.get('suggested_task', "image-classification")
                logger.debug(f"Setting task for {self.session.engine.model_id} to {self.session.engine.task}")
        except Exception as e:
            logger.debug(f"Could not determine task for local model: {e}")

        # Try to save config
        try:
            from src.utils.config_manager import save_config
            save_config(self.session)
            if hasattr(self, 'model_status_label'):
                self.model_status_label.configure(
                    text="✓ Configuration saved",
                    text_color="#2FA572",
                )
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def _load_models(self):
        """Local provider doesn't load models from an API."""
        pass

    def _display_models(self, models):
        """Local provider display is handled by refresh_local_cache."""
        pass

    def _save_provider_config(self, model_id):
        """Local provider-specific config saving is handled by save_local."""
        pass

    def refresh(self):
        """Refresh the tab when it becomes visible."""
        # Update model entry
        if hasattr(self, 'local_model_var'):
            current_model = getattr(self.session.engine, "model_id", "") or ""
            self.local_model_var.set(current_model)

        # Sync probability scoring controls from the session
        self._sync_probability_controls_from_session()


def create_local_tab(parent, session, worker, persist_preference_callback, filter_image_models_func):
    """Factory function to create a local provider tab."""
    return LocalProviderTab(
        parent, session, worker, persist_preference_callback, filter_image_models_func
    )