"""
Step Dedup: Deduplication Wizard Step
=====================================

This module provides the UI for detecting and managing duplicate images
in Daminion collections and searches. Users can:
- Configure hash algorithm and similarity threshold
- Scan for duplicates with visual progress
- Review duplicate groups with thumbnail previews
- Select keep strategy and apply deduplication actions

Author: Synapic Project
"""

import customtkinter as ctk
import logging
import io
from PIL import Image, ImageTk
from typing import Optional, List, Dict, Any, Callable
import threading
from tkinter import messagebox

from src.core.dedup_processor import DaminionDedupProcessor, DedupAction, DedupScanResult
from src.core.dedup import DuplicateGroup, KeepStrategy, DedupDecision

logger = logging.getLogger(__name__)


class DuplicateGroupFrame(ctk.CTkFrame):
    """
    A frame displaying a single duplicate group with thumbnails.
    """
    
    def __init__(
        self,
        parent,
        group: DuplicateGroup,
        group_index: int,
        processor: DaminionDedupProcessor,
        on_selection_changed: Optional[Callable] = None
    ):
        super().__init__(parent, fg_color=("gray85", "gray20"), corner_radius=8)
        
        self.group = group
        self.group_index = group_index
        self.processor = processor
        self.on_selection_changed = on_selection_changed
        
        # Track which item is selected to keep
        self.keep_item = ctk.StringVar(value=group.items[0] if group.items else "")
        
        # Track if this is a false positive (keep all)
        self.keep_all = ctk.BooleanVar(value=False)
        
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))
        
        similarity = min(group.similarity_scores.values()) if group.similarity_scores else 0
        title_text = f"Group {group_index + 1}: {len(group.items)} items ({similarity:.1f}% similar)"
        
        title_label = ctk.CTkLabel(
            header,
            text=title_text,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title_label.pack(side="left")
        
        # Keep All checkbox (false positive)
        self.keep_all_cb = ctk.CTkCheckBox(
            header,
            text="Keep All (Not duplicates)",
            variable=self.keep_all,
            command=self._on_keep_all_changed,
            font=ctk.CTkFont(size=11),
            text_color=("orange", "#FFA500")
        )
        self.keep_all_cb.pack(side="right", padx=10)
        
        algo_label = ctk.CTkLabel(
            header,
            text=f"[{group.hash_type.upper()}]",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray60")
        )
        algo_label.pack(side="right")
        
        # Thumbnails container
        thumbnails_frame = ctk.CTkFrame(self, fg_color="transparent")
        thumbnails_frame.pack(fill="x", padx=10, pady=10)
        
        self.thumbnail_widgets = []
        
        for item_id in group.items:
            item_frame = self._create_item_widget(thumbnails_frame, item_id)
            item_frame.pack(side="left", padx=5)
            self.thumbnail_widgets.append(item_frame)
    
    def _create_item_widget(self, parent, item_id: str) -> ctk.CTkFrame:
        """Create a widget for a single item with thumbnail and selection."""
        frame = ctk.CTkFrame(parent, fg_color=("gray80", "gray25"), corner_radius=6)
        
        # Get thumbnail
        thumbnail_bytes = self.processor.get_item_thumbnail(item_id)
        if thumbnail_bytes:
            try:
                img = Image.open(io.BytesIO(thumbnail_bytes))
                img.thumbnail((100, 100))
                photo = ctk.CTkImage(img, size=(100, 100))
                
                img_label = ctk.CTkLabel(frame, image=photo, text="")
                img_label.image = photo  # Keep reference
                img_label.pack(padx=5, pady=5)
            except Exception as e:
                logger.warning(f"Failed to load thumbnail for {item_id}: {e}")
                placeholder = ctk.CTkLabel(frame, text="[No Image]", width=100, height=100)
                placeholder.pack(padx=5, pady=5)
        else:
            placeholder = ctk.CTkLabel(frame, text="[No Image]", width=100, height=100)
            placeholder.pack(padx=5, pady=5)
        
        # Item ID
        id_label = ctk.CTkLabel(
            frame,
            text=f"ID: {item_id}",
            font=ctk.CTkFont(size=10)
        )
        id_label.pack()
        
        # Similarity score
        similarity = self.group.similarity_scores.get(item_id, 0)
        sim_label = ctk.CTkLabel(
            frame,
            text=f"{similarity:.1f}%",
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray60")
        )
        sim_label.pack()
        
        # Radio button to select keep
        radio = ctk.CTkRadioButton(
            frame,
            text="Keep",
            variable=self.keep_item,
            value=item_id,
            command=self._on_selection_changed,
            font=ctk.CTkFont(size=11)
        )
        radio.pack(pady=5)
        
        return frame
    
    def _on_selection_changed(self):
        """Called when the keep selection changes."""
        if self.on_selection_changed:
            self.on_selection_changed(self.group_index, self.keep_item.get())
    
    def _on_keep_all_changed(self):
        """Called when keep all checkbox changes."""
        if self.on_selection_changed:
            self.on_selection_changed(self.group_index, self.keep_item.get())
    
    def select_by_strategy(self, strategy: str):
        """Auto-select item based on strategy."""
        logger.info(f"[DEDUP GROUP {self.group_index}] Applying strategy: '{strategy}', items: {self.group.items}")
        
        if not self.group.items:
            logger.warning(f"[DEDUP GROUP {self.group_index}] No items in group, skipping")
            return
        
        metadata_map = {}
        for item_id in self.group.items:
            meta = self.processor.get_item_metadata(item_id)
            if meta:
                metadata_map[item_id] = meta
                logger.debug(f"[DEDUP GROUP {self.group_index}] Item {item_id} metadata: FileSize={meta.get('FileSize')}, Created={meta.get('Created')}, DateTimeOriginal={meta.get('DateTimeOriginal')}")
            else:
                logger.warning(f"[DEDUP GROUP {self.group_index}] No metadata found for item {item_id}")
        
        selected = self.group.items[0]  # Default
        
        if strategy == "oldest":
            # Sort by date (ascending) - oldest first
            sorted_items = sorted(
                self.group.items,
                key=lambda x: metadata_map.get(x, {}).get('DateTimeOriginal', '') or metadata_map.get(x, {}).get('Created', '') or '9999'
            )
            selected = sorted_items[0]
            logger.info(f"[DEDUP GROUP {self.group_index}] Oldest strategy selected: {selected}")
        elif strategy == "newest":
            sorted_items = sorted(
                self.group.items,
                key=lambda x: metadata_map.get(x, {}).get('DateTimeOriginal', '') or metadata_map.get(x, {}).get('Created', '') or '',
                reverse=True
            )
            selected = sorted_items[0]
            logger.info(f"[DEDUP GROUP {self.group_index}] Newest strategy selected: {selected}")
        elif strategy == "largest":
            sorted_items = sorted(
                self.group.items,
                key=lambda x: metadata_map.get(x, {}).get('FileSize', 0) or 0,
                reverse=True
            )
            selected = sorted_items[0]
            logger.info(f"[DEDUP GROUP {self.group_index}] Largest strategy selected: {selected} (size: {metadata_map.get(selected, {}).get('FileSize', 'N/A')})")
        elif strategy == "smallest":
            sorted_items = sorted(
                self.group.items,
                key=lambda x: metadata_map.get(x, {}).get('FileSize', 0) or float('inf')
            )
            selected = sorted_items[0]
            logger.info(f"[DEDUP GROUP {self.group_index}] Smallest strategy selected: {selected} (size: {metadata_map.get(selected, {}).get('FileSize', 'N/A')})")
        
        
        # Ensure we have a valid selection before setting
        if selected:
            logger.info(f"[DEDUP GROUP {self.group_index}] Setting keep_item to: {selected}")
            self.keep_item.set(str(selected))
            self._on_selection_changed()
            
            # Force UI update by triggering the widget update
            # CustomTkinter radio buttons don't auto-update when variable changes programmatically
            self.update_idletasks()
        else:
            logger.warning(f"[DEDUP GROUP {self.group_index}] Strategy '{strategy}' failed to find a valid item")
    
    def get_decision(self) -> DedupDecision:
        """Get the dedup decision for this group based on user selection."""
        # If keep_all is checked, return empty remove list (false positive)
        if self.keep_all.get():
            return DedupDecision(
                keep_item=None,
                remove_items=[],
                reason="False positive - keep all"
            )
        
        keep = self.keep_item.get()
        remove = [item for item in self.group.items if item != keep]
        return DedupDecision(
            keep_item=keep,
            remove_items=remove,
            reason="User selection"
        )


class StepDedup(ctk.CTkFrame):
    """
    Deduplication wizard step.
    
    Allows users to scan for duplicates in their Daminion selection
    and apply deduplication actions.
    """
    
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.session = controller.session
        self.logger = logging.getLogger(__name__)
        
        self.processor: Optional[DaminionDedupProcessor] = None
        self.scan_result: Optional[DedupScanResult] = None
        self.group_frames: List[DuplicateGroupFrame] = []
        
        # Confirmation state for Apply button
        self.is_confirming_action = False
        self.default_btn_fg = None
        self.default_btn_hover = None
        self.default_btn_text = "Apply Deduplication"
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the deduplication UI."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.grid(row=0, column=0, sticky="nsew")
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_rowconfigure(3, weight=1)
        
        # ===== Header =====
        header_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        
        title = ctk.CTkLabel(
            header_frame,
            text="🔍 Duplicate Detection",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(side="left")
        
        back_btn = ctk.CTkButton(
            header_frame,
            text="← Back",
            width=80,
            command=self._go_back
        )
        back_btn.pack(side="right")
        
        # ===== Settings Panel =====
        settings_frame = ctk.CTkFrame(self.container)
        settings_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        settings_frame.grid_columnconfigure(0, weight=1)
        settings_frame.grid_columnconfigure(1, weight=2)
        settings_frame.grid_columnconfigure(2, weight=2)
        settings_frame.grid_columnconfigure(3, weight=0)

        algo_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        algo_frame.grid(row=0, column=0, padx=(15, 10), pady=(12, 6), sticky="ew")

        algo_label = ctk.CTkLabel(algo_frame, text="Algorithm:")
        algo_label.pack(anchor="w")

        self.algorithm_var = ctk.StringVar(value="phash")
        self.algorithm_dropdown = ctk.CTkOptionMenu(
            algo_frame,
            values=["phash", "dhash", "ahash", "whash"],
            variable=self.algorithm_var,
            width=120
        )
        self.algorithm_dropdown.pack(fill="x", pady=(6, 0))

        threshold_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        threshold_frame.grid(row=0, column=1, padx=10, pady=(12, 6), sticky="ew")
        threshold_frame.grid_columnconfigure(0, weight=1)
        threshold_frame.grid_columnconfigure(1, weight=0)

        threshold_label = ctk.CTkLabel(threshold_frame, text="Threshold:")
        threshold_label.grid(row=0, column=0, columnspan=2, sticky="w")

        self.threshold_var = ctk.DoubleVar(value=95.0)
        self.threshold_slider = ctk.CTkSlider(
            threshold_frame,
            from_=50,
            to=100,
            variable=self.threshold_var,
            command=self._on_threshold_change
        )
        self.threshold_slider.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        self.threshold_value_label = ctk.CTkLabel(threshold_frame, text="95%", width=50)
        self.threshold_value_label.grid(row=1, column=1, padx=(10, 0), pady=(6, 0), sticky="e")

        action_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        action_frame.grid(row=0, column=2, padx=10, pady=(12, 6), sticky="ew")

        action_label = ctk.CTkLabel(action_frame, text="Action:")
        action_label.pack(anchor="w")

        self.action_var = ctk.StringVar(value="Tag as Duplicate")
        self.action_dropdown = ctk.CTkOptionMenu(
            action_frame,
            values=["Tag as Duplicate", "Remove from Catalog", "Delete from Disk", "Manual Review Only"],
            variable=self.action_var,
            command=self._on_action_change
        )
        self.action_dropdown.pack(fill="x", pady=(6, 0))

        # Warning label for destructive actions (hidden initially)
        self.action_warning_label = ctk.CTkLabel(
            settings_frame, text="", font=ctk.CTkFont(size=10),
            text_color="red"
        )
        self.action_warning_label.grid(row=1, column=0, columnspan=3, padx=15, pady=(0, 10), sticky="w")

        scan_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        scan_frame.grid(row=0, column=3, rowspan=2, padx=(10, 15), pady=10, sticky="e")

        self.scan_btn = ctk.CTkButton(
            scan_frame,
            text="Scan for Duplicates",
            command=self._start_scan,
            fg_color=("green", "darkgreen"),
            hover_color=("darkgreen", "green"),
            width=160
        )
        self.scan_btn.pack(anchor="e", pady=8)
        
        # ===== Progress Bar (own row below settings) =====
        self.progress_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        self.progress_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=5)
        self.progress_frame.grid_remove()  # Hidden initially
        
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, width=400)
        self.progress_bar.pack(side="left", padx=10, fill="x", expand=True)
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(self.progress_frame, text="Scanning...")
        self.progress_label.pack(side="left", padx=10)
        
        self.abort_btn = ctk.CTkButton(
            self.progress_frame,
            text="Abort",
            width=80,
            fg_color=("red", "darkred"),
            command=self._abort_scan
        )
        self.abort_btn.pack(side="right", padx=10)
        
        # ===== Results Area =====
        results_container = ctk.CTkFrame(self.container)
        results_container.grid(row=3, column=0, sticky="nsew", padx=20, pady=10)
        results_container.grid_columnconfigure(0, weight=1)
        results_container.grid_rowconfigure(0, weight=1)
        
        # Scrollable frame for duplicate groups
        self.results_scroll = ctk.CTkScrollableFrame(results_container)
        self.results_scroll.grid(row=0, column=0, sticky="nsew")
        
        # Initial message
        self.initial_label = ctk.CTkLabel(
            self.results_scroll,
            text="Select your scan settings and click 'Scan for Duplicates' to begin.\n\n"
                 "The scan will analyze images from your current Daminion selection.",
            font=ctk.CTkFont(size=14),
            text_color=("gray50", "gray60"),
            justify="center"
        )
        self.initial_label.pack(pady=50)
        
        # ===== Footer =====
        footer_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        footer_frame.grid(row=4, column=0, sticky="ew", padx=20, pady=(10, 20))
        footer_frame.grid_columnconfigure(0, weight=1)
        footer_frame.grid_columnconfigure(1, weight=1)
        footer_frame.grid_columnconfigure(2, weight=0)

        self.stats_label = ctk.CTkLabel(
            footer_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray60")
        )
        self.stats_label.grid(row=0, column=0, sticky="w")

        # Auto-select buttons frame
        self.select_btns_frame = ctk.CTkFrame(footer_frame, fg_color="transparent")
        self.select_btns_frame.grid(row=0, column=1, padx=20)
        
        select_label = ctk.CTkLabel(self.select_btns_frame, text="Select all:", font=ctk.CTkFont(size=11))
        select_label.pack(side="left", padx=(0, 5))
        
        for strategy, label in [("oldest", "Oldest"), ("newest", "Newest"), ("largest", "Largest"), ("smallest", "Smallest")]:
            btn = ctk.CTkButton(
                self.select_btns_frame,
                text=label,
                width=70,
                height=26,
                font=ctk.CTkFont(size=11),
                fg_color=("gray60", "gray40"),
                hover_color=("gray50", "gray30"),
                command=lambda s=strategy: self._select_all_by_strategy(s)
            )
            btn.pack(side="left", padx=2)
        
        self.apply_btn = ctk.CTkButton(
            footer_frame,
            text="Apply Deduplication",
            command=self._apply_dedup,
            state="disabled",
            width=160
        )
        self.apply_btn.grid(row=0, column=2, sticky="e")
    
    def _on_threshold_change(self, value):
        """Update threshold label when slider changes."""
        self.threshold_value_label.configure(text=f"{int(value)}%")
    
    def _go_back(self):
        """Navigate back to Step 1."""
        self.controller.show_step("Step1Datasource")
    
    def _start_scan(self):
        """Start the duplicate scan in a background thread."""
        self._reset_confirmation()
        
        # Check if Daminion is connected
        if not self.session.daminion_client or not self.session.daminion_client.authenticated:
            messagebox.showerror("Error", "Please connect to Daminion first.")
            return
        
        # Check if there are items to scan
        if not hasattr(self.session, 'dedup_items') or not self.session.dedup_items:
            messagebox.showerror("Error", "No items selected for deduplication.\n\n"
                               "Please select a collection or search first.")
            return
        
        # Initialize processor
        self.processor = DaminionDedupProcessor(
            self.session.daminion_client,
            similarity_threshold=self.threshold_var.get()
        )
        
        # Show progress
        self.progress_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=5)
        self.progress_bar.set(0)
        self.progress_label.configure(text="Refreshing items from Daminion...")
        self.scan_btn.configure(state="disabled")
        
        # Clear previous results
        for frame in self.group_frames:
            frame.destroy()
        self.group_frames.clear()
        self.initial_label.pack_forget()
        
        # Re-fetch items from Daminion before scanning to get fresh data
        thread = threading.Thread(target=self._refresh_and_scan, daemon=True)
        thread.start()

    def _refresh_and_scan(self):
        """Re-fetch items from Daminion and then run the scan."""
        try:
            # Re-fetch the item list from Daminion to capture any changes
            # (e.g., items deleted/tagged in a previous round)
            self.after(0, lambda: self.progress_label.configure(text="Re-fetching items from Daminion..."))
            self._reload_base_query_items()
            
            # Now run the actual scan
            self._run_scan()
            
        except Exception as e:
            self.logger.error(f"Refresh and scan failed: {e}", exc_info=True)
            self.after(0, lambda: self._on_scan_error(str(e)))
    
    def _run_scan(self):
        """Run the scan in background thread."""
        try:
            algorithm = self.algorithm_var.get()
            items = self.session.dedup_items
            
            def progress_callback(message, current, total):
                self.after(0, lambda: self._update_progress(message, current, total))
            
            self.scan_result = self.processor.scan_for_duplicates(
                items,
                algorithm=algorithm,
                progress_callback=progress_callback
            )
            
            self.after(0, self._on_scan_complete)
            
        except Exception as e:
            self.logger.error(f"Scan failed: {e}", exc_info=True)
            self.after(0, lambda: self._on_scan_error(str(e)))
    
    def _update_progress(self, message: str, current: int, total: int):
        """Update progress bar and label."""
        self.progress_label.configure(text=message)
        if total > 0:
            self.progress_bar.set(current / total)
    
    def _abort_scan(self):
        """Abort the current scan."""
        if self.processor:
            self.processor.abort()
    
    def _on_scan_complete(self):
        """Called when scan completes successfully."""
        self.progress_frame.grid_remove()
        self.scan_btn.configure(state="normal")
        
        if not self.scan_result:
            return
        
        result = self.scan_result
        
        # Update stats
        self.stats_label.configure(
            text=f"Scanned {result.items_hashed}/{result.total_items} items | "
                 f"Found {len(result.duplicate_groups)} duplicate groups | "
                 f"Algorithm: {result.algorithm.upper()} | "
                 f"Threshold: {result.threshold:.0f}%"
        )
        
        if not result.duplicate_groups:
            no_dupes_label = ctk.CTkLabel(
                self.results_scroll,
                text="✓ No duplicates found!",
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=("green", "lightgreen")
            )
            no_dupes_label.pack(pady=50)
            self.apply_btn.configure(state="disabled")
            return
        
        # Display duplicate groups
        for idx, group in enumerate(result.duplicate_groups):
            group_frame = DuplicateGroupFrame(
                self.results_scroll,
                group,
                idx,
                self.processor
            )
            group_frame.pack(fill="x", padx=10, pady=5)
            self.group_frames.append(group_frame)
        
        self.apply_btn.configure(state="normal")
    
    def _select_all_by_strategy(self, strategy: str):
        """Apply a selection strategy to all duplicate groups."""
        logger.info(f"[DEDUP UI] Strategy button clicked: '{strategy}', group_frames count: {len(self.group_frames)}")
        
        if not self.group_frames:
            logger.warning("[DEDUP UI] No group frames to apply strategy to")
            return
        
        for group_frame in self.group_frames:
            group_frame.select_by_strategy(strategy)
        
        # Update status to show strategy was applied
        self.stats_label.configure(
            text=f"Strategy '{strategy.capitalize()}' applied to {len(self.group_frames)} groups | "
                 f"Review selections and click 'Apply Deduplication' to proceed"
        )
        logger.info(f"[DEDUP UI] Strategy '{strategy}' applied to {len(self.group_frames)} groups")
    
    def _on_scan_error(self, error_message: str):
        """Called when scan fails with an error."""
        self.progress_frame.grid_remove()
        self.scan_btn.configure(state="normal")
        # Show error in initial label
        self.initial_label.configure(text=f"Scan Error:\n{error_message}", text_color="red")
        self.initial_label.pack(pady=50)

    
    def _reset_apply_button(self):
        """Reset the apply button to its default state."""
        try:
            default_color = ctk.ThemeManager.theme["CTkButton"]["fg_color"]
        except Exception:
            default_color = ["#3a7ebf", "#1f538d"]
        self.apply_btn.configure(text="Apply Deduplication", fg_color=default_color)
    
    def _reset_confirmation(self, *args):
        """Reset the apply button to its default state."""
        if self.is_confirming_action:
            if self.default_btn_fg:
                self.apply_btn.configure(fg_color=self.default_btn_fg)
            if self.default_btn_hover:
                self.apply_btn.configure(hover_color=self.default_btn_hover)
            
            self.apply_btn.configure(text=self.default_btn_text)
            self.is_confirming_action = False

    def _on_action_change(self, choice):
        """Called when action dropdown changes."""
        self._reset_confirmation()
        # Show/hide warning for destructive actions
        if choice == "Delete from Disk":
            self.action_warning_label.configure(
                text="⚠️ WARNING: This will permanently delete files from disk!"
            )
        elif choice == "Remove from Catalog":
            self.action_warning_label.configure(
                text="Files will remain on disk."
            )
        else:
            self.action_warning_label.configure(text="")

    def _apply_dedup(self):
        """Apply deduplication actions based on user selections."""
        logger.info(f"[DEDUP APPLY] Apply button clicked. Processor: {self.processor is not None}, Group frames: {len(self.group_frames) if self.group_frames else 0}")
        
        if not self.processor or not self.group_frames:
            return
        
        # Get action
        action_text = self.action_var.get()
        logger.info(f"[DEDUP APPLY] Action selected: '{action_text}'")
        
        if action_text == "Tag as Duplicate":
            action = DedupAction.TAG
        elif action_text == "Remove from Catalog":
            action = DedupAction.REMOVE
        elif action_text == "Delete from Disk":
            action = DedupAction.DELETE
        else:
            action = DedupAction.NONE
            
        # Handle Manual Review (No-op / Info only)
        if action == DedupAction.NONE:
            total_remove = sum(len(f.get_decision().remove_items) for f in self.group_frames)
            messagebox.showinfo("Manual Review", 
                              f"You have selected {total_remove} items as duplicates.\n\n"
                              "No automatic action will be taken.")
            return

        # --- Two-Step Confirmation Logic ---
        if not self.is_confirming_action:
            # First click: Change button to red "Confirm" state
            self.default_btn_fg = self.apply_btn.cget("fg_color")
            self.default_btn_hover = self.apply_btn.cget("hover_color")
            # Save current text just in case, though we default to "Apply Deduplication"
            self.default_btn_text = "Apply Deduplication" 
            
            self.apply_btn.configure(
                text="Confirm to proceed",
                fg_color="red",
                hover_color="darkred"
            )
            self.is_confirming_action = True
            return

        # Second click: Execute Action
        # Reset button state first
        self._reset_confirmation()
        
        # Collect decisions
        decisions = [frame.get_decision() for frame in self.group_frames]
        total_remove = sum(len(d.remove_items) for d in decisions)
        logger.info(f"[DEDUP APPLY] Executing action {action} on {total_remove} items")
        
        # --- Run on background thread with progress ---
        # Disable controls while processing
        self.apply_btn.configure(state="disabled", text="Processing...")
        self.scan_btn.configure(state="disabled")
        self.action_dropdown.configure(state="disabled")

        # Show progress bar
        self.progress_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=5)
        self.progress_bar.set(0)
        self.progress_label.configure(text="Applying deduplication...")
        # Hide abort button during apply (processor.abort() could be wired but keep it simple)
        self.abort_btn.configure(state="disabled")

        # Launch background thread
        thread = threading.Thread(
            target=self._run_apply_action,
            args=(decisions, action),
            daemon=True
        )
        thread.start()

    def _run_apply_action(self, decisions, action):
        """Run the dedup action in a background thread."""
        try:
            def progress_callback(message, current, total):
                self.after(0, lambda m=message, c=current, t=total: self._update_apply_progress(m, c, t))

            results = self.processor.apply_dedup_action(
                decisions, action, progress_callback=progress_callback
            )
            
            self.after(0, lambda: self._on_apply_complete(results))
            
        except Exception as e:
            self.logger.error(f"Apply dedup failed: {e}", exc_info=True)
            self.after(0, lambda err=str(e): self._on_apply_error(err))

    def _update_apply_progress(self, message: str, current: int, total: int):
        """Update progress bar during dedup action."""
        self.progress_label.configure(text=message)
        if total > 0:
            self.progress_bar.set(current / total)

    def _on_apply_complete(self, results):
        """Called on main thread when the dedup action finishes."""
        # Hide progress, re-enable controls
        self.progress_frame.grid_remove()
        self.scan_btn.configure(state="normal")
        self.action_dropdown.configure(state="normal")
        self.abort_btn.configure(state="normal")
        self.apply_btn.configure(state="normal", text="Apply Deduplication")
        
        msg = f"Deduplication complete!\n\n"
        if results.get('tagged', 0) > 0:
            msg += f"• Tagged: {results['tagged']} items\n"
        if results.get('deleted', 0) > 0:
            msg += f"• Deleted: {results['deleted']} items\n"
        if results.get('errors', 0) > 0:
            msg += f"• Errors: {results['errors']} items\n"
        if results.get('skipped', 0) > 0:
            msg += f"• Skipped: {results['skipped']} items\n"
        
        messagebox.showinfo("Deduplication Complete", msg)

        # Always reload the base query and rescan after successful apply so the
        # user can continue deduping on fresh results in the same step.
        self.progress_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=5)
        self.progress_bar.set(0)
        self.progress_label.configure(text="Reloading base query from Daminion...")
        self.abort_btn.configure(state="disabled")
        self.scan_btn.configure(state="disabled")
        self.action_dropdown.configure(state="disabled")
        self.apply_btn.configure(state="disabled")
        threading.Thread(
            target=self._reload_and_rescan_after_apply,
            daemon=True,
        ).start()

    def _on_apply_error(self, error_message: str):
        """Called on main thread when the dedup action fails."""
        self.progress_frame.grid_remove()
        self.scan_btn.configure(state="normal")
        self.action_dropdown.configure(state="normal")
        self.abort_btn.configure(state="normal")
        self.apply_btn.configure(state="normal", text="Apply Deduplication")
        messagebox.showerror("Error", f"Failed to apply deduplication:\n\n{error_message}")

    def _reload_and_rescan_after_apply(self):
        """Reload base query and immediately rescan to keep execution step fresh."""
        try:
            fresh_items = self._reload_base_query_items()
            count = len(fresh_items) if fresh_items else 0

            if count <= 0:
                def _finish_empty():
                    self.progress_frame.grid_remove()
                    self.abort_btn.configure(state="normal")
                    self.scan_btn.configure(state="normal")
                    self.action_dropdown.configure(state="normal")
                    self.apply_btn.configure(state="disabled")
                    for frame in self.group_frames:
                        frame.destroy()
                    self.group_frames.clear()
                    self.scan_result = None
                    self.initial_label.configure(
                        text=(
                            "Base query reloaded, but no items remain.\n\n"
                            "Adjust Step 1 filters or limits and scan again."
                        )
                    )
                    self.initial_label.pack(pady=50)
                    self.stats_label.configure(text="Base query reloaded: 0 items.")

                self.after(0, _finish_empty)
                return

            def _prepare_rescan():
                self.progress_label.configure(
                    text=f"Base query reloaded ({count} items). Rescanning duplicates..."
                )
                self.progress_bar.set(0)
                self.progress_frame.grid_remove()
                self.abort_btn.configure(state="normal")
                for frame in self.group_frames:
                    frame.destroy()
                self.group_frames.clear()
                self.scan_result = None
                self.initial_label.pack_forget()
                self.progress_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=5)
                self.progress_label.configure(
                    text=f"Scanning refreshed result set ({count} items)..."
                )

            self.after(0, _prepare_rescan)

            # Recreate processor to ensure threshold is honored for follow-up scan.
            self.processor = DaminionDedupProcessor(
                self.session.daminion_client,
                similarity_threshold=self.threshold_var.get()
            )
            self._run_scan()
        except Exception as e:
            logger.error(f"[DEDUP REFRESH] Post-apply reload/rescan failed: {e}", exc_info=True)
            self.after(0, lambda: self.progress_frame.grid_remove())
            self.after(0, lambda: self.abort_btn.configure(state="normal"))
            self.after(0, lambda: self.scan_btn.configure(state="normal"))
            self.after(0, lambda: self.action_dropdown.configure(state="normal"))
            self.after(0, lambda: self.apply_btn.configure(state="normal", text="Apply Deduplication"))

    def _reload_base_query_items(self) -> List[Dict]:
        """Reload current Step 1 base query into session.dedup_items."""
        ds = self.session.datasource
        client = self.session.daminion_client

        if not client or ds.type != "daminion":
            return list(getattr(self.session, "dedup_items", []) or [])

        scope = getattr(ds, "daminion_scope", "all") or "all"
        status = getattr(ds, "status_filter", "all") or "all"
        ss_id = None
        col_id = None
        search_term = None

        if scope == "saved_search":
            ss_id = getattr(ds, "daminion_saved_search_id", None)
        elif scope == "collection":
            col_id = getattr(ds, "daminion_collection_id", None)
        elif scope == "search":
            search_term = getattr(ds, "daminion_search_term", None)

        untagged = []
        if getattr(ds, "daminion_untagged_keywords", False):
            untagged.append("Keywords")
        if getattr(ds, "daminion_untagged_categories", False):
            untagged.append("Category")
        if getattr(ds, "daminion_untagged_description", False):
            untagged.append("Description")

        process_limit = getattr(ds, "max_items", 0) or 0
        fetch_limit = 500
        if process_limit > 0:
            fetch_limit = min(process_limit, 500)

        logger.info(
            f"[DEDUP REFRESH] Reloading base query: scope={scope}, ss_id={ss_id}, "
            f"col_id={col_id}, status={status}, limit={fetch_limit}"
        )
        fresh_items = client.get_items_filtered(
            scope=scope,
            saved_search_id=ss_id,
            collection_id=col_id,
            search_term=search_term,
            untagged_fields=untagged,
            status_filter=status,
            max_items=fetch_limit,
        )

        if fresh_items:
            old_count = len(self.session.dedup_items) if getattr(self.session, "dedup_items", None) else 0
            self.session.dedup_items = fresh_items
            logger.info(f"[DEDUP REFRESH] Refreshed base query: {old_count} -> {len(fresh_items)} items")
            return fresh_items

        logger.warning("[DEDUP REFRESH] Reload returned no items, keeping existing set")
        return list(getattr(self.session, "dedup_items", []) or [])
    
    def set_items(self, items: List[Dict]):
        """Set the items to scan for deduplication."""
        self.session.dedup_items = items
        
        # Update initial label
        self.initial_label.configure(
            text=f"Ready to scan {len(items)} items for duplicates.\n\n"
                 "Configure settings above and click 'Scan for Duplicates' to begin."
        )
        self.initial_label.pack(pady=50)
    
    def refresh(self):
        """Refresh the step when navigated to."""
        # Check if we have items
        if hasattr(self.session, 'dedup_items') and self.session.dedup_items:
            count = len(self.session.dedup_items)
            self.initial_label.configure(
                text=f"Ready to scan {count} items for duplicates.\n\n"
                     "Configure settings above and click 'Scan for Duplicates' to begin."
            )
    
    def shutdown(self):
        """Clean up on application shutdown."""
        if self.processor:
            self.processor.abort()
