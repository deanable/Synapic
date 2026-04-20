import logging
import threading
import time
import traceback
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk

from src.core.upscaler import (
    WORKFLOW_BALANCED,
    WORKFLOW_FAST,
    WORKFLOW_QUALITY,
    Swin2SRUpscaler,
    UpscaleOptions,
)
from src.utils.background_worker import BackgroundWorker

logger = logging.getLogger(__name__)

class StepUpscale(ctk.CTkFrame):
    """
    UI component for upscaling filtered Daminion images in paginated batches.
    """

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self._worker = BackgroundWorker(name="UpscaleWorker")
        self._upscaler = Swin2SRUpscaler()
        self._stop_event = threading.Event()
        self._is_running = False

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        ctk.CTkLabel(
            header, text="Daminion Upscale", font=("Roboto", 24, "bold")
        ).pack(side="left")

        # Main Content Area
        content_frame = ctk.CTkFrame(self)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        content_frame.grid_rowconfigure(10, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)

        # Status Label
        self.lbl_status = ctk.CTkLabel(
            content_frame,
            text="Ready to upscale.",
            font=("Roboto", 14)
        )
        self.lbl_status.pack(pady=(20, 8))

        self.lbl_task = ctk.CTkLabel(
            content_frame,
            text="Current task: Idle",
            font=("Roboto", 13),
            text_color="gray70",
        )
        self.lbl_task.pack(pady=(0, 10))

        self.progress_bar = ctk.CTkProgressBar(content_frame, height=18)
        self.progress_bar.pack(fill="x", padx=20, pady=(0, 10))
        self.progress_bar.set(0)

        self.lbl_counter = ctk.CTkLabel(
            content_frame,
            text="0 / 0 images",
            font=("Roboto", 13),
        )
        self.lbl_counter.pack(pady=(0, 4))

        self.lbl_eta = ctk.CTkLabel(
            content_frame,
            text="ETA: --",
            font=("Roboto", 12),
            text_color="#E8A838",
        )
        self.lbl_eta.pack(pady=(0, 14))

        # Upscale Settings
        settings_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        settings_frame.pack(fill="x", padx=20, pady=10)
        settings_frame.grid_columnconfigure(0, weight=1)
        settings_frame.grid_columnconfigure(1, weight=1)
        settings_frame.grid_columnconfigure(2, weight=1)
        settings_frame.grid_columnconfigure(3, weight=1)
        settings_frame.grid_columnconfigure(4, weight=1)
        settings_frame.grid_columnconfigure(5, weight=1)

        ctk.CTkLabel(settings_frame, text="Workflow:").grid(row=0, column=0, padx=6, sticky="w")
        self.workflow_var = ctk.StringVar(value=WORKFLOW_QUALITY)
        self.workflow_dropdown = ctk.CTkComboBox(
            settings_frame,
            values=[WORKFLOW_QUALITY, WORKFLOW_BALANCED, WORKFLOW_FAST],
            variable=self.workflow_var,
            width=140,
            command=self._on_workflow_change,
        )
        self.workflow_dropdown.grid(row=0, column=1, padx=6, sticky="w")

        ctk.CTkLabel(settings_frame, text="Upscale Factor:").grid(row=0, column=2, padx=6, sticky="w")
        self.factor_var = ctk.StringVar(value="2x")
        self.factor_dropdown = ctk.CTkComboBox(
            settings_frame,
            values=["2x", "4x"],
            variable=self.factor_var,
            width=100
        )
        self.factor_dropdown.grid(row=0, column=3, padx=6, sticky="w")

        ctk.CTkLabel(settings_frame, text="Precision:").grid(row=0, column=4, padx=6, sticky="w")
        self.precision_var = ctk.StringVar(value="auto")
        self.precision_dropdown = ctk.CTkComboBox(
            settings_frame,
            values=["auto", "fp16", "fp32"],
            variable=self.precision_var,
            width=100,
        )
        self.precision_dropdown.grid(row=0, column=5, padx=6, sticky="w")

        ctk.CTkLabel(settings_frame, text="Output:").grid(row=1, column=0, padx=6, pady=(8, 0), sticky="w")
        self.output_format_var = ctk.StringVar(value="keep")
        self.output_format_dropdown = ctk.CTkComboBox(
            settings_frame,
            values=["keep", "JPEG", "PNG", "WEBP"],
            variable=self.output_format_var,
            width=140,
        )
        self.output_format_dropdown.grid(row=1, column=1, padx=6, pady=(8, 0), sticky="w")

        ctk.CTkLabel(settings_frame, text="Quality:").grid(row=1, column=2, padx=6, pady=(8, 0), sticky="w")
        self.quality_var = ctk.StringVar(value="95")
        self.quality_entry = ctk.CTkEntry(settings_frame, textvariable=self.quality_var, width=80)
        self.quality_entry.grid(row=1, column=3, padx=6, pady=(8, 0), sticky="w")

        ctk.CTkLabel(settings_frame, text="Denoise:").grid(row=1, column=4, padx=6, pady=(8, 0), sticky="w")
        self.denoise_var = ctk.DoubleVar(value=1.0)
        self.denoise_slider = ctk.CTkSlider(
            settings_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=20,
            variable=self.denoise_var,
        )
        self.denoise_slider.grid(row=1, column=5, padx=6, pady=(8, 0), sticky="ew")

        ctk.CTkLabel(settings_frame, text="Sharpen:").grid(row=2, column=0, padx=6, pady=(8, 0), sticky="w")
        self.sharpen_var = ctk.DoubleVar(value=0.0)
        self.sharpen_slider = ctk.CTkSlider(
            settings_frame,
            from_=0.0,
            to=2.0,
            number_of_steps=20,
            variable=self.sharpen_var,
        )
        self.sharpen_slider.grid(row=2, column=1, columnspan=2, padx=6, pady=(8, 0), sticky="ew")

        self.overwrite_var = ctk.BooleanVar(value=True)
        self.overwrite_checkbox = ctk.CTkCheckBox(
            settings_frame,
            text="Overwrite existing output",
            variable=self.overwrite_var,
        )
        self.overwrite_checkbox.grid(row=2, column=3, columnspan=3, padx=6, pady=(8, 0), sticky="w")

        # Action Buttons
        btn_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        btn_frame.pack(pady=30)

        self.btn_upscale = ctk.CTkButton(
            btn_frame,
            text="Run Upscale Batch",
            fg_color=("purple", "#6b21a8"),
            hover_color=("darkviolet", "#7c3aed"),
            width=200,
            command=self.start_upscale,
        )
        self.btn_upscale.pack(side="left", padx=10)

        self.btn_stop = ctk.CTkButton(
            btn_frame,
            text="Stop",
            fg_color=("red", "#991b1b"),
            hover_color=("darkred", "#7f1d1d"),
            width=120,
            state="disabled",
            command=self.stop_upscale,
        )
        self.btn_stop.pack(side="left", padx=10)

        ctk.CTkLabel(content_frame, text="Execution Log:", anchor="w").pack(
            fill="x", padx=20, pady=(10, 0)
        )
        self.console = ctk.CTkTextbox(content_frame, font=("Consolas", 12), height=180)
        self.console.pack(fill="both", expand=True, padx=20, pady=(5, 20))
        self.console.insert("0.0", "--- Upscale log initialized ---\n")
        self.console.configure(state="disabled")

        # Navigation
        nav_frame = ctk.CTkFrame(self, fg_color="transparent")
        nav_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=20)

        ctk.CTkButton(
            nav_frame,
            text="Back",
            command=lambda: self.controller.show_step("Step1Datasource"),
            width=120,
            fg_color="transparent",
            border_width=1,
            text_color=("black", "white"),
        ).pack(side="left")
        self._on_workflow_change(self.workflow_var.get())

    def start_upscale(self):
        """
        Initiates the paginated upscaling process in a background thread.
        """
        client = self.controller.session.daminion_client
        if not client or not client.authenticated:
            messagebox.showerror("Error", "Not connected to Daminion.")
            return

        self._stop_event.clear()
        self._is_running = True
        self.btn_upscale.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.factor_dropdown.configure(state="disabled")
        self.workflow_dropdown.configure(state="disabled")
        self.precision_dropdown.configure(state="disabled")
        self.output_format_dropdown.configure(state="disabled")
        self.quality_entry.configure(state="disabled")
        self.denoise_slider.configure(state="disabled")
        self.sharpen_slider.configure(state="disabled")
        self.overwrite_checkbox.configure(state="disabled")
        self.progress_bar.set(0)
        self.lbl_counter.configure(text="0 / 0 images")
        self.lbl_eta.configure(text="ETA: --")
        self._update_task("Preparing batch...")
        self._clear_log()
        self._log_event("Upscale run started.")

        factor_str = self.factor_var.get()
        factor = int(factor_str.replace("x", ""))
        options = self._build_upscale_options()
        self._log_event(
            "Parameters: "
            f"workflow={options.workflow}, factor={factor}x, precision={options.precision}, "
            f"output={options.output_format}, quality={options.jpeg_quality}, "
            f"denoise={options.denoise_strength:.2f}, sharpen={options.sharpen_amount:.2f}, "
            f"overwrite={options.overwrite_existing}"
        )
        self._update_status(f"Starting batch upscaling at {factor}x...")

        # Run in worker to avoid blocking UI.
        self._worker.submit(self._run_upscale_batch, factor, options)

    def stop_upscale(self):
        """Request graceful cancellation of the current upscale run."""
        if not self._is_running:
            return
        self._stop_event.set()
        self.btn_stop.configure(state="disabled")
        self._update_status("Stopping after current item...")
        self._update_task("Cancellation requested...")
        self._log_event("Cancellation requested by user.")

    def _run_upscale_batch(self, factor: int, options: UpscaleOptions) -> None:
        """
        Process the current Step 1 Daminion result set across API pages.
        For each item: checkout -> download -> upscale -> checkin.
        """
        client = self.controller.session.daminion_client
        ds = self.controller.session.datasource

        scope = ds.daminion_scope or "all"
        saved_search_id = ds.daminion_saved_search_id or None
        collection_id = ds.daminion_collection_id or None
        search_term = ds.daminion_search_term or None
        status_filter = ds.status_filter or "all"
        untagged_fields = []
        if ds.daminion_untagged_keywords:
            untagged_fields.append("Keywords")
        if ds.daminion_untagged_categories:
            untagged_fields.append("Category")
        if ds.daminion_untagged_description:
            untagged_fields.append("Description")

        process_limit = ds.max_items if ds.max_items and ds.max_items > 0 else None
        expected_total = self._resolve_expected_total(
            client=client,
            scope=scope,
            saved_search_id=saved_search_id,
            collection_id=collection_id,
            search_term=search_term,
            untagged_fields=untagged_fields,
            status_filter=status_filter,
            process_limit=process_limit,
        )
        start_index = 0
        page_size = 500
        page_num = 0
        processed_count = 0
        success_count = 0
        failed_count = 0
        started_at = time.monotonic()
        self._log_event(
            f"Resolved datasource: scope={scope}, saved_search_id={saved_search_id}, "
            f"collection_id={collection_id}, status={status_filter}, "
            f"process_limit={process_limit or 'all'}, expected_total={expected_total or 'unknown'}"
        )

        try:
            while True:
                if self._stop_event.is_set():
                    break
                if process_limit is not None and processed_count >= process_limit:
                    break

                fetch_max = page_size
                if process_limit is not None:
                    fetch_max = min(fetch_max, process_limit - processed_count)
                    if fetch_max <= 0:
                        break

                page_num += 1
                self._update_status(f"Loading page {page_num}...")
                self._update_task("Fetching page from Daminion...")
                items = client.get_items_filtered(
                    scope=scope,
                    saved_search_id=saved_search_id,
                    collection_id=collection_id,
                    search_term=search_term,
                    untagged_fields=untagged_fields,
                    status_filter=status_filter,
                    max_items=fetch_max,
                    start_index=start_index,
                )

                if not items:
                    self._log_event(f"Page {page_num}: no items returned, stopping pagination.")
                    break
                self._log_event(f"Page {page_num}: loaded {len(items)} item(s) from start_index={start_index}.")

                if expected_total <= 0:
                    expected_total = max(expected_total, processed_count + len(items))

                for item in items:
                    if self._stop_event.is_set():
                        break
                    item_id = item.get("id") or item.get("Id")
                    if not item_id:
                        failed_count += 1
                        processed_count += 1
                        self._update_progress(
                            processed_count=processed_count,
                            total_count=expected_total,
                            started_at=started_at,
                        )
                        continue

                    if self._process_single_item(item_id, factor, options):
                        success_count += 1
                    else:
                        failed_count += 1
                    processed_count += 1
                    self._update_progress(
                        processed_count=processed_count,
                        total_count=expected_total,
                        started_at=started_at,
                    )

                    if process_limit is not None and processed_count >= process_limit:
                        break

                # Pagination rule: partial page means no more data
                if self._stop_event.is_set() or len(items) < page_size:
                    if len(items) < page_size:
                        self._log_event(
                            f"Page {page_num}: received partial page ({len(items)} < {page_size}), reached final page."
                        )
                    break
                start_index += page_size

            self.after(
                0,
                lambda: self._on_upscale_complete(
                    {
                        "processed": processed_count,
                        "success": success_count,
                        "failed": failed_count,
                        "cancelled": self._stop_event.is_set(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Upscale batch error: {e}", exc_info=True)
            self._log_event(f"Batch error: {e}")
            self._log_event(traceback.format_exc())
            self.after(0, lambda err=str(e): self._on_upscale_error(err))

    def _process_single_item(self, item_id: int, factor: int, options: UpscaleOptions) -> bool:
        """Run checkout -> download -> upscale -> checkin for one item."""
        client = self.controller.session.daminion_client

        if self._stop_event.is_set():
            self._log_event(f"Item {item_id}: skipped due to cancellation.")
            return False

        self._update_task(f"Item {item_id}: checking out...")
        self._update_status(f"Item {item_id}: checking out...")
        self._log_event(f"Item {item_id}: checkout started.")
        success = client.checkout_item(item_id)
        if not success:
            logger.error(f"Checkout failed for item {item_id}")
            self._log_event(f"Item {item_id}: checkout failed.")
            return False
        self._log_event(f"Item {item_id}: checkout successful.")

        try:
            if self._stop_event.is_set():
                self._log_event(f"Item {item_id}: cancelled before download.")
                return False
            self._update_task(f"Item {item_id}: downloading...")
            self._update_status(f"Item {item_id}: downloading original...")
            original_path = client.download_original(item_id)
            if not original_path:
                self._update_status(f"Item {item_id}: download failed.")
                self._log_event(f"Item {item_id}: download failed.")
                return False
            self._log_event(f"Item {item_id}: downloaded original to '{original_path}'.")

            if self._stop_event.is_set():
                self._log_event(f"Item {item_id}: cancelled before upscale.")
                return False
            self._update_task(f"Item {item_id}: upscaling...")
            self._update_status(f"Item {item_id}: upscaling {factor}x...")
            in_size = self._probe_image_size(original_path)
            if in_size:
                self._log_event(f"Item {item_id}: input size {in_size[0]}x{in_size[1]}.")
            upscaled_path = self._upscaler.upscale(
                input_path=original_path,
                factor=factor,
                status_callback=self._on_upscaler_status,
                options=options,
            )
            self._log_event(f"Item {item_id}: upscaled output generated at '{upscaled_path}'.")
            try:
                upscaled_file_size = int(upscaled_path.stat().st_size)
            except Exception:
                upscaled_file_size = -1
            if upscaled_file_size <= 0:
                self._log_event(
                    f"Item {item_id}: output file missing or empty at '{upscaled_path}'."
                )
                return False
            self._log_event(f"Item {item_id}: output file size {upscaled_file_size} bytes.")
            out_size = self._probe_image_size(upscaled_path)
            if out_size:
                self._log_event(f"Item {item_id}: output size {out_size[0]}x{out_size[1]}.")
            if in_size and out_size and out_size[0] <= in_size[0]:
                self._log_event(
                    f"Item {item_id}: WARNING output width {out_size[0]} did not increase from {in_size[0]}."
                )

            if self._stop_event.is_set():
                self._log_event(f"Item {item_id}: cancelled before check-in.")
                return False
            self._update_task(f"Item {item_id}: checking back in...")
            self._update_status(f"Item {item_id}: checking in...")
            checkin_msg = self._build_checkin_summary(options, factor)
            self._log_event(f"Item {item_id}: check-in message: {checkin_msg}")
            success = client.checkin_item(item_id, str(upscaled_path), message=checkin_msg)
            if success:
                self._flush_upscale_cache(item_id, original_path, upscaled_path)
                self._update_status(f"Item {item_id}: check-in complete.")
                self._log_event(f"Item {item_id}: check-in complete.")
                return True
            self._update_status(f"Item {item_id}: check-in failed.")
            self._log_event(f"Item {item_id}: check-in failed.")
            return False

        except Exception as e:
            logger.error(f"Upscale workflow error for item {item_id}: {e}", exc_info=True)
            self._update_status(f"Item {item_id}: error - {e}")
            self._log_event(f"Item {item_id}: error - {e}")
            self._log_event(traceback.format_exc())
            return False

    def _update_status(self, text: str):
        # Update status label from background thread safely
        if self.winfo_exists():
            self.after(0, lambda: self.lbl_status.configure(text=text))

    def _update_task(self, text: str):
        if self.winfo_exists():
            self.after(0, lambda: self.lbl_task.configure(text=f"Current task: {text}"))

    def _on_upscaler_status(self, text: str):
        self._update_status(text)
        self._log_event(text)

    def _update_progress(self, processed_count: int, total_count: int, started_at: float):
        if not self.winfo_exists():
            return
        total = max(total_count, 1)
        pct = min(processed_count / total, 1.0)
        elapsed = max(time.monotonic() - started_at, 0.0)
        if total_count > 0 and processed_count > 0:
            remaining = max(total_count - processed_count, 0)
            eta_seconds = elapsed / processed_count * remaining
            eta_txt = (
                f"ETA: {self._format_duration(eta_seconds)} "
                f"({self._format_duration(elapsed / processed_count)} / image)"
            )
        else:
            eta_txt = "ETA: --"

        self.after(0, lambda: self.progress_bar.set(pct))
        if total_count > 0:
            self.after(0, lambda: self.lbl_counter.configure(text=f"{processed_count} / {total_count} images"))
        else:
            self.after(0, lambda: self.lbl_counter.configure(text=f"{processed_count} images processed"))
        self.after(0, lambda: self.lbl_eta.configure(text=eta_txt))

    def _format_duration(self, seconds: float) -> str:
        seconds = max(int(seconds), 0)
        if seconds > 48 * 3600:
            days, remainder = divmod(seconds, 86400)
            hours, remainder = divmod(remainder, 3600)
            mins, _ = divmod(remainder, 60)
            return f"{days} days, {hours} hours, {mins} minutes"
        hours, remainder = divmod(seconds, 3600)
        mins, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {mins}m"
        return f"{mins}m {secs}s"

    def _resolve_expected_total(
        self,
        client,
        scope: str,
        saved_search_id,
        collection_id,
        search_term,
        untagged_fields,
        status_filter: str,
        process_limit,
    ) -> int:
        """Best-effort total count used for progress/ETA."""
        try:
            total = client.get_filtered_item_count(
                scope=scope,
                saved_search_id=saved_search_id,
                collection_id=collection_id,
                search_term=search_term,
                untagged_fields=untagged_fields,
                status_filter=status_filter,
            )
            if isinstance(total, int) and total > 0:
                if process_limit is not None:
                    return min(total, process_limit)
                return total
        except Exception as e:
            logger.warning(f"Failed to resolve expected total for upscaling: {e}")

        if process_limit is not None:
            return process_limit
        return 0

    def _on_upscale_complete(self, summary: dict):
        """
        Callback when worker thread finishes.
        """
        if self.winfo_exists():
            self._is_running = False
            self.btn_upscale.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.factor_dropdown.configure(state="normal")
            self.workflow_dropdown.configure(state="normal")
            self.precision_dropdown.configure(state="normal")
            self.output_format_dropdown.configure(state="normal")
            self.quality_entry.configure(state="normal")
            self.denoise_slider.configure(state="normal")
            self.sharpen_slider.configure(state="normal")
            self.overwrite_checkbox.configure(state="normal")
            processed = summary.get("processed", 0)
            success = summary.get("success", 0)
            failed = summary.get("failed", 0)
            cancelled = summary.get("cancelled", False)
            self.lbl_status.configure(
                text=f"Done. Processed {processed}, succeeded {success}, failed {failed}."
            )
            self.lbl_task.configure(text="Current task: Complete")
            if processed > 0:
                self.progress_bar.set(1.0)
            self._log_event(
                f"Run finished. processed={processed}, succeeded={success}, failed={failed}, cancelled={cancelled}"
            )
            messagebox.showinfo(
                "Upscale Stopped" if cancelled else "Upscale Complete",
                f"Processed: {processed}\nSucceeded: {success}\nFailed: {failed}",
            )

    def _on_upscale_error(self, error_message: str):
        if self.winfo_exists():
            self._is_running = False
            self.btn_upscale.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.factor_dropdown.configure(state="normal")
            self.workflow_dropdown.configure(state="normal")
            self.precision_dropdown.configure(state="normal")
            self.output_format_dropdown.configure(state="normal")
            self.quality_entry.configure(state="normal")
            self.denoise_slider.configure(state="normal")
            self.sharpen_slider.configure(state="normal")
            self.overwrite_checkbox.configure(state="normal")
            self.lbl_status.configure(text=f"Upscale batch failed: {error_message}")
            self.lbl_task.configure(text="Current task: Error")
            self._log_event(f"Run failed: {error_message}")
            messagebox.showerror("Upscale Error", error_message)

    def refresh(self):
        """Refresh summary text whenever this step is shown."""
        ds = self.controller.session.datasource
        seed_items = getattr(self.controller.session, "upscale_items", [])
        scope = ds.daminion_scope or "all"
        process_limit = ds.max_items if ds.max_items and ds.max_items > 0 else "all"
        self.lbl_status.configure(
            text=(
                f"Ready. Scope: {scope}. "
                f"Initial loaded items: {len(seed_items)}. "
                f"Will process: {process_limit}."
            )
        )
        self.lbl_task.configure(text="Current task: Idle")
        self.progress_bar.set(0)
        self.lbl_counter.configure(text="0 / 0 images")
        self.lbl_eta.configure(text="ETA: --")
        self._on_workflow_change(self.workflow_var.get())
        self._log_event("Upscale view refreshed.")

    def _build_upscale_options(self) -> UpscaleOptions:
        quality = 95
        try:
            quality = int(float(self.quality_var.get()))
        except Exception:
            quality = 95
        quality = max(70, min(100, quality))

        return UpscaleOptions(
            workflow=self.workflow_var.get().strip().lower(),
            precision=self.precision_var.get().strip().lower(),
            denoise_strength=float(self.denoise_var.get()),
            sharpen_amount=float(self.sharpen_var.get()),
            output_format=self.output_format_var.get().strip(),
            jpeg_quality=quality,
            overwrite_existing=bool(self.overwrite_var.get()),
        )

    def _on_workflow_change(self, workflow: str):
        wf = (workflow or WORKFLOW_QUALITY).strip().lower()
        if wf == WORKFLOW_BALANCED:
            self.denoise_slider.configure(state="normal")
            self.precision_dropdown.configure(state="normal")
        elif wf == WORKFLOW_FAST:
            self.denoise_slider.configure(state="disabled")
            self.precision_dropdown.configure(state="disabled")
        else:
            self.denoise_slider.configure(state="disabled")
            self.precision_dropdown.configure(state="normal")

    def shutdown(self):
        """Cleanup worker."""
        self._stop_event.set()
        if self._worker:
            self._worker.shutdown()

    def _build_checkin_summary(self, options: UpscaleOptions, factor: int) -> str:
        output_fmt = options.output_format.upper() if options.output_format else "KEEP"
        return (
            f"Upscaled using {options.workflow} {factor}x "
            f"(precision={options.precision}, output={output_fmt}, "
            f"quality={options.jpeg_quality}, denoise={options.denoise_strength:.2f}, "
            f"sharpen={options.sharpen_amount:.2f})"
        )

    def _probe_image_size(self, image_path):
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                return img.size
        except Exception:
            return None

    def _clear_log(self):
        if self.winfo_exists():
            self.console.configure(state="normal")
            self.console.delete("1.0", "end")
            self.console.insert("0.0", "--- Upscale log initialized ---\n")
            self.console.configure(state="disabled")

    def _log_event(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        logger.info(line)
        if self.winfo_exists():
            self.after(0, lambda: self._append_log_line(line))

    def _append_log_line(self, line: str):
        if not self.winfo_exists():
            return
        self.console.configure(state="normal")
        self.console.insert("end", f"{line}\n")
        self.console.see("end")
        self.console.configure(state="disabled")

    def _flush_upscale_cache(self, item_id: int, original_path, upscaled_path) -> None:
        """
        Remove cached temp files for this item after successful check-in.
        """
        client = self.controller.session.daminion_client
        cache_root = getattr(client, "temp_dir", None)
        if not cache_root:
            return

        try:
            cache_root = Path(cache_root).resolve()
        except Exception:
            return

        candidates = set()
        for p in (original_path, upscaled_path):
            try:
                if p:
                    candidates.add(Path(p))
            except Exception:
                continue

        # Also sweep any stale variants for this item's original/upscaled temp files.
        try:
            candidates.update(cache_root.glob(f"{item_id}_original*"))
            candidates.update(cache_root.glob(f"{item_id}_original_upscaled*"))
        except Exception:
            pass

        removed = 0
        for cand in candidates:
            try:
                resolved = cand.resolve()
                resolved.relative_to(cache_root)
                if resolved.exists() and resolved.is_file():
                    resolved.unlink()
                    removed += 1
            except Exception:
                continue

        if removed > 0:
            self._log_event(
                f"Item {item_id}: flushed upscale cache ({removed} file(s) removed)."
            )
