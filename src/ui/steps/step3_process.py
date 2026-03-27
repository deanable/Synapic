"""
Step 3: Processing Execution UI
===============================

This module provides the interface for executing and monitoring the batch
tagging process. It acts as the UI frontend for the `ProcessingManager`,
routing log messages and progress updates from background worker threads
to the user interface.

Key Responsibilities:
---------------------
- Lifecycle Management: Starting and aborting the background tagging worker.
- Progress Visualization: Real-time progress bar and item counter updates.
- Execution Logging: Capturing and displaying technical logs from the engine.
- Thread Safety: Ensuring all background callbacks are safely marshaled to
  the main Tkinter thread.

Author: Synapic Project
"""

import tkinter as tk
import customtkinter as ctk


class Step3Process(ctk.CTkFrame):
    """
    UI component for the third step of the tagging wizard.

    This frame manages the active processing phase. It initializes the
    `ProcessingManager` with the current session state and provides buttons
    to control the execution lifecycle.

    Attributes:
        manager: The backend orchestrator (ProcessingManager) for the tagging task.
    """

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main container
        self.container = ctk.CTkFrame(self)
        self.container.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_rowconfigure(4, weight=1)  # Log expands

        # Title
        title = ctk.CTkLabel(
            self.container, text="Step 3: Processing", font=("Roboto", 24, "bold")
        )
        title.grid(row=0, column=0, pady=(20, 30))

        # Controls
        controls_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        controls_frame.grid(row=1, column=0, pady=10, padx=20, sticky="ew")
        for col in range(3):
            controls_frame.grid_columnconfigure(col, weight=1)

        self.btn_start = ctk.CTkButton(
            controls_frame,
            text="Start Processing",
            fg_color="green",
            width=200,
            height=50,
            font=("Roboto", 16, "bold"),
            command=self.start_process,
        )
        self.btn_start.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.btn_stop = ctk.CTkButton(
            controls_frame,
            text="ABORT",
            fg_color="red",
            width=150,
            height=50,
            font=("Roboto", 16, "bold"),
            state="disabled",
            command=self.stop_process,
        )
        self.btn_stop.grid(row=0, column=1, padx=10, pady=10)

        # Pagination option (Daminion API returns max 500 records per request)
        self.auto_paginate_var = tk.BooleanVar(value=True)
        self.chk_paginate = ctk.CTkCheckBox(
            controls_frame,
            text="Auto-paginate (reload search per batch)",
            variable=self.auto_paginate_var,
            font=("Roboto", 13),
        )
        self.chk_paginate.grid(row=0, column=2, padx=10, pady=10, sticky="e")

        # Progress Area
        progress_frame = ctk.CTkFrame(self.container)
        progress_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=20)

        self.lbl_status = ctk.CTkLabel(
            progress_frame, text="Ready to start.", font=("Roboto", 16)
        )
        self.lbl_status.pack(pady=(10, 5))

        self.progress_bar = ctk.CTkProgressBar(progress_frame, height=20)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=20, pady=(5, 10))

        self.lbl_counter = ctk.CTkLabel(progress_frame, text="0 / 0 Images")
        self.lbl_counter.pack(pady=(0, 5))

        self.lbl_eta = ctk.CTkLabel(progress_frame, text="", font=("Roboto", 13))
        self.lbl_eta.pack(pady=(0, 10))

        # Console Log
        ctk.CTkLabel(self.container, text="Execution Log:", anchor="w").grid(
            row=3, column=0, sticky="w", padx=20, pady=(10, 0)
        )

        self.console = ctk.CTkTextbox(self.container, font=("Consolas", 12))
        self.console.grid(row=4, column=0, sticky="nsew", padx=20, pady=(5, 20))
        self.console.insert("0.0", "--- Log initialized ---\n")
        self.console.configure(state="disabled")

        # Navigation Buttons
        nav_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        nav_frame.grid(row=5, column=0, pady=20, sticky="ew")

        ctk.CTkButton(
            nav_frame,
            text="Previous",
            command=lambda: self.controller.show_step("Step2Tagging"),
            width=150,
            fg_color="gray",
        ).pack(side="left", padx=20)
        ctk.CTkButton(
            nav_frame,
            text="View Results",
            command=lambda: self.controller.show_step("Step4Results"),
            width=200,
            height=40,
        ).pack(side="right", padx=20)

    def start_process(self):
        """
        Initialize and launch the background processing task.

        Disables navigation/start buttons and spawns a `ProcessingManager`
        to handle the heavy lifting (IO, AI Inference, DAM Writes) on
        separate threads.
        """
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.lbl_status.configure(text="Initializing...")

        # Start Worker
        from src.core.processing import ProcessingManager

        self.manager = ProcessingManager(
            session=self.controller.session,
            log_callback=self.safe_log,
            progress_callback=self.safe_update_progress,
            auto_paginate=self.auto_paginate_var.get(),
        )
        self.manager.start()

    def stop_process(self):
        """
        Signal the background manager to stop processing immediately.

        Note: Termination might not be instantaneous as it waits for the
        current item to finish processing to prevent catalog corruption.
        """
        if hasattr(self, "manager"):
            self.manager.abort()

        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.lbl_status.configure(text="Stopping...")

    def shutdown(self):
        """Clean up resources on application exit."""
        if hasattr(self, "manager") and self.manager:
            self.manager.shutdown()

    def safe_log(self, message):
        self.after(0, lambda: self.log(message))

    def safe_update_progress(
        self, pct, current, total, more_pages=False, elapsed_seconds=0, etc_seconds=0
    ):
        def _update():
            self.progress_bar.set(pct)
            self.lbl_counter.configure(text=f"{current} / {total} Images")
            # ETA display
            if etc_seconds > 0 and etc_seconds < 86400:
                hours, remainder = divmod(int(etc_seconds), 3600)
                mins, secs = divmod(remainder, 60)
                if hours > 0:
                    eta_str = f"ETA: ~{hours}h {mins}m remaining"
                else:
                    eta_str = f"ETA: ~{mins}m {secs}s remaining"
                self.lbl_eta.configure(text=eta_str, text_color="#E8A838")
            elif pct >= 1.0 and not more_pages:
                self.lbl_eta.configure(text="")
            # Only mark as complete when the backend confirms there are no more
            # pages to fetch (more_pages=False). Without this guard the UI would
            # flip the buttons back to idle at pct=1.0 after every full 500-item
            # page, even while pagination is still running.
            if pct >= 1.0 and not more_pages:
                self.lbl_status.configure(text="Completed.")
                self.btn_start.configure(state="normal")
                self.btn_stop.configure(state="disabled")

        self.after(0, _update)

    def log(self, message):
        self.console.configure(state="normal")
        self.console.insert("end", f"> {message}\n")
        self.console.see("end")
        self.console.configure(state="disabled")
