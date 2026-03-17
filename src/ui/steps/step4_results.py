"""
Step 4: Results and Review UI
=============================

This module defines the final step of the tagging wizard, providing a 
comprehensive review of the processing session. It displays aggregate 
metrics and a granular list of every processed item.

Key Responsibilities:
---------------------
- Metrics Dashboard: Visualizing Success vs. Failure counts.
- Session History: Displaying a list of processed files with their status.
- External Integration: Opening the technical log file in the system default 
  text editor.
- Data Reset: Providing an entry point to restart the wizard and begin a 
  new session.

Author: Synapic Project
"""

import customtkinter as ctk
import os
import subprocess
import platform

class Step4Results(ctk.CTkFrame):
    """
    UI component for the fourth and final step of the tagging wizard.
    
    This frame serves as the post-mortem view of the tagging operations. It 
    extracts the final results from the `Session` object and presents them 
    in a human-readable format.
    
    Attributes:
        controller: The main App instance managing the wizard flow.
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
        self.container.grid_rowconfigure(2, weight=1)

        # Title
        title = ctk.CTkLabel(self.container, text="Step 4: Results & Review", font=("Roboto", 24, "bold"))
        title.grid(row=0, column=0, pady=(20, 30))

        # Metrics Dashboard
        self.metrics_frame = ctk.CTkFrame(self.container)
        self.metrics_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        
        self.create_metric(self.metrics_frame, "Total Processed", "0", 0)
        self.create_metric(self.metrics_frame, "Successful", "0", 1, "green")
        self.create_metric(self.metrics_frame, "Failed", "0", 2, "red")
        self.create_metric(self.metrics_frame, "Skipped", "0", 3, "orange")

        # Results Grid (Simple scrollable frame)
        ctk.CTkLabel(self.container, text="Session Details:", anchor="w").grid(row=2, column=0, sticky="nw", padx=20, pady=(10,0))
        
        self.results_frame = ctk.CTkScrollableFrame(self.container, label_text="Filename | Status | Tags")
        self.results_frame.grid(row=3, column=0, sticky="nsew", padx=20, pady=10)
        
    def tkraise(self, *args, **kwargs):
        super().tkraise(*args, **kwargs)
        self.refresh_stats()

    def refresh_stats(self):
        """
        Synchronize the results UI with the final data stored in the Session.
        
        This method is called every time the frame is shown (tkraise) to ensure 
        the results list is fully populated with the latest processing data.
        """
        s = self.controller.session
        
        # Update metrics
        # We need store references to metric labels or rebuild them.
        # Rebuilding is easier for this prototype.
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()
            
        self.create_metric(self.metrics_frame, "Total Processed", str(s.processed_items), 0)
        self.create_metric(self.metrics_frame, "Successful", str(s.processed_items - s.failed_items), 1, "green")
        self.create_metric(self.metrics_frame, "Failed", str(s.failed_items), 2, "red")
        
        # Update Grid
        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        for res in s.results:
            self.add_result_row(res.get("filename", "?"), res.get("status", "?"), res.get("tags", ""))

        # Actions
        action_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        action_frame.grid(row=4, column=0, pady=20, sticky="ew")
        
        ctk.CTkButton(action_frame, text="Open Log File", command=self.open_logs, fg_color="gray").pack(side="left", padx=20)
        ctk.CTkButton(action_frame, text="Export CSV", command=self.export_report).pack(side="left", padx=20)
        ctk.CTkButton(action_frame, text="New Session", command=self.new_session, fg_color="green", width=200).pack(side="right", padx=20)

    def create_metric(self, parent, label: str, value: str, col: int, color: str = "white"):
        """
        Create a styled metric card.
        
        Args:
            parent: The frame to place the metric in.
            label: Text description of the metric.
            value: The numeric value to display in large font.
            col: Grid column index.
            color: Font color for the value text.
        """
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=0, column=col, padx=10, pady=10, sticky="ew")
        parent.grid_columnconfigure(col, weight=1)
        
        ctk.CTkLabel(frame, text=value, font=("Roboto", 30, "bold"), text_color=color).pack()
        ctk.CTkLabel(frame, text=label, font=("Roboto", 12)).pack()

    def add_result_row(self, filename, status, tags):
        row = ctk.CTkFrame(self.results_frame)
        row.pack(fill="x", pady=2)
        
        ctk.CTkLabel(row, text=filename, width=150, anchor="w").pack(side="left", padx=5)
        
        color = "green" if status == "Success" else "red"
        ctk.CTkLabel(row, text=status, width=80, text_color=color).pack(side="left", padx=5)
        
        ctk.CTkLabel(row, text=tags, anchor="w").pack(side="left", fill="x", expand=True, padx=5)

    def open_logs(self):
        """Open the detailed log file (synapic.log)."""
        from src.utils.logger import LOG_DIR
        log_file = LOG_DIR / "synapic.log"
        
        try:
            if not log_file.exists():
                print(f"Log file does not exist: {log_file}")
                # Fallback to directory
                if LOG_DIR.exists():
                     self._open_path(LOG_DIR)
                return
            
            self._open_path(log_file)
            print(f"Opened log file: {log_file}")
            
        except Exception as e:
            print(f"Failed to open log file: {e}")

    def _open_path(self, path):
        """Helper to open file or folder."""
        system = platform.system()
        if system == 'Windows':
            os.startfile(path)
        elif system == 'Darwin':  # macOS
            subprocess.run(['open', str(path)])
        else:  # Linux
            subprocess.run(['xdg-open', str(path)])

    def export_report(self):
        print("Exporting report...")

    def new_session(self):
        self.controller.show_step("Step1Datasource")
