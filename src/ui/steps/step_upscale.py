import logging
from tkinter import messagebox

import customtkinter as ctk

from src.core.upscaler import Swin2SRUpscaler
from src.utils.background_worker import BackgroundWorker

logger = logging.getLogger(__name__)

class StepUpscale(ctk.CTkFrame):
    """
    UI component for upscaling an image from Daminion.
    """

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self._worker = BackgroundWorker(name="UpscaleWorker")
        self._upscaler = Swin2SRUpscaler()

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

        # Status Label
        self.lbl_status = ctk.CTkLabel(
            content_frame,
            text="Ready to upscale.",
            font=("Roboto", 14)
        )
        self.lbl_status.pack(pady=20)

        # Upscale Settings
        settings_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        settings_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(settings_frame, text="Upscale Factor:").pack(side="left", padx=5)
        self.factor_var = ctk.StringVar(value="2x")
        self.factor_dropdown = ctk.CTkComboBox(
            settings_frame,
            values=["2x", "4x"],
            variable=self.factor_var,
            width=100
        )
        self.factor_dropdown.pack(side="left", padx=5)

        # Action Buttons
        btn_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        btn_frame.pack(pady=30)

        self.btn_upscale = ctk.CTkButton(
            btn_frame,
            text="Checkout & Upscale First Item",
            fg_color=("purple", "#6b21a8"),
            hover_color=("darkviolet", "#7c3aed"),
            width=200,
            command=self.start_upscale,
        )
        self.btn_upscale.pack(side="left", padx=10)

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


    def start_upscale(self):
        """
        Initiates the upscaling process in a background thread.
        """
        if not self.controller.session.daminion_client:
            messagebox.showerror("Error", "Not connected to Daminion.")
            return

        items = self.controller.session.current_items
        if not items:
            messagebox.showerror("Error", "No items in current session. Please select a scope with items in Step 1.")
            return

        # We'll just upscale the first item for demonstration
        item_id = items[0].get("id")
        if not item_id:
            messagebox.showerror("Error", "Selected item has no ID.")
            return

        self.btn_upscale.configure(state="disabled")
        self.lbl_status.configure(text=f"Processing Item ID: {item_id}...")

        factor_str = self.factor_var.get()
        factor = int(factor_str.replace("x", ""))

        # Run in worker to avoid blocking UI
        self._worker.submit(
            self._do_upscale,
            item_id,
            factor,
            callback=self._on_upscale_complete
        )

    def _do_upscale(self, item_id: int, factor: int) -> bool:
        """
        The actual checkout -> download -> upscale -> checkin logic.
        """
        client = self.controller.session.daminion_client

        # 1. Checkout
        self._update_status(f"Checking out item {item_id}...")
        success = client.checkout_item(item_id)
        if not success:
            logger.error(f"Checkout failed for item {item_id}")
            # Consider returning False here, but for testing if API fails we can proceed or abort

        try:
            # 2. Download Original
            self._update_status("Downloading original image...")
            original_path = client.download_original(item_id)
            if not original_path:
                self._update_status("Download failed.")
                return False

            # 3. Upscale with a real model
            upscaled_path = self._upscaler.upscale(
                input_path=original_path,
                factor=factor,
                status_callback=self._update_status,
            )

            # 4. Checkin
            self._update_status("Checking in new version...")
            success = client.checkin_item(item_id, str(upscaled_path))
            if success:
                self._update_status("Successfully checked in upscaled version!")
                return True
            else:
                self._update_status("Checkin failed. Upscaled file kept locally.")
                return False

        except Exception as e:
            logger.error(f"Upscale workflow error: {e}")
            self._update_status(f"Error: {e}")
            return False

    def _update_status(self, text: str):
        # Update status label from background thread safely
        if self.winfo_exists():
            self.after(0, lambda: self.lbl_status.configure(text=text))

    def _on_upscale_complete(self, success: bool):
        """
        Callback when worker thread finishes.
        """
        if self.winfo_exists():
            self.btn_upscale.configure(state="normal")
            if success:
                messagebox.showinfo("Success", "Upscaling and Check-in complete!")

    def shutdown(self):
        """Cleanup worker."""
        if self._worker:
            self._worker.shutdown()
