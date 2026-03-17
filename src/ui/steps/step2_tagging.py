"""
Step 2: Tagging Engine Configuration UI
=======================================

This module defines the UI for selecting and configuring the AI engine used 
for image processing. It supports a unified interface for three distinct 
processing targets:

1. Local Inference: Uses on-device hardware (CPU/GPU) to run cached models.
2. Hugging Face API: Serverless inference via the Hugging Face Hub (requires Internet).
3. OpenRouter API: Access to multimodal models via a unified gateway (requires Internet).

The UI manages model discovery (searching the Hub), cache management (downloading 
models for offline use), and global inference parameters like confidence 
thresholds and device selection.

Key Components:
---------------
- Engine Cards: Visual selection of the processing provider.
- Config Dialog: Modal interface for model selection and API key management.
- Download Manager: Integrated downloader with real-time progress for local models.
- Global Settings: Threshold sliders and device toggles (CPU vs. CUDA).

Author: Synapic Project
"""

import queue
try:
    from src.ui.steps.step_groq_settings import GroqSettingsDialog  # optional
except Exception:
    GroqSettingsDialog = None
import customtkinter as ctk
from src.utils.background_worker import BackgroundWorker
from src.utils.concurrency import DaemonThreadPoolExecutor

class Step2Tagging(ctk.CTkFrame):
    """
    UI component for the second step of the tagging wizard.
    
    This frame serves as the hub for AI configuration. It coordinates between
    local model caching and remote API settings, ensuring the 'EngineConfig'
    is fully populated before moving to the execution phase.
    
    Attributes:
        controller: The main App instance managing the wizard flow.
        session: Global application state and configuration.
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

        # Title
        title = ctk.CTkLabel(self.container, text="Step 2: Tagging Engine", font=("Roboto", 24, "bold"))
        title.grid(row=0, column=0, pady=(20, 30))

        # Engine Selection
        self.engine_var = ctk.StringVar(value=self.controller.session.engine.provider or "huggingface")
        
        # Engine Cards (using Radio buttons for simplicity but styled)
        self.cards_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        self.cards_frame.grid(row=1, column=0, pady=10)
        
        self.create_engine_card(self.cards_frame, "Local Inference", "local", 0)
        self.create_engine_card(self.cards_frame, "Hugging Face", "huggingface", 1)
        self.create_engine_card(self.cards_frame, "OpenRouter", "openrouter", 2)
        self.create_engine_card(self.cards_frame, "Groq", "groq_package", 3)
        self.create_engine_card(self.cards_frame, "Ollama", "ollama", 4)
        self.create_engine_card(self.cards_frame, "Nvidia", "nvidia", 5)
        self.create_engine_card(self.cards_frame, "Google AI", "google_ai", 6)
        self.create_engine_card(self.cards_frame, "Cerebras", "cerebras", 7)

        # Inline Config Container
        self.session = self.controller.session
        self._worker = BackgroundWorker(name="Step2Worker")
        self.config_container = ctk.CTkFrame(self.container, fg_color="transparent")
        self.config_container.grid(row=2, column=0, pady=5, sticky="ew")
        self.config_container.grid_columnconfigure(0, weight=1)

        self.tab_local = ctk.CTkFrame(self.config_container, fg_color="transparent")
        self.tab_hf = ctk.CTkFrame(self.config_container, fg_color="transparent")
        self.tab_or = ctk.CTkFrame(self.config_container, fg_color="transparent")
        self.tab_groq = ctk.CTkFrame(self.config_container, fg_color="transparent")
        self.tab_ollama = ctk.CTkFrame(self.config_container, fg_color="transparent")
        self.tab_nvidia = ctk.CTkFrame(self.config_container, fg_color="transparent")
        self.tab_google_ai = ctk.CTkFrame(self.config_container, fg_color="transparent")
        self.tab_cerebras = ctk.CTkFrame(self.config_container, fg_color="transparent")

        self.init_local_tab()
        self.init_hf_tab()
        self.init_or_tab()
        self.init_groq_tab()
        self.init_ollama_tab()
        self.init_nvidia_tab()
        self.init_google_ai_tab()
        self.init_cerebras_tab()
        
        # === Model Info Section ===
        model_info_frame = ctk.CTkFrame(self.container, fg_color="#2B2B2B", corner_radius=10)
        model_info_frame.grid(row=3, column=0, pady=10, padx=40, sticky="ew")
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
        settings_frame.grid(row=4, column=0, pady=10, padx=40, sticky="ew")
        
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
        
        # Left label: Free
        ctk.CTkLabel(
            slider_container,
            text="Free",
            font=("Roboto", 10),
            text_color="gray"
        ).pack(side="left", padx=(0, 10))
        
        # Slider
        self.threshold_slider = ctk.CTkSlider(
            slider_container,
            from_=1,
            to=100,
            number_of_steps=99,
            command=self.on_threshold_change
        )
        self.threshold_slider.set(self.controller.session.engine.confidence_threshold)
        self.threshold_slider.pack(side="left", fill="x", expand=True)
        
        # Right label: Strict
        ctk.CTkLabel(
            slider_container,
            text="Strict",
            font=("Roboto", 10),
            text_color="gray"
        ).pack(side="left", padx=(10, 0))
        
        # Navigation Buttons
        nav_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        nav_frame.grid(row=5, column=0, pady=20, sticky="ew")
        
        ctk.CTkButton(nav_frame, text="Previous", command=lambda: self.controller.show_step("Step1Datasource"), width=150, fg_color="gray").pack(side="left", padx=20)
        ctk.CTkButton(nav_frame, text="Next Step", command=self.next_step, width=200, height=40).pack(side="right", padx=20)

        # Traces for color coding
        self.engine_var.trace_add("write", lambda *args: self._on_engine_change())
        self._on_engine_change()

        # (Auto-load of Groq models is handled in the ConfigDialog for the Groq tab)

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
        print(f"Device changed to: {value}")

    def _on_engine_change(self):
        engine = self.engine_var.get()
        # Hide all frames
        for frame in [self.tab_local, self.tab_hf, self.tab_or, self.tab_groq, self.tab_ollama, self.tab_nvidia, self.tab_google_ai, self.tab_cerebras]:
            frame.grid_forget()
            
        # Show the correct frame
        if engine == "local":
            self.tab_local.grid(row=0, column=0, sticky="nsew")
        elif engine == "huggingface":
            self.tab_hf.grid(row=0, column=0, sticky="nsew")
        elif engine == "openrouter":
            self.tab_or.grid(row=0, column=0, sticky="nsew")
        elif engine == "groq_package":
            self.tab_groq.grid(row=0, column=0, sticky="nsew")
            if not getattr(self, '_groq_models_loaded', False):
                self._load_and_display_groq_models()
                self._groq_models_loaded = True
        elif engine == "ollama":
            self.tab_ollama.grid(row=0, column=0, sticky="nsew")
            if not getattr(self, '_ollama_models_loaded', False):
                self._load_and_display_ollama_models()
                self._ollama_models_loaded = True
        elif engine == "nvidia":
            self.tab_nvidia.grid(row=0, column=0, sticky="nsew")
            if not getattr(self, '_nvidia_models_loaded', False):
                self._load_and_display_nvidia_models()
                self._nvidia_models_loaded = True
        elif engine == "google_ai":
            self.tab_google_ai.grid(row=0, column=0, sticky="nsew")
            if not getattr(self, '_google_ai_models_loaded', False):
                self._load_and_display_google_ai_models()
                self._google_ai_models_loaded = True
        elif engine == "cerebras":
            self.tab_cerebras.grid(row=0, column=0, sticky="nsew")
            if not getattr(self, '_cerebras_models_loaded', False):
                self._load_and_display_cerebras_models()
                self._cerebras_models_loaded = True

    def _apply_config(self):
        self.update_model_info()

    def create_engine_card(self, parent, text, value, col):
        card = ctk.CTkRadioButton(parent, text=text, variable=self.engine_var, value=value, font=("Roboto", 16))
        card.grid(row=0, column=col, padx=20, pady=20)
        
            
    def update_model_info(self):
        """Update the model info label after configuration changes."""
        self.model_info_label.configure(text=self._get_model_display_text())
        
    def next_step(self):
        # Update session
        self.controller.session.engine.provider = self.engine_var.get()
        
        # We need to retrieve values from the dialog if it was opened, or use defaults/session
        # This UI flow is a bit tricky because the dialog is modal.
        # Ideally, the dialog should update the session directly when "Save" is clicked (if we had a Save button)
        # Or we should have the inputs on the main card.
        
        # For now, let's assume the user configured it via the dialog which we will update to write to session.
        pass
        
        print(f"Selected Engine: {self.controller.session.engine.provider}")
        self.controller.show_step("Step3Process")

    def refresh_stats(self):
        """
        Synchronize the UI elements with the current Session state.
        
        Called by the App coordinator whenever the user navigates to this step
        to ensure all inputs accurately reflect the persisted configuration.
        """
        # Sync engine provider radio button with session
        self.engine_var.set(self.controller.session.engine.provider or "huggingface")
        self._on_engine_change()
        self.update_model_info()
        # Update device and threshold from session
        self.device_var.set(self.controller.session.engine.device)
        self.threshold_slider.set(self.controller.session.engine.confidence_threshold)
        self.threshold_value_label.configure(text=f"{self.controller.session.engine.confidence_threshold}%")



    def _schedule_ui_update(self, callback):
        """Schedule a callback on the UI thread only while dialog exists."""
        if not self.winfo_exists():
            return
        self.after(0, lambda: callback() if self.winfo_exists() else None)

    # Groq auto-load methods moved to ConfigDialog

    def _load_and_display_groq_models(self):
        from src.integrations.groq_package_client import GroqPackageClient
        api_key = self.session.engine.groq_api_key or self._get_groq_api_key_for_refresh()
        client = GroqPackageClient(api_key=api_key)

        def worker():
            try:
                models = client.list_models(limit=40)
            except Exception:
                models = []
            # UI updates must run on the Tk main thread.
            if self.winfo_exists():
                self.after(0, lambda m=models: self._display_groq_models(m))

        self._worker.submit_replacing("groq_models", worker)

    def _display_groq_models(self, models):
        # Lazy create a Groq models panel on the Groq tab if not exists (though init_groq_tab creates it now)
        if not self.winfo_exists() or not hasattr(self, "_groq_models_list"):
             return 

        for w in self._groq_models_list.winfo_children():
            w.destroy()
            
        # Header
        header_text = f"{'Model ID':<40} | {'Capability':^15} | {'Cost':>15}"
        ctk.CTkLabel(
            self._groq_models_list, 
            text=header_text, 
            font=("Courier New", 12, "bold"),
            text_color="gray",
            anchor="w"
        ).pack(fill="x", pady=(5, 10), padx=5)

        if not models:
            ctk.CTkLabel(self._groq_models_list, text="No Groq models found (check API key?).", text_color="gray").pack()
            return

        for m in models:
            mid = m.get('id') or m.get('model_id') or ''
            cap = m.get('capability') or m.get('task') or 'Groq'
            cost = m.get('token_cost') or m.get('token_cost_per_inference') or m.get('cost')
            cost_text = f"{cost} tokens" if cost is not None else "Unknown"
            
            display_text = f"{mid:<40} | {cap:^15} | {cost_text:>15}"
            
            btn = ctk.CTkButton(
                self._groq_models_list, 
                text=display_text, 
                font=("Courier New", 12),
                fg_color="transparent",
                border_width=1,
                anchor="w", 
                width=0,
                command=lambda m_id=mid: self._select_groq_model(m_id)
            )
            btn.pack(fill="x", pady=2)

    def _select_groq_model(self, model_id):
        self.groq_model.delete(0, "end")
        self.groq_model.insert(0, model_id)

    def init_groq_tab(self):
        # Refined Groq tab with multi-key API Key support and Model Selection
        self.tab_groq.grid_columnconfigure(0, weight=1)
        self.tab_groq.grid_rowconfigure(2, weight=1) # List area grows

        # API Keys section
        key_frame = ctk.CTkFrame(self.tab_groq, fg_color="transparent")
        key_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        key_frame.grid_columnconfigure(0, weight=1)

        # Header row with label, count badge, and refresh button
        header_row = ctk.CTkFrame(key_frame, fg_color="transparent")
        header_row.pack(fill="x")

        ctk.CTkLabel(header_row, text="API Keys (one per line):").pack(side="left")

        self.groq_key_count_label = ctk.CTkLabel(
            header_row, text="0 keys", font=("Roboto", 10),
            text_color="gray"
        )
        self.groq_key_count_label.pack(side="left", padx=10)

        ctk.CTkButton(
            header_row, text="Refresh Models",
            command=self._load_and_display_groq_models, width=120
        ).pack(side="right")

        # Multi-line textbox for API keys
        existing_keys = self.session.engine.groq_api_keys or ""
        self.groq_api_keys_textbox = ctk.CTkTextbox(
            key_frame, height=70, font=("Courier New", 11),
            wrap="none", fg_color="#1E1E1E", border_width=1,
            border_color="#444"
        )
        self.groq_api_keys_textbox.pack(fill="x", pady=(5, 2))
        if existing_keys:
            self.groq_api_keys_textbox.insert("1.0", existing_keys)

        # Helper hint
        ctk.CTkLabel(
            key_frame,
            text="💡 Enter multiple Groq API keys (one per line) for automatic rotation when quota is exceeded.",
            font=("Roboto", 9), text_color="#888", anchor="w", wraplength=600
        ).pack(fill="x")

        # Update key count on any change
        self.groq_api_keys_textbox.bind("<KeyRelease>", lambda e: self._update_groq_key_count())
        self._update_groq_key_count()

        # Status / actions
        row_status = ctk.CTkFrame(self.tab_groq, fg_color="transparent")
        row_status.grid(row=1, column=0, sticky="ew", padx=10, pady=0)
        self.groq_status = ctk.CTkLabel(row_status, text="", text_color="gray")
        self.groq_status.pack(side="left")

        # List
        self._groq_models_list = ctk.CTkScrollableFrame(self.tab_groq, label_text="Available Groq Models")
        self._groq_models_list.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        # Selection
        row_sel = ctk.CTkFrame(self.tab_groq, fg_color="transparent")
        row_sel.grid(row=3, column=0, sticky="ew", padx=10, pady=(5,20))
        
        ctk.CTkLabel(row_sel, text="Selected:").pack(side="left")
        self.groq_model = ctk.CTkEntry(row_sel, width=300)
        # Default if not set
        self.groq_model.insert(0, self.session.engine.model_id if self.session.engine.provider == "groq_package" else "llama2-70b-4096") 
        self.groq_model.pack(side="left", padx=10, fill="x", expand=True)
        
        ctk.CTkButton(row_sel, text="Save Config", command=self.save_groq_config).pack(side="right")

    def _update_groq_key_count(self):
        """Update the key count label based on current textbox content."""
        text = self.groq_api_keys_textbox.get("1.0", "end-1c")
        keys = [k.strip() for k in text.splitlines() if k.strip()]
        count = len(keys)
        if count == 0:
            self.groq_key_count_label.configure(text="0 keys", text_color="gray")
        elif count == 1:
            self.groq_key_count_label.configure(text="1 key", text_color="#2FA572")
        else:
            self.groq_key_count_label.configure(text=f"{count} keys (rotation enabled)", text_color="#2FA572")

    def _get_groq_api_key_for_refresh(self):
        """Get the first API key from the textbox for model listing."""
        text = self.groq_api_keys_textbox.get("1.0", "end-1c")
        keys = [k.strip() for k in text.splitlines() if k.strip()]
        return keys[0] if keys else ""

    def test_groq_connection(self):
        # Simplified test calling load_models essentially
        self._load_and_display_groq_models()

    def save_groq_config(self):
        model_id = self.groq_model.get().strip()
        api_keys_text = self.groq_api_keys_textbox.get("1.0", "end-1c").strip()
        
        # Validate: at least one key
        keys = [k.strip() for k in api_keys_text.splitlines() if k.strip()]
        if not keys:
             self.groq_status.configure(text="At least one API Key required", text_color="red")
             return

        self.session.engine.provider = "groq_package"
        self.session.engine.groq_api_keys = api_keys_text
        self.session.engine.groq_current_key_index = 0  # Reset rotation on save
        self.session.engine.model_id = model_id
        # Groq vision models are multi-modal usually or LLMs.
        self.session.engine.task = "image-to-text"
        
        from src.utils.config_manager import save_config
        try:
            save_config(self.session)
        except Exception:
            pass

        key_info = f"{len(keys)} key{'s' if len(keys) > 1 else ''}"
        self.groq_status.configure(text=f"Groq config saved ({key_info})", text_color="green")
        self._apply_config()
    
    # ================================================================
    # OLLAMA API TAB METHODS
    # ================================================================

    def init_ollama_tab(self):
        """Initialize the Ollama configuration tab."""
        self.tab_ollama.grid_columnconfigure(0, weight=1)
        self.tab_ollama.grid_rowconfigure(2, weight=1)

        # Info banner
        info_frame = ctk.CTkFrame(self.tab_ollama, fg_color="#1A6B3C", corner_radius=8)
        info_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 10))

        ctk.CTkLabel(
            info_frame,
            text="🦙 Ollama — Run LLMs locally or connect to a remote server.",
            wraplength=600,
            font=("Roboto", 11),
            text_color="white"
        ).pack(padx=10, pady=8)

        # Host / Key Configuration — split into two sub-rows for breathing room
        config_frame = ctk.CTkFrame(self.tab_ollama, fg_color="transparent")
        config_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        config_frame.grid_columnconfigure(1, weight=1)

        # ── Sub-row 0: Host URL + Cloud / Local shortcuts
        ctk.CTkLabel(config_frame, text="Host URL:").grid(row=0, column=0, padx=(0, 5), sticky="w")
        self.ollama_host_var = ctk.StringVar(value=self.session.engine.ollama_host or "http://localhost:11434")
        ctk.CTkEntry(config_frame, textvariable=self.ollama_host_var).grid(
            row=0, column=1, sticky="ew", padx=5
        )

        shortcut_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        shortcut_frame.grid(row=0, column=2, padx=5)
        ctk.CTkButton(
            shortcut_frame, text="Cloud", width=55, height=26,
            font=("Roboto", 10), fg_color="#4B4B4B", hover_color="#5B5B5B",
            command=lambda: self.ollama_host_var.set("https://ollama.com")
        ).pack(side="left", padx=2)
        ctk.CTkButton(
            shortcut_frame, text="Local", width=55, height=26,
            font=("Roboto", 10), fg_color="#4B4B4B", hover_color="#5B5B5B",
            command=lambda: self.ollama_host_var.set("http://localhost:11434")
        ).pack(side="left", padx=2)

        # ── Sub-row 1: API Key + Refresh + Status
        ctk.CTkLabel(config_frame, text="API Key:").grid(row=1, column=0, padx=(0, 5), pady=(6, 0), sticky="w")
        self.ollama_key_var = ctk.StringVar(value=self.session.engine.ollama_api_key or "")
        ctk.CTkEntry(config_frame, textvariable=self.ollama_key_var, show="*").grid(
            row=1, column=1, sticky="ew", padx=5, pady=(6, 0)
        )

        key_btn_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        key_btn_frame.grid(row=1, column=2, padx=5, pady=(6, 0))
        ctk.CTkButton(
            key_btn_frame, text="Refresh",
            command=self._load_and_display_ollama_models, width=80
        ).pack(side="left", padx=2)
        self._ollama_status = ctk.CTkLabel(key_btn_frame, text="", text_color="gray")
        self._ollama_status.pack(side="left", padx=6)

        # Models list
        self._ollama_models_list = ctk.CTkScrollableFrame(
            self.tab_ollama,
            label_text="Available Ollama Models"
        )
        self._ollama_models_list.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        # Selection row
        row_sel = ctk.CTkFrame(self.tab_ollama, fg_color="transparent")
        row_sel.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 20))

        ctk.CTkLabel(row_sel, text="Selected:").pack(side="left")
        self._ollama_model_entry = ctk.CTkEntry(row_sel, width=300)

        default_model = (
            self.session.engine.model_id
            if self.session.engine.provider == "ollama" and self.session.engine.model_id
            else "llama3:latest"
        )
        self._ollama_model_entry.insert(0, default_model)
        self._ollama_model_entry.pack(side="left", padx=10, fill="x", expand=True)

        ctk.CTkButton(
            row_sel,
            text="Save Config",
            command=self._save_ollama_config
        ).pack(side="right")

    def _load_and_display_ollama_models(self):
        """Load models from Ollama server in a background thread."""
        self._ollama_status.configure(text="Connecting...", text_color="gray")
        host = self.ollama_host_var.get().strip()
        key = self.ollama_key_var.get().strip()

        def worker():
            try:
                from src.integrations.ollama_client import OllamaClient
                client = OllamaClient(host=host, api_key=key)

                if not client.is_available():
                    self._schedule_ui_update(lambda: self._display_ollama_models([]))
                    return

                models = client.list_models()
                self._schedule_ui_update(lambda m=models: self._display_ollama_models(m))

            except Exception as e:
                self._schedule_ui_update(
                    lambda err=str(e): self._ollama_status.configure(
                        text=f"Error: {err}", text_color="red"
                    )
                )

        self._worker.submit_replacing("ollama_models", worker)

    def _display_ollama_models(self, models):
        """Display Ollama models in the scrollable list."""
        if not self.winfo_exists() or not hasattr(self, "_ollama_models_list"):
            return

        for w in self._ollama_models_list.winfo_children():
            w.destroy()

        # Header
        header_text = f"{'Model ID':<40} | {'Family':^12} | {'Type':^10} | {'Size':>10}"
        ctk.CTkLabel(
            self._ollama_models_list,
            text=header_text,
            font=("Courier New", 12, "bold"),
            text_color="gray",
            anchor="w"
        ).pack(fill="x", pady=(5, 10), padx=5)

        if not models:
            ctk.CTkLabel(
                self._ollama_models_list,
                text="No models found or connection failed.\n"
                     "Make sure Ollama is running and accessible.",
                text_color="gray",
                justify="center"
            ).pack(pady=20)
            self._ollama_status.configure(text="Connection Failed", text_color="red")
            return

        self._ollama_status.configure(
            text=f"{len(models)} models found",
            text_color="#2FA572"
        )

        for m in models:
            mid = m.get('id', '')
            family = m.get('family', 'unknown')
            cap = m.get('capability', 'LLM')
            size = m.get('size', '')

            display_id = mid[:38] + ".." if len(mid) > 40 else mid
            display_text = f"{display_id:<40} | {family:^12} | {cap:^10} | {size:>10}"

            btn = ctk.CTkButton(
                self._ollama_models_list,
                text=display_text,
                font=("Courier New", 12),
                fg_color="transparent",
                border_width=1,
                anchor="w",
                width=0,
                command=lambda m_id=mid: self._select_ollama_model(m_id)
            )
            btn.pack(fill="x", pady=2)

    def _select_ollama_model(self, model_id):
        self._ollama_model_entry.delete(0, "end")
        self._ollama_model_entry.insert(0, model_id)

    def _save_ollama_config(self):
        model_id = self._ollama_model_entry.get().strip()
        host = self.ollama_host_var.get().strip()
        key = self.ollama_key_var.get().strip()
        
        if not model_id:
            self._ollama_status.configure(text="Select a model", text_color="red")
            return

        self.session.engine.provider = "ollama"
        self.session.engine.model_id = model_id
        self.session.engine.ollama_host = host
        self.session.engine.ollama_api_key = key
        # Assume image-to-text (VLM or description prompt)
        self.session.engine.task = "image-to-text"
        
        from src.utils.config_manager import save_config
        try:
            save_config(self.session)
        except Exception:
            pass
            
        self._ollama_status.configure(
            text=f"Saved: {model_id}", text_color="green"
        )
        self._apply_config()

    # ================================================================
    # NVIDIA NIM TAB METHODS
    # ================================================================

    def init_nvidia_tab(self):
        """Initialize the Nvidia NIM configuration tab."""
        self.tab_nvidia.grid_columnconfigure(0, weight=1)
        self.tab_nvidia.grid_rowconfigure(2, weight=1)

        # Info banner
        info_frame = ctk.CTkFrame(self.tab_nvidia, fg_color="#34495E", corner_radius=8)
        info_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 10))

        ctk.CTkLabel(
            info_frame,
            text="🟢 Nvidia NIM — High-performance inference via NVIDIA Inference Microservices.",
            wraplength=550,
            font=("Roboto", 11),
            text_color="white"
        ).pack(padx=10, pady=8)

        # API Key Configuration
        row_key = ctk.CTkFrame(self.tab_nvidia, fg_color="transparent")
        row_key.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        row_key.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(row_key, text="API Key:").grid(row=0, column=0, padx=(0, 5), sticky="w")
        self.nvidia_key_var = ctk.StringVar(value=self.session.engine.nvidia_api_key or "")
        ctk.CTkEntry(row_key, textvariable=self.nvidia_key_var, show="*", width=300).grid(row=0, column=1, sticky="ew", padx=5)

        ctk.CTkButton(
            row_key,
            text="Refresh Models",
            command=self._load_and_display_nvidia_models,
            width=120
        ).grid(row=0, column=2, padx=(10, 0))
        
        self._nvidia_status = ctk.CTkLabel(row_key, text="", text_color="gray")
        self._nvidia_status.grid(row=0, column=3, padx=10)

        # Models list
        self._nvidia_models_list = ctk.CTkScrollableFrame(
            self.tab_nvidia,
            label_text="Available Nvidia models"
        )
        self._nvidia_models_list.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        # Selection row
        row_sel = ctk.CTkFrame(self.tab_nvidia, fg_color="transparent")
        row_sel.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 20))

        ctk.CTkLabel(row_sel, text="Selected:").pack(side="left")
        self._nvidia_model_entry = ctk.CTkEntry(row_sel, width=300)
        
        default_model = (
            self.session.engine.model_id
            if self.session.engine.provider == "nvidia" and self.session.engine.model_id
            else "mistralai/mistral-large-3-675b-instruct-2512"
        )
        self._nvidia_model_entry.insert(0, default_model)
        self._nvidia_model_entry.pack(side="left", padx=10, fill="x", expand=True)

        ctk.CTkButton(
            row_sel,
            text="Save Config",
            command=self._save_nvidia_config
        ).pack(side="right")

    def _load_and_display_nvidia_models(self):
        """Load models from Nvidia NIM in a background thread."""
        self._nvidia_status.configure(text="Connecting...", text_color="gray")
        key = self.nvidia_key_var.get().strip()

        def worker():
            try:
                from src.integrations.nvidia_client import NvidiaClient
                client = NvidiaClient(api_key=key)

                if not client.is_available():
                    self._schedule_ui_update(lambda: self._display_nvidia_models([]))
                    return

                models = client.list_models()
                self._schedule_ui_update(lambda m=models: self._display_nvidia_models(m))

            except Exception as e:
                self._schedule_ui_update(
                    lambda err=str(e): self._nvidia_status.configure(
                        text=f"Error: {err}", text_color="red"
                    )
                )

        self._worker.submit_replacing("nvidia_models", worker)

    def _display_nvidia_models(self, models):
        """Display Nvidia models in the scrollable list."""
        if not self.winfo_exists() or not hasattr(self, "_nvidia_models_list"):
            return

        for w in self._nvidia_models_list.winfo_children():
            w.destroy()

        # Header
        header_text = f"{'Model ID':<40} | {'Provider':^15} | {'Capability':>15}"
        ctk.CTkLabel(
            self._nvidia_models_list,
            text=header_text,
            font=("Courier New", 12, "bold"),
            text_color="gray",
            anchor="w"
        ).pack(fill="x", pady=(5, 10), padx=5)

        if not models:
            ctk.CTkLabel(
                self._nvidia_models_list,
                text="No models found or connection failed.\n"
                     "Check your API key.",
                text_color="gray",
                justify="center"
            ).pack(pady=20)
            self._nvidia_status.configure(text="Connection Failed", text_color="red")
            return

        self._nvidia_status.configure(
            text=f"{len(models)} models found",
            text_color="#2FA572"
        )

        for m in models:
            mid = m.get('id', '')
            prov = m.get('provider', 'Nvidia')
            cap = m.get('capability', 'Vision')

            display_id = mid[:38] + ".." if len(mid) > 40 else mid
            display_text = f"{display_id:<40} | {prov:^15} | {cap:>15}"

            btn = ctk.CTkButton(
                self._nvidia_models_list,
                text=display_text,
                font=("Courier New", 12),
                fg_color="transparent",
                border_width=1,
                anchor="w",
                width=0,
                command=lambda m_id=mid: self._select_nvidia_model(m_id)
            )
            btn.pack(fill="x", pady=2)

    def _select_nvidia_model(self, model_id):
        self._nvidia_model_entry.delete(0, "end")
        self._nvidia_model_entry.insert(0, model_id)

    def _save_nvidia_config(self):
        model_id = self._nvidia_model_entry.get().strip()
        key = self.nvidia_key_var.get().strip()
        
        if not model_id:
            self._nvidia_status.configure(text="Select a model", text_color="red")
            return

        self.session.engine.provider = "nvidia"
        self.session.engine.model_id = model_id
        self.session.engine.nvidia_api_key = key
        # Nvidia vision models are multi-modal
        self.session.engine.task = "image-to-text"
        
        from src.utils.config_manager import save_config
        try:
            save_config(self.session)
        except Exception:
            pass
            
        self._nvidia_status.configure(
            text=f"Saved: {model_id}", text_color="green"
        )
        self._apply_config()

    # ================================================================
    # GOOGLE AI STUDIO TAB METHODS
    # ================================================================

    def init_google_ai_tab(self):
        """Initialize the Google AI Studio configuration tab."""
        self.tab_google_ai.grid_columnconfigure(0, weight=1)
        self.tab_google_ai.grid_rowconfigure(2, weight=1)

        # Info banner
        info_frame = ctk.CTkFrame(self.tab_google_ai, fg_color="#1A73E8", corner_radius=8)
        info_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 10))

        ctk.CTkLabel(
            info_frame,
            text="✨ Google AI Studio — Access Gemini models with a free tier.",
            wraplength=550,
            font=("Roboto", 11),
            text_color="white"
        ).pack(padx=10, pady=8)

        # API Key Configuration
        row_key = ctk.CTkFrame(self.tab_google_ai, fg_color="transparent")
        row_key.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        row_key.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(row_key, text="API Key:").grid(row=0, column=0, padx=(0, 5), sticky="w")
        self.google_ai_key_var = ctk.StringVar(value=self.session.engine.google_ai_api_key or "")
        ctk.CTkEntry(row_key, textvariable=self.google_ai_key_var, show="*", width=300).grid(row=0, column=1, sticky="ew", padx=5)

        ctk.CTkButton(
            row_key,
            text="Refresh Models",
            command=self._load_and_display_google_ai_models,
            width=120
        ).grid(row=0, column=2, padx=(10, 0))

        self._google_ai_status = ctk.CTkLabel(row_key, text="", text_color="gray")
        self._google_ai_status.grid(row=0, column=3, padx=10)

        # Models list
        self._google_ai_models_list = ctk.CTkScrollableFrame(
            self.tab_google_ai,
            label_text="Available Google AI Models"
        )
        self._google_ai_models_list.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        # Selection row
        row_sel = ctk.CTkFrame(self.tab_google_ai, fg_color="transparent")
        row_sel.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 20))

        ctk.CTkLabel(row_sel, text="Selected:").pack(side="left")
        self._google_ai_model_entry = ctk.CTkEntry(row_sel, width=300)

        default_model = (
            self.session.engine.model_id
            if self.session.engine.provider == "google_ai" and self.session.engine.model_id
            else "gemini-2.5-flash"
        )
        self._google_ai_model_entry.insert(0, default_model)
        self._google_ai_model_entry.pack(side="left", padx=10, fill="x", expand=True)

        ctk.CTkButton(
            row_sel,
            text="Save Config",
            command=self._save_google_ai_config
        ).pack(side="right")

    def _load_and_display_google_ai_models(self):
        """Load models from Google AI in a background thread."""
        self._google_ai_status.configure(text="Connecting...", text_color="gray")
        key = self.google_ai_key_var.get().strip()

        def worker():
            try:
                from src.integrations.google_ai_client import GoogleAIClient
                client = GoogleAIClient(api_key=key)

                if not client.is_available():
                    self._schedule_ui_update(lambda: self._display_google_ai_models([]))
                    return

                models = client.list_models()
                self._schedule_ui_update(lambda m=models: self._display_google_ai_models(m))

            except Exception as e:
                self._schedule_ui_update(
                    lambda err=str(e): self._google_ai_status.configure(
                        text=f"Error: {err}", text_color="red"
                    )
                )

        self._worker.submit_replacing("google_ai_models", worker)

    def _display_google_ai_models(self, models):
        """Display Google AI models in the scrollable list."""
        if not self.winfo_exists() or not hasattr(self, "_google_ai_models_list"):
            return

        for w in self._google_ai_models_list.winfo_children():
            w.destroy()

        # Header
        header_text = f"{'Model ID':<40} | {'Provider':^15} | {'Capability':>15}"
        ctk.CTkLabel(
            self._google_ai_models_list,
            text=header_text,
            font=("Courier New", 12, "bold"),
            text_color="gray",
            anchor="w"
        ).pack(fill="x", pady=(5, 10), padx=5)

        if not models:
            ctk.CTkLabel(
                self._google_ai_models_list,
                text="No models found or connection failed.\n"
                     "Check your API key.",
                text_color="gray",
                justify="center"
            ).pack(pady=20)
            self._google_ai_status.configure(text="Connection Failed", text_color="red")
            return

        self._google_ai_status.configure(
            text=f"{len(models)} models found",
            text_color="#2FA572"
        )

        for m in models:
            mid = m.get('id', '')
            prov = m.get('provider', 'Google')
            cap = m.get('capability', 'Multi-modal')

            display_id = mid[:38] + ".." if len(mid) > 40 else mid
            display_text = f"{display_id:<40} | {prov:^15} | {cap:>15}"

            btn = ctk.CTkButton(
                self._google_ai_models_list,
                text=display_text,
                font=("Courier New", 12),
                fg_color="transparent",
                border_width=1,
                anchor="w",
                width=0,
                command=lambda m_id=mid: self._select_google_ai_model(m_id)
            )
            btn.pack(fill="x", pady=2)

    def _select_google_ai_model(self, model_id):
        self._google_ai_model_entry.delete(0, "end")
        self._google_ai_model_entry.insert(0, model_id)

    def _save_google_ai_config(self):
        model_id = self._google_ai_model_entry.get().strip()
        key = self.google_ai_key_var.get().strip()

        if not model_id:
            self._google_ai_status.configure(text="Select a model", text_color="red")
            return

        self.session.engine.provider = "google_ai"
        self.session.engine.model_id = model_id
        self.session.engine.google_ai_api_key = key
        # Gemini models are multi-modal
        self.session.engine.task = "image-to-text"

        from src.utils.config_manager import save_config
        try:
            save_config(self.session)
        except Exception:
            pass

        self._google_ai_status.configure(
            text=f"Saved: {model_id}", text_color="green"
        )
        self._apply_config()

    # ================================================================
    # CEREBRAS INFERENCE TAB METHODS
    # ================================================================

    def init_cerebras_tab(self):
        """Initialize the Cerebras Inference configuration tab."""
        self.tab_cerebras.grid_columnconfigure(0, weight=1)
        self.tab_cerebras.grid_rowconfigure(2, weight=1)

        # Info banner — Cerebras orange brand colour
        info_frame = ctk.CTkFrame(self.tab_cerebras, fg_color="#E05C00", corner_radius=8)
        info_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 10))

        ctk.CTkLabel(
            info_frame,
            text=(
                "⚡ Cerebras — World's fastest LLM inference. "
                "Get your API key at cloud.cerebras.ai"
            ),
            wraplength=550,
            font=("Roboto", 11),
            text_color="white",
        ).pack(padx=10, pady=8)

        # API Key Configuration
        row_key = ctk.CTkFrame(self.tab_cerebras, fg_color="transparent")
        row_key.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        row_key.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(row_key, text="API Key:").grid(row=0, column=0, padx=(0, 5), sticky="w")
        self.cerebras_key_var = ctk.StringVar(
            value=self.session.engine.cerebras_api_key or ""
        )
        ctk.CTkEntry(
            row_key, textvariable=self.cerebras_key_var, show="*", width=300
        ).grid(row=0, column=1, sticky="ew", padx=5)

        ctk.CTkButton(
            row_key,
            text="Refresh Models",
            command=self._load_and_display_cerebras_models,
            width=120,
        ).grid(row=0, column=2, padx=(10, 0))

        self._cerebras_status = ctk.CTkLabel(row_key, text="", text_color="gray")
        self._cerebras_status.grid(row=0, column=3, padx=10)

        # Models list
        self._cerebras_models_list = ctk.CTkScrollableFrame(
            self.tab_cerebras, label_text="Available Cerebras Models"
        )
        self._cerebras_models_list.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        # Selection row
        row_sel = ctk.CTkFrame(self.tab_cerebras, fg_color="transparent")
        row_sel.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 20))

        ctk.CTkLabel(row_sel, text="Selected:").pack(side="left")
        self._cerebras_model_entry = ctk.CTkEntry(row_sel, width=300)

        default_model = (
            self.session.engine.model_id
            if self.session.engine.provider == "cerebras" and self.session.engine.model_id
            else "llama3.1-8b"
        )
        self._cerebras_model_entry.insert(0, default_model)
        self._cerebras_model_entry.pack(side="left", padx=10, fill="x", expand=True)

        ctk.CTkButton(
            row_sel,
            text="Save Config",
            command=self._save_cerebras_config,
        ).pack(side="right")

    def _load_and_display_cerebras_models(self):
        """Load models from Cerebras API in a background thread."""
        self._cerebras_status.configure(text="Connecting...", text_color="gray")
        key = self.cerebras_key_var.get().strip()

        def worker():
            try:
                from src.integrations.cerebras_client import CerebrasClient
                client = CerebrasClient(api_key=key)
                models = client.list_models(limit=40)
                self._schedule_ui_update(lambda m=models: self._display_cerebras_models(m))
            except Exception as exc:
                self._schedule_ui_update(
                    lambda err=str(exc): self._cerebras_status.configure(
                        text=f"Error: {err}", text_color="red"
                    )
                )

        self._worker.submit_replacing("cerebras_models", worker)

    def _display_cerebras_models(self, models):
        """Display Cerebras models in the scrollable list."""
        if not self.winfo_exists() or not hasattr(self, "_cerebras_models_list"):
            return

        for w in self._cerebras_models_list.winfo_children():
            w.destroy()

        # Header
        header_text = f"{'Model ID':<35} | {'Provider':^12} | {'Capability':>18}"
        ctk.CTkLabel(
            self._cerebras_models_list,
            text=header_text,
            font=("Courier New", 12, "bold"),
            text_color="gray",
            anchor="w",
        ).pack(fill="x", pady=(5, 10), padx=5)

        if not models:
            ctk.CTkLabel(
                self._cerebras_models_list,
                text="No models found.\nCheck your API key or network connection.",
                text_color="gray",
                justify="center",
            ).pack(pady=20)
            self._cerebras_status.configure(text="No models found", text_color="orange")
            return

        self._cerebras_status.configure(
            text=f"{len(models)} models found", text_color="#2FA572"
        )

        for m in models:
            mid = m.get("id", "")
            prov = m.get("provider", "Cerebras")
            cap = m.get("capability", "LLM")

            display_id = mid[:33] + ".." if len(mid) > 35 else mid
            display_text = f"{display_id:<35} | {prov:^12} | {cap:>18}"

            btn = ctk.CTkButton(
                self._cerebras_models_list,
                text=display_text,
                font=("Courier New", 12),
                fg_color="transparent",
                border_width=1,
                anchor="w",
                width=0,
                command=lambda m_id=mid: self._select_cerebras_model(m_id),
            )
            btn.pack(fill="x", pady=2)

    def _select_cerebras_model(self, model_id):
        self._cerebras_model_entry.delete(0, "end")
        self._cerebras_model_entry.insert(0, model_id)

    def _save_cerebras_config(self):
        model_id = self._cerebras_model_entry.get().strip()
        key = self.cerebras_key_var.get().strip()

        if not key:
            self._cerebras_status.configure(text="API key required", text_color="red")
            return

        if not model_id:
            self._cerebras_status.configure(text="Select a model", text_color="red")
            return

        self.session.engine.provider = "cerebras"
        self.session.engine.model_id = model_id
        self.session.engine.cerebras_api_key = key
        # Cerebras models are text-based (image sent as base64 data URL or text fallback)
        self.session.engine.task = "image-to-text"

        from src.utils.config_manager import save_config
        try:
            save_config(self.session)
        except Exception:
            pass

        self._cerebras_status.configure(
            text=f"Saved: {model_id}", text_color="green"
        )
        self._apply_config()

    # ================================================================
    # CLEANUP
    # ================================================================

    def destroy(self):
        """Override destroy to clean up worker thread."""
        if hasattr(self, '_worker'):
            self._worker.shutdown()
        super().destroy()


    def init_local_tab(self):
        self.tab_local.grid_columnconfigure(0, weight=1)
        self.tab_local.grid_rowconfigure(1, weight=1)  # List area grows

        # Header with cache info
        header = ctk.CTkFrame(self.tab_local, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        ctk.CTkLabel(header, text="📦 Downloaded Models", 
                     font=("Roboto", 16, "bold")).pack(side="left", padx=5)
        
        self.cache_count_label = ctk.CTkLabel(header, text="(0 models)", 
                                              text_color="gray")
        self.cache_count_label.pack(side="left", padx=5)
        
        ctk.CTkButton(header, text="+ Find & Download Models", 
                      command=self.open_download_manager, 
                      width=180).pack(side="right", padx=5)

        # List of cached models ONLY
        self.local_list_frame = ctk.CTkScrollableFrame(
            self.tab_local, 
            label_text="Ready for Local Inference"
        )
        self.local_list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # Add a header label for clarity
        header_text = f"{'Model ID':<40} | {'Capability':^15} | {'Size':>10}"
        self.list_header = ctk.CTkLabel(
            self.local_list_frame, 
            text=header_text, 
            font=("Courier New", 12, "bold"),
            text_color="gray",
            anchor="w"
        )
        self.list_header.pack(fill="x", pady=(5, 10), padx=5)
        
        # Selection and action
        footer = ctk.CTkFrame(self.tab_local, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        
        ctk.CTkLabel(footer, text="Selected:").pack(side="left")
        self.local_model_var = ctk.StringVar(value=self.session.engine.model_id or "")
        ctk.CTkLabel(footer, textvariable=self.local_model_var, 
                     font=("Roboto", 12, "bold")).pack(side="left", padx=10)
        
        ctk.CTkButton(footer, text="Use for Local Inference", 
                      command=self.save_local).pack(side="right")
        
        # Load cached models
        self.refresh_local_cache()

    def open_download_manager(self):
        """Opens a separate dialog for browsing and downloading models."""
        DownloadManagerDialog(self, self.session)

    def refresh_local_cache(self):
        """Refresh the list of locally cached models."""
        for widget in self.local_list_frame.winfo_children():
            widget.destroy()
        
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
                    self.add_cached_model_item(
                        model_id, 
                        info.get('size_str', 'Unknown size'),
                        info.get('capability', 'Unknown')
                    )
                    
        except Exception as e:
            ctk.CTkLabel(
                self.local_list_frame, 
                text=f"Error scanning cache: {e}",
                text_color="red"
            ).pack()

    def add_cached_model_item(self, model_id, size_str, capability):
        """Add a cached model to the list."""
        frame = ctk.CTkFrame(self.local_list_frame)
        frame.pack(fill="x", pady=2)
        
        # Format the text with "columns" using padding/fixed width font if possible, 
        # but for now a nice formatted string.
        display_text = f"✓ {model_id:<40} | {capability:^15} | {size_str:>10}"
        
        btn = ctk.CTkButton(
            frame, 
            text=display_text, 
            font=("Courier New", 12), # Using monospace for column-like look
            fg_color="transparent", 
            border_width=1,
            text_color="#2FA572",
            anchor="w",
            command=lambda m=model_id: self.select_local_model(m)
        )
        btn.pack(side="left", fill="x", expand=True)
        
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
        self.local_model_var.set(model_id)

    def delete_cached_model(self, model_id):
        """Delete a cached model from disk."""
        # Simple confirmation using tkinter.messagebox if available, or just delete for now
        import tkinter.messagebox as mb
        if mb.askyesno("Confirm Delete", f"Are you sure you want to delete {model_id} from local cache?\nThis will free up disk space."):
            try:
                from src.core import huggingface_utils
                import shutil
                import os
                
                path = huggingface_utils.get_model_cache_dir(model_id)
                if os.path.exists(path):
                    shutil.rmtree(path)
                    print(f"Deleted model directory: {path}")
                
                self.refresh_local_cache()
            except Exception as e:
                mb.showerror("Error", f"Failed to delete model: {e}")

    def save_local(self):
        self.session.engine.provider = "local"
        self.session.engine.model_id = self.local_model_var.get()
        # Find task from cache
        try:
            from src.core import huggingface_utils
            local_models = huggingface_utils.find_local_models()
            model_info = local_models.get(self.session.engine.model_id)
            if model_info:
                # Use the newly added suggested_task from hf_utils
                self.session.engine.task = model_info.get('suggested_task', "image-classification")
                print(f"Setting task for {self.session.engine.model_id} to {self.session.engine.task}")
        except:
            pass
        self._apply_config()

    def validate_model_id(self, model_id, provider):
        """Basic validation to prevent using OR models with HF engine and vice versa."""
        if provider == "huggingface":
            if ":" in model_id and not "/" in model_id.split(":")[0]:
                # Looks like 'google/gemini...:free' or similar
                import tkinter.messagebox as mb
                return mb.askyesno("Potential Error", 
                                  f"The model ID '{model_id}' looks like it might be an OpenRouter model.\n\n"
                                  "Are you sure you want to use it with the Hugging Face engine?")
        elif provider == "openrouter":
            if "/" in model_id and ":" not in model_id:
                # Looks like 'org/model' without a suffix, which is common for HF
                # OpenRouter also uses org/model but often has suffixes or specific names
                pass
        return True

    def init_hf_tab(self):
        self.tab_hf.grid_columnconfigure(0, weight=1)
        self.tab_hf.grid_rowconfigure(4, weight=1)  # List area grows

        # API Key
        row1 = ctk.CTkFrame(self.tab_hf, fg_color="transparent")
        row1.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        ctk.CTkLabel(row1, text="API Key:").pack(side="left")
        self.hf_key = ctk.CTkEntry(row1, width=250, show="*")
        self.hf_key.insert(0, self.session.engine.api_key or "")
        self.hf_key.pack(side="left", padx=10, fill="x", expand=True)

        # Rate Limit Warning Banner
        warning_frame = ctk.CTkFrame(self.tab_hf, fg_color="#FF6B35", corner_radius=8)
        warning_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        
        warning_icon = ctk.CTkLabel(warning_frame, text="⚠️", font=("Roboto", 16))
        warning_icon.pack(side="left", padx=10)
        
        warning_text = ctk.CTkLabel(
            warning_frame, 
            text="⚡ API Test Mode: Free tier has rate limits (~15 req/hour). Multi-modal models (Image+Text) are supported.",
            wraplength=500,
            font=("Roboto", 11)
        )
        warning_text.pack(side="left", padx=5, pady=8)

        # Search Tools (Mirroring OR style)
        row3 = ctk.CTkFrame(self.tab_hf, fg_color="transparent")
        row3.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.hf_search = ctk.CTkEntry(row3, placeholder_text="Search multi-modal models (e.g. 'blip', 'vit-gpt2')...")
        self.hf_search.pack(side="left", fill="x", expand=True, padx=(0,5))
        self.hf_search.bind("<Return>", lambda e: self.search_hf_online())
        ctk.CTkButton(row3, text="Search Hub", width=100, command=self.search_hf_online).pack(side="left")

        # List
        self.hf_list = ctk.CTkScrollableFrame(self.tab_hf, label_text="Recommended Multi-modal Models")
        self.hf_list.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)

        # Add header
        header_text = f"{'Model ID':<40} | {'Capability':^15} | {'Size':>10}"
        self.hf_list_header = ctk.CTkLabel(
            self.hf_list, 
            text=header_text, 
            font=("Courier New", 12, "bold"),
            text_color="gray",
            anchor="w"
        )
        self.hf_list_header.pack(fill="x", pady=(5, 10), padx=5)

        # Selection
        row4 = ctk.CTkFrame(self.tab_hf, fg_color="transparent")
        row4.grid(row=4, column=0, sticky="ew", padx=10, pady=(10,20))
        
        ctk.CTkLabel(row4, text="Selected:").pack(side="left")
        self.hf_model = ctk.CTkEntry(row4, width=250)
        self.hf_model.insert(0, self.session.engine.model_id or "Salesforce/blip-image-captioning-base")
        self.hf_model.pack(side="left", padx=10, fill="x", expand=True)
        
        btn_save_config = ctk.CTkButton(row4, text="Save Config", width=100, command=self.save_hf)
        btn_save_config.pack(side="right", padx=5)
        
        btn_download = ctk.CTkButton(row4, text="Download for Local Use", width=150, fg_color="#2FA572", command=self.download_selected_hf_for_local)
        btn_download.pack(side="right", padx=5)

    def download_selected_hf_for_local(self):
        model_id = self.hf_model.get()
        if not model_id: return
        
        # Open download manager directly for this model
        dm = DownloadManagerDialog(self, self.session)
        dm.search_entry.delete(0, "end")
        dm.search_entry.insert(0, model_id)
        dm.start_search()

    def search_hf_online(self):
        query = self.hf_search.get()
        # Clear list
        for w in self.hf_list.winfo_children(): w.destroy()
        ctk.CTkLabel(self.hf_list, text="Searching Hub...", text_color="gray").pack(pady=10)
        
        def worker():
            try:
                from huggingface_hub import list_models
                from src.core import huggingface_utils, config
                
                tasks = [
                    config.MODEL_TASK_IMAGE_CLASSIFICATION,
                    config.MODEL_TASK_IMAGE_TO_TEXT,
                    config.MODEL_TASK_ZERO_SHOT,
                    "visual-question-answering",
                    "image-text-to-text"
                ]
                
                all_results = []
                for t in tasks:
                    models = list_models(filter=t, search=query, limit=5, sort="downloads", direction=-1)
                    for m in models:
                        all_results.append({
                            'id': m.id,
                            'task': t,
                            'capability': huggingface_utils.get_model_capability(t)
                        })
                
                # Deduplicate
                seen = set()
                unique_results = []
                for r in all_results:
                    if r['id'] not in seen:
                        unique_results.append(r)
                        seen.add(r['id'])
                
                # Filter out incompatible models (GPTQ, AWQ, etc.)
                unique_results = [r for r in unique_results if huggingface_utils.is_model_compatible(r['id'])]
                
                # Fetch sizes
                results_with_details = []
                for item in unique_results:
                    mid = item['id']
                    size_bytes = huggingface_utils.get_remote_model_size(mid)
                    item['size_str'] = huggingface_utils.format_size(size_bytes)
                    results_with_details.append(item)

                self.after(0, lambda: self.show_hf_results(results_with_details) if self.winfo_exists() else None)
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda: self.show_hf_results([], error=error_msg) if self.winfo_exists() else None)
        
        # Use submit_replacing so rapid searches only execute the final one
        self._worker.submit_replacing("hf_search", worker)

    def show_hf_results(self, results, error=None):
        for w in self.hf_list.winfo_children(): w.destroy()
        
        # Re-add header
        header_text = f"{'Model ID':<40} | {'Capability':^15} | {'Size':>10}"
        ctk.CTkLabel(self.hf_list, text=header_text, font=("Courier New", 12, "bold"), text_color="gray", anchor="w").pack(fill="x", pady=(5, 10), padx=5)

        if error:
            ctk.CTkLabel(self.hf_list, text=f"Error: {error}", text_color="red").pack()
            return

        if not results:
            ctk.CTkLabel(self.hf_list, text="No models found.").pack()
            return

        for item in results:
            mid = item['id']
            size_str = item.get('size_str', 'Unknown')
            capability = item.get('capability', 'Unknown')
            
            display_text = f"{mid:<40} | {capability:^15} | {size_str:>10}"
            
            btn = ctk.CTkButton(
                self.hf_list, 
                text=display_text, 
                font=("Courier New", 12),
                fg_color="transparent", 
                border_width=1, 
                anchor="w", 
                command=lambda m=mid: self.select_hf_model(m)
            )
            btn.pack(fill="x", pady=2)

    def select_hf_model(self, mid):
        self.hf_model.delete(0, "end")
        self.hf_model.insert(0, mid)

    def init_or_tab(self):
        self.tab_or.grid_columnconfigure(0, weight=1)
        self.tab_or.grid_rowconfigure(2, weight=1)

        # API Key
        row1 = ctk.CTkFrame(self.tab_or, fg_color="transparent")
        row1.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        ctk.CTkLabel(row1, text="API Key:").pack(side="left")
        self.or_key = ctk.CTkEntry(row1, width=250, show="*")
        self.or_key.insert(0, self.session.engine.api_key or "")
        self.or_key.pack(side="left", padx=10, fill="x", expand=True)
        
        # Tools
        row2 = ctk.CTkFrame(self.tab_or, fg_color="transparent")
        row2.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(row2, text="Fetch Available Models", command=self.fetch_or_models).pack(side="left")
        
        self.var_show_paid = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(row2, text="Show Paid Models", variable=self.var_show_paid, 
                        command=self.fetch_or_models).pack(side="left", padx=10)
        
        # List
        self.or_list = ctk.CTkScrollableFrame(self.tab_or, label_text="OpenRouter Vision Models")
        self.or_list.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        # Header
        header_text = f"{'Model ID':<40} | {'Capability':^15} | {'Cost':>15}"
        ctk.CTkLabel(
            self.or_list, 
            text=header_text, 
            font=("Courier New", 12, "bold"),
            text_color="gray",
            anchor="w"
        ).pack(fill="x", pady=(5, 10), padx=5)

        # Selection
        row4 = ctk.CTkFrame(self.tab_or, fg_color="transparent")
        row4.grid(row=3, column=0, sticky="ew", padx=10, pady=(5,20))
        ctk.CTkLabel(row4, text="Selected:").pack(side="left")
        self.or_model = ctk.CTkEntry(row4, width=300)
        self.or_model.insert(0, self.session.engine.model_id or "openai/gpt-4-vision-preview")
        self.or_model.pack(side="left", padx=10, fill="x", expand=True)
        ctk.CTkButton(row4, text="Save Config", command=self.save_or).pack(side="right")

    def fetch_or_models(self):
        # Clear including header
        for w in self.or_list.winfo_children(): w.destroy()
        
        # Restore header
        header_text = f"{'Model ID':<40} | {'Capability':^15} | {'Cost':>15}"
        ctk.CTkLabel(
            self.or_list, 
            text=header_text, 
            font=("Courier New", 12, "bold"),
            text_color="gray",
            anchor="w"
        ).pack(fill="x", pady=(5, 10), padx=5)

        ctk.CTkLabel(self.or_list, text="Fetching...", text_color="gray").pack()
        
        def worker():
            try:
                from src.core import openrouter_utils
                # We can't really filter by 'task' in the same way, but OR utils handles 'image' modality check
                models, _ = openrouter_utils.find_models_by_task(
                    "image-to-text", 
                    limit=100,
                    include_paid=self.var_show_paid.get()
                )
                self.after(0, lambda: self.show_or_results(models) if self.winfo_exists() else None)
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda: self.show_or_results([], error=error_msg) if self.winfo_exists() else None)
        
        # Use submit_replacing so rapid fetches only execute the final one
        self._worker.submit_replacing("or_fetch", worker)

    def show_or_results(self, results, error=None):
        # Clear list but keep header? Actually cleaner to clear and redraw header in one go if I had separate method, 
        # but here I cleared children in fetch.
        # Let's just clear and redraw header to be safe
        for w in self.or_list.winfo_children(): w.destroy()
        
        header_text = f"{'Model ID':<40} | {'Capability':^15} | {'Cost':>15}"
        ctk.CTkLabel(
            self.or_list, 
            text=header_text, 
            font=("Courier New", 12, "bold"),
            text_color="gray",
            anchor="w"
        ).pack(fill="x", pady=(5, 10), padx=5)

        if error:
            ctk.CTkLabel(self.or_list, text=f"Error: {error}\n(Check internet?)", text_color="red").pack()
            return
        
        if not results:
            ctk.CTkLabel(self.or_list, text="No generic vision models found.").pack()
            return
            
        for mid in results:
             # OR results are often just IDs, but check if we have more info
             # The openrouter_utils.find_models_by_task returns list of strings usually
             capability = "Vision"
             cost_str = "Unknown"
             
             display_text = f"{mid:<40} | {capability:^15} | {cost_str:>15}"

             btn = ctk.CTkButton(
                 self.or_list, 
                 text=display_text, 
                 font=("Courier New", 12),
                 fg_color="transparent", 
                 border_width=1, 
                 anchor="w", 
                 command=lambda m=mid: self.select_or_model(m)
             )
             btn.pack(fill="x", pady=2)

    def select_or_model(self, mid):
        self.or_model.delete(0, "end")
        self.or_model.insert(0, mid)

    def save_hf(self):
        model_id = self.hf_model.get()
        if not self.validate_model_id(model_id, "huggingface"):
            return

        self.session.engine.provider = "huggingface"
        self.session.engine.api_key = self.hf_key.get().strip()
        self.session.engine.model_id = model_id
        
        # Try to infer task or default to image-to-text for multi-modal
        # Classification models often have 'vit', 'resnet', 'siglip' but no 'caption' or 'desc'
        if any(x in model_id.lower() for x in ["vit-base-patch", "resnet-", "siglip-", "bits-"]):
             self.session.engine.task = "image-classification"
        else:
             self.session.engine.task = "image-to-text"
             
        self._apply_config()

    def save_or(self):
        model_id = self.or_model.get().strip()
        
        # Validation
        from src.core import openrouter_utils
        if not openrouter_utils.validate_model_id(model_id):
            import tkinter.messagebox as mb
            if not mb.askyesno("Invalid Model ID", 
                               f"The model ID '{model_id}' was not found in the OpenRouter registry.\n\n"
                               "If this is a new model, you can proceed, but it may fail.\n"
                               "Do you want to proceed anyway?"):
                return

        if not self.validate_model_id(model_id, "openrouter"):
            return

        self.session.engine.provider = "openrouter"
        self.session.engine.api_key = self.or_key.get().strip()
        self.session.engine.model_id = model_id
        # OpenRouter is primarily chat/generation -> image-to-text task in our logical mapping
        # But could be zero-shot if we prompt it right. For now, default to image-to-text (captioning/describe)
        self.session.engine.task = "image-to-text" 
        self._apply_config()




class DownloadManagerDialog(ctk.CTkToplevel):
    """
    Dedicated modal for browsing and downloading models for local use.
    
    This dialog interfaces with both the Hugging Face Hub (search) and the
    local filesystem (model caching). It provides real-time download progress
    via a background worker.
    """
    """Separate dialog for browsing Hub and downloading models."""
    
    def __init__(self, parent, session):
        super().__init__(parent)
        self.parent = parent
        self.session = session
        self.title("Download Models from Hugging Face Hub")
        self.geometry("800x600")
        
        # Background worker for thread management (single persistent thread)
        self._worker = BackgroundWorker(name="DownloadManagerWorker")
        
        # Make the dialog modal or at least ensuring it stays on top
        self.transient(parent)
        self.grab_set()
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Search header
        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        # Task dropdown removed to simplify for average user - focusing on multi-modal
        self.task_var = ctk.StringVar(value="image-to-text")
        
        self.search_entry = ctk.CTkEntry(header, placeholder_text="Search multi-modal models (e.g. 'blip', 'vit', 'qwen')...", width=350)
        self.search_entry.pack(side="left", padx=5, fill="x", expand=True)
        self.search_entry.bind("<Return>", lambda e: self.start_search())
        
        ctk.CTkButton(header, text="Search Hub", command=self.start_search, width=120).pack(side="left", padx=5)

        # Results area
        self.results_frame = ctk.CTkScrollableFrame(self, label_text="Hugging Face Hub Results")
        self.results_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
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
        self.footer.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        
        self.lbl_status = ctk.CTkLabel(self.footer, text="Enter a query and click Search", text_color="gray")
        self.lbl_status.pack(side="left", padx=5)
        
        self.progress = ctk.CTkProgressBar(self.footer)
        self.progress.pack(side="right", padx=10, fill="x", expand=True)
        self.progress.set(0)

    def start_search(self):
        query = self.search_entry.get()
        task = self.task_var.get()
        self.lbl_status.configure(text=f"Searching Hub for '{task}'...")
        
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Use submit_replacing so rapid searches only execute the final one
        self._worker.submit_replacing("search", self._search_worker, query, task)

    def _search_worker(self, query, task):
        try:
            from huggingface_hub import list_models
            from src.core import huggingface_utils, config
            
            # search across all relevant image tasks to be helpful
            tasks = [
                config.MODEL_TASK_IMAGE_CLASSIFICATION,
                config.MODEL_TASK_IMAGE_TO_TEXT,
                config.MODEL_TASK_ZERO_SHOT,
                "visual-question-answering",
                "image-text-to-text"
            ]
            
            all_results = []
            for t in tasks:
                models = list_models(filter=t, search=query, limit=10, sort="downloads", direction=-1)
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
            
            # Filter out incompatible models (GPTQ, AWQ, etc.)
            compatible_results = []
            for r in unique_results:
                if huggingface_utils.is_model_compatible(r['id']):
                    compatible_results.append(r)
                else:
                    print(f"Filtered out incompatible model: {r['id']}")
            unique_results = compatible_results
            
            # Fetch sizes concurrently to avoid UI lag
            # from concurrent.futures import ThreadPoolExecutor -> Replaced with Daemon version
            from src.utils.concurrency import DaemonThreadPoolExecutor as ThreadPoolExecutor
            
            def fetch_size(item):
                mid = item['id']
                try:
                    size_bytes = huggingface_utils.get_remote_model_size(mid)
                    item['size_str'] = huggingface_utils.format_size(size_bytes)
                except Exception:
                    item['size_str'] = "Unknown"
                return item

            with ThreadPoolExecutor(max_workers=5) as executor:
                results_with_details = list(executor.map(fetch_size, unique_results))

            self.after(0, lambda: self.show_search_results(results_with_details) if self.winfo_exists() else None)
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: self.lbl_status.configure(text=f"Error: {error_msg}", text_color="red") if self.winfo_exists() else None)

    def show_search_results(self, results):
        self.lbl_status.configure(text=f"Found {len(results)} models.", text_color="gray")
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Re-add header
        header_text = f"{'Model ID':<40} | {'Capability':^15} | {'Size':>10}"
        ctk.CTkLabel(self.results_frame, text=header_text, font=("Courier New", 12, "bold"), text_color="gray", anchor="w").pack(fill="x", pady=(5, 10), padx=5)

        if not results:
             ctk.CTkLabel(self.results_frame, text="No models found matching your query.", text_color="gray").pack(pady=20)
             return
             
        for item in results:
            self.add_result_item(item['id'], item['size_str'], item['capability'])

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
        btn_test = ctk.CTkButton(frame, text="Test via API", width=100, fg_color="#3B8ED0", 
                                 command=lambda m=model_id: self.test_via_api(m))
        btn_test.pack(side="right", padx=5)
        
        btn_download = ctk.CTkButton(frame, text="Download", width=100, fg_color="#2FA572",
                                     command=lambda m=model_id: self.start_download(m))
        btn_download.pack(side="right", padx=5)

    def test_via_api(self, model_id):
        # Switch to HF tab in parent and select model
        self.parent.engine_var.set("huggingface")
        self.parent.hf_model.delete(0, "end")
        self.parent.hf_model.insert(0, model_id)
        # self.parent.hf_task_var.set(self.task_var.get()) # Task var removed from HF tab
        self.lbl_status.configure(text=f"Selected {model_id} for API testing. Switch to 'Hugging Face' tab.")
        # Optional: focus parent
        self.parent.focus_set()

    def start_download(self, model_id):
        from src.core import huggingface_utils
        
        # Check compatibility before downloading
        if not huggingface_utils.is_model_compatible(model_id):
            reason = huggingface_utils.get_incompatibility_reason(model_id)
            self.lbl_status.configure(
                text=f"Cannot download {model_id}: {reason}", 
                text_color="red"
            )
            import tkinter.messagebox as mb
            mb.showerror(
                "Incompatible Model", 
                f"The model '{model_id}' cannot be used with Synapic.\n\n"
                f"Reason: {reason}\n\n"
                "These models require special libraries not included in Synapic.\n"
                "Please choose a different model."
            )
            return
        
        self.lbl_status.configure(text=f"Downloading {model_id}...", text_color="gray")
        self.progress.set(0)
        
        self.download_queue = queue.Queue()
        self._worker.submit(
            huggingface_utils.download_model_worker, 
            model_id, 
            self.download_queue
        )
        
        self.poll_download_queue()

    def poll_download_queue(self):
        try:
            while True:
                msg_type, data = self.download_queue.get_nowait()
                
                if msg_type == "model_download_progress":
                    downloaded, total = data
                    if total > 0:
                        self.progress.set(downloaded / total)
                        # Optional: Update status with %
                        # self.lbl_status.configure(text=f"Downloading... {downloaded/total*100:.1f}%")
                
                elif msg_type == "status_update":
                    self.lbl_status.configure(text=data, text_color="gray")
                
                elif msg_type == "download_complete":
                    self.on_download_complete(data)
                    return # Stop polling
                
                elif msg_type == "error":
                    self.lbl_status.configure(text=f"Download failed: {data}", text_color="red")
                    return # Stop polling
                    
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
                print(f"Auto-selected model {model_id} with task {self.session.engine.task}")
        except Exception as e:
            print(f"Could not determine task for {model_id}: {e}")
            self.session.engine.task = "image-to-text"  # Default fallback
        
        # Refresh parent cache and update selection
        if hasattr(self.parent, 'refresh_local_cache'):
            self.parent.refresh_local_cache()
        if hasattr(self.parent, 'local_model_var'):
            self.parent.local_model_var.set(model_id)
    
    def destroy(self):
        """Override destroy to clean up worker thread."""
        if hasattr(self, '_worker'):
            self._worker.shutdown()
        super().destroy()
