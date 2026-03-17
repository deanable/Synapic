import re

file_path = "src/ui/steps/step2_tagging.py"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace the Configure Button setup (lines 80-82 approx)
btn_regex = re.compile(r'# Configure Button \(renamed to "Select Engine"\)\s+self\.btn_config = ctk\.CTkButton[^\n]+\n\s+self\.btn_config\.grid[^\n]+\n')
inline_panels_code = """# Inline Config Container
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
"""
content = btn_regex.sub(inline_panels_code, content)

# 2. Replace engine_var trace and remove update_config_button_color call in init
trace_regex = re.compile(r'self\.engine_var\.trace_add\("write", lambda \*args: self\.update_config_button_color\(\)\)\s+self\.update_config_button_color\(\)')
trace_new = """self.engine_var.trace_add("write", lambda *args: self._on_engine_change())\n        self._on_engine_change()"""
content = trace_regex.sub(trace_new, content)

# 3. Replace update_config_button_color method (lines 237-288 approx)
update_color_regex = re.compile(r'def update_config_button_color\(self\):.*?elif engine == "cerebras":\n[^\n]*\n[^\n]*\n[^\n]*\n[^\n]*\n[^\n]*\n', re.DOTALL)
on_change_code = """def _on_engine_change(self):
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
"""
content = update_color_regex.sub(on_change_code, content)

# 4. Remove open_config_dialog method
open_dialog_regex = re.compile(r'def open_config_dialog\(self\):.*?self\.update_model_info\(\)\n', re.DOTALL)
content = open_dialog_regex.sub('', content)

# 5. Update refresh_stats to remove update_config_button_color call
refresh_stats_regex = re.compile(r'self\.engine_var\.set\(self\.controller\.session\.engine\.provider or "huggingface"\)\s+self\.update_config_button_color\(\)')
content = refresh_stats_regex.sub('self.engine_var.set(self.controller.session.engine.provider or "huggingface")\n        self._on_engine_change()', content)

# 6. Remove the entirety of ConfigDialog class definition and its __init__ (Lines 347 to 433)
config_dialog_regex = re.compile(r'class ConfigDialog\(ctk\.CTkToplevel\):.*?def _schedule_ui_update\(self, callback\):', re.DOTALL)
content = config_dialog_regex.sub('def _schedule_ui_update(self, callback):', content)

# Now all ConfigDialog methods are part of Step2Tagging! We just need to replace self.destroy() -> self._apply_config()
# but ONLY for the save methods to avoid affecting DownloadManagerDialog which is later.
# Instead of blanket, we'll individually target them or replace all before DownloadManagerDialog.

split_parts = content.split('class DownloadManagerDialog(ctk.CTkToplevel):')
part_tagging = split_parts[0]
part_download = split_parts[1]

part_tagging = part_tagging.replace('self.destroy()', 'self._apply_config()')

content = part_tagging + 'class DownloadManagerDialog(ctk.CTkToplevel):' + part_download

# Update DownloadManagerDialog.test_via_api to use engine_var instead of tabview
dm_test_regex = re.compile(r'self\.parent\.tabview\.set\("Hugging Face"\)')
content = dm_test_regex.sub('self.parent.engine_var.set("huggingface")', content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Rewrite script completed!")
