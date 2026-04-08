"""
Synapic - AI-Powered Image Tagging Application
===============================================

Main entry point for the Synapic application. This application uses AI models
(local, Hugging Face, or OpenRouter) to automatically generate metadata tags
for images, including categories, keywords, and descriptions.

The application can work with:
- Local image folders
- Daminion Digital Asset Management (DAM) system
Author: Dean Kruger
License: Proprietary
"""

import sys
import argparse
import os
import logging

# ============================================================================
# PYTHONW COMPATIBILITY - NULL STREAM SAFETY
# ============================================================================
# When launched via pythonw.exe (e.g. from start_synapic.bat), sys.stdout and
# sys.stderr are None. This silently breaks logging.StreamHandler, print(),
# and any cleanup code that relies on logging for error reporting — causing
# memory leak fixes to be silently skipped.
# Replace None streams with devnull wrappers so all downstream code works.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# ============================================================================
# WINDOWS COMPATIBILITY - HUGGING FACE CACHE SYMLINKS
# ============================================================================
# On Windows, Hugging Face Hub tries to use symlinks for caching which requires
# Developer Mode or Administrator privileges. Setting this env var disables
# symlink warnings and tells HF Hub to copy files instead.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ============================================================================
# PATH SETUP
# ============================================================================
# Ensure the 'src' directory is in Python's module search path.
# This allows us to import modules using 'from src.core import ...' syntax
# regardless of where the script is executed from.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# ============================================================================
# LOGGING INITIALIZATION
# ============================================================================
# Initialize the logging system BEFORE importing any other application modules.
# This ensures all subsequent imports and operations are properly logged.
# The logger writes to both console and a rotating file in the 'logs' directory.
from src.utils.logger import setup_logging

log_file = setup_logging()

# Import the main application UI after logging is configured
import customtkinter
import tkinter

# MONKEYPATCH: CTkToplevel icon methods
# The customtkiner library tries to load a default icon in a delayed callback.
# On some systems/configurations (especially Python 3.14 preview), this fails with
# a TclError even if the file exists. We patch these methods to catch and ignore those errors.


def safe_wm_iconbitmap(self, bitmap=None, default=None):
    try:
        # We can't easily call the original CTkToplevel.wm_iconbitmap because we're replacing it.
        # But we can replicate its logic (setting the flag) and then call the superclass (tkinter.Toplevel).
        self._iconbitmap_method_called = True
        tkinter.Toplevel.wm_iconbitmap(self, bitmap, default)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Ignored error in wm_iconbitmap: {e}")


def safe_iconbitmap(self, bitmap=None, default=None):
    try:
        self._iconbitmap_method_called = True
        tkinter.Toplevel.iconbitmap(self, bitmap, default)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Ignored error in iconbitmap: {e}")


def safe_wm_iconphoto(self, default=False, *args):
    try:
        tkinter.Toplevel.wm_iconphoto(self, default, *args)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Ignored error in wm_iconphoto: {e}")


customtkinter.CTkToplevel.wm_iconbitmap = safe_wm_iconbitmap
customtkinter.CTkToplevel.iconbitmap = safe_iconbitmap
customtkinter.CTkToplevel.wm_iconphoto = safe_wm_iconphoto

from src.ui.app import App


def parse_args():
    parser = argparse.ArgumentParser(description="Synapic launcher (GUI or headless)")
    parser.add_argument(
        "--no-gui",
        "-n",
        action="store_true",
        dest="no_gui",
        help="Run in headless mode without launching GUI",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    """
    Main application entry point.
    
    This function:
    1. Initializes the logger for this module
    2. Creates the main application window (CustomTkinter-based GUI)
    3. Starts the GUI event loop
    4. Handles any fatal errors with comprehensive logging
    5. Ensures proper cleanup on shutdown
    
    The application follows a wizard-style workflow:
    - Step 1: Select data source (local folder or Daminion)
    - Step 2: Configure tagging engine (model selection, device, threshold)
    - Step 3: Process images with AI model
    - Step 4: View and export results
    """
    logger = logging.getLogger(__name__)

    try:
        # Log application startup information
        logger.info("Initializing Synapic application")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {current_dir}")

        if args.no_gui:
            logger.info("No-GUI mode requested. Exiting after initialization steps.")
            logger.info("Application shutdown (headless)")
            from src.utils.logger import shutdown_logging

            shutdown_logging()
            sys.exit(0)

        # Create and display the main application window
        # The App class (from src.ui.app) handles all UI initialization
        app = App()
        logger.info("Application window created successfully")

        # Start the GUI event loop (blocks until window is closed)
        app.mainloop()

    except Exception as e:
        # Log any fatal errors with full stack trace
        logger.critical(f"Fatal error in main application: {e}", exc_info=True)
        raise  # Re-raise to ensure the application exits with error code
    finally:
        # Always perform cleanup, even if an error occurred
        logger.info("Application shutdown")
        from src.utils.logger import shutdown_logging

        shutdown_logging()

        # Explicitly exit to ensure all threads/processes are terminated
        # This is especially important for Windows where pythonw can sometimes hang
        sys.exit(0)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
