"""
Session Management Module
==========================

This module defines the core session and configuration structures for the Synapic application.
The Session class maintains all application state throughout the user's workflow, including:
- Data source configuration (where to get images from)
- Engine configuration (which AI model to use and how)
- Runtime statistics and results
- Daminion client connection (if using DAM system)

The configuration is persisted between sessions using the config_manager utility.
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import logging
from .daminion_client import DaminionClient

# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================


@dataclass
class DatasourceConfig:
    """
    Configuration for the image data source.

    Supports two types of sources:
    1. Local filesystem - scan a folder for images
    2. Daminion DAM - connect to Daminion server and query items

    Attributes:
        type: Either 'local' or 'daminion'

        Local source fields:
            local_path: Absolute path to folder containing images
            local_recursive: Whether to scan subfolders recursively

        Daminion source fields:
            daminion_url: Base URL of Daminion server (e.g., http://server/daminion)
            daminion_user: Username for authentication
            daminion_pass: Password for authentication
            daminion_catalog_id: Display name of the catalog/collection
            current_collection_id: Internal ID for the selected collection
            daminion_scope: Query scope - 'all', 'selection', 'collection', 'saved_search', 'search'
            daminion_saved_search: Display name of saved search
            daminion_saved_search_id: Internal ID of saved search
            daminion_collection_id: Internal ID or access code for collection
            daminion_untagged_keywords: Filter for items missing keywords
            daminion_untagged_categories: Filter for items missing categories
            daminion_untagged_description: Filter for items missing descriptions
            daminion_search_term: Free-text search term
            status_filter: Item status - 'all', 'approved', 'rejected', 'unassigned'
            max_items: Maximum number of items to process (0 = unlimited)
    """

    type: str = "local"  # 'local' or 'daminion'

    # Local filesystem fields
    local_path: str = ""
    local_recursive: bool = False

    # Daminion DAM fields
    daminion_url: str = ""
    daminion_user: str = ""
    daminion_pass: str = ""
    daminion_catalog_id: str = ""  # Display name for collection
    current_collection_id: str = ""  # Internal ID/Code for collection
    daminion_scope: str = (
        "all"  # 'all', 'selection', 'collection', 'saved_search', 'search'
    )
    daminion_saved_search: str = ""  # Display name
    daminion_saved_search_id: str = ""  # Internal ID
    daminion_collection_id: str = ""  # Internal ID / Access Code

    # Daminion filters
    daminion_untagged_keywords: bool = False
    daminion_untagged_categories: bool = False
    daminion_untagged_description: bool = False
    daminion_search_term: str = ""
    status_filter: str = "all"  # 'all', 'approved', 'rejected', 'unassigned'
    max_items: int = 100
    resize_scale: int = 100  # AI inference image scale: 100, 75, 50, 25 (percentage)
    use_thumbnail_override: bool = False  # Override scale to use fixed 200px thumbnail


@dataclass
class EngineConfig:
    """
    Configuration for the AI tagging engine.

    Synapic tags images with locally downloaded Foundation Models (LFM) only.
    Models run through Hugging Face Transformers on the local machine (CPU or
    CUDA GPU) — there is no cloud provider selection.

    Attributes:
        provider: Engine type — always 'local'.
        model_id: Identifier of the local model (e.g., 'Qwen/Qwen2-VL-2B-Instruct').
        system_prompt: Custom system prompt for vision-language models.
        task: Model task type - 'image-classification', 'zero-shot-image-classification',
              'image-to-text', or 'image-text-to-text'
        device: Inference device for local models - 'cpu' or 'cuda' (GPU)
        embedding_rescue_enabled: When True, allow tier 2.5 CLIP-embedding
            scoring to rescue failed local probability passes (candidates
            outside the model's label space). First use downloads ~600MB.
        confidence_threshold: Minimum confidence (1-100) for including tags in results
                            Lower = more permissive, Higher = more strict
    """

    provider: str = "local"  # Synapic runs local (LFM) inference only
    model_id: str = ""
    system_prompt: str = ""  # Custom system prompt for VLM models
    task: str = "image-to-text"  # Default task
    confidence_threshold: int = (
        50  # Confidence threshold (1-100) for category/keyword filtering
    )
    device: str = "cpu"  # 'cpu' or 'cuda' for local inference
    probability_enabled: bool = False
    probability_mode: str = "llm"  # 'llm', 'probability', or 'both'
    probability_candidates: list = field(default_factory=list)
    probability_threshold: float = 0.0
    embedding_rescue_enabled: bool = False  # Opt-in: tier 2.5 CLIP rescue (~600MB download on first use)


# ============================================================================
# SESSION CLASS
# ============================================================================


class Session:
    """
    Main session class that maintains application state.

    This class serves as the central data store for the entire application workflow.
    It holds configuration, runtime state, and results. The session is created once
    at application startup and persists until the application closes.

    The session is passed to all UI steps and processing components, allowing them
    to read configuration and update state.

    Attributes:
        datasource: Configuration for image source (local or Daminion)
        engine: Configuration for AI tagging engine
        daminion_client: Active connection to Daminion server (if using Daminion)
        is_processing: Flag indicating if processing is currently running
        total_items: Total number of items queued for processing
        processed_items: Number of items successfully processed
        failed_items: Number of items that failed processing
        results: List of processing results (dicts with filename, status, tags)
    """

    def __init__(self):
        """Initialize a new session with default configuration."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing new session")

        # Configuration objects (will be populated from saved config or UI)
        self.datasource = DatasourceConfig()
        self.engine = EngineConfig()

        # Daminion connection (initialized when connecting to Daminion server)
        self.daminion_client: Optional[DaminionClient] = None

        # Processing state flag
        self.is_processing = False

        # Runtime statistics (reset at start of each processing job)
        self.total_items = 0
        self.processed_items = 0
        self.failed_items = 0
        self.results: deque = deque(
            maxlen=500
        )  # Bounded: keeps last 500 results for export

        self.logger.debug(
            f"Session initialized - Datasource: {self.datasource.type}, Engine: {self.engine.provider}"
        )

    def connect_daminion(self) -> bool:
        """
        Establish connection to Daminion server.

        This method:
        1. Validates that datasource type is set to 'daminion'
        2. Creates a DaminionClient instance with configured credentials
        3. Attempts authentication with the server
        4. Stores the client for later use if successful

        Returns:
            bool: True if connection and authentication succeeded, False otherwise
        """
        if self.datasource.type != "daminion":
            self.logger.warning(
                "Attempted to connect to Daminion but datasource type is not 'daminion'"
            )
            return False

        try:
            self.logger.info(
                f"Connecting to Daminion server at {self.datasource.daminion_url}"
            )
            self.logger.debug(f"Daminion user: {self.datasource.daminion_user}")

            # Create Daminion client with configured credentials
            self.daminion_client = DaminionClient(
                base_url=self.datasource.daminion_url,
                username=self.datasource.daminion_user,
                password=self.datasource.daminion_pass,
            )

            # Attempt authentication with the server
            success = self.daminion_client.authenticate()

            if success:
                self.logger.info("Successfully authenticated with Daminion server")
            else:
                self.logger.error("Daminion authentication failed")

            return success

        except Exception as e:
            self.logger.error(f"Failed to connect to Daminion: {e}", exc_info=True)
            return False

    def validate_engine(self) -> bool:
        """
        Validate the engine configuration.

        Checks that a local model is selected and downloaded before a
        processing job starts, so misconfiguration is surfaced early rather
        than failing mid-run. The specific reason for any failure is logged.

        Validation rules:
            - A non-empty model_id is required.
            - The model must be downloaded to the local cache.

        Returns:
            bool: True if the engine configuration is valid, False otherwise.
        """
        engine = self.engine
        provider = (engine.provider or "").lower()
        self.logger.info(
            f"Validating engine configuration - Provider: {provider}, Model: {engine.model_id}"
        )

        # A model identifier is required.
        if not engine.model_id or not engine.model_id.strip():
            self.logger.error(
                f"Engine validation failed: no model selected for provider '{provider}'"
            )
            return False

        # Local inference: the model must actually be present in the cache.
        if provider == "local":
            try:
                from src.core.huggingface_utils import is_model_downloaded

                if not is_model_downloaded(engine.model_id):
                    self.logger.error(
                        f"Engine validation failed: local model '{engine.model_id}' "
                        "is not downloaded"
                    )
                    return False
            except Exception as e:
                self.logger.error(
                    f"Engine validation failed: could not verify local model "
                    f"'{engine.model_id}': {e}"
                )
                return False
            self.logger.debug("Engine configuration is valid")
            return True

        # Only local inference is supported; anything else can't be validated.
        self.logger.error(
            f"Engine validation failed: unknown provider '{provider}' "
            "(Synapic supports local models only)"
        )
        return False

    def reset_stats(self):
        """
        Reset processing statistics to zero.

        This method is called at the start of each processing job to clear
        statistics from any previous runs. It resets counters and clears the
        results list while preserving configuration.
        """
        self.logger.info("Resetting session statistics")
        self.logger.debug(
            f"Previous stats - Total: {self.total_items}, Processed: {self.processed_items}, Failed: {self.failed_items}"
        )

        # Reset all counters to zero
        self.total_items = 0
        self.processed_items = 0
        self.failed_items = 0
        self.results = deque(maxlen=500)  # Bounded: keeps last 500 results

        self.logger.info("Session statistics reset complete")

    def validate_workflow_state(self, target_step):
        """
        Validate that the session has sufficient state to proceed to the target step.

        Args:
            target_step: The step identifier to validate for (e.g., "Step2Tagging", "Step3Process")

        Returns:
            tuple: (is_valid, error_message) where is_valid is boolean and error_message is string

        Raises:
            ValueError: If target_step is not recognized
        """
        if target_step == "Step2Tagging":
            # Validate that datasource type is selected
            if not self.datasource.type:
                return False, "Please select a datasource (Local Folder or Daminion Server) in Step 1"

            # If Daminion, validate connection
            if self.datasource.type == "daminion":
                if not self.datasource.daminion_url:
                    return False, "Please enter Daminion server URL"
                if not self.datasource.daminion_user:
                    return False, "Please enter Daminion username"
                if not self.datasource.daminion_pass:
                    return False, "Please enter Daminion password"

        elif target_step == "Step3Process":
            # Validate that engine is configured
            if not self.engine.provider:
                return False, "Please select an AI engine in Step 2"
            if not self.engine.model_id:
                return False, "Please select a model in Step 2"

            # Validate engine configuration (includes local-model download check)
            if not self.validate_engine():
                return False, "Engine configuration is invalid. Please check your settings in Step 2"

            # Validate datasource based on type
            if self.datasource.type == "local":
                if not self.datasource.local_path:
                    return False, "Please select a local folder in Step 1"
                import os
                if not os.path.exists(self.datasource.local_path):
                    return False, f"Selected folder does not exist: {self.datasource.local_path}"
            elif self.datasource.type == "daminion":
                if not self.daminion_client or not self.daminion_client.authenticated:
                    return False, "Please connect to Daminion server in Step 1"

        elif target_step == "Step4Results":
            # Validate that processing has completed or been aborted
            # This is more of a checkpoint - we allow viewing results even if processing failed
            pass
        else:
            raise ValueError(f"Unknown target step: {target_step}")

        return True, ""
