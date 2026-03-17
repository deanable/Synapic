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
from typing import Optional, List
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
    daminion_scope: str = "all"  # 'all', 'selection', 'collection', 'saved_search', 'search'
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

@dataclass
class EngineConfig:
    """
    Configuration for the AI tagging engine.
    
    Supports eight types of providers:
    1. Local - Run models locally using Hugging Face Transformers
    2. Hugging Face - Use Hugging Face Inference API (requires API key)
    3. OpenRouter - Use OpenRouter API for LLM-based tagging (requires API key)
    4. Groq - Use Groq SDK for fast inference (requires API key)
    5. Ollama - Access to local or remote Ollama models (requires running Ollama server)
    6. Nvidia - High-performance inference via NVIDIA NIM (requires API key)
    7. Google AI - Google Gemini API with free tier (requires API key)
    8. Cerebras - Ultra-fast LLM inference via Cerebras Cloud (requires API key)
    
    Attributes:
        provider: Engine type - 'local', 'huggingface', 'openrouter',
                  'groq_package', 'ollama', 'nvidia', 'google_ai', or 'cerebras'
        model_id: Identifier for the model (e.g., 'Qwen/Qwen2-VL-2B-Instruct')
        api_key: API key for cloud providers (not used for local/ollama_free)
        system_prompt: Custom system prompt for LLM-based models
        task: Model task type - 'image-classification', 'zero-shot-image-classification',
              'image-to-text', or 'image-text-to-text'
        confidence_threshold: Minimum confidence (1-100) for including tags in results
                            Lower = more permissive, Higher = more strict
        device: Inference device for local models - 'cpu' or 'cuda' (GPU)
    """
    provider: str = "huggingface"  # 'local', 'huggingface', 'openrouter', 'groq_package', 'ollama', 'nvidia', 'google_ai', 'cerebras'
    model_id: str = ""
    api_key: str = ""
    nvidia_api_key: str = ""  # Nvidia NIM API key
    google_ai_api_key: str = ""  # Google AI Studio (Gemini API) key
    cerebras_api_key: str = ""  # Cerebras Inference API key
    system_prompt: str = ""  # For OpenRouter/LLMs
    task: str = "image-to-text"  # Default task
    confidence_threshold: int = 50  # Confidence threshold (1-100) for category/keyword filtering
    device: str = "cpu"  # 'cpu' or 'cuda' for local inference

    # Groq integration settings (optional)
    groq_base_url: str = ""  # Base URL for Groq API
    groq_api_keys: str = ""  # Newline-separated list of Groq API keys (supports rotation)

    # Ollama integration settings
    ollama_host: str = "http://localhost:11434"  # Ollama server host URL
    ollama_api_key: str = ""  # Ollama API key for authentication

    # Index for Groq API key rotation (not persisted)
    groq_current_key_index: int = 0

    # Set of exhausted Groq API keys for the current run cycle (not persisted)
    groq_exhausted_keys: set = field(default_factory=set)

    @property
    def groq_api_key(self) -> str:
        """Backward-compatible property returning the currently active Groq API key."""
        keys = self.get_groq_key_list()
        if not keys:
            return ""
            
        available_keys = [k for k in keys if k not in self.groq_exhausted_keys]
        if not available_keys:
            return keys[self.groq_current_key_index % len(keys)]
            
        idx = self.groq_current_key_index % len(keys)
        candidate = keys[idx]
        while candidate in self.groq_exhausted_keys:
            idx = (idx + 1) % len(keys)
            candidate = keys[idx]
            
        return candidate

    @groq_api_key.setter
    def groq_api_key(self, value: str):
        """Backward-compatible setter — sets a single key."""
        self.groq_api_keys = value

    def get_groq_key_list(self) -> list:
        """Parse the newline-separated groq_api_keys string into a list of non-empty keys."""
        if not self.groq_api_keys:
            return []
        return [k.strip() for k in self.groq_api_keys.splitlines() if k.strip()]

    def rotate_groq_key(self) -> str:
        """Advance to the next non-exhausted Groq API key and return it. Returns '' if no keys."""
        keys = self.get_groq_key_list()
        if not keys:
            return ""
            
        start_idx = self.groq_current_key_index
        self.groq_current_key_index = (self.groq_current_key_index + 1) % len(keys)
        
        while keys[self.groq_current_key_index] in self.groq_exhausted_keys:
            self.groq_current_key_index = (self.groq_current_key_index + 1) % len(keys)
            if self.groq_current_key_index == start_idx:
                break
                
        return keys[self.groq_current_key_index]

    def mark_groq_key_exhausted(self, key: str):
        """Mark a Groq API key as exhausted for the current run cycle."""
        if key:
            self.groq_exhausted_keys.add(key)

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
        self.results: deque = deque(maxlen=500)  # Bounded: keeps last 500 results for export
        
        self.logger.debug(f"Session initialized - Datasource: {self.datasource.type}, Engine: {self.engine.provider}")
        
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
            self.logger.warning("Attempted to connect to Daminion but datasource type is not 'daminion'")
            return False
            
        try:
            self.logger.info(f"Connecting to Daminion server at {self.datasource.daminion_url}")
            self.logger.debug(f"Daminion user: {self.datasource.daminion_user}")
            
            # Create Daminion client with configured credentials
            self.daminion_client = DaminionClient(
                base_url=self.datasource.daminion_url,
                username=self.datasource.daminion_user,
                password=self.datasource.daminion_pass
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
        
        This method checks that the selected engine and model are properly configured
        and available. Currently a placeholder for future validation logic.
        
        Returns:
            bool: True if engine configuration is valid
        
        TODO: Implement actual validation:
            - For local: Check if model is downloaded
            - For HuggingFace/OpenRouter: Validate API key format
            - Check model compatibility with selected task
        """
        self.logger.info(f"Validating engine configuration - Provider: {self.engine.provider}, Model: {self.engine.model_id}")
        # TODO: Implement verification logic using utils
        self.logger.debug("Engine validation not yet implemented, returning True")
        return True

    def reset_stats(self):
        """
        Reset processing statistics to zero.
        
        This method is called at the start of each processing job to clear
        statistics from any previous runs. It resets counters and clears the
        results list while preserving configuration.
        """
        self.logger.info("Resetting session statistics")
        self.logger.debug(f"Previous stats - Total: {self.total_items}, Processed: {self.processed_items}, Failed: {self.failed_items}")
        
        # Reset all counters to zero
        self.total_items = 0
        self.processed_items = 0
        self.failed_items = 0
        self.results = deque(maxlen=500)  # Bounded: keeps last 500 results
        
        # Clear exhausted API keys for the new run cycle
        self.engine.groq_exhausted_keys.clear()
        
        self.logger.info("Session statistics reset complete")
