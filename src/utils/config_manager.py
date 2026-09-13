"""
Application Configuration Persistence
======================================

This module manages the serialization and deserialization of the Synapic 
application state. It ensures that user preferences, such as selected 
datasource paths, AI engine providers, and API keys, are preserved between 
application restarts.

Key Responsibilities:
---------------------
- File-System Persistence: Stores config in a hidden JSON file in the 
  user's home directory (`~/.synapic_v2_config.json`).
- State Synchronization: Maps JSON keys to the attributes of the `Session`, 
  `DatasourceConfig`, and `EngineConfig` dataclasses.
- Security Logging: Interfaces with the logger to record save/load events 
  while automatically redacting sensitive fields.

Author: Synapic Project
"""

import json
import logging
from pathlib import Path
from dataclasses import asdict

from src.core.session import Session
from src.utils.logger import log_config

CONFIG_PATH = Path.home() / ".synapic_v2_config.json"

def save_config(session: Session):
    """
    Persist the current session state to the configuration file.
    
    This function extracts the `DatasourceConfig` and `EngineConfig` from the 
    global session, converts them to primitive dictionaries, and writes 
    them to disk as a pretty-printed JSON file.
    
    Args:
        session: The active Session object containing the state to be saved.
    """
    logger = logging.getLogger(__name__)
    
    try:
        data = {
            "datasource": asdict(session.datasource),
            "engine": asdict(session.engine)
        }
        
        # Log the configuration being saved (with sensitive data masked)
        log_config("Saving Configuration", data, logger)
        
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Configuration saved successfully to {CONFIG_PATH}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}", exc_info=True)

def load_config(session: Session):
    """
    Load and apply configuration from the hidden JSON file.
    
    If a configuration file exists, this function parses it and updates the 
    attributes of the provided `Session` object. It uses a field-by-field 
    mapping approach to ensure that only valid configuration keys are 
    applied, avoiding the overwriting of any Session logic or methods.
    
    Args:
        session: The Session object to be populated with loaded data.
    """
    logger = logging.getLogger(__name__)
    
    if not CONFIG_PATH.exists():
        logger.info(f"No existing configuration file found at {CONFIG_PATH}")
        return

    try:
        logger.info(f"Loading configuration from {CONFIG_PATH}")
        
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        
        # Log the loaded configuration (with sensitive data masked)
        log_config("Loaded Configuration", data, logger)
            
        if "datasource" in data:
            ds_data = data["datasource"]
            # Safely update fields
            for k, v in ds_data.items():
                if hasattr(session.datasource, k):
                    setattr(session.datasource, k, v)
            logger.debug(f"Datasource configuration updated: type={session.datasource.type}")

        if "engine" in data:
            eng_data = data["engine"]
            for k, v in eng_data.items():
                if hasattr(session.engine, k):
                    if k == "api_key" and isinstance(v, str):
                        v = v.strip()
                    setattr(session.engine, k, v)
            # Migration: pre-2.5 configs only persisted probability_enabled.
            # Treat a saved "enabled" flag as the 'both' mode.
            if (
                hasattr(session.engine, "probability_mode")
                and session.engine.probability_mode == "llm"
                and getattr(session.engine, "probability_enabled", False)
            ):
                session.engine.probability_mode = "both"

            # LFM-only migration: legacy configs may have saved a cloud
            # provider. Synapic tags images with local models only, so any
            # non-local provider is coerced back to 'local' (the saved
            # model_id is kept — the user can re-select it in Step 2 if it
            # is a downloaded local model).
            if session.engine.provider != "local":
                logger.info(
                    f"Migrating legacy provider '{session.engine.provider}' to 'local'"
                )
                session.engine.provider = "local"

            logger.debug(f"Engine configuration updated: provider={session.engine.provider}, task={session.engine.task}")
                    
        logger.info("Configuration loaded and applied successfully")
        
    except json.JSONDecodeError as e:
        logger.error(f"Configuration file is corrupted: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)

