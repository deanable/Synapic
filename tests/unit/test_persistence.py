"""
Manual Persistence Verification
===============================

This file performs an integration-style check against a live Daminion server to
confirm that metadata updates persist and can be read back after a short delay.

It is kept under `tests/unit` for historical reasons, but in practice it is a
manual verification script that depends on real server connectivity and a known
catalog item.
"""

import os
import sys
import logging
import uuid
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.core.daminion_client import DaminionClient
from tests.integration.verify_metadata import get_record_metadata

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_persistence():
    """Write a unique keyword to a known item and verify it can be read back."""
    # Use real credentials from environment or defaults
    server_url = "http://researchserver.juicefilm.local/daminion"
    username = "admin"
    password = "admin"
    
    # Target item ID from logs
    item_id = 883889
    
    client = DaminionClient(server_url, username, password)
    
    try:
        if not client.authenticate():
            logger.error("Failed to authenticate")
            return
            
        # Generate a unique keyword
        unique_kw = f"Synapic_{uuid.uuid4().hex[:8]}"
        logger.info(f"Testing persistence with unique keyword: {unique_kw}")
        
        # Current state
        current = get_record_metadata(client, item_id)
        logger.info(f"Current keywords: {current['keywords']}")
        
        # Add keyword
        logger.info(f"Adding keyword to item {item_id}...")
        success = client.update_item_metadata(
            item_id=item_id,
            keywords=[unique_kw] # Note: update_item_metadata currently ADDS keywords in main-like version
        )
        
        if not success:
            logger.error("update_item_metadata reported failure")
            return
            
        logger.info("Update reported success. Waiting 2 seconds for server to commit...")
        import time
        time.sleep(2)
        
        # Verify
        logger.info("Retrieving metadata to verify persistence...")
        after = get_record_metadata(client, item_id)
        logger.info(f"Keywords AFTER update: {after['keywords']}")
        
        if unique_kw.lower() in [k.lower() for k in after['keywords']]:
            logger.info("SUCCESS: Unique keyword found in catalog!")
        else:
            logger.error(f"FAILURE: Unique keyword '{unique_kw}' NOT FOUND in catalog.")
            
    except Exception as e:
        logger.exception(f"Test failed with error: {e}")
    finally:
        client.logout()

if __name__ == "__main__":
    test_persistence()
