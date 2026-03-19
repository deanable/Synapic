"""
Windows Registry Credential Storage
===================================

This module provides a secure mechanism for storing sensitive application 
credentials (URLs, usernames, and passwords) within the Windows Registry.

By utilizing the Registry (`HKEY_CURRENT_USER\\SOFTWARE\\Synapic`), we ensure 
that:
1. Credentials are kept out of the application's local configuration files.
2. Sensitive data is never accidentally committed to version control.
3. Access is restricted to the current Windows user profile.

Architecture:
-------------
- Root Key: `HKEY_CURRENT_USER\\SOFTWARE\\Synapic`
- Daminion Subkey: `SOFTWARE\\Synapic\\Daminion` (Stores DAM credentials)

Note:
-----
This module is Windows-specific and depends on the `winreg` standard library. 
It should only be used in environments where the Windows OS is present.

Author: Synapic Project
"""
import winreg
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Registry path for Synapic credentials
REGISTRY_KEY = r"SOFTWARE\Synapic"
DAMINION_SUBKEY = r"SOFTWARE\Synapic\Daminion"
UI_PREFS_SUBKEY = r"SOFTWARE\Synapic\UIPreferences"


def _get_or_create_key(key_path: str) -> winreg.HKEYType:
    """Get or create a registry key."""
    try:
        return winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_ALL_ACCESS)
    except FileNotFoundError:
        return winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path)


def save_daminion_credentials(url: str, username: str, password: str) -> bool:
    """
    Persist Daminion server credentials to the Windows Registry.
    
    This function creates or opens the Synapic subkey and saves the 
    URL, username, and password as string values (`REG_SZ`).
    
    Args:
        url: The full Daminion server endpoint (e.g., http://dam.local/daminion).
        username: The account username for authentication.
        password: The account password (stored in plain text within the Registry).
        
    Returns:
        bool: True if the operation succeeded, False otherwise.
    """
    try:
        with _get_or_create_key(DAMINION_SUBKEY) as key:
            winreg.SetValueEx(key, "URL", 0, winreg.REG_SZ, url)
            winreg.SetValueEx(key, "Username", 0, winreg.REG_SZ, username)
            winreg.SetValueEx(key, "Password", 0, winreg.REG_SZ, password)
        
        logger.info(f"Saved Daminion credentials to registry for {url}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save Daminion credentials to registry: {e}")
        return False


def load_daminion_credentials() -> Optional[Dict[str, str]]:
    """
    Retrieve stored Daminion credentials from the Windows Registry.
    
    Attempts to read the Synapic subkey. If the key or values are missing,
    it returns None instead of raising an error.
    
    Returns:
        Optional[Dict]: A dictionary containing 'url', 'username', and 
                        'password', or None if not found.
    """
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, DAMINION_SUBKEY, 0, winreg.KEY_READ) as key:
            url, _ = winreg.QueryValueEx(key, "URL")
            username, _ = winreg.QueryValueEx(key, "Username")
            password, _ = winreg.QueryValueEx(key, "Password")
            
        return {
            "url": url,
            "username": username,
            "password": password
        }
        
    except FileNotFoundError:
        logger.debug("No Daminion credentials found in registry")
        return None
    except Exception as e:
        logger.error(f"Failed to load Daminion credentials from registry: {e}")
        return None


def delete_daminion_credentials() -> bool:
    """
    Purge Daminion credentials from the Windows Registry.
    
    Removes the entire Daminion subkey. This is typically used when the 
    user wants to disconnect or clear their local session data.
    
    Returns:
        bool: True if the key was deleted (or didn't exist), False on error.
    """
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, DAMINION_SUBKEY)
        logger.info("Deleted Daminion credentials from registry")
        return True
    except FileNotFoundError:
        logger.debug("No Daminion credentials to delete")
        return True
    except Exception as e:
        logger.error(f"Failed to delete Daminion credentials from registry: {e}")
        return False


def credentials_exist() -> bool:
    """
    Check if Daminion credentials are currently stored in the Registry.
    
    Returns:
        bool: True if the Daminion subkey can be opened for reading.
    """
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, DAMINION_SUBKEY, 0, winreg.KEY_READ):
            return True
    except FileNotFoundError:
        return False


def save_ui_preferences(preferences: Dict[str, Any]) -> bool:
    """
    Persist lightweight UI preferences to the Windows Registry.

    Boolean values are stored as DWORDs. String values are stored as REG_SZ.
    """
    try:
        with _get_or_create_key(UI_PREFS_SUBKEY) as key:
            for name, value in preferences.items():
                if isinstance(value, bool):
                    winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, int(value))
                else:
                    winreg.SetValueEx(key, name, 0, winreg.REG_SZ, str(value))

        logger.info("Saved UI preferences to registry")
        return True
    except Exception as e:
        logger.error(f"Failed to save UI preferences to registry: {e}")
        return False


def load_ui_preferences() -> Dict[str, Any]:
    """
    Load persisted UI preferences from the Windows Registry.

    Returns an empty dict when no preferences exist.
    """
    preferences: Dict[str, Any] = {}
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, UI_PREFS_SUBKEY, 0, winreg.KEY_READ) as key:
            index = 0
            while True:
                try:
                    name, value, reg_type = winreg.EnumValue(key, index)
                    if reg_type == winreg.REG_DWORD:
                        preferences[name] = bool(value)
                    else:
                        preferences[name] = value
                    index += 1
                except OSError:
                    break
    except FileNotFoundError:
        logger.debug("No UI preferences found in registry")
    except Exception as e:
        logger.error(f"Failed to load UI preferences from registry: {e}")

    return preferences
