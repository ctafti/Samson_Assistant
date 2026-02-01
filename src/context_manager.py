# src/context_manager.py

import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from .config_loader import PROJECT_ROOT
from .logger_setup import logger

# --- Global Variables & Constants ---




# A lock to ensure that file I/O is thread-safe.
_CONTEXT_LOCK = threading.Lock()


# --- Public Functions ---

def get_active_context(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Thread-safely reads the global context from the context.json file.
    ...
    Inputs:
        config (Dict[str, Any]): The application's global configuration object.
    ...
    """
    context_file_path = config['paths']['system_state_dir'] / "context.json"
    
    with _CONTEXT_LOCK:
        try:
            # Ensure the parent directory exists.
            context_file_path.parent.mkdir(parents=True, exist_ok=True)
            if not context_file_path.exists():
                logger.info("Context file does not exist. No active context.")
                return None
            
            with open(context_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    logger.info("Context file is empty. No active context.")
                    return None
                
                # Rewind and parse the JSON content.
                f.seek(0)
                context_data = json.load(f)
                
            return context_data

        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON from context file: {context_file_path}. Treating as no active context.")
            return None
        except Exception as e:
            logger.error(f"Failed to read active context file: {e}", exc_info=True)
            return None

def set_active_context(matter_id: str, matter_name: str, source: str, environmental_context: str, config: Dict[str, Any]) -> None:
    """
    Thread-safely writes the new active context to the context.json file,
    overwriting any previous context.

    Inputs:
        matter_id (str): The unique identifier for the matter.
        matter_name (str): The human-readable name of the matter.
        source (str): The origin of the context change (e.g., 'scheduler', 'voice_command').
        environmental_context (str): The recording context, e.g., 'voip' or 'in_person'.
        config (Dict[str, Any]): The application's global configuration object.

    Outputs:
        None
    """
    context_file_path = config['paths']['system_state_dir'] / "context.json"
    with _CONTEXT_LOCK:
        try:
            # Construct the new context object with a timestamp.
            new_context = {
                "matter_id": matter_id,
                "matter_name": matter_name,
                "source": source,
                "environmental_context": environmental_context,
                "last_updated_utc": datetime.now(timezone.utc).isoformat()
            }
            
            # Ensure the parent directory exists.
            context_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the new context to the file.
            with open(context_file_path, 'w', encoding='utf-8') as f:
                json.dump(new_context, f, indent=2)
            
            logger.info(f"Active context set to matter '{matter_name}' (ID: {matter_id}, Env: {environmental_context}) by '{source}'.")

        except Exception as e:
            logger.error(f"Failed to set active context: {e}", exc_info=True)
            # Re-raise the exception to signal the failure to the caller.
            raise