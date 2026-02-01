# src/event_manager.py

import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config_loader import PROJECT_ROOT
from .logger_setup import logger

# --- Global Variables & Constants ---

# Define the path to the events file. JSON Lines format is used for efficient appends.
_EVENTS_FILE_PATH = PROJECT_ROOT / "data" / "events.jsonl"

# A lock to ensure file I/O operations are atomic and thread-safe.
_EVENTS_LOCK = threading.Lock()


# --- Public Functions ---

def add_item(item: Dict[str, Any]) -> None:
    """
    Thread-safely appends a new event item to the events.jsonl file.

    Inputs:
        item (Dict[str, Any]): A dictionary representing the event to be scheduled.
                               It must contain a unique 'item_id'.

    Outputs:
        None
    """
    with _EVENTS_LOCK:
        try:
            # Ensure the parent directory exists.
            _EVENTS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            # Open in append mode and write the new item as a single line.
            with open(_EVENTS_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item) + '\n')
            
            logger.info(f"Added new event with ID '{item.get('item_id')}' to the event log.")

        except Exception as e:
            logger.error(f"Failed to add item to event log: {e}", exc_info=True)
            raise

def get_all_items(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Thread-safely reads all event items from the events.jsonl file.

    Inputs:
        status (Optional[str]): If provided, filters the returned items to only those
                                with a matching 'status' field.

    Outputs:
        A list of dictionaries, where each dictionary is an event item. Returns
        an empty list if the file doesn't exist or is empty.
    """
    with _EVENTS_LOCK:
        items = []
        try:
            if not _EVENTS_FILE_PATH.exists():
                return []

            with open(_EVENTS_FILE_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            items.append(item)
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping corrupted line in event log: {line}")
            
            if status:
                items = [item for item in items if item.get('status') == status]
            
            return items

        except Exception as e:
            logger.error(f"Failed to get items from event log: {e}", exc_info=True)
            return []

def update_item_status(item_id: str, new_status: str) -> bool:
    """
    Thread-safely updates the status of a specific event item in the events.jsonl file.

    This operation reads all items, updates the target item in memory, and then
    rewrites the entire file.

    Inputs:
        item_id (str): The unique ID of the item to update.
        new_status (str): The new status to set for the item (e.g., 'COMPLETED', 'CANCELLED').

    Outputs:
        True if the item was found and updated successfully, False otherwise.
    """
    with _EVENTS_LOCK:
        try:
            if not _EVENTS_FILE_PATH.exists():
                logger.warning(f"Cannot update item '{item_id}': event log file does not exist.")
                return False

            # Read all items into memory.
            items_in_memory = []
            with open(_EVENTS_FILE_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line: items_in_memory.append(json.loads(line))


            item_found = False
            for item in items_in_memory:
                if item.get('item_id') == item_id:
                    item['status'] = new_status
                    item_found = True
                    break
            
            if not item_found:
                logger.warning(f"Could not find event with ID '{item_id}' to update its status.")
                return False

            # Rewrite the entire file with the updated list.
            with open(_EVENTS_FILE_PATH, 'w', encoding='utf-8') as f:
                for item in items_in_memory:
                    f.write(json.dumps(item) + '\n')
            
            logger.info(f"Updated status of event '{item_id}' to '{new_status}'.")
            return True

        except Exception as e:
            logger.error(f"Failed to update item status for ID '{item_id}': {e}", exc_info=True)
            return False