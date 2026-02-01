import json
import threading
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.logger_setup import logger
from src.config_loader import PROJECT_ROOT, get_config
from src.utils.file_io import atomic_json_dump
from src.utils.file_locking import get_lock


_matters_file_path_cache: Optional[Path] = None


# A lock to ensure thread-safe read/write operations on the JSON file
MATTER_LOCK = get_lock("matters_db")


def _get_matters_file_path() -> Path:
    """Gets the path to matters.json from config, ensuring test isolation."""
    global _matters_file_path_cache
    if _matters_file_path_cache:
        return _matters_file_path_cache

    config = get_config()
    # First, check for an explicit file path for overrides.
    matters_path_str = config.get("paths", {}).get("matters_db_file")

    if matters_path_str:
        path = Path(matters_path_str)
    else:
        # If not explicitly set, look for the standard speaker_db_dir.
        speaker_db_dir_path = config.get("paths", {}).get("speaker_db_dir")
        if speaker_db_dir_path and isinstance(speaker_db_dir_path, Path):
            # This is the expected path for both tests and production.
            path = speaker_db_dir_path / "matters.json"
        else:
            # Fallback for legacy configurations or errors.
            logger.warning("Could not find 'paths.speaker_db_dir' in config, falling back to legacy matters.json path.")
            path = PROJECT_ROOT / "data" / "matters.json"

    _matters_file_path_cache = path
    return path


def _ensure_matters_file_exists():
    """Ensures the matters.json file and its parent directory exist."""
    matters_file_path = _get_matters_file_path()
    if not matters_file_path.exists():
        logger.info(f"Matters data file not found at {matters_file_path}. Creating a new empty file.")
        try:
            matters_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(matters_file_path, 'w', encoding='utf-8') as f:
                json.dump([], f)  # Initialize with an empty list
        except Exception as e:
            logger.error(f"Failed to create matters data file: {e}", exc_info=True)


def _load_matters() -> List[Dict[str, Any]]:
    """Loads all matters from the JSON file. Assumes the caller holds the lock."""
    _ensure_matters_file_exists()
    matters_file_path = _get_matters_file_path()
    try:
        with open(matters_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading or parsing {matters_file_path.name}: {e}", exc_info=True)
        return []


def _save_matters(matters_data: List[Dict[str, Any]]):
    """Saves the provided list of matters to the JSON file. Assumes the caller holds the lock."""
    matters_file_path = _get_matters_file_path()
    try:
        atomic_json_dump(matters_data, matters_file_path)
    except Exception as e:
        logger.error(f"Failed to save {matters_file_path.name}: {e}", exc_info=True)


def run_samson_orchestrator_startup_logic(config: Dict[str, Any]):
    """
    Ensures system-critical data, like the default matter, exists at startup.
    This is called by the main orchestrator or integration tests.

    DEPRECATED: Default matter is no longer a concept. This function is kept for compatibility but does nothing.
    """
    pass

# --- Public API Functions ---

def get_all_matters(include_inactive: bool = False) -> List[Dict[str, Any]]:
    """
    Returns a list of matters. By default, only returns 'active' matters.

    Args:
        include_inactive: If True, includes matters with 'inactive' status.

    Returns:
        A list of matter dictionaries.
    """
    with MATTER_LOCK:
        matters = _load_matters()
        if include_inactive:
            return matters
        return [m for m in matters if m.get('status') == 'active']


def get_matter_by_id(matter_id: str) -> Optional[Dict[str, Any]]:
    """Finds and returns a single matter by its unique matter_id, regardless of status."""
    with MATTER_LOCK:
        matters = _load_matters()
        return next((m for m in matters if m.get('matter_id') == matter_id), None)


def add_matter(matter_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds a new matter with default fields. If a matter with the same name
    already exists, it returns the existing matter instead of creating a duplicate.
    Returns the created or found matter dictionary.

    Args:
        matter_data: A dictionary containing matter properties, 'name' is required.

    Returns:
        The newly created or existing matter dictionary.
    """
    new_matter_name = matter_data.get("name")
    if not new_matter_name:
        logger.error("Attempted to add a matter without a 'name'. Creating entry, but this is indicative of an issue.")

    with MATTER_LOCK:
        matters = _load_matters()

        if new_matter_name:
            existing_matter = next((m for m in matters if m.get('name', '').lower() == new_matter_name.lower()), None)
            if existing_matter:
                logger.warning(f"Matter with name '{new_matter_name}' already exists with ID {existing_matter['matter_id']}. Not adding duplicate.")
                return existing_matter

        # If no existing matter is found, create the new one.
        new_matter = {
            # Use provided matter_id or generate a new one
            "matter_id": matter_data.get('matter_id', str(uuid.uuid4())),
            "status": "active",
            "source": "user_created",
            "description": "",
            "keywords": [],
        }
        # Update with the rest of the data, which might overwrite source etc.
        new_matter.update(matter_data)

        matters.append(new_matter)
        _save_matters(matters)
        logger.info(f"Successfully added new matter: {new_matter['matter_id']}")
        return new_matter

def update_matter(matter_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Updates an existing matter. Can also change the matter_id itself.

    Args:
        matter_id: The current ID of the matter to update.
        updates: A dictionary of fields to update.

    Returns:
        The updated matter dictionary on success, None if not found.
    """
    with MATTER_LOCK:
        matters = _load_matters()
        matter_to_update = None
        matter_index = -1

        for i, matter in enumerate(matters):
            if matter.get('matter_id') == matter_id:
                matter_to_update = matter
                matter_index = i
                break

        if matter_to_update is None:
            logger.warning(f"Matter with ID '{matter_id}' not found. Cannot update.")
            return None

        # Apply the updates
        matters[matter_index].update(updates)
        updated_matter = matters[matter_index]

        _save_matters(matters)
        logger.info(f"Successfully updated matter: {matter_id} -> {updated_matter.get('matter_id')}")
        return updated_matter


def delete_matter(matter_id: str) -> bool:
    """
    Soft-deletes a matter by setting its status to 'inactive'.

    Args:
        matter_id: The ID of the matter to delete.

    Returns:
        True on success, False if matter not found.
    """
    with MATTER_LOCK:
        matters = _load_matters()
        matter_found = False
        for matter in matters:
            if matter.get('matter_id') == matter_id:
                matter['status'] = 'inactive'
                matter_found = True
                break

        if not matter_found:
            logger.warning(f"Matter with ID '{matter_id}' not found. Cannot delete.")
            return False

        _save_matters(matters)
        logger.info(f"Successfully soft-deleted matter: {matter_id}")
    return True