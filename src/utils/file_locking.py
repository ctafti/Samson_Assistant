# File: src/utils/file_locking.py

import threading
from pathlib import Path
from filelock import FileLock, Timeout
from typing import Dict

from src.config_loader import get_config
from src.logger_setup import logger

_LOCK_DIR = Path(get_config()['paths']['system_state_dir']) / "locks"
_LOCK_DIR.mkdir(parents=True, exist_ok=True)

# Cache lock objects in memory to avoid re-creating them constantly
_in_process_locks: Dict[str, FileLock] = {}
_cache_lock = threading.Lock()

def get_lock(resource_name: str) -> FileLock:
    """
    Gets a process-safe FileLock for a given resource name.

    Args:
        resource_name (str): A unique name for the resource to be locked (e.g., "matters_db", "daily_log_2023-10-28").

    Returns:
        A FileLock object for the specified resource.
    """
    with _cache_lock:
        if resource_name not in _in_process_locks:
            # Sanitize the resource name to be a valid filename
            sanitized_name = "".join(c for c in resource_name if c.isalnum() or c in ('-', '_')).rstrip()
            lock_file_path = _LOCK_DIR / f"{sanitized_name}.lock"
            # Create a FileLock object. The lock is acquired/released using a 'with' statement.
            _in_process_locks[resource_name] = FileLock(lock_file_path, timeout=10) # 10-second timeout
        return _in_process_locks[resource_name]