# src/utils/file_io.py

import os
import json
import pathlib
import tempfile
from typing import Any
from src.logger_setup import logger

def atomic_json_dump(data: Any, file_path: pathlib.Path):
    """
    Atomically writes JSON data to a file by writing to a temporary file
    and then renaming it to the final destination. This prevents file
    corruption in case of an interruption.

    Args:
        data (Any): The JSON-serializable data to write.
        file_path (pathlib.Path): The final path of the file to save.
    """
    temp_file_path = None
    try:
        # Create a temporary file in the same directory to ensure atomic rename
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, dir=file_path.parent) as tmp_f:
            temp_file_path = pathlib.Path(tmp_f.name)
            json.dump(data, tmp_f, indent=2)

        # Atomically rename the temporary file to the final destination
        os.rename(temp_file_path, file_path)
        logger.debug(f"Atomically saved data to {file_path}")

    except Exception as e:
        logger.error(f"Failed to perform atomic write to {file_path}: {e}", exc_info=True)
        # Clean up the temporary file if it still exists after an error
        if temp_file_path and temp_file_path.exists():
            os.remove(temp_file_path)
        # Re-raise the exception so the caller knows the save failed
        raise