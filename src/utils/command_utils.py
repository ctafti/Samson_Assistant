# src/utils/command_utils.py

import time
import uuid
import json
from typing import Dict, Any

from src.config_loader import get_config
from src.logger_setup import logger

def queue_command_for_executor(command: Dict[str, Any]):
    """
    Writes a command dictionary to a file in the command_queue_dir for the
    CommandExecutorService to process.

    Args:
        command (Dict[str, Any]): The command to be queued.
    """
    try:
        # This function now becomes the single source of truth for this operation.
        command_queue_dir = get_config()['paths']['command_queue_dir']
        command_queue_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_ms = int(time.time() * 1000)
        unique_id = uuid.uuid4().hex[:8]
        command_file_path = command_queue_dir / f"cmd_{timestamp_ms}_{unique_id}.json"

        with open(command_file_path, 'w', encoding='utf-8') as f:
            json.dump(command, f, indent=2)
        
        logger.info(f"Queued command '{command.get('command_type')}' to file: {command_file_path.name}")
    except Exception as e:
        logger.error(f"Failed to queue command to file: {e}", exc_info=True)