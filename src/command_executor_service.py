import threading
import time
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Tuple
from datetime import datetime, timezone
from thefuzz import process as fuzz_process
from ruamel.yaml import YAML
import pytz
import dateparser
import uuid
import re
import sys
import requests
import ast

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.task_intelligence_manager import (
    _load_tasks, 
    _get_tasks_index_path, 
    _get_tasks_map_path, 
    TASK_INDEX_LOCK
)
from src.utils.file_io import atomic_json_dump

from src.logger_setup import logger
from src.daily_log_manager import update_chunk_metadata, update_matter_for_recent_time_window, update_matter_segments_for_chunk, get_daily_log_data
from src.matter_manager import get_all_matters, get_matter_by_id, add_matter
from src.feedback_manager import log_matter_correction
from src.config_loader import CONFIG_FILE_PATH, PROJECT_ROOT
from src.context_manager import set_active_context
from src.event_manager import add_item as add_scheduled_item
from src.task_intelligence_manager import TaskIntelligenceManager, _load_tasks as load_all_tasks, _save_tasks as save_all_tasks
from src.utils.command_utils import queue_command_for_executor
from src.llm_interface import get_llm_chat_model

class CommandExecutorService:
    def __init__(
        self,
        config: Dict[str, Any],
        shutdown_event: threading.Event,
        queue_item_function: Callable,
        embedding_model: SentenceTransformer
    ):
        self.config = config
        self.shutdown_event = shutdown_event
        self.queue_item_function = queue_item_function
        self.embedding_model = embedding_model
        self.command_queue_dir = Path(config['paths']['command_queue_dir'])
        self.poll_interval = config.get('services', {}).get('command_executor', {}).get('poll_interval_s', 5)
        
        self.command_handlers: Dict[str, Callable] = {
            "SET_MATTER": self._handle_set_matter,
            "FORCE_SET_MATTER": self._handle_force_set_matter,
            "UPDATE_CONFIG": self._handle_update_config,
            "UPDATE_MATTER_FOR_SPAN": self._handle_update_matter_for_span,
            "SCHEDULE_MATTER": self._handle_schedule_matter,
            "EXTRACT_TASKS_FROM_CHUNK": self._handle_extract_tasks_from_chunk,
            "SCHEDULE_TASK_CONFIRMATION": self._handle_schedule_task_confirmation,
            "RELINK_TASKS_FOR_MATTER": self._handle_relink_tasks_for_matter,
            "REBUILD_TASK_INDEX": self._handle_rebuild_task_index,
            "EXECUTE_WINDMILL_WORKFLOW": self._handle_execute_windmill_workflow,
            "GENERATE_WORKFLOW_FROM_PROMPT": self._handle_generate_workflow_from_prompt,
            "RENAME_WORKFLOW_FILE": self._handle_rename_workflow_file,
            "DELETE_WORKFLOW_FILE": self._handle_delete_workflow_file,
        }
        self.max_retries = 3 # Configurable retry count


    def _handle_rebuild_task_index(self, command_data: Dict[str, Any]) -> bool:
        """
        Performs a full rebuild of the task FAISS index and map from the tasks.jsonl file.
        This is a potentially long-running, non-blocking operation.
        """
        logger.info("COMMAND_EXECUTOR: Starting full rebuild of the task vector index.")

        if not self.embedding_model:
            logger.error("Cannot rebuild task index: embedding model is not available to the service.")
            return True # Do not retry, this is a startup configuration error.

        try:
            all_tasks = load_all_tasks()
            with TASK_INDEX_LOCK:
                # 2. Get file paths and clean up old index files
                index_file = _get_tasks_index_path()
                map_file = _get_tasks_map_path()

                if index_file.exists():
                    os.remove(index_file)
                if map_file.exists():
                    os.remove(map_file)
                
                logger.info("Cleared old task index and map files.")

                # 3. Process the loaded tasks
                if not all_tasks:

                    logger.warning("No tasks found in tasks.jsonl. The new index will be empty.")
                    # Create empty files to signify a successful empty build
                    index_file.parent.mkdir(parents=True, exist_ok=True)
                    index_file.touch()
                    with open(map_file, 'w') as f:
                        json.dump({}, f)
                    return True

                # 3. Initialize new index and map
                embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                new_index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
                new_map = {} # Maps sequential int ID -> string task_id UUID
                
                vectors_list = []
                faiss_ids_list = []

                # 4. Generate embeddings for all tasks
                for i, task in enumerate(all_tasks):
                    task_id_str = task.get("task_id")
                    if not task_id_str:
                        continue
                    
                    text_to_embed = f"{task.get('title', '')} {task.get('description', '')}".strip()
                    if not text_to_embed:
                        continue
                    
                    vector = self.embedding_model.encode(text_to_embed, normalize_embeddings=False)
                    vectors_list.append(vector.astype('float32'))
                    
                    faiss_id = i # Use simple sequential integer as the FAISS ID
                    faiss_ids_list.append(faiss_id)
                    new_map[faiss_id] = task_id_str

                if not vectors_list:
                    logger.warning("No valid tasks with text content found to index.")
                    return True

                # 5. Normalize and add to index in a batch
                vectors_np = np.vstack(vectors_list)
                faiss.normalize_L2(vectors_np)
                
                faiss_ids_np = np.array(faiss_ids_list, dtype='int64')
                new_index.add_with_ids(vectors_np, faiss_ids_np)

                # 6. Atomically save the new index and map
                temp_index_path = index_file.with_suffix(index_file.suffix + '.tmp')
                temp_map_path = map_file.with_suffix(map_file.suffix + '.tmp')
                
                faiss.write_index(new_index, str(temp_index_path))
                with open(temp_map_path, 'w', encoding='utf-8') as f:
                    json.dump(new_map, f, indent=2)

                os.rename(temp_index_path, index_file)
                os.rename(temp_map_path, map_file)

            logger.info(f"Successfully rebuilt task index with {new_index.ntotal} tasks.")
            return True

        except Exception as e:
            logger.error(f"Critical error during task index rebuild: {e}", exc_info=True)
            return False # Return False to indicate a retryable error
    def run(self):
        logger.info("Command Executor Service started.")
        while not self.shutdown_event.is_set():
            try:
                self._process_queue()
            except Exception as e:
                logger.error(f"Error in Command Executor Service run loop: {e}", exc_info=True)
            
            self.shutdown_event.wait(self.poll_interval)
        logger.info("Command Executor Service shutting down.")

    def _process_queue(self):
        command_files = sorted(self.command_queue_dir.glob("cmd_*.json"))
        for cmd_file in command_files:
            if self.shutdown_event.is_set(): break
            
            try:
                with open(cmd_file, 'r', encoding='utf-8') as f:
                    command_data = json.load(f)
                logger.info(f"EXECUTOR: Loaded raw command data from {cmd_file.name}: {json.dumps(command_data, indent=2)}")
                
                # Support both 'type' and 'command_type' for backwards compatibility
                cmd_type = command_data.get('type', command_data.get('command_type'))
                handler = self.command_handlers.get(cmd_type)
                
                if handler:
                    logger.info(f"Processing command '{cmd_type}' from file {cmd_file.name}")
                    success = handler(command_data) # This can raise exceptions or return False
                    
                    if success:
                        os.remove(cmd_file)
                        logger.info(f"Successfully processed and removed command file: {cmd_file.name}")
                    else:
                        # Handler reported a retryable failure
                        raise RuntimeError(f"Handler for '{cmd_type}' returned False, indicating a retryable error.")
                else:
                    logger.warning(f"No handler for command type '{cmd_type}'. Deleting unknown command file: {cmd_file.name}")
                    os.remove(cmd_file)

            except Exception as e:
                logger.error(f"Failed to process command file {cmd_file.name}: {e}", exc_info=True)
                
                # Retry logic
                match = re.search(r'\.failed\.(\d+)$', cmd_file.name)
                retries = int(match.group(1)) + 1 if match else 1
                
                if retries > self.max_retries:
                    logger.critical(f"Command {cmd_file.name} failed after {self.max_retries} retries. Moving to error directory.")
                    error_dir = self.command_queue_dir / "error"
                    error_dir.mkdir(exist_ok=True)
                    try:
                        shutil.move(str(cmd_file), str(error_dir / cmd_file.name))
                    except Exception as move_err:
                        logger.error(f"Could not move failed command {cmd_file.name} to error dir: {move_err}")
                else:
                    new_name = re.sub(r'(\.failed\.\d+)?$', f'.failed.{retries}', cmd_file.name)
                    new_path = cmd_file.with_name(new_name)
                    try:
                        os.rename(cmd_file, new_path)
                        logger.warning(f"Command {cmd_file.name} failed, renamed to {new_path.name} for retry.")
                    except OSError as rename_err:
                         logger.error(f"Could not rename failed command file {cmd_file.name}: {rename_err}")
    
    def _find_matter_by_fuzzy_name(self, spoken_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Finds the canonical matter_id from a spoken name using fuzzy matching."""
        all_matters = get_all_matters()
        if not all_matters:
            return None, None

        choices = {}
        for matter in all_matters:
            choices[matter['name'].lower()] = matter
            for alias in matter.get('aliases', []):
                choices[alias.lower()] = matter
        
        best_match = fuzz_process.extractOne(spoken_name.lower(), choices.keys())
        
        if best_match and best_match[1] > 80:
            matched_name = best_match[0]
            canonical_matter = choices[matched_name]
            logger.info(f"Fuzzy matched spoken matter '{spoken_name}' to canonical name '{canonical_matter['name']}' with score {best_match[1]}.")
            return canonical_matter['matter_id'], canonical_matter['name']
        
        logger.warning(f"Could not find a confident matter match for spoken name: '{spoken_name}'. Best guess was '{best_match[0] if best_match else 'None'}' with score {best_match[1] if best_match else 0}.")
        return None, None

    def _handle_force_set_matter(self, command_data: Dict[str, Any]) -> bool:
        """Handles the FORCE_SET_MATTER command for vocal override."""
        payload = command_data.get('payload', {})
        spoken_matter_name = payload.get('matter_id')
        date_str = command_data.get('processing_date')
        chunk_id = command_data.get('source_chunk_id')

        if not all([spoken_matter_name, date_str]):
            logger.error(f"Invalid FORCE_SET_MATTER command payload: {command_data}")
            return False

        canonical_matter_id, _ = self._find_matter_by_fuzzy_name(spoken_matter_name)
        if not canonical_matter_id:
            logger.error(f"FORCE_SET_MATTER failed: Could not resolve '{spoken_matter_name}' to a known matter.")
            return False

        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError:
            logger.error(f"Invalid date format in FORCE_SET_MATTER command: {date_str}")
            return False

        lookback_from_payload = payload.get('lookback_seconds')
        default_lookback = self.config.get('voice_commands', {}).get('vocal_override_lookback_seconds', 60)
        lookback_seconds = lookback_from_payload if lookback_from_payload is not None else default_lookback

        success = update_matter_for_recent_time_window(
            new_matter_id=canonical_matter_id,
            lookback_seconds=lookback_seconds,
            target_date=target_date
        )

        if success:
            logger.info(f"Successfully applied FORCE_SET_MATTER for '{canonical_matter_id}' over the last {lookback_seconds}s.")
            log_matter_correction(
                "vocal_override",
                {
                    "new_matter_id": canonical_matter_id,
                    "source_chunk_id": chunk_id,
                    "lookback_seconds": lookback_seconds,
                    "source": "voice_command_FORCE_SET_MATTER"
                }
            )
        else:
            logger.error(f"Failed to apply FORCE_SET_MATTER for '{canonical_matter_id}'.")
        
        return success

    def _handle_set_matter(self, command_data: Dict[str, Any]) -> bool:
        """Handles the SET_MATTER command by updating the persistent global context."""
        payload = command_data.get('payload', {})
        matter_id_from_payload = payload.get('matter_id')
        matter_name_from_payload = payload.get('matter_name')
        environmental_context = payload.get('environmental_context', 'in_person') # Default context
        source = command_data.get('source', 'unknown')
        source_event_id = command_data.get('source_event_id')

        if not matter_id_from_payload and not matter_name_from_payload:
            logger.error(f"Invalid SET_MATTER command payload, missing 'matter_id' or 'matter_name': {command_data}")
            return True # Discard malformed command

        canonical_matter_id = None
        canonical_matter_name = None

        # AC5: Prioritize canonical ID lookup
        if matter_id_from_payload:
            matter_obj = get_matter_by_id(matter_id_from_payload)
            if matter_obj:
                canonical_matter_id = matter_obj['matter_id']
                canonical_matter_name = matter_obj['name']
        
        # Fallback to fuzzy name matching if ID lookup fails or wasn't provided
        if not canonical_matter_id:
            name_to_search = matter_name_from_payload or matter_id_from_payload # Use whichever is available
            if name_to_search:
                canonical_matter_id, canonical_matter_name = self._find_matter_by_fuzzy_name(name_to_search)

        if not canonical_matter_id:
            logger.error(f"SET_MATTER command failed: Could not resolve '{matter_id_from_payload or matter_name_from_payload}' to a known matter.")
            return True # Discard

        try:
            # Pass the config object to the refactored context manager function
            set_active_context(
                matter_id=canonical_matter_id,
                matter_name=canonical_matter_name,
                source=source,
                environmental_context=environmental_context,
                config=self.config
            )
            logger.info(f"Global context successfully set to matter: '{canonical_matter_name}' (ID: {canonical_matter_id}, Env: {environmental_context}).")

            if source == 'scheduler' and source_event_id:
                # ... (rest of function is the same, no changes needed here) ...
                from . import event_manager
                event_manager.update_item_status(source_event_id, "COMPLETED")
            
            return True
        except Exception as e:
            logger.error(f"Failed to set active context: {e}", exc_info=True)
            return False # Retry on failure

    def _handle_schedule_matter(self, command_data: Dict[str, Any]) -> bool:
        """Handles scheduling a future matter context switch."""
        payload = command_data.get('payload', {})
        matter_name_query = payload.get('matter_name')
        time_string = payload.get('time_string')
        source_info = payload.get('source', 'unknown')
        # Extract environmental_context from the payload
        environmental_context = payload.get('environmental_context', 'in_person')

        if not matter_name_query or not time_string:
            logger.error(f"Invalid SCHEDULE_MATTER command payload: {payload}")
            return True # Discard malformed command

        canonical_matter_id, canonical_matter_name = self._find_matter_by_fuzzy_name(matter_name_query)
        if not canonical_matter_id:
            logger.info(f"SCHEDULE_MATTER: Could not resolve '{matter_name_query}'. Assuming it's a new matter and creating it.")
            # We could send a failure message back here if needed in the future
            new_matter_details = add_matter({
                "name": matter_name_query,
                "description": f"Auto-created from '{source_info}' command.",
                "source": source_info
            })
            if not new_matter_details:
                logger.error(f"SCHEDULE_MATTER: Failed to auto-create new matter '{matter_name_query}'. Command will be retried.")
                return False # Return False to retry the command
            
            canonical_matter_id = new_matter_details['matter_id']
            canonical_matter_name = new_matter_details['name']
            logger.info(f"Successfully auto-created new matter '{canonical_matter_name}' (ID: {canonical_matter_id}).")
        
        try:
            tz_str = self.config.get('timings', {}).get('assumed_recording_timezone', 'UTC')
            local_tz = pytz.timezone(tz_str)
            now_in_local_tz = datetime.now(local_tz)

            time_string_to_parse = time_string
            # Heuristic to improve robustness: If it's currently afternoon/evening and the user
            # gives an ambiguous low-number hour (1-6) without an AM/PM specifier or colon,
            # assume they mean PM. This prevents "meeting at 2" from being scheduled for 2 AM.
            match = re.search(r'\b(at|for|around)\s+([1-6])\b(?!\s*[:ap])', time_string.lower())
            if match and now_in_local_tz.hour >= 12:
                logger.info(f"Ambiguous time '{match.group(0)}' detected after noon. Appending 'PM' to assist parser.")
                time_string_to_parse += " PM"
            
            # Use dateparser to interpret the natural language time string
            target_dt_local = dateparser.parse(
                time_string,
                languages=['en'],
                settings={'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': now_in_local_tz}
            )
            
            if not target_dt_local:
                logger.error(f"Could not parse time string: '{time_string}'")
                return True

            # Ensure the datetime object is timezone-aware before converting to UTC
            if target_dt_local.tzinfo is None:
                target_dt_local = local_tz.localize(target_dt_local)
            
            target_dt_utc = target_dt_local.astimezone(pytz.utc)

            event_item = {
                "item_id": str(uuid.uuid4()),
                "item_type": "MEETING",
                "status": "SCHEDULED",
                "title": f"Switch to Matter: {canonical_matter_name}",
                "description": f"Scheduled via {source_info}.",
                "matter_id": canonical_matter_id,
                "start_time_utc": target_dt_utc.isoformat(),
                "end_time_utc": None,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "metadata": {  # Store context in the metadata field
                    "environmental_context": environmental_context
                },
                "source": {
                    "type": source_info,
                    "original_text": f"matter: '{matter_name_query}', time: '{time_string}'"
                }
            }
            
            add_scheduled_item(event_item)
            logger.info(f"Successfully scheduled event '{event_item['item_id']}' to switch matter to '{canonical_matter_name}' (Env: {environmental_context}) at {target_dt_utc.isoformat()}.")
            return True

        except Exception as e:
            logger.error(f"Error processing SCHEDULE_MATTER command: {e}", exc_info=True)
            return False # Retry on failure
        
    def _handle_update_config(self, command_data: Dict[str, Any]) -> bool:
        """Handles the UPDATE_CONFIG command to safely modify config.yaml."""
        payload = command_data.get('payload', {})
        if not payload:
            logger.warning("UPDATE_CONFIG command received with an an empty payload.")
            return True

        logger.info(f"Applying {len(payload)} configuration changes from the GUI.")
        try:
            yaml = YAML()
            yaml.preserve_quotes = True
            with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                config_data = yaml.load(f)

            for key_path, new_value in payload.items():
                keys = key_path.split('.')
                current_level = config_data
                for i, key in enumerate(keys[:-1]):
                    if key not in current_level:
                        logger.error(f"Invalid key path in UPDATE_CONFIG: '{key}' not found in '{'.'.join(keys[:i+1])}'.")
                        current_level = None
                        break
                    current_level = current_level[key]

                if current_level is not None:
                    final_key = keys[-1]
                    logger.info(f"Updating config: {key_path} = {new_value} (type: {type(new_value)})")
                    current_level[final_key] = new_value

            with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f)
            
            logger.info("Successfully saved updated configuration to config.yaml. A restart is required for changes to take effect.")
            return True

        except Exception as e:
            logger.error(f"Failed to update config.yaml: {e}", exc_info=True)
            return False

    def _handle_update_matter_for_span(self, command_data: Dict[str, Any]) -> bool:
        """Handles commands from the GUI to update a matter for a specific time span."""
        payload = command_data.get('payload', {})
        target_date_str = payload.get('target_date_str')
        chunk_id = payload.get('chunk_id')
        start_time = payload.get('start_time')
        end_time = payload.get('end_time')
        new_matter_id = payload.get('new_matter_id')

        if not all([target_date_str, chunk_id, start_time is not None, end_time is not None, new_matter_id]):
            logger.error(f"Invalid UPDATE_MATTER_FOR_SPAN payload: {payload}")
            return False

        try:
            target_date = datetime.strptime(target_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            
            success = update_matter_segments_for_chunk(
                target_date=target_date,
                chunk_id=chunk_id,
                start_time=float(start_time),
                end_time=float(end_time),
                new_matter_id=new_matter_id
            )
            return success
        except (ValueError, TypeError) as e:
            logger.error(f"Error processing UPDATE_MATTER_FOR_SPAN command: {e}", exc_info=True)
            return False
    
    def _handle_extract_tasks_from_chunk(self, command_data: Dict[str, Any]) -> bool:
        """Loads a transcript chunk and triggers the task extraction process."""
        payload = command_data.get('payload', {})
        chunk_id = payload.get('chunk_id')
        date_str = payload.get('processing_date')
        if not all([chunk_id, date_str]):
            logger.error("Invalid EXTRACT_TASKS_FROM_CHUNK command, missing data.")
            return True  # Discard malformed command, don't retry

        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            daily_log = get_daily_log_data(target_date)
            chunk_data = daily_log.get("chunks", {}).get(chunk_id)

            if not chunk_data:
                logger.error(f"Could not find chunk {chunk_id} in log for {date_str} to extract tasks.")
                return False # Could be a race condition, retry

            task_manager = TaskIntelligenceManager(self.config, self.embedding_model)
            newly_created_task_ids = task_manager.extract_and_save_tasks(chunk_data)

            if newly_created_task_ids:
                # Traceability: Log generated task IDs back to the daily log via main worker
                log_command = {
                    "type": "LOG_TASK_CREATION",
                    "payload": {
                        "chunk_id": chunk_id,
                        "processing_date": date_str,
                        "task_ids": newly_created_task_ids
                    }
                }
                self.queue_item_function(1, log_command) # Normal priority

                # Schedule auto-confirmation for each new task
                cooldown_s = self.config.get('task_intelligence', {}).get('task_creation_cooldown_s', 300)
                confirmation_time_str = f"in {cooldown_s} seconds"
                for task_id in newly_created_task_ids:
                    schedule_command = {
                        "command_type": "SCHEDULE_TASK_CONFIRMATION",
                        "payload": {
                            "task_id": task_id,
                            "time_string": confirmation_time_str
                        }
                    }
                    queue_command_for_executor(schedule_command) 
            return True
        except Exception as e:
            logger.error(f"Failed to process task extraction for chunk {chunk_id}: {e}", exc_info=True)
            return False # Retry on failure

    def _handle_schedule_task_confirmation(self, command_data: Dict[str, Any]) -> bool:
        """Schedules the automatic confirmation of a pending task."""
        payload = command_data.get('payload', {})
        task_id = payload.get('task_id')
        time_string = payload.get('time_string')

        if not all([task_id, time_string]):
            logger.error(f"Invalid SCHEDULE_TASK_CONFIRMATION command: {payload}")
            return True # Discard malformed command
        
        try:
            tz_str = self.config.get('timings', {}).get('assumed_recording_timezone', 'UTC')
            local_tz = pytz.timezone(tz_str)
            now_in_local_tz = datetime.now(local_tz)

            target_dt_local = dateparser.parse(
                time_string,
                settings={'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': now_in_local_tz}
            )
            if not target_dt_local:
                logger.error(f"Could not parse time_string for task confirmation: '{time_string}'")
                return True
            
            if target_dt_local.tzinfo is None:
                target_dt_local = local_tz.localize(target_dt_local)
            
            target_dt_utc = target_dt_local.astimezone(pytz.utc)

            event_item = {
                "item_id": str(uuid.uuid4()),
                "item_type": "TASK_CONFIRMATION",
                "status": "SCHEDULED",
                "title": f"Auto-confirm task {task_id}",
                "start_time_utc": target_dt_utc.isoformat(),
                "metadata": {"task_id": task_id},
            }
            add_scheduled_item(event_item)
            logger.info(f"Successfully scheduled auto-confirmation for task {task_id} at {target_dt_utc.isoformat()}.")
            return True
        except Exception as e:
            logger.error(f"Error processing SCHEDULE_TASK_CONFIRMATION: {e}", exc_info=True)
            return False # Retry

    def _handle_relink_tasks_for_matter(self, command_data: Dict[str, Any]) -> bool:
        """Retroactively updates matter_id for all tasks based on their source logs."""
        logger.warning("Starting retroactive task relinking. This may be a slow operation.")
        try:
            all_tasks = load_all_tasks()
            daily_log_cache = {}
            tasks_updated = 0

            for task in all_tasks:
                for ref in task.get("source_references", []):
                    if ref.get("source_type") != "transcript":
                        continue
                    
                    chunk_id = ref.get("chunk_id")
                    # Assuming chunk_id contains the date, e.g., 'CHUNK_20240101_...'
                    match = re.search(r'_(\d{8})_', chunk_id)
                    if not match: continue
                    
                    date_str = match.group(1)
                    if date_str not in daily_log_cache:
                        dt_obj = datetime.strptime(date_str, '%Y%m%d').replace(tzinfo=timezone.utc)
                        daily_log_cache[date_str] = get_daily_log_data(dt_obj)

                    log = daily_log_cache[date_str]
                    chunk_data = log.get("chunks", {}).get(chunk_id)
                    if not chunk_data: continue

                    # We assume the matter context is consistent for the whole chunk for now
                    chunk_matter_id = chunk_data.get("active_matter_context", {}).get("matter_id")
                    
                    if chunk_matter_id and task.get("matter_id") != chunk_matter_id:
                        logger.info(f"Relinking task {task['task_id']}: old matter '{task.get('matter_id')}' -> new matter '{chunk_matter_id}'.")
                        task['matter_id'] = chunk_matter_id
                        # We can also update matter_name for UI convenience
                        matter_details = get_matter_by_id(chunk_matter_id)
                        if matter_details:
                            task['matter_name'] = matter_details.get('name')
                        tasks_updated += 1
                        break # Go to next task once updated
            
            if tasks_updated > 0:
                save_all_tasks(all_tasks)
            
            logger.info(f"Retroactive task relinking complete. Updated {tasks_updated} tasks.")
            return True
        except Exception as e:
            logger.error(f"Error during RELINK_TASKS_FOR_MATTER: {e}", exc_info=True)
            return False
        
    def _handle_execute_windmill_workflow(self, command_data: Dict[str, Any]) -> bool:
        """Routes to CLI or API execution with automatic fallback."""
        execution_method = self.config.get('windmill', {}).get('execution_method', 'api')
        enable_fallback = self.config.get('windmill', {}).get('enable_fallback', True)
        
        if execution_method == 'cli':
            logger.info("Attempting CLI execution")
            success = self._execute_workflow_cli(command_data)
            
            if not success and enable_fallback:
                logger.warning("CLI execution failed. Falling back to API method.")
                return self._execute_workflow_api(command_data)
            
            return success
        else:
            logger.info("Using API execution method")
            logger.info(f"DEBUG: windmill config = {self.config.get('windmill', {})}")  
            return self._execute_workflow_api(command_data)

    def _execute_workflow_cli(self, command_data: Dict[str, Any]) -> bool:
        """Executes workflow using wmill CLI."""
        payload = command_data.get('payload', command_data.get('command_payload', {}))
        workflow_path = payload.get('workflow_path')
        input_params = payload.get('input_params', {})
        workspace = payload.get('workspace', 'samson')

        if not workflow_path:
            logger.error("EXECUTE_WINDMILL_WORKFLOW failed: 'workflow_path' is missing")
            return True

        try:
            full_file_path = PROJECT_ROOT / workflow_path
            
            if not full_file_path.exists():
                logger.error(f"Workflow file not found: {full_file_path}")
                return False
            
            # Derive windmill path from filename
            workflow_name = full_file_path.stem
            windmill_path = f"f/workflows/{workflow_name}"
            
            # Ensure #path: comment is in the file
            with open(full_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip().startswith('#path:'):
                # Prepend the path comment
                content = f"#path: {windmill_path}\n{content}"
                with open(full_file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Added #path: {windmill_path} to script file")
            
            # Push script via CLI (simplified command)
            logger.info(f"Pushing script via CLI: {windmill_path}")
            push_result = subprocess.run(
                ["wmill", "script", "push", str(full_file_path), "--workspace", workspace],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(PROJECT_ROOT)
            )
            
            if push_result.returncode != 0:
                logger.error(f"CLI push failed: {push_result.stderr}")
                return False
            
            logger.info(f"Script pushed successfully: {push_result.stdout.strip()}")
            
            # Execute via CLI
            logger.info(f"Executing workflow with params: {input_params}")
            exec_cmd = ["wmill", "script", "run", windmill_path, "--workspace", workspace]
            
            if input_params:
                exec_cmd.extend(["--data", json.dumps(input_params)])
            
            exec_result = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(PROJECT_ROOT)
            )
            
            if exec_result.returncode != 0:
                logger.error(f"CLI execution failed: {exec_result.stderr}")
                return False
            
            logger.info(f"Workflow completed. Output: {exec_result.stdout}")
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error in CLI execution: {e}", exc_info=True)
            return False

    def _execute_workflow_api(self, command_data: Dict[str, Any]) -> bool:
        """Executes workflow using Windmill REST API."""
        payload = command_data.get('payload', command_data.get('command_payload', {}))
        workflow_path = payload.get('workflow_path')
        input_params = payload.get('input_params', {})
        workspace = payload.get('workspace', 'samson')

        if not workflow_path:
            logger.error("EXECUTE_WINDMILL_WORKFLOW failed: 'workflow_path' is missing from payload.")
            return True

        try:
            full_file_path = PROJECT_ROOT / workflow_path
            
            if not full_file_path.exists():
                logger.error(f"Workflow file not found: {full_file_path}")
                return False
            
            # Read the script content
            try:
                with open(full_file_path, 'r', encoding='utf-8') as f:
                    script_content = f.read()
                    if not script_content.strip():
                        logger.error(f"Workflow file is empty: {full_file_path}")
                        return True
            except Exception as e:
                logger.error(f"Could not read workflow file: {e}")
                return False
            
            # The authoritative windmill path is derived from the filename.
            # This avoids duplicating the path inside the script content when sending to the API.
            workflow_name = full_file_path.stem
            windmill_path = f"f/workflows/{workflow_name}"
            logger.info(f"Derived windmill path from filename: {windmill_path}")

            # For backwards compatibility, remove the #path: line from content for the API call if it exists.
            lines = script_content.splitlines()
            if lines and lines[0].strip().startswith('#path:'):
                script_content = '\n'.join(lines[1:]).lstrip()
            
            # Create/update script using Windmill REST API
            import requests
            
            # Get token from wmill.yaml
            yaml = YAML()
            with open(PROJECT_ROOT / 'wmill.yaml', 'r') as f:
                wmill_config = yaml.load(f)
            
            token = wmill_config.get('remotes', {}).get('production', {}).get('token')
            base_url = wmill_config.get('remotes', {}).get('production', {}).get('remote', 'http://localhost:80')
            
            if not token:
                logger.error("No token found in wmill.yaml")
                return False
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            # Ensure folder structure exists before creating script
            path_parts = windmill_path.split("/")
            if len(path_parts) > 2:  # e.g., ["f", "workflows", "script_name"]
                # The folder name is the second part (index 1)
                folder_name = path_parts[1]
                logger.info(f"Ensuring folder exists: {folder_name}")
                
                folder_url = f"{base_url}/api/w/{workspace}/folders/create"
                folder_data = {"name": folder_name}
                
                folder_response = requests.post(folder_url, json=folder_data, headers=headers)
                if folder_response.status_code in [200, 201]:
                    logger.info(f"Successfully created folder: {folder_name}")
                elif folder_response.status_code == 409 or "already exists" in folder_response.text.lower():
                    logger.info(f"Folder {folder_name} already exists")
                else:
                    # Log warning but continue - folder might exist with different error code
                    logger.warning(f"Folder creation returned status {folder_response.status_code}: {folder_response.text}")
            
            # Prepare script data
            script_data = {
                "path": windmill_path,
                "summary": f"Auto-generated workflow",
                "description": "",
                "content": script_content,
                "language": "python3",
                "kind": "script",
                
            }
            
            # Try to create the script
            api_url = f"{base_url}/api/w/{workspace}/scripts/create"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"Creating script via API: {api_url}")
            response = requests.post(api_url, json=script_data, headers=headers)
            
            if response.status_code in [200, 201]:
                logger.info(f"Successfully created script {windmill_path}")
            elif response.status_code == 409 or (response.status_code == 400 and "path conflict" in response.text.lower()) or (response.status_code == 400 and "already exists" in response.text.lower()):
                # Script exists with different content, or has same content but we want to force a refresh. Delete then recreate.
                logger.info(f"Script {windmill_path} exists (status: {response.status_code}), deleting and recreating to ensure fresh state and dependency check...")
                
                # Delete the old script
                delete_url = f"{base_url}/api/w/{workspace}/scripts/delete/p/{windmill_path.replace('/', '%2F')}"
                delete_response = requests.post(delete_url, headers=headers)
                
                if delete_response.status_code in [200, 201]:
                    logger.info(f"Successfully deleted old script. Waiting briefly before recreating.")
                    time.sleep(1.0) # Add a 1-second delay to allow the deletion to fully propagate.
                    # Now create the new one
                    create_response = requests.post(
                        f"{base_url}/api/w/{workspace}/scripts/create",
                        json=script_data,
                        headers=headers
                    )
                    if create_response.status_code not in [200, 201]:
                        logger.error(f"Failed to recreate script: {create_response.status_code} - {create_response.text}")
                        return False
                else:
                    logger.error(f"Failed to delete script: {delete_response.status_code} - {delete_response.text}")
                    return False
            else:
                logger.error(f"Failed to create script: {response.status_code} - {response.text}")
                return False
            logger.info(f"Script operation completed, polling for readiness...")

            # Poll for script to be ready (Dependencies job must complete)
            max_wait = 60  # seconds
            poll_interval = 2  # second
            waited = 0
            ready = False
            
            while waited < max_wait:
                time.sleep(poll_interval)
                waited += poll_interval
                
                # Check if script is ready by trying to get its details
                check_url = f"{base_url}/api/w/{workspace}/scripts/get/p/{windmill_path.replace('/', '%2F')}"
                check_response = requests.get(check_url, headers=headers)
                
                if check_response.status_code == 200:
                    script_details = check_response.json()
                    dep_job = script_details.get("last_dependencies_job")
                    # A script is ready if it has no dependency job, or that job has completed successfully.
                    if not dep_job or (dep_job.get("result") and dep_job.get("result", {}).get("success")):
                        logger.info(f"Script dependencies are ready after {waited}s.")
                        ready = True
                        break
                    else:
                        job_id = dep_job.get("id", "N/A")
                        logger.debug(f"Script exists but dependencies job ({job_id}) is not complete. Waited {waited}s...")
                else:
                    logger.debug(f"Script metadata not found yet, waited {waited}s... (Status: {check_response.status_code})")

            if not ready:
                logger.warning(f"Script did not become ready after {max_wait}s, attempting execution anyway...")
            
            # Execute via API with retries for the 404 race condition
            run_url = f"{base_url}/api/w/{workspace}/jobs/run/p/{windmill_path.replace('/', '%2F')}"
            logger.info(f"Executing workflow via API: {run_url}")
            logger.info(f"Original path: {windmill_path}, Encoded path: {windmill_path.replace('/', '%2F')}")

            max_exec_retries = 5
            exec_retry_delay = 2.0  # seconds
            run_response = None

            for attempt in range(max_exec_retries):
                run_response = requests.post(
                    run_url,
                    json=input_params,
                    headers=headers,
                    timeout=30
                )
                if run_response.status_code != 404:
                    # It's either success or a real error, break the retry loop
                    break
                
                logger.warning(f"Execution attempt {attempt + 1}/{max_exec_retries} failed with 404. Retrying in {exec_retry_delay}s...")
                time.sleep(exec_retry_delay)

            # Check the final response after the loop
            if run_response is None or run_response.status_code not in [200, 201]:
                logger.error(f"Failed to run workflow: {run_response.status_code} - {run_response.text}")
                return False
            
            # If we reach here, the response was successful.
            job_id = run_response.text.strip('"')
            logger.info(f"Successfully started job {job_id} for workflow '{windmill_path}'")
            
            # Optionally poll for completion
            status_url = f"{base_url}/api/w/{workspace}/jobs/completed/get/{job_id}"
            max_wait = 300  # 5 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                time.sleep(2)
                status_response = requests.get(status_url, headers=headers)
                if status_response.status_code == 200:
                    result = status_response.json()
                    logger.info(f"Workflow completed: {result}")
                    return True
                elif status_response.status_code == 404:
                    # Job still running
                    continue
                else:
                    logger.error(f"Error checking job status: {status_response.text}")
                    return False
            
            logger.warning(f"Job {job_id} still running after {max_wait}s timeout")
            return True  # Job started successfully even if we don't wait for completion

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return False

    def _handle_generate_workflow_from_prompt(self, command_data: Dict[str, Any]) -> bool:
        """
        Handles generating a Windmill workflow script from a user prompt using an LLM.
        """
        logger.info(f"HANDLER_GENERATE_WORKFLOW: Received command data: {json.dumps(command_data, indent=2)}")
        
        payload = command_data.get('payload', command_data.get('command_payload', {}))
        logger.info(f"HANDLER_GENERATE_WORKFLOW: Extracted payload dictionary: {payload}")
        
        user_prompt = payload.get('user_prompt')
        logger.info(f"HANDLER_GENERATE_WORKFLOW: Extracted user_prompt: '{user_prompt}'")

        if not user_prompt:
            logger.error("GENERATE_WORKFLOW failed: 'user_prompt' is missing from payload.")
            return True

        try:
            # Get the configured profile name
            workflow_gen_profile_name = self.config.get('llm', {}).get('workflow_generator_profile')
            if not workflow_gen_profile_name:
                logger.error("Cannot generate workflow: 'llm.workflow_generator_profile' is not set in config.yaml.")
                return True

            # Get the actual profile configuration
            profiles = self.config.get('llm', {}).get('profiles', {})
            profile_config = profiles.get(workflow_gen_profile_name)
            
            if not profile_config:
                logger.error(f"Cannot generate workflow: Profile '{workflow_gen_profile_name}' not found in llm.profiles.")
                return True

            # Initialize the LLM based on provider
            provider = profile_config.get('provider')
            model_name = profile_config.get('model_name')
            temperature = profile_config.get('temperature', 0.2)
            
            if provider == "lmstudio":
                from langchain_openai.chat_models import ChatOpenAI
                llm = ChatOpenAI(
                    model_name=model_name,
                    temperature=float(temperature),
                    base_url=profile_config.get('base_url'),
                    api_key="lm-studio"
                )
            elif provider == "google_gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=profile_config.get('api_key'),
                    temperature=float(temperature)
                )
            elif provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(
                    model=model_name,
                    anthropic_api_key=profile_config.get('api_key'),
                    temperature=float(temperature)
                )
            else:
                logger.error(f"Unsupported provider '{provider}' for workflow generation.")
                return False

            logger.info(f"Successfully initialized LLM using profile '{workflow_gen_profile_name}'")

            # Generate safe filename and Windmill path
            safe_filename_base = re.sub(r'[^a-z0-9_]+', '_', user_prompt.lower())[:50]
            workflow_name = f"ai_{safe_filename_base}_{uuid.uuid4().hex[:6]}"
            windmill_path = f"f/workflows/{workflow_name}"

            # Load the prompt template
            prompt_path = PROJECT_ROOT / "src/prompts/generate_windmill_workflow_prompt.txt"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
            
            prompt = prompt_template.format(user_prompt=user_prompt)

            # Call the LLM
            logger.info(f"Invoking LLM to generate workflow for prompt: '{user_prompt[:80]}...'")
            response = llm.invoke(prompt)
            generated_code = response.content
            
            # Extract Python code
            code_match = re.search(r"```python\n(.*?)```", generated_code, re.DOTALL)
            if code_match:
                extracted_code = code_match.group(1).strip()
            else:
                extracted_code = generated_code.strip()

            if not extracted_code:
                logger.error("LLM returned an empty response or no valid code block.")
                return False
            
            # --- Validation Only: Check if AI included dependencies ---
            has_dependency_block = bool(re.search(r'# /// script\s*\n# dependencies', extracted_code))

            if has_dependency_block:
                logger.info("AI-generated code already includes dependency block. Skipping generation.")
                final_code = extracted_code
            else:
                logger.warning("AI did not generate dependency block. Creating one from AST parsing.")
            # --- Dependency Detection ---
                dependencies = set()
                try:
                    tree = ast.parse(extracted_code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                dependencies.add(alias.name.split('.')[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                dependencies.add(node.module.split('.')[0])
                except SyntaxError as e:
                    logger.warning(f"Could not parse generated code for dependencies due to syntax error: {e}. Proceeding without dependency block.")

                # Filter out standard libraries
                standard_libs = set(sys.stdlib_module_names) if sys.version_info >= (3, 10) else set()
                standard_libs.update(['sys', 'os', 'json', 'collections', 'datetime', 're', 'time', 'pathlib', 'typing', 'ast', 'uuid', 'threading'])

                package_map = {
                    "bs4": "beautifulsoup4", "PIL": "Pillow", "cv2": "opencv-python",
                    "skimage": "scikit-image", "sklearn": "scikit-learn", "yaml": "PyYAML",
                    "thefuzz": "thefuzz", "pypdf2": "PyPDF2"
                }
                final_dependencies = {package_map.get(dep.lower(), dep) for dep in dependencies if dep not in standard_libs}

                dependency_block = ""
                if final_dependencies:
                    deps_list_formatted = ',\n'.join([f'#   "{dep}"' for dep in sorted(list(final_dependencies))])
                    dependency_block = f"# /// script\n# dependencies = [\n{deps_list_formatted}\n# ]\n# ///"
                    logger.info(f"Detected dependencies and generated block:\n{dependency_block}")

                if dependency_block:
                    # DON'T include #path in the content - it's set via API
                    final_code = f"{dependency_block}\n\n{extracted_code}"
                else:
                    final_code = extracted_code

            # Save the generated code
            new_filename = f"{workflow_name}.py"
            workflows_dir = PROJECT_ROOT / "workflows"
            workflows_dir.mkdir(exist_ok=True)
            output_path = workflows_dir / new_filename

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_code)
            
            logger.info(f"Successfully generated and saved new workflow to: {output_path}")
            logger.info(f"Windmill path will be: {windmill_path}")
            return True

        except Exception as e:
            logger.error(f"An unexpected error occurred during workflow generation: {e}", exc_info=True)
            return False


    def _handle_rename_workflow_file(self, command_data: Dict[str, Any]) -> bool:
        """Handles renaming a workflow file."""
        payload = command_data.get('payload', command_data.get('command_payload', {}))
        relative_path_str = payload.get('workflow_path')
        new_name = payload.get('new_name')

        if not all([relative_path_str, new_name]):
            logger.error(f"Invalid RENAME_WORKFLOW_FILE payload: {payload}")
            return True # Discard malformed command

        try:
            old_path = PROJECT_ROOT / relative_path_str
            if not old_path.is_file():
                logger.error(f"Workflow file to rename not found at: {old_path}")
                return True

            # Extract UUID from old filename if it's an AI script pattern
            old_stem = old_path.stem
            uuid_part = None
            match = re.search(r'_([a-f0-9]{6})$', old_stem)
            if old_stem.startswith("ai_") and match:
                uuid_part = match.group(1)

            # Sanitize the new display name into a filesystem-safe stem
            # e.g., "AI: My New Workflow" -> "ai_my_new_workflow"
            processed_name = new_name
            if processed_name.lower().startswith("ai:"):
                processed_name = "ai_" + processed_name[3:].strip()
            
            new_stem = re.sub(r'[^a-z0-9_]+', '_', processed_name.lower()).strip('_')
            
            # If the original file had a UUID and the new name still implies it's an AI script, preserve the UUID.
            if uuid_part and new_stem.startswith("ai_"):
                new_stem = f"{new_stem}_{uuid_part}"

            if not new_stem:
                logger.error(f"New name '{new_name}' resulted in an empty/invalid file stem.")
                return True
            
            # Using .with_name() is safer as it replaces the entire filename (stem + suffix)
            new_path = old_path.with_name(f"{new_stem}.py")

            if new_path.exists() and new_path != old_path:
                logger.error(f"Cannot rename workflow, destination file already exists: {new_path}")
                return True

            os.rename(old_path, new_path)
            logger.info(f"Successfully renamed workflow from '{old_path.name}' to '{new_path.name}'.")
            return True

        except Exception as e:
            logger.error(f"Failed to rename workflow file '{relative_path_str}': {e}", exc_info=True)
            return False

    def _handle_delete_workflow_file(self, command_data: Dict[str, Any]) -> bool:
        """Handles deleting a workflow file."""
        payload = command_data.get('payload', command_data.get('command_payload', {}))
        relative_path_str = payload.get('workflow_path')

        if not relative_path_str:
            logger.error(f"Invalid DELETE_WORKFLOW_FILE payload: {payload}")
            return True

        try:
            path_to_delete = PROJECT_ROOT / relative_path_str
            if not path_to_delete.is_file():
                logger.warning(f"Workflow file to delete not found (already deleted?): {path_to_delete}")
                return True

            os.remove(path_to_delete)
            logger.info(f"Successfully deleted workflow file: {path_to_delete}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete workflow file '{relative_path_str}': {e}", exc_info=True)
            return False