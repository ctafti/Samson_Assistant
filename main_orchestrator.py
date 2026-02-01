
import sys
import os
import time
from pathlib import Path
import json
import threading
from typing import Dict, Any, List, Optional, Tuple, Set # Added Set
import functools
import subprocess
import shutil
from collections import Counter
from datetime import datetime, timezone, date, timedelta # Ensure timedelta is imported

import pytz
import re
import queue
import numpy as np
import faiss
import torch # Added for audio snippet processing
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
import uuid
import logging
from sentence_transformers import SentenceTransformer
from langchain_core.language_models.base import BaseLanguageModel
import dateparser # For parsing human-readable time strings
import json
import time

from src.api_server import app as api_flask_app
from src.command_executor_service import CommandExecutorService
from src.utils.command_utils import queue_command_for_executor
from src.speaker_profile_manager import get_enrolled_speaker_names
from src.utils.file_locking import get_lock

# --- DESIGN NOTE ON DATE FORMATS ---
# This application intentionally uses two date formats: '%Y-%m-%d' for all file system paths and API endpoints,
# and '%Y%m%d' for internal identifiers and command payloads to maintain compatibility with the GUI.

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config_loader import get_config, PROJECT_ROOT, CONFIG_FILE_PATH, ensure_config_exists
from src.logger_setup import setup_logging, logger

from src.folder_monitor import start_monitoring, AudioFileEventHandler

from src.audio_pipeline_controller import (
    initialize_audio_models,
    cleanup_audio_models,
    process_single_audio_file,
    _loaded_models_cache # Directly access the cache for models
)
from src.audio_processing_suite.persistence import (
    load_or_create_faiss_index,
    load_or_create_speaker_map,
    save_faiss_index,
    save_speaker_map
)


from src.audio_processing_suite import speaker_id as aps_speaker_id
from src.audio_processing_suite import text_processing as aps_text_processing
from src.audio_processing_suite import audio_processing as aps_audio # Added for snippet extraction
from src.signal_interface import send_message, start_signal_listener_thread
from src.daily_log_manager import (
    add_processed_audio_entry,
    get_day_start_time,
    get_all_dialogue_for_speaker_id,
    set_day_start_time,
    extract_sequence_number_from_filename,
    get_daily_log_data,
    save_daily_log_data,
    relabel_single_word, 
    modify_word_span_and_speaker, 
    update_matter_segments_for_chunk,
    check_if_sequence_processed,
    relabel_speaker_id_in_daily_log,
    get_highest_processed_sequence,
    get_samson_today,
    update_matter_for_recent_time_window,
    load_daily_flags_queue,
    save_daily_flags_queue,
    get_log_file_path,
    update_chunk_metadata
)
from src.master_daily_transcript import get_master_transcript_path, append_processed_chunk_to_master_log, regenerate_master_log_for_day
from src.llm_interface import get_llm_chat_model
from src.context_manager import get_active_context,  set_active_context
from src.speaker_profile_manager import add_segment_embedding_for_evolution, get_enrolled_speaker_names, create_speaker_profile, get_all_speaker_profiles, get_all_segment_embeddings, clear_segment_embeddings_for_context, update_speaker_profile, update_or_remove_evolution_segment, get_speaker_profile, add_segment_embedding_for_evolution
from src.feedback_manager import log_correction_feedback
from src.speaker_intelligence import SpeakerIntelligenceBackgroundService
from src.tools.health_check import run_health_check
from src.matter_analysis_service import MatterAnalysisService
from src.scheduler_service import SchedulerService
from src.matter_manager import get_all_matters, add_matter
from src.task_intelligence_manager import TaskIntelligenceManager, _load_tasks


try:
    from thefuzz import process as fuzz_process
    from thefuzz import fuzz
    THEFUZZ_AVAILABLE = True
except ImportError:
    THEFUZZ_AVAILABLE = False
    # A warning is logged by the main orchestrator function now.


# --- Global Variables ---
AUDIO_PROCESSING_QUEUE = queue.PriorityQueue(maxsize=100)
TIE_BREAKER_COUNTER = 0
COUNTER_LOCK = threading.Lock()

ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE: Set[date] = set()
SPEAKER_DB_LOCK = get_lock("speaker_db") # Lock for FAISS index and speaker map
ORCHESTRATOR_SHUTDOWN_EVENT = threading.Event()
AUDIO_PROCESSING_THREAD: Optional[threading.Thread] = None
ORCHESTRATOR_FOLDER_MONITOR_OBSERVER: Optional[Any] = None
ORCHESTRATOR_EVENT_HANDLER: Optional[AudioFileEventHandler] = None
HTTP_SERVER_THREAD: Optional[threading.Thread] = None
HTTP_SERVER_INSTANCE: Optional[socketserver.TCPServer] = None
API_SERVER_THREAD: Optional[threading.Thread] = None        #Windmill Server
COMMAND_QUEUE_DIR = PROJECT_ROOT / "data" / "command_queue"
COMMAND_QUEUE_MONITOR_THREAD: Optional[threading.Thread] = None
COMMAND_EXECUTOR_THREAD: Optional[threading.Thread] = None
SCHEDULER_SERVICE_THREAD: Optional[threading.Thread] = None
ORCHESTRATOR_SIGNAL_LISTENER_THREAD: Optional[threading.Thread] = None

SCHEDULED_TASKS: List[Dict[str, Any]] = []
SCHEDULED_TASKS_LOCK = threading.Lock()

GUI_TASK_STATUS: Dict[str, str] = {}
GUI_TASK_STATUS_LOCK = threading.Lock()

ACTIVE_REVIEW_SESSION_FLAGS: List[Dict[str, Any]] = []
ACTIVE_REVIEW_SESSION_CURRENT_INDEX: int = -1
ACTIVE_REVIEW_SESSION_LOCK = threading.Lock()
LAST_EOD_SUMMARY_DATE: Optional[date] = None
EOD_TASKS_LOCK = threading.Lock()

# Centralized state for managing interactive Signal sessions.
# This will hold context for flag reviews, command confirmations, etc.
SIGNAL_SESSION_STATE: Dict[str, Any] = {}
SIGNAL_SESSION_LOCK = threading.Lock()


def _queue_item_with_priority(priority: int, data: Any):
    """Safely adds an item to the priority queue with a tie-breaker."""
    global TIE_BREAKER_COUNTER
    with COUNTER_LOCK:
        TIE_BREAKER_COUNTER += 1
        tie_breaker = TIE_BREAKER_COUNTER
    AUDIO_PROCESSING_QUEUE.put((priority, tie_breaker, data))



# --- HTTP Snippet Server ---
class AudioSnippetRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    Handles GET requests for audio snippets and POST for corrections.
    """
    def do_OPTIONS(self):
        """Handles pre-flight CORS requests."""
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*') # Allow any origin
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _send_cors_error(self, code, message):
        """Helper to send an error response with CORS headers."""
        self.send_response(code)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        error_body = {'status': 'error', 'message': message}
        self.wfile.write(json.dumps(error_body).encode('utf-8'))

    def do_POST(self):
        """
        Handles POST requests for saving multi-word corrections by parsing the
        GUI's data structure and dispatching a SINGLE BATCH command to the worker queue.
        """
        if self.path != '/save_corrections':
            return self._send_cors_error(404, "Endpoint not found. Use /save_corrections")

        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)

            command_type = data.get('command_type')
            command_payload = data.get('command_payload')

            if not command_type or not command_payload:
                return self._send_cors_error(400, "Bad Request: 'command_type' and 'command_payload' are required.")

            # This will be the ID returned to the client for polling the job's status.
            polling_task_id = str(uuid.uuid4())
            with GUI_TASK_STATUS_LOCK:
                GUI_TASK_STATUS[polling_task_id] = "pending"
            
            if command_type == "UPDATE_TASK_STATUS":
                logger.info(f"HTTP Server: Routing '{command_type}' to AudioProcessingWorker queue.")
                # The original payload contains the correct 'task_id'. We inject the new 'polling_task_id'
                # so the worker can report the status of this specific job.
                command_payload['polling_task_id'] = polling_task_id
                command_to_queue = {"type": "UPDATE_TASK_STATUS_FROM_GUI", "payload": command_payload}
                _queue_item_with_priority(0, command_to_queue) # High priority for UI actions

            elif command_type == "UPDATE_TASK_STATUS_FROM_GUI":
                logger.info(f"HTTP Server: Routing '{command_type}' to AudioProcessingWorker queue.")
                # This is the correct handler for task edits from the UI.
                # Inject the polling_task_id for status checks, but crucially, do NOT overwrite the original task_id.
                command_payload['polling_task_id'] = polling_task_id
                command_to_queue = {"type": command_type, "payload": command_payload}
                _queue_item_with_priority(0, command_to_queue)
            
            elif command_type == "RELINK_TASKS_FOR_MATTER":
                logger.info(f"HTTP Server: Routing '{command_type}' to file-based CommandExecutor.")
                # This command's structure is already correct for the executor.
                # The 'data' variable contains the full {'command_type': ..., 'command_payload': ...} dict.
                queue_command_for_executor(data)

            elif command_type == "EXECUTE_WINDMILL_WORKFLOW":
                logger.info(f"HTTP Server: Routing '{command_type}' to file-based CommandExecutor.")
                queue_command_for_executor(data)

            elif command_type == "GENERATE_WORKFLOW_FROM_PROMPT":
                logger.info(f"HTTP Server: Routing '{command_type}' to file-based CommandExecutor.")
                queue_command_for_executor(data)

            elif command_type == "RENAME_WORKFLOW_FILE":
                logger.info(f"HTTP Server: Routing '{command_type}' to file-based CommandExecutor.")
                queue_command_for_executor(data)

            elif command_type == "DELETE_WORKFLOW_FILE":
                logger.info(f"HTTP Server: Routing '{command_type}' to file-based CommandExecutor.")
                queue_command_for_executor(data)
            else:
                # Default behavior: for new jobs (e.g., corrections), inject the polling ID as the main task_id.
                logger.info(f"HTTP Server: Routing command '{command_type}' to AudioProcessingWorker queue (default).")
                command_payload['task_id'] = polling_task_id
                command_to_queue = {
                    "type": command_type,
                    "payload": command_payload
                }
                _queue_item_with_priority(0, command_to_queue)

        
           
            logger.info(f"HTTP Server: Dispatched command '{command_type}' to the worker queue.")

            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # --- MODIFIED: Return the task_id to the client ---
            response_body = {
                'status': 'queued', 
                'message': f'Successfully queued command: {command_type}.',
                'task_id': polling_task_id
            }
            self.wfile.write(json.dumps(response_body).encode('utf-8'))

        except json.JSONDecodeError as e:
            logger.error(f"HTTP Server: JSON decode error in /save_corrections: {e}", exc_info=True)
            return self._send_cors_error(400, f"Bad Request: Invalid JSON. {e}")
        except Exception as e:
            logger.error(f"HTTP Server: Unhandled error in /save_corrections: {e}", exc_info=True)
            return self._send_cors_error(500, f"Internal Server Error: {e}")

    def do_GET(self):
        try:
            parsed_url = urlparse(self.path)
            
            if parsed_url.path == '/task_status':
                params = parse_qs(parsed_url.query)
                task_id = params.get('id', [None])[0]

                if not task_id:
                    return self._send_cors_error(400, "Bad Request: Missing 'id' parameter for task status check.")

                with GUI_TASK_STATUS_LOCK:
                    status = GUI_TASK_STATUS.get(task_id, "not_found")
                    # Clean up completed tasks to prevent memory growth
                    if status == "complete":
                        del GUI_TASK_STATUS[task_id]
                
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response_body = {'status': status}
                self.wfile.write(json.dumps(response_body).encode('utf-8'))
                return # End execution here for this path

            elif parsed_url.path == '/snippet':
                params = parse_qs(parsed_url.query)
                date_str = params.get('date', [None])[0]
                flag_id = params.get('flag_id', [None])[0]
                # --- START: New parameters for transcript snippets ---
                file_stem = params.get('file_stem', [None])[0]
                start_s_str = params.get('start', [None])[0]
                end_s_str = params.get('end', [None])[0]
                # --- END: New parameters ---

                if not date_str:
                    return self._send_cors_error(400, "Bad Request: 'date' parameter is required.")
                
                
                # Branch 1: Handle Flag-based snippet requests (existing logic)
                if flag_id:
                    # AC5: Security - Construct and validate path
                    try:
                        config = get_config()
                        base_snippets_dir = config['paths']['flag_snippets_dir']
                        
                        # Sanitize inputs to prevent directory traversal
                        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str) or not re.match(r'^FLAG_[\w\-]+$', flag_id):
                             return self._send_cors_error(403, "Forbidden: Invalid parameter format for flag.")

                        snippet_path = Path(base_snippets_dir) / date_str / f"{flag_id}.wav"
                        # ... (rest of existing flag snippet logic) ...
                        if not snippet_path.resolve().is_relative_to(Path(base_snippets_dir).resolve()) or not snippet_path.is_file():
                            logger.warning(f"Forbidden access attempt for snippet: {snippet_path}")
                            return self._send_cors_error(403, "Forbidden: Access denied.")

                        with open(snippet_path, 'rb') as f:
                            snippet_bytes = f.read()
                        
                        self.send_response(200)
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.send_header('Content-type', 'audio/wav')
                        self.send_header('Content-Length', str(len(snippet_bytes)))
                        self.end_headers()
                        self.wfile.write(snippet_bytes)

                    except Exception as e:
                        logger.error(f"HTTP Server: Error serving snippet for flag {flag_id}: {e}", exc_info=True)
                        return self._send_cors_error(500, "Internal Server Error.")
                    return # End execution for this branch

                # Branch 2: Handle Transcript-based snippet requests (new logic)
                elif file_stem and start_s_str and end_s_str:
                    try:
                        config = get_config()
                        archived_audio_folder = config['paths']['archived_audio_folder']
                        ffmpeg_path = config['tools'].get('ffmpeg_path', 'ffmpeg')
                        
                        # Sanitize inputs
                        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str) or not re.match(r'^[\w\-]+$', file_stem):
                            return self._send_cors_error(403, "Forbidden: Invalid parameter format for transcript snippet.")
                        
                        start_s = float(start_s_str)
                        end_s = float(end_s_str)

                        date_dir = Path(archived_audio_folder) / date_str
                        # Find the original file, could be .aac, .mp3, etc.
                        original_audio_path = next(date_dir.glob(f"{file_stem}.*"), None)

                        if not original_audio_path or not original_audio_path.is_file():
                             return self._send_cors_error(404, f"Source audio file for stem '{file_stem}' not found.")

                        # Create snippet on the fly using ffmpeg
                        ffmpeg_command = [
                            ffmpeg_path,
                            "-hide_banner", "-loglevel", "error", "-i", str(original_audio_path),
                            "-ss", str(start_s), "-to", str(end_s),
                            "-f", "wav", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-"
                        ]
                        
                        result = subprocess.run(ffmpeg_command, capture_output=True, timeout=15)

                        if result.returncode == 0:
                            snippet_bytes = result.stdout
                            self.send_response(200)
                            self.send_header('Access-Control-Allow-Origin', '*')
                            self.send_header('Content-type', 'audio/wav')
                            self.send_header('Content-Length', str(len(snippet_bytes)))
                            self.end_headers()
                            self.wfile.write(snippet_bytes)
                        else:
                            logger.error(f"ffmpeg failed for transcript snippet: {result.stderr.decode('utf-8', errors='ignore').strip()}")
                            return self._send_cors_error(500, "Failed to extract audio snippet.")
                    
                    except Exception as e:
                        logger.error(f"HTTP Server: Error serving transcript snippet: {e}", exc_info=True)
                        return self._send_cors_error(500, "Internal Server Error.")
                    return # End execution for this branch
                
                else:
                    return self._send_cors_error(400, "Bad Request: requires either (date, flag_id) or (date, file_stem, start, end).")
            else:
                self.send_error(404, "Endpoint not found. Use /snippet or /task_status")
                return

        except Exception as e:
            logger.error(f"HTTP Server: Unhandled error in do_GET: {e}", exc_info=True)
            if not self.wfile.closed: # Check if headers have already been sent or connection closed
                try:
                    # Avoid sending error if headers already sent (e.g., during self.wfile.write)
                    # A more robust check might involve checking self.headers_sent attribute if available/reliable
                    # or simply logging if sending error itself fails.
                    if not getattr(self, '_headers_sent', False): # A common pattern, but not standard for BaseHTTPRequestHandler
                        self.send_error(500, f"Internal Server Error: {e}")
                except Exception as e_send: 
                    logger.error(f"HTTP Server: Failed to send error response: {e_send}")

    def log_message(self, format_str, *args):
        # Filter out "GET /snippet HTTP/1.1" 200 - (successful logs) if too verbose
        # Or customize logging further as needed.
        # For now, default behavior will log to logger.debug
        # Check if the message contains "200" for success or other codes for more detailed logging
        if " 200 " in (format_str % args):
            logger.debug(f"HTTP Server: {self.address_string()} - {format_str % args}")
        else:
            logger.info(f"HTTP Server: {self.address_string()} - {format_str % args}")


# --- Helper for Interactive Flag Review ---
def get_closest_name_match(name_to_check: str, list_of_names: List[str], threshold_ratio: int = 85) -> Optional[str]:
    if not THEFUZZ_AVAILABLE or not list_of_names:
        return None
    # Use a processor to make the matching case-insensitive, while returning the original cased name.
    processor = lambda s: str(s).lower().strip()
    best_match = fuzz_process.extractOne(name_to_check, list_of_names, scorer=fuzz.WRatio, processor=processor)
    if best_match and best_match[1] >= threshold_ratio:
        return best_match[0]
    return None

def get_multiple_name_matches(name_to_check: str, list_of_names: List[str], threshold_ratio: int = 75) -> List[str]:
    """Finds all fuzzy matches for a name in a list above a given threshold."""
    if not THEFUZZ_AVAILABLE or not list_of_names:
        return []
    processor = lambda s: str(s).lower().strip()
    matches = fuzz_process.extract(name_to_check, list_of_names, scorer=fuzz.WRatio, processor=processor, limit=5)
    good_matches = [match[0] for match in matches if match[1] >= threshold_ratio]
    return good_matches

def _format_datetime_for_user(dt_object: datetime, tz_str: str) -> str:
    """Formats a datetime object into a user-friendly string in the specified timezone."""
    if not dt_object:
        return "an unknown time"
    
    try:
        target_tz = pytz.timezone(tz_str)
    except pytz.UnknownTimeZoneError:
        target_tz = timezone.utc
    
    dt_aware = dt_object.astimezone(target_tz) if dt_object.tzinfo else target_tz.localize(dt_object)
    now_in_tz = datetime.now(target_tz)

    if dt_aware.date() == now_in_tz.date():
        return f"Today at {dt_aware.strftime('%H:%M')}"
    else:
        day = dt_aware.strftime('%d').lstrip('0')
        return f"{dt_aware.strftime('%B')} {day} at {dt_aware.strftime('%H:%M')}"

def parse_flag_id_for_date(flag_id: str) -> Optional[datetime]:
    match = re.match(r"FLAG_(\d{8})_.+", flag_id.upper())
    if match:
        date_str = match.group(1)
        try: return datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError: logger.error(f"Could not parse date from flag_id: {flag_id}")
    return None

def _clear_active_review_session():
    global ACTIVE_REVIEW_SESSION_FLAGS, ACTIVE_REVIEW_SESSION_CURRENT_INDEX
    logger.debug("Entering _clear_active_review_session.")
    ACTIVE_REVIEW_SESSION_FLAGS = []
    logger.debug("_clear_active_review_session: ACTIVE_REVIEW_SESSION_FLAGS cleared.")
    ACTIVE_REVIEW_SESSION_CURRENT_INDEX = -1
    logger.debug("_clear_active_review_session: ACTIVE_REVIEW_SESSION_CURRENT_INDEX has been set to -1.")
    logger.info("Active flag review session cleared.") 
    logger.debug("Exiting _clear_active_review_session successfully.")

def _update_flag_status_in_queue(flag_id_to_update: str, new_status: str, original_flag_date: Optional[datetime] = None, final_assigned_name: Optional[str] = None):
    logger.info(f"Entering _update_flag_status_in_queue with flag_id: {flag_id_to_update}, new_status: {new_status}, original_flag_date: {original_flag_date}, final_assigned_name: {final_assigned_name}")
    if original_flag_date is None: original_flag_date = parse_flag_id_for_date(flag_id_to_update)
    if not original_flag_date:
        logger.error(f"Cannot update flag {flag_id_to_update}: could not determine original date.")
        logger.info("Exiting _update_flag_status_in_queue: original date not determinable.")
        return False

    daily_flags = load_daily_flags_queue(original_flag_date) 
    updated = False
    for flag_item in daily_flags:
        if flag_item.get("flag_id") == flag_id_to_update:
            flag_item["status"] = new_status
            if final_assigned_name: flag_item["assigned_name"] = final_assigned_name
            flag_item["timestamp_resolved_utc"] = datetime.now(timezone.utc).isoformat()
            updated = True
            logger.info(f"Flag {flag_id_to_update} found in queue. Updating status to '{new_status}'.")
            break
    if updated:
        save_daily_flags_queue(daily_flags, original_flag_date) 
        logger.info(f"Flag {flag_id_to_update} status updated to '{new_status}' in queue for {original_flag_date.strftime('%Y-%m-%d')}.")
        logger.info("Exiting _update_flag_status_in_queue: Update successful.")
        return True
    else:
        logger.warning(f"Flag {flag_id_to_update} not found in queue for {original_flag_date.strftime('%Y-%m-%d')} to update status.")
        logger.info("Exiting _update_flag_status_in_queue: Flag not found.")
        return False

def get_pending_flags() -> List[Dict[str, Any]]:
    config = get_config()
    max_days_lookback = int(config.get('audio_suite_settings', {}).get('review_flags_max_days_lookback', 7))
    logger.info(f"Scanning for pending flags across the last {max_days_lookback} days.")

    all_flags_before_status_filter = []
    for i in range(max_days_lookback):
        target_date_for_load = datetime.now(timezone.utc) - timedelta(days=i)
        flags_for_date = load_daily_flags_queue(target_date_for_load)
        all_flags_before_status_filter.extend(flags_for_date)

    pending_flags = [f for f in all_flags_before_status_filter if f.get('status') == 'pending_review']
   
    sorted_pending_flags = sorted(pending_flags, key=lambda x: x.get('timestamp_logged_utc', ''))
   
    logger.info(f"Found a total of {len(sorted_pending_flags)} flags pending review.")
    return sorted_pending_flags


WORKER_QUEUE_COMMANDS = {
    'RESOLVE_FLAG', 
    'BATCH_CORRECT_SPEAKER_ASSIGNMENTS',
    'CORRECT_TEXT_AND_SPEAKER',
    'RECALCULATE_SPEAKER_PROFILES',
    'REBUILD_SPEAKER_DATABASE'
}
def queue_command_from_gui(command: Dict[str, Any]) -> bool:
    """
    Acts as a router for commands from the GUI. It dispatches commands
    to the appropriate queue based on their type.
    """
    command_type = command.get('type')

    if command_type in WORKER_QUEUE_COMMANDS:
        logger.info(f"Dispatching GUI command '{command_type}' to in-memory AudioProcessingWorker queue.")
        try:
            _queue_item_with_priority(0, command)
            return True
        except Exception as e:
            logger.error(f"Failed to dispatch command '{command_type}' to worker queue: {e}", exc_info=True)
            return False
    else:
        logger.info(f"Dispatching GUI command '{command_type}' to file-based CommandExecutorService queue.")
        try:
            queue_command_for_executor(command)
            return True
        except Exception as e:
            logger.error(f"Failed to dispatch command '{command_type}' to executor file queue: {e}", exc_info=True)
            return False

def resolve_flag(flag_id: str, resolution: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Constructs a 'RESOLVE_FLAG' command dictionary for the processing queue.
    This function no longer performs the action itself but prepares it for the worker.

    Args:
        flag_id: The ID of the flag to resolve.
        resolution: A dictionary detailing the user's action (e.g., assign, skip, context).

    Returns:
        A command dictionary to be placed on the queue, or None if validation fails.
    """
    logger.info(f"Preparing RESOLVE_FLAG command for flag '{flag_id}' with resolution: {resolution}")

    flag_date = parse_flag_id_for_date(flag_id)
    if not flag_date:
        logger.error(f"Cannot prepare command for flag '{flag_id}': could not parse date from ID.")
        return None

    all_flags_for_date = load_daily_flags_queue(flag_date)
    flag_data = next((f for f in all_flags_for_date if f.get("flag_id") == flag_id), None)

    if not flag_data:
        logger.error(f"Cannot prepare command for flag '{flag_id}': not found in queue for {flag_date.strftime('%Y-%m-%d')}.")
        return None

    # Package all necessary information into the command payload
    command_payload = {
        "flag_id": flag_id,
        "flag_data": flag_data, # Pass the entire flag data for the worker
        "resolution": resolution
    }

    logger.info(f"[Orchestrator] Successfully packaged RESOLVE_FLAG command for worker. Payload size: {len(str(command_payload))} bytes.")
    return {
        "type": "RESOLVE_FLAG",
        "payload": command_payload
    }

def _finalize_review_session_updates(config_from_caller: Dict[str, Any]):
    global ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE, AUDIO_PROCESSING_QUEUE
    logger.info("Entering _finalize_review_session_updates.")

    if ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE:
        logger.info(f"_finalize_review_session_updates: Processing master log regeneration for days: {sorted(list(ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE))}")
        sorted_dates_for_regen = sorted(list(ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE))
       
        for day_date_obj in sorted_dates_for_regen: 
            day_str = day_date_obj.strftime('%Y-%m-%d')
            logger.info(f"_finalize_review_session_updates: Processing day {day_str}")
           
            day_datetime_obj = datetime.combine(day_date_obj, datetime.min.time(), tzinfo=timezone.utc)
           
            day_start_time_for_regen = get_day_start_time(day_datetime_obj)
            if day_start_time_for_regen:
                logger.info(f"Sanitizing daily log for {day_str} by re-running timestamp calculations.")
                # Calling set_day_start_time forces recalculation of absolute timestamps and saving.
                set_day_start_time(day_start_time_for_regen, day_datetime_obj)
            else:
                logger.error(f"Cannot sanitize daily log for {day_str}: day_start_time is not set. Skipping this day's regeneration.")
                continue

            try:
                logger.info(f"_finalize_review_session_updates: Before calling _post_process_fill_gaps_in_daily_log for {day_str}")
                _post_process_fill_gaps_in_daily_log(day_datetime_obj, config_from_caller)
            except Exception as e_post_process:
                logger.error(f"_finalize_review_session_updates: Error during _post_process_fill_gaps_in_daily_log for {day_str}: {e_post_process}", exc_info=True)

            try:
                logger.info(f"_finalize_review_session_updates: Before calling regenerate_master_log_for_day for {day_str}")
                regenerate_master_log_for_day(day_date_obj, config_from_caller)
               
                rebuilt_cmd_payload = f"CMD_MASTER_LOG_REBUILT_FOR_DATE:{day_str}"
                _queue_item_with_priority(0, rebuilt_cmd_payload) 
                logger.info(f"_finalize_review_session_updates: Queued command to worker: {rebuilt_cmd_payload}")
            except Exception as e_regen:
                logger.error(f"_finalize_review_session_updates: Error during regenerate_master_log_for_day for {day_str}: {e_regen}", exc_info=True)

        logger.info("_finalize_review_session_updates: Finished processing regeneration queue.")
        ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE.clear()
        logger.info("_finalize_review_session_updates: ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE cleared.")
    else:
        logger.info("_finalize_review_session_updates: ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE is empty. No regeneration needed.")
    logger.info("Exiting _finalize_review_session_updates.")

# Placeholder for future logic to fill gaps in the daily log after review updates.
# This could involve checking for missing sequence numbers for 'target_date'
# and creating placeholder entries or performing other integrity checks.
def _post_process_fill_gaps_in_daily_log(target_date: datetime, config: Dict[str, Any]):
    logger.info(f"Placeholder: _post_process_fill_gaps_in_daily_log called for date: {target_date.strftime('%Y-%m-%d')}")
    # For now, this function does nothing beyond logging.
    pass

def _present_next_flag_for_review(recipient_phone: str, config: Dict[str, Any]):
    global ACTIVE_REVIEW_SESSION_FLAGS, ACTIVE_REVIEW_SESSION_CURRENT_INDEX, ACTIVE_REVIEW_SESSION_LOCK

    attachment_path: Optional[Path] = None

    try:
        logger.debug("Attempting to present next flag for review.")
        current_flag = None
        with ACTIVE_REVIEW_SESSION_LOCK:
            ACTIVE_REVIEW_SESSION_CURRENT_INDEX += 1
            logger.info(f"Advanced review index to: {ACTIVE_REVIEW_SESSION_CURRENT_INDEX}. Total flags: {len(ACTIVE_REVIEW_SESSION_FLAGS)}")

            if ACTIVE_REVIEW_SESSION_CURRENT_INDEX >= len(ACTIVE_REVIEW_SESSION_FLAGS):
                msg = "All pending flags have been reviewed. Review session ended."
                logger.info(f"Sending Signal message: '{msg}'")
                send_message(recipient_phone, msg, config)
                _clear_active_review_session()
                _finalize_review_session_updates(config)
                return

            current_flag = ACTIVE_REVIEW_SESSION_FLAGS[ACTIVE_REVIEW_SESSION_CURRENT_INDEX]

        flag_id = current_flag.get('flag_id', 'N/A')
        source_file_name = current_flag.get('source_file_name')
        snippet_server_params = current_flag.get('snippet_server_params')
        attachments_to_send = []

        if source_file_name and snippet_server_params:
            paths_cfg = config.get('paths', {})
            archived_audio_folder = paths_cfg.get('archived_audio_folder')
            temp_processing_dir = paths_cfg.get('temp_processing_dir')
            tools_cfg = config.get('tools', {})
            ffmpeg_path = tools_cfg.get('ffmpeg_path', 'ffmpeg')

            if all([archived_audio_folder, temp_processing_dir, ffmpeg_path]):
                date_str = snippet_server_params.get('date')
                start_s = snippet_server_params.get('start')
                end_s = snippet_server_params.get('end')

                if all([date_str, isinstance(start_s, (int, float)), isinstance(end_s, (int, float))]):
                    original_audio_path = Path(archived_audio_folder) / date_str / source_file_name
                    if original_audio_path.exists():
                        temp_dir_path = Path(temp_processing_dir)
                        temp_dir_path.mkdir(parents=True, exist_ok=True)
                        attachment_filename = f"review_snippet_{flag_id}.wav"
                        attachment_path = temp_dir_path / attachment_filename

                        ffmpeg_command = [
                            ffmpeg_path,
                            "-hide_banner", "-loglevel", "error",
                            "-i", str(original_audio_path),
                            "-ss", str(start_s),
                            "-to", str(end_s),
                            "-acodec", "pcm_s16le",
                            "-ar", "16000",
                            "-ac", "1",
                            "-y",
                            str(attachment_path)
                        ]
                        
                        logger.info(f"Executing ffmpeg to create snippet for flag {flag_id}")
                        result = subprocess.run(ffmpeg_command, capture_output=True, text=True, timeout=15)

                        if result.returncode == 0:
                            logger.info(f"Successfully created temporary snippet: {attachment_path}")
                            attachments_to_send.append(str(attachment_path))
                        else:
                            logger.error(f"ffmpeg failed for flag {flag_id} with return code {result.returncode}.")
                            logger.error(f"ffmpeg stderr: {result.stderr.strip()}")
                    else:
                        logger.warning(f"Could not find source audio for snippet: {original_audio_path}")
                else:
                    logger.warning(f"Flag {flag_id} has incomplete snippet parameters: {snippet_server_params}")
            else:
                logger.warning("Snippet creation skipped: paths or tools not fully configured.")
        else:
            logger.warning(f"Flag {flag_id} is missing source file or snippet params. Cannot create audio snippet.")

        reason = current_flag.get('reason_for_flag', 'N/A')
        source_file = current_flag.get('source_file_name', 'N/A')
        tentative_speaker = current_flag.get('tentative_speaker_name', 'Unknown')
        dialogue_snippet = current_flag.get('text_preview')

        message_lines = [
            f"Flag for Review ({ACTIVE_REVIEW_SESSION_CURRENT_INDEX + 1}/{len(ACTIVE_REVIEW_SESSION_FLAGS)}):",
            # f"ID: {flag_id}", # REMOVED as per request
            # f"File: {source_file}", # REMOVED as per request
            f"Reason: {reason}",
            f"Tentative Speaker: {tentative_speaker}"
        ]
        if dialogue_snippet:
            message_lines.append(f"Dialogue: \"{dialogue_snippet[:150]}{'...' if len(dialogue_snippet) > 150 else ''}\"")

        message_lines.append("\nOptions:")
        enrolled_speakers = get_enrolled_speaker_names()
        options_map = {}

        if tentative_speaker and tentative_speaker != 'UNKNOWN_SPEAKER':
            message_lines.append(f"  YES - Confirm '{tentative_speaker}'")
        
        option_num = 1
        if enrolled_speakers:
            for name in enrolled_speakers:
                message_lines.append(f"  {option_num}. Assign to '{name}'")
                options_map[option_num] = name
                option_num += 1
        
        if THEFUZZ_AVAILABLE and tentative_speaker != 'UNKNOWN_SPEAKER' and enrolled_speakers:
            closest_match = get_closest_name_match(tentative_speaker, enrolled_speakers)
            if closest_match and closest_match.lower() != tentative_speaker.lower():
                message_lines.append(f"  (Suggestion: '{closest_match}' seems similar to '{tentative_speaker}')")

        message_lines.append(f"  <Name> [context] - Assign by typing name (e.g., 'Name' or 'Name in_person').")
        message_lines.append("  SKIP - Skip this flag for now.")
        message_lines.append("  DONE - End review session.")
        
        current_flag['potential_matches_with_indices'] = options_map
        full_message = "\n".join(message_lines)
        
        logger.info(f"Presenting flag {flag_id} to user with {len(attachments_to_send)} attachment(s).")
        send_message(recipient_phone, full_message, config, attachments=attachments_to_send)

    finally:
        if attachment_path and attachment_path.exists():
            try:
                attachment_path.unlink()
                logger.info(f"Cleaned up temporary attachment: {attachment_path}")
            except Exception as e:
                logger.error(f"Failed to delete temporary attachment {attachment_path}: {e}")

def _try_present_next_flag_or_handle_error(recipient_phone: str, config: Dict[str, Any]):
    try: _present_next_flag_for_review(recipient_phone, config)
    except Exception as e_present:
        error_msg = "Error presenting next flag. Review session terminated."
        logger.error(f"Error during _present_next_flag_or_review: {e_present}", exc_info=True)
        _clear_active_review_session()
        logger.info(f"Sending Signal message: '{error_msg}'")
        send_message(recipient_phone, error_msg, config)

# <<< START: New End-of-Day Handler Function >>>
def _handle_end_of_day_tasks(config: Dict[str, Any]):
    """
    Runs the end-of-day tasks, including the health check, and sends a report.
    """
    global LAST_EOD_SUMMARY_DATE, EOD_TASKS_LOCK
   
    samson_today = get_samson_today()
   
    with EOD_TASKS_LOCK:
        if LAST_EOD_SUMMARY_DATE == samson_today:
            logger.debug("EOD tasks for today have already been run. Skipping.")
            return
       
        logger.info(f"--- Running End-of-Day Tasks for {samson_today.strftime('%Y-%m-%d')} ---")
       
        # 1. Run the health check
        health_report = run_health_check()
       
        # 2. Format the report for Signal
        report_lines = [f"Samson End-of-Day Report: {samson_today.strftime('%Y-%m-%d')}"]
       
        if health_report['status'] == 'ok':
            report_lines.append("\n✅ Data Health Check: OK")
            report_lines.append("All checks passed.")
        else:
            report_lines.append(f"\n⚠️ Data Health Check: {len(health_report['issues'])} Issue(s) Found")
            for idx, issue in enumerate(health_report['issues']):
                report_lines.append(f"  {idx + 1}. {issue}")
       
        # 3. Send the health report
        recipient = config.get('signal', {}).get('recipient_phone_number')
        if recipient:
            send_message(recipient, "\n".join(report_lines), config)
        else:
            logger.error("Cannot send EOD report: recipient_phone_number not configured.")

        # 4. Format and send the Samson Daily Digest
        try:
            queue_size = AUDIO_PROCESSING_QUEUE.qsize()
            pending_flags = get_pending_flags()
            
            digest_lines = [f"Samson Daily Digest: {samson_today.strftime('%Y-%m-%d')}"]
            digest_lines.append(f"\nSystem Status:")
            digest_lines.append(f"  - Audio Queue Size: {queue_size}")
            digest_lines.append(f"  - Pending Flags: {len(pending_flags)}")
            
            # Query for open tasks and group them by matter
            all_tasks = _load_tasks()
            open_tasks = [
                t for t in all_tasks 
                if t.get('status') in ['confirmed', 'in_progress']
            ]

            if open_tasks:
                digest_lines.append(f"\nOpen Tasks ({len(open_tasks)}):")
                
                # Create a map for quick matter name lookup
                all_matters = get_all_matters(include_inactive=True)
                matter_map = {m['matter_id']: m['name'] for m in all_matters}
                
                # Group tasks by matter name
                tasks_by_matter = {}
                for task in open_tasks:
                    matter_id = task.get('matter_id')
                    matter_name = matter_map.get(matter_id, "Unassigned")
                    if matter_name not in tasks_by_matter:
                        tasks_by_matter[matter_name] = []
                    tasks_by_matter[matter_name].append(task)
                
                # Format the grouped tasks for the message
                for matter_name, tasks in sorted(tasks_by_matter.items()):
                    digest_lines.append(f"  - {matter_name}:")
                    for task in tasks:
                        digest_lines.append(f"    - {task.get('title', 'Untitled Task')}")
            else:
                digest_lines.append("\nNo open tasks.")

            digest_message = "\n".join(digest_lines)
            
            if recipient:
                send_message(recipient, digest_message, config)
        except Exception as e_digest:
            logger.error(f"Failed to generate or send Samson Daily Digest: {e_digest}", exc_info=True)
           
        # 5. Mark today's EOD tasks as complete
        LAST_EOD_SUMMARY_DATE = samson_today
        logger.info("--- End-of-Day Tasks Complete ---")

# --- New Signal Command Helpers (Stateful) ---

def _present_speaker_flag(flag: Dict[str, Any]):
    """Presents a speaker-related flag to the user for review."""
    config = get_config()
    sender = SIGNAL_SESSION_STATE['sender']
    total_flags = len(SIGNAL_SESSION_STATE['review_flags'])
    current_index = SIGNAL_SESSION_STATE['current_index']

    reason = flag.get('reason_for_flag', 'N/A')
    tentative_speaker = flag.get('tentative_speaker_name', 'Unknown')
    dialogue_snippet = flag.get('text_preview', flag.get('summary', ''))

    message_lines = [
        f"Speaker Flag ({current_index + 1}/{total_flags}):",
        f"Reason: {reason}",
        f"Tentative Speaker: {tentative_speaker}"
    ]
    if dialogue_snippet:
        message_lines.append(f"Dialogue: \"{dialogue_snippet[:150]}{'...' if len(dialogue_snippet) > 150 else ''}\"")

    message_lines.append("\nOptions:")
    enrolled_speakers = get_enrolled_speaker_names()
    options_map = {}
    
    option_num = 1
    if enrolled_speakers:
        for name in enrolled_speakers:
            message_lines.append(f"  {option_num}. Assign to '{name}'")
            options_map[str(option_num)] = name
            option_num += 1
    
    message_lines.append("  <Name> - Assign by typing a new or existing name.")
    message_lines.append("  SKIP - Skip this flag.")
    message_lines.append("  DONE - End review session.")
    
    SIGNAL_SESSION_STATE['current_flag_options'] = enrolled_speakers
    full_message = "\n".join(message_lines)
    send_message(sender, full_message, config)

def _present_matter_flag(flag: Dict[str, Any]):
    """Presents a matter conflict flag to the user for review."""
    config = get_config()
    sender = SIGNAL_SESSION_STATE['sender']
    total_flags = len(SIGNAL_SESSION_STATE['review_flags'])
    current_index = SIGNAL_SESSION_STATE['current_index']

    conflicting_matters = flag.get('conflicting_matters', [])
    dialogue_snippet = flag.get('text_preview', '')
    
    message_lines = [
        f"Matter Conflict Flag ({current_index + 1}/{total_flags}):",
        f"Reason: The following dialogue was difficult to assign to a single matter.",
        f'Dialogue: "{dialogue_snippet[:150]}{"..." if len(dialogue_snippet) > 150 else ""}"',
        "\nPotential Matters:"
    ]
    
    option_num = 1
    for matter in conflicting_matters:
        message_lines.append(f"  {option_num}. {matter['name']}")
        option_num += 1
        
    message_lines.append("\nOptions:")
    option_num = 1
    for matter in conflicting_matters:
        message_lines.append(f"  {option_num}. Assign to '{matter['name']}'")
        option_num += 1
        
    message_lines.append("  NEW <Name> - Create & assign a new matter.")
    message_lines.append("  SKIP - Skip this flag.")
    message_lines.append("  DONE - End review session.")

    SIGNAL_SESSION_STATE['current_flag_options'] = conflicting_matters
    full_message = "\n".join(message_lines)
    send_message(sender, full_message, config)
    
def _present_next_flag_for_review():
    """Presents the next flag in the session state or ends the session."""
    config = get_config()
    if SIGNAL_SESSION_STATE.get('current_index', -1) >= len(SIGNAL_SESSION_STATE.get('review_flags', [])):
        send_message(SIGNAL_SESSION_STATE['sender'], "Flag review complete.", config)
        SIGNAL_SESSION_STATE.clear()
        return

    flag = SIGNAL_SESSION_STATE['review_flags'][SIGNAL_SESSION_STATE['current_index']]
    flag_type = flag.get('flag_type')

    if flag_type in ['new_speaker_detected', 'ambiguous_speaker']:
        _present_speaker_flag(flag)
    elif flag_type == 'matter_conflict':
        _present_matter_flag(flag)
    else:
        logger.warning(f"Unknown flag type '{flag_type}' encountered during review. Skipping.")
        SIGNAL_SESSION_STATE['current_index'] += 1
        _present_next_flag_for_review()

def _start_flag_review_session(sender: str):
    """Initializes and starts a flag review session for a user."""
    config = get_config()
    pending_flags = get_pending_flags()
    if not pending_flags:
        send_message(sender, "There are no pending flags to review.", config)
        return

    SIGNAL_SESSION_STATE.clear()
    SIGNAL_SESSION_STATE['mode'] = 'flag_review'
    SIGNAL_SESSION_STATE['review_flags'] = pending_flags
    SIGNAL_SESSION_STATE['current_index'] = 0
    SIGNAL_SESSION_STATE['sender'] = sender
    
    _present_next_flag_for_review()

def _handle_fuzzy_match_response(sender: str, message: str):
    """Handles YES/NO response to a fuzzy name match suggestion."""
    config = get_config()
    message_clean = message.strip()
    prefix = "set matter to "
    if not message_clean.lower().startswith(prefix):
        send_message(sender, 'Invalid format. Use: set matter to <Name> [at <time>]', config)
        return

    body = message_clean[len(prefix):]
    
    matter_name_query = body
    time_string = "now" # Default

    # Find the last occurrence of a time keyword to avoid splitting matter names
    keywords = [" at ", " on ", " in "]
    last_keyword_pos = -1
    keyword_len = 0

    for kw in keywords:
        # Case-insensitive find from the right
        pos = body.lower().rfind(kw)
        if pos > last_keyword_pos:
            last_keyword_pos = pos
            keyword_len = len(kw)

    if last_keyword_pos != -1:
        potential_matter_name = body[:last_keyword_pos].strip()
        # The part after the keyword could be a time
        potential_time_str = body[last_keyword_pos + keyword_len:].strip()
        
        # Use dateparser to validate if it's a real time string
        if dateparser.parse(potential_time_str):
             matter_name_query = potential_matter_name
             time_string = potential_time_str
    matter_name_query = matter_name_query.strip()
    time_string = time_string.strip() if time_string else "now"
    if re.match(r'^\d{3,4}$', time_string):
        if len(time_string) == 3:
            # e.g., "900" -> "09:00"
            time_string = f"0{time_string[0]}:{time_string[1:]}"
        else:
            # e.g., "1700" -> "17:00"
            time_string = f"{time_string[:2]}:{time_string[2:]}"

    # Infer environmental context
    message_lower = message.strip().lower()
    environmental_context = "in_person"
    voip_keywords = ["teams", "call", "meeting", "zoom", "google meet"]
    if any(keyword in message_lower for keyword in voip_keywords):
        environmental_context = "voip"

    all_matter_names = [m['name'] for m in get_all_matters()]
    potential_matches = get_multiple_name_matches(matter_name_query, all_matter_names)

    if not potential_matches:
        send_message(sender, f"No matters found matching '{matter_name_query}'. Use 'new matter' to create it.", config)
        return

    if len(potential_matches) > 1:
        SIGNAL_SESSION_STATE['mode'] = 'matter_clarification'
        SIGNAL_SESSION_STATE['clarification_options'] = potential_matches
        SIGNAL_SESSION_STATE['pending_command_details'] = {'time_string': time_string}
        
        message_lines = [f"Which matter did you mean for '{matter_name_query}'?"]
        for i, name in enumerate(potential_matches):
            message_lines.append(f"  {i+1}. {name}")
        send_message(sender, "\n".join(message_lines), config)
        return

    target_matter_name = potential_matches[0]
    target_matter = next((m for m in get_all_matters() if m['name'] == target_matter_name), None)
    if not target_matter:
        send_message(sender, f"Internal error: could not retrieve details for '{target_matter_name}'.", config)
        return

    local_tz_str = config.get('timings', {}).get('assumed_recording_timezone', 'UTC')
    local_tz = pytz.timezone(local_tz_str)
    now_in_local_tz = datetime.now(local_tz)
    target_dt = dateparser.parse(time_string, settings={'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': now_in_local_tz})
    if not target_dt:
        send_message(sender, f"Could not understand the time '{time_string}'.", config)
        return

    local_tz_str = config.get('timings', {}).get('assumed_recording_timezone', 'UTC')
    logger.info(f"Queueing SCHEDULE_MATTER command for '{target_matter_name}' with time string '{time_string}'.")
    command_to_queue = {
        # The command type is incorrect here. It should be SCHEDULE_MATTER for the executor,
        # but the logic inside the executor can handle it. The main issue is the queue.
        # "type": is not a key for the command executor, it's "command_type"
        "command_type": "SCHEDULE_MATTER",
        "payload": {
            "matter_name": target_matter_name,
            "time_string": time_string,
            "source": "signal_command",
            "environmental_context": environmental_context # Add the context here
        }
    }

    queue_command_for_executor(command_to_queue)

    formatted_time = _format_datetime_for_user(target_dt, local_tz_str)
    send_message(sender, f"OK. Matter set to '{target_matter_name}' starting {formatted_time}.", config)

def _handle_matter_clarification_response(sender: str, message: str):
    """Handles user response to a matter clarification prompt."""
    config = get_config()
    response = message.strip()
    options = SIGNAL_SESSION_STATE.get('clarification_options', [])
    
    target_matter_name = None
    if response.isdigit() and 1 <= int(response) <= len(options):
        target_matter_name = options[int(response) - 1]
    else:
        closest_match = get_closest_name_match(response, options)
        if closest_match:
            target_matter_name = closest_match

    if not target_matter_name:
        send_message(sender, "Invalid selection. Please choose a number from the list.", config)
        return

    time_string = SIGNAL_SESSION_STATE['pending_command_details']['time_string']
    target_matter = next((m for m in get_all_matters() if m['name'] == target_matter_name), None)

    # Pre-process time string to handle HHMM format and guide dateparser
    if re.match(r'^\d{3,4}$', time_string):
        if len(time_string) == 3:
            # e.g., "900" -> "09:00"
            time_string = f"0{time_string[0]}:{time_string[1:]}"
        else:
            # e.g., "1700" -> "17:00"
            time_string = f"{time_string[:2]}:{time_string[2:]}"
    
    local_tz_str = config.get('timings', {}).get('assumed_recording_timezone', 'UTC')
    local_tz = pytz.timezone(local_tz_str)
    now_in_local_tz = datetime.now(local_tz)
    target_dt = dateparser.parse(time_string, settings={'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': now_in_local_tz})
    
    local_tz_str = config.get('timings', {}).get('assumed_recording_timezone', 'UTC')
    logger.info(f"Queueing SCHEDULE_MATTER command for '{target_matter_name}' with time string '{time_string}' after clarification.")
    command_to_queue = {
        "command_type": "SCHEDULE_MATTER",
        "payload": {
            "matter_name": target_matter_name,
            "time_string": time_string,
            "source": "signal_clarification"
        }
    }
    queue_command_for_executor(command_to_queue)


    formatted_time = _format_datetime_for_user(target_dt, local_tz_str)
    send_message(sender, f"OK. Matter set to '{target_matter_name}' starting {formatted_time}.", config)
    SIGNAL_SESSION_STATE.clear()


def _handle_flag_review_response(sender: str, message: str):
    """Processes user input during a flag review session."""
    config = get_config()
    command = message.strip().lower()
    
    if command == "done":
        send_message(sender, "Review ended.", config)
        SIGNAL_SESSION_STATE.clear()
        return
        
    if command == "skip":
        SIGNAL_SESSION_STATE['current_index'] += 1
        _present_next_flag_for_review()
        return

    current_flag = SIGNAL_SESSION_STATE['review_flags'][SIGNAL_SESSION_STATE['current_index']]
    flag_type = current_flag.get('flag_type')

    if flag_type in ['new_speaker_detected', 'ambiguous_speaker']:
        # Speaker flag handling
        speaker_options = SIGNAL_SESSION_STATE.get('current_flag_options', [])
        
        # Check if it's a number corresponding to an option
        if command.isdigit() and int(command) <= len(speaker_options):
            name_to_assign = speaker_options[int(command) - 1]
        else:
            name_to_assign = message.strip() # Treat as a name

        closest_match = get_closest_name_match(name_to_assign, speaker_options)
        
        if closest_match and closest_match.lower() != name_to_assign.lower():
            SIGNAL_SESSION_STATE['mode'] = 'fuzzy_match_confirm'
            SIGNAL_SESSION_STATE['fuzzy_match_original_input'] = name_to_assign
            SIGNAL_SESSION_STATE['fuzzy_match_suggestion'] = closest_match
            send_message(sender, f"Did you mean '{closest_match}'? Reply YES or NO.", config)
            return
        
        # No fuzzy match, or user entered an existing name directly. Resolve it.
        resolution = {"action": "assign", "name": name_to_assign, "context": "in_person", "source": "signal_review"}
        command_to_queue = resolve_flag(current_flag['flag_id'], resolution)
        if command_to_queue:
            _queue_item_with_priority(0, command_to_queue)
            send_message(sender, f"OK. Queued resolution for '{name_to_assign}'.", config)
        
    elif flag_type == 'matter_conflict':
        # Matter flag handling
        matter_options = SIGNAL_SESSION_STATE.get('current_flag_options', [])
        
        if SIGNAL_SESSION_STATE.get('mode') == 'matter_clarification':
            _handle_matter_clarification_response(sender, message)
            return
        
        if command.startswith("new "):
            matter_name = message.strip()[4:]
            if not matter_name:
                send_message(sender, "Please provide a name for the new matter.", config)
                return
            
            new_matter = add_matter({"name": matter_name, "source": "signal_flag_review"})
            if not new_matter:
                send_message(sender, f"Failed to create matter '{matter_name}'.", config)
                return
            send_message(sender, f"Created matter '{matter_name}'. Assigning dialogue...", config)
            # Update the chunk with the new matter ID
            update_command = {
                'type': 'UPDATE_MATTER_FOR_SPAN',
                'payload': {
                    'target_date_str': parse_flag_id_for_date(current_flag['flag_id']).strftime('%Y-%m-%d'),
                    'chunk_id': current_flag['chunk_id'],
                    'start_time': current_flag['start_time'],
                    'end_time': current_flag['end_time'],
                    'new_matter_id': new_matter['matter_id']
                }
            }
            _queue_item_with_priority(0, update_command)
            # Mark the flag as resolved
            _update_flag_status_in_queue(current_flag['flag_id'], f"resolved_as_{new_matter['name']}", parse_flag_id_for_date(current_flag['flag_id']))
        else:
            target_matter = None
            if command.isdigit() and int(command) <= len(matter_options):
                target_matter = matter_options[int(command) - 1]
            else:
                # Fuzzy match on matter name
                best_match = get_closest_name_match(command, [m['name'] for m in matter_options])
                if best_match:
                    target_matter = next((m for m in matter_options if m['name'] == best_match), None)
            
            if target_matter:
                send_message(sender, f"Assigning dialogue to matter '{target_matter['name']}'...", config)
                update_command = {
                    'type': 'UPDATE_MATTER_FOR_SPAN',
                    'payload': {
                        'target_date_str': parse_flag_id_for_date(current_flag['flag_id']).strftime('%Y-%m-%d'),
                        'chunk_id': current_flag['chunk_id'],
                        'start_time': current_flag['start_time'],
                        'end_time': current_flag['end_time'],
                        'new_matter_id': target_matter['matter_id']
                    }
                }
                _queue_item_with_priority(0, update_command)
                _update_flag_status_in_queue(current_flag['flag_id'], f"resolved_as_{target_matter['name']}", parse_flag_id_for_date(current_flag['flag_id']))
            else:
                send_message(sender, "Could not match your response to an option.", config)
                return # Don't advance to next flag on error
    
    # If successful, advance to the next flag
    SIGNAL_SESSION_STATE['current_index'] += 1
    _present_next_flag_for_review()

def _format_matter_name_if_legal_case(name: str) -> str:
    """
    If a matter name resembles 'party1 v party2', it formats it to 'Party1 v. Party2'.
    Otherwise, it capitalizes the name using title case.
    It handles 'v', 'vs', 'v.', and 'vs.' and standardizes on 'v.'.
    """
    name_stripped = name.strip()
    # This regex is designed to be simple and capture the common legal case format.
    # It captures party1, the versus separator, and party2.
    match = re.match(r'(.+?)\s+(v|vs|v\.|vs\.)\s+(.+)', name_stripped, re.IGNORECASE)
    
    if match:
        # Use title() instead of capitalize() to correctly handle multi-word party names
        # e.g., "john depp" becomes "John Depp"
        party1 = match.group(1).strip().title()
        party2 = match.group(3).strip().title()
        
        # Return the standardized and formatted name.
        return f"{party1} v. {party2}"
        
    # If it doesn't match the legal case pattern, return the title-cased version.
    return name_stripped.title()
   

def _handle_new_matter_command(sender: str, message: str):
    """Handles the 'NEW MATTER' command with optional time modifiers."""
    config = get_config()
    message_clean = message.strip()
    command_prefix_lower = "new matter"

    if ':' not in message_clean or not message_clean.lower().startswith(command_prefix_lower):
        send_message(sender, 'Invalid format. Use: new matter: <Name> or new matter [time]: <Name>', config)
        return

    # Split on the last colon to correctly separate a time modifier (which may contain colons)
    # from the matter name.
    command_part, matter_name = message_clean.rsplit(':', 1)
    matter_name = matter_name.strip()
    
    # Extract the modifier string (e.g., "", "now", "at 8:00 p.m. today")
    modifier = command_part[len(command_prefix_lower):].strip()
    matter_name = _format_matter_name_if_legal_case(matter_name)

    all_matters = [m['name'] for m in get_all_matters()]
    closest_match = get_closest_name_match(matter_name, all_matters)

    if closest_match:
        SIGNAL_SESSION_STATE['active_confirmation'] = {
            'command_details': {
                'type': 'force_create_matter',
                'payload': {'name': matter_name, 'modifier': modifier}
            },
            'expires_at': time.time() + 300
        }
        send_message(sender, f"A matter named '{closest_match}' already exists. Create '{matter_name}' anyway? Reply YES to proceed.", config)
        return
    if "at" in modifier:
        time_string = modifier.replace("at", "").strip()
        # Pre-process HHMM time format to guide dateparser
        # Find a 3 or 4 digit number that looks like a time
        time_match = re.search(r'\b(\d{3,4})\b', time_string)
        if time_match:
            hhmm_part = time_match.group(1)
            if len(hhmm_part) == 3:
                # e.g., "900" -> "09:00"
                formatted_time = f"0{hhmm_part[0]}:{hhmm_part[1:]}"
            else:
                # e.g., "1800" -> "18:00"
                formatted_time = f"{hhmm_part[:2]}:{hhmm_part[2:]}"
            # Replace the ambiguous time with a clear one
            time_string = time_string.replace(hhmm_part, formatted_time, 1)
        local_tz_str = config.get('timings', {}).get('assumed_recording_timezone', 'UTC')
        local_tz = pytz.timezone(local_tz_str)
        now_in_local_tz = datetime.now(local_tz)
        target_dt = dateparser.parse(time_string, settings={'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': now_in_local_tz})

        if target_dt:
            if target_dt.tzinfo is None:
                target_dt = local_tz.localize(target_dt)
            target_dt_utc = target_dt.astimezone(timezone.utc)

            with SCHEDULED_TASKS_LOCK:
                SCHEDULED_TASKS.append({
                    "type": "CREATE_MATTER",
                    "payload": {"name": matter_name, "source": "signal_scheduled"},
                    "run_at_utc": target_dt_utc
                })
            formatted_time = _format_datetime_for_user(target_dt, local_tz_str)
            send_message(sender, f"OK, I've scheduled matter '{matter_name}' to be created {formatted_time}.", config)
        else:
            send_message(sender, f"I couldn't understand the time '{time_string}'. Matter not created.", config)
    else:  # "now" or no modifier
        new_matter = add_matter({"name": matter_name, "source": "signal_command"})
        if not new_matter:
            send_message(sender, f"Error: Could not create matter '{matter_name}'.", config)
            return
        send_message(sender, f"Successfully created matter: '{matter_name}'.", config)
        # If the modifier is "now" or there's no time modifier, set this new matter as the active context.
        set_active_context(
            matter_id=new_matter['matter_id'],
            matter_name=new_matter['name'],
            source="signal_new_matter_command",
            environmental_context="in_person", # Default context
            config=config
        )
        if "now" in modifier:
            lookback_s = int(config.get('audio_suite_settings', {}).get('new_matter_now_lookback_s', 600))
            command = {
                'type': 'UPDATE_MATTER_FOR_RECENT_WINDOW',
                'payload': {'matter_id': new_matter['matter_id'], 'duration_s': lookback_s}
            }
            _queue_item_with_priority(0, command)
            logger.info(f"Queued command to update matter for last {lookback_s}s.")


def _handle_confirmation_response(sender: str, message: str):
    """Handles a generic YES/NO confirmation response."""
    config = get_config()
    confirmation = SIGNAL_SESSION_STATE.get('active_confirmation')

    if not confirmation or time.time() > confirmation.get('expires_at', 0):
        send_message(sender, "Sorry, that confirmation request has expired.", config)
        SIGNAL_SESSION_STATE.pop('active_confirmation', None)
        return

    response = message.strip().lower()
    details = confirmation['command_details']

    if response in ['yes', 'y']:
        if details['type'] == 'force_create_matter':
            payload = details['payload']
            matter_name = payload['name']
            modifier = payload.get('modifier', '')

            # --- Refactored logic to mirror _handle_new_matter_command ---
            if "at" in modifier:
                time_string = modifier.replace("at", "").strip()
                # Pre-process HHMM time format to guide dateparser
                time_match = re.search(r'\b(\d{3,4})\b', time_string)
                if time_match:
                    hhmm_part = time_match.group(1)
                    if len(hhmm_part) == 3:
                        formatted_time = f"0{hhmm_part[0]}:{hhmm_part[1:]}"
                    else:
                        formatted_time = f"{hhmm_part[:2]}:{hhmm_part[2:]}"
                    time_string = time_string.replace(hhmm_part, formatted_time, 1)
                local_tz_str = config.get('timings', {}).get('assumed_recording_timezone', 'UTC')
                local_tz = pytz.timezone(local_tz_str)
                now_in_local_tz = datetime.now(local_tz)
                target_dt = dateparser.parse(time_string, settings={'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': now_in_local_tz})

                if target_dt:
                    if target_dt.tzinfo is None:
                        target_dt = local_tz.localize(target_dt)
                    with SCHEDULED_TASKS_LOCK:
                        SCHEDULED_TASKS.append({"type": "CREATE_MATTER", "payload": {"name": matter_name, "source": "signal_scheduled_confirmed"}, "run_at_utc": target_dt.astimezone(timezone.utc)})
                    
                    formatted_time = _format_datetime_for_user(target_dt, local_tz_str)
                    send_message(sender, f"OK. Scheduled matter '{matter_name}' to be created {formatted_time}.", config)
                else:
                    send_message(sender, f"I couldn't understand the time '{time_string}'. Matter not created.", config)
            else:  # "now" or no modifier
                new_matter = add_matter({"name": matter_name, "source": "signal_command_confirmed"})
                if not new_matter:
                    send_message(sender, f"Error: Could not create matter '{matter_name}'.", config)
                else:
                    send_message(sender, f"OK. Successfully created matter: '{matter_name}'.", config)
                    if "now" in modifier:
                        lookback_s = int(config.get('audio_suite_settings', {}).get('new_matter_now_lookback_s', 600))
                        command = {'type': 'UPDATE_MATTER_FOR_RECENT_WINDOW', 'payload': {'matter_id': new_matter['matter_id'], 'duration_s': lookback_s}}
                        _queue_item_with_priority(0, command)
    elif response in ['no', 'n']:
        send_message(sender, "OK. Action cancelled.", config)

    SIGNAL_SESSION_STATE.pop('active_confirmation', None)

def _send_enhanced_help_message(sender: str):
    """Sends a detailed help message with system status."""
    config = get_config()
    queue_size = AUDIO_PROCESSING_QUEUE.qsize()
    flags = get_pending_flags()
    
    message_lines = [
        "Samson Control Interface",
        "\nSystem Status:",
        f"  - Audio Queue Size: {queue_size}",
        f"  - Flags Pending Review: {len(flags)}",
        "\nAvailable Commands:",
         '  - review flags',
        '  - new matter: [name]',
        '      - new matter now: [name]',
        '      - new matter at [24H Time] tomorrow: [name]',
        '  - set matter to [Name] at [24H Time]',
        "  - help: Show this message."
    ]
    
    send_message(sender, "\n".join(message_lines), config)


def main_orchestrator_signal_command_handler(message_data: Dict[str, Any], config_from_caller: Dict[str, Any]):
    """Handles all incoming Signal commands using a stateful session model."""
    try:
        if not isinstance(message_data, dict) or 'message' not in message_data:
            envelope = message_data.get('envelope', {})
            sender = envelope.get('sourceNumber')
            message = envelope.get("dataMessage", {}).get('message', "")
        else: # Direct message format
            sender = message_data.get('source_number')
            message = message_data.get('message', '')

        admin_phone_number = config_from_caller.get('signal', {}).get('recipient_phone_number')

        if sender and admin_phone_number and str(sender) != str(admin_phone_number):
            logger.warning(f"Ignoring command from unauthorized number: {sender}")
            return
        
        # If sender is missing (e.g., in some test scenarios), assume it's the admin
        if not sender:
            sender = admin_phone_number
            if not sender:
                logger.error("No sender and no admin phone number configured. Cannot process command.")
                return

        if not message or not message.strip(): return
        logger.info(f"ORCHESTRATOR_SIGNAL_HANDLER: Processing command: '{message}' from {sender}")
        
        with SIGNAL_SESSION_LOCK:
            # Check for an active confirmation request, which takes precedence
            if 'active_confirmation' in SIGNAL_SESSION_STATE:
                _handle_confirmation_response(sender, message)
                return

            # Check if the user is in a flag review session
            if SIGNAL_SESSION_STATE.get('mode') == 'flag_review':
                _handle_flag_review_response(sender, message)
                return
                
            # Check if user is responding to a fuzzy match prompt
            if SIGNAL_SESSION_STATE.get('mode') == 'fuzzy_match_confirm':
                _handle_fuzzy_match_response(sender, message)
                return
            
            if SIGNAL_SESSION_STATE.get('mode') == 'matter_clarification':
                _handle_matter_clarification_response(sender, message)
                return

        # If no active session, treat as a new command.
        # Acquire the lock again to prevent race conditions.
        with SIGNAL_SESSION_LOCK:
            command_lower = message.strip().lower()

            if command_lower.startswith("review flags"):
                _start_flag_review_session(sender)

            elif command_lower.startswith("new matter"):
                _handle_new_matter_command(sender, message) # Pass original message
            
            elif command_lower.startswith("set matter to"):
                _handle_set_matter_command(sender, message)

            elif command_lower == "help":
                _send_enhanced_help_message(sender)
            elif re.match(r'^(?:settime\s+)?([\d:.]+)$', command_lower):
                logger.info(f"Matched SETTIME command: '{message.strip()}'")
                match = re.match(r'^(?:settime\s+)?([\d:.]+)$', command_lower)
                time_str_raw = match.group(1)
                
                try:
                    # Normalize the time string by removing separators
                    time_str_normalized = time_str_raw.replace(':', '').replace('.', '')

                    # ADDED: Handle 3-digit time like "905" -> "0905"
                    if len(time_str_normalized) == 3:
                        time_str_normalized = "0" + time_str_normalized
                    
                    # Ensure the length is valid before parsing
                    if not 3 <= len(time_str_normalized) <= 4:
                        raise ValueError("Invalid time format length.")

                    parsed_time_obj = datetime.strptime(time_str_normalized, '%H%M').time()

                    samson_today_date = get_samson_today()
                    local_dt_naive = datetime.combine(samson_today_date, parsed_time_obj)
                    
                    local_tz_str = config_from_caller.get('timings', {}).get('assumed_recording_timezone', 'UTC')
                    local_tz = pytz.timezone(local_tz_str)
                    local_dt_aware = local_tz.localize(local_dt_naive)
                    
                    utc_start_time = local_dt_aware.astimezone(pytz.utc)

                    target_date_for_log = datetime.combine(samson_today_date, datetime.min.time())
                    
                    if set_day_start_time(utc_start_time, target_date=target_date_for_log):
                        success_msg = f"OK. Recording start time for {samson_today_date.strftime('%Y-%m-%d')} set to {local_dt_aware.strftime('%H:%M')} {local_tz_str}. Processing resumed."
                        send_message(sender, success_msg, config_from_caller)
                        if ORCHESTRATOR_EVENT_HANDLER:
                            ORCHESTRATOR_EVENT_HANDLER.process_initial_batch_after_time_set()
                            logger.info("Signaled folder monitor to resume processing after SETTIME.")
                        else:
                            logger.warning("ORCHESTRATOR_EVENT_HANDLER not found. Cannot resume processing automatically.")
                    else:
                        send_message(sender, "Error: Failed to set the start time in the daily log.", config_from_caller)

                except ValueError:
                    send_message(sender, f"Error: Invalid time format '{time_str_raw}'. Please use HH:MM or HHMM.", config_from_caller)
                except Exception as e_settime:
                    logger.error(f"Error processing SETTIME command: {e_settime}", exc_info=True)
                    send_message(sender, "An internal error occurred while setting the time.", config_from_caller)
            else:
                # Fallback to the existing LLM-based command parser
                logger.info("No structured command matched. Falling back to LLM-based entity extraction.")
                from src.llm_interface import extract_structured_entities
                from src.matter_manager import get_all_matters, add_matter
                
                llm_command_parser = get_llm_chat_model(config_from_caller, 'llm_command_parser')
                if llm_command_parser:
                    extracted_data = extract_structured_entities(message.strip(), llm_command_parser)
                    if extracted_data:
                        command_type = extracted_data.get("command_type")

                        if not command_type:
                            send_message(sender, "I couldn't fully understand your request.", config_from_caller)
                            return
                        
                        if command_type == "GENERATE_WORKFLOW_FROM_PROMPT":
                            user_prompt = extracted_data.get("user_prompt")
                            if not user_prompt:
                                send_message(sender, "I understood you want a workflow, but I didn't get the description. Please try again.", config_from_caller)
                                return

                            logger.info(f"Signal command: Queuing workflow generation for prompt: '{user_prompt}'")
                            command_for_executor = {
                                "command_type": "GENERATE_WORKFLOW_FROM_PROMPT",
                                "command_payload": {
                                    "user_prompt": user_prompt
                                }
                            }
                            queue_command_for_executor(command_for_executor)
                            send_message(sender, f"OK, I've started generating a workflow for: '{user_prompt}'. It will appear in the UI when complete.", config_from_caller)
                            return

                        # --- If not a workflow command, proceed with matter-related logic ---
                        matter_name_from_llm = extracted_data.get("matter_name")
                        if not matter_name_from_llm:
                            send_message(sender, "I couldn't fully understand your request for a matter-related command.", config_from_caller)
                            return

                        from src.command_executor_service import CommandExecutorService
                        temp_executor = CommandExecutorService(config_from_caller, threading.Event(), (lambda p, d: None), None)
                        resolved_id, resolved_name = temp_executor._find_matter_by_fuzzy_name(matter_name_from_llm)

                        # Handle matter creation if it doesn't exist
                        if not resolved_id:
                            logger.info(f"Signal command for non-existent matter '{matter_name_from_llm}'. Creating it.")
                            new_matter = add_matter({"name": matter_name_from_llm, "source": "signal_llm_auto_create"})
                            if not new_matter:
                                send_message(sender, f"Failed to create new matter '{matter_name_from_llm}'.", config_from_caller)
                                return
                            resolved_id = new_matter['matter_id']
                            resolved_name = new_matter['name']

                        if command_type == "FORCE_SET_MATTER":
                            duration_str = extracted_data.get("duration")
                            if not duration_str:
                                send_message(sender, "Please specify a duration (e.g., 'for the last 5 minutes').", config_from_caller)
                                return
                            
                            from src.daily_log_manager import parse_duration_to_minutes
                            lookback_s = parse_duration_to_minutes(duration_str) * 60
                            
                            force_command = {
                                "command_type": "FORCE_SET_MATTER",
                                "payload": {
                                    "matter_id": resolved_id,
                                    "lookback_seconds": lookback_s
                                }
                            }
                            queue_command_for_executor(force_command)
                            send_message(sender, f"OK. Forcing matter to '{resolved_name}' for the last {duration_str}.", config_from_caller)

                        elif command_type == "SCHEDULE_MATTER":
                            time_string = extracted_data.get("time_string", "now")
                            
                            if time_string.lower().strip() == "now":
                                # Queue for the worker's in-memory queue
                                timed_update_cmd = {
                                    "type": "APPLY_TIMED_MATTER_UPDATE",
                                    "payload": {
                                        "new_matter_id": resolved_id,
                                        "start_time_utc": datetime.now(timezone.utc),
                                        "source": "signal_now"
                                    }
                                }
                                _queue_item_with_priority(0, timed_update_cmd)
                                send_message(sender, f"OK. Setting matter to '{resolved_name}' effective now.", config_from_caller)
                            else:
                                local_tz_str = config_from_caller.get('timings', {}).get('assumed_recording_timezone', 'UTC')
                                local_tz = pytz.timezone(local_tz_str)
                                now_in_local_tz = datetime.now(local_tz)
                                target_dt = dateparser.parse(time_string, settings={'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': now_in_local_tz})
                                
                                if not target_dt:
                                    send_message(sender, f"I couldn't understand the time '{time_string}'. The matter was not scheduled.", config_from_caller)
                                    return

                                if target_dt.tzinfo is None:
                                    target_dt = local_tz.localize(target_dt)

                                schedule_cmd = {
                                    "command_type": "SCHEDULE_MATTER",
                                    "payload": {
                                        "matter_name": resolved_name,
                                        "time_string": target_dt.astimezone(timezone.utc).isoformat(),
                                        "source": "signal_llm_parser",
                                        "environmental_context": extracted_data.get("environmental_context", "in_person")
                                    }
                                }
                                queue_command_for_executor(schedule_cmd)

                                formatted_time = _format_datetime_for_user(target_dt, local_tz_str)
                                send_message(sender, f"OK, I've scheduled a switch to matter '{resolved_name}' for {formatted_time}.", config_from_caller)
                        
                        elif command_type == "CREATE_MATTER":
                            time_string = extracted_data.get("time_string", "now")
                            if time_string.lower().strip() != "now":
                                schedule_cmd = {
                                    "command_type": "SCHEDULE_MATTER",
                                    "payload": { "matter_name": resolved_name, "time_string": time_string, "source": "signal_llm_parser" }
                                }
                                queue_command_for_executor(schedule_cmd)
                                send_message(sender, f"OK, created matter '{resolved_name}' and scheduled it to start at '{time_string}'.", config_from_caller)
                            else:
                                set_active_context(resolved_id, resolved_name, "signal_new_matter", extracted_data.get("environmental_context", "in_person"), config_from_caller)
                                send_message(sender, f"OK, created and activated matter '{resolved_name}'.", config_from_caller)

                        return # Stop processing

                # If we reach here, it's truly an unknown command
                unknown_cmd_msg = f"Unknown command: '{message.strip()}'. Try HELP."
                send_message(sender, unknown_cmd_msg, config_from_caller)

    except Exception as e:
        logger.error(f"ORCHESTRATOR_SIGNAL_HANDLER: Unhandled exception in main handler: {e}", exc_info=True)


def on_new_audio_file_detected(file_path: Path):
    """Callback function for the folder monitor. Puts new files in the queue."""
    logger.info(f"ORCHESTRATOR: File '{file_path.name}' passed monitor checks, adding to processing queue.")
    # Priority 1 for normal audio files. 0 could be for high-priority commands.
    _queue_item_with_priority(1, str(file_path))

def _move_file_to_error_folder(file_to_move: Optional[Path], worker_config: Dict[str, Any], reason: str = "unknown error"):
    """Moves a file to the configured error folder."""
    if not file_to_move or not file_to_move.exists(): 
        logger.warning(f"Attempted to move non-existent file ({file_to_move}) to error folder due to: {reason}.")
        return
    audio_error_folder_path = worker_config.get('paths', {}).get('audio_error_folder')
    if not audio_error_folder_path: 
        logger.error(f"audio_error_folder not configured. Cannot move {file_to_move.name} (reason: {reason}).")
        return
    try:
        error_folder_path = Path(audio_error_folder_path)
        error_folder_path.mkdir(parents=True, exist_ok=True)
        target_error_path = error_folder_path / file_to_move.name
        if file_to_move.resolve() == target_error_path.resolve(): 
            logger.warning(f"File {file_to_move.name} is already in the target error location. Skipping move. Reason: {reason}")
            return
        shutil.move(str(file_to_move), str(target_error_path))
        logger.info(f"Moved {file_to_move.name} to error folder {target_error_path} due to: {reason}.")
    except Exception as move_err: 
        logger.error(f"Failed to move {file_to_move.name} to error folder (reason: {reason}): {move_err}")

class CommandQueueMonitor:
    """
    A simple monitor that watches a directory for command files, loads them,
    and puts them onto the main processing queue for the worker.
    """
    def __init__(self, queue_dir: Path, processing_queue: queue.PriorityQueue, shutdown_event: threading.Event):
        self.queue_dir = queue_dir
        self.processing_queue = processing_queue
        self.shutdown_event = shutdown_event
        self.thread_name = "CommandQueueMonitor"
        logger.info(f"{self.thread_name}: Initialized to watch '{self.queue_dir}'.")

    def run(self):
        logger.info(f"{self.thread_name}: Run loop started.")
        self.queue_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists at start

        while not self.shutdown_event.is_set():
            try:
                # Find all command files
                command_files = list(self.queue_dir.glob("cmd_*.json"))
                if command_files:
                    logger.debug(f"{self.thread_name}: Found {len(command_files)} command file(s).")
                    for cmd_file in command_files:
                        self._process_command_file(cmd_file)
                        if self.shutdown_event.is_set(): break # Exit loop early if shutdown
            except Exception as e:
                logger.error(f"{self.thread_name}: Error during directory scan: {e}", exc_info=True)

            # Wait for a short interval before scanning again
            self.shutdown_event.wait(timeout=2.0)

        logger.info(f"{self.thread_name}: Run loop ended.")

    def _process_command_file(self, cmd_file_path: Path):
        """Loads, queues, and deletes a single command file."""
        logger.info(f"{self.thread_name}: Processing command file: {cmd_file_path.name}")
        try:
            with open(cmd_file_path, 'r', encoding='utf-8') as f:
                command_data = json.load(f)

            # Priority 0 for GUI commands to ensure they are handled promptly
            _queue_item_with_priority(0, command_data)
            logger.info(f"{self.thread_name}: Queued command of type '{command_data.get('type')}' for worker.")

        except json.JSONDecodeError:
            logger.error(f"{self.thread_name}: Corrupted command file (invalid JSON): {cmd_file_path.name}. Moving to 'corrupted'.")
            corrupted_dir = self.queue_dir / "corrupted"
            corrupted_dir.mkdir(exist_ok=True)
            try:
                shutil.move(str(cmd_file_path), str(corrupted_dir / cmd_file_path.name))
            except Exception as e_move:
                logger.error(f"Could not move corrupted command file: {e_move}")
                os.remove(cmd_file_path) # Fallback to just deleting it
        except Exception as e:
            logger.error(f"{self.thread_name}: Error processing command file {cmd_file_path.name}: {e}", exc_info=True)
        finally:
            # Ensure the file is deleted if it still exists
            if cmd_file_path.exists():
                try:
                    os.remove(cmd_file_path)
                except OSError as e_del:
                    logger.error(f"{self.thread_name}: CRITICAL - Failed to delete processed command file {cmd_file_path.name}: {e_del}")
class AudioProcessingWorker(threading.Thread):
    def __init__(self, worker_config: Dict[str, Any], 
                 dynamic_config_store: Dict[str, float],
                 db_lock: threading.Lock, 
                 faiss_index_path: Optional[Path],
                 speaker_map_path: Optional[Path],
                 summary_llm: BaseLanguageModel,
                 matter_analysis_service: MatterAnalysisService,
                 embedding_model: SentenceTransformer):
        super().__init__(daemon=True, name="AudioProcessingWorker")
        self.config = worker_config
        self.shutdown_event = threading.Event()
        self.master_log_current_day_str: Optional[str] = None
        self.master_log_last_speaker: Optional[str] = None
        self.master_log_next_timestamp_marker_abs_utc: Optional[datetime] = None
        self.force_header_rewrite_for_day: Optional[str] = None
        self.dynamic_config_store = dynamic_config_store
        self.db_lock = db_lock # Store the lock
        self.summary_llm = summary_llm
        self.matter_analysis_service = matter_analysis_service
        self.embedding_model = embedding_model

        self.faiss_idx_path = faiss_index_path 
        self.speaker_map_path = speaker_map_path 
       
        self.faiss_index = None
        self.speaker_map = None

        # <<< START MODIFICATION >>>
        self.pending_matter_updates: List[Dict[str, Any]] = []
        # last_processed_turn_matter_id is already present, which is correct.
        
        # Load persisted state on startup
        try:
            state_file = self.config['paths']['system_state_dir'] / "worker_state.json"
            if state_file.exists():
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.pending_matter_updates = state.get('pending_matter_updates', [])
                    # Deserialize datetime strings back to datetime objects
                    for update in self.pending_matter_updates:
                        if 'start_time_utc' in update['payload'] and isinstance(update['payload']['start_time_utc'], str):
                            update['payload']['start_time_utc'] = datetime.fromisoformat(update['payload']['start_time_utc'])
                    self.last_processed_turn_matter_id = state.get('last_processed_turn_matter_id')
                    logger.info("Successfully loaded persisted worker state.")
        except Exception as e:
            logger.warning(f"Could not load persisted worker state: {e}", exc_info=True)
        # <<< END MODIFICATION >>>

        # Inside AudioProcessingWorker.__init__
        self.pending_refinements: List[Dict[str, Any]] = []
        self.refinement_commit_timer: Optional[threading.Timer] = None
        self.timer_lock = threading.Lock()
        self.batch_commit_delay_s: int = self.config.get('speaker_intelligence', {}).get('batch_commit_inactivity_seconds', 120)
        self.last_processed_turn_matter_id: Optional[str] = None
        self.last_word_end_time_utc: Optional[str] = None # AC1: New state variable

    # Add these new methods inside the AudioProcessingWorker class
    def _schedule_refinement_commit(self):
        """Schedules or resets the timer to commit the pending refinement batch."""
        with self.timer_lock:
            if self.refinement_commit_timer:
                self.refinement_commit_timer.cancel()
            self.refinement_commit_timer = threading.Timer(
                self.batch_commit_delay_s,
                lambda: _queue_item_with_priority(0, {"type": "COMMIT_REFINEMENT_BATCH"})
            )
            self.refinement_commit_timer.start()
            logger.info(f"Scheduled refinement batch commit in {self.batch_commit_delay_s}s.")

    def _commit_pending_refinements(self):
        """Commits the current batch of pending refinements to the database."""
        with self.timer_lock:
            if not self.pending_refinements:
                logger.info("Commit refinements called, but batch is empty.")
                return
            logger.info(f"Committing a batch of {len(self.pending_refinements)} refinements to the speaker database.")
            refinements_to_commit = self.pending_refinements.copy()
            self.pending_refinements.clear()
        
        aps_speaker_id.update_faiss_embeddings_for_refinement(
            refinements_to_commit,
            self.faiss_idx_path,
            self.speaker_map_path,
            self.db_lock
        )

    #  New helper for surgical correction audio processing.
    def _enroll_and_populate_new_speaker(self, new_speaker_name: str, corrections_list: List[Dict[str, Any]], correction_context: str, chunk_id: str, target_date_utc_dt: datetime):
        """
        Consolidated helper to enroll a new speaker and populate their profile with
        a high-quality embedding derived from a list of corrections.
        """
        logger.info(f"WORKER: HELPER - Starting enrollment and population for new speaker '{new_speaker_name}'.")
        
        audio_tensors_for_concatenation: List[torch.Tensor] = []
        files_to_load_cache: Dict[str, Tuple[torch.Tensor, int]] = {}
        total_duration_s = 0.0
    
        for correction in corrections_list:
            try:
                original_file_name = correction["original_file_name"]
                word_data = correction["word_data"]
                start_time, end_time = float(word_data["start"]), float(word_data["end"])
                
                processed_word_data = self._get_audio_and_embedding_for_correction(
                    original_file_name, target_date_utc_dt, start_time, end_time, files_to_load_cache
                )
                if processed_word_data:
                    audio_tensor, _ = processed_word_data
                    audio_tensors_for_concatenation.append(audio_tensor)
            except (KeyError, ValueError) as e_inner:
                logger.error(f"WORKER: HELPER - Skipping a correction snippet due to invalid data: {correction}. Error: {e_inner}")
    
        if not audio_tensors_for_concatenation:
            logger.error(f"WORKER: HELPER - Could not extract any audio for new speaker '{new_speaker_name}'. Aborting enrollment.")
            return
    
        # Generate ONE high-quality embedding from the concatenated audio
        concatenated_audio = torch.cat(audio_tensors_for_concatenation, dim=1)
        total_duration_s = float(concatenated_audio.shape[-1]) / 16000.0
        
        embedding_model = _loaded_models_cache.get("embedding_model")
        with torch.no_grad():
            device = self.config.get('global_ml_device', 'cpu')
            raw_embedding = embedding_model.encode_batch(concatenated_audio.to(device))
        
        main_embedding_np = raw_embedding.mean(dim=1).squeeze(0).cpu().numpy() if raw_embedding.dim() == 3 else raw_embedding.squeeze(0).cpu().numpy()
    
        # Enroll using the robust embedding
        success, new_index, new_map = aps_speaker_id.enroll_speakers_programmatic(
            [{'name_to_enroll': new_speaker_name, 'embedding': main_embedding_np, 'context': correction_context}],
            self.faiss_idx_path, self.speaker_map_path, self.config['audio_suite_settings']['embedding_dim'], self.db_lock
        )
    
        if not success:
            logger.error(f"WORKER: HELPER - Programmatic enrollment failed for '{new_speaker_name}'.")
            return
    
        self.faiss_index, self.speaker_map = new_index, new_map
        final_faiss_id = next((fid for fid, s_info in new_map.items() if isinstance(s_info, dict) and s_info.get('name') == new_speaker_name), -1)
        
        if final_faiss_id != -1:
            create_speaker_profile(faiss_id=final_faiss_id, name=new_speaker_name)
            
            # Add this data as the first evolution segment
            update_or_remove_evolution_segment(
                faiss_id=final_faiss_id, context=correction_context, chunk_id=chunk_id,
                new_embedding=main_embedding_np, new_duration_s=total_duration_s
            )
            
            # Log feedback
            log_correction_feedback("new_speaker_enroll_from_correction", {
                "faiss_id_of_correct_speaker": final_faiss_id, "corrected_speaker_id": new_speaker_name,
                "source": "gui_correction", "audio_context": correction_context, "chunk_id": chunk_id,
                "num_words_corrected": len(corrections_list), "duration_s": total_duration_s 
            })
            logger.info(f"WORKER: HELPER - Successfully enrolled and populated profile for '{new_speaker_name}' with ID {final_faiss_id}.")
        else:
            logger.error(f"WORKER: HELPER - Could not find FAISS ID for '{new_speaker_name}' after successful enrollment.")

    def _apply_pending_updates(self, chunk_result: Dict[str, Any], target_date: datetime) -> Optional[str]:
        """
        Applies time-synced matter updates from the pending queue to the just-processed chunk.
        """
        if not self.pending_matter_updates:
            return None

        try:
            chunk_start_utc_str = chunk_result.get('chunk_start_time_utc')
            if not chunk_start_utc_str:
                logger.error("Cannot apply pending updates: chunk_start_time_utc is missing from chunk result.")
                return None
            
            chunk_start_utc = datetime.fromisoformat(chunk_start_utc_str)
            
            max_relative_end_s = 0.0
            for segment in chunk_result.get('identified_transcript', []):
                for word in segment.get('words', []):
                    max_relative_end_s = max(max_relative_end_s, word.get('end', 0.0))
            
            chunk_end_utc = chunk_start_utc + timedelta(seconds=max_relative_end_s)

            # Select updates that fall before or within this chunk's timeframe
            updates_to_process = [
                upd for upd in self.pending_matter_updates 
                if upd['payload']['start_time_utc'] <= chunk_end_utc
            ]

            if not updates_to_process:
                return None
            
            # Sort updates chronologically to apply them in the correct order
            updates_to_process.sort(key=lambda x: x['payload']['start_time_utc'])
            
            last_applied_matter_id: Optional[str] = None
            processed_update_payloads = []

            for update in updates_to_process:
                payload = update['payload']
                update_time_utc = payload['start_time_utc']
                
                # Calculate when the update starts relative to the chunk's beginning
                relative_start_s = max(0.0, (update_time_utc - chunk_start_utc).total_seconds())
                
                logger.info(f"Applying timed update for matter '{payload['new_matter_id']}' at relative time {relative_start_s:.2f}s.")
                
                update_matter_segments_for_chunk(
                    target_date=target_date,
                    chunk_id=chunk_result['chunk_id'],
                    start_time=relative_start_s,
                    end_time=max_relative_end_s, # Apply until the end of the chunk
                    new_matter_id=payload['new_matter_id']
                )

                last_applied_matter_id = payload['new_matter_id']
                # After surgically updating the past chunk, set the global context for all future chunks.
                from src import context_manager
                
                new_matter_id = payload.get('new_matter_id')
                new_matter_name = payload.get('new_matter_name')

                if new_matter_id and new_matter_name:
                    context_manager.set_active_context(
                        matter_id=new_matter_id,
                        matter_name=new_matter_name,
                        source=payload.get('source', 'timed_update'),
                        environmental_context=payload.get('environmental_context', 'in_person'),
                        config=self.config
                    )
                    logger.info(f"Global context updated to '{new_matter_name}' by timed update.")
                else:
                    logger.error(f"Could not update global context: new_matter_id or new_matter_name missing from timed update payload: {payload}")
                
                # If the update came from the scheduler, mark the event as complete
                if payload.get('source') == 'scheduler' and payload.get('source_event_id'):
                    from src import event_manager
                    event_manager.update_item_status(payload['source_event_id'], "COMPLETED")
                    logger.info(f"Marked scheduled event {payload['source_event_id']} as COMPLETED.")
                
                processed_update_payloads.append(update)

            # Remove the processed updates from the main pending list
            self.pending_matter_updates = [upd for upd in self.pending_matter_updates if upd not in processed_update_payloads]
            
            return last_applied_matter_id

        except Exception as e:
            logger.error(f"Error in _apply_pending_updates: {e}", exc_info=True)
            return None

    def _audit_all_logs_for_speaker_data(self, start_date: date, end_date: date) -> Dict[str, List[Dict[str, Any]]]:
        """
        Performs a comprehensive, word-level audit of all daily logs within a date range.

        Returns:
            A dictionary mapping each speaker_name to a list of their word objects,
            where each word is enriched with necessary context for reprocessing.
        """
        logger.info(f"WORKER: AUDIT - Starting full word-level audit from {start_date} to {end_date}.")
        speaker_word_collection: Dict[str, List[Dict[str, Any]]] = {}
        
        current_date = start_date
        while current_date <= end_date:
            log_path = get_log_file_path(datetime.combine(current_date, datetime.min.time()))
            if not log_path.exists():
                current_date += timedelta(days=1)
                continue

            log_data = get_daily_log_data(datetime.combine(current_date, datetime.min.time()))
            for chunk_id, entry in log_data.get("chunks", {}).items():
                if not isinstance(entry, dict): continue
                original_file_name = entry.get("original_file_name")
                if not original_file_name: continue

                wlt = entry.get("processed_data", {}).get("word_level_transcript_with_absolute_times", [])
                for segment in wlt:
                    if not isinstance(segment, dict): continue
                    for word in segment.get("words", []):
                        if not isinstance(word, dict): continue
                        speaker_name = word.get("speaker")
                        if not speaker_name: continue
                        
                        # Enrich the word object with context needed for reprocessing
                        enriched_word = word.copy()
                        if 'absolute_start_utc' in word:
                            enriched_word['absolute_start_utc'] = word['absolute_start_utc']

                        enriched_word['audio_file_stem'] = Path(original_file_name).stem
                        enriched_word['processing_date_of_log'] = current_date.strftime('%Y-%m-%d')
                        
                        if speaker_name not in speaker_word_collection:
                            speaker_word_collection[speaker_name] = []
                        speaker_word_collection[speaker_name].append(enriched_word)
            
            current_date += timedelta(days=1)
            
        logger.info(f"WORKER: AUDIT - Audit complete. Found dialogue for {len(speaker_word_collection)} unique speakers.")
        return speaker_word_collection

    def _audit_and_group_speaker_diarization(self, start_date: date, end_date: date) -> Dict[str, List[Dict[str, Any]]]:
        """
        Audits logs and groups consecutive words by the same speaker into continuous segments.
        This provides "natural" speech clips for high-quality embedding generation, respecting
        post-correction speaker assignments at the word level.
    
        Returns:
            A dictionary mapping each speaker_name to a list of their aggregated speech segments.
        """
        logger.info(f"WORKER: AUDIT - Starting segment-grouping audit from {start_date} to {end_date}.")
        speaker_segments_collection: Dict[str, List[Dict[str, Any]]] = {}
        
        current_date = start_date
        while current_date <= end_date:
            log_path = get_log_file_path(datetime.combine(current_date, datetime.min.time()))
            if not log_path.exists():
                current_date += timedelta(days=1)
                continue
    
            log_data = get_daily_log_data(datetime.combine(current_date, datetime.min.time()))
            
            for chunk_id, entry in log_data.get("chunks", {}).items():
                if not isinstance(entry, dict): continue
                original_file_name = entry.get("original_file_name")
                if not original_file_name: continue
                
                all_words_in_chunk = []
                wlt = entry.get("processed_data", {}).get("word_level_transcript_with_absolute_times", [])
                for segment in wlt:
                    all_words_in_chunk.extend(segment.get("words", []))
                
                all_words_in_chunk.sort(key=lambda x: x.get('start', float('inf')))
                if not all_words_in_chunk: continue
                
                current_segment = None
                gap_threshold_s = 0.75 # A reasonable gap to define a new utterance

                for word in all_words_in_chunk:
                    speaker = word.get("speaker")
                    if not speaker or speaker.startswith("CUSID_"): continue

                    # Calculate word duration once, and skip if invalid.
                    word_duration = word.get('end', 0.0) - word.get('start', 0.0)
                    if word_duration <= 0: continue

                    word_date = None
                    if 'absolute_start_utc' in word and word['absolute_start_utc']:
                        try:
                            word_date = datetime.fromisoformat(word['absolute_start_utc']).date()
                        except (ValueError, TypeError): pass
    
                    # Continue the segment only if the speaker is the same AND the gap is small
                    if current_segment and speaker == current_segment['speaker'] and (word.get('start', 0.0) - current_segment.get('end_s', 0.0)) < gap_threshold_s and word_date == current_segment.get('_word_date'):
                        current_segment['end_s'] = word['end']
                        # Accumulate the accurate word duration
                        current_segment['summed_word_duration_s'] += word_duration
                    else:
                        if current_segment:
                            if current_segment['speaker'] not in speaker_segments_collection: speaker_segments_collection[current_segment['speaker']] = []
                            speaker_segments_collection[current_segment['speaker']].append(current_segment)
                        
                        # Start a new segment, initializing the summed duration
                        current_segment = {'speaker': speaker, 'start_s': word['start'], 'end_s': word['end'],
                                         'summed_word_duration_s': word_duration, # Initialize with the first word's duration
                                         'absolute_start_utc': word.get('absolute_start_utc'),
                                         'audio_file_stem': Path(original_file_name).stem,
                                         'processing_date_of_log': current_date.strftime('%Y-%m-%d'),
                                         '_word_date': word_date
                                         }
                if current_segment:
                    if current_segment['speaker'] not in speaker_segments_collection: speaker_segments_collection[current_segment['speaker']] = []
                    speaker_segments_collection[current_segment['speaker']].append(current_segment)
            current_date += timedelta(days=1)
        return speaker_segments_collection

    def _group_words_into_segments(self, words_list: List[Dict[str, Any]], gap_threshold_s: float = 0.5) -> List[Dict[str, Any]]:
            """Groups a sorted list of words from a single speaker into continuous speech segments."""
            if not words_list:
                return []

            segments = []
            # Ensure words are sorted by start time, as the input list might not be.
            sorted_words = sorted(words_list, key=lambda x: x.get('start', float('inf')))
            
            current_segment = None
            for word in sorted_words:
                word_duration = word.get('end', 0.0) - word.get('start', 0.0)
                if word_duration <= 0: continue

                # Start a new segment if it's the first word, or if there's a large gap
                if current_segment is None or (word['start'] - current_segment['end_s']) > gap_threshold_s:
                    if current_segment:
                        segments.append(current_segment)
                    
                    current_segment = {
                        'start_s': word['start'],
                        'end_s': word['end'],
                        'summed_word_duration_s': word_duration
                    }
                else: # Continue the current segment
                    current_segment['end_s'] = word['end']
                    current_segment['summed_word_duration_s'] += word_duration
            
            if current_segment:
                segments.append(current_segment)

            return segments





    # <<< ADD THIS NEW METHOD >>>
    def _initialize_db_state(self):
        """Initializes or re-initializes the in-memory speaker DB from disk if not already loaded."""
        if self.faiss_index is not None and self.speaker_map is not None:
            # Already initialized, no need to do anything.
            return

        logger.info("WORKER: In-memory DB state is not initialized. Attempting to load from disk...")
        if self.faiss_idx_path and self.speaker_map_path:
            try:
                with self.db_lock:
                    embedding_dim = self.config.get('audio_suite_settings', {}).get('embedding_dim', 192)
                    # Load or create the FAISS index
                    self.faiss_index = load_or_create_faiss_index(self.faiss_idx_path, embedding_dim)
                    # Load or create the speaker map
                    self.speaker_map = load_or_create_speaker_map(self.speaker_map_path)
                    logger.info(f"WORKER: Speaker DB loaded successfully. Index has {self.faiss_index.ntotal} vectors. Map has {len(self.speaker_map)} speakers.")
            except Exception as e:
                logger.error(f"WORKER: CRITICAL - Failed to load speaker database during initialization: {e}", exc_info=True)
                # Set to None to allow re-attempt on next loop iteration
                self.faiss_index = None
                self.speaker_map = None
        else:
            logger.warning("WORKER: Speaker DB paths not configured; cannot initialize speaker state.")

    def _get_audio_and_embedding_for_correction(self, file_name: str, file_date: datetime, start_s: float, end_s: float, loaded_files_cache: dict) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
        """Helper to get a word's audio tensor and its individual embedding, caching loaded audio files.

        Returns:
            A tuple of (audio_tensor, embedding_ndarray), or None.
        """
        if file_name in loaded_files_cache:
            waveform, sample_rate = loaded_files_cache[file_name]
        else:
            archive_folder = self.config.get('paths', {}).get('archived_audio_folder')
            if not archive_folder:
                logger.error("WORKER: Cannot get embedding, archive folder not configured.")
                return None
           
            original_audio_path = Path(archive_folder) / file_date.strftime('%Y-%m-%d') / file_name
            if not original_audio_path.exists():
                logger.warning(f"WORKER: Cannot get embedding, original audio file not found at {original_audio_path}")
                return None
           
            waveform, sample_rate = aps_audio.load_audio(original_audio_path, target_sr=16000)
            if waveform is None:
                return None
            loaded_files_cache[file_name] = (waveform, sample_rate)

        start_sample = int(start_s * sample_rate)
        end_sample = int(end_s * sample_rate)
        start_sample, end_sample = max(0, start_sample), min(waveform.shape[-1], end_sample)

        if start_sample >= end_sample:
            return None
       
        snippet_tensor = waveform[:, start_sample:end_sample]
        embedding_model = _loaded_models_cache.get("embedding_model")
        if not embedding_model:
            return None
       
        with torch.no_grad():
            device = self.config.get('global_ml_device', 'cpu')
            raw_embedding = embedding_model.encode_batch(snippet_tensor.to(device))
       
        embedding_np = None
        if raw_embedding.dim() == 3: embedding_np = raw_embedding.mean(dim=1).squeeze(0).cpu().numpy()
        elif raw_embedding.dim() == 2: embedding_np = raw_embedding.squeeze(0).cpu().numpy()
        elif raw_embedding.dim() == 1: embedding_np = raw_embedding.cpu().numpy()
        
        if embedding_np is None:
            return None

        return snippet_tensor, embedding_np

    def _save_speaker_db(self):
        # This function is no longer needed as writes are saved immediately.
        # The lock ensures that the final state is consistent on disk.
        logger.info("Audio worker shutdown: DB state is saved upon modification, no final save needed.")
        pass

    def stop(self): self.shutdown_event.set()

    def run(self):
        logger.info("Audio processing worker started.")
        while not self.shutdown_event.is_set():
            # --- Ensure DB state is initialized for this loop iteration ---
            self._initialize_db_state()
            if self.faiss_index is None or self.speaker_map is None:
                logger.error("WORKER: DB state is not available, cannot process items. Retrying in 5 seconds.")
                self.shutdown_event.wait(5)
                continue # Skip to next loop iteration to retry initialization

            audio_file_path_for_error_handling: Optional[Path] = None; dequeued_item_info: Any = None # Changed type to Any
            try:
                # In AudioProcessingWorker.run(), at the start of the 'try:' block
                priority, _, dequeued_item = AUDIO_PROCESSING_QUEUE.get(timeout=1)
                logger.info(f"[Worker] Dequeued item with priority {priority}. Type: {type(dequeued_item)}. Content: {str(dequeued_item)[:500]}...")
                logger.debug("WORKER: Dequeued item (Priority: %s): %s", priority, json.dumps(dequeued_item, indent=2, default=str))
                dequeued_item_info = (priority, dequeued_item)
                

                if isinstance(dequeued_item, dict) and dequeued_item.get('type') == 'APPLY_TIMED_MATTER_UPDATE':
                    logger.info("WORKER: Received APPLY_TIMED_MATTER_UPDATE command.")
                    # Basic validation
                    payload = dequeued_item.get('payload', {})
                    if 'new_matter_id' in payload and 'start_time_utc' in payload:
                        self.pending_matter_updates.append(dequeued_item)
                        logger.info(f"Added timed matter update to pending queue. Queue size: {len(self.pending_matter_updates)}.")
                    else:
                        logger.error(f"Invalid APPLY_TIMED_MATTER_UPDATE command, missing required keys: {payload}")


                elif isinstance(dequeued_item, dict) and dequeued_item.get('type') == 'COMMIT_REFINEMENT_BATCH':
                    logger.info("WORKER: Processing COMMIT_REFINEMENT_BATCH command.")
                    self._commit_pending_refinements()
                
                elif isinstance(dequeued_item, dict) and dequeued_item.get('type') == 'REANALYZE_MATTERS_FOR_DATE_RANGE':
                    logger.info("WORKER: Processing REANALYZE_MATTERS_FOR_DATE_RANGE command.")
                    try:
                        from src import matter_manager
                        payload = dequeued_item.get('payload', {})
                        lookback_days = int(payload.get('lookback_days', 1))
                        
                        all_matters_fresh = matter_manager.get_all_matters(include_inactive=True)
                        self.matter_analysis_service._update_matter_embeddings_cache(all_matters_fresh)
                        logger.info(f"Refreshed matter embeddings cache with {len(all_matters_fresh)} matters.")

                        end_date = datetime.now(timezone.utc)
                        start_date = end_date - timedelta(days=lookback_days)
                        all_profiles = get_all_speaker_profiles()
                        
                        current_date = start_date
                        while current_date.date() <= end_date.date():
                            log_data = get_daily_log_data(current_date)
                            if not log_data or not log_data.get("chunks"):
                                current_date += timedelta(days=1)
                                continue
                            
                            for chunk_id, chunk_data in log_data.get("chunks", {}).items():
                                original_segments = chunk_data.get("matter_segments", [])
                                needs_reanalysis = any(s.get("matter_id") is None for s in original_segments)
                                
                                if needs_reanalysis:
                                    logger.info(f"Re-analyzing chunk {chunk_id} from {current_date.date()} due to null matter segments.")
                                    transcript = chunk_data.get("processed_data", {}).get("word_level_transcript_with_absolute_times", [])
                                    if transcript:
                                        new_segments, _, _, _ = self.matter_analysis_service.analyze_chunk(transcript, all_matters_fresh, all_profiles, None)
                                        
                                        if new_segments != original_segments:
                                            update_chunk_metadata(current_date, chunk_id, {"matter_segments": new_segments})
                                            ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE.add(current_date.date())
                                            logger.info(f"Updated matter segments for chunk {chunk_id} and queued day for master log regen.")
                            
                            current_date += timedelta(days=1)
                        logger.info("Re-analysis complete.")
                    except Exception as e:
                        logger.error(f"Error during REANALYZE_MATTERS_FOR_DATE_RANGE: {e}", exc_info=True)

                elif isinstance(dequeued_item, dict) and dequeued_item.get('type') == "UPDATE_MATTER_FOR_SPAN":
                    logger.info("WORKER: Entering handler for 'UPDATE_MATTER_FOR_SPAN'.")
                    payload = dequeued_item.get('payload', {})
                    task_id = payload.get('task_id')
                    try:
                        logger.debug("WORKER: Payload for UPDATE_MATTER_FOR_SPAN: %s", json.dumps(payload, indent=2))
                        target_date_str = payload['target_date_str']
                        chunk_id = payload['chunk_id']
                        start_time = payload['start_time']
                        end_time = payload['end_time']
                        new_matter_id = payload['new_matter_id']

                        logger.debug(f"WORKER: Received new_matter_id '{new_matter_id}' from GUI command.")
                        if new_matter_id == "m_unassigned":
                            new_matter_id = None
                            logger.info("WORKER: Converted special matter_id 'm_unassigned' to None for backend processing.")
                        
                        target_date = datetime.strptime(target_date_str, '%Y%m%d').replace(tzinfo=timezone.utc)
                        
                        success = update_matter_segments_for_chunk(
                            target_date=target_date,
                            chunk_id=chunk_id,
                            start_time=start_time,
                            end_time=end_time,
                            new_matter_id=new_matter_id
                        )
                        if success:
                            logger.info(f"Successfully updated matter for span in chunk {chunk_id}.")
                        else:
                            logger.error(f"Failed to update matter for span in chunk {chunk_id}.")

                        if task_id:
                            with GUI_TASK_STATUS_LOCK:
                                GUI_TASK_STATUS[task_id] = "complete" if success else "error"
                            logger.info(f"WORKER: Marked UPDATE_MATTER_FOR_SPAN task {task_id} as {'complete' if success else 'error'}.")

                    except Exception as e:
                        logger.error(f"Error handling UPDATE_MATTER_FOR_SPAN: {e}", exc_info=True)
                        if task_id:
                            with GUI_TASK_STATUS_LOCK:
                                GUI_TASK_STATUS[task_id] = "error"

                elif isinstance(dequeued_item, dict) and dequeued_item.get('type') == "REACTIVATE_MATTER":
                    logger.info("WORKER: Processing REACTIVATE_MATTER command.")
                    try:
                        from src import matter_manager
                        payload = dequeued_item['payload']
                        matter_id = payload['matter_id']
                        updated_matter = matter_manager.update_matter(matter_id, {"status": "active"})
                        if updated_matter:
                            logger.info(f"Successfully reactivated matter ID: {matter_id}")
                        else:
                            logger.warning(f"Could not reactivate matter ID: {matter_id}. It might not exist.")
                    except Exception as e:
                        logger.error(f"Error handling REACTIVATE_MATTER: {e}", exc_info=True)
                
                elif isinstance(dequeued_item, dict) and dequeued_item.get('type') == "CREATE_MATTER_AND_FLAG":
                    logger.info("WORKER: Processing CREATE_MATTER_AND_FLAG command.")
                    try:
                        from src import matter_manager
                        payload = dequeued_item['payload']
                        matter_data = payload['matter_data']
                        flag_data = payload['flag_data']
                        
                        new_matter = matter_manager.add_matter(matter_data)
                        if not new_matter:
                            logger.error("Failed to create new matter from command. Aborting.")
                        else:
                            new_matter_id = new_matter['matter_id']
                            logger.info(f"Auto-created new matter '{matter_data.get('name')}' with ID {new_matter_id}.")
                            
                            flag_data['matter_id'] = new_matter_id
                            flag_data['reason_for_flag'] = f"New Matter Auto-Detected: {matter_data.get('name')}"
                            
                            flag_file_date_obj = datetime.now(timezone.utc)
                            flag_id = f"FLAG_{flag_file_date_obj.strftime('%Y%m%d')}_{uuid.uuid4().hex[:12]}"
                            
                            flag_to_create = {
                                "flag_id": flag_id,
                                "timestamp_logged_utc": datetime.now(timezone.utc).isoformat(),
                                "status": "pending_review",
                            }
                            flag_to_create.update(flag_data)
                            
                            daily_flags_list = load_daily_flags_queue(flag_file_date_obj)
                            daily_flags_list.append(flag_to_create)
                            save_daily_flags_queue(daily_flags_list, flag_file_date_obj)
                            logger.info(f"WORKER: Saved new matter flag to queue for {flag_file_date_obj.strftime('%Y-%m-%d')}.")
                            
                            recipient = self.config.get('signal', {}).get('recipient_phone_number')
                            if recipient:
                                message = (
                                    f"Samson has auto-detected a potential new project/matter:\n"
                                    f"Name: {matter_data.get('name')}\n"
                                    f"Summary: {matter_data.get('description', 'N/A')}\n\n"
                                    f"A flag has been created for your review. Please check the GUI or use 'review flags'."
                                )
                                send_message(recipient, message, self.config)
                                logger.info("Sent Signal notification for auto-created matter.")
                    except Exception as e:
                        logger.error(f"Error handling CREATE_MATTER_AND_FLAG: {e}", exc_info=True)

                elif isinstance(dequeued_item, dict) and dequeued_item.get('type') == 'REGENERATE_MASTER_LOG':
                    logger.info("WORKER: Processing REGENERATE_MASTER_LOG command from queue.")
                    payload = dequeued_item.get('payload', {})
                    date_str_to_regen = payload.get('date_str')
                    if not date_str_to_regen:
                        logger.error("WORKER: Invalid REGENERATE_MASTER_LOG payload, 'date_str' missing.")
                    else:
                        try:
                            date_obj_to_regen = datetime.strptime(date_str_to_regen, '%Y-%m-%d').date()
                            regenerate_master_log_for_day(date_obj_to_regen, self.config)
                        except Exception as e:
                            logger.error(f"WORKER: Failed to execute master log regeneration for {date_str_to_regen}: {e}", exc_info=True)

                elif isinstance(dequeued_item, dict) and dequeued_item.get('type') == 'RESOLVE_FLAG':
                    logger.info("[Worker-RESOLVE] Entered RESOLVE_FLAG handler.")
                    logger.info("WORKER: Processing RESOLVE_FLAG command from queue.")
                    payload = dequeued_item.get('payload', {})
                    flag_id = payload.get('flag_id')
                    flag_data = payload.get('flag_data')
                    resolution = payload.get('resolution')

                    logger.info(f"[Worker-RESOLVE] Processing flag '{flag_id}' with resolution action: {resolution.get('action')}")

                    if not all([flag_id, flag_data, resolution]):
                        logger.error(f"WORKER: Invalid RESOLVE_FLAG payload. Skipping. Payload: {payload}")
                    else:
                        with self.db_lock:
                            try:
                                action = resolution.get("action")
                                flag_date = parse_flag_id_for_date(flag_id) # Should always succeed as it was checked before queuing

                                
                                correction_context = resolution.get("context", "unknown") # Context from GUI/Signal resolution
                                # chunk_id is added during flag creation in the pipeline (see below in success handler)
                                chunk_id_from_flag = flag_data.get("chunk_id", "unknown") 

                                if action == "skip":
                                    _update_flag_status_in_queue(flag_id, "skipped", flag_date)
                                    logger.info(f"WORKER: Flag '{flag_id}' resolution complete (skipped).")

                                elif action == "resolved":
                                    assigned_matter_id = resolution.get("assigned_matter_id")
                                    if not assigned_matter_id:
                                        logger.error(f"WORKER: Cannot resolve matter flag '{flag_id}': 'resolved' action requires an 'assigned_matter_id'.")
                                    else:
                                        try:
                                            update_success = update_matter_segments_for_chunk(
                                                target_date=flag_date,
                                                chunk_id=flag_data.get('chunk_id'),
                                                start_time=flag_data.get('start_time'),
                                                end_time=flag_data.get('end_time'),
                                                new_matter_id=assigned_matter_id
                                            )
                                            if update_success:
                                                ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE.add(flag_date.date())
                                                logger.info(f"WORKER: Successfully updated matter segments for flag {flag_id} resolution.")
                                            else:
                                                logger.error(f"WORKER: Failed to update matter segments in daily log for flag {flag_id}")
                                        except Exception as e_update:
                                            logger.error(f"WORKER: Exception while updating matter segments for flag {flag_id}: {e_update}", exc_info=True)
                                        # Find the matter name from the original flag data for a better status message
                                        matter_name = "Unknown Matter"
                                        conflicting_matters = flag_data.get("conflicting_matters", [])
                                        assigned_matter = next((m for m in conflicting_matters if m.get("matter_id") == assigned_matter_id), None)
                                        if assigned_matter:
                                            matter_name = assigned_matter.get("name", "Unknown Matter")

                                        new_status = f"resolved_as_{matter_name}"
                                        _update_flag_status_in_queue(flag_id, new_status, flag_date)
                                        logger.info(f"WORKER: Flag '{flag_id}' resolution complete ({new_status}).")

                                elif action == "assign":
                                    target_name = resolution.get("name")
                                    if not target_name:
                                        logger.error(f"WORKER: Cannot resolve flag '{flag_id}': 'assign' action requires a 'name'.")
                                    else:
                                        # This is the logic moved from the old global resolve_flag function
                                        embedding_to_use: Optional[np.ndarray] = None
                                        embedding_source = (
                                            flag_data.get('consolidated_embedding') or 
                                            flag_data.get('segment_embedding') or 
                                            flag_data.get('embedding')
                                        )
                                        if isinstance(embedding_source, list):
                                            embedding_to_use = np.array(embedding_source, dtype=np.float32)
                                        elif isinstance(embedding_source, np.ndarray):
                                            embedding_to_use = embedding_source

                                        if embedding_to_use is None or embedding_to_use.size == 0:
                                            logger.error(f"WORKER: Cannot resolve flag '{flag_id}': missing or invalid embedding for assignment.")
                                        else:

                                            enroll_data = [{
                                                "temp_id": f"enroll_from_flag_{flag_id}",
                                                "name_to_enroll": target_name,
                                                "embedding": embedding_to_use.squeeze(),
                                                "context": correction_context  # Pass the context from the resolution
                                            }]
                                            enrollment_success, new_index, new_map = aps_speaker_id.enroll_speakers_programmatic(
                                                enrollment_data_list=enroll_data,
                                                faiss_index_path=self.faiss_idx_path,
                                                speaker_map_path=self.speaker_map_path,
                                                embedding_dim=self.config.get('audio_suite_settings', {}).get('embedding_dim', 192),
                                                db_lock=self.db_lock
                                            )
                                            if not enrollment_success:
                                                logger.error(f"WORKER: Failed to update speaker database for '{target_name}' while resolving flag '{flag_id}'.")
                                            else:
                                                self.faiss_index, self.speaker_map = new_index, new_map
                                                final_faiss_id_for_feedback = next((fid for fid, speaker_info in self.speaker_map.items() if isinstance(speaker_info, dict) and speaker_info.get('name', '').lower() == target_name.lower()), None)
                                                
                                                # Step 1: Create profile shell for the target speaker
                                                if final_faiss_id_for_feedback is not None:
                                                    create_speaker_profile(faiss_id=final_faiss_id_for_feedback, name=target_name)
                                                    logger.info(f"WORKER: Created/updated speaker profile for '{target_name}' with FAISS ID {final_faiss_id_for_feedback} from flag resolution.")
                                                
                                                # Step 2: Before relabeling, get the accurate duration of the dialogue being moved
                                                duration_to_add = 0.0
                                                original_speaker_to_replace = flag_data.get('cusid') or flag_data.get('tentative_speaker_name')
                                                if original_speaker_to_replace and flag_date:
                                                    # Perform a robust, word-level audit to get the true duration of the CUSID's speech.
                                                    # This avoids the error from get_all_dialogue_for_speaker_id, which only checks segment-level speaker tags.
                                                    daily_log_data = get_daily_log_data(flag_date)
                                                    duration_to_add = 0.0
                                                    for chunk in daily_log_data.get('chunks', {}).values():
                                                        wlt = chunk.get('processed_data', {}).get('word_level_transcript_with_absolute_times', [])
                                                        for segment in wlt:
                                                            for word in segment.get('words', []):
                                                                if word.get('speaker') == original_speaker_to_replace:
                                                                    try:
                                                                        # Add duration only if start and end are valid floats
                                                                        word_duration = float(word.get('end', 0.0)) - float(word.get('start', 0.0))
                                                                        if word_duration > 0:
                                                                            duration_to_add += word_duration
                                                                    except (TypeError, ValueError):
                                                                        continue # Skip words with malformed timestamps
                                                    
                                                    # Now, perform the relabeling in the log
                                                    relabel_speaker_id_in_daily_log(flag_date, original_speaker_to_replace, target_name)

                                                # Step 3: Add the evolution segment from the flag and update lifetime audio with the correct delta
                                                duration_s_for_evol_segment = flag_data.get('segment_duration_s')
                                                if duration_s_for_evol_segment is None:
                                                    start_s = flag_data.get('start_time')
                                                    end_s = flag_data.get('end_time')
                                                    if start_s is not None and end_s is not None:
                                                        try:
                                                            duration_s_for_evol_segment = float(end_s) - float(start_s)
                                                        except (ValueError, TypeError): duration_s_for_evol_segment = None

                                                if final_faiss_id_for_feedback is not None and isinstance(final_faiss_id_for_feedback, int):
                                                    if duration_s_for_evol_segment is not None:
                                                        add_segment_embedding_for_evolution(
                                                            faiss_id=final_faiss_id_for_feedback, embedding=embedding_to_use, duration_s=duration_s_for_evol_segment,
                                                            confidence_score=1.0, context=correction_context,
                                                            processing_timestamp=datetime.now(timezone.utc), chunk_id=chunk_id_from_flag
                                                        )
                                                    
                                                    # This logic is now self-contained and correctly handles the delta.
                                                    # The `duration_to_add` is the total duration of the dialogue being moved from the CUSID.
                                                    # We treat this as a delta to be added to the speaker's existing lifetime audio.
                                                    if duration_to_add > 0:
                                                        profile_to_update = get_speaker_profile(final_faiss_id_for_feedback)
                                                        if profile_to_update:
                                                            current_total_s = profile_to_update.get('lifetime_total_audio_s', 0.0)
                                                            # The operation is explicitly an addition of the CUSID's dialogue duration.
                                                            new_total_s = current_total_s + duration_to_add
                                                            update_speaker_profile(faiss_id=final_faiss_id_for_feedback, lifetime_total_audio_s=new_total_s)
                                                            logger.info(f"WORKER: FLAG - Updated lifetime audio for '{target_name}' from {current_total_s:.2f}s to {new_total_s:.2f}s.")

                                                else:
                                                    logger.error(f"WORKER: Could not find FAISS ID for '{target_name}' after enrollment. Profile will not be created and feedback logging will be incomplete.")
                                                    final_faiss_id_for_feedback = -1

                                                # <<< MODIFIED: Update feedback details with context and chunk_id >>>
                                                feedback_details = {
                                                    "faiss_id_of_correct_speaker": final_faiss_id_for_feedback,
                                                    "original_speaker_id": flag_data.get("tentative_speaker_name", "N/A"),
                                                    "corrected_speaker_id": target_name,
                                                    "source": resolution.get("source", "unknown"),
                                                    "audio_context": correction_context,
                                                    "chunk_id": chunk_id_from_flag,
                                                    "duration_s": duration_to_add, # Use the correct total duration for the feedback log
                                                    "flag_details": { "flag_id": flag_id, "reason_for_flag": flag_data.get("reason_for_flag"), "flag_type": flag_data.get('flag_type')}
                                                    
                                                }
                                                log_correction_feedback("flag_resolution", feedback_details)
                                            
                                                _update_flag_status_in_queue(flag_id, f"resolved_as_{target_name}", flag_date, final_assigned_name=target_name)

                                                if flag_date:
                                                    ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE.add(flag_date.date())
                                                    logger.info(f"WORKER: Added {flag_date.date().strftime('%Y-%m-%d')} to master log regeneration queue after flag resolution.")
                                                
                                                
                                else:
                                    logger.error(f"WORKER: Unknown resolution action '{action}' for flag '{flag_id}'.")
                                # At the end of a successful resolution (e.g., after _update_flag_status_in_queue)
                                try:
                                    if flag_date:
                                        config = get_config()
                                        snippet_path_to_delete = config['paths']['flag_snippets_dir'] / flag_date.strftime('%Y-%m-%d') / f"{flag_id}.wav"
                                        if snippet_path_to_delete.exists():
                                            snippet_path_to_delete.unlink()
                                            logger.info(f"WORKER: Cleaned up resolved snippet file: {snippet_path_to_delete.name}")
                                except Exception as e_cleanup:
                                    logger.warning(f"WORKER: Could not clean up snippet for resolved flag {flag_id}: {e_cleanup}")
                            except Exception as e:
                                logger.error(f"WORKER: Unhandled error during flag resolution for '{flag_id}': {e}", exc_info=True)
                       
                elif isinstance(dequeued_item, dict) and dequeued_item.get('type') == 'BATCH_CORRECT_SPEAKER_ASSIGNMENTS':
                    logger.info(f"WORKER: Processing BATCH Correction Command.")
                    payload = dequeued_item.get('payload', {})
                    
                    # This entire block is now under the single DB lock for atomicity
                    with self.db_lock:
                        # In AudioProcessingWorker.run(), replace the entire try...except block inside the BATCH_CORRECT_SPEAKER_ASSIGNMENTS handler
                        try:
                            # --- Step 1: Extract and Validate Batch Data ---
                            corrections_list = payload.get("corrections", [])
                            new_speaker_name = payload.get("new_speaker_name")
                            target_date_str = payload.get("target_date_str")
                            correction_source = payload.get("source", "gui_batch")
                            correction_context = payload.get("context", "unknown")
                            chunk_id_from_batch = payload.get("chunk_id", "unknown")
                            
                            original_speakers_to_clean = set()
                        
                            if not all([corrections_list, new_speaker_name, target_date_str, chunk_id_from_batch != "unknown"]):
                                logger.error(f"WORKER: Invalid or incomplete BATCH command payload: {payload}. Discarding.")
                            else:
                                target_date_utc_dt = datetime.strptime(target_date_str, '%Y%m%d').replace(tzinfo=timezone.utc)
                            
                                # --- Step 2: Load Log Data ONCE ---
                                log_data = get_daily_log_data(target_date_utc_dt)
                                if not log_data or chunk_id_from_batch not in log_data.get('chunks', {}):
                                    logger.error(f"WORKER: BATCH - Could not find chunk_id '{chunk_id_from_batch}' in log data for {target_date_str}. Aborting batch.")
                                else:
                                    target_entry = log_data['chunks'][chunk_id_from_batch]
                                    wlt = target_entry.get('processed_data', {}).get('word_level_transcript_with_absolute_times', [])
                                    
                                    # --- Step 3: Relabel Words in Memory ---
                                    words_relabelled_count = 0
                                    for correction in corrections_list:
                                        try:
                                            seg_idx, word_idx = int(correction["segment_index"]), int(correction["word_index"])
                                            original_speaker = correction["word_data"].get("speaker")
                                            if original_speaker: original_speakers_to_clean.add(original_speaker)
                                            wlt[seg_idx]['words'][word_idx]['speaker'] = new_speaker_name
                                            words_relabelled_count += 1
                                        except (IndexError, KeyError, TypeError, ValueError) as e_inner:
                                            logger.error(f"WORKER: BATCH - Skipping a correction due to invalid index or data: {correction}. Error: {e_inner}", exc_info=True)
                                
                                    if words_relabelled_count == 0:
                                        logger.warning("WORKER: BATCH - No words were successfully relabelled. Aborting.")
                                    else:
                                        # --- Step 4: Recalculate dominant speakers and save log ---
                                        for segment in wlt:
                                            if any(word.get('speaker') == new_speaker_name for word in segment.get('words', [])):
                                                speaker_counts = Counter(w.get('speaker', 'UNKNOWN_SPEAKER') for w in segment.get('words', []))
                                                if speaker_counts:
                                                    segment['speaker'] = speaker_counts.most_common(1)[0][0]
                                
                                        save_daily_log_data(target_date_utc_dt, log_data)
                                        logger.info(f"WORKER: BATCH - Successfully relabelled {words_relabelled_count} words and saved daily log.")
                                
                                        # --- Step 5: Update Evolution Data & Lifetime Stats ---
                                        self._enroll_and_populate_new_speaker(
                                            new_speaker_name=new_speaker_name,
                                            corrections_list=corrections_list,
                                            correction_context=correction_context,
                                            chunk_id=chunk_id_from_batch,
                                            target_date_utc_dt=target_date_utc_dt
                                        )
                                        
                                        # --- START: Corrected Lifetime Audio Statistics Update ---
                                        # Calculate duration deltas based on the specific words moved.
                                        duration_delta_by_speaker = Counter()
                                        for correction in corrections_list:
                                            word_data = correction.get("word_data", {})
                                            original_speaker = word_data.get("speaker")
                                            try:
                                                duration = float(word_data.get('end', 0.0)) - float(word_data.get('start', 0.0))
                                                if original_speaker and duration > 0:
                                                    duration_delta_by_speaker[original_speaker] -= duration
                                                    duration_delta_by_speaker[new_speaker_name] += duration
                                            except (TypeError, ValueError):
                                                logger.warning(f"Could not calculate duration for word, skipping stats update: {word_data}")
                                        
                                        logger.info(f"WORKER: BATCH - Calculated lifetime audio deltas: {dict(duration_delta_by_speaker)}")

                                        # For original speakers, the evolution segment from this chunk is now tainted and must be removed.
                                        for speaker_name in original_speakers_to_clean:
                                            if speaker_name != new_speaker_name and not speaker_name.startswith("CUSID_"):
                                                profile = next((p for p in get_all_speaker_profiles() if p.get('name') == speaker_name), None)
                                                if profile:
                                                    # Step A: Always remove the old segment as it's tainted.
                                                    update_or_remove_evolution_segment(profile['faiss_id'], correction_context, chunk_id_from_batch)
                                                    logger.info(f"WORKER: BATCH - Cleared tainted evolution segment for '{speaker_name}' from chunk '{chunk_id_from_batch}'.")

                                                    # Step B: Find all remaining words for this speaker in this chunk to rebuild the segment.
                                                    remaining_words = []
                                                    original_file_name_for_rebuild = target_entry.get("original_file_name")
                                                    if original_file_name_for_rebuild:
                                                        for seg in wlt: # wlt is the modified transcript in memory
                                                            for word in seg.get('words', []):
                                                                if word.get('speaker') == speaker_name:
                                                                    remaining_words.append(word)

                                                    if remaining_words:
                                                        logger.info(f"WORKER: BATCH - Found {len(remaining_words)} words remaining for '{speaker_name}'. Rebuilding evolution segment.")
                                                        audio_tensors, files_cache = [], {}
                                                        for word in remaining_words:
                                                            try:
                                                                start_time, end_time = float(word["start"]), float(word["end"])
                                                                processed = self._get_audio_and_embedding_for_correction(original_file_name_for_rebuild, target_date_utc_dt, start_time, end_time, files_cache)
                                                                if processed: audio_tensors.append(processed[0])
                                                            except (KeyError, ValueError): continue
                                                        
                                                        if audio_tensors:
                                                            concatenated_audio = torch.cat(audio_tensors, dim=1)
                                                            new_duration_s = float(concatenated_audio.shape[-1]) / 16000.0
                                                            embedding_model = _loaded_models_cache.get("embedding_model")
                                                            with torch.no_grad():
                                                                device = self.config.get('global_ml_device', 'cpu')
                                                                raw_embedding = embedding_model.encode_batch(concatenated_audio.to(device))
                                                            new_embedding_np = raw_embedding.mean(dim=1).squeeze(0).cpu().numpy() if raw_embedding.dim() == 3 else raw_embedding.squeeze(0).cpu().numpy()
                                                            
                                                            update_or_remove_evolution_segment(
                                                                faiss_id=profile['faiss_id'], context=correction_context, chunk_id=chunk_id_from_batch,
                                                                new_embedding=new_embedding_np, new_duration_s=new_duration_s
                                                            )
                                                            logger.info(f"WORKER: BATCH - Rebuilt evolution segment for '{speaker_name}' with duration {new_duration_s:.2f}s.")
                                        # Apply the accurately calculated duration changes to the lifetime stats for all affected speakers.
                                        for speaker_name, delta_s in duration_delta_by_speaker.items():
                                            if delta_s == 0: continue
                                            
                                            # We need to look up the profile again as the new speaker might have just been created.
                                            profile = next((p for p in get_all_speaker_profiles() if p.get('name') == speaker_name), None)
                                            if profile:
                                                current_total_s = profile.get('lifetime_total_audio_s', 0.0)
                                                new_total_s = max(0.0, current_total_s + delta_s)
                                                update_speaker_profile(faiss_id=profile['faiss_id'], lifetime_total_audio_s=new_total_s)
                                                logger.info(f"WORKER: BATCH - Adjusted lifetime audio for '{speaker_name}' by {delta_s:+.2f}s. New total: {new_total_s:.2f}s")
                                            else:
                                                logger.warning(f"WORKER: BATCH - Could not find profile for '{speaker_name}' to update lifetime audio stats.")
                                        # --- END: Corrected Lifetime Audio Statistics Update ---
                                
                                        # --- Step 6: Finalize ---
                                        ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE.add(target_date_utc_dt.date())
                                        logger.info(f"WORKER: Added {target_date_str} to master log regeneration queue.")
                                        
                                        task_id = payload.get('task_id')
                                        if task_id:
                                            with GUI_TASK_STATUS_LOCK: GUI_TASK_STATUS[task_id] = "complete"
                                            logger.info(f"WORKER: Marked BATCH task {task_id} as complete.")
                        
                        except Exception as e:
                            logger.error(f"WORKER: Unhandled error processing BATCH command: {e}", exc_info=True)
                            task_id = payload.get('task_id')
                            if task_id:
                                with GUI_TASK_STATUS_LOCK: GUI_TASK_STATUS[task_id] = "error"

                elif isinstance(dequeued_item, dict) and dequeued_item.get('type') == 'CORRECT_TEXT_AND_SPEAKER':
                    logger.info("WORKER: Processing CORRECT_TEXT_AND_SPEAKER command.")
                    payload = dequeued_item.get('payload', {})
                    task_id_for_text_corr = payload.get('task_id') # Get task_id early for error handling
                    try:
                        # Call the new function in daily_log_manager
                        success, original_words, new_words = modify_word_span_and_speaker(
                            corrections=payload['corrections'],
                            new_speaker_name=payload.get('new_speaker_name'),
                            new_text=payload['new_text'],
                            correction_context=payload['context'],
                            correction_source=payload['source'],
                            target_date_str=payload['target_date_str']
                        )
                        if not success:
                            logger.error(f"Failed to execute text & speaker correction for chunk {payload.get('chunk_id')}.")
                            if task_id_for_text_corr:
                                with GUI_TASK_STATUS_LOCK: GUI_TASK_STATUS[task_id_for_text_corr] = "error"
                        else:

                            new_speaker_name = payload.get("new_speaker_name")
                            if new_speaker_name:
                                # This call is now unconditional to leverage the idempotent nature of enroll_speakers_programmatic.
                                # It will create a new speaker or update an existing one's embedding by averaging.
                                logger.info(f"WORKER: Enrolling/updating speaker '{new_speaker_name}' from text correction.")
                                target_date_utc_dt = datetime.strptime(payload['target_date_str'], '%Y%m%d').replace(tzinfo=timezone.utc)
                                self._enroll_and_populate_new_speaker(
                                    new_speaker_name=new_speaker_name,
                                    corrections_list=payload['corrections'],
                                    correction_context=payload['context'],
                                    chunk_id=payload['chunk_id'],
                                    target_date_utc_dt=target_date_utc_dt
                                )
                            # --- END: Unconditional helper call ---

                            # Refresh the original speaker's embedding and evolution data, mirroring the BATCH handler's logic.
                            if original_words:
                                original_speakers_to_clean = {word.get('speaker') for word in original_words if word.get('speaker')}
                                chunk_id_from_payload = payload.get("chunk_id")
                                target_date_utc_dt_for_refresh = datetime.strptime(payload['target_date_str'], '%Y%m%d').replace(tzinfo=timezone.utc)
                                
                                # Reload the log data to get the now-modified transcript
                                log_data_after_modify = get_daily_log_data(target_date_utc_dt_for_refresh)
                                target_entry_after_modify = log_data_after_modify['chunks'][chunk_id_from_payload]
                                wlt_after_modify = target_entry_after_modify.get('processed_data', {}).get('word_level_transcript_with_absolute_times', [])
                                
                                for speaker_name_to_refresh in original_speakers_to_clean:
                                    if speaker_name_to_refresh == new_speaker_name or speaker_name_to_refresh.startswith("CUSID_"):
                                        continue

                                    # Remove tainted evolution segment for the original speaker
                                    profile = next((p for p in get_all_speaker_profiles() if p.get('name') == speaker_name_to_refresh), None)
                                    if profile:
                                        update_or_remove_evolution_segment(profile['faiss_id'], payload['context'], chunk_id_from_payload)
                                        logger.info(f"WORKER: TEXT_CORRECT - Cleared tainted evolution segment for '{speaker_name_to_refresh}' from chunk '{chunk_id_from_payload}'.")

                                        # Rebuild evolution segment from remaining words
                                        remaining_words_for_rebuild = []
                                        original_file_for_rebuild = target_entry_after_modify.get("original_file_name")
                                        if original_file_for_rebuild and wlt_after_modify:
                                            for seg in wlt_after_modify:
                                                for word in seg.get('words', []):
                                                    if word.get('speaker') == speaker_name_to_refresh:
                                                        remaining_words_for_rebuild.append(word)

                                        if remaining_words_for_rebuild:
                                            logger.info(f"WORKER: TEXT_CORRECT - Found {len(remaining_words_for_rebuild)} words remaining for '{speaker_name_to_refresh}'. Rebuilding evolution segment.")
                                            audio_tensors, files_cache = [], {}
                                            for word in remaining_words_for_rebuild:
                                                try:
                                                    start_time, end_time = float(word["start"]), float(word["end"])
                                                    processed = self._get_audio_and_embedding_for_correction(original_file_for_rebuild, target_date_utc_dt_for_refresh, start_time, end_time, files_cache)
                                                    if processed: audio_tensors.append(processed[0])
                                                except (KeyError, ValueError): continue

                                            if audio_tensors:
                                                concatenated_audio = torch.cat(audio_tensors, dim=1)
                                                new_duration_s = float(concatenated_audio.shape[-1]) / 16000.0
                                                embedding_model = _loaded_models_cache.get("embedding_model")
                                                with torch.no_grad():
                                                    device = self.config.get('global_ml_device', 'cpu')
                                                    raw_embedding = embedding_model.encode_batch(concatenated_audio.to(device))
                                                new_embedding_np = raw_embedding.mean(dim=1).squeeze(0).cpu().numpy() if raw_embedding.dim() == 3 else raw_embedding.squeeze(0).cpu().numpy()

                                                update_or_remove_evolution_segment(
                                                    faiss_id=profile['faiss_id'], context=payload['context'], chunk_id=chunk_id_from_payload,
                                                    new_embedding=new_embedding_np, new_duration_s=new_duration_s
                                                )
                                                logger.info(f"WORKER: TEXT_CORRECT - Rebuilt evolution segment for '{speaker_name_to_refresh}' with duration {new_duration_s:.2f}s.")

                            # --- START: Immediate Lifetime Audio Update (for Text Correction) ---
                            if original_words is not None and new_words is not None:
                                try:
                                    duration_delta = Counter()

                                    # Subtract duration of old words from their original speakers
                                    for word in original_words:
                                        original_speaker = word.get('speaker')
                                        duration = float(word.get('end', 0.0)) - float(word.get('start', 0.0))
                                        if duration > 0 and original_speaker:
                                            duration_delta[original_speaker] -= duration
                                    
                                    # Add duration of new words to the final speaker
                                    final_speaker_for_delta = new_words[0]['speaker'] if new_words and 'speaker' in new_words[0] else None
                                    if final_speaker_for_delta:
                                        for word in new_words:
                                            duration = float(word.get('end', 0.0)) - float(word.get('start', 0.0))
                                            if duration > 0:
                                                duration_delta[final_speaker_for_delta] += duration

                                    logger.info(f"WORKER: TEXT_CORRECT - Recalculating lifetime audio. Deltas: {dict(duration_delta)}")

                                    # Atomically update profiles for all affected speakers
                                    for speaker_name, delta_s in duration_delta.items():
                                        if delta_s == 0: continue
                                        
                                        if speaker_name.startswith("CUSID_"):
                                            logger.debug(f"WORKER: TEXT_CORRECT - Skipping lifetime audio update for ephemeral ID '{speaker_name}'.")
                                            continue
                                        
                                        profile_faiss_id = None
                                        if self.speaker_map:
                                            for fid, s_info in self.speaker_map.items():
                                                 if isinstance(s_info, dict) and s_info.get('name') == speaker_name:
                                                    profile_faiss_id = fid
                                                    break
                                        
                                        if profile_faiss_id is None:
                                            logger.warning(f"WORKER: TEXT_CORRECT - Could not find FAISS ID for '{speaker_name}' to update lifetime audio. Skipping.")
                                            continue
                                    
                                        current_profile = get_speaker_profile(profile_faiss_id)
                                        if current_profile:
                                            current_total_s = current_profile.get('lifetime_total_audio_s', 0.0)
                                            new_total_s = max(0.0, current_total_s + delta_s)
                                            update_speaker_profile(faiss_id=profile_faiss_id, lifetime_total_audio_s=new_total_s)
                                            logger.info(f"WORKER: TEXT_CORRECT - Updated lifetime audio for '{speaker_name}' from {current_total_s:.2f}s to {new_total_s:.2f}s.")
                                        else:
                                            logger.warning(f"WORKER: TEXT_CORRECT - Could not find profile for FAISS ID {profile_faiss_id} ('{speaker_name}') to update lifetime audio.")
                                
                                except Exception as e_stat:
                                    logger.error(f"WORKER: TEXT_CORRECT - Failed to perform immediate lifetime audio update: {e_stat}", exc_info=True)

                            if task_id_for_text_corr:
                                with GUI_TASK_STATUS_LOCK:
                                    GUI_TASK_STATUS[task_id_for_text_corr] = "complete"
                                logger.info(f"WORKER: Marked TEXT_AND_SPEAKER task {task_id_for_text_corr} as complete.")

                    except KeyError as e:
                        logger.error(f"WORKER: Invalid payload for text correction: missing key {e}.")
                        if task_id_for_text_corr:
                            with GUI_TASK_STATUS_LOCK: GUI_TASK_STATUS[task_id_for_text_corr] = "error"
                    except Exception as e_text_corr_main:
                        logger.error(f"WORKER: Unhandled error in CORRECT_TEXT_AND_SPEAKER: {e_text_corr_main}", exc_info=True)
                        if task_id_for_text_corr:
                            with GUI_TASK_STATUS_LOCK: GUI_TASK_STATUS[task_id_for_text_corr] = "error"

                # This handler is now redundant and has been removed.
                
                elif isinstance(dequeued_item, dict) and dequeued_item.get('type') == 'RECALCULATE_SPEAKER_PROFILES':
                    logger.info("WORKER: Processing RECALCULATE_SPEAKER_PROFILES command.")
                    with self.db_lock:
                        # In AudioProcessingWorker.run(), replace the RECALCULATE_SPEAKER_PROFILES handler's try block
                        try:
                            # --- Step 1: Initialization ---
                            logger.info("WORKER: RECALCULATE - Starting full speaker database re-audit and rebuild.")
                            if not self.faiss_idx_path or not self.faiss_idx_path.exists():
                                logger.error("WORKER: RECALCULATE - Cannot run, FAISS index file does not exist.")
                            else:
                                archive_folder = self.config.get('paths', {}).get('archived_audio_folder')
                                if not archive_folder or not Path(archive_folder).exists():
                                    logger.error("WORKER: RECALCULATE - Cannot run, 'archived_audio_folder' not configured or does not exist.")
                                else:
                                    old_faiss_index = faiss.read_index(str(self.faiss_idx_path))
                                    old_speaker_map = load_or_create_speaker_map(self.speaker_map_path)
                                    embedding_model = _loaded_models_cache.get("embedding_model")
                                
                                    if not embedding_model:
                                        logger.error("WORKER: RECALCULATE - Cannot run, embedding model is not loaded.")
                                    else:
                                        # --- Step 2: Perform a single, comprehensive word-level audit for all speakers ---
                                        dialogue_start_date = datetime(2020, 1, 1).date()
                                        dialogue_end_date = datetime.now(timezone.utc).date()
                                        
                                        all_historical_segments_by_speaker = self._audit_and_group_speaker_diarization(
                                            start_date=dialogue_start_date, end_date=dialogue_end_date
                                        )
                                        new_embeddings_for_db = []
                                        new_map_data_for_db = []
                                    
                                        for faiss_id, speaker_info in old_speaker_map.items():
                                            if not isinstance(speaker_info, dict) or 'name' not in speaker_info:
                                                logger.warning(f"WORKER: RECALCULATE - Skipping malformed entry in speaker map with ID {faiss_id}.")
                                                continue
                                            
                                            speaker_name = speaker_info['name']
                                            logger.info(f"WORKER: RECALCULATE - Processing collected history for '{speaker_name}' (ID: {faiss_id}).")
                                            logger.debug(f"WORKER: RECALCULATE - DIAGNOSTICS FOR '{speaker_name}'")
                                            
                                            # Use the correctly audited segments from the comprehensive audit
                                            historical_segments = all_historical_segments_by_speaker.get(speaker_name, [])
    
    
                                        
                                            if not historical_segments:
                                                logger.warning(f"No dialogue found for '{speaker_name}' after full word-level audit. They will be removed from the active search index but their profile will be preserved.")
                                                update_speaker_profile(int(faiss_id), segment_embeddings_for_evolution={})
                                                continue
    
                                            # --- 2a. Calculate true lifetime audio and extract audio snippets ---
                                            new_lifetime_total_s = 0.0
                                            
                                            # --- REFACTOR START: Combine loops to prevent index mismatch ---
                                            segment_embeddings_and_weights = []
                                            loaded_audio_waveforms = {}
                                            expected_dim = old_faiss_index.d
                                            decay_rate = self.config.get('speaker_intelligence', {}).get('recalculation_recency_decay_rate', 0.0)
    
                                            # --- NEW: Calculate "Speaking Day Age" based on actual utterance timestamps ---
                                            all_speaking_dates = []
                                            if decay_rate > 0: # Only do this if decay is enabled
                                                all_speaking_dates = sorted(list(set(
                                                    datetime.fromisoformat(seg['absolute_start_utc']).date()
                                                    for seg in historical_segments if 'absolute_start_utc' in seg and seg.get('absolute_start_utc')
                                                )))
                                            
                                            total_speaking_days = len(all_speaking_dates)
                                            # Create a map of date -> age, where the most recent day has age 0.
                                            date_to_speaking_day_age_map = {
                                                d: total_speaking_days - 1 - i 
                                                for i, d in enumerate(all_speaking_dates)
                                            } if all_speaking_dates else {}
                                            
                                            for segment_data in historical_segments:
                                                try:
                                                    duration = segment_data.get('summed_word_duration_s', 0.0)
                                                    if duration <= 0: continue
                                                   
                                                    # This must be incremented for every segment, not just the first one from a file.
                                                    # new_lifetime_total_s += duration
    
                                                    file_stem = segment_data['audio_file_stem']
                                                    if file_stem not in loaded_audio_waveforms:
                                                        processing_date_str = segment_data.get('processing_date_of_log')
                                                        if not processing_date_str: continue
                                                        
                                                        audio_file_path = next((Path(archive_folder) / processing_date_str).glob(f"{file_stem}.*"), None)
                                                        if not audio_file_path or not audio_file_path.exists(): continue
                                                        
                                                        waveform, sample_rate = aps_audio.load_audio(audio_file_path, target_sr=16000)
                                                        if waveform is None: 
                                                            loaded_audio_waveforms[file_stem] = (None, None) # Cache failure
                                                            continue
                                                        loaded_audio_waveforms[file_stem] = (waveform, sample_rate)
                                                    
                                                    waveform, sample_rate = loaded_audio_waveforms[file_stem]
                                                    if waveform is None: continue # Skip if failed on previous attempt
                                                                                            
                                                    start_s, end_s = segment_data.get('start_s'), segment_data.get('end_s')
                                                    if start_s is None or end_s is None: continue
    
                                                    start_sample, end_sample = int(start_s * sample_rate), int(end_s * sample_rate)
                                                    audio_snippet = waveform[:, start_sample:end_sample]
    
                                                    # Generate embedding for this snippet
                                                    with torch.no_grad():
                                                        device = self.config.get('global_ml_device', 'cpu')
                                                        raw_embedding = embedding_model.encode_batch(audio_snippet.to(device))
                                                    if not (raw_embedding is not None and raw_embedding.numel() > 0): continue
                                                    
                                                    if raw_embedding.dim() == 3: segment_emb_np = raw_embedding.mean(dim=1).squeeze(0).cpu().numpy()
                                                    else: segment_emb_np = raw_embedding.squeeze(0).cpu().numpy()
                                                    if segment_emb_np.shape[0] != expected_dim: continue
                                                    faiss.normalize_L2(segment_emb_np.reshape(1, -1))
    
                                                    # Calculate recency weight using speaking day age
                                                    recency_weight = 1.0
                                                    speaking_day_age_for_log = 0.0
                                                    segment_timestamp_str = segment_data.get('absolute_start_utc')
                                                    if segment_timestamp_str and decay_rate > 0 and date_to_speaking_day_age_map:
                                                        try:
                                                            segment_date = datetime.fromisoformat(segment_timestamp_str).date()
                                                            speaking_day_age = date_to_speaking_day_age_map.get(segment_date, 0)
                                                            recency_weight = np.exp(-decay_rate * speaking_day_age)
                                                            speaking_day_age_for_log = speaking_day_age
                                                        except (ValueError, TypeError): pass
                                                    
                                                    # The final weight is a product of duration and recency
                                                    final_weight = duration * recency_weight
                                                    logger.debug(f"  - Segment for '{speaker_name}': start_utc='{segment_timestamp_str}', "
                                                        f"speaking_day_age={speaking_day_age_for_log:.2f}, recency_weight={recency_weight:.6f}, "
                                                        f"duration={duration:.4f}, final_weight={final_weight:.6f}")
                                                    segment_embeddings_and_weights.append({'embedding': segment_emb_np, 'weight': final_weight, 'duration': duration})
                                                
                                                except Exception as e_seg:
                                                    logger.error(f"WORKER: RECALCULATE - Error processing a segment for '{speaker_name}': {e_seg}", exc_info=True)
    
                                            # --- 2b. Generate new master embedding with recency weighting ---
                                            new_master_embedding = None
                                            if not segment_embeddings_and_weights:
                                                logger.warning(f"WORKER: RECALCULATE - No valid embeddings generated for '{speaker_name}'. Using old one.")
                                                new_master_embedding = old_faiss_index.reconstruct(int(faiss_id)).reshape(1, -1)
                                            else:
                                                total_weight = sum(item['weight'] for item in segment_embeddings_and_weights)
                                                weighted_sum_embedding = np.zeros(expected_dim, dtype=np.float32)
                                                for item in segment_embeddings_and_weights:
                                                    # item['weight'] is already duration * recency_weight
                                                    weighted_sum_embedding += item['embedding'] * item['weight']
                                                
                                                if total_weight > 1e-6:
                                                    final_embedding = weighted_sum_embedding / total_weight
                                                else: 
                                                    # Fallback to simple average of embeddings if total weight is zero
                                                    final_embedding = np.mean([item['embedding'] for item in segment_embeddings_and_weights], axis=0)
    
                                                new_master_embedding = final_embedding.astype(np.float32).reshape(1, -1)
                                                faiss.normalize_L2(new_master_embedding)
    
                                            logger.debug(f"WORKER: RECALCULATE - Final new embedding for '{speaker_name}': "
                                                f"norm={np.linalg.norm(new_master_embedding):.6f}, "
                                                f"preview={new_master_embedding[0, :5]}")
                                            
                                            # --- 2c. Store results and handle potential model failures ---
                                            expected_dim = old_faiss_index.d
    
                                            # --- Step 2d: Consolidate all profile updates and perform one atomic write ---
                                            profile_updates = {
                                                
                                                "segment_embeddings_for_evolution": {},  # Clear all pending evolution data
                                            }
                                            
                                            # Set the evolution timestamp for all relevant contexts
                                            current_profile_for_ts = get_speaker_profile(int(faiss_id))
                                            if current_profile_for_ts:
                                                now_iso = datetime.now(timezone.utc).isoformat()
                                                
                                                # Robustly get or create the timestamp dictionary
                                                new_evo_ts_dict = current_profile_for_ts.get('profile_last_evolved_utc')
                                                if not isinstance(new_evo_ts_dict, dict):
                                                    new_evo_ts_dict = {}
    
                                                # A full re-audit is an evolution event for ALL of a speaker's contexts.
                                                # Iterate over the keys of the existing timestamp dictionary to update them all.
                                                # This correctly reflects that the master embedding was rebuilt from all historical data.
                                                for context in list(new_evo_ts_dict.keys()):
                                                    new_evo_ts_dict[context] = now_iso
    
                                                # Also, capture any contexts that only had pending data but no prior evolution timestamp.
                                                if isinstance(current_profile_for_ts.get('segment_embeddings_for_evolution'), dict):
                                                    for context in current_profile_for_ts.get('segment_embeddings_for_evolution'):
                                                        new_evo_ts_dict[context] = now_iso # This will add new contexts if they weren't in the original dict.
    
                                                profile_updates['profile_last_evolved_utc'] = new_evo_ts_dict

                                                new_lifetime_total_s = sum(item['duration'] for item in segment_embeddings_and_weights)
                                                logger.info(f"WORKER: RECALCULATE - New lifetime total audio for '{speaker_name}': {new_lifetime_total_s:.2f}s")
    
                                                #Add the newly calculated lifetime audio duration to the updates.
                                                profile_updates['lifetime_total_audio_s'] = new_lifetime_total_s
    
                                            # Determine which embedding to use (new or fallback)
                                            if new_master_embedding is not None and new_master_embedding.shape == (1, expected_dim):
                                                new_embeddings_for_db.append(new_master_embedding.astype(np.float32))
                                            else:
                                                bad_shape = new_master_embedding.shape if hasattr(new_master_embedding, 'shape') else 'N/A'
                                                logger.critical(f"WORKER: RECALCULATE - Model produced a malformed embedding for '{speaker_name}' (shape: {bad_shape}). Using old embedding as fallback.")
                                                old_embedding = old_faiss_index.reconstruct(int(faiss_id)).reshape(1, -1)
                                                new_embeddings_for_db.append(old_embedding.astype(np.float32))
                                            
                                            # Add the speaker to the new map and commit all profile changes at once
                                            new_map_data_for_db.append(speaker_info)
                                            update_speaker_profile(int(faiss_id), **profile_updates)
                                    
                                        # --- Step 3: Atomically build and save the new database ---
                                        if new_embeddings_for_db:
                                            new_faiss_index = faiss.IndexFlatIP(old_faiss_index.d)
                                            new_faiss_index.add(np.vstack(new_embeddings_for_db))
                                            new_speaker_map = {i: data for i, data in enumerate(new_map_data_for_db)}
                                            
                                            save_faiss_index(new_faiss_index, self.faiss_idx_path)
                                            save_speaker_map(new_speaker_map, self.speaker_map_path)
                                            
                                            self.faiss_index = new_faiss_index
                                            self.speaker_map = new_speaker_map
                                            logger.info(f"WORKER: RECALCULATE - Speaker database rebuilt successfully with {new_faiss_index.ntotal} vectors.")
                                        else:
                                            logger.warning("WORKER: RECALCULATE - Rebuild resulted in an empty speaker set.")
                                            empty_index = faiss.IndexFlatIP(old_faiss_index.d)
                                            save_faiss_index(empty_index, self.faiss_idx_path)
                                            save_speaker_map({}, self.speaker_map_path)
                                            self.faiss_index = empty_index
                                            self.speaker_map = {}
                        
                        except Exception as e:
                            logger.error(f"WORKER: Unhandled error during RECALCULATE_SPEAKER_PROFILES: {e}", exc_info=True)
                elif isinstance(dequeued_item, dict) and dequeued_item.get('type') == 'REBUILD_SPEAKER_DATABASE':
                    logger.info("WORKER: Processing REBUILD_SPEAKER_DATABASE command.")
                    with self.db_lock:
                        try:
                            # Step 1: Get the list of speakers that should exist from the profiles
                            all_profiles = get_all_speaker_profiles()
                            valid_speaker_names = {profile.get('name') for profile in all_profiles if profile.get('name')}
                            logger.info(f"Rebuilding database for {len(valid_speaker_names)} valid speakers found in profiles.")
                            
                            if not self.faiss_idx_path or not self.speaker_map_path:
                                logger.error("WORKER: Cannot rebuild database. FAISS index or speaker map paths are not configured for the worker.")
                            else:
                                dimension = self.config.get('audio_suite_settings', {}).get('embedding_dim', 192)
                                new_faiss_index = faiss.IndexFlatIP(dimension)
                                new_speaker_map = {}

                                # If there are no profiles, the new empty DB is correct.
                                # If there are profiles, we need the old DB to get their embeddings.
                                if valid_speaker_names and self.faiss_idx_path.exists() and self.speaker_map_path.exists():
                                    old_faiss_index = faiss.read_index(str(self.faiss_idx_path))
                                    with open(self.speaker_map_path, 'r', encoding='utf-8') as f:
                                        old_speaker_map = {int(k): v for k, v in json.load(f).items()}
                                    
                                    dimension = old_faiss_index.d # Use dimension from old index
                                    new_faiss_index = faiss.IndexFlatIP(dimension) # Re-create with correct dimension
                                # Step 4: Populate the new database with only the valid speakers
                                    speakers_retained = 0
                                    for old_id, profile_info in old_speaker_map.items():
                                        speaker_name = None
                                        if isinstance(profile_info, dict):
                                            speaker_name = profile_info.get('name')
                                        elif isinstance(profile_info, str):
                                            speaker_name = profile_info
                                        
                                        if speaker_name and speaker_name in valid_speaker_names:
                                            if old_id < old_faiss_index.ntotal:
                                                embedding = old_faiss_index.reconstruct(old_id).reshape(1, -1)
                                                new_faiss_index.add(embedding)
                                                new_id = new_faiss_index.ntotal - 1
                                                # Preserve the full dictionary structure in the new map
                                                new_speaker_map[new_id] = profile_info
                                                speakers_retained += 1
                                            else:
                                                logger.warning(f"WORKER: REBUILD - ID {old_id} for '{speaker_name}' is out of bounds for the old FAISS index (ntotal={old_faiss_index.ntotal}). Skipping.")

                                    logger.info(f"WORKER: REBUILD - Retained {speakers_retained} speakers. New index will have {new_faiss_index.ntotal} vectors.")
                                    
                                # Step 5: Atomically save the new database, replacing the old one
                                db_index_path = Path(self.faiss_idx_path)
                                map_path = Path(self.speaker_map_path)

                                temp_index_path = db_index_path.with_suffix(db_index_path.suffix + '.tmp')
                                temp_map_path = map_path.with_suffix(map_path.suffix + '.tmp')

                                faiss.write_index(new_faiss_index, str(temp_index_path))
                                with open(temp_map_path, 'w', encoding='utf-8') as f:
                                    json.dump(new_speaker_map, f, indent=2)

                                os.rename(temp_index_path, db_index_path)
                                os.rename(temp_map_path, map_path)
                                
                                logger.info(f"WORKER: Speaker database rebuilt and saved successfully. New index has {new_faiss_index.ntotal} vectors.")

                                # Step 6: Update the worker's in-memory instances
                                self.faiss_index = new_faiss_index
                                self.speaker_map = new_speaker_map

                        except Exception as e:
                            logger.error(f"WORKER: Failed during REBUILD_SPEAKER_DATABASE: {e}", exc_info=True)
                            # Cleanup .tmp files if they exist
                            if 'temp_index_path' in locals() and os.path.exists(str(temp_index_path)):
                                os.remove(str(temp_index_path))
                            if 'temp_map_path' in locals() and os.path.exists(str(temp_map_path)):
                                os.remove(str(temp_map_path))
                elif isinstance(dequeued_item, dict) and dequeued_item.get('type') == 'LOG_TASK_CREATION':
                    logger.info("WORKER: Processing LOG_TASK_CREATION command.")
                    payload = dequeued_item.get('payload', {})
                    try:
                        date_obj = datetime.strptime(payload['processing_date'], '%Y-%m-%d').replace(tzinfo=timezone.utc)
                        update_chunk_metadata(
                            date_obj,
                            payload['chunk_id'],
                            {'generated_task_ids': payload['task_ids']}
                        )
                        logger.info(f"Successfully logged task creation for chunk {payload['chunk_id']}.")
                    except Exception as e:
                        logger.error(f"Error logging task creation: {e}", exc_info=True)

                elif isinstance(dequeued_item, dict) and dequeued_item.get('type') == 'CONFIRM_TASK':
                    logger.info("WORKER: Processing CONFIRM_TASK command from Scheduler.")
                    payload = dequeued_item.get('payload', {})
                    task_id = payload.get('task_id')
                    if task_id:
                        try:
                            task_manager = TaskIntelligenceManager(self.config, self.embedding_model)
                            task_list = task_manager.get_tasks_by_status("pending_confirmation")
                            task_to_confirm = next((t for t in task_list if t.get('task_id') == task_id), None)
                            
                            if task_to_confirm:
                                task_manager.update_task(task_id, {"status": "confirmed"})
                                logger.info(f"Task {task_id} successfully auto-confirmed by scheduler.")
                            else:
                                logger.info(f"Task {task_id} was not in 'pending_confirmation' state. No automatic update applied (likely handled manually).")
                        except Exception as e:
                            logger.error(f"Error auto-confirming task {task_id}: {e}", exc_info=True)

                elif isinstance(dequeued_item, dict) and dequeued_item.get('type') == 'UPDATE_TASK_STATUS_FROM_GUI':
                    logger.info("WORKER: Processing UPDATE_TASK_STATUS_FROM_GUI command.")
                    payload = dequeued_item.get('payload', {})
                    task_id_to_update = payload.get('task_id')
                    polling_task_id = payload.get('polling_task_id')
                    
                    if not task_id_to_update:
                        logger.error("WORKER: Invalid UPDATE_TASK_STATUS command, missing 'task_id' in payload.")
                        if polling_task_id:
                             with GUI_TASK_STATUS_LOCK: GUI_TASK_STATUS[polling_task_id] = "error"
                        continue

                    success = False
                    try:
                        task_manager = TaskIntelligenceManager(self.config, self.embedding_model)
                        updates_to_apply = payload.get('updates', {})
                        if 'new_status' in payload:
                            updates_to_apply['status'] = payload['new_status']
                        
                        if updates_to_apply:
                            author_name = self.config.get('voice_commands', {}).get('user_speaker_name', 'GUI User')
                            updated_task = task_manager.update_task(
                                task_id_to_update, 
                                updates_to_apply,
                                author=author_name
                            )
                            if updated_task:
                                success = True
                        else:
                            logger.warning(f"Received empty update for task {task_id_to_update} from GUI.")
                    except Exception as e:
                        logger.error(f"Error updating task {task_id_to_update} from GUI: {e}", exc_info=True)
                    finally:
                        if polling_task_id:
                            with GUI_TASK_STATUS_LOCK:
                                GUI_TASK_STATUS[polling_task_id] = "complete" if success else "error"
                elif isinstance(dequeued_item, str) and dequeued_item.startswith("CMD_"):
                    # --- Start of new block for CMD_CORRECT_SPEAKER_ASSIGNMENT ---
                    if dequeued_item.startswith("CMD_RESET_MASTER_LOG_TS_STATE_FOR_DATE:"):
                        try:
                            date_to_reset_for_cmd = dequeued_item.split(":", 1)[1]; datetime.strptime(date_to_reset_for_cmd, "%Y-%m-%d")
                            logger.info(f"WORKER: Received CMD_RESET for UTC date: {date_to_reset_for_cmd}")
                            if self.master_log_current_day_str == date_to_reset_for_cmd:
                                logger.info(f"WORKER: Resetting master log rendering state for active UTC day {date_to_reset_for_cmd}.")
                                self.master_log_next_timestamp_marker_abs_utc = None
                                self.master_log_last_speaker = None
                            self.force_header_rewrite_for_day = date_to_reset_for_cmd
                        except (IndexError, ValueError) as e_cmd_parse:
                            logger.error(f"WORKER: Invalid CMD_RESET format: {dequeued_item}. Error: {e_cmd_parse}")
                       
                    elif dequeued_item.startswith("CMD_MASTER_LOG_REBUILT_FOR_DATE:"):
                        try:
                            date_str_rebuilt = dequeued_item.split(":", 1)[1]
                            datetime.strptime(date_str_rebuilt, "%Y-%m-%d") 
                            logger.info(f"WORKER: Received CMD_REBUILT for UTC date: {date_str_rebuilt}")
                            if self.master_log_current_day_str == date_str_rebuilt:
                                logger.info(f"WORKER: Master log for current day {date_str_rebuilt} was rebuilt. Resetting live append state.")
                                self.master_log_last_speaker = None
                                self.master_log_next_timestamp_marker_abs_utc = None
                                self.force_header_rewrite_for_day = date_str_rebuilt
                        except (IndexError, ValueError) as e_cmd_parse:
                            logger.error(f"WORKER: Invalid CMD_REBUILT format: {dequeued_item}. Error: {e_cmd_parse}")
                       
                    # After processing any command, continue to the next queue item

                # --- AUDIO FILE PROCESSING BLOCK (Existing logic, now after command handling) ---
                elif isinstance(dequeued_item, str):
                    from src.daily_log_manager import parse_duration_to_minutes
                    current_audio_file_path = Path(str(dequeued_item)); audio_file_path_for_error_handling = current_audio_file_path
                    logger.info(f"Audio worker dequeued: {current_audio_file_path.name} (Priority: {priority})")
                
                    # Define samson_tz for flag_file_date which is used later in the original code section for flag generation.
                    # This was previously defined inside the block that is now being replaced by the patch.
                    samson_tz_name = self.config.get('timings', {}).get('assumed_recording_timezone', 'UTC')
                    try:
                        samson_tz = pytz.timezone(samson_tz_name)
                    except pytz.UnknownTimeZoneError:
                        logger.error(f"WORKER: Invalid Samson timezone configured: {samson_tz_name}. Defaulting to UTC.")
                        samson_tz = timezone.utc # Fallback to UTC timezone object

                    # Correct Date Determination
                    samson_today_date_obj = get_samson_today()
                    target_date_for_log_operations = datetime.combine(samson_today_date_obj, datetime.min.time())

                    # Check for redundancy using the CORRECT date.
                    seq_num_worker_check = extract_sequence_number_from_filename(current_audio_file_path.stem)
                    if seq_num_worker_check is not None and check_if_sequence_processed(seq_num_worker_check, target_date_for_log_operations):
                        logger.warning(f"WORKER: REDUNDANCY CHECK - Sequence #{seq_num_worker_check} ('{current_audio_file_path.name}') is already processed for date {target_date_for_log_operations.date()}. Discarding job.")
                        archive_folder = self.config.get('paths', {}).get('archived_audio_folder');
                        if archive_folder and isinstance(archive_folder, Path):
                            try:
                                date_subfolder_for_dup = archive_folder / target_date_for_log_operations.strftime('%Y-%m-%d')
                                date_subfolder_for_dup.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(current_audio_file_path), str(date_subfolder_for_dup / f"DUPLICATE_{current_audio_file_path.name}"))
                            except Exception as e_mv: logger.error(f"Error moving duplicate file {current_audio_file_path.name} to archive: {e_mv}")
                        continue

                    # Handle Day Start Time setting (Method A: Automatic)
                    day_start_time_is_set = get_day_start_time(target_date_for_log_operations)
                    if not day_start_time_is_set:
                        logger.info(f"WORKER: First audio file of the day ({current_audio_file_path.name}) detected for an uninitialized day log. Automatically setting day_start_time.")
                    
                        # New logic: DayStartTime = Current UTC wall-clock time - Chunk Duration
                        chunk_duration_s = parse_duration_to_minutes(self.config.get('timings', {}).get('audio_chunk_expected_duration', "10m")) * 60.0
                        auto_start_time_utc = datetime.now(timezone.utc) - timedelta(seconds=chunk_duration_s)
                    
                        day_start_set_successful = set_day_start_time(auto_start_time_utc, target_date=target_date_for_log_operations)
                    
                        if day_start_set_successful:
                            logger.info(f"WORKER: Successfully set day_start_time for {target_date_for_log_operations.strftime('%Y-%m-%d')} to {auto_start_time_utc.isoformat()}.")
                        else:
                            logger.error(f"WORKER: Failed to set day_start_time for {target_date_for_log_operations.strftime('%Y-%m-%d')}.")
                            # Depending on desired robustness, you might want to move the file to error and continue.
                
                    # Get dynamic thresholds for this run
                    audio_suite_settings_cfg = self.config.get('audio_suite_settings', {})
                    current_similarity_threshold = self.dynamic_config_store.get('similarity_threshold', audio_suite_settings_cfg.get('similarity_threshold', 0.63))
                    current_ambiguity_upper_bound = self.dynamic_config_store.get('ambiguity_similarity_upper_bound_for_review', audio_suite_settings_cfg.get('ambiguity_similarity_upper_bound_for_review', 0.80))

                    active_context = get_active_context(config=self.config)
                    logger.info(f"WORKER: Processing file with active matter context: {active_context.get('matter_name') if active_context else 'None'}")

                    processing_result = process_single_audio_file(
                        current_audio_file_path,
                        cfg_similarity_threshold=current_similarity_threshold,
                        cfg_ambiguity_upper_bound=current_ambiguity_upper_bound,
                        db_lock=self.db_lock,
                        processing_date_utc=target_date_for_log_operations,
                        active_matter_context=active_context
                    )

                    if not processing_result or "processing_status" not in processing_result:
                        err_msg = "critical processing failure, invalid result"; _move_file_to_error_folder(current_audio_file_path, self.config, err_msg)
                        self.last_word_end_time_utc = None
                    else:
                        status = processing_result.get("processing_status")
                        archive_folder = self.config.get('paths', {}).get('archived_audio_folder')
                        if status in ["no_dialogue_detected", "processing_error"]:
                            self.last_word_end_time_utc = None
                            if status == "no_dialogue_detected":
                                logger.info(f"No dialogue detected in {current_audio_file_path.name}. Archiving.")
                                if archive_folder and isinstance(archive_folder, Path):
                                    try: 
                                        date_subfolder = archive_folder / target_date_for_log_operations.strftime('%Y-%m-%d')
                                        date_subfolder.mkdir(parents=True, exist_ok=True)
                                        shutil.move(str(current_audio_file_path), str(date_subfolder / current_audio_file_path.name))
                                    except Exception as e_mv: logger.error(f"Error moving file {current_audio_file_path.name} to dated archive: {e_mv}")
                            else: # processing_error
                                error_message = processing_result.get('error_message', 'Unknown error'); original_path_str_for_err = processing_result.get('original_file_path_for_error_handling'); file_to_handle_on_error = Path(original_path_str_for_err) if original_path_str_for_err else current_audio_file_path
                                _move_file_to_error_folder(file_to_handle_on_error, self.config, f"processing error: {error_message}");
                        
                        elif status == "success":
                            if not processing_result.get("identified_transcript"):
                                self.last_word_end_time_utc = None

                            authoritative_chunk_id = processing_result.get("chunk_id")
                            if not authoritative_chunk_id:
                                logger.error(f"WORKER: CRITICAL - chunk_id missing from successful processing_result for {current_audio_file_path.name}. Generating a fallback ID.")
                                # Fallback to a UUID if process_single_audio_file fails to provide an ID.
                                authoritative_chunk_id = str(uuid.uuid4())

                            # STEP 1: Run Matter Analysis FIRST to populate processing_result.
                            try:
                                from src import matter_manager
                                all_matters = matter_manager.get_all_matters()
                                all_profiles = get_all_speaker_profiles()
                                
                                # Before matter analysis
                                chunk_start_time_utc_str = processing_result.get('chunk_start_time_utc')
                                if self.last_word_end_time_utc and chunk_start_time_utc_str:
                                    try:
                                        last_word_dt = datetime.fromisoformat(self.last_word_end_time_utc)
                                        chunk_start_dt = datetime.fromisoformat(chunk_start_time_utc_str)
                                        silence_gap = (chunk_start_dt - last_word_dt).total_seconds()
                                        
                                        threshold = self.config.get('audio_suite_settings', {}).get('true_silence_reset_threshold_seconds', 90)
                                        if silence_gap > threshold:
                                            logger.info(f"True silence gap of {silence_gap:.1f}s detected (>{threshold}s). Resetting matter context.")
                                            self.last_processed_turn_matter_id = None
                                    except (ValueError, TypeError) as e_ts:
                                        logger.warning(f"Could not parse timestamps for silence gap calculation: {e_ts}")


                                active_matter_id_from_context = active_context.get('matter_id') if active_context else None
                                if self.last_processed_turn_matter_id and active_matter_id_from_context and self.last_processed_turn_matter_id != active_matter_id_from_context:
                                    logger.info(f"Context change detected. Previous matter '{self.last_processed_turn_matter_id}' differs from new active matter '{active_matter_id_from_context}'. Resetting stickiness.")
                                    self.last_processed_turn_matter_id = None # Reset the sticky context
                                matter_segments, commands_to_create, final_matter_id_for_chunk, last_word_end_time = self.matter_analysis_service.analyze_chunk(
                                    processing_result['identified_transcript'],
                                    all_matters,
                                    all_profiles,
                                    self.last_processed_turn_matter_id,
                                    active_matter_id_from_context # Pass the active matter ID
                                )
                                
                                self.last_processed_turn_matter_id = final_matter_id_for_chunk
                                processing_result['matter_segments'] = matter_segments
                                self.last_word_end_time_utc = last_word_end_time

                            except Exception as e:
                                logger.error(f"Error during matter analysis: {e}", exc_info=True)
                                processing_result['matter_segments'] = []
                                commands_to_create = []
                                self.last_word_end_time_utc = None

                            if processing_result.get('matter_segments'):
                                transcript_for_propagation = processing_result.get('identified_transcript', [])
                                matter_segments_for_propagation = processing_result['matter_segments']
                                for asr_segment in transcript_for_propagation:
                                    for word in asr_segment.get('words', []):
                                        word_start = word.get('start')
                                        if word_start is not None:
                                            # Default to None if no segment matches
                                            word['matter_id'] = None 
                                            for matter_seg in matter_segments_for_propagation:
                                                # Check if the word's start time is within the matter segment's time range
                                                if matter_seg.get('start_time') <= word_start < matter_seg.get('end_time'):
                                                    word['matter_id'] = matter_seg.get('matter_id')
                                                    break # Found the segment, no need to check further
                                logger.info(f"Successfully propagated matter_id from {len(matter_segments_for_propagation)} segments to words in the transcript.")

                            # STEP 2: Write the complete, analyzed chunk to the log file.
                            add_processed_audio_entry(
                                original_file_path=current_audio_file_path,
                                processing_result=processing_result,
                                target_date_for_log=target_date_for_log_operations,
                                chunk_id=authoritative_chunk_id
                            )
                            logger.info("[TASK_DEBUG] Audio processing for chunk %s successful. Checking if task extraction is enabled.", authoritative_chunk_id)
                            current_config_for_tasks = get_config()
                            task_intel_config = current_config_for_tasks.get('task_intelligence', {})
                            is_task_intel_enabled = task_intel_config.get('enabled', False)
                            logger.info("[TASK_DEBUG] Value of 'task_intelligence.enabled' from config: %s", is_task_intel_enabled)

                            if is_task_intel_enabled:
                                logger.info("[TASK_DEBUG] Condition met. Queuing 'EXTRACT_TASKS_FROM_CHUNK' command for chunk %s.", authoritative_chunk_id)
                                logger.info(f"Task intelligence is enabled. Queuing task extraction for chunk {authoritative_chunk_id} to Command Executor Service.")
                                command_for_executor = {
                                    "command_type": "EXTRACT_TASKS_FROM_CHUNK",
                                    "payload": {
                                        "chunk_id": authoritative_chunk_id,
                                        "processing_date": target_date_for_log_operations.strftime('%Y-%m-%d')
                                    }
                                }
                                # Use the file-based command executor queue     
                                queue_command_for_executor(command_for_executor)
                            # STEP 2.5: Apply any pending time-synced updates to the log entry we just wrote.
                            last_update_id = self._apply_pending_updates(processing_result, target_date_for_log_operations)
                            if last_update_id is not None:
                                # Ensures the new matter "sticks" to the subsequent chunk's analysis
                                self.last_processed_turn_matter_id = last_update_id
                                logger.info(f"Matter context stickiness updated to '{last_update_id}' from timed update.")
                                # This change will require master log regeneration for consistency
                                ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE.add(target_date_for_log_operations.date())
                           
                            try:
                            
                                logger.info(f"WORKER: Refreshing top-level matter list for log date {target_date_for_log_operations.date()}.")
                                all_current_matters = get_all_matters(include_inactive=True)
                                log_data_for_update = get_daily_log_data(target_date_for_log_operations)
                                if log_data_for_update:
                                    log_data_for_update['matters'] = all_current_matters
                                    save_daily_log_data(target_date_for_log_operations, log_data_for_update)
                                    logger.info(f"WORKER: Successfully updated matter list in daily log with {len(all_current_matters)} matters.")
                                else:
                                    logger.warning(f"WORKER: Could not reload log data for {target_date_for_log_operations.date()} to refresh matter list.")
                            except Exception as e_matter_refresh:
                                logger.error(f"WORKER: Failed to refresh top-level matter list in daily log: {e_matter_refresh}", exc_info=True)
                            logger.debug(f"WORKER: Pre-refinement state for '{current_audio_file_path.name}':")
                            if processing_result.get("faiss_index_instance"):
                                logger.debug(f"  - FAISS index instance has {processing_result.get('faiss_index_instance').ntotal} vectors.")
                            if processing_result.get("speaker_map_instance"):
                                logger.debug(f"  - Speaker map instance has {len(processing_result.get('speaker_map_instance'))} speakers: {list(processing_result.get('speaker_map_instance').values())}")
                            
                            # In AudioProcessingWorker.run(), inside 'elif status == "success":'
                            refinement_data_for_update = processing_result.get("refinement_data_for_update", [])
                            
                            if refinement_data_for_update:
                                for ref_item in refinement_data_for_update:
                                    add_segment_embedding_for_evolution(
                                        faiss_id=ref_item['faiss_id'],
                                        embedding=ref_item['new_segment_embedding'],
                                        duration_s=ref_item['segment_duration_s'],
                                        confidence_score=ref_item.get('diarization_confidence', 1.0),
                                        context=ref_item.get('context', 'in_person'),
                                        processing_timestamp=datetime.now(timezone.utc),
                                        chunk_id=authoritative_chunk_id
                                    )

                                logger.info(f"WORKER: Added {len(refinement_data_for_update)} segments to speaker profiles for evolution.")
                                self.pending_refinements.extend(refinement_data_for_update)
                                logger.info(f"Added {len(refinement_data_for_update)} items to refinement batch. Total pending: {len(self.pending_refinements)}.")
                                self._schedule_refinement_commit()

                            # Eagerly update lifetime audio for already-known speakers to maintain consistency.
                            identified_transcript = processing_result.get("identified_transcript", [])
                            duration_deltas = Counter()
                            if identified_transcript:
                                for segment in identified_transcript:
                                    speaker_name = segment.get("speaker")
                                    # Only update for speakers who are already known (not new CUSIDs)
                                    if speaker_name and not speaker_name.startswith("CUSID_"):
                                        # Check if this speaker has a profile to update
                                        profile_to_update = next((p for p in get_all_speaker_profiles() if p.get('name') == speaker_name), None)
                                        if profile_to_update:
                                            segment_duration = sum(w.get('end', 0.0) - w.get('start', 0.0) for w in segment.get('words', []))
                                            if segment_duration > 0:
                                                duration_deltas[speaker_name] += segment_duration

                            if duration_deltas:
                                logger.info(f"WORKER: EAGER_UPDATE - Calculated lifetime audio deltas for identified speakers: {dict(duration_deltas)}")
                                for speaker_name, delta_s in duration_deltas.items():
                                    profile = next((p for p in get_all_speaker_profiles() if p.get('name') == speaker_name), None)
                                    if profile:
                                        current_total_s = profile.get('lifetime_total_audio_s', 0.0)
                                        new_total_s = current_total_s + delta_s
                                        update_speaker_profile(faiss_id=profile['faiss_id'], lifetime_total_audio_s=new_total_s)
                                        logger.info(f"WORKER: EAGER_UPDATE - Updated lifetime audio for '{speaker_name}' from {current_total_s:.2f}s to {new_total_s:.2f}s.")
                            
                            # --- VOICE COMMAND CONFIRMATION LOGIC ---
                            if processing_result.get('recognized_command'):
                                command_data = processing_result['recognized_command']
                                
                                payload = command_data.get('payload', {})
                                
                                # Check for retroactive "duration" based commands first
                                if 'duration' in payload and payload['duration']:
                                    logger.info(f"Retroactive voice command '{command_data['command_type']}' detected. Queuing for CommandExecutor.")
                                    
                                    # Parse duration (e.g., "10 minutes") into seconds
                                    try:
                                        lookback_s = parse_duration_to_minutes(payload['duration']) * 60
                                        payload['lookback_seconds'] = lookback_s # Add to payload for executor
                                        
                                        # Queue for the file-based executor
                                        command_for_executor = {
                                            "command_type": command_data['command_type'],
                                            "payload": payload,
                                            "source_chunk_id": processing_result['chunk_id'],
                                            "processing_date": target_date_for_log_operations.strftime('%Y-%m-%d'),
                                            # Other metadata as needed by CommandExecutorService
                                        }
                                        queue_command_for_executor(command_for_executor)

                                    except Exception as e:
                                        logger.error(f"Failed to parse duration '{payload['duration']}' from voice command: {e}")

                                else: # It's a transcript-synced command
                                    confidence_threshold = self.config.get('voice_commands', {}).get('voice_command_confidence_threshold', 0.85)
                                    confidence_score = command_data.get('confidence_score', 0.0)

                                    if confidence_score >= confidence_threshold:
                                        logger.info(f"High-confidence transcript-synced command '{command_data['command_type']}' detected.")
                                        
                                        # Find the precise timestamp of the command
                                        matched_phrase = command_data.get('matched_phrase', '').lower().strip()
                                        transcript_words = [word for seg in processing_result['identified_transcript'] for word in seg.get('words', [])]
                                        
                                        command_start_s = 0.0 # Default to chunk start
                                        found_phrase = False
                                        if matched_phrase:
                                            phrase_tokens = matched_phrase.split()
                                            for i in range(len(transcript_words) - len(phrase_tokens) + 1):
                                                window = transcript_words[i : i + len(phrase_tokens)]
                                                window_text = ' '.join(w.get('word', '').lower() for w in window)
                                                if window_text == matched_phrase:
                                                    command_start_s = window[0].get('start', 0.0)
                                                    found_phrase = True
                                                    break
                                        
                                        if not found_phrase:
                                            logger.warning(f"Could not find exact matched_phrase '{matched_phrase}' in transcript. Defaulting to chunk start time.")

                                        chunk_start_utc = datetime.fromisoformat(processing_result['chunk_start_time_utc'])
                                        command_start_utc = chunk_start_utc + timedelta(seconds=command_start_s)
                                        
                                        # Resolve matter name to ID
                                        from src.command_executor_service import CommandExecutorService
                                        temp_executor = CommandExecutorService(self.config, threading.Event(), lambda: None)
                                        resolved_id, _ = temp_executor._find_matter_by_fuzzy_name(payload.get('matter_id', ''))
                                        
                                        if resolved_id:
                                            # Queue the time-synced command for the worker itself
                                            command_to_queue = {
                                                "type": "APPLY_TIMED_MATTER_UPDATE",
                                                "payload": {
                                                    "new_matter_id": resolved_id,
                                                    "start_time_utc": command_start_utc,
                                                    "source": "vocal_command"
                                                }
                                            }
                                            _queue_item_with_priority(0, command_to_queue)
                                            logger.info(f"Queued transcript-synced vocal command for matter '{resolved_id}' at {command_start_utc.isoformat()}.")
                                        else:
                                            logger.error(f"Could not resolve matter '{payload.get('matter_id')}' from vocal command. Command dropped.")
                                    
                                    else: # Low confidence
                                        logger.info(f"Command '{command_data['command_type']}' has low confidence ({confidence_score:.2f}). Sending to Signal for confirmation.")

                            # --- Flag Generation and Saving ---
                            all_flags_for_this_chunk = []
                            flag_file_date_obj = target_date_for_log_operations
                            
                            # 1. Collect flags from MatterAnalysisService results
                            if commands_to_create:
                                for command in commands_to_create:
                                    payload = command.get("payload", {})
                                    if payload.get("flag_type") == "matter_conflict":
                                        try:
                                            logger.info(f"Preparing a 'matter_conflict' flag for chunk {processing_result['chunk_id']}.")
                                            # The date is now derived from the processing result for consistency
                                            flag_date_from_chunk = datetime.fromisoformat(processing_result['chunk_start_time_utc'])
                                            new_flag = {
                                                "flag_id": f"FLAG_{flag_date_from_chunk.strftime('%Y%m%d')}_{uuid.uuid4().hex[:12]}",
                                                "timestamp_logged_utc": datetime.now(timezone.utc).isoformat(),
                                                "status": "pending_review",
                                                "flag_type": "matter_conflict",
                                                "chunk_id": processing_result['chunk_id'],
                                                "source_file_name": processing_result['source_file_name'],
                                                "text_preview": payload.get("text"),
                                                "start_time": payload.get("start_time"),
                                                "end_time": payload.get("end_time"),
                                                "conflicting_matters": payload.get("conflicting_matters")
                                            }
                                            all_flags_for_this_chunk.append(new_flag)
                                        except Exception as flag_err:
                                            logger.error(f"Failed to prepare matter_conflict flag: {flag_err}", exc_info=True)

                            # 2. Collect flags from audio pipeline results
                            ambiguous_segments = processing_result.get("ambiguous_segments_flagged", [])
                            new_speakers = processing_result.get("new_speaker_enrollment_data", [])
                            original_file_stem = current_audio_file_path.stem
                            original_thresholds_for_flag = {
                                "similarity_threshold": current_similarity_threshold,
                                "ambiguity_similarity_upper_bound_for_review": current_ambiguity_upper_bound
                            }

                            for seg_data in ambiguous_segments:
                                flag_id = f"FLAG_{flag_file_date_obj.strftime('%Y%m%d')}_{uuid.uuid4().hex[:12]}"
                            
                                relative_start_s = seg_data.get('relative_start_s_in_wav', 0.0)
                                relative_end_s = seg_data.get('relative_end_s_in_wav', relative_start_s + 5.0)

                                snippet_params = {
                                    'date': flag_file_date_obj.strftime('%Y-%m-%d'),
                                    'file_stem': original_file_stem,
                                    'start': round(relative_start_s, 3),
                                    'end': round(relative_end_s, 3)
                                }
                                logger.info(f"WORKER: Generated snippet server params for flag {flag_id}: {snippet_params}")

                                # Clean up and prepare seg_data before merging
                                seg_data.pop('snippet_path_abs', None)
                                seg_data['snippet_server_params'] = snippet_params
                                if isinstance(seg_data.get('segment_embedding'), np.ndarray):
                                    seg_data['segment_embedding'] = seg_data.pop('segment_embedding').tolist()
                            
                                # Create the new flag object safely
                                new_ambiguous_flag = {
                                    "flag_id": flag_id,
                                    "chunk_id": authoritative_chunk_id, # <<< ADDED chunk_id
                                    "timestamp_logged_utc": datetime.now(timezone.utc).isoformat(),
                                    "status": "pending_review",
                                    "source_file_name": current_audio_file_path.name,
                                    "original_identification_thresholds": original_thresholds_for_flag,
                                }
                                # Safely merge the rest of the data
                                new_ambiguous_flag.update(seg_data)
                                all_flags_for_this_chunk.append(new_ambiguous_flag)
                        
                            for speaker_data in new_speakers:
                                flag_id = f"FLAG_{flag_file_date_obj.strftime('%Y%m%d')}_{uuid.uuid4().hex[:12]}"
                            
                                # Get relative start/end times from the speaker_data payload
                                relative_start_s = speaker_data.get('start_time', 0.0)
                                relative_end_s = speaker_data.get('end_time', relative_start_s + 5.0)

                                # Create the snippet_server_params, just like for ambiguous flags
                                snippet_params = {
                                    'date': flag_file_date_obj.strftime('%Y-%m-%d'),
                                    'file_stem': original_file_stem,
                                    'start': round(relative_start_s, 3),
                                    'end': round(relative_end_s, 3)
                                }
                                logger.info(f"WORKER: Generated snippet server params for new speaker flag {flag_id}: {snippet_params}")
                            
                                # Convert numpy embedding to list for JSON serialization if needed
                                embedding_for_json = speaker_data.get('embedding')
                                if isinstance(embedding_for_json, np.ndarray):
                                    embedding_for_json = embedding_for_json.tolist()

                                # Reconstruct the flag dictionary explicitly instead of using **
                                new_speaker_flag = {
                                    "flag_id": flag_id,
                                    "chunk_id": authoritative_chunk_id, # <<< ADDED chunk_id
                                    "timestamp_logged_utc": datetime.now(timezone.utc).isoformat(),
                                    "status": "pending_review",
                                    "source_file_name": current_audio_file_path.name,
                                    "reason_for_flag": "New unknown speaker detected.",
                                    "cusid": speaker_data.get("cusid"),
                                    "tentative_speaker_name": speaker_data.get("temp_id"),
                                    "snippet_server_params": snippet_params,
                                    "embedding": embedding_for_json,
                                    "summary": speaker_data.get("summary"),
                                    "start_time": speaker_data.get("start_time"),
                                    "end_time": speaker_data.get("end_time"),
                                    "original_label": speaker_data.get("original_label"),
                                    "original_identification_thresholds": original_thresholds_for_flag
                                }
                                all_flags_for_this_chunk.append(new_speaker_flag)

                            # 3. Atomically save all collected flags
                            if all_flags_for_this_chunk:
                                daily_flags_list = load_daily_flags_queue(flag_file_date_obj)
                                daily_flags_list.extend(all_flags_for_this_chunk)
                                save_daily_flags_queue(daily_flags_list, flag_file_date_obj)
                                logger.info(f"WORKER: Saved {len(all_flags_for_this_chunk)} new flags to queue for {flag_file_date_obj.strftime('%Y-%m-%d')}.")

                        
                            
                            # --- Log and Master Transcript Processing ---
                            # Determine the day's start time for master log calculations (using target_date_for_log_operations)
                            day_start_time = get_day_start_time(target_date=target_date_for_log_operations)
                            if not day_start_time:
                                logger.error(f"WORKER: CRITICAL - day_start_time is None for {target_date_for_log_operations.date()}. Master log append may be incorrect.")
                            else:
                                # Calculate chunk's absolute start time
                                chunk_duration_s = parse_duration_to_minutes(self.config.get('timings', {}).get('audio_chunk_expected_duration', "10m")) * 60.0
                                chunk_start_utc_for_master_log = day_start_time + timedelta(seconds=(seq_num_worker_check - 1) * chunk_duration_s) if seq_num_worker_check else day_start_time
                            
                                # Use the correct target date for master log
                                actual_master_log_day_str = target_date_for_log_operations.strftime('%Y-%m-%d')
                                master_log_path_obj = get_master_transcript_path(self.config, target_date=target_date_for_log_operations)

                                is_first_chunk = self.master_log_current_day_str != actual_master_log_day_str or \
                                                    (master_log_path_obj and not master_log_path_obj.exists()) or \
                                                    self.force_header_rewrite_for_day == actual_master_log_day_str
                                if is_first_chunk:
                                    logger.info(f"WORKER: MasterLogAppend: New day or forced rewrite for '{actual_master_log_day_str}'. Resetting render state.");
                                    self.master_log_current_day_str = actual_master_log_day_str
                                    self.master_log_last_speaker = None
                                    self.master_log_next_timestamp_marker_abs_utc = None
                                    self.force_header_rewrite_for_day = None # Consume the flag

                                self.master_log_last_speaker, self.master_log_next_timestamp_marker_abs_utc = append_processed_chunk_to_master_log(
                                    master_log_path=master_log_path_obj, 
                                    word_level_transcript_for_chunk=processing_result.get("identified_transcript"),
                                    audio_chunk_start_time_utc=chunk_start_utc_for_master_log,
                                    is_first_chunk_of_day_for_master_log=is_first_chunk,
                                    current_master_log_day_str=actual_master_log_day_str,
                                    last_speaker_in_master_log=self.master_log_last_speaker,
                                    next_timestamp_marker_abs_utc=self.master_log_next_timestamp_marker_abs_utc,
                                    config=self.config
                                )
                            
                                # Archive the original file
                                if archive_folder and isinstance(archive_folder, Path):
                                    try:
                                        date_subfolder = archive_folder / target_date_for_log_operations.strftime('%Y-%m-%d')
                                        date_subfolder.mkdir(parents=True, exist_ok=True)
                                        shutil.move(str(current_audio_file_path), str(date_subfolder / current_audio_file_path.name))
                                    except Exception as e_mv:
                                        logger.error(f"Error moving successfully processed file {current_audio_file_path.name} to dated archive: {e_mv}")
                                else:
                                    logger.warning(f"No archive folder configured. Processed file '{current_audio_file_path.name}' remains in monitored folder (RISK OF REPROCESSING ON RESTART).")
                        
                        else:
                            _move_file_to_error_folder(current_audio_file_path, self.config, f"unknown status: {status}")

            except queue.Empty:
                dequeued_item_info = None # No item was processed, do nothing
            except Exception as e:
                logger.error(f"Audio processing worker: Unhandled error in main loop: {e}", exc_info=True)
                self.last_word_end_time_utc = None
                # Ensure audio_file_path_for_error_handling is a Path before moving
                if isinstance(audio_file_path_for_error_handling, Path):
                    _move_file_to_error_folder(audio_file_path_for_error_handling, self.config, f"unhandled exception in worker: {e}")
                else:
                    logger.warning(f"Could not move item to error folder as it was not a valid file path. Item: {str(audio_file_path_for_error_handling)[:200]}")
                time.sleep(1)
            finally:
                if dequeued_item_info:
                    AUDIO_PROCESSING_QUEUE.task_done()
       
        logger.info("Audio processing worker shutting down.")
        self._save_speaker_db()

def run_samson_orchestrator():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1' # For macOS Accelerate framework
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    # On macOS, some libraries (like PyTorch) used in subprocesses can print a harmless
    # but noisy "MallocStackLogging" message to stderr. This environment variable
    # prevents that message from appearing.
    if 'darwin' in sys.platform:
        os.environ['MallocStackLogging'] = 'no'
        os.environ['OBJC_DISABLE_INITIALIZE_FOR_KIDS'] = '1'

    global THEFUZZ_AVAILABLE, AUDIO_PROCESSING_THREAD, logger, ORCHESTRATOR_FOLDER_MONITOR_OBSERVER, ORCHESTRATOR_EVENT_HANDLER, HTTP_SERVER_THREAD, HTTP_SERVER_INSTANCE, LAST_EOD_SUMMARY_DATE, COMMAND_QUEUE_MONITOR_THREAD, COMMAND_EXECUTOR_THREAD, SCHEDULER_SERVICE_THREAD # <<< ADD THIS
    ensure_config_exists(); config = get_config()
    log_folder_path = config.get('paths', {}).get('log_folder'); log_file_name_str = config.get('paths', {}).get('log_file_name')
    if not isinstance(log_folder_path, Path) or not log_file_name_str: print(f"FATAL: Log folder/file name not configured. Exiting."); sys.exit(1)
    setup_logging(log_folder=log_folder_path, log_file_name=log_file_name_str)
    if not THEFUZZ_AVAILABLE: logger.warning("The 'thefuzz' library not installed. Fuzzy name matching disabled. `pip install \"thefuzz[speedup]\"`")


    logger.setLevel(logging.DEBUG)
   
    logger.info("SAMSON Orchestrator starting up...")
    

    # --- NEW: Voice Command User Validation ---
    vc_config = config.get('voice_commands', {})
    if vc_config.get('enabled', False):
        user_speaker_name = vc_config.get('user_speaker_name')
        if not user_speaker_name:
            logger.critical("CONFIG ERROR: Voice commands enabled, but 'user_speaker_name' is missing.")
        else:
            try:
                if user_speaker_name not in get_enrolled_speaker_names():
                    logger.critical(f"CONFIG ERROR: Voice command user '{user_speaker_name}' is not an enrolled speaker.")
            except Exception as e:
                logger.error(f"Could not validate voice command user: {e}")
    

    audio_suite_cfg = config.get('audio_suite_settings', {})
    dynamic_config_store = {
        "similarity_threshold": audio_suite_cfg.get('similarity_threshold', 0.63),
        "ambiguity_similarity_upper_bound_for_review": audio_suite_cfg.get('ambiguity_similarity_upper_bound_for_review', 0.80)
    }
    logger.info(f"Dynamic configuration store initialized: {dynamic_config_store}")

    initialize_audio_models()

    # --- Initialize Summary LLM ---
    summary_llm_instance = None
    try:
        summary_llm_instance = get_llm_chat_model(config)
        logger.info("Summary LLM instance initialized.")
    except Exception as e:
        logger.error(f"Fatal error: Could not initialize summary LLM. {e}", exc_info=True)
        return

    logger.info("Loading shared text embedding model...")
    shared_text_embedding_model = None
    try:
        embedding_model_config = config['llm']['text_embedding_model']
        model_name = embedding_model_config['model_name']
        shared_text_embedding_model = SentenceTransformer(model_name)
        logger.info(f"Successfully loaded shared embedding model: {model_name}")
    except Exception as e:
        logger.error(f"Fatal error: Could not load shared text embedding model. {e}", exc_info=True)
        return

    
    matter_analysis_service = MatterAnalysisService(config, shared_text_embedding_model)
    logger.info("Matter Analysis Service initialized.")

    paths_cfg = config.get('paths', {})
    aps_cfg = config.get('audio_suite_settings', {})
    speaker_db_dir_str = paths_cfg.get('speaker_db_dir')
    faiss_index_filename = aps_cfg.get('faiss_index_filename')
    speaker_map_filename = aps_cfg.get('speaker_map_filename')
    embedding_dim = aps_cfg.get('embedding_dim', 192)

    faiss_index_instance = None
    speaker_map_instance = None
    faiss_index_path_obj = None
    speaker_map_path_obj = None

    if all([speaker_db_dir_str, faiss_index_filename, speaker_map_filename]):
        db_dir = Path(speaker_db_dir_str)
        faiss_index_path_obj = db_dir / faiss_index_filename
        speaker_map_path_obj = db_dir / speaker_map_filename
        try:
            faiss_index_instance = load_or_create_faiss_index(faiss_index_path_obj, embedding_dim)
            speaker_map_instance = load_or_create_speaker_map(speaker_map_path_obj)
            if faiss_index_instance is not None and speaker_map_instance is not None:
                logger.info(f"Main Orchestrator: FAISS index ({faiss_index_instance.ntotal} vectors) and speaker map ({len(speaker_map_instance)} speakers) loaded.")
            else:
                logger.error("Main Orchestrator: FAISS index or speaker map is None after attempting to load/create.")
        except Exception as e:
            logger.error(f"Main Orchestrator: Failed to load FAISS/speaker_map: {e}", exc_info=True)
    else:
        logger.warning("Main Orchestrator: Speaker DB paths not fully configured. Speaker ID and Intelligence features will be limited.")

    audio_worker_instance = AudioProcessingWorker(
        worker_config=config, 
        dynamic_config_store=dynamic_config_store,
        db_lock=SPEAKER_DB_LOCK,
        faiss_index_path=faiss_index_path_obj, 
        speaker_map_path=speaker_map_path_obj,
        summary_llm=summary_llm_instance,
        matter_analysis_service=matter_analysis_service,
        embedding_model=shared_text_embedding_model
    )
    AUDIO_PROCESSING_THREAD = threading.Thread(target=audio_worker_instance.run, daemon=True); AUDIO_PROCESSING_THREAD.start()


    # --- HTTP Server Initialization --- (Consolidated here)
    try:
        http_server_port = int(config.get('gui', {}).get('http_server_port', 8001))
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        # Use ThreadingTCPServer for concurrent request handling
        HTTP_SERVER_INSTANCE = socketserver.ThreadingTCPServer(("", http_server_port), AudioSnippetRequestHandler)
        HTTP_SERVER_THREAD = threading.Thread(target=HTTP_SERVER_INSTANCE.serve_forever, daemon=True)
        HTTP_SERVER_THREAD.start()
        logger.info(f"ORCHESTRATOR: Audio snippet HTTP server started on port {http_server_port}.")
    except Exception as e:
        logger.error(f"ORCHESTRATOR: Failed to start audio snippet HTTP server: {e}", exc_info=True)
        HTTP_SERVER_INSTANCE = None # Ensure it's None if startup fails
        HTTP_SERVER_THREAD = None

    try:
        api_server_port = config.get('api_server', {}).get('port', 5000)
        API_SERVER_THREAD = threading.Thread(
            target=lambda: api_flask_app.run(
                host='0.0.0.0', 
                port=api_server_port, 
                debug=False, 
                use_reloader=False
            ), 
            daemon=True,
            name="APIServerThread"
        )
        API_SERVER_THREAD.start()
        logger.info(f"ORCHESTRATOR: Flask API server started in a background thread on port {api_server_port}.")
    except Exception as e:
        logger.error(f"ORCHESTRATOR: Failed to start Flask API server thread: {e}", exc_info=True)
    

    if all([faiss_index_path_obj, speaker_map_path_obj]):
        speaker_intel_service = SpeakerIntelligenceBackgroundService(
            audio_processing_queue=AUDIO_PROCESSING_QUEUE,
            global_config=config, 
            shutdown_event=ORCHESTRATOR_SHUTDOWN_EVENT,
            dynamic_config_store=dynamic_config_store, 
            db_lock=SPEAKER_DB_LOCK,
            # Pass paths as strings for the service to create Path objects
            faiss_index_path_str=str(faiss_index_path_obj),
            speaker_map_path_str=str(speaker_map_path_obj)
        )
        speaker_intel_service.start()
    else:
        logger.warning("Skipping Speaker Intelligence Service initialization due to missing FAISS/map resources.")

    # --- Command Executor Service Initialization ---
    services_cfg = config.get('services', {})
    if services_cfg.get('command_executor', {}).get('enabled', False):
        logger.info("Initializing Command Executor Service...")
        command_executor_service = CommandExecutorService(
            config=config,
            shutdown_event=ORCHESTRATOR_SHUTDOWN_EVENT,
            queue_item_function=_queue_item_with_priority,
            embedding_model=shared_text_embedding_model  
        )
        COMMAND_EXECUTOR_THREAD = threading.Thread(
            target=command_executor_service.run,
            name="CommandExecutorServiceThread",
            daemon=True
        )
        COMMAND_EXECUTOR_THREAD.start()
        logger.info("Command Executor Service started.")

        # --- Scheduler Service Initialization ---
        logger.info("Initializing Scheduler Service...")
        scheduler_service = SchedulerService(
            config=config,
            shutdown_event=ORCHESTRATOR_SHUTDOWN_EVENT,
            queue_item_function=_queue_item_with_priority 
        )
        SCHEDULER_SERVICE_THREAD = threading.Thread(
            target=scheduler_service.run,
            name="SchedulerServiceThread",
            daemon=True
        )
        SCHEDULER_SERVICE_THREAD.start()
        logger.info("Scheduler Service started.")

    if config.get('signal', {}).get('samson_phone_number') and config.get('signal', {}).get('recipient_phone_number'):
        logger.info("ORCHESTRATOR: Signal integration is ENABLED. Starting listener.")
        configured_signal_handler = functools.partial(main_orchestrator_signal_command_handler, config_from_caller=config)
        ORCHESTRATOR_SIGNAL_LISTENER_THREAD = start_signal_listener_thread(configured_signal_handler, config, ORCHESTRATOR_SHUTDOWN_EVENT)
   
    monitored_folder_path = config.get('paths', {}).get('monitored_audio_folder')
    if monitored_folder_path and isinstance(monitored_folder_path, Path):
        try:
            monitored_folder_path.mkdir(parents=True, exist_ok=True)
            today_for_startup = datetime.now(timezone.utc) # For startup, this can be UTC today
            # If get_samson_today() is available and robust, it could be used here too for consistency,
            # but daily_log_manager.get_highest_processed_sequence expects a datetime object for its `target_date`
            samson_today_date_for_startup = get_samson_today()
            startup_datetime_for_log = datetime.combine(samson_today_date_for_startup, datetime.min.time())

            last_processed_seq_startup = get_highest_processed_sequence(startup_datetime_for_log)
            logger.info(f"ORCHESTRATOR: Initializing folder monitor. Highest processed sequence for Samson today ({startup_datetime_for_log.date()}) is #{last_processed_seq_startup}.")
            ORCHESTRATOR_FOLDER_MONITOR_OBSERVER, ORCHESTRATOR_EVENT_HANDLER = start_monitoring(
                folder_to_watch=monitored_folder_path,
                callback_on_new_file=on_new_audio_file_detected,
                last_known_processed_sequence=last_processed_seq_startup,
                file_extensions=config.get('folder_monitor', {}).get('watched_extensions', ['.aac', '.m4a', '.wav', '.mp3']),
                app_config=config
            )
        except Exception as e: logger.error(f"ORCHESTRATOR: Failed to start folder monitor: {e}", exc_info=True)

    logger.info("SAMSON Orchestrator is running. Press Ctrl+C to exit.")
    try:
        while not ORCHESTRATOR_SHUTDOWN_EVENT.is_set():
            # Check for EOD tasks
            try:
                timings_cfg = config.get('timings', {})
                eod_time_str = timings_cfg.get('end_of_day_summary_time', '21:00')
                assumed_tz_str = timings_cfg.get('assumed_recording_timezone', 'UTC')
               
                # Get current time in the configured timezone
                local_tz = pytz.timezone(assumed_tz_str)
                now_local = datetime.now(local_tz)
               
                # Parse EOD time
                eod_time_obj = datetime.strptime(eod_time_str, "%H:%M").time()
               
                # Check if it's after the EOD time and tasks for today haven't run
                if now_local.time() >= eod_time_obj:
                    # Thread-safe check and execution of EOD tasks
                    _handle_end_of_day_tasks(config)

            except Exception as e_eod:
                logger.error(f"Orchestrator: Error during end-of-day task check: {e_eod}", exc_info=True)

            # Check for pending master log regenerations
            if ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE:
                # Create a copy of the set to iterate over, in case it's modified elsewhere
                days_to_process = list(ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE)
                logger.info(f"Orchestrator: Found {len(days_to_process)} day(s) in the regeneration queue: {days_to_process}")
               
                for day_date_obj in days_to_process:                      
                    # Create and queue a command instead of calling directly
                    regen_command = {
                        "type": "REGENERATE_MASTER_LOG",
                        "payload": {"date_str": day_date_obj.strftime('%Y-%m-%d')}
                    }
                    _queue_item_with_priority(0, regen_command) # High priority
                    logger.info(f"Orchestrator: Queued REGENERATE_MASTER_LOG command for {day_date_obj.strftime('%Y-%m-%d')}.")
                ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE.clear() # Clear the queue now that commands are sent
               
                logger.info("Orchestrator: Finished processing regeneration queue for this cycle.")

            # The main loop can now sleep for a longer, more reasonable interval.
            ORCHESTRATOR_SHUTDOWN_EVENT.wait(timeout=60) # Check for shutdown every minute

    except KeyboardInterrupt: logger.info("Ctrl+C received. Shutting down SAMSON Orchestrator...")
    finally:
        logger.info("Initiating shutdown sequence...")
        ORCHESTRATOR_SHUTDOWN_EVENT.set()
        if ORCHESTRATOR_EVENT_HANDLER: logger.info("Stopping folder monitor event handler..."); ORCHESTRATOR_EVENT_HANDLER.stop()
        if ORCHESTRATOR_FOLDER_MONITOR_OBSERVER and ORCHESTRATOR_FOLDER_MONITOR_OBSERVER.is_alive(): logger.info("Stopping folder monitor observer..."); ORCHESTRATOR_FOLDER_MONITOR_OBSERVER.stop(); ORCHESTRATOR_FOLDER_MONITOR_OBSERVER.join(timeout=5)
       
# Add this inside the `finally` block at the end of the function
        # --- HTTP Server Shutdown ---
        if HTTP_SERVER_INSTANCE:
            logger.info("Stopping HTTP server...");
            HTTP_SERVER_INSTANCE.shutdown() # This will stop serve_forever
            HTTP_SERVER_INSTANCE.server_close() # This closes the server socket
        if HTTP_SERVER_THREAD and HTTP_SERVER_THREAD.is_alive():
            HTTP_SERVER_THREAD.join(timeout=5)
            if HTTP_SERVER_THREAD.is_alive():
                logger.warning("HTTP Server thread did not terminate cleanly.")
            else:
                logger.info("HTTP Server thread terminated.")
           
        # --- Command Queue Monitor Shutdown ---
       # if COMMAND_QUEUE_MONITOR_THREAD and COMMAND_QUEUE_MONITOR_THREAD.is_alive():
       #     logger.info("Waiting for command queue monitor to finish...")
        #    COMMAND_QUEUE_MONITOR_THREAD.join(timeout=5)
        if ORCHESTRATOR_SIGNAL_LISTENER_THREAD and ORCHESTRATOR_SIGNAL_LISTENER_THREAD.is_alive():
            logger.info("Waiting for Signal listener thread to finish...")
            # The timeout should be generous enough for signal-cli to terminate
            ORCHESTRATOR_SIGNAL_LISTENER_THREAD.join(timeout=12)
            if ORCHESTRATOR_SIGNAL_LISTENER_THREAD.is_alive():
                logger.warning("Signal listener thread did not terminate cleanly.")
            else:
                logger.info("Signal listener thread has finished.")

        # Add shutdown logic for the new thread
        if COMMAND_EXECUTOR_THREAD and COMMAND_EXECUTOR_THREAD.is_alive():
            logger.info("Waiting for Command Executor thread to finish...")
            COMMAND_EXECUTOR_THREAD.join(timeout=10)
            if COMMAND_EXECUTOR_THREAD.is_alive():
                logger.warning("Command Executor thread did not terminate cleanly.")
            else:
                logger.info("Command Executor thread has finished.")

        # Add shutdown logic for the new scheduler thread
        if SCHEDULER_SERVICE_THREAD and SCHEDULER_SERVICE_THREAD.is_alive():
            logger.info("Waiting for Scheduler Service thread to finish...")
            SCHEDULER_SERVICE_THREAD.join(timeout=10)
            if SCHEDULER_SERVICE_THREAD.is_alive():
                logger.warning("Scheduler Service thread did not terminate cleanly.")
            else:
                logger.info("Scheduler Service thread has finished.")

        if 'audio_worker_instance' in locals():
            logger.info("Persisting worker state before shutdown...")
            try:
                state_to_persist = {
                    'pending_matter_updates': audio_worker_instance.pending_matter_updates,
                    'last_processed_turn_matter_id': audio_worker_instance.last_processed_turn_matter_id
                }
                state_file = config['paths']['system_state_dir'] / "worker_state.json"
                with open(state_file, 'w', encoding='utf-8') as f:
                    # Custom serializer to handle datetime objects
                    def json_serializer(obj):
                        if isinstance(obj, datetime):
                            return obj.isoformat()
                        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
                    json.dump(state_to_persist, f, indent=2, default=json_serializer)
                logger.info("Worker state persisted successfully.")
            except Exception as e:
                logger.error(f"Failed to persist worker state on shutdown: {e}", exc_info=True)


            logger.info("Performing final commit of pending refinements before shutdown...")
            audio_worker_instance._commit_pending_refinements()
            logger.info("Signaling audio processing worker to shut down..."); audio_worker_instance.stop()
            if AUDIO_PROCESSING_THREAD and AUDIO_PROCESSING_THREAD.is_alive(): logger.info("Waiting for audio processing worker to finish..."); AUDIO_PROCESSING_THREAD.join(timeout=20)
        cleanup_audio_models(); logger.info("SAMSON Orchestrator shut down gracefully.")

if __name__ == "__main__":
    run_samson_orchestrator()
