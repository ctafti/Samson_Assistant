
from pathlib import Path
import json
from datetime import datetime, timedelta, timezone, date # Added date
import os
import pytz
from typing import Dict, Any, List, Optional, Tuple
import re # Ensure re is imported
import threading # Added for concurrency control
from collections import Counter # Import Counter for segment speaker recalculation
import uuid # For migration
import time # For corrupted file backup timestamp
from filelock import FileLock
from src.utils.file_locking import get_lock

from src.logger_setup import logger
from src.config_loader import get_config
from src.feedback_manager import log_matter_correction, log_correction_feedback
import shutil

# Global variable to cache the day's start time for performance.
# Key: "YYYY-MM-DD", Value: datetime object in UTC
_day_start_times_cache: Dict[str, Optional[datetime]] = {}

# The _daily_log_data_cache has been removed to ensure data freshness from the file system.
# --- Concurrency Control ---
_daily_log_file_locks: Dict[str, threading.RLock] = {}
_management_lock_for_locks_dict = threading.Lock() # To protect _daily_log_file_locks itself


def _get_lock_for_date_str(date_str: str) -> FileLock:
    """
    Safely retrieves a process-safe lock for a given date_str.
    """
    return get_lock(f"daily_log_{date_str}")

# --- End Concurrency Control ---

def get_samson_today() -> date:
    """
    Returns the current 'Samson date' based on the configured timezone.
    This is the single source of truth for what "today" means, ensuring
    that the GUI and backend services are always aligned.
    """
    config = get_config()
    assumed_tz_str = config.get('timings', {}).get('assumed_recording_timezone', 'UTC')
    try:
        samson_tz = pytz.timezone(assumed_tz_str)
    except pytz.UnknownTimeZoneError:
        logger.warning(f"Invalid timezone '{assumed_tz_str}' in config, falling back to UTC.")
        samson_tz = timezone.utc

    return datetime.now(samson_tz).date()

def parse_duration_to_minutes(duration_config_value: Any, default_minutes: float = 10.0) -> float:
    """
    Parses a duration string (e.g., "10m", "600s") or number into minutes.
    If just a number, assumes minutes.
    """
    if isinstance(duration_config_value, (int, float)):
        return float(duration_config_value)
    if isinstance(duration_config_value, str):
        duration_str = duration_config_value.strip().lower()
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(m|s)?", duration_str)
        if match:
            value_str, unit = match.groups()
            try:
                value = float(value_str)
                if unit == 's':
                    return value / 60.0
                # If unit is 'm' or None (no unit specified, assume minutes)
                return value
            except ValueError:
                logger.warning(f"DailyLogManager: Could not parse duration value '{value_str}' from '{duration_config_value}'. Using default {default_minutes}m.")
                return default_minutes
        else:
            logger.warning(f"DailyLogManager: Unrecognized duration format '{duration_config_value}'. Using default {default_minutes}m.")
            return default_minutes
            
    logger.warning(f"DailyLogManager: Invalid type for duration_config_value '{type(duration_config_value)}'. Using default {default_minutes}m.")
    return default_minutes

def extract_sequence_number_from_filename(filename_stem: str) -> Optional[int]:
    """
    Extracts the sequence number from a filename stem like 'alibi-recording-audio_recordings-1'.
    Returns the number as an int, or None if not found.
    """
    # Regex to match the specific prefix and capture the number at the end
    match = re.fullmatch(r".*-(\d+)", filename_stem, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            logger.error(f"DailyLogManager: Could not convert extracted sequence number '{match.group(1)}' to int for stem '{filename_stem}'.")
            return None
    logger.debug(f"DailyLogManager: Sequence number not found in filename stem '{filename_stem}' using pattern 'alibi-recording-audio_recordings-NUMBER'.")
    return None

def get_log_file_path(target_date: Optional[datetime] = None) -> Path:
    """Gets the path to the daily log JSON file for the given date (or today if None)."""
    if target_date is None:
        # Use the centralized function to get today's date according to Samson's timezone
        target_date = datetime.combine(get_samson_today(), datetime.min.time())
    date_str = target_date.strftime("%Y-%m-%d")

    config = get_config()
    paths_config = config.get('paths', {})
    daily_log_folder_path = paths_config.get('daily_log_folder')

    if not isinstance(daily_log_folder_path, Path):
        logger.critical("DailyLogManager: 'paths.daily_log_folder' is not configured as a Path object in config_loader. This is a critical setup error.")
        # Attempt a fallback to a default location if critical path is missing
        # This should ideally not happen if config_loader is robust.
        from src.config_loader import PROJECT_ROOT # Local import to avoid circular dependency at module load time
        daily_log_folder_path = PROJECT_ROOT / "data" / "daily_logs_fallback"
        logger.warning(f"DailyLogManager: Fallback: Using default daily log folder: {daily_log_folder_path}")


    daily_log_folder_path.mkdir(parents=True, exist_ok=True)
    return daily_log_folder_path / f"{date_str}_samson_log.json"

def _load_log_from_file(log_path: Path) -> Dict[str, Any]:
    """Loads the JSON log file. Returns a default structure if not found or error."""
    default_log_structure = {"schema_version": "2.0", "day_start_timestamp_utc": None, "chunks": {}}
    if log_path.exists():
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "entries" in data and "chunks" not in data:
                logger.info(f"Migrating old log file {log_path.name} to new chunk-based format.")
                migrated_chunks = {}
                for entry in data.get("entries", []):
                    # Create a placeholder chunk_id if one doesn't exist
                    chunk_id = entry.get("chunk_id", str(uuid.uuid4()))
                    entry["chunk_id"] = chunk_id # Ensure it's in the entry
                    migrated_chunks[chunk_id] = entry
                data["chunks"] = migrated_chunks
                del data["entries"]
                data["schema_version"] = "2.0"
            
            if not isinstance(data, dict) or "schema_version" not in data:
                logger.warning(f"Log file {log_path} is not in expected format. Initializing with default structure.")
                return default_log_structure
            
            if isinstance(data.get("day_start_timestamp_utc"), str):
                try:
                    # Ensure 'Z' is handled for ISO 8601 UTC
                    dt_str = data["day_start_timestamp_utc"]
                    if dt_str.endswith("Z"):
                        dt_str = dt_str[:-1] + "+00:00"
                    data["day_start_timestamp_utc"] = datetime.fromisoformat(dt_str)
                except (ValueError, TypeError):
                    logger.error(f"Invalid day_start_timestamp_utc format in {log_path}. Resetting to None.")
                    data["day_start_timestamp_utc"] = None
            elif not isinstance(data.get("day_start_timestamp_utc"), (datetime, type(None))):
                 data["day_start_timestamp_utc"] = None 

            if "chunks" not in data or not isinstance(data["chunks"], dict):
                data["chunks"] = {}

            return data
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from log file {log_path}. Backing up and returning default structure.")
            try:
                backup_path = log_path.with_suffix(f".corrupted.{int(time.time())}.json")
                shutil.copy(log_path, backup_path)
                logger.info(f"Backed up corrupted log file to {backup_path}")
            except Exception as e:
                logger.error(f"Failed to backup corrupted log file {log_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading log file {log_path}: {e}. Initializing with default structure.")
    return default_log_structure

def _save_log_to_file(log_path: Path, log_data: Dict[str, Any]):
    """
    Saves the log data to the JSON file.
    ASSUMES THE CALLER HOLDS THE APPROPRIATE DATE-SPECIFIC LOCK.
    """
    logger.debug(f"Entering _save_log_to_file for path: {log_path}")
    try:
        data_to_save = log_data.copy()
        if isinstance(data_to_save.get("day_start_timestamp_utc"), datetime):
            data_to_save["day_start_timestamp_utc"] = data_to_save["day_start_timestamp_utc"].isoformat()
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        logger.info(f"Daily log successfully saved to {log_path} via json.dump.")
    except Exception as e:
        logger.error(f"Failed to save daily log to {log_path}: {e}", exc_info=True)

def _fetch_or_load_daily_log_data_locked(date_str: str, target_date_obj: datetime) -> Dict[str, Any]:
    """
    Internal helper to get daily log data.
    ASSUMES THE CALLER HOLDS THE LOCK FOR date_str.
    This function now ALWAYS loads from the file to ensure freshness across processes.
    """
    log_path = get_log_file_path(target_date_obj)
    log_data = _load_log_from_file(log_path)
    # The caching logic has been intentionally removed.
    return log_data

def get_daily_log_data(target_date: Optional[datetime] = None) -> Dict[str, Any]:
    """Loads daily log data, using a cache for the current day. Thread-safe."""
    target_date_obj = target_date or datetime.now(timezone.utc)
    date_str = target_date_obj.strftime("%Y-%m-%d")
    date_specific_lock = _get_lock_for_date_str(date_str)
    with date_specific_lock:
        return _fetch_or_load_daily_log_data_locked(date_str, target_date_obj)

def save_daily_log_data(target_date: datetime, log_data: Dict[str, Any]) -> bool:
    """
    Saves the provided log data for the given target_date.
    This function will acquire a lock for the specific date, save the data to the
    corresponding log file, and update the in-memory cache.
    Args:
        target_date: The datetime object representing the date for which to save data.
                     The date part is used to determine the log file and cache key.
        log_data: A dictionary containing the log data to be saved. It's recommended
                  that 'day_start_timestamp_utc', if present, is a datetime object or
                  an ISO 8601 string. It will be stored as a datetime object in cache.

    Returns:
        True if the data was saved successfully, False otherwise (currently always True
        as _save_log_to_file relies on exceptions for errors).
    """
    if not isinstance(target_date, datetime):
        logger.error("DailyLogManager.save_daily_log_data: target_date must be a datetime object.")
        # Potentially raise TypeError or return False, but for now logging and proceeding.
        # Depending on strictness, one might want to return False here.
        # For now, let's allow it to proceed to see if strftime works or fails.
        pass # Or raise TypeError("target_date must be a datetime object")

    date_str = target_date.strftime("%Y-%m-%d")
    date_specific_lock = _get_lock_for_date_str(date_str)

    with date_specific_lock:
        log_path = get_log_file_path(target_date)
        
        # _save_log_to_file handles conversion of datetime to string for JSON.
        _save_log_to_file(log_path, log_data)

        # The internal module-level cache update logic has been removed to
        # ensure data freshness when multiple processes are involved.
        
    return True

def _update_absolute_times_in_entry(entry_data: Dict[str, Any], day_start_time_utc: datetime, audio_chunk_duration_minutes_float: float):
    """Helper to update absolute timestamps within a single log entry's processed_data."""
    file_seq_num = entry_data.get("file_sequence_number")
    if not (isinstance(file_seq_num, int) and file_seq_num > 0):
        logger.debug(f"Skipping absolute time update for entry {entry_data.get('original_file_name', 'N/A')} due to invalid/missing file_sequence_number: {file_seq_num}")
        entry_data["audio_chunk_start_utc"] = None
        entry_data["audio_chunk_end_utc"] = None
        return
    if not day_start_time_utc: 
        logger.warning(f"Day start time UTC not provided for entry {entry_data.get('original_file_name', 'N/A')}. Cannot calculate absolute timestamps.")
        entry_data["audio_chunk_start_utc"] = None 
        entry_data["audio_chunk_end_utc"] = None
        return

    offset_total_seconds = (file_seq_num - 1) * audio_chunk_duration_minutes_float * 60.0
    current_chunk_start_time_utc = day_start_time_utc + timedelta(seconds=offset_total_seconds)

    entry_data["audio_chunk_start_utc"] = current_chunk_start_time_utc.isoformat()
    current_chunk_end_time_utc = current_chunk_start_time_utc + timedelta(seconds=audio_chunk_duration_minutes_float * 60.0)
    entry_data["audio_chunk_end_utc"] = current_chunk_end_time_utc.isoformat()

    processed_data = entry_data.get("processed_data", {})
    wlt_key = "word_level_transcript_with_absolute_times" 

    if wlt_key in processed_data and isinstance(processed_data[wlt_key], list):
        for segment in processed_data[wlt_key]: 
            if isinstance(segment, dict):
                if 'start' in segment and isinstance(segment['start'], (int, float)):
                    abs_start_dt = current_chunk_start_time_utc + timedelta(seconds=segment['start'])
                    segment['absolute_start_utc'] = abs_start_dt.isoformat()
                if 'end' in segment and isinstance(segment['end'], (int, float)):
                    abs_end_dt = current_chunk_start_time_utc + timedelta(seconds=segment['end'])
                    segment['absolute_end_utc'] = abs_end_dt.isoformat()
                
                if 'words' in segment and isinstance(segment['words'], list):
                    for word_item in segment['words']: 
                        if isinstance(word_item, dict):
                            if 'start' in word_item and isinstance(word_item['start'], (int, float)):
                                abs_word_start_dt = current_chunk_start_time_utc + timedelta(seconds=word_item['start'])
                                word_item['absolute_start_utc'] = abs_word_start_dt.isoformat()
                            if 'end' in word_item and isinstance(word_item['end'], (int, float)):
                                abs_word_end_dt = current_chunk_start_time_utc + timedelta(seconds=word_item['end'])
                                word_item['absolute_end_utc'] = abs_word_end_dt.isoformat()

def set_day_start_time(start_time: datetime, target_date: Optional[datetime] = None) -> bool:
    """Sets the official start time for the day's recordings and updates existing entries. Thread-safe."""
    target_date_obj = target_date or datetime.now(timezone.utc) # If target_date is None, it's for "today's" log (UTC)
    date_str = target_date_obj.strftime("%Y-%m-%d")
    date_specific_lock = _get_lock_for_date_str(date_str)
    with date_specific_lock:
        log_path = get_log_file_path(target_date_obj)
        log_data = _fetch_or_load_daily_log_data_locked(date_str, target_date_obj)

        if start_time.tzinfo is None or start_time.tzinfo.utcoffset(start_time) is None:
            logger.warning("set_day_start_time received a naive datetime. Assuming UTC.")
            start_time_utc = start_time.replace(tzinfo=timezone.utc)
        elif start_time.tzinfo != timezone.utc:
            start_time_utc = start_time.astimezone(timezone.utc)
        else:
            start_time_utc = start_time

        log_data["day_start_timestamp_utc"] = start_time_utc
        _day_start_times_cache[date_str] = start_time_utc 
        
        logger.info(f"Day start time for {date_str} set to: {start_time_utc.isoformat()}. Re-calculating absolute timestamps for existing entries...")

        config = get_config()
        duration_config_val = config.get('timings', {}).get('audio_chunk_expected_duration', "10m")
        audio_chunk_duration_minutes_float = parse_duration_to_minutes(duration_config_val)
        
        updated_entries_count = 0
        for entry_data in log_data.get("chunks", {}).values():
            _update_absolute_times_in_entry(entry_data, start_time_utc, audio_chunk_duration_minutes_float)
            updated_entries_count +=1
                
        if updated_entries_count > 0:
            logger.info(f"Updated absolute timestamps for {updated_entries_count} existing entries in log for {date_str}.")
        
        _save_log_to_file(log_path, log_data)
    return True

def get_day_start_time(target_date: Optional[datetime] = None) -> Optional[datetime]:
    """Gets the day's official start time. Returns UTC datetime object or None. Thread-safe."""
    target_date_obj = target_date or datetime.now(timezone.utc)
    date_str = target_date_obj.strftime("%Y-%m-%d")
    date_specific_lock = _get_lock_for_date_str(date_str)
    with date_specific_lock:
        if date_str in _day_start_times_cache:
            cached_time = _day_start_times_cache[date_str]
            if isinstance(cached_time, str): 
                logger.warning(f"Day start time for {date_str} was a string in _day_start_times_cache. Attempting conversion.")
                try:
                    dt_str_cache = cached_time
                    if dt_str_cache.endswith("Z"): dt_str_cache = dt_str_cache[:-1] + "+00:00"
                    dt = datetime.fromisoformat(dt_str_cache)
                    _day_start_times_cache[date_str] = dt 
                    return dt
                except ValueError:
                    logger.error(f"Could not convert cached day_start_time string '{cached_time}' to datetime.")
                    _day_start_times_cache[date_str] = None 
            elif isinstance(cached_time, datetime) or cached_time is None:
                return cached_time
            else: 
                logger.error(f"Invalid type in _day_start_times_cache for {date_str}: {type(cached_time)}. Resetting.")
                _day_start_times_cache[date_str] = None

        log_data = _fetch_or_load_daily_log_data_locked(date_str, target_date_obj)
        day_start_dt = log_data.get("day_start_timestamp_utc") 
        
        if isinstance(day_start_dt, datetime) or day_start_dt is None:
            _day_start_times_cache[date_str] = day_start_dt 
            return day_start_dt
        
        if isinstance(day_start_dt, str):
            logger.warning(f"Day start time for {date_str} was a string in log_data after fetch. Attempting conversion.")
            try:
                dt_str_log = day_start_dt
                if dt_str_log.endswith("Z"): dt_str_log = dt_str_log[:-1] + "+00:00"
                converted_dt = datetime.fromisoformat(dt_str_log)
                _day_start_times_cache[date_str] = converted_dt
                # The primary data cache (_daily_log_data_cache) has been removed.
                # The in-memory log_data object is now temporary and does not need to be updated.
                return converted_dt
            except ValueError:
                logger.error(f"Could not convert day_start_timestamp_utc string '{day_start_dt}' from log data for {date_str}.")
                _day_start_times_cache[date_str] = None
                return None
        
        logger.error(f"Unexpected type for day_start_timestamp_utc in log_data for {date_str}: {type(day_start_dt)}")
        _day_start_times_cache[date_str] = None
        return None

def check_if_sequence_processed(sequence_number: int, target_date: Optional[datetime] = None) -> bool:
    """
    Checks the daily log to see if a given sequence number has already been processed. Thread-safe.
    This is a critical function for preventing duplicate processing.
    Args:
        sequence_number: The sequence number to check for.
        target_date: The UTC date for the log file to check. Defaults to today's UTC date.

    Returns:
        True if an entry with the sequence number exists, False otherwise.
    """
    if not isinstance(sequence_number, int) or sequence_number <= 0:
        logger.warning(f"check_if_sequence_processed called with invalid sequence number: {sequence_number}. Returning False.")
        return False

    target_date_obj = target_date or datetime.combine(get_samson_today(), datetime.min.time())
    date_str = target_date_obj.strftime("%Y-%m-%d")
    date_specific_lock = _get_lock_for_date_str(date_str)

    with date_specific_lock:
        log_data = _fetch_or_load_daily_log_data_locked(date_str, target_date_obj)
        
        # Using a generator expression with any() is efficient
        return any(
            entry.get("file_sequence_number") == sequence_number
            for entry in log_data.get("chunks", {}).values()
            if isinstance(entry, dict)
        )

def get_highest_processed_sequence(target_date: Optional[datetime] = None) -> int:
    """
    Gets the highest sequence number that has been successfully logged for a given day.
    Used by the folder monitor to know where to resume processing from on startup. Thread-safe.
    Args:
        target_date: The UTC date for the log file to check. Defaults to today's UTC date.

    Returns:
        The highest sequence number found, or 0 if no entries exist for that day.
    """
    target_date_obj = target_date or datetime.combine(get_samson_today(), datetime.min.time())
    date_str = target_date_obj.strftime("%Y-%m-%d")
    date_specific_lock = _get_lock_for_date_str(date_str)

    with date_specific_lock:
        log_data = _fetch_or_load_daily_log_data_locked(date_str, target_date_obj)
        
        highest_seq = 0
        for entry in log_data.get("chunks", {}).values():
            if isinstance(entry, dict):
                seq_num = entry.get("file_sequence_number")
                if isinstance(seq_num, int):
                    highest_seq = max(highest_seq, seq_num)
                    
        return highest_seq

def add_processed_audio_entry(
    original_file_path: Path,
    processing_result: Dict[str, Any],
    target_date_for_log: datetime,
    chunk_id: str
):
    """
    Adds an entry for a processed audio file to the daily log for the specified date.
    The caller is responsible for determining the correct target_date_for_log and a unique chunk_id.
    """
    date_str = target_date_for_log.strftime("%Y-%m-%d")
    date_specific_lock = _get_lock_for_date_str(date_str)
    with date_specific_lock:
        log_path = get_log_file_path(target_date_for_log)
        log_data = _fetch_or_load_daily_log_data_locked(date_str, target_date_for_log)
        
        config = get_config()
        duration_config_val = config.get('timings', {}).get('audio_chunk_expected_duration', "10m")
        audio_chunk_duration_minutes_float = parse_duration_to_minutes(duration_config_val)

        day_start_time_utc_for_this_entrys_day = get_day_start_time(target_date_for_log)

        file_sequence_num: Optional[int] = extract_sequence_number_from_filename(original_file_path.stem)
        if file_sequence_num is None:
            # Replicate the logic of get_highest_processed_sequence here to avoid deadlocks
            highest_seq = 0
            for entry in log_data.get("chunks", {}).values():
                if isinstance(entry, dict):
                    seq_num = entry.get("file_sequence_number")
                    if isinstance(seq_num, int):
                        highest_seq = max(highest_seq, seq_num)
            file_sequence_num = highest_seq + 1
            logger.info(f"Dynamically assigned sequence number {file_sequence_num} to file '{original_file_path.name}' for date {date_str}.")
        
        log_entry = {
            "chunk_id": chunk_id,
            "entry_creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "original_file_name": original_file_path.name,
            "file_sequence_number": file_sequence_num if file_sequence_num is not None and file_sequence_num > 0 else None,
            "audio_chunk_start_utc": None, 
            "audio_chunk_end_utc": None,
            "matter_segments": processing_result.get('matter_segments', []),
            "processed_data": {
                "word_level_transcript_with_absolute_times": processing_result.get("identified_transcript", []),
                "speaker_turns_final_text": processing_result.get("speaker_turns_processed_text", [])
            },
            "source_job_output_dir": str(processing_result.get("job_output_dir")),
            "processing_date_utc": target_date_for_log.strftime('%Y-%m-%d'),
        }

        if isinstance(day_start_time_utc_for_this_entrys_day, datetime) and \
           file_sequence_num is not None and file_sequence_num > 0:
            _update_absolute_times_in_entry(log_entry, day_start_time_utc_for_this_entrys_day, audio_chunk_duration_minutes_float)
        else:
            if not isinstance(day_start_time_utc_for_this_entrys_day, datetime): 
                logger.warning(f"DailyLogManager: Day start time for {date_str} not set. Absolute timestamps in log for {original_file_path.name} will be missing until set.")
            if not (file_sequence_num is not None and file_sequence_num > 0):
                logger.warning(f"DailyLogManager: Could not determine valid sequence number for {original_file_path.name}. Timestamps might be affected.")
                
        log_data.setdefault("chunks", {})[chunk_id] = log_entry
        _save_log_to_file(log_path, log_data)
    logger.info(f"Added entry for '{original_file_path.name}' (chunk_id: {chunk_id}) to daily log {log_path.name}.")

def update_chunk_metadata(target_date: datetime, chunk_id: str, new_data: Dict[str, Any]) -> bool:
    """Atomically updates a specific chunk in a daily log."""
    log_file_path = get_log_file_path(target_date)
    date_str = target_date.strftime("%Y-%m-%d")
    date_specific_lock = _get_lock_for_date_str(date_str)
    with date_specific_lock:
        # NOTE: The _fetch_or_load_daily_log_data_locked function expects a datetime object, not a path.
        # It correctly gets the path internally via get_log_file_path. So we pass target_date.
        log_data = _fetch_or_load_daily_log_data_locked(date_str, target_date)
        if "chunks" in log_data and chunk_id in log_data["chunks"]:
            log_data["chunks"][chunk_id].update(new_data)
            logger.info(f"Updated metadata for chunk {chunk_id} in log for {date_str}.")
            _save_log_to_file(log_file_path, log_data)
            return True
        else:
            logger.warning(f"Chunk ID {chunk_id} not found in log for {date_str}. Cannot update.")
            return False

def update_matter_segments_for_chunk(target_date: datetime, chunk_id: str, start_time: float, end_time: float, new_matter_id: Optional[str]) -> bool:
    """
    Surgically updates the matter_segments for a specific time range within a chunk.
    """
    date_str = target_date.strftime("%Y-%m-%d")
    lock = _get_lock_for_date_str(date_str)
    with lock:
        log_data = _fetch_or_load_daily_log_data_locked(date_str, target_date)
        if not log_data or "chunks" not in log_data:
            logger.error(f"No log data found for date {date_str} to update matter segments.")
            return False

        chunk = log_data["chunks"].get(chunk_id)
        if not chunk:
            logger.error(f"Chunk ID {chunk_id} not found in log for date {date_str}.")
            return False

        original_segments = chunk.get("matter_segments", [])
        logger.debug("Updating segments for chunk '%s'. Original segments: %s", chunk_id, json.dumps(original_segments, indent=2))
        logger.debug("Update details: start=%.2f, end=%.2f, new_matter_id='%s'", start_time, end_time, new_matter_id)
        
        new_segments = []
        TOLERANCE = 0.01

        # 1. Shatter existing segments that overlap with the update range
        for existing_segment in original_segments:
            ex_start = existing_segment['start_time']
            ex_end = existing_segment['end_time']
            ex_id = existing_segment['matter_id']

            # Case 1: No overlap (segment is completely before or after the update range)
            if ex_end <= start_time + TOLERANCE or ex_start >= end_time - TOLERANCE:
                new_segments.append(existing_segment)
                continue

            # Case 2: Overlap exists, shatter the segment
            # Add the part before the update, if it exists
            if ex_start < start_time - TOLERANCE:
                new_segments.append({'start_time': ex_start, 'end_time': start_time, 'matter_id': ex_id})

            # Add the part after the update, if it exists
            if ex_end > end_time + TOLERANCE:
                new_segments.append({'start_time': end_time, 'end_time': ex_end, 'matter_id': ex_id})
        
        # 2. Add the new segment if it's not a deletion operation
        if new_matter_id is not None:
            new_segments.append({'start_time': start_time, 'end_time': end_time, 'matter_id': new_matter_id})

        # 3. Sort by start time to prepare for merging
        new_segments.sort(key=lambda s: s['start_time'])

        # 4. Merge adjacent segments that have the same matter_id
        if not new_segments:
            merged_segments = []
        else:
            merged_segments = [new_segments[0]]
            for current_segment in new_segments[1:]:
                last_segment = merged_segments[-1]
                
                # Check for merge condition: same ID and adjacent/overlapping times
                if (current_segment['matter_id'] == last_segment['matter_id'] and
                    current_segment['start_time'] - last_segment['end_time'] < TOLERANCE):
                    # Merge by extending the end time of the last segment
                    last_segment['end_time'] = max(last_segment['end_time'], current_segment['end_time'])
                else:
                    # No merge, so append as a new segment
                    merged_segments.append(current_segment)

        # 5. Filter out any tiny or invalid segments that might have been created by float inaccuracies
        final_segments = [
            s for s in merged_segments if (s['end_time'] - s['start_time']) > 0
        ]
        words_updated_count = 0
        wlt = chunk.get("processed_data", {}).get("word_level_transcript_with_absolute_times", [])
        for segment in wlt:
            if isinstance(segment, dict):
                for word in segment.get("words", []):
                    if isinstance(word, dict):
                        word_start = word.get("start")
                        # Check if the word's start time is within the updated span
                        if word_start is not None and start_time <= word_start < end_time:
                            word['matter_id'] = new_matter_id
                            words_updated_count += 1
        
        if words_updated_count > 0:
            logger.info(f"Consistently updated matter_id for {words_updated_count} individual words in chunk '{chunk_id}'.")
        # 6. Update the chunk in the log data and save the entire file
        chunk["matter_segments"] = final_segments
        logger.debug("Saving updated segments for chunk '%s'. New segments: %s", chunk_id, json.dumps(new_segments, indent=2))
        log_path = get_log_file_path(target_date)
        _save_log_to_file(log_path, log_data)

        # Log this manual correction as a feedback event
        log_matter_correction(
            "gui_manual_correction",
            {
                "new_matter_id": new_matter_id,
                "source_chunk_id": chunk_id,
                "start_time_s": start_time,
                "end_time_s": end_time,
                "source": "gui_transcript_editor_matter_update"
            }
        )
    return True

def relabel_single_word(
    target_date_utc: datetime,
    chunk_id: str,
    segment_index: int,
    word_index: int,
    new_speaker_id: str,
    new_speaker_faiss_id: int,
    log_feedback: bool = True
) -> bool:
    """
    Atomically finds and relabels a single word in a daily log file based on its
    exact coordinates, recalculates the parent segment's speaker, and logs the
    correction as a feedback event.
    Args:
        target_date_utc: The UTC date of the log to modify.
        chunk_id: The ID of the chunk within the log's "chunks" dictionary.
        segment_index: The index of the segment within the entry's word-level transcript.
        word_index: The index of the word within the segment's "words" list.
        new_speaker_id: The new speaker name to assign to the word.
        new_speaker_faiss_id: The FAISS ID of the new speaker, for feedback logging.
        log_feedback: Whether to log the change as a feedback event.

    Returns:
        True if the word was successfully found and relabeled, False otherwise.
    """
    date_str = target_date_utc.strftime("%Y-%m-%d")
    date_specific_lock = _get_lock_for_date_str(date_str)

    with date_specific_lock:
        log_data = _fetch_or_load_daily_log_data_locked(date_str, target_date_utc)
        
        try:
            # --- Navigation and Validation ---
            target_entry = log_data['chunks'][chunk_id]
            wlt = target_entry['processed_data']['word_level_transcript_with_absolute_times']
            target_segment = wlt[segment_index]
            target_word = target_segment['words'][word_index]

            original_speaker = target_word.get('speaker', 'UNKNOWN_SPEAKER')
            if original_speaker == new_speaker_id:
                logger.info(f"Relabel word: No change needed. Word at [{chunk_id}][{segment_index}][{word_index}] on {date_str} already has speaker '{new_speaker_id}'.")
                return True

            # --- Atomic Change in Log File ---
            logger.info(f"Relabeling word at [{chunk_id}][{segment_index}][{word_index}] on {date_str} from '{original_speaker}' to '{new_speaker_id}'.")
            target_word['speaker'] = new_speaker_id

            # --- Recalculate Parent Segment Speaker ---
            all_segment_words = target_segment.get('words', [])
            if all_segment_words:
                speaker_counts = Counter(w.get('speaker', 'UNKNOWN_SPEAKER') for w in all_segment_words)
                dominant_speaker = speaker_counts.most_common(1)[0][0]
                if target_segment.get('speaker') != dominant_speaker:
                    logger.debug(f"Segment speaker updated from '{target_segment.get('speaker')}' to dominant speaker '{dominant_speaker}'.")
                    target_segment['speaker'] = dominant_speaker
            
            # --- Save the modified log file ---
            _save_log_to_file(get_log_file_path(target_date_utc), log_data)
            # The in-memory cache update has been removed to ensure freshness.
            
            # --- Log Feedback Event ---
            if log_feedback:
                feedback_details = {
                    "faiss_id_of_correct_speaker": new_speaker_faiss_id,
                    "original_speaker_id": original_speaker,
                    "corrected_speaker_id": new_speaker_id,
                    "source": "gui_transcript_editor", # Assume GUI is the source for this surgical tool
                    "context": {
                        "log_date": date_str,
                        "chunk_id": chunk_id,
                        "segment_index": segment_index,
                        "word_index": word_index,
                        "word_text": target_word.get('word', 'N/A'),
                        "relative_start_s": target_word.get('start', 'N/A')
                    }
                }
                log_correction_feedback("manual_relabel", feedback_details)

            return True

        except (IndexError, KeyError, TypeError) as e:
            logger.error(f"Failed to relabel word at coordinates [{chunk_id}][{segment_index}][{word_index}] on {date_str}: Invalid path or data structure. Error: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during relabel_single_word for {date_str}: {e}", exc_info=True)
            return False

def modify_word_span_and_speaker(
    corrections: List[Dict[str, Any]],
    new_speaker_name: Optional[str],
    new_text: str,
    correction_context: str, # Context like 'in_person' or 'voip'
    correction_source: str, # Source like 'gui_transcript_editor'
    target_date_str: str
) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    """
    Atomically finds a span of words across one or more segments, merges the affected
    segments, and replaces the text content by interpolating timestamps. It also
    updates the speaker if a new name is provided.
    Args:
        corrections: The list of word objects selected in the GUI.
        new_speaker_name: The new speaker to assign. If None, the original speaker is kept.
        new_text: The new block of text to replace the selection with.
        correction_context: The context of the correction (e.g., 'in_person').
        correction_source: The origin of the correction call.
        target_date_str: The date string for the log ('YYYYMMDD').

    Returns:
        A tuple of (success, original_words_list, new_words_list).
        On success, lists contain data for stats updates. On failure, they are None.
    """
    if not corrections:
        logger.error("modify_word_span_and_speaker called with no corrections.")
        return False, None, None

    if new_text is None:
        logger.error("modify_word_span_and_speaker requires a 'new_text' value (use empty string for deletion).")
        return False, None, None

    chunk_id = corrections[0].get("chunk_id")
    if not chunk_id:
        logger.error("Cannot modify word span: chunk_id missing from correction data.")
        return False, None, None
        
    try:
        target_date = datetime.strptime(target_date_str, '%Y%m%d').replace(tzinfo=timezone.utc)
    except ValueError:
        logger.error(f"Invalid date string format: {target_date_str}. Expected YYYYMMDD.")
        return False, None, None

    date_lock_str = target_date.strftime("%Y-%m-%d")
    date_specific_lock = _get_lock_for_date_str(date_lock_str)

    with date_specific_lock:
        log_data = _fetch_or_load_daily_log_data_locked(date_lock_str, target_date)
        try:
            target_entry = log_data['chunks'][chunk_id]
            wlt = target_entry['processed_data']['word_level_transcript_with_absolute_times']
            initial_segment_count = len(wlt)
            chunk_start_utc_str = target_entry.get("audio_chunk_start_utc")
            if not chunk_start_utc_str:
                logger.error(f"Cannot perform text/speaker modification: The chunk '{chunk_id}' is missing its 'audio_chunk_start_utc'. Aborting.")
                return False, None, None
            chunk_start_time_utc = datetime.fromisoformat(chunk_start_utc_str.replace('Z', '+00:00'))
            
            affected_seg_indices = sorted(list({corr['segment_index'] for corr in corrections}))
            affected_segments = [wlt[i] for i in affected_seg_indices]
            corrected_word_coords = {(corr['segment_index'], corr['word_index']) for corr in corrections}
            
            unaffected_words = [
                word for i, seg in enumerate(affected_segments)
                for j, word in enumerate(seg.get('words', []))
                if (affected_seg_indices[i], j) not in corrected_word_coords
            ]
            
            original_words_data = sorted([corr['word_data'] for corr in corrections], key=lambda w: w.get('start', float('inf')))
            start_time = original_words_data[0]['start']
            end_time = original_words_data[-1]['end']
            total_duration = max(0.0, end_time - start_time)
            
            final_speaker = new_speaker_name if new_speaker_name else original_words_data[0].get('speaker', 'UNKNOWN_SPEAKER')
            
            new_word_list = new_text.strip().split()
            new_word_objects = []
            num_new_words = len(new_word_list)

            logger.info(f"Modifying word span in chunk {chunk_id}: Replacing {len(original_words_data)} words with {num_new_words} words.")
            logger.debug(f"  - Original span: start={start_time:.3f}s, end={end_time:.3f}s, duration={total_duration:.3f}s.")
            logger.debug(f"  - New text: '{new_text}'")

            if num_new_words > 0:
                if num_new_words == len(original_words_data):
                    logger.info("  - Word count is identical. Applying 1-to-1 timestamp mapping for maximum accuracy.")
                    for i, word_str in enumerate(new_word_list):
                        original_word = original_words_data[i]
                        word_start_time = original_word['start']
                        word_end_time = original_word['end']

                        if word_end_time <= word_start_time:
                             word_end_time = word_start_time + 0.01 # Add a minimal 10ms duration
                             logger.warning(f"  - Adjusted end time for mapped word '{word_str}' to prevent zero/negative duration. Start: {word_start_time}, New End: {word_end_time}")

                        new_word_objects.append({
                            "word": word_str, "start": word_start_time, "end": word_end_time,
                            "absolute_start_utc": (chunk_start_time_utc + timedelta(seconds=word_start_time)).isoformat(),
                            "absolute_end_utc": (chunk_start_time_utc + timedelta(seconds=word_end_time)).isoformat(),
                            "speaker": final_speaker, "probability": 1.0,
                            "source_of_correction": correction_source,
                            "correction_context": correction_context
                        })
                        logger.debug(f"  - Mapped new word '{word_str}': start={word_start_time:.3f}, end={word_end_time:.3f}")
                else:
                    logger.info("  - Word count differs. Applying proportional timestamp distribution based on character length.")
                    total_chars = sum(len(w) for w in new_word_list)

                    if total_chars > 0 and total_duration > 0:
                        time_per_char = total_duration / total_chars
                        logger.debug(f"  - Calculated time per character: {time_per_char:.6f}s")

                        current_word_start_time = start_time
                        for i, word_str in enumerate(new_word_list):
                            word_duration = len(word_str) * time_per_char
                            word_end_time = current_word_start_time + word_duration

                            # CRITICAL: Clamp the final word to the exact end time to prevent float drift
                            if i == num_new_words - 1:
                                word_end_time = end_time

                            word_start_time_rounded = round(current_word_start_time, 3)
                            word_end_time_rounded = round(word_end_time, 3)

                            # Robustness: Ensure duration is positive
                            if word_end_time_rounded <= word_start_time_rounded:
                                word_end_time_rounded = word_start_time_rounded + 0.01
                                logger.warning(f"  - Adjusted end time for new word '{word_str}' to prevent zero/negative duration. Start: {word_start_time_rounded:.3f}, New End: {word_end_time_rounded:.3f}")

                            new_word_objects.append({
                                "word": word_str, "start": word_start_time_rounded, "end": word_end_time_rounded,
                                "absolute_start_utc": (chunk_start_time_utc + timedelta(seconds=word_start_time_rounded)).isoformat(),
                                "absolute_end_utc": (chunk_start_time_utc + timedelta(seconds=word_end_time_rounded)).isoformat(),
                                "speaker": final_speaker, "probability": 1.0,
                                "source_of_correction": correction_source,
                                "correction_context": correction_context
                            })
                            logger.debug(f"  - Created new word '{word_str}': start={word_start_time_rounded:.3f}, end={word_end_time_rounded:.3f}")

                            current_word_start_time = word_end_time # Use unrounded for next iteration
                    else:
                        # Fallback for zero-length text or zero-duration span
                        logger.warning("  - Proportional distribution not possible (total_chars or total_duration is 0). Falling back to linear distribution.")
                        duration_per_word = total_duration / num_new_words if num_new_words > 0 else 0
                        for i, word_str in enumerate(new_word_list):
                            word_start_time_fallback = round(start_time + (i * duration_per_word), 3)
                            word_end_time_fallback = round(min(start_time + ((i + 1) * duration_per_word), end_time), 3)
                            if word_end_time_fallback <= word_start_time_fallback:
                                word_end_time_fallback = word_start_time_fallback + 0.01

                            new_word_objects.append({
                                "word": word_str, "start": word_start_time_fallback, "end": word_end_time_fallback,
                                "absolute_start_utc": (chunk_start_time_utc + timedelta(seconds=word_start_time_fallback)).isoformat(),
                                "absolute_end_utc": (chunk_start_time_utc + timedelta(seconds=word_end_time_fallback)).isoformat(),
                                "speaker": final_speaker, "probability": 1.0,
                                "source_of_correction": correction_source,
                                "correction_context": correction_context
                            })
                            logger.debug(f"  - Created new word (linear fallback) '{word_str}': start={word_start_time_fallback:.3f}, end={word_end_time_fallback:.3f}")
            else: # Case for deletion
                logger.info("  - New text is empty. Deleting selected word span.")

            final_word_list = sorted(unaffected_words + new_word_objects, key=lambda w: w.get('start', float('inf')))
            
            merged_seg_start_rel = final_word_list[0]['start'] if final_word_list else affected_segments[0]['start']
            merged_seg_end_rel = final_word_list[-1]['end'] if final_word_list else affected_segments[-1]['end']

            new_merged_segment = {
                "start": merged_seg_start_rel, "end": merged_seg_end_rel, "speaker": final_speaker,
                "text": " ".join(w['word'] for w in final_word_list), "words": final_word_list,
                "absolute_start_utc": (chunk_start_time_utc + timedelta(seconds=merged_seg_start_rel)).isoformat(),
                "absolute_end_utc": (chunk_start_time_utc + timedelta(seconds=merged_seg_end_rel)).isoformat(),
            }
            
            for seg_idx in sorted(affected_seg_indices, reverse=True):
                del wlt[seg_idx]
            
            wlt.insert(affected_seg_indices[0], new_merged_segment)

            final_segment_count = len(wlt)
            logger.info(f"Successfully modified text/speaker in chunk {chunk_id}, merging {len(affected_seg_indices)} segments into 1. Segment count changed from {initial_segment_count} to {final_segment_count}.")
            
            _save_log_to_file(get_log_file_path(target_date), log_data)
            
            #from main_orchestrator import ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE
            #ORCHESTRATOR_DAYS_FOR_REGEN_QUEUE.add(target_date.date())
            
            return True, original_words_data, new_word_objects

        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.error(f"Error modifying text/speaker in chunk {chunk_id}: {e}", exc_info=True)
            return False, None, None

def relabel_speaker_id_in_daily_log(
    target_date_of_log: datetime,
    old_speaker_id: str,
    new_speaker_name: str
) -> bool:
    """
    Finds all occurrences of an old_speaker_id in a specific daily log and
    replaces them with new_speaker_name. This is a global replace for the entire day.
    Args:
        target_date_of_log: The date of the log to modify.
        old_speaker_id: The speaker ID to find and replace (e.g., "CUSID_...").
        new_speaker_name: The new, correct speaker name to assign.

    Returns:
        True if changes were made and saved, False otherwise.
    """
    logger.info(f"Globally relabeling all occurrences of '{old_speaker_id}' to '{new_speaker_name}' for log date {target_date_of_log.strftime('%Y-%m-%d')}.")

    date_str = target_date_of_log.strftime("%Y-%m-%d")
    date_specific_lock = _get_lock_for_date_str(date_str)

    with date_specific_lock:
        log_data = _fetch_or_load_daily_log_data_locked(date_str, target_date_of_log)
        if not log_data or "chunks" not in log_data:
            logger.error(f"Failed to load or parse log data for {date_str} during relabeling.")
            return False

        update_count = 0
        # Iterate through all entries, all segments, all words
        for entry in log_data.get("chunks", {}).values():
            if not isinstance(entry, dict): continue
            processed_data = entry.get("processed_data", {})
            word_level_transcript = processed_data.get("word_level_transcript_with_absolute_times", [])

            for segment in word_level_transcript:
                if not isinstance(segment, dict): continue
                words_in_segment = segment.get("words", [])
                if not isinstance(words_in_segment, list): continue

                segment_changed = False
                for word_obj in words_in_segment:
                    if isinstance(word_obj, dict) and word_obj.get('speaker') == old_speaker_id:
                        word_obj['speaker'] = new_speaker_name
                        update_count += 1
                        segment_changed = True

                # If any word in the segment was changed, recalculate the segment's dominant speaker
                if segment_changed:
                    speaker_counts = Counter(w.get('speaker', 'UNKNOWN_SPEAKER') for w in words_in_segment)
                    if speaker_counts:
                        dominant_speaker = speaker_counts.most_common(1)[0][0]
                        if segment.get('speaker') != dominant_speaker:
                            logger.debug(f"Recalculating segment speaker for seg at {segment.get('start', 'N/A'):.2f}s to '{dominant_speaker}'.")
                            segment['speaker'] = dominant_speaker

        if update_count > 0:
            logger.info(f"Successfully relabeled {update_count} word instances from '{old_speaker_id}' to '{new_speaker_name}'. Saving daily log for {date_str}.")
            # Save the fully modified log file
            log_path = get_log_file_path(target_date_of_log)
            _save_log_to_file(log_path, log_data)
            # The in-memory cache update has been removed to ensure freshness.
            return True
        else:
            logger.warning(f"Did not find any occurrences of speaker '{old_speaker_id}' to relabel in log for {date_str}.")
            return False

def update_speaker_label_in_daily_log(
    log_entry_original_file_name: str,
    segment_relative_start_s: float,
    segment_relative_end_s: Optional[float],
    original_speaker_to_replace: str,
    new_speaker_name: str,
    target_date_of_log: datetime
) -> bool:
    """
    DEPRECATED for CUSID resolution. Use relabel_speaker_id_in_daily_log instead.
    Updates speaker labels for all words within a given time range that match the original speaker.
    This is time-range based, not global.
    """
    logger.info(f"Entering update_speaker_label_in_daily_log. Target: '{log_entry_original_file_name}', Range: [{segment_relative_start_s:.2f}-{segment_relative_end_s if segment_relative_end_s is not None else 'N/A'}], Replace '{original_speaker_to_replace}' -> '{new_speaker_name}'")
    time_tolerance_s = 0.05 # Increased tolerance slightly for float comparisons
    if target_date_of_log.tzinfo is None:
        logger.warning(f"update_speaker_label_in_daily_log received a naive datetime for target_date_of_log. Assuming UTC date part: {target_date_of_log.date()}")

    date_str = target_date_of_log.strftime("%Y-%m-%d")
    date_specific_lock = _get_lock_for_date_str(date_str)

    with date_specific_lock:
        log_data = _fetch_or_load_daily_log_data_locked(date_str, target_date_of_log)
        if not log_data or "chunks" not in log_data:
            logger.error(f"Failed to load or parse log data for {date_str}")
            return False

        found_entry_data: Optional[Dict[str, Any]] = None
        original_file_stem = Path(log_entry_original_file_name).stem
        for entry in log_data["chunks"].values():
            if Path(entry.get("original_file_name", "")).stem == original_file_stem:
                found_entry_data = entry
                break

        if not found_entry_data:
            logger.warning(f"Entry with original_file_stem '{original_file_stem}' not found in log for {date_str}.")
            return False

        wlt_key = "word_level_transcript_with_absolute_times"
        processed_data = found_entry_data.get("processed_data", {})
        if wlt_key not in processed_data or not isinstance(processed_data.get(wlt_key), list):
            logger.warning(f"'{wlt_key}' not found or not a list in entry '{found_entry_data.get('original_file_name')}'.")
            return False

        word_level_transcript: List[Dict[str, Any]] = processed_data[wlt_key]
        words_updated_count = 0

        # --- REWRITTEN LOGIC ---
        # Loop through all segments and all words to find ones within the time range
        for segment in word_level_transcript:
            if not isinstance(segment, dict) or 'words' not in segment or not isinstance(segment['words'], list):
                continue
            
            for word_obj in segment['words']:
                if not isinstance(word_obj, dict): continue

                word_start = word_obj.get('start')
                word_speaker = word_obj.get('speaker')

                if not isinstance(word_start, (float, int)) or word_speaker is None:
                    continue

                # Check if word's start time is within the target range
                # and if it belongs to the speaker we are trying to replace.
                # If segment_relative_end_s is None, we match only on start time, for robustness.
                is_within_time_range = (
                    (segment_relative_start_s - time_tolerance_s) <= word_start and
                    (segment_relative_end_s is None or word_start < (segment_relative_end_s + time_tolerance_s))
                )

                if is_within_time_range and word_speaker == original_speaker_to_replace:
                    word_obj['speaker'] = new_speaker_name
                    words_updated_count += 1
                    logger.debug(f"Relabeled word '{word_obj.get('word')}' at time {word_start:.2f} from '{original_speaker_to_replace}' to '{new_speaker_name}'.")

        if words_updated_count > 0:
            logger.info(f"Updated {words_updated_count} words to speaker '{new_speaker_name}' in entry '{found_entry_data.get('original_file_name')}'.")

            # After updating words, recalculate the dominant speaker for each segment
            for segment in word_level_transcript:
                 if 'words' in segment and isinstance(segment['words'], list) and segment['words']:
                    speaker_counts = Counter(w.get('speaker', 'UNKNOWN_SPEAKER') for w in segment['words'])
                    if speaker_counts:
                        # Set segment speaker to the most common one in its words
                        dominant_speaker = speaker_counts.most_common(1)[0][0]
                        if segment.get('speaker') != dominant_speaker:
                            logger.debug(f"Recalculating segment speaker for seg at {segment.get('start', 'N/A'):.2f}s to '{dominant_speaker}'.")
                            segment['speaker'] = dominant_speaker
            
            # Now, save the modified log data
            log_path = get_log_file_path(target_date_of_log)
            logger.info(f"Calling _save_log_to_file for path: {log_path} after speaker update.")
            _save_log_to_file(log_path, log_data)
            # The in-memory cache update has been removed to ensure freshness.
            return True
        else:
            logger.warning(f"No words matching speaker '{original_speaker_to_replace}' found within time range [{segment_relative_start_s:.2f}-{segment_relative_end_s if segment_relative_end_s is not None else 'N/A'}] in entry '{found_entry_data.get('original_file_name')}'. No changes made.")
            return False

def get_all_dialogue_for_speaker_id(
    speaker_id: str,
    start_date_dt: datetime,
    end_date_dt: datetime
) -> List[Dict[str, Any]]:
    """
    Collects all dialogue segments attributed to a specific speaker_id across a date range.
    This function has been optimized to only scan dates where archived data exists, making
    it much faster for large date ranges.
    Args:
        speaker_id: The speaker ID (e.g., an enrolled name or a CUSID) to search for.
        start_date_dt: The start date of the range (inclusive).
        end_date_dt: The end date of the range (inclusive).

    Returns:
        A list of dictionaries, each representing a dialogue segment and containing all
        necessary metadata for re-auditing and re-embedding.
    """
    if not speaker_id:
        logger.warning("get_all_dialogue_for_speaker_id called with empty speaker_id.")
        return []
    if start_date_dt > end_date_dt:
        logger.warning(f"Start date {start_date_dt.date()} is after end date {end_date_dt.date()}. Returning empty list.")
        return []

    config = get_config()
    archive_folder_path_str = config.get('paths', {}).get('archived_audio_folder')
    if not archive_folder_path_str:
        logger.error("get_all_dialogue_for_speaker_id: 'archived_audio_folder' not configured. Cannot perform audit. Aborting.")
        return []

    archive_folder_path = Path(archive_folder_path_str)
    if not archive_folder_path.is_dir():
        logger.warning(f"get_all_dialogue_for_speaker_id: Archive folder not found at {archive_folder_path}. No historical data can be retrieved.")
        return []

    # --- NEW: Efficiently find dates with data ---
    active_dates_with_archives = []
    for item in archive_folder_path.iterdir():
        if item.is_dir():
            try:
                # Assumes directory name is 'YYYY-MM-DD'
                active_date = datetime.strptime(item.name, '%Y-%m-%d').date()
                active_dates_with_archives.append(active_date)
            except ValueError:
                logger.debug(f"Ignoring non-date directory in archive folder: {item.name}")
                continue

    active_dates_in_range = [
        d for d in active_dates_with_archives 
        if start_date_dt.date() <= d <= end_date_dt.date()
    ]

    if not active_dates_in_range:
        logger.info(f"No archived audio found within the specified date range: {start_date_dt.date()} to {end_date_dt.date()}")
        return []

    collected_dialogue_segments: List[Dict[str, Any]] = []
    logger.info(f"Searching for dialogue by speaker_id '{speaker_id}' in {len(active_dates_in_range)} active archive date(s) between {start_date_dt.strftime('%Y-%m-%d')} and {end_date_dt.strftime('%Y-%m-%d')}")

    # --- MODIFIED LOOP ---
    # The loop now iterates over a much smaller, pre-validated list of dates
    for current_date in sorted(active_dates_in_range):
        date_str_for_path = current_date.strftime('%Y-%m-%d')
        current_datetime_utc = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc)

        # File System Validation: Check log file (archive folder already confirmed to exist)
        log_path = get_log_file_path(current_datetime_utc)
        if not log_path.exists():
            logger.warning(f"Data inconsistency: Archive folder exists for {date_str_for_path}, but daily log file is missing. Skipping day.")
            continue

        # Safe File Loading and Parsing
        logger.debug(f"Parsing log file: {log_path.name}")
        log_data = _load_log_from_file(log_path) # Handles JSONDecodeError and returns default structure

        if not log_data or "chunks" not in log_data or not log_data["chunks"]:
            continue # Move to next day if log is empty or malformed

        # Correct Data Parsing
        for chunk_id, entry in log_data.get("chunks", {}).items():
            if not isinstance(entry, dict): continue

            processed_data = entry.get("processed_data", {})
            word_level_transcript = processed_data.get("word_level_transcript_with_absolute_times", [])

            for segment in word_level_transcript:
                if not isinstance(segment, dict): continue

                if segment.get("speaker") == speaker_id:
                    # Data Extraction and Validation
                    original_file_name = entry.get("original_file_name")
                    abs_start_utc = segment.get("absolute_start_utc")
                    rel_start_s = segment.get("start")
                    rel_end_s = segment.get("end")

                    if all([original_file_name, abs_start_utc, isinstance(rel_start_s, (float, int)), isinstance(rel_end_s, (float, int))]):
                        segment_words = segment.get("words", [])
                        collected_dialogue_segments.append({
                            "speaker_id": speaker_id,
                            "timestamp_utc": abs_start_utc,
                            "audio_file_stem": Path(original_file_name).stem,
                            "relative_start_s": rel_start_s,
                            "relative_end_s": rel_end_s,
                            "chunk_id": chunk_id,
                            "processing_date_of_log": date_str_for_path,
                            "segment_duration": rel_end_s - rel_start_s,  # Add segment duration for comparison
                            "word_count": len(segment_words),            # Add word count for validation
                            "words": segment_words                       # Add the full word list
                        })
                    else:
                        logger.debug(f"Skipping segment for speaker '{speaker_id}' in chunk '{chunk_id}' due to missing essential audit data (file_name, timestamps, etc.).")

    # Enhanced Logging at end
    if collected_dialogue_segments:
        logger.info(f"Found {len(collected_dialogue_segments)} dialogue segments for speaker_id '{speaker_id}'.")
    else:
        logger.warning(f"Found 0 dialogue segments for speaker_id '{speaker_id}'. This may lead to incorrect profile calculations or deletion.")

    return collected_dialogue_segments

def get_daily_flags_queue_path(target_date: Optional[datetime] = None) -> Path:
    """
    Constructs the path to the daily flags queue JSON file for a given date.
    This function is self-contained and reads from the config.
    """
    config = get_config()
    flags_queue_dir_str = config.get('paths', {}).get('flags_queue_dir')
    if not flags_queue_dir_str:
        from src.config_loader import PROJECT_ROOT
        logger.error("Configuration for 'paths.flags_queue_dir' is missing. Using fallback.")
        flags_queue_dir = PROJECT_ROOT / "data" / "flags_queue_fallback"
    else:
        flags_queue_dir = Path(flags_queue_dir_str)
    flags_queue_dir.mkdir(parents=True, exist_ok=True)

    date_obj = target_date if target_date else datetime.now(timezone.utc)
    filename = f"{date_obj.strftime('%Y-%m-%d')}_flags_queue.json"
    return flags_queue_dir / filename

def load_daily_flags_queue(target_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Loads the daily flags queue by calling its path helper."""
    path = get_daily_flags_queue_path(target_date)
    if not path.exists():
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return []
            f.seek(0)
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from flags queue: {path.name}", exc_info=True)
        # Backup corrupted file
        backup_path = path.with_suffix(f".corrupted.{int(time.time())}.json")
        try:
            shutil.copy(path, backup_path)
            logger.info(f"Backed up corrupted flags queue to {backup_path}")
        except Exception as e_backup:
            logger.error(f"Failed to backup corrupted flags queue {path.name}: {e_backup}")
        return []
    except Exception as e:
        logger.error(f"Error loading flags queue {path.name}: {e}", exc_info=True)
        return []

def save_daily_flags_queue(data: List[Dict[str, Any]], target_date: Optional[datetime] = None):
    """Saves the daily flags queue by calling its path helper."""
    path = get_daily_flags_queue_path(target_date)
    logger.info(f"Saving {len(data)} flags to {path.name}.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving flags queue {path.name}: {e}", exc_info=True)

def update_matter_for_recent_time_window(new_matter_id: str, lookback_seconds: int, target_date: datetime) -> bool:
    """
    Finds and updates the matter_id for the last N seconds of conversation, searching
    backwards across chunk and day boundaries to meet user expectations.

    Args:
        new_matter_id: The canonical matter ID to assign.
        lookback_seconds: How far back in time to "paint over".
        target_date: The date the command was issued, used as a starting point.

    Returns:
        True if at least one chunk was successfully updated, False otherwise.
    """
    logger.info(f"Vocal Override: Initiating update for matter '{new_matter_id}' with a {lookback_seconds}s lookback from {target_date.strftime('%Y-%m-%d')}.")
    
    all_chunks = []
    
    # 1. Collect chunks from the target day and the previous day to handle boundaries.
    for i in range(2): # 0 = today, 1 = yesterday
        current_date = target_date - timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        
        lock = _get_lock_for_date_str(date_str)
        with lock:
            # Note: We are reading data here, but updates will happen via a separate function call
            # that re-acquires the lock. This is safe.
            log_data = _fetch_or_load_daily_log_data_locked(date_str, current_date)
            if log_data and log_data.get("chunks"):
                # Add the processing date to each chunk for later use
                for chunk_id, chunk in log_data["chunks"].items():
                    chunk['processing_date_obj'] = current_date
                    all_chunks.append(chunk)

    if not all_chunks:
        logger.warning(f"Vocal Override: No chunks found for {target_date.strftime('%Y-%m-%d')} or the previous day.")
        return False
        
    # 2. Sort all collected chunks in reverse chronological order by their end time.
    # We filter out any chunks that are missing the critical timestamp metadata.
    valid_chunks = [
        c for c in all_chunks 
        if c.get("audio_chunk_end_utc") and c.get("audio_chunk_start_utc")
    ]
    valid_chunks.sort(key=lambda x: x["audio_chunk_end_utc"], reverse=True)

    if not valid_chunks:
        logger.error(f"Vocal Override: Found chunks, but none had valid start/end UTC timestamps.")
        return False

    # 3. Iterate backwards through chunks, applying the update until the lookback is satisfied.
    remaining_lookback = float(lookback_seconds)
    updates_performed = 0

    for chunk in valid_chunks:
        if remaining_lookback <= 0:
            break

        try:
            chunk_start_utc = datetime.fromisoformat(chunk["audio_chunk_start_utc"])
            chunk_end_utc = datetime.fromisoformat(chunk["audio_chunk_end_utc"])
            chunk_duration = (chunk_end_utc - chunk_start_utc).total_seconds()
            
            if chunk_duration <= 0: continue

            # Determine how much of the remaining lookback applies to this chunk
            update_duration_for_chunk = min(remaining_lookback, chunk_duration)
            
            relative_end_s = chunk_duration
            relative_start_s = chunk_duration - update_duration_for_chunk

            logger.info(
                f"Vocal Override: Applying update to chunk '{chunk['chunk_id']}' "
                f"from relative time {relative_start_s:.2f}s to {relative_end_s:.2f}s."
            )

            # Use the robust surgical update logic. This will re-acquire the necessary locks.
            success = update_matter_segments_for_chunk(
                target_date=chunk['processing_date_obj'],
                chunk_id=chunk['chunk_id'],
                start_time=relative_start_s,
                end_time=relative_end_s,
                new_matter_id=new_matter_id
            )
            
            if success:
                updates_performed += 1
                remaining_lookback -= update_duration_for_chunk
            else:
                logger.error(f"Vocal Override: Failed to update matter segments for chunk '{chunk['chunk_id']}'.")

        except (ValueError, KeyError) as e:
            logger.error(f"Vocal Override: Could not process chunk '{chunk.get('chunk_id', 'Unknown')}' due to error: {e}", exc_info=True)
            continue
    
    if updates_performed == 0:
        logger.warning("Vocal Override: Completed, but no chunk segments were updated.")
        return False
        
    logger.info(f"Vocal Override: Successfully completed, updating {updates_performed} chunk(s).")
    return True
