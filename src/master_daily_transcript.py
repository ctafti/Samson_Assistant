# File: src/master_daily_transcript.py
import json
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta, date
from typing import List, Optional, Dict, Any, Tuple
import textwrap # <<< ADDED THIS IMPORT

import pytz

from src.logger_setup import logger
from src.daily_log_manager import get_daily_log_data, get_day_start_time

MASTER_LOG_LOCK = threading.Lock()

# --- START: New Helper Function ---
def _format_speaker_turn_with_wrapping(
    speaker_label: str,
    text_block: str,
    line_width: int
) -> str:
    """
    Formats a speaker's turn with a hanging indent for readability.
    """
    initial_indent = f"[{speaker_label}]: "
    subsequent_indent = ' ' * len(initial_indent)
    
    wrapped_text = textwrap.fill(
        text_block,
        width=line_width,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        replace_whitespace=True
    )
    
    return wrapped_text
# --- END: New Helper Function ---


def get_master_transcript_path(config: Dict[str, Any], target_date: Optional[datetime] = None) -> Path:
    paths_cfg = config.get('paths', {})
    base_folder_val = paths_cfg.get('database_folder') 
    
    base_folder: Path
    if isinstance(base_folder_val, str) and base_folder_val.strip():
        base_folder = Path(base_folder_val)
    elif isinstance(base_folder_val, Path):
        base_folder = base_folder_val
    else:
        logger.warning(f"MasterTranscript: 'paths.database_folder' ('{base_folder_val}') not configured or invalid. Using fallback.")
        from src.config_loader import PROJECT_ROOT 
        base_folder = PROJECT_ROOT / "data" / "master_dialogues_fallback"
    
    base_folder.mkdir(parents=True, exist_ok=True)

    date_to_use = target_date if target_date else datetime.now(timezone.utc)
    if date_to_use.tzinfo is None: 
        date_to_use = date_to_use.replace(tzinfo=timezone.utc)

    date_str_for_file = date_to_use.strftime("%Y-%m-%d")
    file_name = f"MASTER_DIALOGUE_{date_str_for_file}.txt"
    return base_folder / file_name


def append_processed_chunk_to_master_log(
    master_log_path: Path,
    word_level_transcript_for_chunk: List[Dict[str, Any]], 
    audio_chunk_start_time_utc: datetime, 
    is_first_chunk_of_day_for_master_log: bool,
    current_master_log_day_str: str, 
    last_speaker_in_master_log: Optional[str],
    next_timestamp_marker_abs_utc: Optional[datetime], 
    config: Dict[str, Any]
) -> Tuple[Optional[str], Optional[datetime]]:
    global MASTER_LOG_LOCK

    local_next_timestamp_marker = next_timestamp_marker_abs_utc 
    lines_to_write: List[str] = []

    aps_cfg = config.get('audio_suite_settings', {})
    timings_cfg = config.get('timings', {})
    periodic_interval_s = int(aps_cfg.get('timestamped_transcript_interval_seconds', 120))
    assumed_tz_str = timings_cfg.get('assumed_recording_timezone', 'UTC')
    # <<< MODIFIED: Get line width from config or use a default >>>
    master_log_line_width = int(aps_cfg.get('master_log_line_width', 90))

    try:
        display_timezone = pytz.timezone(assumed_tz_str)
    except pytz.UnknownTimeZoneError:
        logger.warning(f"MasterLogAppend: Unknown timezone '{assumed_tz_str}', defaulting to UTC.")
        display_timezone = timezone.utc
    master_log_timestamp_format = timings_cfg.get('master_log_timestamp_format', "%b%d, %Y - %H:%M")

    if local_next_timestamp_marker is None or is_first_chunk_of_day_for_master_log:
        new_anchor_time = audio_chunk_start_time_utc
        local_next_timestamp_marker = new_anchor_time 
        
        if not is_first_chunk_of_day_for_master_log and master_log_path.exists() and master_log_path.stat().st_size > 0:
            lines_to_write.append(f"\n--- Note: Day start time may have been adjusted. Periodic timestamps re-anchored. ---\n")

    if is_first_chunk_of_day_for_master_log:
        day_start_display = audio_chunk_start_time_utc.astimezone(display_timezone).strftime(master_log_timestamp_format)
        lines_to_write.append(f"## Samson Master Log for {current_master_log_day_str} ##\n")
        lines_to_write.append(f"Day recording started around: {day_start_display} ({assumed_tz_str})\n")

    current_turn_speaker_label = last_speaker_in_master_log
    current_turn_words_buffer: List[str] = []
    all_words_in_chunk_with_abs_time: List[Dict[str, Any]] = []

    if word_level_transcript_for_chunk:
        for segment in word_level_transcript_for_chunk: 
            if isinstance(segment, dict) and 'words' in segment and isinstance(segment['words'], list):
                for word_obj in segment['words']:
                    if isinstance(word_obj, dict) and word_obj.get('start') is not None and word_obj.get('word') is not None: 
                        word_copy = word_obj.copy() 
                        try:
                            relative_start_s = float(word_obj['start'])
                            word_copy['absolute_start_utc'] = audio_chunk_start_time_utc + timedelta(seconds=relative_start_s)
                            all_words_in_chunk_with_abs_time.append(word_copy)
                        except (ValueError, TypeError):
                            continue
        all_words_in_chunk_with_abs_time.sort(key=lambda w: w['absolute_start_utc'])

    for word_obj in all_words_in_chunk_with_abs_time:
        word_abs_start_utc = word_obj['absolute_start_utc']
        
        if local_next_timestamp_marker is None: local_next_timestamp_marker = audio_chunk_start_time_utc

        while local_next_timestamp_marker and word_abs_start_utc >= local_next_timestamp_marker:
            if current_turn_words_buffer: 
                full_text_block = ' '.join(current_turn_words_buffer)
                formatted_turn = _format_speaker_turn_with_wrapping(current_turn_speaker_label or 'UNKNOWN', full_text_block, master_log_line_width)
                lines_to_write.append(f"\n{formatted_turn}\n")
                current_turn_words_buffer = []
            
            marker_display_time = local_next_timestamp_marker.astimezone(display_timezone)
            lines_to_write.append(f"\n--- [{marker_display_time.strftime('%H:%M:%S %Z')}] ---\n")
            local_next_timestamp_marker += timedelta(seconds=periodic_interval_s)
            current_turn_speaker_label = None 

        word_text = str(word_obj.get("word", "")).strip() 
        if not word_text: continue

        speaker_for_word = str(word_obj.get("speaker", "UNKNOWN_SPEAKER")) 

        if speaker_for_word != current_turn_speaker_label:
            if current_turn_words_buffer:
                full_text_block = ' '.join(current_turn_words_buffer)
                formatted_turn = _format_speaker_turn_with_wrapping(current_turn_speaker_label or 'UNKNOWN', full_text_block, master_log_line_width)
                lines_to_write.append(f"\n{formatted_turn}\n")
            current_turn_speaker_label = speaker_for_word
            current_turn_words_buffer = [word_text] 
        else:
            current_turn_words_buffer.append(word_text)
    
    if current_turn_words_buffer:
        full_text_block = ' '.join(current_turn_words_buffer)
        formatted_turn = _format_speaker_turn_with_wrapping(current_turn_speaker_label or 'UNKNOWN', full_text_block, master_log_line_width)
        lines_to_write.append(f"\n{formatted_turn}\n")

    if not lines_to_write: 
        return current_turn_speaker_label, local_next_timestamp_marker

    with MASTER_LOG_LOCK:
        try:
            with open(master_log_path, "a", encoding="utf-8") as f:
                f.writelines(lines_to_write)
        except Exception as e:
            logger.error(f"Failed to append to master log {master_log_path.name}: {e}", exc_info=True)

    return current_turn_speaker_label, local_next_timestamp_marker

def _render_master_log_content_for_words(
    words_for_day: List[Dict[str, Any]], 
    day_start_time_utc: datetime, 
    config: Dict[str, Any], 
    current_master_log_day_str: str
) -> List[str]:
    lines_to_write: List[str] = []

    aps_cfg = config.get('audio_suite_settings', {})
    timings_cfg = config.get('timings', {})
    periodic_interval_s = int(aps_cfg.get('timestamped_transcript_interval_seconds', 120))
    assumed_tz_str = timings_cfg.get('assumed_recording_timezone', 'UTC')
    master_log_line_width = int(aps_cfg.get('master_log_line_width', 90))

    try:
        display_timezone = pytz.timezone(assumed_tz_str)
    except pytz.UnknownTimeZoneError:
        display_timezone = timezone.utc
    master_log_timestamp_format = timings_cfg.get('master_log_timestamp_format', "%b%d, %Y - %H:%M")

    day_start_display = day_start_time_utc.astimezone(display_timezone).strftime(master_log_timestamp_format)
    lines_to_write.append(f"## Samson Master Log for {current_master_log_day_str} ##\n")
    lines_to_write.append(f"Day recording started around: {day_start_display} ({assumed_tz_str})\n")

    local_next_timestamp_marker = day_start_time_utc 
    current_turn_speaker_label: Optional[str] = None
    current_turn_words_buffer: List[str] = []

    for word_obj in words_for_day:
        word_abs_start_utc = word_obj.get('absolute_start_utc')
        if not isinstance(word_abs_start_utc, datetime): continue

        speaker_for_word = str(word_obj.get('speaker', 'UNKNOWN_SPEAKER'))
        word_text = str(word_obj.get('word', '')).strip()
        if not word_text: continue

        while local_next_timestamp_marker and word_abs_start_utc >= local_next_timestamp_marker:
            if current_turn_words_buffer:
                full_text_block = ' '.join(current_turn_words_buffer)
                formatted_turn = _format_speaker_turn_with_wrapping(current_turn_speaker_label or 'UNKNOWN', full_text_block, master_log_line_width)
                lines_to_write.append(f"\n{formatted_turn}\n")
                current_turn_words_buffer = []
            
            marker_display_time = local_next_timestamp_marker.astimezone(display_timezone)
            lines_to_write.append(f"\n--- [{marker_display_time.strftime('%H:%M:%S %Z')}] ---\n")
            local_next_timestamp_marker += timedelta(seconds=periodic_interval_s)
            current_turn_speaker_label = None

        if speaker_for_word != current_turn_speaker_label:
            if current_turn_words_buffer:
                full_text_block = ' '.join(current_turn_words_buffer)
                formatted_turn = _format_speaker_turn_with_wrapping(current_turn_speaker_label or 'UNKNOWN', full_text_block, master_log_line_width)
                lines_to_write.append(f"\n{formatted_turn}\n")
            current_turn_speaker_label = speaker_for_word
            current_turn_words_buffer = [word_text]
        else:
            current_turn_words_buffer.append(word_text)

    if current_turn_words_buffer:
        full_text_block = ' '.join(current_turn_words_buffer)
        formatted_turn = _format_speaker_turn_with_wrapping(current_turn_speaker_label or 'UNKNOWN', full_text_block, master_log_line_width)
        lines_to_write.append(f"\n{formatted_turn}\n")
    
    return lines_to_write

def regenerate_master_log_for_day(target_date_obj: date, config: Dict[str, Any]):
    logger.info(f"Regenerating master log for: {target_date_obj.strftime('%Y-%m-%d')}")
    date_str_for_log = target_date_obj.strftime('%Y-%m-%d')
    
    global MASTER_LOG_LOCK
    with MASTER_LOG_LOCK:
        try:
            target_datetime = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=timezone.utc)
            daily_log = get_daily_log_data(target_datetime)
            if not (daily_log and daily_log.get('chunks')):
                logger.warning(f"No daily log chunks for {date_str_for_log} to regenerate master log.")
                return

            official_day_start_utc = get_day_start_time(target_datetime)
            if not official_day_start_utc:
                logger.error(f"Cannot regenerate master log for {date_str_for_log}: Official day start time not set.")
                return
            
            all_words_for_day: List[Dict[str, Any]] = []
            for entry in daily_log.get('chunks', {}).values():
                for segment in entry.get('processed_data', {}).get('word_level_transcript_with_absolute_times', []):
                    for word_obj in segment.get('words', []):
                        if not isinstance(word_obj, dict): continue
                        word_data = word_obj.copy()
                        abs_start_utc_str = word_data.get('absolute_start_utc')
                        
                        if isinstance(abs_start_utc_str, str):
                            try: word_data['absolute_start_utc'] = datetime.fromisoformat(abs_start_utc_str.replace('Z', '+00:00'))
                            except ValueError: continue
                        elif isinstance(abs_start_utc_str, datetime): word_data['absolute_start_utc'] = abs_start_utc_str
                        else: continue
                        
                        all_words_for_day.append(word_data)
            
            all_words_for_day.sort(key=lambda x: x['absolute_start_utc'])
            log_lines = _render_master_log_content_for_words(all_words_for_day, official_day_start_utc, config, date_str_for_log)
            
            master_log_path = get_master_transcript_path(config, target_datetime)
            logger.info(f"Writing regenerated master log to: {master_log_path}")
            
            with open(master_log_path, "w", encoding="utf-8") as f:
                f.writelines(log_lines)
            
            logger.info(f"Successfully regenerated master log: {master_log_path.name}")
        except Exception as e:
            logger.error(f"Error regenerating master log for {date_str_for_log}: {e}", exc_info=True)