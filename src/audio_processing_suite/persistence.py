# Samson/src/audio_processing_suite/persistence.py
import json
import faiss
import numpy as np # Retained as faiss might interact with numpy arrays implicitly or in other uses
import math # For SRT timestamps and potentially other calculations
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from src.logger_setup import logger
from .text_processing import reconstruct_text_from_word_objects # Added for the new function

# --- FAISS / Speaker Map ---
def load_or_create_faiss_index(index_path: Path, dimension: int) -> faiss.Index:
    """Loads a FAISS index or creates a new one."""
    if index_path.exists():
        logger.info(f"Attempting to load existing FAISS index from {index_path}")
        try:
            index = faiss.read_index(str(index_path)) # faiss might need str
            if index.d != dimension:
                logger.warning(f"Index dimension ({index.d}) differs from model dimension ({dimension}). Creating new index at {index_path}.")
                index = faiss.IndexFlatIP(dimension) # Use Inner Product
            else:
                logger.info(f"FAISS index loaded successfully from {index_path} with {index.ntotal} embeddings (Dim: {index.d}).")
        except Exception as e:
            logger.error(f"Error loading FAISS index from {index_path}: {e}. Creating a new one.", exc_info=True)
            index = faiss.IndexFlatIP(dimension)
    else:
        logger.info(f"FAISS index not found at {index_path}. Creating a new one with dimension {dimension}.")
        index = faiss.IndexFlatIP(dimension)
    return index

def save_faiss_index(index: faiss.Index, index_path: Path):
    """Saves the FAISS index to disk."""
    logger.info(f"Saving FAISS index to {index_path} with {index.ntotal} embeddings...")
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path)) # faiss might need str
        logger.info(f"FAISS index saved successfully to {index_path}.")
    except Exception as e:
        logger.error(f"Error saving FAISS index to {index_path}: {e}", exc_info=True)

def load_or_create_speaker_map(map_path: Path) -> Dict[int, Dict[str, Any]]:
    """Loads the speaker name map (FAISS ID -> {name, context}) from JSON."""
    speaker_map: Dict[int, Dict[str, Any]] = {}
    if map_path.exists():
        logger.info(f"Loading speaker map from {map_path}")
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                loaded_map = json.load(f)
            
            # Migration logic to handle old and new formats
            for key, value in loaded_map.items():
                if isinstance(value, str):
                    # This is the old format (str), migrate it to the new dict format
                    logger.info(f"Migrating old speaker map format for speaker '{value}' (ID: {key}).")
                    speaker_map[int(key)] = {"name": value, "context": "unknown"}
                else:
                    # This is assumed to be the new format (dict)
                    speaker_map[int(key)] = value
            logger.info(f"Speaker map loaded from {map_path} with {len(speaker_map)} entries.")
        except Exception as e:
            logger.error(f"Error loading speaker map from {map_path}: {e}. Creating a new one.", exc_info=True)
            speaker_map = {} # Initialize to empty on error
    else:
        logger.info(f"Speaker map not found at {map_path}. Creating a new one.")
        speaker_map = {}
    return speaker_map

def save_speaker_map(speaker_map: Dict[int, Dict[str, Any]], map_path: Path):
    """Saves the speaker name map to JSON."""
    logger.info(f"Saving speaker map to {map_path} with {len(speaker_map)} entries...")
    try:
        map_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure keys are strings for JSON compatibility
        save_map = {str(k): v for k, v in speaker_map.items()}
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(save_map, f, indent=2)
        logger.info(f"Speaker map saved successfully to {map_path}.")
    except Exception as e:
        logger.error(f"Error saving speaker map to {map_path}: {e}", exc_info=True)


# --- Transcript Saving / Loading ---

def save_word_level_transcript(transcript_data: List[Dict[str, Any]], output_path: Path):
    """Saves word-level transcript data (intermediate or final) to a JSON file."""
    logger.info(f"Saving word-level transcript JSON to {output_path}...")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Word-level transcript JSON saved successfully to {output_path}.")
    except Exception as e:
        logger.error(f"Error saving word-level transcript JSON to {output_path}: {e}", exc_info=True)
        raise

def load_word_level_transcript(input_path: Path) -> List[Dict[str, Any]]:
    """Loads the word-level transcript data from a JSON file."""
    logger.info(f"Loading word-level transcript from {input_path}...")
    if not input_path.exists():
        logger.error(f"Word-level transcript file not found: {input_path}")
        raise FileNotFoundError(f"Word-level transcript file not found: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            transcript_data: List[Dict[str, Any]] = json.load(f)
        logger.info(f"Word-level transcript loaded successfully from {input_path}.")
        return transcript_data
    except json.JSONDecodeError as jde:
        logger.error(f"Error decoding JSON from word-level transcript file {input_path}: {jde}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error loading word-level transcript JSON from {input_path}: {e}", exc_info=True)
        raise

def save_final_transcript(punctuated_turns: List[Tuple[Optional[str], str]], output_path: Path):
    """Saves the final, formatted transcript (speaker: text block) to a text file.
    This uses the cleaned and punctuated text.
    """
    logger.info(f"Saving final standard transcript (speaker: text block) to {output_path}...")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            if punctuated_turns:
                for speaker, text in punctuated_turns:
                    speaker_str = str(speaker) if speaker is not None else "UNKNOWN_SPEAKER"
                    text_str = str(text).strip() if text is not None else ""
                    f.write(f"[{speaker_str}]:\n{text_str}\n\n")
                logger.info(f"Final standard transcript saved successfully to {output_path} ({len(punctuated_turns)} speaker turns).")
            else:
                logger.warning(f"Final transcript data for {output_path} is empty or None. Saving an empty file.")
                f.write("")
    except Exception as e:
        logger.error(f"Error saving final standard transcript text file to {output_path}: {e}", exc_info=True)
        raise

def _format_timestamp(seconds: Optional[float], format_srt: bool = False) -> str:
    """Helper to format seconds into HH:MM:SS,ms or HH:MM:SS.ms format."""
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
        return "00:00:00,000" if format_srt else "00:00:00.000"
    
    seconds_float = float(seconds) # Ensure it's float for math ops
    milliseconds = math.floor((seconds_float - math.floor(seconds_float)) * 1000)
    total_seconds_int = math.floor(seconds_float)
    
    hours = total_seconds_int // 3600
    minutes = (total_seconds_int % 3600) // 60
    secs = total_seconds_int % 60
    
    separator = "," if format_srt else "."
    # For the specific periodic timestamp, we don't need milliseconds.
    # The prompt's example "[HH:MM:SS]" suggests no milliseconds.
    # Let's adapt the format_time helper inside the new function if needed, or this global one.
    # The prompt's example `format_time` was:
    # def format_time(seconds: float) -> str:
    #     s = int(seconds)
    #     h = s // 3600
    #     m = (s % 3600) // 60
    #     s = s % 60
    #     return f"{h:02d}:{m:02d}:{s:02d}"
    # This _format_timestamp is for SRT and is more general. The new function has its own.
    # I will keep this _format_timestamp as is, for SRT, and use the simpler one for periodic timestamps.
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}"


def save_transcript_with_timestamps(
    word_level_transcript: List[Dict[str, Any]],
    output_path: Path,
    interval_seconds: int = 60
) -> None:
    """
    Saves a human-readable transcript with speaker labels and periodic timestamps.
    The text is reconstructed from word objects to ensure proper formatting and timing.
    Timestamps are printed at regular intervals (e.g., every 60 seconds).

    Args:
        word_level_transcript: List of segment dictionaries, where each segment
                               contains a list of word dictionaries. Words must have
                               'word', 'start', 'end', and 'speaker' (final label).
        output_path: Path to save the timestamped transcript.
        interval_seconds: How often to print a [HH:MM:SS] timestamp marker.
                          Defaults to 60 seconds.
    """
    logger.info(f"Saving transcript with periodic timestamps ({interval_seconds}s interval) to {output_path}...")

    if interval_seconds <= 0:
        logger.warning(f"interval_seconds ({interval_seconds}s) must be positive. Defaulting to 60s.")
        interval_seconds = 60
    
    if not word_level_transcript:
        logger.warning(f"Word level transcript data for {output_path} is empty. Saving an empty file.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            pass # Create empty file
        return

    output_lines: List[str] = []
    
    def format_time_hhmmss(seconds: float) -> str:
        """Formats seconds to HH:MM:SS."""
        if not isinstance(seconds, (int, float)) or seconds < 0:
            seconds = 0.0
        s_int = int(round(seconds)) # Round to nearest second for HH:MM:SS display
        h = s_int // 3600
        m = (s_int % 3600) // 60
        s = s_int % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    next_timestamp_marker_time: float = 0.0
    current_speaker_label: Optional[str] = None
    current_speaker_word_objects: List[Dict[str, Any]] = []

    all_words: List[Dict[str, Any]] = []
    for segment in word_level_transcript:
        if isinstance(segment, dict) and 'words' in segment and isinstance(segment['words'], list):
            for word_data in segment['words']:
                if isinstance(word_data, dict) and \
                   word_data.get('start') is not None and \
                   word_data.get('word') is not None: # Basic check for usable word
                    all_words.append(word_data)
    
    if not all_words:
        logger.warning(f"No valid words with start times found in word_level_transcript for {output_path}. Saving an empty file.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            pass
        return
        
    all_words.sort(key=lambda w: w['start']) # Sort globally by start time
    
    for word_obj in all_words:
        word_start_time = word_obj.get('start')
        # word_start_time should not be None due to filter above, but defensive check
        if word_start_time is None: 
            logger.debug(f"Skipping word with no start time: {word_obj.get('word')}")
            continue 

        # Check for and print any timestamp markers that are due *before* or at this word's time.
        while word_start_time >= next_timestamp_marker_time:
            if current_speaker_word_objects:
                reconstructed_text = reconstruct_text_from_word_objects(
                    current_speaker_word_objects, current_speaker_label
                )
                if reconstructed_text.strip(): # Only print if there's actual text
                    output_lines.append(f"[{current_speaker_label or 'UNKNOWN_SPEAKER'}]:")
                    output_lines.append(reconstructed_text)
                    output_lines.append("") 
                current_speaker_word_objects = []
            
            output_lines.append(f"--- Timestamp: [{format_time_hhmmss(next_timestamp_marker_time)}] ---")
            next_timestamp_marker_time += interval_seconds
            current_speaker_label = None # Force re-print of speaker header for text after this timestamp

        # Process the current word for speaker and text accumulation.
        speaker = str(word_obj.get("speaker", "UNKNOWN_SPEAKER"))
        
        if speaker != current_speaker_label:
            # Speaker has changed, or it's the first word after a timestamp (label was None).
            if current_speaker_word_objects: # Flush previous speaker's text if any
                reconstructed_text = reconstruct_text_from_word_objects(
                    current_speaker_word_objects, current_speaker_label 
                )
                if reconstructed_text.strip():
                    output_lines.append(f"[{current_speaker_label or 'UNKNOWN_SPEAKER'}]:")
                    output_lines.append(reconstructed_text)
                    output_lines.append("")
            
            current_speaker_label = speaker
            current_speaker_word_objects = [word_obj]
        else: # Same speaker, append word
            current_speaker_word_objects.append(word_obj)

    # Add the very last speaker's turn
    if current_speaker_word_objects:
        reconstructed_text = reconstruct_text_from_word_objects(
            current_speaker_word_objects, current_speaker_label
        )
        if reconstructed_text.strip():
            output_lines.append(f"[{current_speaker_label or 'UNKNOWN_SPEAKER'}]:")
            output_lines.append(reconstructed_text)
            output_lines.append("")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for line_idx, line in enumerate(output_lines):
                f.write(line)
                # Add newline unless it's the very last line and it's an empty line from speaker separation
                if not (line_idx == len(output_lines) - 1 and line == ""):
                     f.write('\n')
            # Ensure file ends with a newline if content exists, unless it's an empty file overall
            if output_lines and output_lines[-1] != "": # if last content line wasn't already blank from speaker sep
                 if not output_lines[-1].endswith('\n'): # check if previous write already did it
                    if len(output_lines) > 0 : # only write newline if file is not empty
                        f.write('\n')


        logger.info(f"Transcript with periodic timestamps saved successfully to {output_path}.")
    except Exception as e:
        logger.error(f"Failed to save transcript with timestamps to {output_path}: {e}", exc_info=True)
        raise


def save_transcript_srt(transcript_data: List[Dict[str, Any]], output_path: Path, words_per_line: int = 10):
    """Saves transcript in SRT subtitle format."""
    logger.info(f"Saving transcript in SRT format to {output_path}...")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            subtitle_index = 1
            all_words: List[Dict[str, Any]] = []
            for segment in transcript_data:
                if 'words' in segment and isinstance(segment.get('words'), list) and segment['words']:
                    all_words.extend(w for w in segment['words'] if isinstance(w, dict) and 'start' in w and 'end' in w and 'word' in w)

            if not all_words:
                 logger.warning(f"No valid timed words found in transcript data for SRT generation for {output_path}.")
                 f.write("")
                 return

            # Sort words by start time if not already, crucial for SRT chunking
            all_words.sort(key=lambda w: w['start'])

            for i in range(0, len(all_words), words_per_line):
                chunk = all_words[i : i + words_per_line]
                if not chunk: continue
                
                start_time = chunk[0].get('start')
                end_time = chunk[-1].get('end') # Use end time of the last word in the chunk
                
                # Speaker of the first word in chunk, fallback for consistency
                speaker = str(chunk[0].get('speaker', 'UNKNOWN_SPEAKER')) 

                start_fmt = _format_timestamp(start_time, format_srt=True)
                end_fmt = _format_timestamp(end_time, format_srt=True)
                
                line_text_parts = [str(w.get('word', '')).strip() for w in chunk if str(w.get('word', '')).strip()]
                line_text = " ".join(line_text_parts)
                
                if not line_text.strip(): # Skip empty lines
                    continue

                f.write(f"{subtitle_index}\n")
                f.write(f"{start_fmt} --> {end_fmt}\n")
                # Optional: Include speaker in SRT line, or remove for cleaner SRT
                # f.write(f"[{speaker}]: {line_text}\n\n") # With speaker
                f.write(f"{line_text}\n\n") # Without speaker, more standard for SRT
                subtitle_index += 1

        logger.info(f"SRT transcript saved successfully to {output_path} ({subtitle_index - 1} entries).")
    except Exception as e:
        logger.error(f"Error saving SRT transcript to {output_path}: {e}", exc_info=True)
        raise

def save_segment_level_json(transcript_data: List[Dict[str, Any]], output_path: Path):
    """
    Saves the transcript as a JSON file containing segment-level information
    (start, end, text, speaker), omitting the detailed word list.
    """
    logger.info(f"Saving segment-level transcript JSON to {output_path}...")
    segment_level_data: List[Dict[str, Any]] = []
    try:
        for segment_idx, segment in enumerate(transcript_data):
            if not isinstance(segment, dict):
                logger.debug(f"Skipping non-dict segment at index {segment_idx}")
                continue

            seg_start = segment.get('start')
            seg_end = segment.get('end')
            # Use pre-computed segment text if available and non-empty
            segment_text = str(segment.get('text', "")).strip() 
            primary_speaker = str(segment.get('speaker', 'UNKNOWN_SPEAKER'))

            # If segment text is empty but words exist, reconstruct from words.
            # Also ensure start/end times are derived from words if missing at segment level.
            word_list = []
            if 'words' in segment and isinstance(segment.get('words'), list) and segment['words']:
                word_list = [w for w in segment['words'] if isinstance(w, dict) and 'word' in w]

            if not segment_text and word_list:
                segment_text_parts: List[str] = [str(w.get('word', '')).strip() for w in word_list]
                segment_text = " ".join(filter(None, segment_text_parts))
            
            if word_list: # Try to get more accurate start/end from words if segment level is missing/generic
                valid_word_starts = [w.get('start') for w in word_list if isinstance(w.get('start'), (float, int))]
                valid_word_ends = [w.get('end') for w in word_list if isinstance(w.get('end'), (float, int))]

                if not isinstance(seg_start, (float, int)) and valid_word_starts:
                    seg_start = min(valid_word_starts)
                if not isinstance(seg_end, (float, int)) and valid_word_ends:
                    seg_end = max(valid_word_ends)
                
                # Attempt to use the first word's speaker if segment speaker is generic and words have specific speakers
                first_word_speaker_candidate = str(word_list[0].get('speaker', primary_speaker))
                if (primary_speaker == 'UNKNOWN_SPEAKER' or primary_speaker == 'SPEAKER_FROM_SEGMENT') and \
                   first_word_speaker_candidate not in ['UNKNOWN_SPEAKER', 'SPEAKER_FROM_SEGMENT', None, '']:
                     primary_speaker = first_word_speaker_candidate


            if isinstance(seg_start, (float, int)) and isinstance(seg_end, (float, int)) and segment_text:
                segment_level_data.append({
                    "start": seg_start,
                    "end": seg_end,
                    "text": segment_text,
                    "speaker": primary_speaker
                })
            else:
                logger.debug(f"Skipping segment for {output_path} due to missing start/end/text. Segment info: start={seg_start}, end={seg_end}, text='{segment_text}', speaker='{primary_speaker}'")


        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(segment_level_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Segment-level transcript JSON saved successfully to {output_path} ({len(segment_level_data)} segments).")

    except Exception as e:
        logger.error(f"Error saving segment-level transcript JSON to {output_path}: {e}", exc_info=True)
        raise