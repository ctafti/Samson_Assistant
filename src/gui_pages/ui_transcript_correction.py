
import streamlit as st
import streamlit.components.v1 as components
from datetime import date, datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple
import json
import hashlib
import time
import urllib.parse
import pytz

# --- Local Samson Imports ---
from src.daily_log_manager import get_daily_log_data, get_samson_today
from src.speaker_profile_manager import get_enrolled_speaker_names, get_all_speaker_profiles
from main_orchestrator import queue_command_from_gui
from src.logger_setup import logger, setup_logging
from src.config_loader import get_config
from src.context_manager import get_active_context
from src.matter_manager import get_all_matters

# --- Constants ---
DEFAULT_MATTER_ID = "m_unassigned"
DEFAULT_MATTER_NAME = "Unassigned"

def get_color_for_matter_id(matter_id: str) -> str:
    """
    Deterministically assigns a pleasant, light pastel color to a matter_id
    using HSL color space for virtually unlimited, readable colors.
    """
    if not matter_id:
        return "transparent"
    
    # Use MD5 hash to get a deterministic integer from the matter_id string
    hash_object = hashlib.md5(matter_id.encode())
    hash_digest = hash_object.hexdigest()
    hash_int = int(hash_digest, 16)
    
    # Map the integer to a hue value (0-359 degrees on the color wheel)
    hue = hash_int % 360
    
    # Use fixed high lightness and moderate saturation for pleasant pastels
    saturation = 55
    lightness = 90
    
    return f"hsl({hue}, {saturation}%, {lightness}%)"

# --- All Python functions are unchanged ---
@st.cache_data(ttl=60)
def _get_speaker_name_to_id_map() -> Dict[str, int]:
    """Fetches speaker profiles and creates a name -> faiss_id map."""
    profiles = get_all_speaker_profiles()
    return {p['name']: p['faiss_id'] for p in profiles if 'name' in p and 'faiss_id' in p}

@st.cache_data
def _prepare_dialogue_data_for_display(chunk_data, all_matters_map, flags_data):
    """
    Prepares transcript data for rendering, including logic for conflict flags and null matters.
    This function handles the nested segment/word structure from the audio pipeline.
    """
    segments = chunk_data.get('processed_data', {}).get('word_level_transcript_with_absolute_times', [])
    if not segments:
        return []

    # Create a quick lookup for conflict flags relevant to this chunk
    chunk_id = chunk_data.get('chunk_id')
    conflict_flags_map = {}
    if chunk_id and flags_data:
        for flag in flags_data:
            payload = flag.get('payload', {})
            if flag.get('type') == 'matter_conflict' and payload.get('source_chunk_id') == chunk_id:
                # Store the flag by a time-based key for quick lookup
                start_time = payload.get('start_time')
                if start_time is not None:
                    conflict_flags_map[float(start_time)] = flag

    def find_conflict_flag_for_word(word_start_time, flags_map):
        # Find if a word falls within any flagged segment
        for flag_start_time, flag in flags_map.items():
            flag_end_time = flag.get('payload', {}).get('end_time')
            if flag_end_time is not None and flag_start_time <= word_start_time < float(flag_end_time):
                return flag
        return None

    dialogue_for_display = []

    # FIX: Iterate through each segment, then through the words list inside the segment.
    for seg_idx, segment in enumerate(segments):
        for word_idx, word_info in enumerate(segment.get("words", [])):
            speaker_name = word_info.get('speaker_name') or word_info.get('speaker') or "Unknown"

            current_matter_id = word_info.get('matter_id')
            matter_badge_html = ''
            
            # 1. Check for an overlapping conflict flag first
            conflict_flag = find_conflict_flag_for_word(word_info.get('start'), conflict_flags_map)

            if conflict_flag:
                conflicting_matters = conflict_flag.get('payload', {}).get('conflicting_matters', [])
                tooltip_text = "Conflict Detected:\\n" + "\\n".join([
                    f"- {m.get('name', 'Unknown')} (Score: {m.get('score', 0)*100:.1f}%)" for m in conflicting_matters
                ])
                matter_badge_html = f'<span class="matter-badge conflict-badge" title="{tooltip_text}">🟠 Conflict</span>'
            
            # 2. If no conflict, check for a valid, assigned matter
            elif current_matter_id and current_matter_id is not None:
                matter_details = all_matters_map.get(current_matter_id, {})
                matter_name = matter_details.get('name', "Unknown Matter")
                matter_badge_html = f'<span class="matter-badge matter-badge-{current_matter_id}" title="Matter: {matter_name}">{matter_name}</span>'
                
            # 3. If matter_id is null or missing, and there is no conflict, matter_badge_html remains an empty string.
            
            # --- START OF FIX: Pass through all necessary metadata ---
            dialogue_for_display.append({
                'word': word_info.get('word'),
                'start': word_info.get('start'),
                'end': word_info.get('end'),
                'speaker': speaker_name,
                'matter_badge': matter_badge_html,
                'matter_id': current_matter_id,
                'absolute_start_utc': word_info.get('absolute_start_utc'),
                'segment_index': seg_idx,
                'word_index': word_idx
            })
            # --- END OF FIX ---
            
    return dialogue_for_display


def _prepare_dialogue_and_word_map_from_log(log_data: Dict, matters: List[Dict], flags: List[Dict], selected_date: date) -> Tuple[str, Dict]:
    """
    Orchestrates the preparation of dialogue data from a full day's log,
    generates the display HTML, and creates a word map. This function restores
    the logic that was lost during an incomplete refactoring.
    """
    all_words_for_display = []
    matters_map = {m['matter_id']: m for m in matters}

    sorted_chunks = sorted(log_data.get("chunks", {}).values(), key=lambda c: c.get('file_sequence_number', 0))
    for chunk in sorted_chunks:
        words_in_chunk = _prepare_dialogue_data_for_display(chunk, matters_map, flags)
        chunk_id = chunk.get('chunk_id')
        original_file = chunk.get('original_file_name')
        for word in words_in_chunk:
             word['chunk_id'] = chunk_id
             word['original_file_name'] = original_file
        all_words_for_display.extend(words_in_chunk)


    dialogue_html_parts = []
    word_map = {}
    current_speaker = None
    config = get_config()
    interval_seconds = config.get('audio_suite_settings', {}).get('timestamped_transcript_interval_seconds', 60)
    # Add a safeguard against invalid config values (e.g., 0)
    if interval_seconds <= 0:
        interval_seconds = 60
    last_interval_index = -1
    current_highlight_matter_id = None
    word_counter = 0

    last_matter_id_for_badge = None
    speaker_turns_since_last_badge = 0

    tz_str = get_config().get('timings', {}).get('assumed_recording_timezone', 'UTC')
    local_tz = pytz.timezone(tz_str)

    for word_info in all_words_for_display:
        word_id = word_counter
        word_map[word_id] = {
            "word_data": word_info,
            "chunk_id": word_info.get('chunk_id'),
            "original_file_name": word_info.get('original_file_name'),
            "segment_index": word_info.get('segment_index'),
            "word_index": word_info.get('word_index')
        }


        absolute_start_utc_str = word_info.get('absolute_start_utc')
        current_ts = None
        if absolute_start_utc_str:
            try:
                # Handle Z timezone format for broader compatibility
                if absolute_start_utc_str.endswith('Z'):
                    absolute_start_utc_str = absolute_start_utc_str[:-1] + '+00:00'
                current_ts = datetime.fromisoformat(absolute_start_utc_str).astimezone(local_tz)
            except (ValueError, TypeError):
                pass # current_ts will remain None

        if not current_ts:
            # Fallback for words without a valid absolute timestamp
            if current_speaker != word_info['speaker']:
                if current_speaker is not None: dialogue_html_parts.append('</div>')
                current_speaker = word_info['speaker']
                dialogue_html_parts.append('<div class="speaker-turn">')
                dialogue_html_parts.append(f'<span class="speaker-label" data-word-id="{word_id}">{current_speaker}:</span> ')
            dialogue_html_parts.append(f'<span class="word" data-word-id="{word_id}">{word_info["word"]}</span>')
            word_counter += 1
            continue

        unix_seconds = int(current_ts.timestamp())
        current_interval_index = unix_seconds // interval_seconds
        current_matter_id = word_info.get('matter_id')

        if current_interval_index > last_interval_index:
            if current_speaker is not None:
                dialogue_html_parts.append('</div>')
            if current_highlight_matter_id:
                dialogue_html_parts.append('</div>')
            dialogue_html_parts.append(f'<div class="timestamp-marker">[{current_ts.strftime("%H:%M")}]</div>')
            last_interval_index = current_interval_index
            current_highlight_matter_id = None
            current_speaker = None

        if current_matter_id != current_highlight_matter_id:
            if current_speaker is not None:
                dialogue_html_parts.append('</div>')
            if current_highlight_matter_id:
                dialogue_html_parts.append('</div>')
            if current_matter_id:
                dialogue_html_parts.append(f'<div class="matter-highlight matter-highlight-{current_matter_id}">')
            current_highlight_matter_id = current_matter_id
            current_speaker = None

        if word_info['speaker'] != current_speaker:
            if current_speaker is not None:
                dialogue_html_parts.append('</div>')
            
            current_speaker = word_info['speaker']
            
            show_badge = False
            if word_info['matter_badge']:
                if current_matter_id != last_matter_id_for_badge:
                    show_badge = True
                    speaker_turns_since_last_badge = 1
                    last_matter_id_for_badge = current_matter_id
                else:
                    speaker_turns_since_last_badge += 1
                    if speaker_turns_since_last_badge >= 30:
                        show_badge = True
                        speaker_turns_since_last_badge = 1
            else:
                last_matter_id_for_badge = None
                speaker_turns_since_last_badge = 0
            
            if show_badge:
                dialogue_html_parts.append(f'<div class="matter-badge-container">{word_info["matter_badge"]}</div>')
            
            dialogue_html_parts.append('<div class="speaker-turn">')
            dialogue_html_parts.append(f'<span class="speaker-label" data-word-id="{word_id}">{current_speaker}:</span> ')
        
        dialogue_html_parts.append(f'<span class="word" data-word-id="{word_id}">{word_info["word"]}</span>')
        word_counter += 1

    if current_speaker is not None:
        dialogue_html_parts.append('</div>')
    if current_highlight_matter_id:
        dialogue_html_parts.append('</div>')

    dialogue_html = "".join(dialogue_html_parts)
    date_str = selected_date.strftime('%Y%m%d')
    full_html_container = f'<div id="transcript-container" data-date="{date_str}">{dialogue_html}</div>'

    return full_html_container, word_map


def render_correction_toolkit(selected_date: date, word_map: Dict, enrolled_speakers: List[str]):
    """
    Renders messages in the sidebar based on the current selection state.
    All corrections are now handled by the popup component.
    """
    if 'selected_word_ids' not in st.session_state or not st.session_state.selected_word_ids:
        return

    # If we are here, it means words are selected.
    st.sidebar.subheader("Selection Active 📝")
    st.sidebar.markdown(f"**Selected Words:** {len(st.session_state.selected_word_ids)}")

    preview_words = []
    # Use word_map if available to show a preview
    if word_map:
        for word_id in st.session_state.selected_word_ids[:5]:
            if word_id in word_map:
                word_text = word_map[word_id]["word_data"].get("word", "")
                preview_words.append(word_text)

    preview_text = " ".join(preview_words)
    if len(st.session_state.selected_word_ids) > 5:
        preview_text += "..."

    st.sidebar.markdown(f"**Preview:** `{preview_text}`")

def create_robust_transcript_component(dialogue_html: str, selected_word_ids: List[int], enrolled_speakers: List[str], component_id: str, word_map: Dict, matters: List[Dict[str, Any]], audio_volume: float = 0.7, auto_play_on_select: bool = False, search_query: str = "", search_match_index: int = 0) -> str:
    """
    Creates a robust transcript component with multi-select capability.
    """

    # Dynamically generate CSS for matter highlight colors
    dynamic_styles = []
    for matter in matters:
        matter_id = matter.get("matter_id")
        if matter_id:
            color = get_color_for_matter_id(matter_id)
            highlight_color = color.replace("hsl(", "hsla(").replace(")", ", 0.4)")
            # CSS class for the highlight
            dynamic_styles.append(f".matter-highlight-{matter_id} {{ background-color: {highlight_color} !important; }}")
            dynamic_styles.append(f".matter-badge-{matter_id} {{ background-color: {color} !important; border: 1px solid rgba(0,0,0,0.15); }}")
    
    dynamic_css = "<style>\n" + "\n".join(dynamic_styles) + "\n</style>"


    component_css = """
    <style>
        .main{
            overflow: hidden;
        }
        html, body {
            margin: 0;
            padding: 0;
            height: 100%;
        }
        #component-body {
            height: 100%; 
            border: 1px solid #ccc;
            border-radius: 0.5rem;
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            position: relative; /* MODIFICATION: Added for overlay positioning */
        }
        #transcript-container {
            flex-grow: 1;
            overflow-y: auto;
            overflow-x: hidden;
            min-height: 0;
            padding: 1rem;
            background: #fafafa;
            position: relative;
            line-height: 2.0;
            font-family: monospace;
            user-select: none;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
        }
        .timestamp-marker {
            font-weight: bold;
            color: #666;
            margin: 1rem 0 0.5rem 0;
            font-family: monospace;
            font-size: 0.9em;
        }
        .speaker-label {
            font-weight: bold;
            color: #2c5aa0;
            cursor: pointer;
            padding: 0.2rem 0.4rem;
            margin-right: 0.5rem;
            border-radius: 0.3rem;
            transition: all 0.2s ease;
            vertical-align: middle;
            
        }

        .matter-highlight {
            padding: 10px 1rem;
            margin: 8px -1rem;
            border-radius: 8px;
            border: 1px solid rgba(0,0,0,0.08);
            display: block;
        }
        
        .speaker-turn {
            margin-bottom: 0.5rem; /* Provides spacing between consecutive speaker turns */
        }

        .matter-badge-container {
            margin-bottom: 8px; /* Space between badge and the first speaker line */
        }

        .matter-badge {
            background-color: #e0e0e0;
            color: #333;
            border-radius: 5px;
            padding: 3px 8px;
            font-size: 0.75em;
            font-weight: bold;
            margin-right: 6px;
            display: inline-block; /* Allows badge to size to its content */
            font-family: sans-serif;
            vertical-align: middle;
            white-space: normal;
            line-height: 1.3;
            text-align: center;
        }

        .conflict-badge {
            background-color: #FFB74D; /* Orange 500 */
            color: #212121; /* Dark Grey Text */
            font-family: sans-serif;
            cursor: help;
            border: 1px solid #E65100; /* Darker Orange Border */
        }

        .speaker-label:hover, .word:hover {
            background-color: #e3f2fd;
            transform: scale(1.02);
        }
        .word {
            cursor: pointer;
            border-radius: 0.2rem;
            padding: 0.1rem 0.2rem;
            margin: 0.05rem;
            transition: all 0.2s ease;
            display: inline-block;
        }
        .word-selected {
            background-color: #1976d2 !important;
            color: white !important;
            font-weight: bold;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }
        .word-multi-selected {
            background-color: #ff9800 !important;
            color: white !important;
            font-weight: bold;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }
        .word-drag-hover {
            background-color: #ffeb3b !important;
            color: black !important;
        }

        .word-search-highlight {
            background-color: #ffda00 !important;
            color: black !important;
            border-radius: 3px;
            box-shadow: 0 0 5px #ffda00;
        }

        .speaker-label-pending-update {
            color: #2e7d32 !important; /* Dark Green */
            font-style: italic;
        }
        .word-pending-update {
            background-color: #e8f5e9 !important; /* Light Green Background */
            color: black !important;
            font-style: italic;
            cursor: wait;
        }
        .editor-locked-overlay {
            position: absolute; /* MODIFICATION: Positions relative to #component-body */
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(200, 200, 200, 0.8);
            z-index: 5000;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 1.2em;
            color: #333;
            font-weight: bold;
            cursor: wait;
        }

        .status-indicator {
            position: fixed;
            top: 10px;
            right: 10px;
            background: #4caf50;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            z-index: 6000; /* Higher than overlay */
            opacity: 0;
            transition: opacity 0.3s ease, transform 0.3s ease;
            transform: translateY(-20px);
        }
        .status-indicator.show {
            opacity: 1;
            transform: translateY(0);
        }
        .multi-select-popup {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            z-index: 10000;
            min-width: 750px;
            max-width: 900px;
            min-height: 500px; /* Set a minimum height */
            font-family: sans-serif;
            display: flex; /* Use flexbox for robust layout */
            flex-direction: column; /* Stack children vertically */
            resize: both;
            overflow: auto;
        }
        .popup-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.3);
            z-index: 9999;
        }
        .popup-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #eee; flex-shrink: 0; /* Prevent header from shrinking */ }
        .popup-title { font-size: 1.1rem; font-weight: 600; color: #333; }
        .close-btn { background: transparent; color: #777; border: none; border-radius: 50%; width: 30px; height: 30px; cursor: pointer; font-size: 20px; line-height: 30px; text-align: center; }
        .close-btn:hover { background: #f0f0f0; color: #333; }
        .popup-content { margin-bottom: 20px; display: flex; flex-direction: column; flex-grow: 1; /* Allow content area to grow */ }
        .popup-buttons { display: flex; gap: 10px; justify-content: flex-end; margin-top: 20px; flex-shrink: 0; /* Prevent buttons from shrinking */ }

        .popup-form-element {
            width: 100%;
            padding: 9px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 12px;
            font-size: 0.95rem;
            box-sizing: border-box; /* Crucial for alignment */
        }
        
        #multi-text-edit.popup-form-element {
            font-family: monospace;
            resize: none; /* User resizes the whole popup, not the textarea */
            overflow-y: auto; /* Show scrollbar if content overflows max-height */
            min-height: 100px; /* Start with ~5 lines */
            max-height: 300px; /* Grow up to ~18 lines */
        }
        
        .speaker-assignment label { display: block; margin-bottom: 6px; font-weight: 500; font-size: 0.95rem; color: #444; }
        .popup-volume-control { margin-top: auto; padding-top: 15px; border-top: 1px solid #eee; }
        .popup-volume-control label { display: inline-block; margin-right: 10px; vertical-align: middle; font-weight: 500; }
        .popup-volume-control input[type="range"] { width: calc(100% - 130px); vertical-align: middle; }
        .popup-volume-control #popup-volume-value { display: inline-block; width: 40px; text-align: right; font-family: monospace; font-size: 0.9rem; color: #333; vertical-align: middle; }

        .btn { padding: 9px 18px; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem; font-weight: 500; transition: background-color 0.2s ease, transform 0.1s ease; }
        .btn-primary { background: #2196F3; color: white; }
        .btn-secondary { background: #9E9E9E; color: white; }
        .btn-success { background: #4CAF50; color: white; }
        .btn:hover { opacity: 0.85; }
        .btn:active { transform: scale(0.98); }
        .btn:disabled { background: #BDBDBD; cursor: not-allowed; }
        .btn-processing { background: #FF9800 !important; color: white !important; cursor: wait !important; }
        .word-count { color: #777; font-size: 0.85rem; margin-bottom: 8px; }
        
        /* --- Single Word Quick Editor --- */
        .single-word-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: transparent;
            z-index: 10001;
        }
        .single-word-editor {
            position: fixed;
            background: white;
            border: 1px solid #999;
            border-radius: 4px;
            padding: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 10002;
            display: flex;
            align-items: center;
        }
        .single-word-input {
            border: 1px solid #ccc;
            padding: 8px;
            font-size: 1rem;
            border-radius: 3px;
            outline: none;
            font-family: monospace;
        }
        .single-word-input:focus {
            border-color: #2196F3;
            box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.2);
        }
    </style>
    """

    selected_word_ids_for_js = json.dumps(selected_word_ids) if selected_word_ids else '[]'
    enrolled_speakers_js = json.dumps(enrolled_speakers)
    word_map_js = json.dumps(word_map)
    matters_js = json.dumps(matters)
    auto_play_on_select_js = json.dumps(auto_play_on_select)
    search_query_js = json.dumps(search_query)
    search_match_index_js = json.dumps(search_match_index)

    component_js = f"""
    <script>
        // Global initialization guard prevents the entire script from re-running
        if (window.transcriptComponentInitialized) {{
            console.log("[JS Component] Initialization blocked: Script has already run.");
        }} else {{

        // --- Component State Variables ---
        let currentComponentId = '{component_id}';
        let currentSelectedWordIds = {selected_word_ids_for_js};
        let currentWordMap = {word_map_js};
        let currentEnrolledSpeakers = {enrolled_speakers_js};
        let currentMatters = {matters_js};
        let currentAudioVolume = {audio_volume};
        let lastKnownPythonVolume = {audio_volume};
        let currentAutoPlayOnSelect = {auto_play_on_select_js};
        let currentSearchQuery = {search_query_js};
        let currentSearchMatchIndex = {search_match_index_js};
        let isSaveInProgress = false;
        let pendingUpdateWordIds = new Set();
        let isMouseDownOnPopup = false;

        const VOLUME_STORAGE_KEY = 'samson_cockpit_volume';
        const storedVolumeStr = localStorage.getItem(VOLUME_STORAGE_KEY);
        if (storedVolumeStr !== null) {{
            currentAudioVolume = parseFloat(storedVolumeStr);
        }}

        let isDragging = false;
        let dragStartX = 0, dragStartY = 0; // Kept for drag threshold detection
        let dragStartWordId = null; // The word ID where dragging started
        let dragSelectedWords = new Set();
        const DRAG_THRESHOLD = 5; 
        let potentialClickTargetForMouseUp = null;
        let popupAudioPlayer = null;

        const SCROLL_POS_KEY = 'transcriptScrollTop';

        function saveScrollPosition() {{
            const container = document.getElementById('transcript-container');
            if (container) {{
                sessionStorage.setItem(SCROLL_POS_KEY, container.scrollTop.toString());
                jsLog(`Saved scroll position: ${{container.scrollTop}}`);
            }}
        }}

        function restoreScrollPosition() {{
            const container = document.getElementById('transcript-container');
            const savedScrollTop = sessionStorage.getItem(SCROLL_POS_KEY);
            if (container && savedScrollTop !== null) {{
                container.scrollTop = parseInt(savedScrollTop, 10);
                jsLog(`Restored scroll position: ${{savedScrollTop}}`);
                sessionStorage.removeItem(SCROLL_POS_KEY); // Clean up after use
            }}
        }}

        function jsLog(message, ...args) {{
            console.log(`[JS Component ${{currentComponentId}}] ${{message}}`, ...args);
        }}
        
        function showStatus(message, color = '#4caf50') {{
            let indicator = document.getElementById('status-indicator');
            if (!indicator) {{
                indicator = document.createElement('div');
                indicator.id = 'status-indicator';
                indicator.className = 'status-indicator';
                document.body.appendChild(indicator);
            }}
            indicator.textContent = message;
            indicator.style.background = color;
            indicator.classList.add('show');
            setTimeout(() => {{
                indicator.classList.remove('show');
            }}, 3000); // Increased duration
        }}
        
        function autosizeTextarea(textarea) {{
            if (!textarea) return;
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        }}

        function sendSelectionToStreamlit(data) {{
            jsLog('Sending data to Streamlit:', data);
            try {{
                if (window.Streamlit) {{
                    window.Streamlit.setComponentValue(data);
                    return true;
                }}
            }} catch(e) {{ jsLog(`Streamlit.setComponentValue failed:`, e); }}
            return false;
        }}

        function selectMultipleWords(wordIds) {{
            jsLog(`selectMultipleWords called with ${{wordIds.length}} word IDs.`);
            const data = {{ action: 'multi_select', selected_word_ids: wordIds, timestamp: Date.now(), componentId: currentComponentId }};
            highlightMultipleWordsUI(wordIds);
            sendSelectionToStreamlit(data);
            currentSelectedWordIds = wordIds;
            if (wordIds && wordIds.length > 0) {{
                showMultiSelectPopup(wordIds);
            }} else {{
                closeMultiSelectPopup();
            }}
        }}

        function highlightMultipleWordsUI(wordIdsToHighlight) {{
            document.querySelectorAll('.word-selected, .word-multi-selected').forEach(el => el.classList.remove('word-selected', 'word-multi-selected'));
            let firstElement = null;
            if (wordIdsToHighlight && wordIdsToHighlight.length > 0) {{
                wordIdsToHighlight.forEach((id, index) => {{
                    const el = document.querySelector(`#transcript-container .word[data-word-id='${{id}}']`);
                    if (el) {{
                        el.classList.add('word-multi-selected');
                        if (index === 0) firstElement = el;
                    }}
                }});
                if (firstElement) scrollToElementIfNeeded(firstElement);
            }}
        }}

        function scrollToElementIfNeeded(element) {{
            const container = document.getElementById('transcript-container');
            if (container && element) {{
                const rect = element.getBoundingClientRect();
                const containerRect = container.getBoundingClientRect();
                if (rect.bottom > containerRect.bottom || rect.top < containerRect.top) {{
                    element.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }});
                }}
            }}
        }}
        
        function handleMouseDown(event) {{
            const popup = document.querySelector('.multi-select-popup, .single-word-editor');
            if ((popup && popup.contains(event.target)) || event.button !== 0) return;

            const container = document.getElementById('transcript-container');
            if (!container || !container.contains(event.target)) return;

            potentialClickTargetForMouseUp = event.target.closest('.word, .speaker-label');
            
            const containerRect = container.getBoundingClientRect();
            dragStartX = event.clientX - containerRect.left + container.scrollLeft;
            dragStartY = event.clientY - containerRect.top + container.scrollTop;
            
            isDragging = false;
            dragSelectedWords.clear();
            dragStartWordId = null;

            if (potentialClickTargetForMouseUp) {{
                const wordId = parseInt(potentialClickTargetForMouseUp.dataset.wordId, 10);
                if (!isNaN(wordId)) {{
                    dragStartWordId = wordId;
                }}
            }}
        }}

        function handleMouseMove(event) {{
            if (document.querySelector('.multi-select-popup, .single-word-editor') || event.buttons !== 1 || dragStartWordId === null) return;
            
            const container = document.getElementById('transcript-container');
            if (!container) return;

            if (!isDragging) {{
                const containerRect = container.getBoundingClientRect();
                const currentX = event.clientX - containerRect.left + container.scrollLeft;
                const currentY = event.clientY - containerRect.top + container.scrollTop;
                if (Math.abs(currentX - dragStartX) > DRAG_THRESHOLD || Math.abs(currentY - dragStartY) > DRAG_THRESHOLD) {{
                    isDragging = true;
                    highlightMultipleWordsUI([]);
                    container.style.userSelect = 'none';
                    potentialClickTargetForMouseUp = null;
                }}
            }}

            if (!isDragging) return;

            const targetWordEl = event.target.closest('.word, .speaker-label');
            if (!targetWordEl) return;
            
            const dragEndWordId = parseInt(targetWordEl.dataset.wordId, 10);
            if (isNaN(dragEndWordId)) return;

            const minId = Math.min(dragStartWordId, dragEndWordId);
            const maxId = Math.max(dragStartWordId, dragEndWordId);
            
            const newSelection = new Set();
            for (let i = minId; i <= maxId; i++) {{
                newSelection.add(i);
            }}

            const currentHovered = new Set(Array.from(document.querySelectorAll('#transcript-container .word-drag-hover')).map(el => parseInt(el.dataset.wordId, 10)));
            
            currentHovered.forEach(id => {{
                if (!newSelection.has(id)) {{
                    document.querySelector(`#transcript-container .word[data-word-id='${{id}}']`)?.classList.remove('word-drag-hover');
                }}
            }});

            newSelection.forEach(id => {{
                if (!currentHovered.has(id)) {{
                    document.querySelector(`#transcript-container .word[data-word-id='${{id}}']`)?.classList.add('word-drag-hover');
                }}
            }});

            dragSelectedWords = newSelection;
        }}

        function handleMouseUp(event) {{
            const isMouseUpOnPopup = !!document.querySelector('.multi-select-popup, .single-word-editor')?.contains(event.target);

            const container = document.getElementById('transcript-container');
            if (container) container.style.setProperty('user-select', '');
            
            if (isDragging) {{
                const sortedSelection = Array.from(dragSelectedWords).sort((a, b) => a - b);
                selectMultipleWords(sortedSelection);
            
            }} else if (isMouseUpOnPopup) {{
                // This is not a click outside, do nothing.
            
            }} else if (potentialClickTargetForMouseUp) {{
                if (potentialClickTargetForMouseUp.classList.contains('word')) {{
                    showSingleWordEditor(event);
                }} else {{
                    const wordId = parseInt(potentialClickTargetForMouseUp.dataset.wordId, 10);
                    if (!isNaN(wordId)) {{
                        if (currentSelectedWordIds.length === 1 && currentSelectedWordIds[0] === wordId) {{
                            selectMultipleWords([]);
                        }} else {{
                            selectMultipleWords([wordId]);
                        }}
                    }}
                }}
            
            }} else if (!isMouseDownOnPopup) {{
                selectMultipleWords([]);
            }}

            isDragging = false;
            dragStartWordId = null;
            isMouseDownOnPopup = false;
            potentialClickTargetForMouseUp = null;
            document.querySelectorAll('#transcript-container .word-drag-hover').forEach(el => el.classList.remove('word-drag-hover'));
        }}

        function closeSingleWordEditor() {{
            document.getElementById('single-word-overlay')?.remove();
            document.getElementById('single-word-editor')?.remove();
        }}

        function showSingleWordEditor(event) {{
            closeMultiSelectPopup(); 
            closeSingleWordEditor(); 

            const target = event.target.closest('.word');
            if (!target) return;

            const wordId = parseInt(target.dataset.wordId, 10);
            if (isNaN(wordId)) return;

            const originalText = target.textContent.trim();

            const overlay = document.createElement('div');
            overlay.id = 'single-word-overlay';
            overlay.className = 'single-word-overlay';
            overlay.addEventListener('click', closeSingleWordEditor);

            const editor = document.createElement('div');
            editor.id = 'single-word-editor';
            editor.className = 'single-word-editor';
            
            editor.innerHTML = `<input type="text" id="single-word-input" class="single-word-input" />`;
            
            document.body.appendChild(overlay);
            document.body.appendChild(editor);

            const input = document.getElementById('single-word-input');
            input.value = originalText;
            input.style.width = `${{Math.max(100, originalText.length * 10)}}px`;
            
            editor.style.left = `${{event.clientX}}px`;
            editor.style.top = `${{event.clientY + 10}}px`;

            input.focus();
            input.select();

            input.addEventListener('keydown', (e) => {{
                if (e.key === 'Enter') {{
                    e.preventDefault();
                    const newText = input.value.trim();
                    if (newText && newText !== originalText) {{
                        saveSingleWordChange(wordId, newText);
                    }}
                    closeSingleWordEditor();
                }} else if (e.key === 'Escape') {{
                    e.preventDefault();
                    closeSingleWordEditor();
                }}
            }});
        }}

        function saveSingleWordChange(wordId, newText) {{
            if (isSaveInProgress) {{
                showStatus("A save is already in progress. Please wait.", "#ff9800");
                return;
            }}
            if (!currentWordMap[wordId]) {{
                jsLog(`Error: No data found in word map for ID ${{wordId}}`);
                showStatus("Error: Could not find word data.", "#f44336");
                return;
            }}

            isSaveInProgress = true;
            pendingUpdateWordIds = new Set([wordId]);

            const wordElement = document.querySelector(`.word[data-word-id='${{wordId}}']`);
            if (wordElement) {{
                wordElement.classList.add('word-pending-update');
            }}

            const componentBody = document.getElementById('component-body');
            let overlay = document.getElementById('editor-locked-overlay');
            if (!overlay && componentBody) {{
                overlay = document.createElement('div');
                overlay.className = 'editor-locked-overlay';
                overlay.id = 'editor-locked-overlay';
                overlay.innerText = 'Processing word correction...';
                componentBody.appendChild(overlay);
            }}

            const wordDetails = currentWordMap[wordId];
            const targetDateStr = document.getElementById('transcript-container').dataset.date;

            const payload = {{
                command_type: "CORRECT_TEXT_AND_SPEAKER",
                command_payload: {{
                    new_speaker_name: null,
                    new_text: newText,
                    corrections: [wordDetails],
                    target_date_str: targetDateStr,
                    context: "in_person",
                    chunk_id: wordDetails.chunk_id,
                    source: "gui_transcript_editor_quick_edit"
                }}
            }};
            
            fetch('http://localhost:8001/save_corrections', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(payload)
            }})
            .then(res => res.json())
            .then(result => {{
                const taskId = result.task_id;
                if (!taskId) {{
                    throw new Error("Backend command failed to queue.");
                }}
                jsLog(`Single word change task queued. Task ID: ${{taskId}}. Starting polling.`);
                pollForTaskCompletion([taskId], true); 
            }})
            .catch(error => {{
                jsLog("Save initiation failed:", error);
                showStatus(`❌ Save failed. Check logs. (${{error.message}})`, '#f44336');
                alert("A critical error occurred while saving. The UI may be out of sync. Please refresh the page (F5 or Cmd+R).");
                pollForTaskCompletion([], true);
            }});
        }}
        
        function showMultiSelectPopup(wordIds) {{
            jsLog(`showMultiSelectPopup called for ${{wordIds.length}} words.`);
            closeMultiSelectPopup(); 
            const words = wordIds.map(id => {{
                const wordEl = document.querySelector(`#transcript-container .word[data-word-id='${{id}}']`);
                return wordEl?.textContent.trim() || '';
            }}).filter(Boolean);
            
            const fullText = words.join(' '); 

            const overlay = document.createElement('div');
            overlay.id = 'popup-overlay';
            overlay.className = 'popup-overlay';
            const popup = document.createElement('div');
            popup.id = 'multi-select-popup';
            popup.className = 'multi-select-popup';

            const speakerOptionsHtml = currentEnrolledSpeakers.map(s => '<option value="' + s + '">' + s + '</option>').join('');
            const wordCountHtml = wordIds.length + ' item' + ((wordIds.length > 1 && 's') || '') + ' selected';

            let matterOptionsHtml = '<option value="">-- No Change --</option>';
            matterOptionsHtml += `<option value="{DEFAULT_MATTER_ID}">{DEFAULT_MATTER_NAME}</option>`;
            currentMatters.forEach(m => {{
                matterOptionsHtml += `<option value="${{m.matter_id}}">${{m.display_name}}</option>`;
            }});

            popup.innerHTML =
                '<div class="popup-header"><div class="popup-title">Edit Selection</div><button class="close-btn" id="popup-close-btn">×</button></div>' +
                '<div class="popup-content">' +
                    '<div class="word-count">' + wordCountHtml + '</div>' +
                    '<label for="multi-text-edit" style="font-weight: 500;">Corrected Text:</label>' +
                    '<textarea id="multi-text-edit" class="popup-form-element"></textarea>' +
                    '<div class="speaker-assignment">' +
                        '<label for="multi-speaker-select">Reassign to Speaker:</label>' +
                        '<select id="multi-speaker-select" class="popup-form-element"><option value="">Select a speaker...</option>' + speakerOptionsHtml + '</select>' +
                        '<label for="multi-new-speaker-input" style="margin-top:10px;">Or enter new speaker name:</label>' +
                        '<input type="text" id="multi-new-speaker-input" placeholder="Enter new name..." class="popup-form-element">' +
                        '<label for="matter-select" style="margin-top:10px;">Reassign to Matter:</label>' +
                        '<select id="matter-select" class="popup-form-element">' + matterOptionsHtml + '</select>' +
                        '<label for="correction-context-select" style="margin-top:10px;">Context:</label>' +
                        '<select id="correction-context-select" class="popup-form-element"><option value="in_person">In-Person</option><option value="voip">VoIP</option></select>' +
                    '</div>' +
                    '<div class="popup-volume-control">' +
                        '<label for="popup-volume-slider">🔊 Volume</label>' +
                        '<input type="range" id="popup-volume-slider" min="0" max="1" step="0.01" /><span id="popup-volume-value"></span>' +
                    '</div>' +
                '</div>' +
                '<div class="popup-buttons">' +
                    '<button class="btn btn-secondary" id="popup-cancel-btn">Cancel</button>' +
                    '<button class="btn btn-primary" id="popup-play-btn">▶ Play Audio</button>' +
                    '<button class="btn btn-success" id="popup-save-btn">💾 Save Changes</button>' +
                '</div>';

            document.body.appendChild(overlay);
            document.body.appendChild(popup);
            
            const textarea = document.getElementById('multi-text-edit');
            textarea.value = fullText;
            popup.dataset.originalText = fullText;
            autosizeTextarea(textarea);
            
            setupPopupEventListeners(overlay, popup, wordIds);

            const saveBtn = document.getElementById('popup-save-btn');
            if (saveBtn && isSaveInProgress) {{
                saveBtn.disabled = true;
                saveBtn.textContent = '💾 Saving...';
                saveBtn.title = 'An existing save is in progress. Please wait for it to complete.';
                
                const wordCountDiv = popup.querySelector('.word-count');
                if (wordCountDiv) {{
                    const warningEl = document.createElement('div');
                    warningEl.className = 'save-in-progress-warning';
                    warningEl.textContent = 'A save is already in progress. Please wait.';
                    warningEl.style.color = 'orange';
                    warningEl.style.fontWeight = 'bold';
                    warningEl.style.marginBottom = '10px';
                    wordCountDiv.parentNode.insertBefore(warningEl, wordCountDiv.nextSibling);
                }}
            }}
            
            const newSpeakerInput = document.getElementById('multi-new-speaker-input');
            if (newSpeakerInput) {{
                newSpeakerInput.focus({{ preventScroll: true }});
                jsLog('Focused on new speaker input field.');
            }}
            
            if (currentAutoPlayOnSelect && wordIds && wordIds.length > 0) {{
                jsLog("Auto-play is enabled, triggering playback.");
                setTimeout(() => playSelectedAudioFromPopup(), 100);
            }}
            
            window.currentMultiSelectionForPopup = wordIds;
        }}

        function setupPopupEventListeners(overlay, popup) {{
            overlay.addEventListener('click', (event) => {{ if (event.target === overlay) closeMultiSelectPopup(); }});
            popup.addEventListener('click', (event) => {{
                event.stopPropagation();
                const button = event.target.closest('button');
                if (!button) return;
                event.preventDefault();
                jsLog(`Popup button #${{button.id}} clicked.`);
                switch (button.id) {{
                    case 'popup-close-btn': case 'popup-cancel-btn': closeMultiSelectPopup(); break;
                    case 'popup-play-btn': playSelectedAudioFromPopup(); break;
                    case 'popup-save-btn': saveMultipleCorrectionsFromPopup(); break;
                }}
            }});
            
            const textarea = document.getElementById('multi-text-edit');
            if (textarea) {{
                textarea.addEventListener('input', () => autosizeTextarea(textarea));
            }}

            const newSpeakerInput = document.getElementById('multi-new-speaker-input');
            newSpeakerInput?.addEventListener('keydown', (event) => {{
                event.stopPropagation();
                if (event.key === 'Enter') {{
                    event.preventDefault();
                    saveMultipleCorrectionsFromPopup();
                }}
            }});
            const volumeSlider = document.getElementById('popup-volume-slider');
            const volumeValueSpan = document.getElementById('popup-volume-value');
            if (volumeSlider && volumeValueSpan) {{
                const setVolumeUI = (volume) => {{ volumeSlider.value = volume; volumeValueSpan.textContent = `${{Math.round(volume * 100)}}%`; }};
                setVolumeUI(currentAudioVolume);
                volumeSlider.addEventListener('input', (event) => {{
                    const newVolume = parseFloat(event.target.value);
                    currentAudioVolume = newVolume;
                    setVolumeUI(newVolume);
                    if (popupAudioPlayer) popupAudioPlayer.volume = newVolume;
                    localStorage.setItem(VOLUME_STORAGE_KEY, newVolume.toString());
                }});
            }}
        }}

        window.closeMultiSelectPopup = function(isActionSave = false) {{
            popupAudioPlayer?.pause();
            popupAudioPlayer = null;
            const popup = document.getElementById('multi-select-popup');
            if (popup) {{
                jsLog('Closing multi-select popup.');
                if (!isActionSave && currentAudioVolume !== lastKnownPythonVolume) {{
                    jsLog(`Volume changed from ${{lastKnownPythonVolume}} to ${{currentAudioVolume}}. Syncing with Python.`);
                    sendSelectionToStreamlit({{ action: 'volume_update', new_volume: currentAudioVolume, componentId: currentComponentId, timestamp: Date.now() }});
                    lastKnownPythonVolume = currentAudioVolume;
                }}
                popup.remove();
                document.getElementById('popup-overlay')?.remove();
            }}
            window.currentMultiSelectionForPopup = null;
        }};

        window.playSelectedAudioFromPopup = function() {{
            const wordIds = window.currentMultiSelectionForPopup;
            if (!wordIds?.length) return;
            popupAudioPlayer?.pause();
            popupAudioPlayer = null;

            if (!currentWordMap || !Object.keys(currentWordMap).length) {{
                showStatus('❌ Word map data not available.', '#f44336');
                return;
            }}

            try {{
                const yyyymmdd = currentComponentId.match(/(\\d{{8}})_/)[1];
                const dateStr = `${{yyyymmdd.slice(0, 4)}}-${{yyyymmdd.slice(4, 6)}}-${{yyyymmdd.slice(6, 8)}}`;

                const fileSegments = new Map();
                wordIds.forEach(id => {{
                    const wordDetails = currentWordMap[id];
                    if (!wordDetails) return;

                    const fileName = wordDetails.original_file_name;
                    if (!fileSegments.has(fileName)) {{
                        fileSegments.set(fileName, {{ minStart: Infinity, maxEnd: -Infinity }});
                    }}
                    const segment = fileSegments.get(fileName);
                    const wordData = wordDetails.word_data;
                    segment.minStart = Math.min(segment.minStart, wordData.start);
                    segment.maxEnd = Math.max(segment.maxEnd, wordData.end);
                }});

                const orderedFileNames = [...new Set(wordIds.map(id => currentWordMap[id]?.original_file_name).filter(Boolean))];
                const playlist = [];
                for (const fileName of orderedFileNames) {{
                    const segment = fileSegments.get(fileName);
                    if (segment) {{
                        const fileStem = fileName.substring(0, fileName.lastIndexOf('.'));
                        const snippetUrl = `http://localhost:8001/snippet?date=${{dateStr}}&file_stem=${{encodeURIComponent(fileStem)}}&start=${{segment.minStart}}&end=${{segment.maxEnd}}`;
                        playlist.push(snippetUrl);
                    }}
                }}

                if (playlist.length === 0) {{
                    showStatus('❌ Could not construct audio snippet.', '#f44336');
                    return;
                }}

                let currentTrack = 0;
                const playNextTrack = () => {{
                    if (currentTrack >= playlist.length) {{
                        jsLog("Playlist finished.");
                        popupAudioPlayer = null;
                        return;
                    }}
                    const url = playlist[currentTrack];
                    jsLog(`Playing track ${{currentTrack + 1}}/${{playlist.length}}: ${{url}}`);
                    popupAudioPlayer = new Audio(url);
                    popupAudioPlayer.volume = currentAudioVolume;
                    popupAudioPlayer.addEventListener('ended', () => {{
                        currentTrack++;
                        playNextTrack();
                    }});
                    popupAudioPlayer.play().catch(e => {{
                        showStatus(`❌ Audio failed: ${{e.message}}`, '#f44336');
                        jsLog("Audio playback error:", e);
                    }});
                }};
                playNextTrack();
            }} catch (e) {{
                showStatus(`❌ Error preparing audio: ${{e.message}}`, '#f44336');
                jsLog("Error in playSelectedAudioFromPopup:", e);
            }}
        }};

        window.saveMultipleCorrectionsFromPopup = function() {{
            const selectedWordIds = window.currentMultiSelectionForPopup;
            if (!selectedWordIds?.length) return;
            if (isSaveInProgress) {{
                showStatus("A save is already in progress. Please wait.", "#ff9800");
                return;
            }}
        
            const originalText = document.getElementById('multi-select-popup').dataset.originalText;
            const newText = document.getElementById('multi-text-edit').value.trim();
            const speakerSelect = document.getElementById('multi-speaker-select');
            const newSpeakerInput = document.getElementById('multi-new-speaker-input');
            const assignedSpeaker = newSpeakerInput?.value.trim() || speakerSelect?.value;
            const selectedMatterId = document.getElementById('matter-select').value;
        
            const textHasChanged = originalText !== newText;
            const speakerHasBeenAssigned = !!assignedSpeaker;
            const matterHasBeenAssigned = !!selectedMatterId;
        
            if (!speakerHasBeenAssigned && !textHasChanged && !matterHasBeenAssigned) {{
                showStatus('No changes detected. Please edit text, select a speaker, or choose a matter.', '#ff9800');
                return;
            }}
            if (textHasChanged && !newText) {{
                showStatus('Corrected text cannot be empty.', '#f44336');
                return;
            }}
        
            isSaveInProgress = true;
            pendingUpdateWordIds = new Set(window.currentMultiSelectionForPopup);
            
            const selectedWordElements = Array.from(pendingUpdateWordIds).map(id => document.querySelector(`.word[data-word-id='${{id}}']`)).filter(Boolean);
        
            if (speakerHasBeenAssigned && selectedWordElements.length > 0) {{
                const firstWordId = selectedWordElements[0].dataset.wordId;
                const speakerLabel = document.querySelector(`.speaker-label[data-word-id='${{firstWordId}}']`);
                if (speakerLabel) {{
                    speakerLabel.innerText = `${{assignedSpeaker}}:`;
                    speakerLabel.classList.add('speaker-label-pending-update');
                }}
            }}
            
            if (textHasChanged && selectedWordElements.length > 0) {{
                const firstWordElement = selectedWordElements[0];
                firstWordElement.innerText = ` ${{newText}}`; 
                firstWordElement.classList.add('word-pending-update');
        
                for (let i = 1; i < selectedWordElements.length; i++) {{
                    selectedWordElements[i].style.display = 'none';
                }}
                
                const componentBody = document.getElementById('component-body');
                if (componentBody) {{
                    const overlay = document.createElement('div');
                    overlay.className = 'editor-locked-overlay';
                    overlay.id = 'editor-locked-overlay';
                    overlay.innerText = 'Processing text change... Please wait.';
                    componentBody.appendChild(overlay);
                }}

            }} else {{
                 selectedWordElements.forEach(el => el.classList.add('word-pending-update'));
            }}

            const targetDateStr = document.getElementById('transcript-container').dataset.date;
            const context = document.getElementById('correction-context-select').value;
            const allCorrectionsData = selectedWordIds.map(id => currentWordMap[id]).filter(Boolean);

            const correctionsByChunk = allCorrectionsData.reduce((acc, corr) => {{
                const chunkId = corr.chunk_id;
                if (!acc[chunkId]) {{ acc[chunkId] = []; }}
                acc[chunkId].push(corr);
                return acc;
            }}, {{}});

            const commandPromises = [];
            if (textHasChanged) {{
                const newWords = newText.split(/\\s+/);
                let wordCursor = 0;
                const chunkIdsInOrder = [...new Set(allCorrectionsData.map(c => c.chunk_id))];

                for (let i = 0; i < chunkIdsInOrder.length; i++) {{
                    const chunkId = chunkIdsInOrder[i];
                    const chunkCorrections = correctionsByChunk[chunkId];
                    const originalWordCountInChunk = chunkCorrections.length;
                    
                    let wordsForThisChunkCount;
                    if (i === chunkIdsInOrder.length - 1) {{
                        wordsForThisChunkCount = newWords.length - wordCursor;
                    }} else {{
                        wordsForThisChunkCount = Math.round(newWords.length * (originalWordCountInChunk / allCorrectionsData.length));
                    }}

                    const textForThisChunk = newWords.slice(wordCursor, wordCursor + wordsForThisChunkCount).join(' ');
                    wordCursor += wordsForThisChunkCount;

                    if (textForThisChunk) {{
                        const payload = {{
                            command_type: "CORRECT_TEXT_AND_SPEAKER",
                            command_payload: {{
                                new_speaker_name: assignedSpeaker || null, new_text: textForThisChunk,
                                corrections: chunkCorrections, target_date_str: targetDateStr,
                                context: context, chunk_id: chunkId, source: "gui_transcript_editor"
                            }}
                        }};
                        commandPromises.push(fetch('http://localhost:8001/save_corrections', {{
                            method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify(payload)
                        }}).then(res => res.json()));
                    }}
                }}
            }} else if (speakerHasBeenAssigned) {{ // Only speaker change
                for (const chunkId in correctionsByChunk) {{
                    const payload = {{
                        command_type: "BATCH_CORRECT_SPEAKER_ASSIGNMENTS",
                        command_payload: {{
                            new_speaker_name: assignedSpeaker, corrections: correctionsByChunk[chunkId],
                            target_date_str: targetDateStr, context: context,
                            chunk_id: chunkId, source: "gui_transcript_editor"
                        }}
                    }};
                    commandPromises.push(fetch('http://localhost:8001/save_corrections', {{
                        method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(payload)
                    }}).then(res => res.json()));
                }}
            }}

            if (matterHasBeenAssigned) {{
                for (const chunkId in correctionsByChunk) {{
                    const chunkCorrections = correctionsByChunk[chunkId];
                    if (chunkCorrections.length === 0) continue;
                    
                    const minStartTime = Math.min(...chunkCorrections.map(c => c.word_data.start));
                    const maxEndTime = Math.max(...chunkCorrections.map(c => c.word_data.end));

                    const payload = {{
                        command_type: "UPDATE_MATTER_FOR_SPAN",
                        command_payload: {{
                            chunk_id: chunkId,
                            target_date_str: targetDateStr,
                            start_time: minStartTime,
                            end_time: maxEndTime,
                            new_matter_id: selectedMatterId,
                            source: "gui_transcript_editor"
                        }}
                    }};
                    commandPromises.push(fetch('http://localhost:8001/save_corrections', {{
                        method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(payload)
                    }}).then(res => res.json()));
                }}
            }}
            
            closeMultiSelectPopup(true);
        
            Promise.all(commandPromises)
                .then(results => {{
                    const taskIds = results.map(r => r.task_id).filter(Boolean);
                    if (taskIds.length !== commandPromises.length) {{
                        throw new Error("Some backend commands failed to queue.");
                    }}
                    jsLog(`All tasks queued. Task IDs: ${{taskIds.join(', ')}}. Starting polling.`);
                    pollForTaskCompletion(taskIds, textHasChanged || matterHasBeenAssigned);
                }})
                .catch(error => {{
                    jsLog("Save initiation failed:", error);
                    showStatus(`❌ Save failed. Check logs. (${{error.message}})`, '#f44336');
                    alert("A critical error occurred while saving. The UI may be out of sync. Please refresh the page (F5 or Cmd+R).");
                    pollForTaskCompletion([], textHasChanged || matterHasBeenAssigned);
                }});
        }};

        function reEnableSaveButtonInOpenPopup() {{
            const popup = document.getElementById('multi-select-popup');
            if (!popup) return;
            const saveBtn = document.getElementById('popup-save-btn');
            if (saveBtn && saveBtn.disabled) {{
                saveBtn.disabled = false;
                saveBtn.textContent = '💾 Save Changes';
                saveBtn.title = '';
                saveBtn.classList.remove('btn-processing');
                popup.querySelector('.save-in-progress-warning')?.remove();
            }}
        }}

        function pollForTaskCompletion(taskIds, needsFullRefresh) {{
            if (!taskIds || taskIds.length === 0) {{
                isSaveInProgress = false;
                document.getElementById('editor-locked-overlay')?.remove();
                reEnableSaveButtonInOpenPopup();
                return;
            }}
            
            const maxAttempts = 60; // 15 seconds timeout
            let currentAttempt = 0;

            const cleanupAndFinalize = (isSuccess) => {{
                isSaveInProgress = false;
                reEnableSaveButtonInOpenPopup();
                document.getElementById('editor-locked-overlay')?.remove();
                if (!isSuccess) {{
                    alert("Operation failed or timed out. Please refresh the page (F5 or Cmd+R) to ensure UI consistency.");
                }}
                pendingUpdateWordIds.clear();
            }};

            const simulateRefreshClick = () => {{
                jsLog("Attempting to programmatically click the 'Refresh Data' button.");
                const sidebar = parent.document.querySelector('[data-testid="stSidebar"]');
                if (!sidebar) {{
                    jsLog("Error: Could not find the Streamlit sidebar element. Falling back to page reload.");
                    window.location.reload();
                    return;
                }}

                const buttons = sidebar.querySelectorAll('button');
                let refreshButton = null;
                for (const button of buttons) {{
                    if (button.textContent.includes('Refresh Data')) {{
                        refreshButton = button;
                        break;
                    }}
                }}

                if (refreshButton) {{
                    jsLog("Found 'Refresh Data' button. Simulating click.", refreshButton);
                    showStatus('✅ Save complete! Refreshing view...', '#4CAF50');
                    refreshButton.click();
                }} else {{
                    jsLog("Error: Could not find the 'Refresh Data' button. Falling back to page reload.");
                    window.location.reload();
                }}
            }};

            const intervalId = setInterval(() => {{
                if (currentAttempt++ > maxAttempts) {{
                    clearInterval(intervalId);
                    jsLog(`Polling timed out for tasks: ${{taskIds.join(', ')}}`);
                    showStatus('❌ Save timed out. Confirmation failed.', '#f44336');
                    cleanupAndFinalize(false);
                    return;
                }}

                const statusPromises = taskIds.map(taskId => 
                    fetch(`http://localhost:8001/task_status?id=${{taskId}}`)
                        .then(response => {{
                            if (!response.ok) throw new Error(`Status check failed for ${{taskId}} with status: ${{response.status}}`);
                            return response.json();
                        }})
                );

                Promise.all(statusPromises)
                    .then(results => {{
                        const isAllComplete = results.every(r => r.status === 'complete');
                        const isAnyFailed = results.some(r => r.status !== 'complete' && r.status !== 'pending');

                        if (isAllComplete) {{
                            clearInterval(intervalId);
                            jsLog(`All tasks [${{taskIds.join(', ')}}] are complete.`);
                            document.getElementById('editor-locked-overlay')?.remove();
                            
                            if (needsFullRefresh) {{
                                jsLog(`Change requires full refresh. Simulating refresh button press.`);
                                saveScrollPosition();
                                setTimeout(simulateRefreshClick, 100);
                            }} else {{
                                showStatus('✅ Speaker assigned!', '#4CAF50');
                                jsLog(`Speaker-only change successful. Removing pending styles manually.`);
                                pendingUpdateWordIds.forEach(id => {{
                                    document.querySelector(`.word[data-word-id='${{id}}']`)?.classList.remove('word-pending-update');
                                    const speakerLabel = document.querySelector(`.speaker-label[data-word-id='${{id}}']`);
                                    if (speakerLabel) speakerLabel.classList.remove('speaker-label-pending-update');
                                }});
                                cleanupAndFinalize(true);
                            }}
                        }} else if (isAnyFailed) {{
                            clearInterval(intervalId);
                            const failedTask = results.find(r => r.status !== 'complete' && r.status !== 'pending');
                            jsLog(`Polling failed for at least one task. Example status: ${{failedTask?.status}}`);
                            cleanupAndFinalize(false);
                        }} else {{
                            jsLog(`Tasks still pending (attempt ${{currentAttempt}}).`);
                        }}
                    }})
                    .catch(error => {{
                        clearInterval(intervalId);
                        jsLog(`Error during polling for tasks [${{taskIds.join(', ')}}]:`, error);
                        cleanupAndFinalize(false);
                    }});
            }}, 250);
        }}

        function performAndNavigateSearch() {{
            // 1. Clear previous highlights
            document.querySelectorAll('.word-search-highlight').forEach(el => {{
                el.classList.remove('word-search-highlight');
            }});

            const query = currentSearchQuery.trim().toLowerCase();
            if (!query) return;

            // 2. Perform search
            const wordIds = Object.keys(currentWordMap).map(id => parseInt(id, 10)).sort((a, b) => a - b);
            const queryWords = query.split(/\\s+/).filter(Boolean);
            const matches = [];

            if (queryWords.length > 0) {{
                for (let i = 0; i <= wordIds.length - queryWords.length; i++) {{
                    let isMatch = true;
                    for (let j = 0; j < queryWords.length; j++) {{
                        const wordId = wordIds[i + j];
                        const wordInfo = currentWordMap[wordId];
                        if (!wordInfo || wordInfo.word_data.word.toLowerCase() !== queryWords[j]) {{
                            isMatch = false;
                            break;
                        }}
                    }}
                    if (isMatch) {{
                        matches.push({{ start_word_id: wordIds[i], length: queryWords.length }});
                        i += queryWords.length - 1;
                    }}
                }}
            }}

            if (matches.length === 0) return;

            // 3. Navigate to the correct match
            let effectiveIndex = currentSearchMatchIndex;
            if (effectiveIndex >= matches.length) {{
                effectiveIndex = matches.length - 1;
            }}
            if (effectiveIndex < 0) {{
                effectiveIndex = 0;
            }}

            const match = matches[effectiveIndex];
            let firstElement = null;

            for (let i = 0; i < match.length; i++) {{
                const wordId = match.start_word_id + i;
                const el = document.querySelector(`.word[data-word-id='${{wordId}}']`);
                if (el) {{
                    el.classList.add('word-search-highlight');
                    if (i === 0) firstElement = el;
                }}
            }}

            if (firstElement) {{
                firstElement.scrollIntoView({{ block: 'center', behavior: 'smooth' }});
            }}
        }}


        function initializeTranscriptComponent() {{
            jsLog(`Initializing component...`);
            const container = document.getElementById('transcript-container');
            if (!container) {{ setTimeout(initializeTranscriptComponent, 100); return; }}
            
            container.addEventListener('scroll', saveScrollPosition);
            container.addEventListener('mousedown', handleMouseDown);
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            document.addEventListener('keydown', handleKeyDown); 
            
            document.addEventListener('mousedown', function(event) {{
                const popup = document.querySelector('.multi-select-popup');
                if (popup && popup.contains(event.target)) {{
                    isMouseDownOnPopup = true;
                }} else {{
                    isMouseDownOnPopup = false;
                }}
            }}, true);

            const storedVolumeOnLoad = localStorage.getItem(VOLUME_STORAGE_KEY);
            if (storedVolumeOnLoad !== null) {{
                const loadedVolume = parseFloat(storedVolumeOnLoad);
                sendSelectionToStreamlit({{
                    action: 'initial_volume_sync',
                    loaded_volume: loadedVolume,
                    componentId: currentComponentId,
                    timestamp: Date.now()
                }});
                currentAudioVolume = loadedVolume;
            }}

            restoreScrollPosition();

            jsLog(`Event listeners attached.`);
            if (currentSelectedWordIds?.length > 0) {{
                highlightMultipleWordsUI(currentSelectedWordIds);
                showMultiSelectPopup(currentSelectedWordIds);
            }}

            performAndNavigateSearch(); // Run search on component load

            window.transcriptComponentInitialized = true;
            jsLog("Initialization complete. Guard flag is now ON.");
        }}

        function handleKeyDown(event) {{
            if (event.key === 'Escape') {{
                event.preventDefault(); 
                if (document.getElementById('multi-select-popup')) {{
                    selectMultipleWords([]);
                }}
            }}
        }}

        initializeTranscriptComponent(); 
        
        }} // Closes the main guard `if/else` block
    </script>
    """

    return f"{dynamic_css}{component_css}<div id='component-body'>{dialogue_html}</div>{component_js}"


def render_page():
    st.markdown("""
        <style>
            .block-container {
                padding-top: 3rem;
                padding-bottom: 0rem;
                padding-left: 3rem;
                padding-right: 3rem;
            }
            .main > div {
                padding-bottom: 0rem;
            }
            iframe {
                display: block;
            }
            footer {
                visibility: hidden;
            }
        </style>
    """, unsafe_allow_html=True)
    
    if 'volume_loaded_from_js' not in st.session_state: st.session_state.volume_loaded_from_js = False
    if 'selected_word_ids' not in st.session_state: st.session_state.selected_word_ids = []
    if 'component_counter' not in st.session_state: st.session_state.component_counter = 0
    if 'selected_date' not in st.session_state: st.session_state.selected_date = get_samson_today()
    if 'auto_play_on_select' not in st.session_state: st.session_state.auto_play_on_select = False

    # --- Sidebar Controls ---
    st.sidebar.title("Transcript Correction Tool")
    st.sidebar.markdown("---")
    
    prev_selected_date = st.session_state.selected_date
    
    selected_date_from_input = st.sidebar.date_input(
        "Select date:",
        value=st.session_state.selected_date,
        max_value=date.today(),
        key="date_selector_transcript_page"
    )

    if selected_date_from_input != prev_selected_date:
        st.session_state.selected_date = selected_date_from_input
        st.session_state.selected_word_ids = []
        _prepare_dialogue_data_for_display.clear()
        st.session_state.component_counter += 1
        st.rerun()
    
    selected_date = st.session_state.selected_date

    if st.sidebar.button("🔄 Refresh Data", use_container_width=True, key="refresh_transcript_button"):
        logger.info("GUI: Manual Refresh Data on transcript page.")
        _prepare_dialogue_data_for_display.clear()
        st.session_state.selected_word_ids = []
        st.session_state.component_counter += 1
        st.rerun()
    
    st.sidebar.markdown("---")
    
    st.sidebar.toggle(
        "Auto-play selection",
        key="auto_play_on_select",
        help="Automatically play the audio for the selected region when the edit popup appears."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Search Transcript")

    if 'search_match_index' not in st.session_state:
        st.session_state.search_match_index = 0

    def on_search_change():
        st.session_state.search_match_index = 0

    search_query = st.sidebar.text_input(
        "Find text:", 
        key="transcript_search_query",
        on_change=on_search_change,
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▲ Prev", use_container_width=True):
            if st.session_state.search_match_index > 0:
                st.session_state.search_match_index -= 1

    with col2:
        if st.button("▼ Next", use_container_width=True):
            st.session_state.search_match_index += 1


    st.sidebar.markdown("---")

    # --- Main Page Content ---

    if not selected_date:
        st.warning("Please select a date.")
        st.stop()

    try:
        target_dt = datetime.combine(selected_date, datetime.min.time())
        log_data = get_daily_log_data(target_dt)

        if not log_data or not isinstance(log_data.get("chunks"), dict) or not log_data.get("chunks"):
            st.info(f"No transcribed data found for {selected_date}.")
            render_correction_toolkit(selected_date, {}, [])
            st.stop()

        matters_from_log = log_data.get("matters", [])
        
        # (ADDITION) This block creates a context-aware list of matters for the dropdown

        # 1. Fetch all data
        all_matters = get_all_matters(include_inactive=False) # Get only active matters for the dropdown
        active_context = get_active_context(get_config())
        active_matter_id = active_context.get('matter_id') if active_context else None

        # 2. Prepare the list for sorting
        sorted_matters_for_dropdown = []
        active_matter_details = None

        # 3. Separate the active matter from the rest
        for matter in all_matters:
            # Create a 'display_name' key for the template
            matter['display_name'] = matter['name']
            if matter['matter_id'] == active_matter_id:
                active_matter_details = matter
            else:
                sorted_matters_for_dropdown.append(matter)

        # 4. Sort the non-active matters alphabetically
        sorted_matters_for_dropdown.sort(key=lambda m: m['name'])

        # 5. Prepend the active matter to the top of the list and modify its display name
        if active_matter_details:
            active_matter_details['display_name'] = f"(Active) {active_matter_details['name']}"
            sorted_matters_for_dropdown.insert(0, active_matter_details)
        
        flags = log_data.get("flags", [])
        dialogue_html, word_map = _prepare_dialogue_and_word_map_from_log(log_data, matters_from_log, flags, selected_date)
        
        enrolled_speakers = get_enrolled_speaker_names()
        if "UNKNOWN_SPEAKER" not in enrolled_speakers:
            enrolled_speakers.insert(0, "UNKNOWN_SPEAKER")
        
    except Exception as e:
        st.error(f"Failed to load data for {selected_date}: {e}")
        logger.error(f"GUI: Error preparing data for {selected_date}: {e}", exc_info=True)
        st.stop()

    if not dialogue_html or not word_map:
        st.info(f"No transcribed data found for {selected_date}.")
        render_correction_toolkit(selected_date, word_map or {}, enrolled_speakers or [])
        st.stop()

    component_key = f"transcript_comp_{selected_date.strftime('%Y%m%d')}_{st.session_state.component_counter}"

    logger.debug(f"GUI: Rendering transcript component. Component Key: {component_key}")
    logger.debug(f"GUI: State before render -> selected_word_ids: {st.session_state.selected_word_ids}")

    full_html = create_robust_transcript_component(
        dialogue_html,
        st.session_state.selected_word_ids,
        enrolled_speakers,
        component_key,
        word_map,
        sorted_matters_for_dropdown,
        st.session_state.get('global_audio_volume', 0.7),
        auto_play_on_select=st.session_state.auto_play_on_select,
        search_query=st.session_state.get("transcript_search_query", ""),
        search_match_index=st.session_state.get("search_match_index", 0)
    )

    component_result = components.html(full_html, height=850, scrolling=False)

    if isinstance(component_result, dict):
        logger.debug(f"GUI: Component returned result: {json.dumps(component_result)}")
        action = component_result.get("action", "")
        component_id_from_result = component_result.get("componentId")

        if component_id_from_result != component_key:
            logger.warning(f"Stale component result from '{component_id_from_result}' (expected '{component_key}'). Ignoring.")
        else:
            if action == "multi_select":
                selected_ids_str = component_result.get("selected_word_ids", [])
                try:
                    selected_ids = [int(sid) for sid in selected_ids_str]
                    if set(st.session_state.selected_word_ids) != set(selected_ids):
                        st.session_state.selected_word_ids = selected_ids
                        logger.info(f"GUI: {len(selected_ids)} words selected via component.")
                        st.rerun()
                except (ValueError, TypeError):
                       logger.error(f"Invalid word IDs from component multi_select: {selected_ids_str}")
            
            elif action == 'initial_volume_sync':
                if not st.session_state.volume_loaded_from_js:
                    loaded_volume = component_result.get('loaded_volume')
                    if loaded_volume is not None:
                        logger.info(f"GUI: Syncing initial global volume from component: {loaded_volume}")
                        st.session_state.global_audio_volume = float(loaded_volume)
                        st.session_state.volume_loaded_from_js = True
                        st.rerun()
            
            elif action == 'volume_update':
                new_volume = component_result.get('new_volume')
                if new_volume is not None:
                    try:
                        new_volume_float = float(new_volume)
                        if st.session_state.get('global_audio_volume') != new_volume_float:
                            logger.info(f"GUI: Global volume updated from component popup to {new_volume_float}")
                            st.session_state.global_audio_volume = new_volume_float
                    except (ValueError, TypeError):
                        logger.error(f"Invalid volume value received from component: {new_volume}")

    render_correction_toolkit(selected_date, word_map, enrolled_speakers)


if __name__ == "__main__":
    setup_logging()
    render_page()
