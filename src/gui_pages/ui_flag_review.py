import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import re
from datetime import datetime
import json
import requests
from typing import Optional, Dict

from main_orchestrator import get_pending_flags, resolve_flag, queue_command_from_gui
from src.matter_manager import get_all_matters
from src.speaker_profile_manager import get_enrolled_speaker_names, get_all_speaker_profiles
from src.logger_setup import logger
from src.task_intelligence_manager import get_pending_task_flags

@st.cache_data(ttl=60) # Reduce TTL to see changes faster during dev
def get_data_for_review_page():
    """Fetches and caches all data needed for the review page."""
    logger.info("GUI: Caching all data for flag review page.")
    # Speaker Data
    all_profiles = get_all_speaker_profiles()
    enrolled_names = sorted([p.get('name') for p in all_profiles if p.get('name')])
    # Task Data
    pending_tasks = get_pending_task_flags()
    return enrolled_names, pending_tasks

def handle_task_resolution(action: str, task_id: str, updates: Optional[Dict] = None):
    """
    Constructs and sends a command to the backend via HTTP to resolve a task flag.
    """
    logger.info(f"Handling task resolution: action='{action}', task_id='{task_id}'")
    
    command_payload = {"task_id": task_id}
    if action == 'confirm':
        command_payload['new_status'] = 'confirmed'
    elif action == 'reject':
        command_payload['new_status'] = 'cancelled'
    elif action == 'edit' and updates:
        command_payload['updates'] = updates
        command_payload['new_status'] = 'confirmed' # Editing implicitly confirms
    else:
        st.error("Invalid task resolution action.")
        return

    http_payload = {
        "command_type": "UPDATE_TASK_STATUS",
        "command_payload": command_payload
    }
    
    try:
        response = requests.post("http://localhost:8001/save_corrections", json=http_payload, timeout=5)
        if response.status_code == 200 and response.json().get('status') == 'queued':
            st.toast(f"Task '{task_id[:8]}...' resolution queued!", icon="✅")
            if 'task_flags' in st.session_state:
                st.session_state.task_flags = [
                    task for task in st.session_state.task_flags if task.get('task_id') != task_id]
            # Clear cache to force a re-fetch of pending tasks
            get_data_for_review_page.clear()
           # st.session_state.active_tab_index = 1
            st.rerun()
            
        else:
            st.error(f"Backend error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: Could not reach backend: {e}")

def initialize_session_state():
    """Initializes session state variables for the flag review page."""
    if 'review_flags' not in st.session_state:
        st.session_state.review_flags = []
    if 'speaker_flag_index' not in st.session_state:
        st.session_state.speaker_flag_index = 0

    if 'active_tab_index' not in st.session_state:
        st.session_state.active_tab_index = 0

def render_shared_volume_audio_player(audio_url: str, component_key: str):
    """
    Renders an HTML audio player that syncs its volume with other players
    across the app using localStorage.
    """
    initial_volume = st.session_state.get('global_audio_volume', 0.7)
    
    html_string = f"""
    <div id="audio-container-{component_key}"></div>
    <script>
    (() => {{
        const container = document.getElementById('audio-container-{component_key}');
        if (container.querySelector('audio')) {{ return; }} // Already initialized

        const audio = new Audio('{audio_url}');
        audio.controls = true;
        audio.style.width = '100%';
        
        const storageKey = 'samson_audio_volume';
        let currentVolume = {initial_volume};

        // 1. Set initial volume, preferring the value from localStorage
        try {{
            const storedVolume = localStorage.getItem(storageKey);
            if (storedVolume !== null) {{
                currentVolume = parseFloat(storedVolume);
            }}
        }} catch (e) {{
            console.warn('Could not read volume from localStorage', e);
        }}
        audio.volume = currentVolume;

        // 2. When this player's volume changes, update localStorage for other components
        audio.addEventListener('volumechange', () => {{
            try {{
                // Only update if the volume has meaningfully changed to avoid loops
                if (Math.abs(audio.volume - parseFloat(localStorage.getItem(storageKey) || '{initial_volume}')) > 0.01) {{
                    localStorage.setItem(storageKey, audio.volume.toString());
                }}
            }} catch (e) {{
                console.warn('Could not save volume to localStorage', e);
            }}
        }});

        // 3. Listen for changes made by other components/pages
        window.addEventListener('storage', (event) => {{
            if (event.key === storageKey && event.newValue !== null) {{
                const newVolume = parseFloat(event.newValue);
                if (audio.volume !== newVolume) {{
                    audio.volume = newVolume;
                }}
            }}
        }});

        container.appendChild(audio);
    }})();
    </script>
    """
    components.html(html_string, height=55)

def render_page():
    """Renders the Flag Review page."""
    st.title("Flag Review Workflow")
    st.markdown("---")
    
    initialize_session_state()

    if 'speaker_flags' not in st.session_state: st.session_state.speaker_flags = []
    if 'matter_flags' not in st.session_state: st.session_state.matter_flags = []
    if 'task_flags' not in st.session_state: st.session_state.task_flags = []

    if st.button("Load All Pending Flags", type="primary"):
        try:
            enrolled_names, pending_tasks = get_data_for_review_page()
            all_pending_flags = get_pending_flags()
            
            for flag in all_pending_flags:
                flag['display_reason'] = flag.get('reason_for_flag', 'No reason specified.')
                flag_type = flag.get('flag_type')
                if flag_type in ['ambiguous_speaker', 'speaker_ambiguity']:
                    if 'candidates' in flag and isinstance(flag['candidates'], list) and flag['candidates']:
                        descriptions = []
                        for candidate in flag['candidates']:
                            desc = f"{candidate['name']} ({candidate['score'] * 100:.0f}%)"
                            if candidate.get('in_context'):
                                desc += " (in current matter)"
                            descriptions.append(desc)
                        flag['display_reason'] = f"Ambiguous Speaker. Top candidates: {'; '.join(descriptions)}."
                    elif 'tentative_speaker_name' in flag:
                        flag['display_reason'] = f"Ambiguous Speaker (Legacy): Tentatively ID'd as {flag.get('tentative_speaker_name', 'Unknown')}."

            st.session_state.speaker_flags = [f for f in all_pending_flags if f.get('flag_type') != 'matter_conflict']
            st.session_state.matter_flags = [f for f in all_pending_flags if f.get('flag_type') == 'matter_conflict']
            st.session_state.task_flags = pending_tasks
            
            st.session_state.speaker_flag_index = 0
            
            if not any([st.session_state.speaker_flags, st.session_state.matter_flags, st.session_state.task_flags]):
                st.success("No flags are currently pending review. Great job!")
            
            
        except Exception as e:
            st.error(f"Failed to load pending flags: {e}")
            logger.error(f"GUI: Error in get_pending_flags: {e}", exc_info=True)

    speaker_flags = st.session_state.get('speaker_flags', [])
    matter_conflict_flags = st.session_state.get('matter_flags', [])
    task_flags = st.session_state.get('task_flags', [])

    tab_titles = [
        f"Speaker Flags ({len(speaker_flags)})", 
        f"Task Flags ({len(task_flags)})", 
        f"Matter Conflicts ({len(matter_conflict_flags)})"
    ]
    

    st.session_state.active_tab_index = min(st.session_state.active_tab_index, len(tab_titles) - 1 if tab_titles else 0)

   # Use a persistent session state variable as the single source of truth for the active tab.
    
    tab_options = list(range(len(tab_titles)))
    selected_index = st.radio(
        label="Flag review categories",
        options=tab_options,
        index=st.session_state.active_tab_index,
        format_func=lambda index: tab_titles[index], 
        horizontal=True,
        label_visibility="collapsed",
        key="active_tab_index"
    )
    

   

    if tab_titles and st.session_state.active_tab_index == 0: # Replaces "with tab_speaker:"
        st.header(f"Review Pending Speaker Identifications ({len(speaker_flags)})")
        
        if not speaker_flags:
            st.info("No speaker flags are pending review.")
        else:
            current_index = st.session_state.speaker_flag_index

            if current_index >= len(speaker_flags):
                st.success("🎉 All pending speaker flags have been reviewed!")
                if st.button("Start New Review Session"):
                    st.session_state.speaker_flags = []
                    st.session_state.matter_flags = []
                    st.session_state.task_flags = []
                    st.rerun()
            else:
                current_flag = speaker_flags[current_index]
                flag_id = current_flag.get("flag_id", "N/A")

                st.subheader(f"Reviewing Flag {current_index + 1} of {len(speaker_flags)}", divider="rainbow")
                
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Flag ID:** `{flag_id}`")
                    st.markdown(f"**Reason:** {current_flag.get('display_reason', 'No reason specified.')}")
                    
                    tentative_name = current_flag.get('tentative_speaker_name', 'Unknown')
                    is_generic_id = tentative_name.startswith(("CUSID_", "UNMAPPED_FAISS_ID_", "UNKNOWN_"))

                    if not is_generic_id:
                        st.info(f"**System's Tentative Guess:** **{tentative_name}** (Similarity: {current_flag.get('tentative_similarity', 0.0):.2f})")
                    else:
                        st.warning(f"**System's Tentative Guess:** `{tentative_name}` (Requires manual assignment)")
                    
                    snippet_available = False
                    snippet_url = ""
                    
                    snippet_params = current_flag.get('snippet_server_params')
                    if isinstance(snippet_params, dict) and all(k in snippet_params for k in ['date', 'file_stem', 'start', 'end']):
                        try:
                            date_str = snippet_params['date']
                            file_stem = snippet_params['file_stem']
                            start_s = float(snippet_params['start'])
                            end_s = float(snippet_params['end'])
                            
                            snippet_url = f"http://localhost:8001/snippet?date={date_str}&file_stem={file_stem}&start={start_s}&end={end_s}"
                            snippet_available = True
                            logger.info(f"Generated on-the-fly snippet URL for flag {flag_id}")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Invalid snippet_server_params for flag {flag_id}: {snippet_params}. Error: {e}")
                    
                    if snippet_available:
                        render_shared_volume_audio_player(audio_url=snippet_url, component_key=f"flag-audio-{flag_id}")
                    else:
                        st.info("No audio snippet available for this flag.")

                with col2:
                    st.subheader("Resolution Actions")
                    
                    def handle_resolution(resolution: dict):
                        resolution['source'] = 'cockpit_gui_flag_review'
                        resolution['context'] = st.session_state.get("flag_review_context_select", "in_person")
                        
                        command_to_queue = resolve_flag(flag_id, resolution)
                        
                        if not command_to_queue:
                            st.error(f"Failed to prepare resolution command for flag {flag_id}. Check logs for details.")
                            return

                        http_payload = {
                            "command_type": command_to_queue.get("type"),
                            "command_payload": command_to_queue.get("payload")
                        }
                        
                        try:
                            response = requests.post(
                                "http://localhost:8001/save_corrections",
                                json=http_payload,
                                timeout=5
                            )
                            
                            if response.status_code == 200 and response.json().get('status') == 'queued':
                                st.toast(f"Resolution for flag {flag_id} sent to backend!", icon="✅")
                                st.session_state.speaker_flag_index += 1
                                st.rerun()
                            else:
                                st.error(f"Backend error. Status: {response.status_code}. Details: {response.text}")
                                logger.error(f"GUI: Backend returned an error for flag resolution: {response.status_code} - {response.text}")

                        except requests.exceptions.RequestException as e:
                            st.error("Connection Error: Could not reach the backend service.")
                            logger.error(f"GUI: Failed to send command to backend via HTTP: {e}", exc_info=True)
                    st.selectbox(
                        "Correction Context:",
                        options=["in_person", "voip"],
                        key="flag_review_context_select",
                        help="Select the recording context. This is crucial for adaptive learning."
                    )

                    if not is_generic_id:
                        if st.button(f"✅ Confirm as '{tentative_name}'", use_container_width=True):
                            handle_resolution({"action": "assign", "name": tentative_name})

                    enrolled_names, _ = get_data_for_review_page()
                    st.selectbox(
                        "Assign to existing speaker:",
                        options=[""] + enrolled_names,
                        key="reassign_speaker_select"
                    )
                    if st.button("Assign to Selected Speaker", use_container_width=True, disabled=not st.session_state.reassign_speaker_select):
                        handle_resolution({"action": "assign", "name": st.session_state.reassign_speaker_select})

                    st.text_input("...or assign a new name:", key="new_speaker_name_input")
                    if st.button("Assign as New Speaker", use_container_width=True, disabled=not st.session_state.new_speaker_name_input):
                        handle_resolution({"action": "assign", "name": st.session_state.new_speaker_name_input.strip()})

                    st.markdown("---")
                    if st.button("Skip Flag For Now", use_container_width=True):
                        handle_resolution({"action": "skip"})

    elif tab_titles and st.session_state.active_tab_index == 1: # Replaces "with tab_task:"
        st.header(f"Review Pending Task Suggestions ({len(task_flags)})", divider="rainbow")
        if not task_flags:
            st.success("All pending task suggestions have been reviewed!")
        else:
            enrolled_names, _ = get_data_for_review_page()
            for task in task_flags:
                task_id = task.get("task_id", "N/A")
                with st.container(border=True):
                    st.subheader(task.get('title', 'No Title'))
                    st.caption(f"Task ID: `{task_id}`")
                    st.markdown(f"**Description:**\n> {task.get('description', 'No Description')}")
                    
                    st.markdown(f"""
                    - **Assigner:** `{task.get('assigner_id', 'N/A')}`
                    - **Assignees:** `{', '.join(task.get('assignee_ids', []))}`
                    - **Matter:** `{task.get('matter_name', 'Unassigned')}`
                    """)

                    col1, col2, col3 = st.columns([1.5, 1, 3])
                    with col1:
                        if st.button("✅ Confirm Task", key=f"confirm_{task_id}", use_container_width=True, type="primary"):
                            handle_task_resolution('confirm', task_id)
                    with col2:
                        if st.button("❌ Reject Task", key=f"reject_{task_id}", use_container_width=True):
                            handle_task_resolution('reject', task_id)

                    with st.expander("Edit Task Details"):
                        with st.form(key=f"edit_form_{task_id}"):
                            new_title = st.text_input("Title", value=task.get('title', ''), key=f"title_{task_id}")
                            new_desc = st.text_area("Description", value=task.get('description', ''), key=f"desc_{task_id}", height=150)
                            
                            assignees_str = ", ".join(task.get('assignee_ids', []))
                            new_assignees_str = st.text_input("Assignees (comma-separated)", value=assignees_str, key=f"assignees_{task_id}")

                            if st.form_submit_button("Save & Confirm"):
                                updates = {}
                                if new_title != task.get('title'): updates['title'] = new_title
                                if new_desc != task.get('description'): updates['description'] = new_desc
                                
                                new_assignee_list = [name.strip() for name in new_assignees_str.split(',') if name.strip()]
                                if set(new_assignee_list) != set(task.get('assignee_ids', [])):
                                    updates['assignee_ids'] = new_assignee_list
                                
                                if updates:
                                    handle_task_resolution('edit', task_id, updates)
                                else:
                                    st.toast("No changes detected.", icon="🤷")

    elif tab_titles and st.session_state.active_tab_index == 2: # Replaces "with tab_matter:"
        st.header(f"Resolve High-Confidence Matter Conflicts ({len(matter_conflict_flags)})", divider="rainbow")
        if not matter_conflict_flags:
            st.success("No matter conflicts are pending review.")
        else:
            for flag in matter_conflict_flags:
                flag_id = flag.get("flag_id")
                conflicts = flag.get("conflicting_matters", [])
                
                with st.container(border=True):
                    st.markdown(f"**Flag ID:** `{flag_id}`")
                    st.text_area(
                        label="Conflicting Dialogue Snippet:",
                        value=flag.get('text_preview', 'No dialogue preview available.'),
                        height=100,
                        disabled=True
                    )

                    if len(conflicts) >= 2:
                        matter_a = conflicts[0]
                        matter_b = conflicts[1]
                        
                        st.write(f"**Conflict:** `{matter_a.get('name')}` ({matter_a.get('score',0)*100:.1f}%) vs. `{matter_b.get('name')}` ({matter_b.get('score',0)*100:.1f}%)")

                        col1, col2, col3, col4 = st.columns(4)
                        
                        def send_resolution_command(command_to_send, current_flag_obj):
                            """Generic helper to send a command and update UI state."""
                            if not command_to_send:
                                st.error(f"Failed to prepare resolution command for flag {current_flag_obj.get('flag_id')}.")
                                return

                            http_payload = {
                                "command_type": command_to_send.get("type"),
                                "command_payload": command_to_send.get("payload")
                            }

                            try:
                                response = requests.post("http://localhost:8001/save_corrections", json=http_payload, timeout=5)
                                if response.status_code == 200 and response.json().get('status') == 'queued':
                                    flag_id_to_remove = current_flag_obj.get('flag_id')
                                    st.toast(f"Action for flag `{flag_id_to_remove}` was queued successfully.", icon="✅")
                                    
                                    st.session_state.matter_flags = [
                                        f for f in st.session_state.matter_flags if f.get('flag_id') != flag_id_to_remove
                                    ]
                                   # st.session_state.active_tab_index = 2
                                    st.rerun()
                                else:
                                    st.error(f"Backend error. Status: {response.status_code}. Details: {response.text}")
                            except requests.exceptions.RequestException as e:
                                st.error(f"Connection Error: Could not reach the backend service: {e}")

                        with col1:
                            if st.button(f"Assign to `{matter_a.get('name')}`", key=f"assign_a_{flag_id}", type="primary"):
                                resolution = {"action": "resolved", "assigned_matter_id": matter_a.get('matter_id')}
                                command = resolve_flag(flag_id, resolution)
                                send_resolution_command(command, flag)
                        with col2:
                            if st.button(f"Assign to `{matter_b.get('name')}`", key=f"assign_b_{flag_id}"):
                                resolution = {"action": "resolved", "assigned_matter_id": matter_b.get('matter_id')}
                                command = resolve_flag(flag_id, resolution)
                                send_resolution_command(command, flag)
                        with col4:
                            if st.button("Dismiss", key=f"skip_{flag_id}"):
                                resolution = {"action": "skip"}
                                command = resolve_flag(flag_id, resolution)
                                send_resolution_command(command, flag)

if __name__ == '__main__':
    # This part is for local testing if you run this file directly.
    # In a real app, render_page() would be called by your main Streamlit app router.
    
    # Mocking necessary functions and objects for standalone execution
    def mock_get_pending_flags():
        return [
            {"flag_id": "flag123", "flag_type": "speaker_identification", "reason_for_flag": "Low confidence", "tentative_speaker_name": "SpeakerA", "tentative_similarity": 0.65, "text_preview": "This is a test preview for flag 1.", "summary": "Test summary 1.", "snippet_server_params": {"date": "2023-01-01", "file_stem": "test_audio_1", "start": 0, "end": 5000}},
            {"flag_id": "flag456", "flag_type": "speaker_identification", "reason_for_flag": "Possible misidentification", "tentative_speaker_name": "CUSID_12345", "tentative_similarity": 0.0, "text_preview": "Another preview text for flag 2.", "summary": "Test summary 2.", "snippet_server_params": {"date": "2023-01-02", "file_stem": "test_audio_2", "start": 1000, "end": 6000}},
            {"flag_id": "flag789", "flag_type": "matter_ambiguity", "text_preview": "A third preview about Project Phoenix.", "snippet_server_params": {"date": "2023-01-03", "file_stem": "test_audio_3", "start": 2000, "end": 8000}},
        ]

    def mock_resolve_flag(flag_id, resolution):
        logger.info(f"Mock resolve_flag called for {flag_id} with {resolution}")
        # In a real scenario, this might return a command, but for this mock, we just log.
        # For speaker flags, it returns a command. For matter flags, we build it ourselves.
        # To make the mock work for both, we return None for matter flags.
        if resolution.get("action") == "manual_assignment":
            return None
        return {"command_type": "RESOLVE_FLAG_MOCK", "flag_id": flag_id, "resolution": resolution}

    def mock_queue_command_from_gui(command):
        logger.info(f"Mock queue_command_from_gui called with {command}")
        return True # Simulate success

    def mock_get_all_speaker_profiles():
        return [
            {"name": "Alice", "faiss_id": 1},
            {"name": "Bob", "faiss_id": 2},
            {"name": "Charlie", "faiss_id": 3},
        ]

    def mock_get_matters():
        return [
            {"name": "Project Phoenix", "matter_id": "matter_001"},
            {"name": "Q3 Financials", "matter_id": "matter_002"},
            {"name": "Internal Server Upgrade", "matter_id": "matter_003"},
        ]

    # Replace real functions with mocks for standalone testing
    get_pending_flags = mock_get_pending_flags
    resolve_flag = mock_resolve_flag
    queue_command_from_gui = mock_queue_command_from_gui
    get_all_speaker_profiles = mock_get_all_speaker_profiles
    get_all_matters = mock_get_matters

    # Mock logger if not already set up
    if not hasattr(logger, 'hasHandlers') or not logger.hasHandlers():
        import logging
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)

    render_page()