import streamlit as st
from datetime import datetime
from typing import List, Dict, Any
import requests
import json
import time
import hashlib

# --- Samson Imports ---
from src.task_intelligence_manager import _load_tasks
from src.matter_manager import get_all_matters
from src.speaker_profile_manager import get_enrolled_speaker_names
from src.logger_setup import logger

# --- Constants ---
SORT_OPTIONS = {
    "Newest First": ("created_utc", True),
    "Oldest First": ("created_utc", False),
}

# --- Data Loading (Cached) ---
@st.cache_data(ttl=30)
def load_all_task_data() -> List[Dict[str, Any]]:
    """Loads all tasks and enriches them with matter names."""
    logger.info("UI_TODO: Loading all task data.")
    all_tasks = _load_tasks()
    all_matters = get_all_matters(include_inactive=True)
    matter_map = {m['matter_id']: m['name'] for m in all_matters}
    
    for task in all_tasks:
        task['matter_name'] = matter_map.get(task.get('matter_id'), "Unassigned")
    return all_tasks

# --- Backend Communication ---
def send_task_update_command(task_id: str, updates: Dict[str, Any]):
    """Sends a command to the backend to update a task."""
    payload = {
        "command_type": "UPDATE_TASK_STATUS_FROM_GUI",
        "command_payload": {
            "task_id": task_id,
            "updates": updates
        }
    }
    try:
        response = requests.post("http://localhost:8001/save_corrections", json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
        if result.get('status') == 'queued':
            st.toast(f"✅ Update for task '{task_id[:8]}...' queued.", icon="✅")
            return True
        else:
            st.toast(f"❌ Failed to queue update: {result.get('message', 'Unknown error')}", icon="❌")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"UI_TODO: Failed to send command to backend: {e}")
        st.error(f"Connection Error: Could not send update command. Is the orchestrator running?")
        return False
def get_color_for_matter_id(matter_id: str) -> str:
    """
    Deterministically assigns a pleasant, light pastel color to a matter_id
    using HSL color space for virtually unlimited, readable colors.
    """
    if not matter_id:
        return "#f0f2f6" # A neutral light gray for unassigned matters

    # Use MD5 hash to get a deterministic integer from the matter_id string
    hash_object = hashlib.md5(matter_id.encode())
    hash_digest = hash_object.hexdigest()
    hash_int = int(hash_digest, 16)
    
    # Map the integer to a hue value (0-359 degrees on the color wheel)
    hue = hash_int % 360
    
    # Use fixed high lightness and moderate saturation for pleasant pastels
    saturation = 75
    lightness = 92
    
    return f"hsl({hue}, {saturation}%, {lightness}%)"
# --- UI Components ---
def display_edit_dialog(task: Dict[str, Any], assignees_list: List[str]):
    """Renders a dialog with a form to edit a task."""
    task_id = task['task_id']

    with st.form(key=f"edit_form_{task_id}"):
        st.subheader(f"Edit Task: {task.get('title')}")
        
        new_title = st.text_input("Title", value=task.get('title', ''))
        new_description = st.text_area("Description", value=task.get('description', ''), height=150)
        new_assignees = st.multiselect("Assignees", options=assignees_list, default=task.get('assignee_ids', []))

        submitted = st.form_submit_button("Save Changes")
        if submitted:
            updates = {}
            if new_title != task.get('title', ''):
                updates['title'] = new_title
            if new_description != task.get('description', ''):
                updates['description'] = new_description
            if sorted(new_assignees) != sorted(task.get('assignee_ids', [])):
                updates['assignee_ids'] = new_assignees
            
            if updates:
                if send_task_update_command(task_id, updates):
                    st.session_state[f'editing_{task_id}'] = False
                    st.session_state.expanded_task_id = task_id # FIX: Preserve expanded state
                    st.session_state.scroll_to_task_id = task_id # <<< FIX: Set scroll target
                    time.sleep(0.5)
                    load_all_task_data.clear()
                    st.rerun()
            else:
                st.toast("No changes were made.", icon="ℹ️")
                st.session_state[f'editing_{task_id}'] = False
                st.session_state.expanded_task_id = task_id # FIX: Preserve expanded state
                st.session_state.scroll_to_task_id = task_id # <<< FIX: Set scroll target
                st.rerun()

# --- Main Page Render Function ---
def render_page():
    st.title("✅ Task Management Dashboard")

    # --- Initialize Session State ---
    if 'expanded_task_id' not in st.session_state: st.session_state.expanded_task_id = None # FIX: Add state for expanded task
    if 'task_filter_status' not in st.session_state: st.session_state.task_filter_status = "All"
    if 'task_filter_assignee' not in st.session_state: st.session_state.task_filter_assignee = "All"
    if 'task_sort_by' not in st.session_state: st.session_state.task_sort_by = "Newest First"

    # --- Load Data ---
    tasks = load_all_task_data()
    matters = [{"matter_id": "All", "name": "All Matters"}] + get_all_matters(include_inactive=True)
    assignees = ["All"] + get_enrolled_speaker_names()
    
    # --- Sidebar for Filtering and Sorting ---
    with st.sidebar:
        st.header("Filters & Sorting")
        st.toggle("Show Completed Tasks", key="show_completed_tasks", value=False)
        st.toggle("Show Cancelled Tasks", key="show_cancelled_tasks", value=False)
        st.markdown("---") # Visual separator
        st.selectbox("Matter", options=matters, format_func=lambda m: m.get('name', 'Unknown'), key="task_filter_matter_select", index=0)
        selected_matter = st.session_state.task_filter_matter_select
        st.session_state.task_filter_matter = selected_matter['matter_id']
        st.selectbox("Assignee", assignees, key="task_filter_assignee")
        st.selectbox("Sort By", list(SORT_OPTIONS.keys()), key="task_sort_by")
        if st.button("🔄 Refresh Tasks", use_container_width=True):
            load_all_task_data.clear()
            st.rerun()

    # --- Filter and Sort Logic ---
    # (This section is the same as the previous plan, it is correct)
    filtered_tasks = tasks
    if not st.session_state.get("show_completed_tasks", False):
        filtered_tasks = [t for t in filtered_tasks if t.get('status') != 'completed']
    
    if not st.session_state.get("show_cancelled_tasks", False):
        filtered_tasks = [t for t in filtered_tasks if t.get('status') != 'cancelled']

    if st.session_state.task_filter_matter != "All":
        filtered_tasks = [t for t in filtered_tasks if t.get('matter_id') == st.session_state.task_filter_matter]
    if st.session_state.task_filter_assignee != "All":
        filtered_tasks = [t for t in filtered_tasks if st.session_state.task_filter_assignee in t.get('assignee_ids', [])]
    sort_key, reverse_order = SORT_OPTIONS[st.session_state.task_sort_by]
    filtered_tasks.sort(key=lambda t: t.get(sort_key, ""), reverse=reverse_order)
    
    # --- Display Tasks ---
    if not filtered_tasks:
        st.info("No tasks match the current filters.")
        st.stop()

    st.markdown(f"**Showing {len(filtered_tasks)} of {len(tasks)} total tasks**")

    tasks_by_matter = {}
    for task in filtered_tasks:
        matter_name = task.get('matter_name', 'Unassigned')
        matter_id = task.get('matter_id')
        # Use a tuple as the key to keep name and ID together for sorting and coloring
        matter_key = (matter_name, matter_id)
        if matter_key not in tasks_by_matter:
            tasks_by_matter[matter_key] = []
        tasks_by_matter[matter_key].append(task)

    # Sort the matters alphabetically by name for a consistent order
    sorted_matters = sorted(tasks_by_matter.items(), key=lambda item: item[0][0])

    for (matter_name, matter_id), tasks_in_matter in sorted_matters:
        color = get_color_for_matter_id(matter_id)
        
        # Display a styled header for each matter group
        st.markdown(f"""
            <div style="background-color: {color}; padding: 10px; border-radius: 8px; margin-top: 20px; margin-bottom: 10px; border: 1px solid #ddd;">
                <h3 style="margin: 0; color: #1E1E1E;">{matter_name} ({len(tasks_in_matter)})</h3>
            </div>
        """, unsafe_allow_html=True)

        for task in tasks_in_matter:
            task_id = task.get('task_id', 'no-id')
            status = task.get('status', 'unknown').replace('_', ' ').title()
            status_color = {"Pending Confirmation": "orange", "Confirmed": "blue", "In Progress": "violet", "Completed": "green", "Cancelled": "gray"}.get(status, "gray")

            # <<< FIX: Add an invisible div with the task_id as its HTML id to act as a scroll anchor.
            st.markdown(f'<div id="{task_id}"></div>', unsafe_allow_html=True)

            if f'editing_{task_id}' not in st.session_state:
                st.session_state[f'editing_{task_id}'] = False

            if st.session_state[f'editing_{task_id}']:
                display_edit_dialog(task, get_enrolled_speaker_names())
            else:
                # FIX: Control expanded state via session_state instead of hardcoding to False
                is_expanded = task_id == st.session_state.get('expanded_task_id')
                with st.expander(f"**{task.get('title', 'Untitled Task')}**", expanded=is_expanded):
                    st.markdown(f"<span style='color:{status_color};'>●</span> **Status:** {status}", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # The original code had a bug here, referencing `matter_name` instead of `task.get('matter_name')`. This is also corrected.
                        st.markdown(f"**Matter:** `{task.get('matter_name', 'N/A')}`")
                        st.markdown(f"**Assigner:** `{task.get('assigner_id', 'N/A')}`")
                    with col2:
                        st.markdown(f"**Assignees:** `{', '.join(task.get('assignee_ids', ['N/A']))}`")
                        created_time = datetime.fromisoformat(task['created_utc']).strftime("%Y-%m-%d %H:%M") if 'created_utc' in task else 'N/A'
                        st.markdown(f"**Created:** `{created_time}`")

                    st.markdown("**Description:**")
                    st.text_area("Description", value=task.get('description', ''), key=f"desc_{task_id}", disabled=True, height=100)
                    
                    # --- Version History Display ---
                    version_history = task.get('version_history', [])
                    if version_history:
                        # FIX: Add a callback to the checkbox to set the active task, preventing the expander from closing
                        def set_active_task(tid):
                            st.session_state.expanded_task_id = tid
                            st.session_state.scroll_to_task_id = tid # <<< FIX: Set scroll target
                        if st.checkbox("Show Modification History", key=f"history_toggle_{task_id}", on_change=set_active_task, args=(task_id,)):
                            st.markdown("---")
                            st.markdown("**Modification History:**")
                            history_container = st.container(height=150)
                            with history_container:
                                for entry in sorted(version_history, key=lambda x: x.get('timestamp_utc', ''), reverse=True):
                                    ts = datetime.fromisoformat(entry['timestamp_utc']).strftime('%Y-%m-%d %H:%M')
                                    author = entry.get('change_author', 'System')
                                    summary = entry.get('change_summary', 'No summary.')
                                    st.markdown(f"- **{ts}** by `{author}`: *{summary}*")
                    
                    # --- Action Buttons ---
                    st.markdown("**Actions:**")
                    action_cols = st.columns(5)
                    
                    with action_cols[0]:
                        if st.button("Confirm", key=f"confirm_{task_id}", use_container_width=True, disabled=status not in ["Pending Confirmation"]):
                            st.session_state.expanded_task_id = task_id # FIX: Preserve expanded state
                            st.session_state.scroll_to_task_id = task_id # <<< FIX: Set scroll target
                            if send_task_update_command(task_id, {"status": "confirmed"}): time.sleep(0.5); load_all_task_data.clear(); st.rerun()

                    with action_cols[1]:
                        if st.button("Edit ✏️", key=f"edit_{task_id}", use_container_width=True, disabled=status in ["Completed", "Cancelled"]):
                            st.session_state.expanded_task_id = task_id # FIX: Preserve expanded state
                            st.session_state.scroll_to_task_id = task_id # <<< FIX: Set scroll target
                            st.session_state[f'editing_{task_id}'] = True
                            st.rerun()

                    with action_cols[2]:
                        if st.button("Complete ✔️", key=f"complete_{task_id}", use_container_width=True, disabled=status not in ["Confirmed", "In Progress"]):
                            st.session_state.expanded_task_id = task_id # FIX: Preserve expanded state
                            st.session_state.scroll_to_task_id = task_id # <<< FIX: Set scroll target
                            if send_task_update_command(task_id, {"status": "completed"}): time.sleep(0.5); load_all_task_data.clear(); st.rerun()
                    
                    with action_cols[3]:
                        if st.button("Cancel ❌", key=f"cancel_{task_id}", type="secondary", use_container_width=True, disabled=status in ["Completed", "Cancelled"]):
                            st.session_state.expanded_task_id = task_id # FIX: Preserve expanded state
                            st.session_state.scroll_to_task_id = task_id # <<< FIX: Set scroll target
                            if send_task_update_command(task_id, {"status": "cancelled"}): time.sleep(0.5); load_all_task_data.clear(); st.rerun()