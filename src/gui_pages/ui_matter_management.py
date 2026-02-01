
# src/gui_pages/ui_matter_management.py

import requests
import streamlit as st
import uuid
from typing import List, Dict, Any

from src.matter_manager import get_all_matters, add_matter, update_matter, delete_matter
from src.logger_setup import logger
from main_orchestrator import queue_command_from_gui
from src.task_intelligence_manager import get_tasks_by_matter


@st.cache_data(ttl=60)
def load_matters_cached() -> List[Dict[str, Any]]:
    """Loads and caches all matters for display, including inactive ones."""
    logger.info("GUI: Loading all matters (including inactive) for management page.")
    try:
        # Step 2: Change call to include inactive matters
        return sorted(get_all_matters(include_inactive=True), key=lambda m: m.get('name', '').lower())
    except Exception as e:
        logger.error(f"GUI: Failed to load matters: {e}", exc_info=True)
        return []

def render_page():
    """Renders the Matter Management page."""
    st.title("🗂️ Matter Management")
    
    # Initialize session state for showing re-analysis options
    if "show_reanalysis_for_matter_id" not in st.session_state:
        st.session_state.show_reanalysis_for_matter_id = None

    # --- Add New Matter Form ---
    with st.expander("➕ Add New Matter", expanded=not st.session_state.show_reanalysis_for_matter_id):
        with st.form(key="add_matter_form"):
            new_matter_name = st.text_input("Matter Name*", help="A unique, descriptive name for the matter.")
            new_matter_desc = st.text_area("Description", help="A brief description of what this matter entails.")
            new_matter_keywords = st.text_input("Keywords", help="Comma-separated keywords for better matching.")
            
            submitted = st.form_submit_button("Create Matter")
            if submitted:
                if new_matter_name:
                    new_matter_id = str(uuid.uuid4())
                    new_matter_data = {
                        "matter_id": new_matter_id,
                        "name": new_matter_name,
                        "description": new_matter_desc,
                        "keywords": [k.strip() for k in new_matter_keywords.split(',') if k.strip()],
                        "source": "manual_gui",
                        "status": "active"
                    }
                    if add_matter(new_matter_data):
                        st.success(f"Matter '{new_matter_name}' created successfully!")
                        load_matters_cached.clear()
                        # Set state to show re-analysis options instead of sleeping
                        st.session_state.show_reanalysis_for_matter_id = new_matter_id
                        st.rerun() # Rerun immediately
                    else:
                        st.error(f"Failed to create matter. Does a matter with ID '{new_matter_id}' already exist?")
                else:
                    st.warning("Matter Name is a required field.")

    # --- Re-analysis Section (conditionally rendered) ---
    if st.session_state.show_reanalysis_for_matter_id:
        with st.container(border=True):
            st.subheader("🚀 Re-analyze Recent Conversations?")
            st.info("You can now re-analyze recent audio to see if any previously unidentified conversations belong to this new matter.", icon="💡")
            
            col_reanalyze1, col_reanalyze2, _ = st.columns([1, 1, 2])
            matter_id = st.session_state.show_reanalysis_for_matter_id

            def on_reanalyze_click():
                # This callback will run before the rerun, resetting the state
                st.session_state.show_reanalysis_for_matter_id = None

            with col_reanalyze1:
                if st.button("Analyze Last 24 Hours", key=f"reanalyze_1_{matter_id}", on_click=on_reanalyze_click):
                    command = {'type': 'REANALYZE_MATTERS_FOR_DATE_RANGE', 'payload': {'matter_id': matter_id, 'lookback_days': 1}}
                    queue_command_from_gui(command)
                    st.toast("Re-analysis for the last 24 hours has been queued.", icon="✅")

            with col_reanalyze2:
                if st.button("Analyze Last 7 Days", key=f"reanalyze_7_{matter_id}", on_click=on_reanalyze_click):
                    command = {'type': 'REANALYZE_MATTERS_FOR_DATE_RANGE', 'payload': {'matter_id': matter_id, 'lookback_days': 7}}
                    queue_command_from_gui(command)
                    st.toast("Re-analysis for the last 7 days has been queued.", icon="✅")

    st.divider()

    st.subheader("Existing Matters")
    
    # Step 3: Add toggle
    st.toggle("Show Inactive Matters", key="show_inactive_matters", value=False)

    matters = load_matters_cached()

    # Step 4: Filter matters based on toggle
    if not st.session_state.get("show_inactive_matters", False):
        matters = [m for m in matters if m.get('status', 'active') == 'active']

    if not matters:
        st.warning("No matters found matching the current filter.")
        return
        
    for matter in matters:
        matter_id = matter.get("matter_id", "N/A")
        with st.container(border=True):
            matter_name = matter.get('name', 'Unnamed Matter')
            
            # Step 5: Display status differently
            if matter.get('status') == 'inactive':
                st.subheader(f"{matter_name} (Inactive)")
            else:
                st.subheader(matter_name)

            st.caption(f"ID: `{matter_id}`")
            
            tasks_for_matter = get_tasks_by_matter(matter_id)
            open_tasks = [t for t in tasks_for_matter if t.get('status') not in ['completed', 'cancelled']]
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric("Open Tasks", f"{len(open_tasks)}")
            with metric_cols[1]:
                 st.metric("Total Tasks", f"{len(tasks_for_matter)}")
            
            with st.expander("Edit or Delete"):
                # Step 6: Conditional block for active vs inactive
                if matter.get('status') == 'inactive':
                    st.write("This matter is inactive.")
                    if st.button("♻️ Reactivate", key=f"reactivate_{matter_id}"):
                        # Step 7: Queue reactivate command
                        command = {'type': 'REACTIVATE_MATTER', 'payload': {'matter_id': matter_id}}
                        queue_command_from_gui(command)
                        st.toast(f"Reactivation request for '{matter.get('name')}' sent.", icon="✅")
                        load_matters_cached.clear()
                        st.rerun()
                else: # Matter is active
                    new_name = st.text_input("Name", value=matter.get("name", ""), key=f"name_{matter_id}")
                    
                    # Step 8: Add new fields
                    new_description = st.text_area(
                        "Description", 
                        value=matter.get("description", ""), 
                        key=f"desc_{matter_id}",
                        help="A brief summary of what this matter is about."
                    )
                    
                    keywords_list = matter.get("keywords", [])
                    keywords_str = ", ".join(keywords_list)
                    new_keywords_str = st.text_input(
                        "Keywords (comma-separated)", 
                        value=keywords_str, 
                        key=f"keywords_{matter_id}",
                        help="Relevant keywords that can help in classifying conversations."
                    )
                    
                    aliases_str = ", ".join(matter.get("aliases", []))
                    new_aliases = st.text_input("Aliases", value=aliases_str, key=f"aliases_{matter_id}")

                    st.markdown("---")
                    st.markdown("**Advanced Actions**")
                    if st.button("🔄 Relink Tasks to this Matter", key=f"relink_{matter_id}", help="Scans all existing tasks and re-associates any that belong to this matter based on their original transcript context."):
                        with st.spinner("Queueing background job to relink tasks..."):
                            command = {'command_type': 'RELINK_TASKS_FOR_MATTER', 'command_payload': {'matter_id': matter_id}}
                            # Use requests.post for consistency with other GUI->backend communication
                            try:
                                response = requests.post("http://localhost:8001/save_corrections", json=command, timeout=5)
                                if response.status_code == 200 and response.json().get('status') == 'queued':
                                    st.toast("Task relinking command queued successfully!", icon="✅")
                                else:
                                    st.error(f"Backend error: {response.text}")
                            except requests.RequestException as e:
                                st.error(f"Connection error: {e}")
                    
                    col1, col2, _ = st.columns([1, 1, 3])
                    with col1:
                        if st.button("Save Changes", key=f"save_{matter_id}"):
                            # Step 9: Update save logic
                            parsed_keywords = [k.strip() for k in new_keywords_str.split(',') if k.strip()]
                            updates = {
                                "name": new_name.strip(),
                                "aliases": [a.strip() for a in new_aliases.split(',') if a.strip()],
                                "description": new_description.strip(),
                                "keywords": parsed_keywords
                            }
                            if update_matter(matter_id, updates):
                                st.toast("Changes saved!", icon="✅")
                                load_matters_cached.clear()
                                st.rerun()
                            else:
                                st.error("Failed to save changes.")
                    
                    with col2:
                        if st.button("🚨 Delete", key=f"delete_{matter_id}", help="This will mark the matter as inactive. It can be reactivated later."):
                            if delete_matter(matter_id):
                                st.toast(f"Matter '{matter.get('name')}' deactivated.", icon="🗑️")
                                load_matters_cached.clear()
                                st.rerun()
                            else:
                                st.error("Failed to deactivate matter.")
