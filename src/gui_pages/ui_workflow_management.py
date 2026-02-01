import streamlit as st
from pathlib import Path
import ast
import json
import requests
import time
import os
import subprocess
import sys
import re

from src.logger_setup import logger
from src.config_loader import PROJECT_ROOT, reload_config

# --- Constants ---
SORT_OPTIONS = {
    "Newest First": ("created_timestamp", True),
    "Oldest First": ("created_timestamp", False),
    "Alphabetically (A-Z)": ("name", False),
    "Alphabetically (Z-A)": ("name", True),
}

# --- Helper Functions ---
@st.cache_data(ttl=60)
def get_shared_files() -> list[str]:
    """Scans the configured windmill_shared_folder for files."""
    try:
        config = st.session_state.get('config', {})
        shared_folder_path_str = config.get('paths', {}).get('windmill_shared_folder')

        if not shared_folder_path_str:
            return []

        shared_folder_path = Path(shared_folder_path_str)
        if not shared_folder_path.exists() or not shared_folder_path.is_dir():
            return []
        
        files = []
        for f in shared_folder_path.rglob('*'):
            if f.is_file():
                relative_path = f.relative_to(shared_folder_path)
                files.append(str(relative_path))
        return sorted(files)

    except Exception as e:
        logger.error(f"GUI: Error scanning shared files directory: {e}")
        # Display a non-crashing warning in the UI if possible
        st.toast(f"Warning: Could not read shared files folder. See logs.", icon="⚠️")
        return []


@st.cache_data(ttl=30)
def get_workflows() -> list[dict[str, str]]:
    """Scans the /workflows directory and parses Python files for details."""
    workflows_dir = PROJECT_ROOT / "workflows"
    workflow_files = []
    if not workflows_dir.exists():
        return []
    
    for file_path in workflows_dir.glob("*.py"):
        if file_path.name.startswith("_"):
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content)
                docstring = ast.get_docstring(tree) or "No description provided."
                
                stem = file_path.stem
                display_name = ""
                # Check for AI-generated pattern: ai_..._uuid
                match = re.match(r'^ai_(.+)_[a-f0-9]{6}$', stem)
                if match:
                    # e.g., "ai_summarize_tasks_cfa3b2" -> "AI: Summarize Tasks"
                    base_name = match.group(1)
                    display_name = f"AI: {base_name.replace('_', ' ').title()}"
                else:
                    # e.g., "my_custom_script" -> "My Custom Script"
                    display_name = stem.replace("_", " ").title()

                workflow_files.append({
                    "name": display_name,
                    "path": str(file_path.relative_to(PROJECT_ROOT)),
                    "description": docstring.strip(),
                    "code": content,
                    "created_timestamp": file_path.stat().st_ctime
                })
        except Exception as e:
            logger.error(f"Failed to parse workflow file {file_path.name}: {e}")
            
    return workflow_files

def queue_workflow_execution(workflow_path: str, params: dict, workspace: str):
    """Sends the EXECUTE_WINDMILL_WORKFLOW command to the backend."""
    http_payload = {
        "command_type": "EXECUTE_WINDMILL_WORKFLOW",
        "command_payload": {
            "workflow_path": workflow_path,
            "input_params": params,
            "workspace": workspace,
        }
    }
    
    try:
        response = requests.post("http://localhost:8001/save_corrections", json=http_payload, timeout=10)
        if response.status_code == 200 and response.json().get('status') == 'queued':
            st.toast(f"Workflow '{Path(workflow_path).stem}' queued for execution!", icon="✅")
            return True
        else:
            st.error(f"Backend Error: {response.status_code} - {response.text}")
            return False
    except requests.RequestException as e:
        st.error(f"Connection Error: Could not reach backend to queue workflow. Is main_orchestrator.py running?")
        logger.error(f"GUI: Failed to send EXECUTE_WINDMILL_WORKFLOW command: {e}")
        return False
    
def queue_workflow_rename(workflow_path: str, new_name: str):
    """Sends the RENAME_WORKFLOW_FILE command to the backend."""
    http_payload = {
        "command_type": "RENAME_WORKFLOW_FILE",
        "command_payload": {
            "workflow_path": workflow_path, 
            "new_name": new_name
        }
    }
    try:
        response = requests.post("http://localhost:8001/save_corrections", json=http_payload, timeout=10)
        if response.status_code == 200 and response.json().get('status') == 'queued':
            st.toast(f"Workflow rename queued!", icon="✅")
            return True
        else:
            st.error(f"Backend Error: {response.status_code} - {response.text}")
            return False
    except requests.RequestException as e:
        st.error(f"Connection Error: Could not reach backend to queue rename. Error: {e}")
        logger.error(f"GUI: Failed to send RENAME_WORKFLOW_FILE command: {e}")
        return False

def queue_workflow_delete(workflow_path: str):
    """Sends the DELETE_WORKFLOW_FILE command to the backend."""
    http_payload = {
        "command_type": "DELETE_WORKFLOW_FILE",
        "command_payload": {
            "workflow_path": workflow_path
        }
    }
    try:
        response = requests.post("http://localhost:8001/save_corrections", json=http_payload, timeout=10)
        if response.status_code == 200 and response.json().get('status') == 'queued':
            st.toast(f"Workflow deletion queued!", icon="✅")
            return True
        else:
            st.error(f"Backend Error: {response.status_code} - {response.text}")
            return False
    except requests.RequestException as e:
        st.error(f"Connection Error: Could not reach backend to queue deletion. Error: {e}")
        logger.error(f"GUI: Failed to send DELETE_WORKFLOW_FILE command: {e}")
        return False

def queue_ai_generation(prompt: str):
    """Sends the GENERATE_WORKFLOW_FROM_PROMPT command to the backend."""
    http_payload = {
        "command_type": "GENERATE_WORKFLOW_FROM_PROMPT",
        "command_payload": {
            "user_prompt": prompt
        }
    }
    logger.info(f"GUI: Queuing AI workflow generation with payload: {json.dumps(http_payload, indent=2)}")
    try:
        response = requests.post("http://localhost:8001/save_corrections", json=http_payload, timeout=120)
        if response.status_code == 200 and response.json().get('status') == 'queued':
            st.toast("🤖 Workflow generation queued! The new workflow will appear below when complete.", icon="✅")
            return True
        else:
            st.error(f"Backend Error: {response.status_code} - {response.text}")
            return False
    except requests.RequestException as e:
        st.error(f"Connection Error: Could not reach backend to queue generation.")
        logger.error(f"GUI: Failed to send GENERATE_WORKFLOW_FROM_PROMPT command: {e}")
        return False

# --- Main Page Rendering ---

def create_page():
    """Renders the Workflow Management page."""
    # Per user request, reload config on every page load/rerun and clear dependent caches
    try:
        st.session_state['config'] = reload_config()
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Critical: Failed to load configuration: {e}")
        st.stop()

    st.title("⚙️ Workflow Management")
    if 'scroll_to_workflow_id' not in st.session_state:
        st.session_state.scroll_to_workflow_id = None
    if 'workflow_sort_by' not in st.session_state:
        st.session_state.workflow_sort_by = "Newest First"
    st.markdown(
        "Manage, inspect, and run custom data processing pipelines powered by Windmill. "
        "Workflows are defined as Python scripts in the `/workflows` directory."
    )
    
    config = st.session_state.get('config', {})
    windmill_config = config.get('windmill', {})
    base_url = windmill_config.get('base_url', 'http://localhost:80')
    workspace = windmill_config.get('workspace', 'samson')

    with st.sidebar:
        st.header("Actions")
        if st.button("🔄 Refresh Workflows", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.selectbox(
            "Sort By",
            options=list(SORT_OPTIONS.keys()),
            key="workflow_sort_by"
        )

        # The manual reload button is kept for convenience, even with auto-reload on.
        if st.button("⚙️ Reload Configuration", use_container_width=True, help="Reloads the config.yaml file from disk if you have made changes."):
            try:
                st.session_state['config'] = reload_config()
                st.cache_data.clear()
                st.toast("Configuration reloaded successfully!", icon="✅")
                time.sleep(1) 
                st.rerun()
            except Exception as e:
                st.error(f"Failed to reload configuration: {e}")

    col1, col2 = st.columns(2)
    with col1:
        st.link_button("↗️ Open Windmill Editor", base_url, help="Opens the Windmill web UI in a new tab for visual editing and monitoring.", use_container_width=True)
    
    output_dir_path = ""
    shared_folder = config.get('paths', {}).get('windmill_shared_folder')
    if shared_folder:
        output_dir_path = os.path.join(shared_folder, 'output')

    with col2:
        if output_dir_path and st.button("📂 Open Output Folder", use_container_width=True, help="Opens the folder where workflows save their output files."):
            if sys.platform == "darwin": # macOS
                try:
                    os.makedirs(output_dir_path, exist_ok=True)
                    subprocess.run(["open", output_dir_path], check=True)
                except Exception as e:
                    st.error(f"Could not open folder: {e}")
            else:
                st.warning(f"This feature is currently supported on macOS. Path: {output_dir_path}")

    st.divider()
            
    st.subheader("✨ Create New Workflow with AI")
    with st.form("ai_generation_form"):
        user_prompt = st.text_area(
            "Describe the workflow you want to build:",
            placeholder="e.g., 'Create a report that counts how many tasks are assigned to each person.'",
            height=100
        )
        submitted = st.form_submit_button("Generate Workflow")
        if submitted:
            if not user_prompt.strip():
                st.warning("Please describe the workflow you want to create.")
            else:
                initial_workflow_count = len(get_workflows())
                generation_queued = queue_ai_generation(user_prompt)

                if generation_queued:
                    with st.spinner("🤖 AI is building your workflow... This may take up to a minute. Please wait."):
                        success = False
                        for _ in range(60):
                            get_workflows.clear() 
                            if len(get_workflows()) > initial_workflow_count:
                                success = True
                                break 
                            time.sleep(2) 

                    if success:
                        st.success("✨ New workflow generated successfully!")
                        time.sleep(2) 
                    else:
                        st.warning("Workflow generation is taking longer than expected. It may still be in progress. The page will now refresh.")
                        time.sleep(3)
                    st.rerun()
    workflows = get_workflows()
    
    # --- Sorting Logic ---
    if st.session_state.workflow_sort_by in SORT_OPTIONS:
        sort_key, reverse_order = SORT_OPTIONS[st.session_state.workflow_sort_by]
        
        # Use .lower() for case-insensitive sorting on name, otherwise sort normally
        if sort_key == 'name':
            workflows.sort(key=lambda w: w.get(sort_key, "").lower(), reverse=reverse_order)
        else:
            workflows.sort(key=lambda w: w.get(sort_key, 0), reverse=reverse_order)
    
    if not workflows:
        st.warning("No workflows found. Add Python scripts to the `/workflows` directory to get started.")
        st.code("def main(name: str = 'world'):\n    return f'Hello, {name}'", language="python")
        return

    st.subheader(f"Discovered {len(workflows)} Workflows")

    for wf in workflows:
        wf_path = wf['path']
        workflow_id = wf['path'].replace('/', '__').replace('.', '_')
        st.markdown(f'<div id="{workflow_id}"></div>', unsafe_allow_html=True)
        
        with st.expander(f"**{wf['name']}**"):
            st.caption(wf['description'])
            
            # --- Primary Actions ---
            action_cols = st.columns([1, 2, 1])

            with action_cols[1]:
                shared_files = get_shared_files()
                file_options = [""] + shared_files
                selected_file = st.selectbox(
                    "Select a file to process (optional)",
                    options=file_options,
                    key=f"file_{wf_path}",
                    format_func=lambda x: "No file selected" if x == "" else x,
                    label_visibility="collapsed"
                )
            
            with action_cols[0]:
                if st.button("▶️ Run", key=f"run_{wf_path}", use_container_width=True, help="Run the workflow with the selected file and any additional instructions provided in the manage section."):
                    st.session_state.scroll_to_workflow_id = workflow_id
                    params = {}
                    if selected_file:
                        params['selected_file'] = selected_file
                    
                    additional_instructions = st.session_state.get(f"params_{wf_path}", "")
                    if additional_instructions and additional_instructions.strip():
                        params['additional_instructions'] = additional_instructions
                    
                    if queue_workflow_execution(wf['path'], params, workspace):
                        time.sleep(0.5) # Allow toast to be seen
                        # No rerun needed as it's a backend action

            with action_cols[2]:
                if st.button("Manage ▾", key=f"manage_{wf_path}", use_container_width=True, help="Expand to see management options like rename and delete."):
                    st.session_state.scroll_to_workflow_id = workflow_id
                    st.session_state[f"show_manage_{wf_path}"] = not st.session_state.get(f"show_manage_{wf_path}", False)
                    st.rerun()
            
            # --- Management Section (conditionally displayed) ---
            if st.session_state.get(f"show_manage_{wf_path}", False):
                st.divider()
                
                # --- Rename & Instructions ---
                col1, col2 = st.columns(2)
                with col1:
                    with st.form(key=f"rename_form_{wf_path}"):
                        st.markdown("###### Rename Workflow")
                        new_name = st.text_input("New Name", value=wf['name'], label_visibility="collapsed")
                        if st.form_submit_button("Save Name", use_container_width=True):
                            st.session_state.scroll_to_workflow_id = workflow_id
                            if new_name and new_name.strip() and new_name != wf['name']:
                                if queue_workflow_rename(wf_path, new_name):
                                    st.session_state[f"show_manage_{wf_path}"] = False
                                    time.sleep(1); st.cache_data.clear(); st.rerun()
                            else:
                                st.session_state[f"show_manage_{wf_path}"] = False
                                st.rerun()
                with col2:
                    st.markdown("###### Additional Instructions")
                    st.text_area(
                        "Plain text instructions for the worker AI, or other parameters.",
                        key=f"params_{wf_path}",
                        label_visibility="collapsed",
                        height=105,
                        help="This text will be passed as 'additional_instructions' in the parameters."
                    )
                
                # --- Code & Danger Zone ---
                st.markdown("---")
                if st.checkbox("View Code & Danger Zone", key=f"view_code_{wf_path}"):
                    st.code(wf['code'], language="python")
                    st.markdown("---")
                    st.warning("Deleting a workflow is permanent and cannot be undone.", icon="⚠️")
                    if st.button("🗑️ Delete Workflow", key=f"delete_{wf_path}", use_container_width=True, type="primary"):
                        st.session_state.scroll_to_workflow_id = workflow_id
                        if queue_workflow_delete(wf_path):
                            time.sleep(1); st.cache_data.clear(); st.rerun()