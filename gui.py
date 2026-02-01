

#File: /gui.py
#Main entry point for the Samson Cockpit Streamlit application.

import streamlit as st
import sys
from pathlib import Path

# --- Path Setup ---
# Add the 'src' directory to the Python path to allow importing modules like config_loader.
# This is necessary because we are running this script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


# --- Page Imports ---
# Import the render functions from their respective modules within the gui_pages directory.
try:
    from gui_pages import ui_transcript_correction
    from gui_pages import ui_flag_review
    from gui_pages import ui_speaker_management
    from gui_pages import ui_matter_management
    from gui_pages import ui_settings
    from gui_pages import ui_todo_list
    from gui_pages import ui_workflow_management
    from logger_setup import setup_logging, logger # Import logger setup
    from config_loader import get_config # Import config loader
except ImportError as e:
    # Use st.exception for better error display in the browser if Streamlit is running
    try:
        st.exception(e)
    except:
        print(f"\nERROR: Could not import necessary Samson modules from 'src'.")
        print(f"Please ensure the 'src' directory and its contents (like logger_setup.py, config_loader.py, and the 'gui_pages' folder) exist and are accessible.")
        print(f"Error details: {e}")
    sys.exit(1)


# --- Main Application ---

@st.cache_resource
def initialize_backend():
    """
    Loads configuration and sets up logging.
    This function is decorated with @st.cache_resource, so it runs only once
    per session, preventing repeated initialization and log spam on every script re-run.
    Any exceptions raised here will be cached and re-raised on subsequent calls.
    """
    config = get_config()
    setup_logging(
        log_folder=config['paths']['log_folder'],
        log_file_name=config['paths']['log_file_name']
    )
    # This log message will now only appear on the very first run of the process.
    logger.info("--- Samson Cockpit GUI Initializing ---")
    logger.info(f"GUI: Successfully loaded configuration.")
    paths_from_config = config.get('paths', {})
    logger.info(f"GUI: Daily Log Folder Path: {paths_from_config.get('daily_log_folder')} (Type: {type(paths_from_config.get('daily_log_folder'))})")
    logger.info(f"GUI: Speaker DB Dir Path: {paths_from_config.get('speaker_db_dir')} (Type: {type(paths_from_config.get('speaker_db_dir'))})")
    logger.info(f"GUI: Flags Queue Dir Path: {paths_from_config.get('flags_queue_dir')} (Type: {type(paths_from_config.get('flags_queue_dir'))})")
    logger.info("-----------------------------------------")
    return config

def run_gui():
    """
    Sets up the Streamlit page configuration and sidebar navigation,
    and renders the selected page.
    """
    # Set the overall page configuration
    st.set_page_config(
        page_title="Samson Cockpit",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Initialize Backend ---
    # Load config and set up logging. @st.cache_resource ensures this runs only once.
    # The try/except block handles any initialization failures gracefully.
    try:
        config = initialize_backend()
    except Exception as e:
        st.error(f"FATAL: Could not initialize backend configuration or logger: {e}")
        st.warning("Please ensure 'config/config.yaml' is correctly set up and paths are valid.")
        st.exception(e) # Show the full traceback in the browser
        st.stop()

    # This creates a single source of truth for volume on the server side.
    # The client-side components will sync with this via localStorage.
    if 'global_audio_volume' not in st.session_state:
        st.session_state.global_audio_volume = 0.7


    # --- Page Definitions ---
    # This dictionary maps the user-facing page names in the sidebar
    # to the Python functions that render each page.
    PAGES = {
        "Transcript Correction": ui_transcript_correction.render_page,
        "Flag Review": ui_flag_review.render_page,
        "Task Management": ui_todo_list.render_page,
        "Workflow Management": ui_workflow_management.create_page,
        "Speaker Management": ui_speaker_management.render_page,
        "Matter Management": ui_matter_management.render_page,
        "Settings": ui_settings.render_page,
    }

    # --- Sidebar Navigation ---
    st.sidebar.title("Samson Cockpit")
    st.sidebar.markdown("---")

    # Define page-specific state keys that must be cleared on navigation to prevent state leakage.
    TRANSIENT_STATE_KEYS = [
        # Keys for ui_transcript_correction (Note: 'selected_date' and 'component_counter' persist intentionally for this page)
        'selected_word_ids',
        'volume_loaded_from_js',
        # Keys for ui_flag_review
        'review_flags',
        'review_current_index',
        'review_session_active',
        # Keys for ui_speaker_management
        'maintenance_expander_state',
        'recalc_select_all',
        'task_filter_status', 
        'task_filter_matter', 
        'task_filter_assignee', 
        'task_sort_by',
        # Any other page-specific keys can be added here. Dynamic keys like 'recalc_{faiss_id}'
        # are implicitly handled by the page refresh.
    ]

    # Initialize the current page in session state if it doesn't exist.
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Transcript Correction"

    # Create the navigation radio button.
    selection = st.sidebar.radio("Go to", list(PAGES.keys()), key="main_nav_radio")

    # Check if the user has selected a new page.
    if selection != st.session_state.current_page:
        logger.info(f"GUI: Navigating from '{st.session_state.current_page}' to '{selection}'. Clearing transient state.")
        st.session_state.current_page = selection
        
        # Clear all identified transient keys to ensure a clean slate for the new page.
        for key in TRANSIENT_STATE_KEYS:
            if key in st.session_state:
                del st.session_state[key]
        
        # Force a full re-run of the script to render the new page cleanly.
        st.rerun()

    st.sidebar.markdown("---")

    # --- Page Rendering ---
    # Get the function corresponding to the currently active page.
    page_function = PAGES[st.session_state.current_page]

    # Call the function to render the page content.
    page_function()

if __name__ == "__main__":
    try:
        run_gui()
    except Exception as e:
        # <<< MODIFICATION: Use st.exception for better error reporting in the browser >>>
        # This will catch any unhandled errors from the page rendering functions
        # and display a full, interactive traceback in the Streamlit app itself.
        logger.error(f"GUI: An unhandled exception reached the main GUI entry point: {e}", exc_info=True)
        try:
            st.exception(e)
        except:
            # Fallback to printing if Streamlit itself has failed
            print(f"\nAn unexpected error occurred: {e}")
            print("Please ensure all dependencies from Requirements.txt are installed.")
        # sys.exit(1) # Commenting out exit to allow Streamlit to handle the shutdown