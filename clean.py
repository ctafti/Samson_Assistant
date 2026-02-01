# File: clean.py
# Place this script in your Samson project root directory (e.g., alongside main_orchestrator.py)

import shutil
import sys
from pathlib import Path
import json # For printing config if needed

# Ensure src is in path to load config_loader
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

try:
    from src.config_loader import get_config, CONFIG_FILE_PATH, ensure_config_exists, PROJECT_ROOT
    from src.logger_setup import setup_logging, logger # Import logger for consistency
    from src.matter_manager import _get_matters_file_path
    
    
    
except ImportError as e:
    print(f"ERROR: Could not import necessary modules from 'src'. Ensure this script is in the project root.")
    print(f"Details: {e}")
    sys.exit(1)

# --- Configuration ---
DRY_RUN = False # Set to True to see what would be deleted without actually deleting

def confirm_action(prompt_message: str) -> bool:
    """Gets user confirmation."""
    if DRY_RUN:
        print(f"DRY RUN: Would ask: {prompt_message}")
        return True # In dry run, assume yes for reporting purposes
    
    while True:
        response = input(f"{prompt_message} (y/n): ").strip().lower()
        if response == "y":
            return True
        elif response == "n":
            return False
        else:
            print("Invalid input. Please type 'y' or 'n'.")

def delete_path(path_to_delete: Path, is_file: bool = False):
    """Deletes a file or directory, handling errors."""
    if not path_to_delete.exists():
        logger.info(f"{'(DRY RUN) ' if DRY_RUN else ''}Skipping non-existent: {path_to_delete}")
        return

    try:
        if DRY_RUN:
            if is_file:
                logger.info(f"DRY RUN: Would delete file: {path_to_delete}")
            else:
                logger.info(f"DRY RUN: Would delete directory and all its contents: {path_to_delete}")
        else:
            if is_file:
                path_to_delete.unlink()
                logger.info(f"Deleted file: {path_to_delete}")
            else:
                shutil.rmtree(path_to_delete)
                logger.info(f"Deleted directory: {path_to_delete}")
    except Exception as e:
        logger.error(f"Error deleting {path_to_delete}: {e}", exc_info=True)

def main():
    # --- Config Loading ---
    if not ensure_config_exists(CONFIG_FILE_PATH):
        print(f"Config file {CONFIG_FILE_PATH} was missing and could not be created. Exiting.")
        sys.exit(1)
    
    try:
        config = get_config()
    except Exception as e:
        print(f"FATAL: Error loading configuration from {CONFIG_FILE_PATH}: {e}. Exiting.")
        sys.exit(1)

    # --- Logging Setup ---
    log_folder_path_cfg = config.get('paths', {}).get('log_folder')
    log_file_name_str_cfg = config.get('paths', {}).get('log_file_name')
    if isinstance(log_folder_path_cfg, Path) and log_file_name_str_cfg:
        setup_logging(log_folder=log_folder_path_cfg, log_file_name=log_file_name_str_cfg, console_level="INFO")
    else:
        print("Warning: Could not fully set up application logger for this script. Using basic prints.")
        global logger
        import logging
        logger = logging.getLogger("ClearTestDataScript")
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)

    logger.info("==============================================")
    logger.info("Samson Project - Test Data Cleanup Script")
    if DRY_RUN:
        logger.warning("DRY RUN IS ENABLED. NO FILES WILL BE DELETED.")
    logger.info("==============================================")

    paths_to_clean_and_recreate = [] # Directories to delete and then re-create empty
    files_to_clean = []

        # 1. Speaker Database Files
    speaker_db_dir = config.get('paths', {}).get('speaker_db_dir')
    if speaker_db_dir and isinstance(speaker_db_dir, Path):
        logger.info(f"\n--- Speaker Database Files in: {speaker_db_dir} ---")
        faiss_filename = config.get('audio_suite_settings', {}).get('faiss_index_filename')
        map_filename = config.get('audio_suite_settings', {}).get('speaker_map_filename')

        if faiss_filename:
            files_to_clean.append(speaker_db_dir / faiss_filename)
        if map_filename:
            files_to_clean.append(speaker_db_dir / map_filename)
        
        files_to_clean.append(speaker_db_dir / "speaker_profiles.json") # <<< ADD THIS LINE
    else:
        logger.warning("Speaker database directory ('paths.speaker_db_dir') not configured.")

    # 2. Temporary Audio Processing Directory
    temp_processing_dir = config.get('paths', {}).get('temp_processing_dir')
    if temp_processing_dir and isinstance(temp_processing_dir, Path):
        logger.info(f"\n--- Temporary Audio Processing Directory (job outputs): {temp_processing_dir} ---")
        paths_to_clean_and_recreate.append(temp_processing_dir)
    else:
        logger.warning("Temporary processing directory ('paths.temp_processing_dir') not configured.")

    # 3. Audio Error Folder
    audio_error_folder = config.get('paths', {}).get('audio_error_folder')
    if audio_error_folder and isinstance(audio_error_folder, Path):
        logger.info(f"\n--- Audio Error Files Directory: {audio_error_folder} ---")
        paths_to_clean_and_recreate.append(audio_error_folder)
    else:
        logger.warning("Audio error folder ('paths.audio_error_folder') not configured.")

    # 4. Log Files Directory
    log_folder = config.get('paths', {}).get('log_folder')
    if log_folder and isinstance(log_folder, Path):
        logger.info(f"\n--- Log Files Directory: {log_folder} ---")
        paths_to_clean_and_recreate.append(log_folder)
    else:
        logger.warning("Log folder ('paths.log_folder') not configured.")
        
    # 5. Daily Log JSON files (Now reads correctly from config)
    daily_logs_dir = config.get('paths', {}).get('daily_log_folder')
    if daily_logs_dir and isinstance(daily_logs_dir, Path):
        logger.info(f"\n--- Daily JSON Logs Directory: {daily_logs_dir} ---")
        paths_to_clean_and_recreate.append(daily_logs_dir)
    else:
        logger.warning("Daily JSON logs directory ('paths.daily_log_folder') not configured.")

    # 6. Master Daily Transcript Files
    #    These are stored in paths.database_folder
    master_transcript_base_folder = config.get('paths', {}).get('database_folder')
    if master_transcript_base_folder and isinstance(master_transcript_base_folder, Path):
        logger.info(f"\n--- Master Daily Transcript Files in: {master_transcript_base_folder} ---")
        if master_transcript_base_folder.exists():
            for item in master_transcript_base_folder.glob("MASTER_DIALOGUE_*.txt"):
                if item.is_file():
                    files_to_clean.append(item)
        else:
            logger.info(f"Master transcript base folder '{master_transcript_base_folder}' does not exist. Skipping.")
    else:
        logger.warning("Master transcript base folder ('paths.database_folder') not configured for master transcripts. Skipping.")

    # 7. Archive Folders
    #    This now handles both 'archived_audio_folder' and the legacy 'archive_folder' from config
    archive_folders_to_check = {
        config.get('paths', {}).get('archived_audio_folder'),
        config.get('paths', {}).get('archive_folder')
    }
    
    for archive_folder in archive_folders_to_check:
        if archive_folder and isinstance(archive_folder, Path):
            logger.info(f"\n--- Audio Archive Directory: {archive_folder} ---")
            paths_to_clean_and_recreate.append(archive_folder)
        else:
            logger.info(f"An audio archive folder path was not configured or invalid. Skipping one.")
            
    # 8. Flags Queue Directory
    flags_queue_dir = config.get('paths', {}).get('flags_queue_dir')
    if flags_queue_dir and isinstance(flags_queue_dir, Path):
        logger.info(f"\n--- Flags Queue Directory: {flags_queue_dir} ---")
        paths_to_clean_and_recreate.append(flags_queue_dir)
    else:
        logger.warning("Flags queue directory ('paths.flags_queue_dir') not configured.")

    # --- START OF FIX ---
    # 9. Monitored Audio Folder
    monitored_audio_folder = config.get('paths', {}).get('monitored_audio_folder')
    if monitored_audio_folder and isinstance(monitored_audio_folder, Path):
        logger.info(f"\n--- Monitored Audio Input Directory: {monitored_audio_folder} ---")
        paths_to_clean_and_recreate.append(monitored_audio_folder)
    else:
        logger.warning("Monitored audio folder ('paths.monitored_audio_folder') not configured.")

    # 10. Command Queue Directory
    command_queue_dir = config.get('paths', {}).get('command_queue_dir')
    if command_queue_dir and isinstance(command_queue_dir, Path):
        logger.info(f"\n--- Command Queue Directory: {command_queue_dir} ---")
        paths_to_clean_and_recreate.append(command_queue_dir)
    else:
        logger.warning("Command queue directory ('paths.command_queue_dir') not configured.")

    # 11. Core Data and State Files
    logger.info(f"\n--- Core Data and State Files ---")
    
    # Matters file (location is derived within its own module)
    files_to_clean.append(_get_matters_file_path())

    # Context and Events files (now in a configurable system_state_dir)
    system_state_dir = config.get('paths', {}).get('system_state_dir')
    if system_state_dir and isinstance(system_state_dir, Path):
        context_file = system_state_dir / "context.json"
        events_file = system_state_dir / "events.jsonl"
        
        files_to_clean.append(context_file)
        files_to_clean.append(events_file)
        logger.info(f"Targeting state files in: {system_state_dir}")
    else:
        logger.warning("System state directory ('paths.system_state_dir') not configured. Cannot clean context.json or events.jsonl.")
    # 12. Flag Snippets Directory
    flag_snippets_dir = config.get('paths', {}).get('flag_snippets_dir')
    if flag_snippets_dir and isinstance(flag_snippets_dir, Path):
        logger.info(f"\n--- Flag Snippets Directory: {flag_snippets_dir} ---")
        paths_to_clean_and_recreate.append(flag_snippets_dir)
    else:
        logger.warning("Flag snippets directory ('paths.flag_snippets_dir') not configured.")

    # Add new persistent state files from Phase 3.7.5
    data_dir = PROJECT_ROOT / "data"
    files_to_clean.append(data_dir / "context.json")
    files_to_clean.append(data_dir / "events.jsonl")
   
    # 13. System State and Tasks Folders
    logger.info(f"\n--- System State and Tasks Folders in /data ---")
    system_state_data_dir = PROJECT_ROOT / "data" / "system_state"
    tasks_data_dir = PROJECT_ROOT / "data" / "tasks"
    
    paths_to_clean_and_recreate.append(system_state_data_dir)
    paths_to_clean_and_recreate.append(tasks_data_dir)
    
    # --- Display what will be deleted ---
    # Remove duplicates from paths_to_clean_and_recreate before displaying/processing
    unique_paths_to_clean_and_recreate = sorted(list(set(p for p in paths_to_clean_and_recreate if p)))
    unique_files_to_clean = sorted(list(set(p for p in files_to_clean if p)))

    logger.info("\nTHE FOLLOWING ITEMS ARE TARGETED FOR CLEANUP:")
    logger.info("------------------------------------------")
    if not unique_paths_to_clean_and_recreate and not unique_files_to_clean:
        logger.info("No paths or files configured or found for cleanup based on current config. Exiting.")
        return

    for p_dir in unique_paths_to_clean_and_recreate:
        logger.info(f"Directory (and all contents): {p_dir.resolve()}")
    for p_file in unique_files_to_clean:
        logger.info(f"File: {p_file.resolve()}")
    logger.info("------------------------------------------")

    if not confirm_action("Proceed with deleting the items listed above?"):
        logger.info("Cleanup cancelled by user.")
        return

    logger.info("Starting cleanup...")
    for p_dir in unique_paths_to_clean_and_recreate:
        delete_path(p_dir, is_file=False)
        if not DRY_RUN and not p_dir.exists(): 
            try:
                p_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Re-created directory: {p_dir}")
            except Exception as e:
                logger.error(f"Failed to re-create directory {p_dir}: {e}")

    for p_file in unique_files_to_clean:
        delete_path(p_file, is_file=True)

    logger.info("\nCleanup process complete.")
    if DRY_RUN:
        logger.warning("DRY RUN WAS ENABLED. NO ACTUAL DELETION TOOK PLACE.")

if __name__ == "__main__":
    print("WARNING: This script will delete data based on your 'config/config.yaml'.")
    print("It targets speaker databases, temporary processing files, error files, logs, daily logs, master transcripts, and archive folders.")
    
    if not DRY_RUN:
        proceed = input("Are you absolutely sure you want to run this cleanup script? (y/n): ").strip().lower()
        if proceed != "y":
            print("Exiting without running cleanup.")
            sys.exit(0)
    else:
        print("Script is in DRY_RUN mode. It will list actions but not delete anything.")
    
    main()