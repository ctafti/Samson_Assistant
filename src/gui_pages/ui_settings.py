# /src/gui_pages/ui_settings.py
import streamlit as st
from src.config_loader import get_config
from main_orchestrator import queue_command_from_gui
from src.speaker_profile_manager import get_enrolled_speaker_names
import requests
import logging

# Helper function to safely get nested config values
def get_nested_config(cfg, key_path, default=None):
    keys = key_path.split('.')
    val = cfg
    for key in keys:
        if isinstance(val, dict):
            val = val.get(key)
        else:
            return default
    return val if val is not None else default

# --- START FIX: Add helper function for LMStudio models ---
def get_lmstudio_models(base_url="http://localhost:1234/v1"):
    """Fetches a list of loaded models from the LM Studio API."""
    if not base_url:
        return []
    try:
        # Ensure the URL is properly formatted
        models_url = f"{base_url.rstrip('/')}/models"
        response = requests.get(models_url, timeout=3)
        response.raise_for_status()
        models_data = response.json()
        model_ids = [model['id'] for model in models_data.get('data', [])]
        return model_ids
    except requests.exceptions.RequestException as e:
        logging.warning(f"Could not connect to LM Studio at {base_url} to get models: {e}")
        st.toast(f"Could not get models from LM Studio at {base_url}.", icon="⚠️")
        return []
# --- END FIX ---

def render_page():
    st.title("⚙️ System Settings")
    st.info("Changes saved here will be written to `config.yaml`. A restart of the main application is required for them to take effect.", icon="ℹ️")

    config = get_config()
    if 'settings_saved' not in st.session_state:
        st.session_state.settings_saved = False

    # Create a more comprehensive tab structure
    tab_list = [
        "General", "Paths & Tools", "Speech-to-Text", "Speaker ID", 
        "Matter Analysis", "LLM Settings", "Voice Commands", "Integrations", "Task Intelligence", "Windmill", "Advanced"
    ]
    tabs = st.tabs(tab_list)

    # --- WIDGETS ---
    with tabs[0]: # General
        st.subheader("Timings & Timezone")
        st.selectbox(
            label="Application Timezone",
            options=["America/New_York", "America/Chicago", "Europe/London", "UTC"],
            index=["America/New_York", "America/Chicago", "Europe/London", "UTC"].index(get_nested_config(config, 'timings.assumed_recording_timezone', 'UTC')),
            key="cfg_timings_assumed_recording_timezone",
            help="The local timezone for displaying timestamps and scheduling end-of-day tasks."
        )
        st.text_input(
            label="End-of-Day Summary Time (24h format)",
            value=get_nested_config(config, 'timings.end_of_day_summary_time', '21:00'),
            key="cfg_timings_end_of_day_summary_time",
            help="The local time (e.g., '22:00') when daily summary and health check tasks should run."
        )
        st.number_input(
            label="Folder Monitor Process Delay (seconds)",
            min_value=1, max_value=60, step=1,
            value=int(get_nested_config(config, 'timings.folder_monitor_process_delay_s', 5)),
            key="cfg_timings_folder_monitor_process_delay_s",
            help="Delay to process chunk N-1 after chunk N appears."
        )
        st.text_input(
            label="Audio Processing Retry Delays (seconds, comma-separated)",
            value=", ".join(map(str, get_nested_config(config, 'timings.audio_processing_retry_delays_seconds', [60, 300, 900]))),
            key="cfg_timings_audio_processing_retry_delays_seconds",
            help="A list of delays for retrying a failed audio file processing."
        )
        st.divider()
        st.subheader("Intervals")
        st.number_input(
            label="Email Poll Interval (minutes)",
            min_value=1, max_value=120, step=1,
            value=int(get_nested_config(config, 'timings.email_poll_interval_minutes', 15)),
            key="cfg_timings_email_poll_interval_minutes"
        )
        st.number_input(
            label="Calendar Refresh Interval (minutes)",
            min_value=5, max_value=240, step=5,
            value=int(get_nested_config(config, 'timings.calendar_refresh_interval_minutes', 60)),
            key="cfg_timings_calendar_refresh_interval_minutes"
        )
    
    with tabs[1]: # Paths & Tools
        st.subheader("File System Paths")
        st.text_input("Monitored Audio Folder", value=get_nested_config(config, 'paths.monitored_audio_folder', ''), key="cfg_paths_monitored_audio_folder")
        st.text_input("Archived Audio Folder", value=get_nested_config(config, 'paths.archived_audio_folder', ''), key="cfg_paths_archived_audio_folder")
        st.text_input("Audio Error Folder", value=get_nested_config(config, 'paths.audio_error_folder', ''), key="cfg_paths_audio_error_folder")
        st.text_input("Database Folder", value=get_nested_config(config, 'paths.database_folder', ''), key="cfg_paths_database_folder")
        st.text_input("Log Folder", value=get_nested_config(config, 'paths.log_folder', ''), key="cfg_paths_log_folder")
        st.text_input("Daily Log Folder", value=get_nested_config(config, 'paths.daily_log_folder', ''), key="cfg_paths_daily_log_folder")
        st.text_input("Speaker DB Directory", value=get_nested_config(config, 'paths.speaker_db_dir', ''), key="cfg_paths_speaker_db_dir")
        st.text_input("Temp Processing Directory", value=get_nested_config(config, 'paths.temp_processing_dir', ''), key="cfg_paths_temp_processing_dir")

        st.divider()
        st.subheader("External Tools")
        st.text_input(
            label="FFmpeg Path",
            value=get_nested_config(config, 'tools.ffmpeg_path', 'ffmpeg'),
            key="cfg_tools_ffmpeg_path",
            help="Path to the ffmpeg executable. If in system PATH, 'ffmpeg' is sufficient."
        )

    with tabs[2]: # Speech-to-Text
        st.subheader("STT Engine")
        st.selectbox(
            label="Engine",
            options=["parakeet_mlx", "openai_whisper"],
            index=["parakeet_mlx", "openai_whisper"].index(get_nested_config(config, 'stt.engine', 'parakeet_mlx')),
            key="cfg_stt_engine"
        )
        st.divider()
        st.subheader("Parakeet MLX Settings")
        st.text_input(
            label="Model Path or Hub ID",
            value=get_nested_config(config, 'stt.parakeet_mlx.model_path', ''),
            key="cfg_stt_parakeet_mlx_model_path"
        )
        st.text_input(
            label="Language",
            value=get_nested_config(config, 'stt.parakeet_mlx.CLI.language', 'en'),
            key="cfg_stt_parakeet_mlx_CLI_language"
        )
        st.toggle(
            label="Use VAD Filter",
            value=bool(get_nested_config(config, 'stt.parakeet_mlx.CLI.vad_filter', True)),
            key="cfg_stt_parakeet_mlx_CLI_vad_filter"
        )

    with tabs[3]: # Speaker ID
        st.subheader("Core Models")
        st.text_input("Diarization Model", value=get_nested_config(config, 'audio_suite_settings.diarization_model', ''), key="cfg_audio_suite_settings_diarization_model")
        st.text_input("Embedding Model", value=get_nested_config(config, 'audio_suite_settings.embedding_model', ''), key="cfg_audio_suite_settings_embedding_model")
        st.text_input("Punctuation Model", value=get_nested_config(config, 'audio_suite_settings.punctuation_model', ''), key="cfg_audio_suite_settings_punctuation_model")

        st.divider()
        st.subheader("Identification & Enrollment")
        st.slider("Similarity Threshold", min_value=0.50, max_value=0.95, step=0.01, value=float(get_nested_config(config, 'audio_suite_settings.similarity_threshold', 0.63)), key="cfg_audio_suite_settings_similarity_threshold")
        st.slider("Consolidate New Speaker Threshold", min_value=0.70, max_value=0.99, step=0.01, value=float(get_nested_config(config, 'audio_suite_settings.consolidate_new_speaker_threshold', 0.88)), key="cfg_audio_suite_settings_consolidate_new_speaker_threshold")
        st.number_input("Min Segments for Enrollment Prompt", min_value=1, max_value=10, step=1, value=int(get_nested_config(config, 'audio_suite_settings.min_segments_for_enroll_prompt', 1)), key="cfg_audio_suite_settings_min_segments_for_enroll_prompt")
        
        st.divider()
        st.subheader("Live Refinement")
        st.toggle("Enable Live Embedding Refinement", value=bool(get_nested_config(config, 'audio_suite_settings.enable_live_embedding_refinement', True)), key="cfg_audio_suite_settings_enable_live_embedding_refinement")
        st.slider("Refinement Min Similarity", min_value=0.70, max_value=0.95, step=0.01, value=float(get_nested_config(config, 'audio_suite_settings.live_refinement_min_similarity', 0.82)), key="cfg_audio_suite_settings_live_refinement_min_similarity")
        st.slider("Refinement Min Segment Duration (s)", min_value=1.0, max_value=10.0, step=0.1, value=float(get_nested_config(config, 'audio_suite_settings.live_refinement_min_segment_duration_s', 3.8)), key="cfg_audio_suite_settings_live_refinement_min_segment_duration_s")

    with tabs[4]: # Matter Analysis
        st.subheader("Assignment & Context")
        st.slider("Matter Assignment Similarity", min_value=0.60, max_value=0.95, step=0.01, value=float(get_nested_config(config, 'context_management.matter_assignment_min_similarity', 0.75)), key="cfg_context_management_matter_assignment_min_similarity")
        st.slider("Auto-Create New Matter Threshold", min_value=0.70, max_value=0.99, step=0.01, value=float(get_nested_config(config, 'context_management.auto_create_matter_threshold', 0.88)), key="cfg_context_management_auto_create_matter_threshold")
        st.number_input("Silence Reset Threshold (s)", min_value=10, max_value=600, step=5, value=int(get_nested_config(config, 'context_management.silence_reset_threshold_seconds', 90)), key="cfg_context_management_silence_reset_threshold_seconds")
        st.number_input("New Matter Trigger Duration (s)", min_value=10, max_value=300, step=5, value=int(get_nested_config(config, 'context_management.new_matter_trigger_duration_s', 45)), key="cfg_context_management_new_matter_trigger_duration_s")
        # --- START FIX: Add missing Matter Analysis widget ---
        st.number_input("Matter Change Threshold", min_value=1, max_value=10, step=1, value=int(get_nested_config(config, 'context_management.matter_change_threshold', 3)), key="cfg_context_management_matter_change_threshold", help="Number of consecutive differing matter assignments to trigger a context change.")
        # --- END FIX ---
        st.toggle("Enable Matter Context Stickiness", value=bool(get_nested_config(config, 'context_management.enable_matter_context_stickiness', True)), key="cfg_context_management_enable_matter_context_stickiness")
        st.slider("Matter Stickiness Bonus", min_value=0.01, max_value=0.20, step=0.01, value=float(get_nested_config(config, 'context_management.matter_context_stickiness_bonus', 0.05)), key="cfg_context_management_matter_context_stickiness_bonus")
        
        st.divider()
        st.subheader("Conflict Flagging")
        st.toggle("Enable Matter Conflict Flagging", value=bool(get_nested_config(config, 'context_management.enable_matter_conflict_flagging', True)), key="cfg_context_management_enable_matter_conflict_flagging")
        st.slider("Conflict High Confidence Threshold", min_value=0.70, max_value=0.99, step=0.01, value=float(get_nested_config(config, 'context_management.matter_conflict_high_confidence_threshold', 0.85)), key="cfg_context_management_matter_conflict_high_confidence_threshold")
        st.slider("Conflict Delta Threshold", min_value=0.01, max_value=0.10, step=0.01, value=float(get_nested_config(config, 'context_management.matter_conflict_delta_threshold', 0.03)), key="cfg_context_management_matter_conflict_delta_threshold")

        st.divider()
        st.subheader("Conversation Triggers")
        st.text_area(
            "End of Conversation Triggers (one per line)",
            value="\n".join(get_nested_config(config, 'context_management.end_of_conversation_triggers', [])),
            key="cfg_context_management_end_of_conversation_triggers",
            height=150
        )

    # --- START FIX: Replace entire LLM Settings tab ---
    with tabs[5]: # LLM Settings
        st.subheader("Global Settings")
        st.selectbox("Global ML Device", options=["mps", "cpu", "cuda"], index=["mps", "cpu", "cuda"].index(get_nested_config(config, 'global_ml_device', 'mps')), key="cfg_global_ml_device")
        
        st.divider()
        llm_profiles = ["main_llm", "classification_llm", "summary_llm", "llm_command_parser", "llm_task_extractor"]
        lm_studio_models = None # Cache models to avoid repeated API calls

        for profile in llm_profiles:
            st.subheader(f"{profile.replace('_', ' ').title()} Profile")
            
            # Use columns for a cleaner layout
            col1, col2 = st.columns(2)
            
            with col1:
                # The key for st.text_input must be unique for Streamlit to track state
                provider_key = f"cfg_llm_{profile}_provider"
                provider = st.text_input(f"Provider##{profile}", value=get_nested_config(config, f'llm.{profile}.provider', 'lmstudio'), key=provider_key, help="e.g., lmstudio, ollama, openai")
            
            with col2:
                base_url_key = f"cfg_llm_{profile}_base_url"
                base_url = st.text_input(f"Base URL##{profile}", value=get_nested_config(config, f'llm.{profile}.base_url', 'http://localhost:1234/v1'), key=base_url_key, help="API base URL, e.g., for LM Studio.")

            current_model = get_nested_config(config, f'llm.{profile}.model_name', '')
            model_name_key = f"cfg_llm_{profile}_model_name"

            # Logic for model selection dropdown
            if provider.lower() == 'lmstudio':
                if lm_studio_models is None: # Fetch only once per page render for the current base_url
                    # Pass the specific base_url for this profile
                    lm_studio_models = get_lmstudio_models(st.session_state.get(base_url_key, base_url))
                
                if lm_studio_models: # If models were fetched successfully
                    try:
                        # Ensure current model is in the list to avoid errors with index()
                        if current_model and current_model not in lm_studio_models:
                            lm_studio_models.insert(0, current_model)
                        model_index = lm_studio_models.index(current_model) if current_model in lm_studio_models else 0
                    except ValueError:
                        model_index = 0
                    
                    st.selectbox(f"Model Name##{profile}", options=lm_studio_models, index=model_index, key=model_name_key)
                else: # Fallback to text input if connection fails or no models
                    st.text_input(f"Model Name##{profile}", value=current_model, key=model_name_key, help="Enter model name. Could not fetch from LMStudio.")
            else: # For other providers like ollama, openai, etc.
                st.text_input(f"Model Name##{profile}", value=current_model, key=model_name_key)

            temp_col, gpu_col = st.columns(2)
            with temp_col:
                st.slider(f"Temperature##{profile}", min_value=0.0, max_value=2.0, step=0.1, value=float(get_nested_config(config, f'llm.{profile}.temperature', 0.3)), key=f"cfg_llm_{profile}_temperature")
            with gpu_col:
                st.number_input(f"GPU Layers##{profile}", min_value=-1, max_value=100, step=1, value=int(get_nested_config(config, f'llm.{profile}.num_gpu', -1)), key=f"cfg_llm_{profile}_num_gpu", help="-1 for all layers")
    # --- END FIX ---

    with tabs[6]: # Voice Commands
        st.subheader("General Settings")
        st.toggle("Enable Voice Commands", value=bool(get_nested_config(config, 'voice_commands.enabled', True)), key="cfg_voice_commands_enabled")
        
        enrolled_speakers = get_enrolled_speaker_names()
        current_user = get_nested_config(config, 'voice_commands.user_speaker_name', '')
        user_index = enrolled_speakers.index(current_user) if current_user in enrolled_speakers else 0
        
        st.selectbox(
            label="Voice Command User", options=enrolled_speakers, index=user_index,
            key="cfg_voice_commands_user_speaker_name", help="Select the enrolled speaker authorized to give voice commands."
        )
        st.slider(
            label="Vocal Override Lookback (seconds)", min_value=15, max_value=300, step=5,
            value=int(get_nested_config(config, 'voice_commands.vocal_override_lookback_seconds', 60)),
            key="cfg_voice_commands_vocal_override_lookback_seconds"
        )
        st.slider(
            label="Voice Command Confidence Threshold", min_value=0.70, max_value=0.99, step=0.01,
            value=float(get_nested_config(config, 'voice_commands.voice_command_confidence_threshold', 0.85)),
            key="cfg_voice_commands_voice_command_confidence_threshold"
        )
        st.text_area(
            "Trigger Phrases (one per line)",
            value="\n".join(get_nested_config(config, 'voice_commands.trigger_phrases', [])),
            key="cfg_voice_commands_trigger_phrases",
            height=100
        )

    with tabs[7]: # Integrations
        st.subheader("Signal Messaging")
        st.text_input("Samson Bot Phone Number", value=get_nested_config(config, 'signal.samson_phone_number', ''), key="cfg_signal_samson_phone_number")
        st.text_input("Recipient Phone Number", value=get_nested_config(config, 'signal.recipient_phone_number', ''), key="cfg_signal_recipient_phone_number")
        st.text_input("signal-cli Path", value=get_nested_config(config, 'signal.signal_cli_path', 'signal-cli'), key="cfg_signal_signal_cli_path")
        st.text_input("signal-cli Data Path", value=get_nested_config(config, 'signal.signal_cli_data_path', ''), key="cfg_signal_signal_cli_data_path")
        st.divider()
        st.subheader("Hugging Face")
        st.text_input("Hugging Face Token", value=get_nested_config(config, 'audio_suite_settings.hf_token', ''), key="cfg_audio_suite_settings_hf_token", type="password", help="Required for models like pyannote/speaker-diarization-3.1")

    with tabs[8]: # Task Intelligence
        st.subheader("Task Intelligence Engine")
        st.toggle(
            "Enable Task Intelligence",
            value=bool(get_nested_config(config, 'task_intelligence.enabled', False)),
            key="cfg_task_intelligence_enabled"
        )
        st.text_input(
            "Task Data File Path",
            value=get_nested_config(config, 'task_intelligence.task_data_file', 'data/tasks/tasks.jsonl'),
            key="cfg_task_intelligence_task_data_file"
        )
        st.number_input(
            "Task Auto-Confirmation Cooldown (seconds)",
            min_value=60, max_value=3600, step=60,
            value=int(get_nested_config(config, 'task_intelligence.task_creation_cooldown_s', 300)),
            key="cfg_task_intelligence_task_creation_cooldown_s",
            help="Time to wait before a 'pending' task is automatically confirmed."
        )
        st.text_area(
            "Task Completion Keywords (one per line)",
            value="\n".join(get_nested_config(config, 'task_intelligence.completion_keywords', [])),
            key="cfg_task_intelligence_completion_keywords",
            height=150
        )
        # --- START FIX: Add missing Task Intelligence widget ---
        st.slider(
            "Task Similarity Threshold", min_value=0.50, max_value=0.99, step=0.01,
            value=float(get_nested_config(config, 'task_intelligence.task_similarity_threshold', 0.80)),
            key="cfg_task_intelligence_task_similarity_threshold",
            help="Similarity score required to consider a new task a duplicate of an existing one."
        )
        # --- END FIX ---


    with tabs[9]: # Windmill
        st.subheader("Windmill Workflow Engine")
        st.info(
            "Configure the connection to your Windmill instance for workflow execution.",
            icon="ℹ️"
        )
        st.toggle(
            "Enable Windmill Integration",
            value=bool(get_nested_config(config, 'windmill.enabled', False)),
            key="cfg_windmill_enabled"
        )
        st.text_input(
            "Workspace",
            value=get_nested_config(config, 'windmill.workspace', ''),
            key="cfg_windmill_workspace"
        )
        st.text_input(
            "Base URL (for API)",
            value=get_nested_config(config, 'windmill.base_url', 'http://localhost:80'),
            key="cfg_windmill_base_url",
            help="The base URL of your Windmill instance (e.g., http://localhost:8000)."
        )
        st.text_input(
            "Remote URL (for CLI)",
            value=get_nested_config(config, 'windmill.remote', 'http://localhost:80'),
            key="cfg_windmill_remote",
            help="The remote URL used by the Windmill CLI."
        )
        st.selectbox(
            "Execution Method",
            options=["cli", "api"],
            index=["cli", "api"].index(get_nested_config(config, 'windmill.execution_method', 'cli')),
            key="cfg_windmill_execution_method"
        )
        st.text_input(
            "API Token",
            value=get_nested_config(config, 'windmill.api_token', ''),
            key="cfg_windmill_api_token",
            type="password",
            help="API token for authenticating with Windmill."
        )

        st.divider()
        st.subheader("AI Workflow Generation")
        st.info(
            "Configure the LLM profiles used by Windmill to generate new workflows from conversations.",
            icon="🤖"
        )

        # Get profiles from config, default to an empty dict if not found
        workflow_profiles = get_nested_config(config, 'llm.profiles', {})
        profile_names = list(workflow_profiles.keys())
        current_profile = get_nested_config(config, 'llm.workflow_generator_profile', '')

        # Ensure current_profile is in the list to avoid index errors
        if current_profile and current_profile not in profile_names:
            profile_names.insert(0, current_profile)

        profile_index = profile_names.index(current_profile) if current_profile in profile_names else 0

        st.selectbox(
            "Active Workflow Generator Profile",
            options=profile_names,
            index=profile_index,
            key="cfg_llm_workflow_generator_profile",
            help="Select the default LLM profile for generating Windmill workflows."
        )

        st.write("Edit Workflow Generator Profiles:")
        for profile_name, profile_data in workflow_profiles.items():
            with st.expander(f"Profile: `{profile_name}`"):
                st.text_input(
                    "Provider",
                    value=profile_data.get('provider', ''),
                    key=f"cfg_llm_profiles_{profile_name}_provider"
                )
                st.text_input(
                    "Model Name",
                    value=profile_data.get('model_name', ''),
                    key=f"cfg_llm_profiles_{profile_name}_model_name"
                )
                if 'api_key' in profile_data:
                    st.text_input(
                        "API Key",
                        value=profile_data.get('api_key', ''),
                        key=f"cfg_llm_profiles_{profile_name}_api_key",
                        type="password"
                    )
                if 'base_url' in profile_data:
                    st.text_input(
                        "Base URL",
                        value=profile_data.get('base_url', ''),
                        key=f"cfg_llm_profiles_{profile_name}_base_url"
                    )
                if 'temperature' in profile_data:
                    st.slider(
                        "Temperature",
                        min_value=0.0, max_value=2.0, step=0.1,
                        value=float(profile_data.get('temperature', 0.2)),
                        key=f"cfg_llm_profiles_{profile_name}_temperature"
                    )
                if 'num_gpu' in profile_data:
                    st.number_input(
                        "GPU Layers",
                        min_value=-1, max_value=100, step=1,
                        value=int(profile_data.get('num_gpu', -1)),
                        key=f"cfg_llm_profiles_{profile_name}_num_gpu"
                    )

    with tabs[10]: # Advanced
        st.subheader("Background Services")
        st.toggle("Enable Command Executor", value=bool(get_nested_config(config, 'services.command_executor.enabled', True)), key="cfg_services_command_executor_enabled")
        st.number_input("Command Executor Poll Interval (s)", min_value=1, max_value=60, step=1, value=int(get_nested_config(config, 'services.command_executor.poll_interval_s', 5)), key="cfg_services_command_executor_poll_interval_s")
        st.divider()
        st.subheader("Knowledge Graph")
        st.toggle("Enable Knowledge Graph", value=bool(get_nested_config(config, 'knowledge_graph.enabled', False)), key="cfg_knowledge_graph_enabled")
        st.text_input("Ontology Schema Path", value=get_nested_config(config, 'knowledge_graph.ontology_schema_path', ''), key="cfg_knowledge_graph_ontology_schema_path")

    st.divider()
    
    # --- SAVE LOGIC ---
    if st.button("Save Settings", type="primary", use_container_width=True):
        # Parse text area/list-based settings
        eoc_triggers = [line.strip() for line in st.session_state.cfg_context_management_end_of_conversation_triggers.split('\n') if line.strip()]
        vc_triggers = [line.strip() for line in st.session_state.cfg_voice_commands_trigger_phrases.split('\n') if line.strip()]
        retry_delays = [int(x.strip()) for x in st.session_state.cfg_timings_audio_processing_retry_delays_seconds.split(',') if x.strip().isdigit()]
        task_completion_keywords = [line.strip() for line in st.session_state.cfg_task_intelligence_completion_keywords.split('\n') if line.strip()]
        # --- START FIX: Update LLM profiles list ---
        llm_profiles = ["main_llm", "classification_llm", "summary_llm", "llm_command_parser", "llm_task_extractor"]
        # --- END FIX ---

        payload = {
            # General
            "timings.assumed_recording_timezone": st.session_state.cfg_timings_assumed_recording_timezone,
            "timings.end_of_day_summary_time": st.session_state.cfg_timings_end_of_day_summary_time,
            "timings.folder_monitor_process_delay_s": st.session_state.cfg_timings_folder_monitor_process_delay_s,
            "timings.audio_processing_retry_delays_seconds": retry_delays,
            "timings.email_poll_interval_minutes": st.session_state.cfg_timings_email_poll_interval_minutes,
            "timings.calendar_refresh_interval_minutes": st.session_state.cfg_timings_calendar_refresh_interval_minutes,
            # Paths & Tools
            "paths.monitored_audio_folder": st.session_state.cfg_paths_monitored_audio_folder,
            "paths.archived_audio_folder": st.session_state.cfg_paths_archived_audio_folder,
            "paths.audio_error_folder": st.session_state.cfg_paths_audio_error_folder,
            "paths.database_folder": st.session_state.cfg_paths_database_folder,
            "paths.log_folder": st.session_state.cfg_paths_log_folder,
            "paths.daily_log_folder": st.session_state.cfg_paths_daily_log_folder,
            "paths.speaker_db_dir": st.session_state.cfg_paths_speaker_db_dir,
            "paths.temp_processing_dir": st.session_state.cfg_paths_temp_processing_dir,
            "tools.ffmpeg_path": st.session_state.cfg_tools_ffmpeg_path,
            # STT
            "stt.engine": st.session_state.cfg_stt_engine,
            "stt.parakeet_mlx.model_path": st.session_state.cfg_stt_parakeet_mlx_model_path,
            "stt.parakeet_mlx.CLI.language": st.session_state.cfg_stt_parakeet_mlx_CLI_language,
            "stt.parakeet_mlx.CLI.vad_filter": st.session_state.cfg_stt_parakeet_mlx_CLI_vad_filter,
            # Speaker ID
            "audio_suite_settings.diarization_model": st.session_state.cfg_audio_suite_settings_diarization_model,
            "audio_suite_settings.embedding_model": st.session_state.cfg_audio_suite_settings_embedding_model,
            "audio_suite_settings.punctuation_model": st.session_state.cfg_audio_suite_settings_punctuation_model,
            "audio_suite_settings.similarity_threshold": st.session_state.cfg_audio_suite_settings_similarity_threshold,
            "audio_suite_settings.consolidate_new_speaker_threshold": st.session_state.cfg_audio_suite_settings_consolidate_new_speaker_threshold,
            "audio_suite_settings.min_segments_for_enroll_prompt": st.session_state.cfg_audio_suite_settings_min_segments_for_enroll_prompt,
            "audio_suite_settings.enable_live_embedding_refinement": st.session_state.cfg_audio_suite_settings_enable_live_embedding_refinement,
            "audio_suite_settings.live_refinement_min_similarity": st.session_state.cfg_audio_suite_settings_live_refinement_min_similarity,
            "audio_suite_settings.live_refinement_min_segment_duration_s": st.session_state.cfg_audio_suite_settings_live_refinement_min_segment_duration_s,
            # Matter Analysis
            "context_management.matter_assignment_min_similarity": st.session_state.cfg_context_management_matter_assignment_min_similarity,
            "context_management.auto_create_matter_threshold": st.session_state.cfg_context_management_auto_create_matter_threshold,
            "context_management.silence_reset_threshold_seconds": st.session_state.cfg_context_management_silence_reset_threshold_seconds,
            "context_management.new_matter_trigger_duration_s": st.session_state.cfg_context_management_new_matter_trigger_duration_s,
            # --- START FIX: Add new matter key to payload ---
            "context_management.matter_change_threshold": st.session_state.cfg_context_management_matter_change_threshold,
            # --- END FIX ---
            "context_management.enable_matter_context_stickiness": st.session_state.cfg_context_management_enable_matter_context_stickiness,
            "context_management.matter_context_stickiness_bonus": st.session_state.cfg_context_management_matter_context_stickiness_bonus,
            "context_management.enable_matter_conflict_flagging": st.session_state.cfg_context_management_enable_matter_conflict_flagging,
            "context_management.matter_conflict_high_confidence_threshold": st.session_state.cfg_context_management_matter_conflict_high_confidence_threshold,
            "context_management.matter_conflict_delta_threshold": st.session_state.cfg_context_management_matter_conflict_delta_threshold,
            "context_management.end_of_conversation_triggers": eoc_triggers,
            # LLM Settings
            "global_ml_device": st.session_state.cfg_global_ml_device,
            # --- START FIX: Update payload generation for LLM profiles ---
            **{f"llm.{p}.provider": st.session_state[f"cfg_llm_{p}_provider"] for p in llm_profiles},
            **{f"llm.{p}.base_url": st.session_state[f"cfg_llm_{p}_base_url"] for p in llm_profiles},
            **{f"llm.{p}.model_name": st.session_state[f"cfg_llm_{p}_model_name"] for p in llm_profiles},
            **{f"llm.{p}.temperature": st.session_state[f"cfg_llm_{p}_temperature"] for p in llm_profiles},
            **{f"llm.{p}.num_gpu": st.session_state[f"cfg_llm_{p}_num_gpu"] for p in llm_profiles},
            # --- END FIX ---
            # Voice Commands
            "voice_commands.enabled": st.session_state.cfg_voice_commands_enabled,
            "voice_commands.user_speaker_name": st.session_state.cfg_voice_commands_user_speaker_name,
            "voice_commands.vocal_override_lookback_seconds": st.session_state.cfg_voice_commands_vocal_override_lookback_seconds,
            "voice_commands.voice_command_confidence_threshold": st.session_state.cfg_voice_commands_voice_command_confidence_threshold,
            "voice_commands.trigger_phrases": vc_triggers,
            
            # Task Intelligence
            "task_intelligence.enabled": st.session_state.cfg_task_intelligence_enabled,
            "task_intelligence.task_data_file": st.session_state.cfg_task_intelligence_task_data_file,
            "task_intelligence.task_creation_cooldown_s": st.session_state.cfg_task_intelligence_task_creation_cooldown_s,
            "task_intelligence.completion_keywords": task_completion_keywords,
            # --- START FIX: Add new task key to payload ---
            "task_intelligence.task_similarity_threshold": st.session_state.cfg_task_intelligence_task_similarity_threshold,
            # --- END FIX ---

            # Integrations
            "signal.samson_phone_number": st.session_state.cfg_signal_samson_phone_number,
            "signal.recipient_phone_number": st.session_state.cfg_signal_recipient_phone_number,
            "signal.signal_cli_path": st.session_state.cfg_signal_signal_cli_path,
            "signal.signal_cli_data_path": st.session_state.cfg_signal_signal_cli_data_path,
            "audio_suite_settings.hf_token": st.session_state.cfg_audio_suite_settings_hf_token,

            # Windmill
            "windmill.enabled": st.session_state.cfg_windmill_enabled,
            "windmill.workspace": st.session_state.cfg_windmill_workspace,
            "windmill.base_url": st.session_state.cfg_windmill_base_url,
            "windmill.remote": st.session_state.cfg_windmill_remote,
            "windmill.execution_method": st.session_state.cfg_windmill_execution_method,
            "windmill.api_token": st.session_state.cfg_windmill_api_token,
            
            # Advanced
            "llm.workflow_generator_profile": st.session_state.cfg_llm_workflow_generator_profile,
            "services.command_executor.enabled": st.session_state.cfg_services_command_executor_enabled,
            "services.command_executor.poll_interval_s": st.session_state.cfg_services_command_executor_poll_interval_s,
            "knowledge_graph.enabled": st.session_state.cfg_knowledge_graph_enabled,
            "knowledge_graph.ontology_schema_path": st.session_state.cfg_knowledge_graph_ontology_schema_path,
        }
        # Dynamically add the workflow profile settings to the payload
        workflow_profiles = get_nested_config(config, 'llm.profiles', {})
        for profile_name, profile_data in workflow_profiles.items():
            payload[f"llm.profiles.{profile_name}.provider"] = st.session_state[f"cfg_llm_profiles_{profile_name}_provider"]
            payload[f"llm.profiles.{profile_name}.model_name"] = st.session_state[f"cfg_llm_profiles_{profile_name}_model_name"]
            if 'api_key' in profile_data:
                payload[f"llm.profiles.{profile_name}.api_key"] = st.session_state[f"cfg_llm_profiles_{profile_name}_api_key"]
            if 'base_url' in profile_data:
                payload[f"llm.profiles.{profile_name}.base_url"] = st.session_state[f"cfg_llm_profiles_{profile_name}_base_url"]
            if 'temperature' in profile_data:
                payload[f"llm.profiles.{profile_name}.temperature"] = st.session_state[f"cfg_llm_profiles_{profile_name}_temperature"]
            if 'num_gpu' in profile_data:
                payload[f"llm.profiles.{profile_name}.num_gpu"] = st.session_state[f"cfg_llm_profiles_{profile_name}_num_gpu"]
        
        command = {"type": "UPDATE_CONFIG", "payload": payload}
        if queue_command_from_gui(command):
            st.toast("Settings update command queued successfully!", icon="✅")
            st.session_state.settings_saved = True
        else:
            st.error("Failed to queue settings update command. Check logs for details.")

    if st.session_state.settings_saved:
        st.warning(
            "**Restart Required:** Settings have been saved to `config.yaml`. "
            "Please restart the main application for all changes to take effect.",
            icon="⚠️"
        )