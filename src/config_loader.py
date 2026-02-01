# Samson/src/config_loader.py
import yaml
from pathlib import Path
import os
import json
from typing import Any, Optional, Dict # Added Dict for type hinting
import threading

# Determine PROJECT_ROOT once when the module is loaded
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yaml"

_config: Optional[Dict[str, Any]] = None # Global variable to cache the loaded configuration
_config_lock = threading.Lock()

def _create_default_config_content() -> str:
    """Generates the content for a default config.yaml file."""
    return """# -----------------------------------------------------------------------------
# Configuration for Samson AI Assistant
#
# This file defines all the settings for the application.
# - Paths are relative to the project root unless specified as absolute.
# - For settings requiring credentials (like hf_token or signal numbers),
#   it's recommended to use placeholders and configure them for your system.
# - Comments marked with '<<< VERIFY' indicate settings you must check
#   and likely change for your specific setup.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# PATHS
# Define directories for data, logs, models, etc.
# -----------------------------------------------------------------------------
paths:
  monitored_audio_folder: "data/audio_input"      # <<< VERIFY: This is where new audio files to be processed should be placed.
  archived_audio_folder: "data/audio_archive"     # Where successfully processed original audio files are moved.
  audio_error_folder: "data/audio_errors"         # For audio files that fail during processing.
  database_folder: "data/databases"               # Where daily SQLite databases will be stored.
  log_folder: "logs"                              # For general application log files.
  log_file_name: "samson_orchestrator.log"          # The main log file name.
  daily_log_folder: "data/daily_logs"             # Where daily structured JSON logs (YYYY-MM-DD_samson_log.json) are stored.
  temp_processing_dir: "data/temp_audio_processing" # For intermediate files during the audio pipeline.
  flags_queue_dir: "data/flags_queue"             # Directory for daily flag queue files (for review, etc.).
  speaker_db_dir: "data/speaker_database"         # Stores FAISS index and speaker name map for speaker recognition.
  command_queue_dir: "data/command_queue"             # For file-based command queuing

# -----------------------------------------------------------------------------
# TOOLS
# Configuration for external command-line tools.
# -----------------------------------------------------------------------------
tools:
  # Path to the ffmpeg executable. "ffmpeg" is sufficient if it's in your system's PATH.
  # On macOS via Homebrew, it might be "/opt/homebrew/bin/ffmpeg".
  # Use `which ffmpeg` in your terminal to find the correct path if needed.
  ffmpeg_path: "ffmpeg"

# -----------------------------------------------------------------------------
# STT (SPEECH-TO-TEXT)
# -----------------------------------------------------------------------------
stt:
  engine: "parakeet_mlx"  # Recommended engine. Options: "parakeet_mlx", "openai_whisper"

  openai_whisper:
    model_name_or_path: "tiny.en" # e.g., "tiny.en", "base.en", "medium.en"
    language: "en" # Set to null for auto-detection.
    # device: "cpu" # Uncomment to force CPU if MPS/GPU is problematic.

  parakeet_mlx:
    model_path: "mlx-community/parakeet-tdt-0.6b-v2" # Hugging Face model ID or local path.
    CLI:
      language: "en"     # Language hint for the model.
      sample_rate: 16000 # Expected sample rate for the model.
      vad_filter: true   # Use Voice Activity Detection to filter silence.

# Device for non-STT ML models (like Pyannote, SpeechBrain).
# Options: "mps" (for Apple Silicon), "cuda" (for NVIDIA), "cpu".
global_ml_device: "mps" # <<< VERIFY: Change to "cuda" or "cpu" based on your hardware.

# -----------------------------------------------------------------------------
# AUDIO PROCESSING SUITE
# Settings for diarization, speaker identification, and text processing.
# -----------------------------------------------------------------------------
audio_suite_settings:
  # --- Model Names (Hugging Face Hub IDs or local paths) ---
  diarization_model: "pyannote/speaker-diarization-3.1"  # <<< VERIFY: Requires accepting license on Hugging Face.
  embedding_model: "speechbrain/spkrec-ecapa-voxceleb"
  punctuation_model: "felflare/bert-restore-punctuation" # Set to null to disable punctuation restoration.

  # --- Hugging Face Token ---
  # Required for gated models like pyannote/speaker-diarization-3.1.
  # Get a 'read' token from: huggingface.co/settings/tokens
  # Store directly here, or better, as "ENV:HF_TOKEN" to read from an environment variable.
  hf_token: null # <<< VERIFY: e.g., "hf_YourTokenGoesHere" or "ENV:HUGGING_FACE_TOKEN"

  # --- Speaker Database Filenames (inside paths.speaker_db_dir) ---
  faiss_index_filename: "global_speaker_embeddings.index"
  speaker_map_filename: "global_speaker_names.json"

  # --- Model Cache ---
  # Subdirectory for the embedding model cache, relative to the project root.
  embedding_model_cache_subdir: "data/models/embedding_cache"

  # --- Processing Parameters ---
  embedding_dim: 192          # Embedding dimension for the speaker model (ECAPA-TDNN is 192).
  min_segment_duration_for_embedding: 3.0 # Min audio duration (seconds) from a speaker to generate a reliable embedding.
  punctuation_chunk_size: 256 # Text chunk size for the punctuation model.
  review_flags_max_days_lookback: 14 # How many days back to look for unresolved review flags on startup.

  # --- Speaker Identification Thresholds ---
  initial_similarity_thresholds:
    voip: 0.85      # Stricter threshold for clear audio like phone calls.
    in_person: 0.82 # More lenient threshold for variable in-person recordings.
    context_match_bonus: 0.05 # Bonus added if a known speaker's context matches the audio's context.

  # --- Diarization & Transcript Consolidation ---
  unknown_consolidation_max_bridge_s: 3.0   # Max time (s) an UNKNOWN word can be from a known speaker to be re-assigned.
  diarization_consolidation_min_segment_ms: 700 # Min duration (ms) for a diarized segment to be considered stable.
  diarization_consolidation_max_gap_ms: 100   # Max silence (ms) between segments to merge them.

  # --- New Speaker Enrollment ---
  active_candidate_merge_threshold: 0.85      # Similarity to merge a new segment into an "active" unknown speaker session.
  min_segments_for_enroll_prompt: 1           # Min distinct segments from a new speaker before prompting for their name.
  min_duration_ms_for_enroll_prompt: 5000     # Min total audio (ms) from a new speaker before prompting.
  consolidate_new_speaker_threshold: 0.88   # Similarity threshold to merge a newly-enrolled speaker with a similar existing one.

  # --- Live Embedding Refinement (improves speaker profiles over time) ---
  enable_live_embedding_refinement: true
  live_refinement_min_similarity: 0.82        # Min similarity for an identified segment to be used for refining its speaker's profile.
  live_refinement_min_segment_duration_s: 3.8 # Min duration of an identified segment to be used for refinement.

  # --- Ambiguity Detection (for flagging uncertain identifications) ---
  ambiguity_similarity_lower_bound: 0.60           # Min similarity to be considered a potential match.
  ambiguity_similarity_upper_bound_for_review: 0.80 # If best match is below this, flag for review.
  ambiguity_max_similarity_delta_for_multiple_matches: 0.07 # If top 2 matches are this close, flag for review.

  # --- Log Formatting ---
  timestamped_transcript_interval_seconds: 120 # How often to print [HH:MM:SS] in the transcript.
  master_log_line_width: 90                    # Character width for formatted log output.

# -----------------------------------------------------------------------------
# LARGE LANGUAGE MODELS (LLM)
# For summaries, classification, Q&A, etc. Assumes Ollama provider.
# -----------------------------------------------------------------------------
llm:
  llm_command_parser:
    provider: "ollama"
    model_name: "llama3:8b-instruct-q4_K_M" # A fast model suitable for classification/extraction
    temperature: 0.0
  classification_llm:
    provider: "ollama"
    model_name: "llama3:8b" # <<< VERIFY: A fast model for classification tasks.
    temperature: 0.1
    num_gpu: -1             # Number of GPU layers to offload (-1 for all).

  main_llm:
    provider: "ollama"
    model_name: "llama3:8b" # <<< VERIFY: Your primary, powerful model for Q&A and reasoning.
    temperature: 0.3
    num_gpu: -1

  summary_llm:
    provider: "ollama"
    model_name: "llama3:8b" # <<< VERIFY: Can be a smaller model for faster summaries.
    temperature: 0.2
    num_gpu: -1

# -----------------------------------------------------------------------------
# SIGNAL MESSAGING INTERFACE
# -----------------------------------------------------------------------------
signal:
  samson_phone_number: "+1xxxxxxxxxx"     # <<< VERIFY: Your Samson bot's Signal number (must be registered with signal-cli).
  recipient_phone_number: "+1yyyyyyyyyy"  # <<< VERIFY: Your personal Signal number to receive messages.
  signal_cli_path: "signal-cli"           # Path to the signal-cli executable.
  signal_cli_data_path: "~/.local/share/signal-cli" # <<< VERIFY: Path to signal-cli's data directory.

# -----------------------------------------------------------------------------
# TIMINGS AND INTERVALS
# -----------------------------------------------------------------------------
timings:
  # --- System Operation Intervals ---
  audio_processing_retry_delays_seconds: [60, 300, 900] # Delays between retries for a failed audio file.
  folder_monitor_process_delay_s: 5       # Seconds to wait after a new file appears before processing the previous one (for chunked recording).
  audio_chunk_expected_duration: "2m"     # Expected duration of audio chunks (e.g., "10m", "90s"). Helps detect incomplete files.

  # --- Scheduled Tasks ---
  end_of_day_summary_time: "21:00"        # <<< VERIFY: Your preferred time for the daily summary (HH:MM 24-hour local time).
  # email_poll_interval_minutes: 15       # (Example for future use)
  # calendar_refresh_interval_minutes: 60 # (Example for future use)

  # --- Time & Date Formatting ---
  assumed_recording_timezone: "America/New_York"  # <<< VERIFY: Timezone for timestamps if not in metadata (e.g., "UTC", "Europe/London"). See `tzdata` for names.
  master_log_timestamp_format: "%b%d, %Y - %H:%M" # Python strftime format for display in logs.

# -----------------------------------------------------------------------------
# SPEAKER INTELLIGENCE
# Advanced features for managing speaker profiles over time.
# -----------------------------------------------------------------------------
speaker_intelligence:
  enable_dynamic_thresholds: true         # Allow the system to adjust speaker similarity thresholds based on user corrections.
  dynamic_threshold_learning_rate: 0.01
  dynamic_threshold_min_corrections: 5
  dynamic_threshold_adjustment_frequency_hours: 24
  dynamic_threshold_adjustment_buffer: 0.02

  batch_commit_inactivity_seconds: 120    # Seconds of inactivity before committing a batch of new speaker segments to the database.
  recalculation_recency_decay_rate: 0.01

  enable_llm_role_assignment: false       # Use an LLM to guess the role of a speaker in a conversation.
  llm_role_assignment_model: null         # Set a model name or leave as null to use `llm.main_llm`.
  llm_role_assignment_prompt_template: "Based on the following dialogue, what is the primary role of this speaker (e.g., Interviewer, Subject Matter Expert, Assistant, Caller, Host, Guest)? Dialogue: {dialogue_text}"
  role_assignment_frequency_days: 14

  enable_automatic_profile_recalculation: true # Periodically recalculate all speaker embeddings for better accuracy.
  profile_recalculation_time_utc: "03:00" # Time for the daily recalculation task (HH:MM 24-hour UTC).
  profile_recalculation_min_new_segments_threshold: 50

  profile_evolution:
    recency_weight_enabled: true          # Give more weight to recent audio snippets when calculating speaker profiles.
    recency_decay_half_life_days: 90      # How many days it takes for a snippet's weight to be halved.

# -----------------------------------------------------------------------------
# VOICE COMMANDS
# -----------------------------------------------------------------------------
voice_commands:
  enabled: true
  user_speaker_name: "Samson Owner" # The exact speaker name to monitor for commands
  trigger_phrases:
    - "samson"
    - "sampson" # Common mispronunciations
  llm_prompt_template_path: "src/prompts/voice_command_prompt.txt"
  command_definitions:
    - command: "SET_MATTER"
      description: "Assigns the current conversation to a specific matter, case, or project."
      examples:
        - "Samson, set matter to Project Phoenix."
        - "Samson, this is for the Miller case."
      parameters:
        - name: "matter_id"
          type: "string"
          description: "The unique identifier or name of the matter."
    # Add other command definitions here in the future

# -----------------------------------------------------------------------------
# KNOWLEDGE GRAPH (Future Feature)
# -----------------------------------------------------------------------------
knowledge_graph:
  enabled: false
  ontology_schema_path: "config/ontology.json"
  ground_truth_file: "data/entity_ground_truth.jsonl"

# -----------------------------------------------------------------------------
# BACKGROUND SERVICES
# -----------------------------------------------------------------------------
services:
  command_executor:
    enabled: true
    poll_interval_s: 5 # How often the service checks for new commands
"""

def ensure_config_exists(config_file_path: Path = CONFIG_FILE_PATH) -> bool:
    if not config_file_path.exists():
        print(f"ConfigLoader: Configuration file not found at {config_file_path}.")
        print("ConfigLoader: Attempting to create a default config.yaml.")
        try:
            config_file_path.parent.mkdir(parents=True, exist_ok=True)
            default_content = _create_default_config_content()
            with open(config_file_path, 'w', encoding='utf-8') as f:
                f.write(default_content)
            print(f"ConfigLoader: Default config.yaml created at {config_file_path}.")
            print("ConfigLoader: IMPORTANT: Please review the generated 'config/config.yaml', "
                  "especially paths, API keys, phone numbers, and model names, then run the assistant again.")
            return True
        except Exception as e:
            print(f"ConfigLoader: ERROR creating default configuration file at {config_file_path}: {e}")
            return False
    return True

def _resolve_path_value(value: Any, base_dir: Path, is_known_identifier: bool = False) -> Optional[Any]:
    """
    Helper to resolve a path string or Path object, expanding user and making absolute.
    If is_known_identifier is True, returns the value string as is (for HF IDs).
    Otherwise, treats as a file/directory path.
    """
    if value is None:
        return None

    value_str = str(value)

    if is_known_identifier:
        # print(f"DEBUG ConfigLoader: Treating '{value_str}' as known identifier string.")
        return value_str

    # Proceed with normal path resolution
    path_obj = Path(value_str).expanduser()
    if path_obj.is_absolute():
        # print(f"DEBUG ConfigLoader: Resolved absolute path '{value_str}' to '{path_obj.resolve()}'.")
        return path_obj.resolve()
    else:
        resolved_path = (base_dir / path_obj).resolve()
        # print(f"DEBUG ConfigLoader: Resolved relative path '{value_str}' with base '{base_dir}' to '{resolved_path}'.")
        return resolved_path

def _ensure_critical_dirs_exist(config_data: Dict[str, Any], project_root: Path):
    print_prefix = "ConfigLoader(EnsureDirs):"
    paths_to_create: list[Path] = []
    paths_config = config_data.get('paths', {})

    # Keys that represent directory paths and should be Path objects
    # Confirmed: flags_queue_dir is present as requested.
    dir_keys_in_paths = [
        'monitored_audio_folder', 'audio_error_folder', 'database_folder',
        'log_folder', 'speaker_db_dir', 'temp_processing_dir',
        'archived_audio_folder', 'daily_log_folder', 'flags_queue_dir', 'command_queue_dir',
        'windmill_shared_folder'
    ]
    for key in dir_keys_in_paths:
        path_val = paths_config.get(key) # Should be a Path object after processing
        if path_val and isinstance(path_val, Path):
            paths_to_create.append(path_val)
        elif path_val: # If path_val exists in config but isn't resolved to Path (e.g. an error or it's None)
            print(f"{print_prefix} WARNING: Path 'paths.{key}' value '{path_val}' (type: {type(path_val)}) was not resolved to a Path object or is None. Skipping directory creation if it's not a Path.")
        # If path_val is None (key not in config), it's silently skipped, which is usually desired.

    audio_suite_cfg = config_data.get('audio_suite_settings', {})
    embedding_cache_subdir_name = audio_suite_cfg.get('embedding_model_cache_subdir')
    if embedding_cache_subdir_name and isinstance(embedding_cache_subdir_name, str):
        base_model_dir = project_root / "data" / "models" # Consistent base
        # This subdir is relative to base_model_dir
        embedding_cache_full_path = (base_model_dir / embedding_cache_subdir_name).resolve()
        paths_to_create.append(embedding_cache_full_path)
        audio_suite_cfg['_resolved_embedding_model_cache_dir'] = embedding_cache_full_path


    stt_cfg = config_data.get('stt', {})
    stt_engine = stt_cfg.get('engine')

    if stt_engine == 'openai_whisper':
        whisper_settings = stt_cfg.get('openai_whisper', {})
        whisper_model_dl_root = whisper_settings.get('download_root_for_model_cache') # Should be Path or None
        if whisper_model_dl_root and isinstance(whisper_model_dl_root, Path):
            paths_to_create.append(whisper_model_dl_root)
    elif stt_engine == 'parakeet_mlx':
        parakeet_settings = stt_cfg.get('parakeet_mlx', {})
        parakeet_model_path_val = parakeet_settings.get('model_path') # Could be Path or str (HF ID)
        if parakeet_model_path_val and isinstance(parakeet_model_path_val, Path):
            if not parakeet_model_path_val.exists(): # Only check existence if it resolved to a local Path
                print(f"{print_prefix} WARNING: Configured Parakeet MLX local model path does not exist: {parakeet_model_path_val}")

    for dir_path in paths_to_create:
        if dir_path: # Ensure dir_path is not None
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                print(f"{print_prefix} WARNING: Could not create directory {dir_path}: {e}")


def load_config(config_path: Path = CONFIG_FILE_PATH) -> Dict[str, Any]:
    global _config, PROJECT_ROOT

    if not config_path.exists():
        if not ensure_config_exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} was missing and could not be created.")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        print(f"ConfigLoader: CRITICAL ERROR parsing YAML from {config_path}: {e}")
        raise ValueError(f"Invalid YAML format in {config_path}") from e
    except Exception as e:
        print(f"ConfigLoader: CRITICAL ERROR loading configuration file {config_path}: {e}")
        raise
    # Apply container path rewrite if needed, before path resolution.
    container_root_str = os.environ.get('SAMSON_ROOT_IN_CONTAINER')
    if container_root_str:
        container_root = Path(container_root_str)
        if 'paths' in raw_config:
            for key, path_str in raw_config['paths'].items():
                if isinstance(path_str, str): # Only apply to strings from YAML
                    parts = Path(path_str).parts
                    # Only rewrite relative paths starting with 'data'
                    if parts and parts[0] == 'data' and not Path(path_str).is_absolute():
                        new_abs_path_str = str(container_root.joinpath(*parts[1:]))
                        raw_config['paths'][key] = new_abs_path_str
    processed_config: Dict[str, Any] = {}

    # --- Define keys that should remain as strings (not resolved as paths) ---
    # or are known to be HF string identifiers
    string_keys_in_paths = ["log_file_name", "database_name_template", "faiss_index_filename", "speaker_map_filename"]
    hf_id_keys_in_stt_parakeet = ["model_path"] # For stt.parakeet_mlx.model_path if it's an ID
    # Note: embedding_model_cache_subdir is a string but used to *construct* a path later.

    # Process 'paths' section
    processed_config['paths'] = {}
    raw_paths_config = raw_config.get('paths', {}) or {}
    for key, value in raw_paths_config.items():
        if key in string_keys_in_paths:
            processed_config['paths'][key] = str(value) if value is not None else None
        else: # All other keys in 'paths' are assumed to be directory or file paths
            resolved = _resolve_path_value(value, PROJECT_ROOT)
            processed_config['paths'][key] = resolved

    # Process 'stt' section
    processed_config['stt'] = raw_config.get('stt', {}) or {}
    stt_engine = processed_config['stt'].get('engine')

    if stt_engine == 'openai_whisper':
        whisper_settings = processed_config['stt'].get('openai_whisper', {}) or {}
        dl_root = whisper_settings.get('download_root_for_model_cache')
        # This is a directory path, resolve it.
        whisper_settings['download_root_for_model_cache'] = _resolve_path_value(dl_root, PROJECT_ROOT) if dl_root else None
        # 'model_name_or_path' for Whisper can be an ID ("tiny.en") or a local path.
        # If it's a local path, it also needs resolution.
        # For simplicity, assuming Whisper IDs for now. If local paths are needed, more logic here.
        processed_config['stt']['openai_whisper'] = whisper_settings

    elif stt_engine == 'parakeet_mlx':
        parakeet_settings = processed_config['stt'].get('parakeet_mlx', {}) or {}
        model_p_val = parakeet_settings.get('model_path')
        # model_path for parakeet_mlx can be a local path or an HF ID.
        # _resolve_path_value now has 'is_known_identifier'
        # We need to determine if model_p_val is an ID or a path.
        # Heuristic: if it doesn't exist locally and contains '/', assume ID.
        is_hf_id = False
        if isinstance(model_p_val, str) and '/' in model_p_val:
            # Check if it's NOT a local existing path
            temp_path = Path(model_p_val).expanduser()
            if not (temp_path.is_absolute() and temp_path.exists()) and \
               not ((PROJECT_ROOT / temp_path).exists()):
                is_hf_id = True
        
        parakeet_settings['model_path'] = _resolve_path_value(model_p_val, PROJECT_ROOT, is_known_identifier=is_hf_id)
        processed_config['stt']['parakeet_mlx'] = parakeet_settings
    
    processed_config['global_ml_device'] = raw_config.get('global_ml_device', 'cpu') # Default to cpu

    # Load other sections, being mindful of paths vs. string identifiers
    sections_to_copy_directly = [
        'tools', 'llm', 'timings', 'huggingface', 'voice_commands', 
        'speaker_intelligence', 'knowledge_graph', 'services', 'context_management', 'task_intelligence', 'windmill'
    ]
    for section_name in sections_to_copy_directly:
        processed_config[section_name] = raw_config.get(section_name, {}) or {}

    # Process 'signal' section specifically for 'signal_cli_data_path'
    raw_signal_config = raw_config.get('signal', {}) or {}
    processed_config['signal'] = raw_signal_config.copy() # Start by copying all
    cli_data_path = raw_signal_config.get('signal_cli_data_path')
    if cli_data_path: # This path allows '~' and should be expanded
        processed_config['signal']['signal_cli_data_path'] = _resolve_path_value(cli_data_path, Path()) # Pass Path() as base for user expansion

    # Process 'audio_suite_settings'
    processed_config['audio_suite_settings'] = raw_config.get('audio_suite_settings', {}) or {}
    # Keys like 'diarization_model', 'embedding_model', 'punctuation_model' are usually HF IDs (strings).
    # 'embedding_model_cache_subdir' is a string used to construct a path.
    # 'faiss_index_filename', 'speaker_map_filename' are strings.
    # No path resolution needed here directly, they are used as strings or to build paths later.

    _ensure_critical_dirs_exist(processed_config, PROJECT_ROOT)
    _config = processed_config
    # print("DEBUG ConfigLoader: Final processed config:", json.dumps(_config, indent=2, cls=PathEncoder))
    return _config


def get_config() -> Dict[str, Any]:
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = load_config()
    return _config

class PathEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        return super().default(o)

def reload_config() -> Dict[str, Any]:
    """Forces a reload of the configuration from the YAML file."""
    global _config
    with _config_lock:
        # Directly call load_config to bypass the cache check and update the global variable.
        _config = load_config()
    # This function is called to get the updated config to store in session_state
    return get_config()

if __name__ == "__main__":
    print(f"ConfigLoader Self-Test: PROJECT_ROOT is {PROJECT_ROOT}")
    
    # Create a temporary config file for testing if it doesn't exist
    # This ensures the self-test can run even if the user hasn't set up a config yet
    temp_test_config_path = PROJECT_ROOT / "config" / "temp_test_config.yaml"
    if not CONFIG_FILE_PATH.exists():
        print(f"ConfigLoader Self-Test: Main config {CONFIG_FILE_PATH} not found.")
        print(f"ConfigLoader Self-Test: Using temporary config {temp_test_config_path} for this test run.")
        if not ensure_config_exists(temp_test_config_path):
            print("ConfigLoader Self-Test: Default config.yaml could not be created for temp test. Exiting.")
            exit(1)
        config_to_use = temp_test_config_path
    else:
        config_to_use = CONFIG_FILE_PATH

    print(f"ConfigLoader Self-Test: Using config file: {config_to_use}")
    
    # Ensure the (possibly temporary) config file exists before loading
    if not ensure_config_exists(config_to_use):
        print("ConfigLoader Self-Test: config.yaml might have been newly created. "
              "Please review it (if it's the main one) and run again for a full test. Exiting self-test.")
        exit(0)
        
    try:
        print("\nConfigLoader Self-Test: Attempting to load configuration...")
        # Temporarily point _config to None and CONFIG_FILE_PATH to the test config for this run
        _config = None 
        original_config_path_for_test = CONFIG_FILE_PATH # Store original global
        config_file_path_to_use_in_test = config_to_use # Variable for clarity

        # Override global CONFIG_FILE_PATH for the get_config call during the test
        # This simulates loading the specific test config
        globals()['CONFIG_FILE_PATH'] = config_file_path_to_use_in_test
        
        config = get_config() # This will now use the (potentially temporary) config_file_path_to_use_in_test
        
        # Restore global CONFIG_FILE_PATH
        globals()['CONFIG_FILE_PATH'] = original_config_path_for_test


        print("ConfigLoader Self-Test: Configuration loaded successfully.")
        print("\n--- Loaded Configuration (Paths as strings for readability) ---")
        print(json.dumps(config, indent=2, cls=PathEncoder))
        
        print("\nConfigLoader Self-Test: Verifying critical directory creation/existence (from resolved Path objects)...")
        paths_section = config.get('paths', {})
        
        dir_keys_to_check = [
            'monitored_audio_folder', 'audio_error_folder', 'database_folder',
            'log_folder', 'speaker_db_dir', 'temp_processing_dir',
            'archived_audio_folder', 'daily_log_folder', 'flags_queue_dir'
        ]
        
        for key in dir_keys_to_check:
            val_path = paths_section.get(key)
            if val_path is None: 
                 print(f"  paths.{key}: NOT FOUND in config (Type: None)")
            else:
                print(f"  paths.{key}: '{val_path}' (Type: {type(val_path)}, Is Path: {isinstance(val_path, Path)}" +
                      (f", Exists: {val_path.exists()}" if isinstance(val_path, Path) else "") + ")")
        
        print(f"  paths.log_file_name: '{paths_section.get('log_file_name')}' (Type: {type(paths_section.get('log_file_name'))})")


        aps_config = config.get('audio_suite_settings', {})
        emb_cache_subdir_name_from_config = aps_config.get('embedding_model_cache_subdir') # This is the string from config
        
        # Check the _resolved_ path which is created during _ensure_critical_dirs_exist
        resolved_emb_cache_path = aps_config.get('_resolved_embedding_model_cache_dir')

        print(f"  audio_suite_settings.embedding_model_cache_subdir (from config): '{emb_cache_subdir_name_from_config}'")
        if resolved_emb_cache_path and isinstance(resolved_emb_cache_path, Path):
            print(f"  audio_suite_settings._resolved_embedding_model_cache_dir: {resolved_emb_cache_path} (Exists: {resolved_emb_cache_path.exists()})")
            # Verify it's in the correct location based on the new default
            expected_cache_path = PROJECT_ROOT / "data" / "models" / "embedding_cache"
            if resolved_emb_cache_path == expected_cache_path:
                print(f"    Verification: Resolved path matches expected {expected_cache_path}")
            else:
                print(f"    VERIFICATION FAILED: Resolved path {resolved_emb_cache_path} DOES NOT MATCH expected {expected_cache_path}")
        else:
             print(f"  audio_suite_settings._resolved_embedding_model_cache_dir: NOT RESOLVED or not a Path (Value: {resolved_emb_cache_path})")


        stt_config_main = config.get('stt', {})
        stt_engine_main = stt_config_main.get('engine')
        print(f"\n  STT Engine: {stt_engine_main}")
        if stt_engine_main == 'openai_whisper':
            whisper_settings = stt_config_main.get('openai_whisper', {})
            whisper_dl_root = whisper_settings.get('download_root_for_model_cache')
            print(f"  stt.openai_whisper.download_root_for_model_cache: '{whisper_dl_root}' (Type: {type(whisper_dl_root)}, Is Path: {isinstance(whisper_dl_root, Path)}" +
                  (f", Exists: {whisper_dl_root.exists()}" if isinstance(whisper_dl_root, Path) else "") + ")")
        elif stt_engine_main == 'parakeet_mlx':
            parakeet_settings = stt_config_main.get('parakeet_mlx', {})
            parakeet_model_p = parakeet_settings.get('model_path')
            print(f"  stt.parakeet_mlx.model_path: '{parakeet_model_p}' (Type: {type(parakeet_model_p)})")


        print("\nConfigLoader Self-Test: Completed successfully.")
    except Exception as e:
        import traceback
        print(f"ConfigLoader Self-Test: ERROR - An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        # Restore global CONFIG_FILE_PATH if it was changed by the test setup
        if 'original_config_path_for_test' in locals() and original_config_path_for_test:
             globals()['CONFIG_FILE_PATH'] = original_config_path_for_test
        
        if config_to_use == temp_test_config_path and temp_test_config_path.exists():
            # print(f"ConfigLoader Self-Test: Cleaning up temporary config file {temp_test_config_path}")
            # temp_test_config_path.unlink() # You might want to uncomment this for automated tests
            print(f"ConfigLoader Self-Test: Temporary config file {temp_test_config_path} was used and is left for inspection.")
            pass