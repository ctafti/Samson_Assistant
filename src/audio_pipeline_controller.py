from pathlib import Path
import shutil
import time
import subprocess # Ensure this is here
import json      # Ensure this is here
import sys       # Ensure this is here
import re        # Ensure this is here
import uuid
import threading
from typing import Dict, Any, Optional, List, Tuple # Ensure Tuple is here
from datetime import datetime, timezone, timedelta # Added timezone
from collections import Counter, defaultdict # Ensure Counter and defaultdict are imported
import numpy as np # For embedding comparisons
# import copy # Task 3: Removed unused import

from src.logger_setup import logger
from src.config_loader import get_config, PROJECT_ROOT

from .audio_processing_suite import models as aps_models
from .audio_processing_suite import audio_processing as aps_audio
from .audio_processing_suite import speaker_id as aps_speaker_id
from .audio_processing_suite import persistence as aps_persistence
from .audio_processing_suite import text_processing as aps_text_processing
from .audio_processing_suite import utils as aps_utils
from .master_daily_transcript import get_master_transcript_path # MODIFIED: append_to_master_transcript removed
from . import speaker_profile_manager
from src.voice_command_processor import process_voice_command
from src.daily_log_manager import get_day_start_time, extract_sequence_number_from_filename, parse_duration_to_minutes
from src.audio_processing_suite.audio_processing import has_speech
from src.context_manager import get_active_context

# Conditional import for LLM interface, as it might not always be present/needed
try:
    from src.llm_interface import get_ollama_chat_model, summarize_speaker_text
    LLM_INTERFACE_AVAILABLE = True
except ImportError:
    LLM_INTERFACE_AVAILABLE = False
    get_ollama_chat_model = None # type: ignore
    summarize_speaker_text = None # type: ignore
    logger.warning("src.llm_interface not found. LLM-dependent features like summarization will be unavailable.")

# --- Utility functions for managing daily flags queue ---

def _get_daily_flags_queue_path(target_date: datetime, config: Dict[str, Any]) -> Path:
    """
    Constructs the path to the daily flags queue JSON file for a given date.
    Ensures the directory for the flags queue exists.
    """
    paths_cfg = config.get('paths', {})
    flags_queue_dir_str = paths_cfg.get('flags_queue_dir')
    if not flags_queue_dir_str:
        logger.error("Configuration for 'paths.flags_queue_dir' is missing.")
        # Fallback to a default path or raise an error, depending on desired strictness
        # For now, raising an error as it's a critical path configuration.
        raise ValueError("Flags queue directory not configured.")

    flags_queue_dir = Path(flags_queue_dir_str)
    flags_queue_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    filename = f"{target_date.strftime('%Y-%m-%d')}_flags_queue.json"
    return flags_queue_dir / filename

def _load_daily_flags_queue(flags_queue_path: Path) -> List[Dict[str, Any]]:
    """
    Loads the daily flags queue from the specified JSON file.
    Handles FileNotFoundError and JSONDecodeError by returning an empty list and logging a warning.
    """
    if not flags_queue_path.exists():
        logger.info(f"Flags queue file not found: {flags_queue_path}. Returning empty list.")
        return []
    try:
        with open(flags_queue_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return []
            f.seek(0)
            flags_data = json.load(f)
        if not isinstance(flags_data, list):
            logger.warning(f"Flags queue file {flags_queue_path} does not contain a list. Returning empty list.")
            return []
        # Optionally, validate structure of each item here
        return flags_data
    except FileNotFoundError:
        logger.info(f"Flags queue file not found (race condition?): {flags_queue_path}. Returning empty list.")
        return []
    except json.JSONDecodeError:
        logger.warning(f"Error decoding JSON from flags queue file: {flags_queue_path}. Returning empty list.", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading flags queue {flags_queue_path}: {e}", exc_info=True)
        return []

def _json_serializer_helper(obj: Any) -> Any:
    """Helper to serialize datetime objects to ISO 8601 and numpy arrays to lists."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    # Add handling for other non-serializable types if necessary
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def _save_daily_flags_queue(flags_queue_path: Path, flags_data: List[Dict[str, Any]]) -> None:
    """
    Saves the daily flags queue data to the specified JSON file.
    Datetime objects are converted to ISO8601 strings.
    Numpy arrays (embeddings) are converted to lists.
    """
    try:
        flags_queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(flags_queue_path, 'w', encoding='utf-8') as f:
            json.dump(flags_data, f, indent=2, default=_json_serializer_helper)
        logger.info(f"Successfully saved daily flags queue to: {flags_queue_path}")
    except TypeError as te:
        logger.error(f"TypeError during JSON serialization for flags queue {flags_queue_path}: {te}. Data might contain non-serializable types not handled by _json_serializer_helper.", exc_info=True)
        # Potentially log problematic part of data if possible, or re-raise
    except Exception as e:
        logger.error(f"Error writing flags queue to {flags_queue_path}: {e}", exc_info=True)

def _calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculates cosine similarity between two numpy array embeddings.
    Embeddings are L2 normalized before the dot product.
    """
    if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
        logger.warning("_calculate_similarity: Inputs must be numpy arrays.")
        return 0.0
    if embedding1.ndim != 1 or embedding2.ndim != 1:
        logger.warning(f"_calculate_similarity: Embeddings must be 1D arrays. Got shapes {embedding1.shape}, {embedding2.shape}")
        return 0.0
    if embedding1.size == 0 or embedding2.size == 0: # Handles empty arrays
        logger.warning("_calculate_similarity: One or both embeddings are empty.")
        return 0.0
    if embedding1.shape != embedding2.shape:
        logger.warning(f"_calculate_similarity: Embedding shapes do not match: {embedding1.shape} vs {embedding2.shape}")
        return 0.0

    # L2 normalization
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0: # Avoid division by zero if a vector is all zeros
        logger.debug("_calculate_similarity: One or both embeddings have zero norm.")
        return 0.0

    e1_norm = embedding1 / norm1
    e2_norm = embedding2 / norm2

    similarity = np.dot(e1_norm, e2_norm)
    return float(similarity)

# --- End of utility functions for managing daily flags queue ---

_loaded_models_cache: Dict[str, Any] = {
    "stt_engine": None,
    "stt_model": None,
    "stt_tokenizer": None,
    "diarization": None,
    "embedding_model": None,
    "embedding_dim": None,
    "punctuation": None
}

def initialize_audio_models() -> None:
    if _loaded_models_cache.get("stt_model") is not None and \
       _loaded_models_cache.get("stt_model") != "PARAKEET_CLI_MODE": # Check against CLI mode placeholder
        logger.info("Audio models appear to be already initialized.")
        return
    if _loaded_models_cache.get("stt_model") == "PARAKEET_CLI_MODE" and \
       _loaded_models_cache.get("stt_engine") == get_config().get('stt', {}).get('engine'):
        logger.info("Parakeet STT already set to CLI mode. Other models might re-initialize if configured.")

    logger.info("Initializing audio processing models...")
    config = get_config()
    stt_master_cfg = config.get('stt', {})
    aps_cfg = config.get('audio_suite_settings', {})
    
    stt_engine_name = stt_master_cfg.get('engine', 'openai_whisper')
    _loaded_models_cache["stt_engine"] = stt_engine_name
    logger.info(f"Selected STT Engine: {stt_engine_name}")

    pytorch_processing_device = aps_utils.check_device(config.get('global_ml_device', 'cpu'))

    try:
        if stt_engine_name == 'openai_whisper':
            whisper_cfg = stt_master_cfg.get('openai_whisper', {})
            stt_model_name = whisper_cfg.get('model_name_or_path')
            if not stt_model_name:
                raise ValueError("STT model name ('stt.openai_whisper.model_name_or_path') not configured.")
            
            whisper_model_cache_path_str = whisper_cfg.get('download_root_for_model_cache')
            whisper_model_cache_path = Path(whisper_model_cache_path_str) if whisper_model_cache_path_str else None

            logger.info(f"Loading OpenAI Whisper STT model: {stt_model_name} on device: {pytorch_processing_device}")
            if whisper_model_cache_path:
                    logger.info(f"Using Whisper STT model cache path: {whisper_model_cache_path}")

            _loaded_models_cache["stt_model"] = aps_models.load_whisper_model(
                stt_model_name,
                pytorch_processing_device,
                download_root=whisper_model_cache_path
            )
            if not _loaded_models_cache["stt_model"]:
                raise RuntimeError(f"OpenAI Whisper STT model '{stt_model_name}' failed to load.")

        elif stt_engine_name == 'parakeet_mlx':
            parakeet_cfg = stt_master_cfg.get('parakeet_mlx', {})
            parakeet_model_id = parakeet_cfg.get('model_path')
            if not parakeet_model_id:
                raise ValueError("Parakeet MLX model path/ID ('stt.parakeet_mlx.model_path') not configured.")

            logger.info(f"Setting up Parakeet MLX STT model for CLI processing: {parakeet_model_id}")
            model_indicator, tokenizer_indicator = aps_models.load_parakeet_mlx_model(parakeet_model_id)
            
            if model_indicator != "PARAKEET_CLI_MODE":
                raise RuntimeError(f"Parakeet MLX STT model '{parakeet_model_id}' failed to initialize for CLI mode correctly.")
            _loaded_models_cache["stt_model"] = model_indicator
            _loaded_models_cache["stt_tokenizer"] = tokenizer_indicator
            logger.info(f"Parakeet MLX STT configured for CLI processing using model: {parakeet_model_id}")
        else:
            raise ValueError(f"Unsupported STT engine configured: {stt_engine_name}")

        diar_model_name = aps_cfg.get('diarization_model')
        hf_token_from_config = aps_utils.load_huggingface_token()
        if diar_model_name:
            logger.info(f"Loading Diarization model: {diar_model_name} on device: {pytorch_processing_device}")
            _loaded_models_cache["diarization"] = aps_models.load_diarization_pipeline(
                diar_model_name, pytorch_processing_device, hf_token_from_config
            )
            if _loaded_models_cache["diarization"]: logger.info("Diarization model loaded successfully.")
            else: logger.warning("Diarization model failed to load. External diarization may be skipped.")
        else:
            logger.info("Diarization model not configured. Skipping external diarization model loading.")

        emb_model_name = aps_cfg.get('embedding_model')
        if emb_model_name:
            aps_config_from_getter = get_config().get('audio_suite_settings', {})
            emb_cache_dir_resolved = aps_config_from_getter.get('_resolved_embedding_model_cache_dir')

            if not emb_cache_dir_resolved :
                    default_base_model_dir = PROJECT_ROOT / "data" / "models"
                    embedding_cache_subdir_name = aps_cfg.get('embedding_model_cache_subdir', 'models/embedding_cache_default')
                    emb_cache_dir_resolved = (default_base_model_dir / str(embedding_cache_subdir_name)).resolve()
                    logger.warning(f"_resolved_embedding_model_cache_dir not found in config, using default: {emb_cache_dir_resolved}")

            logger.info(f"Loading Embedding model: {emb_model_name} on device: {pytorch_processing_device}")
            _loaded_models_cache["embedding_model"], emb_dim = \
                aps_models.load_embedding_model(emb_model_name, pytorch_processing_device, emb_cache_dir_resolved) # type: ignore
            
            _loaded_models_cache["embedding_dim"] = emb_dim if emb_dim and emb_dim > 0 else aps_cfg.get('embedding_dim')

            if not _loaded_models_cache["embedding_model"]:
                logger.warning("Embedding model failed to load. Speaker ID will be limited/skipped.")
            else:
                logger.info(f"Embedding model loaded. Dimension: {_loaded_models_cache['embedding_dim']}")
        else:
            logger.info("Embedding model not configured. Skipping embedding model loading.")

        punc_model_name = aps_cfg.get('punctuation_model')
        if punc_model_name:
            logger.info(f"Loading Punctuation model: {punc_model_name} on device: {pytorch_processing_device}")
            _loaded_models_cache["punctuation"] = aps_models.load_punctuation_model(
                punc_model_name, pytorch_processing_device
            )
            if _loaded_models_cache["punctuation"]: logger.info("Punctuation model loaded successfully.")
            else: logger.warning("Punctuation model failed to load. Punctuation will be skipped.")
        else:
            logger.info("Punctuation model not configured. Skipping punctuation model loading.")

        logger.info("All configured audio processing models initialized (or set for CLI).")

    except Exception as e:
        logger.critical(f"Failed to initialize one or more audio processing models: {e}", exc_info=True)
        cleanup_audio_models()
        raise

def cleanup_audio_models() -> None:
    logger.info("Cleaning up audio processing models...")
    config = get_config()
    pytorch_processing_device = aps_utils.check_device(config.get('global_ml_device', 'cpu'))
    use_cuda = (pytorch_processing_device == 'cuda')

    stt_model_obj_to_clean = None
    if _loaded_models_cache.get("stt_engine") == "openai_whisper":
        stt_model_obj_to_clean = _loaded_models_cache.get("stt_model")

    aps_utils.cleanup_resources(
        stt_model_obj_to_clean,
        _loaded_models_cache.get("diarization"),
        _loaded_models_cache.get("embedding_model"),
        _loaded_models_cache.get("punctuation"),
        use_cuda=use_cuda
    )
    
    for key in list(_loaded_models_cache.keys()):
        _loaded_models_cache[key] = None
    logger.info("Audio processing models cache cleared and resources cleaned up.")


def _convert_aac_to_wav(aac_file_path: Path, output_wav_path: Path, target_sr: int) -> bool:
    config = get_config()
    tools_cfg = config.get('tools', {})
    ffmpeg_path = tools_cfg.get('ffmpeg_path', 'ffmpeg')
    try:
        logger.info(f"Converting {aac_file_path} to {output_wav_path} (SR: {target_sr}Hz) using '{ffmpeg_path}'...")
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        
        command = [
            ffmpeg_path,
            "-hide_banner", "-loglevel", "error",
            "-i", str(aac_file_path),
            "-acodec", "pcm_s16le",
            "-ar", str(target_sr),
            "-ac", "1",
            "-y",
            str(output_wav_path)
        ]
        process = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8')
        
        if process.returncode == 0:
            logger.info(f"Successfully converted '{aac_file_path.name}' to '{output_wav_path.name}' (SR: {target_sr}Hz)")
            return True
        else:
            logger.error(f"FFMPEG conversion failed for '{aac_file_path.name}'. SR: {target_sr}Hz. Code: {process.returncode}")
            if process.stdout: logger.error(f"FFMPEG stdout: {process.stdout.strip()}")
            if process.stderr:
                # Filter out harmless MallocStackLogging messages on macOS
                filtered_stderr = "\n".join(
                    line for line in process.stderr.splitlines()
                    if "MallocStackLogging: can't turn off malloc stack logging because it was not enabled." not in line
                ).strip()
                if filtered_stderr:
                    logger.error(f"FFMPEG stderr: {filtered_stderr}")
            return False
    except FileNotFoundError:
        logger.error(f"FFMPEG executable not found at '{ffmpeg_path}'. Please ensure it's installed and in PATH or configured correctly.")
        return False
    except Exception as e:
        logger.error(f"Error during FFMPEG conversion for '{aac_file_path.name}': {e}", exc_info=True)
        return False

def _reconstruct_words_from_parakeet_tokens(
    parakeet_tokens: List[Dict[str, Any]],
    sentence_speaker: str
) -> List[Dict[str, Any]]:
    """
    Reconstructs whole words from Parakeet's potentially sub-word tokens.
    Merges consecutive tokens that seem to form a single word.
    """
    if not parakeet_tokens:
        return []

    reconstructed_words: List[Dict[str, Any]] = []
    current_word_parts: List[str] = []
    current_word_start_time: Optional[float] = None
    current_word_end_time: Optional[float] = None
    current_word_probabilities: List[float] = []

    for i, token_info in enumerate(parakeet_tokens):
        # Get the original token text for checking leading spaces, and stripped text for word building
        raw_token_text_unstripped = token_info.get("text", "") # Original form
        token_text_stripped = raw_token_text_unstripped.strip()   # Stripped form for word parts

        start_time = token_info.get("start")
        end_time = token_info.get("end")
        # Ensure probability is float, handle None or empty string from get()
        prob_val = token_info.get("score", token_info.get("probability"))
        probability = float(prob_val) if prob_val is not None and str(prob_val).strip() != "" else 0.0


        # If the stripped token is empty, it means the original was only whitespace or empty.
        # Such tokens don't contribute to word parts but might signal the end of a previous word.
        if not token_text_stripped:
            if current_word_parts: # Finalize the current word if parts have been accumulated
                reconstructed_words.append({
                    "word": "".join(current_word_parts),
                    "start": current_word_start_time,
                    "end": current_word_end_time,
                    "probability": sum(current_word_probabilities) / len(current_word_probabilities) if current_word_probabilities else 0.0,
                    "speaker": sentence_speaker
                })
                current_word_parts = []
                current_word_start_time = None
                current_word_end_time = None # Reset end time as well
                current_word_probabilities = []
            continue # Move to the next token

        # If we are at the beginning of building a word or if the current token signals a new word
        if not current_word_parts:
            current_word_parts.append(token_text_stripped)
            current_word_start_time = start_time
            current_word_end_time = end_time
            if probability is not None: current_word_probabilities.append(probability)
        else:
            # Heuristic: if the raw (unstripped) token does NOT start with a space,
            # and it's not empty after stripping, it's likely a continuation of the current word.
            if not raw_token_text_unstripped.startswith(" ") and token_text_stripped:
                current_word_parts.append(token_text_stripped)
                # Update end time to the end time of this token
                if end_time is not None:
                    current_word_end_time = end_time
                if probability is not None: current_word_probabilities.append(probability)
            else:
                # The token starts a new word (e.g., it started with a space, or previous conditions not met)
                # Finalize the previously accumulated word
                if current_word_parts: # Should always be true here due to the outer 'if not current_word_parts'
                    reconstructed_words.append({
                        "word": "".join(current_word_parts),
                        "start": current_word_start_time,
                        "end": current_word_end_time,
                        "probability": sum(current_word_probabilities) / len(current_word_probabilities) if current_word_probabilities else 0.0,
                        "speaker": sentence_speaker
                    })
                
                # Start a new word with the current token
                current_word_parts = [token_text_stripped]
                current_word_start_time = start_time
                current_word_end_time = end_time
                current_word_probabilities = [probability] if probability is not None else []

    # After the loop, add any last accumulated word
    if current_word_parts:
        reconstructed_words.append({
            "word": "".join(current_word_parts),
            "start": current_word_start_time,
            "end": current_word_end_time,
            "probability": sum(current_word_probabilities) / len(current_word_probabilities) if current_word_probabilities else 0.0,
            "speaker": sentence_speaker
        })

    return reconstructed_words


def _adapt_parakeet_tdt_output(parakeet_result: Any) -> List[Dict[str, Any]]:
    """
    Adapts the JSON output from Parakeet TDT (especially via mlx_audio.stt.generate)
    to the expected internal format (List of segments, each with words).
    Reconstructs whole words from Parakeet tokens.
    """
    adapted_segments: List[Dict[str, Any]] = []

    if not isinstance(parakeet_result, dict):
        logger.warning(f"Adapter: Parakeet TDT result is not a dict. Got: {type(parakeet_result)}, Content: {parakeet_result}")
        return []

    if 'sentences' in parakeet_result and isinstance(parakeet_result['sentences'], list):
        logger.debug(f"Adapter: Found 'sentences' key. Processing {len(parakeet_result['sentences'])} sentences as segments.")
        for p_sentence in parakeet_result['sentences']:
            if not isinstance(p_sentence, dict):
                logger.warning(f"Adapter: Skipping non-dict item in sentences list: {p_sentence}")
                continue

            parakeet_tokens_in_sentence: List[Dict[str, Any]] = []
            if 'tokens' in p_sentence and isinstance(p_sentence['tokens'], list):
                parakeet_tokens_in_sentence = p_sentence['tokens']
            
            sentence_speaker = str(p_sentence.get('speaker', 'UNKNOWN_SPEAKER'))

            # Reconstruct words from tokens for this sentence
            reconstructed_words_for_sentence = _reconstruct_words_from_parakeet_tokens(
                parakeet_tokens_in_sentence, sentence_speaker
            )
            
            final_words_for_segment: List[Dict[str, Any]] = []
            for r_word_info in reconstructed_words_for_sentence:
                word_text = r_word_info.get("word", "")
                # Filter out reconstructed words that are ONLY punctuation or underscore
                if word_text and not re.fullmatch(r'[\W_]+', word_text): # Word must exist and not be solely punctuation/underscore
                    final_words_for_segment.append(r_word_info)
                else:
                    logger.debug(f"Adapter: Filtering out reconstructed punctuation/empty/underscore-only word: '{word_text}'")
            
            # Use original sentence text if available and not empty, otherwise reconstruct from our new words
            segment_text = str(p_sentence.get('text', '')).strip() 
            if not segment_text and final_words_for_segment: 
                segment_text = " ".join(w['word'] for w in final_words_for_segment)
                # Apply basic punctuation spacing fixes for the reconstructed segment text
                segment_text = segment_text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
                segment_text = segment_text.replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re").replace(" 'll", "'ll").replace(" 'd", "'d").replace(" n't", "n't")
                # Potentially more rules from a text normalization function could be applied here if needed.
                logger.debug(f"Adapter: Reconstructed segment text from reconstructed words: '{segment_text}'")


            segment_start = p_sentence.get('start') # Keep as original type for now, convert to float later
            segment_end = p_sentence.get('end')

            # If segment start/end are missing or 0.0, try to derive from words
            # Ensure start/end are float before comparison or assignment
            try:
                s_start_float = float(segment_start) if segment_start is not None else 0.0
                s_end_float = float(segment_end) if segment_end is not None else 0.0
            except ValueError:
                logger.warning(f"Adapter: Invalid start/end time for sentence, defaulting. Sentence: {p_sentence}")
                s_start_float = 0.0
                s_end_float = 0.0

            if (s_start_float == 0.0 and s_end_float == 0.0) and final_words_for_segment:
                if final_words_for_segment[0].get('start') is not None:
                        s_start_float = float(final_words_for_segment[0]['start'])
                if final_words_for_segment[-1].get('end') is not None:
                        s_end_float = float(final_words_for_segment[-1]['end'])
                logger.debug(f"Adapter: Derived segment start/end from reconstructed words: {s_start_float}-{s_end_float}")

            # Only add segment if it has meaningful text or actual words
            if segment_text or final_words_for_segment:
                adapted_segments.append({
                    "start": float(s_start_float), # Ensure float
                    "end": float(s_end_float),     # Ensure float
                    "text": segment_text, 
                    "words": final_words_for_segment, 
                    "speaker": str(sentence_speaker)
                })
            else:
                logger.debug(f"Adapter: Skipping empty sentence (no text and no reconstructed words): {p_sentence}")
        return adapted_segments

    # Case 2: Output has "segments" (this might need word reconstruction too if words are tokenized)
    elif 'segments' in parakeet_result and isinstance(parakeet_result['segments'], list):
        segments_to_process = parakeet_result['segments']
        logger.debug(f"Adapter: Found 'segments' key (not 'sentences'). Processing {len(segments_to_process)} segments in fallback mode.")
        for p_segment in segments_to_process:
            if not isinstance(p_segment, dict):
                logger.warning(f"Adapter: Skipping non-dict item in segments list: {p_segment}")
                continue

            segment_speaker_candidate = str(p_segment.get('speaker', 'UNKNOWN_SPEAKER'))
            
            # Tentatively apply word reconstruction if 'words' are like Parakeet tokens
            # This assumes 'p_segment['words']' would be a list of token-like dicts
            raw_words_in_segment = p_segment.get('words', [])
            if isinstance(raw_words_in_segment, list) and raw_words_in_segment and isinstance(raw_words_in_segment[0], dict) and "text" in raw_words_in_segment[0]:
                    logger.debug(f"Adapter (segment mode): Attempting to reconstruct words from tokens for segment.")
                    reconstructed_segment_words = _reconstruct_words_from_parakeet_tokens(raw_words_in_segment, segment_speaker_candidate)
                    
                    # Filter punctuation-only words from these reconstructed words
                    processed_segment_words: List[Dict[str,Any]] = []
                    for r_word_info in reconstructed_segment_words:
                        word_text = r_word_info.get("word", "")
                        if word_text and not re.fullmatch(r'[\W_]+', word_text):
                            processed_segment_words.append(r_word_info)
                        else:
                            logger.debug(f"Adapter (segment mode): Filtering out reconstructed punctuation-only word: '{word_text}'")
                    current_adapted_words = processed_segment_words
            elif isinstance(raw_words_in_segment, list): # If it's a list but not fitting token structure, process as before
                logger.debug(f"Adapter (segment mode): Processing words as pre-defined (not reconstructing).")
                current_adapted_words = []
                for p_word_info in raw_words_in_segment:
                    if not isinstance(p_word_info, dict): continue
                    word_text_val = p_word_info.get('text', p_word_info.get('word'))
                    if word_text_val is None: continue
                    clean_word_text = str(word_text_val).strip()
                    if not clean_word_text or re.fullmatch(r'[\W_]+', clean_word_text): continue
                    
                    start_time = p_word_info.get('start')
                    end_time = p_word_info.get('end')
                    if start_time is None or end_time is None: continue

                    current_adapted_words.append({
                        "word": clean_word_text,
                        "start": float(start_time),
                        "end": float(end_time),
                        "probability": float(p_word_info.get('score', p_word_info.get('probability', 0.0))),
                        "speaker": str(p_word_info.get('speaker', segment_speaker_candidate)) # Prefer word's speaker
                    })
            else: # words key is missing or not a list
                current_adapted_words = []


            segment_text = str(p_segment.get('text', '')).strip()
            if not segment_text and current_adapted_words:
                segment_text = " ".join(w['word'] for w in current_adapted_words)
                segment_text = segment_text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!") # etc.
                logger.debug(f"Adapter (segment mode): Reconstructed segment text from words: '{segment_text}'")
            
            segment_start_val = p_segment.get('start', 0.0)
            segment_end_val = p_segment.get('end', 0.0)

            if segment_text or current_adapted_words:
                adapted_segments.append({
                    "start": float(segment_start_val),
                    "end": float(segment_end_val),
                    "text": segment_text,
                    "words": current_adapted_words,
                    "speaker": segment_speaker_candidate 
                })
            else:
                logger.debug(f"Adapter (segment mode): Skipping empty segment: {p_segment}")
        return adapted_segments

    # Case 3: Output has only top-level "text" (fallback)
    elif 'text' in parakeet_result:
        full_text = str(parakeet_result.get('text', '')).strip()
        logger.info(f"Adapter: Parakeet output has 'text' ('{full_text[:100]}...') but no 'segments' or 'sentences' array. Creating a single placeholder segment.")
        if full_text:
            language = parakeet_result.get('language', 'unknown')
            adapted_segments.append({
                "start": 0.0,
                "end": 0.0, 
                "text": full_text,
                "words": [], 
                "speaker": "UNKNOWN_SPEAKER",
                "language": language 
            })
        else:
            logger.info("Adapter: Parakeet 'text' field was empty, and no segments/sentences. Returning empty.")
        return adapted_segments

    # Case 4: Unrecognized dictionary format or empty dictionary
    else:
        logger.warning(f"Adapter: Parakeet TDT result is a dict, but not in a recognized format (no 'sentences', 'segments', or 'text' key found). Content: {parakeet_result}")
        return []


def _run_stt_on_wav(wav_file_path: Path) -> List[Dict[str, Any]]:
    config = get_config()
    stt_engine = _loaded_models_cache.get("stt_engine")
    python_executable = sys.executable
    transcript_results: List[Dict[str, Any]] = []

    if stt_engine == "openai_whisper":
        stt_model = _loaded_models_cache.get("stt_model")
        if not stt_model:
            logger.error(f"STT model ({stt_engine}) not loaded. Cannot perform transcription.")
            # In case of critical model load failure, an empty list is returned.
            # The caller (process_single_audio_file) will interpret this as "no_dialogue_found"
            # or if an exception was raised during model loading, it would be a "processing_error".
            return []
        whisper_cfg = config.get('stt', {}).get('openai_whisper', {})
        stt_language = whisper_cfg.get('language')
        
        logger.info(f"Running OpenAI Whisper STT on: {wav_file_path}" + (f" with language: {stt_language}" if stt_language else " (auto-detect lang)"))
        transcribe_options: Dict[str, Any] = {"word_timestamps": True, "verbose": None} # verbose=None should use model default
        if stt_language:
            transcribe_options["language"] = stt_language
        
        try:
            result = stt_model.transcribe(str(wav_file_path), **transcribe_options)
            
            if result and 'segments' in result:
                for segment in result['segments']:
                    words_list: List[Dict[str, Any]] = []
                    if 'words' in segment and isinstance(segment['words'], list):
                        for w in segment['words']:
                            try:
                                word_text_val = w.get('word')
                                if word_text_val is None: continue
                                
                                clean_word_text = str(word_text_val).strip()
                                if not clean_word_text: continue 

                                words_list.append({
                                    "word": clean_word_text, 
                                    "start": float(w['start']),
                                    "end": float(w['end']),
                                    "probability": float(w.get('probability', w.get('score', 0.0)))
                                })
                            except (TypeError, ValueError, KeyError) as e:
                                logger.warning(f"Skipping invalid word data in Whisper segment: {w}. Error: {e}")
                    
                    transcript_results.append({
                        "start": float(segment.get('start', 0.0)),
                        "end": float(segment.get('end', 0.0)),
                        "text": str(segment.get('text', '')).strip(),
                        "words": words_list
                    })
            elif result and 'text' in result and not transcript_results: 
                logger.warning("Whisper STT returned only full text. Creating a single segment.")
                full_text = str(result['text']).strip()
                if full_text:
                    transcript_results.append({"start":0.0, "end":0.0, "text": full_text, "words":[]})
            return transcript_results
        except Exception as e:
            logger.error(f"OpenAI Whisper STT process failed critically for {wav_file_path.name}: {e}", exc_info=True)
            # This exception, if not caught and handled to return [], would propagate up
            # and be caught by process_single_audio_file's main try-except, resulting in "processing_error".
            # For now, returning [] to let the "no_dialogue_found" logic in caller handle it,
            # unless the prompt implies _run_stt_on_wav should raise for tool crashes.
            # Given the prompt's example for STT tool crash was hypothetical, returning [] for now.
            # If a critical failure is to be distinguished, this function should raise an exception.
            return []


    elif stt_engine == "parakeet_mlx":
        parakeet_cfg = config.get('stt', {}).get('parakeet_mlx', {})
        parakeet_model_id = parakeet_cfg.get('model_path')
        if not parakeet_model_id:
            logger.error("Parakeet MLX model path/ID ('stt.parakeet_mlx.model_path') not configured for CLI usage.")
            return [] # Indicates failure to produce transcript
        
        output_basename_str = f"{wav_file_path.stem}_stt_cli_output"
        cli_output_arg_path = wav_file_path.parent / output_basename_str 
        expected_stt_output_json_path = cli_output_arg_path.with_suffix(".json") 

        logger.info(f"Running Parakeet MLX STT via CLI for: {wav_file_path}")
        logger.info(f"  Model: {parakeet_model_id}")
        logger.info(f"  Expecting temporary JSON output at: {expected_stt_output_json_path}")

        cmd = [
            python_executable, "-m", "mlx_audio.stt.generate",
            "--model", str(parakeet_model_id),
            "--audio", str(wav_file_path),
            "--output", str(cli_output_arg_path), 
            "--format", "json" 
        ]

        logger.debug(f"Executing command: {' '.join(cmd)}")
        parakeet_raw_output = None 
        process = None 
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8')
            
            if process.stdout: logger.debug(f"Parakeet CLI stdout: {process.stdout.strip()}")
            if process.stderr: logger.debug(f"Parakeet CLI stderr: {process.stderr.strip()}") 

            if process.returncode == 0:
                logger.info(f"Parakeet MLX CLI process reported success. Reading output from {expected_stt_output_json_path}")
                if expected_stt_output_json_path.exists():
                    with open(expected_stt_output_json_path, 'r', encoding='utf-8') as f:
                        parakeet_raw_output = json.load(f)
                    
                    logger.debug(f"Raw Parakeet CLI JSON output loaded (type: {type(parakeet_raw_output)}). Preview: {str(parakeet_raw_output)[:500]}") 
                    
                    transcript_results = _adapt_parakeet_tdt_output(parakeet_raw_output)
                    
                    if not transcript_results and parakeet_raw_output: 
                        logger.warning(f"Parakeet adapter returned empty results despite non-empty raw output. Check adapter logs.")
                else:
                    logger.error(f"Parakeet CLI process reported success, but output file {expected_stt_output_json_path} not found.")
                    return [] # Indicates failure, leads to no_dialogue or processing_error if exception propagates
            else:
                logger.error(f"Parakeet MLX CLI process failed with return code {process.returncode}.")
                if process.stderr: logger.error(f"Parakeet CLI stderr (on error): {process.stderr.strip()}")
                return [] # Indicates failure
        except FileNotFoundError:
            logger.error(f"Python executable '{python_executable}' or mlx_audio.stt.generate module not found.")
            # This is a critical error. Raising an exception would be appropriate to signal "processing_error".
            # For now, following pattern of returning [].
            raise RuntimeError(f"STT subprocess execution failed: Python or mlx_audio.stt.generate not found.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from Parakeet CLI output file {expected_stt_output_json_path}: {e}")
            try:
                with open(expected_stt_output_json_path, 'r', encoding='utf-8') as f_err:
                    file_content_preview = f_err.read(1000)
                    logger.error(f"Content of {expected_stt_output_json_path} (first 1KB): {file_content_preview}...")
            except Exception as fe_read:
                logger.error(f"Could not read content of error file {expected_stt_output_json_path}: {fe_read}")
            raise RuntimeError(f"STT output JSON decoding failed for {wav_file_path.name}.") # Propagate as error
        except Exception as e:
            logger.error(f"An unexpected error occurred during Parakeet CLI STT execution: {e}", exc_info=True)
            raise # Re-raise to be caught by process_single_audio_file as a "processing_error"
        finally:
            if expected_stt_output_json_path.exists():
                logger.info(f"Temporary STT CLI output RETAINED for inspection: {expected_stt_output_json_path}")
            elif process and process.returncode == 0: 
                logger.warning(f"Parakeet CLI reported success but {expected_stt_output_json_path} was not found post-execution.")

        if not transcript_results:
            logger.warning(f"Parakeet STT for {wav_file_path.name} resulted in no processable transcript segments. Raw output was: {str(parakeet_raw_output)[:500]}")
        
        return transcript_results
    else:
        logger.error(f"Unknown STT engine '{stt_engine}' encountered in _run_stt_on_wav.")
        return []

def _consolidate_pyannote_turns(
    raw_turns: List[Dict[str, Any]],
    min_segment_duration_s: float,
    max_gap_between_segments_to_merge_s: float
) -> List[Dict[str, Any]]:
    if not raw_turns:
        return []

    # Make a mutable copy, sorted by start time. dict.copy() is sufficient.
    current_turns = [turn.copy() for turn in sorted(raw_turns, key=lambda x: x['start'])]

    # --- Pass 1: Merge short segments into neighbors ---
    pass1_iteration_count = 0
    MAX_ITERATIONS_PASS_1 = len(current_turns) + 5  # Heuristic limit

    while pass1_iteration_count < MAX_ITERATIONS_PASS_1:
        pass1_iteration_count += 1
        made_change_in_this_iteration_pass1 = False
        
        if not current_turns:
            break

        indices_to_remove = [False] * len(current_turns)
        
        for i in range(len(current_turns)):
            if indices_to_remove[i]:
                continue

            turn = current_turns[i]
            turn_duration = turn['end'] - turn['start']

            if turn_duration < min_segment_duration_s:
                prev_turn_info: Optional[Tuple[Dict[str, Any], int]] = None
                for k in range(i - 1, -1, -1):
                    if not indices_to_remove[k]:
                        prev_turn_info = (current_turns[k], k)
                        break
                
                next_turn_info: Optional[Tuple[Dict[str, Any], int]] = None
                for k in range(i + 1, len(current_turns)):
                    if not indices_to_remove[k]: 
                        next_turn_info = (current_turns[k], k)
                        break
                
                gap_to_prev = float('inf')
                if prev_turn_info:
                    gap_to_prev = turn['start'] - prev_turn_info[0]['end']

                gap_to_next = float('inf')
                if next_turn_info:
                    gap_to_next = next_turn_info[0]['start'] - turn['end']

                can_merge_prev = prev_turn_info and gap_to_prev <= max_gap_between_segments_to_merge_s
                can_merge_next = next_turn_info and gap_to_next <= max_gap_between_segments_to_merge_s
                
                merged_this_short_turn = False
                if can_merge_prev and can_merge_next:
                    if gap_to_prev <= gap_to_next: 
                        prev_idx = prev_turn_info[1]
                        current_turns[prev_idx]['end'] = turn['end']
                        indices_to_remove[i] = True
                        made_change_in_this_iteration_pass1 = True
                        merged_this_short_turn = True
                    else: 
                        next_idx = next_turn_info[1]
                        current_turns[next_idx]['start'] = turn['start']
                        indices_to_remove[i] = True
                        made_change_in_this_iteration_pass1 = True
                        merged_this_short_turn = True
                elif can_merge_prev:
                    prev_idx = prev_turn_info[1]
                    current_turns[prev_idx]['end'] = turn['end']
                    indices_to_remove[i] = True
                    made_change_in_this_iteration_pass1 = True
                    merged_this_short_turn = True
                elif can_merge_next:
                    next_idx = next_turn_info[1]
                    current_turns[next_idx]['start'] = turn['start']
                    indices_to_remove[i] = True
                    made_change_in_this_iteration_pass1 = True
                    merged_this_short_turn = True
        
        if made_change_in_this_iteration_pass1:
            new_current_turns = []
            for i_rebuild in range(len(current_turns)):
                if not indices_to_remove[i_rebuild]:
                    new_current_turns.append(current_turns[i_rebuild])
            current_turns = new_current_turns
            current_turns.sort(key=lambda x: x['start']) 
        else:
            break 
    
    if pass1_iteration_count >= MAX_ITERATIONS_PASS_1 and made_change_in_this_iteration_pass1:
        logger.warning(f"_consolidate_pyannote_turns: Pass 1 reached max iterations ({MAX_ITERATIONS_PASS_1}).")

    if not current_turns:
        return []


    return current_turns

def _run_diarization_on_wav(wav_file_path: Path) -> Optional[List[Dict[str, Any]]]:
    diarization_pipeline = _loaded_models_cache.get("diarization")
    stt_engine_name = _loaded_models_cache.get("stt_engine") 

    if stt_engine_name == "parakeet_mlx": 
        logger.info("Parakeet MLX (TDT) is STT engine. Its output might already include speaker info. External diarization (Pyannote) will be run if configured, and results merged if Parakeet's STT output lacks speakers.")

    if not diarization_pipeline:
        logger.info("External Diarization model (Pyannote) not loaded. Skipping external diarization.")
        return None 

    logger.info(f"Running external diarization (Pyannote) on: {wav_file_path}")
    try:
        diarization_output = diarization_pipeline(str(wav_file_path)) 
        diarization_results_raw: List[Dict[str, Any]] = []
        if diarization_output: 
            for turn, _, label in diarization_output.itertracks(yield_label=True):
                diarization_results_raw.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(label)})
            diarization_results_raw.sort(key=lambda x: x['start'])
        
        if diarization_results_raw:
            try:
                sample_log_count = min(len(diarization_results_raw), 5)
                logger.debug(f"Pyannote raw diarization_results (first {sample_log_count} of {len(diarization_results_raw)}): "
                             f"{json.dumps(diarization_results_raw[:sample_log_count], indent=2)}")
            except TypeError: 
                logger.debug(f"Pyannote raw diarization_results (could not serialize to JSON, logging raw sample): "
                             f"{str(diarization_results_raw[:5])}...")
        
        config = get_config()
        aps_cfg = config.get('audio_suite_settings', {})
        min_seg_ms = aps_cfg.get('diarization_consolidation_min_segment_ms', 750)
        max_gap_ms = aps_cfg.get('diarization_consolidation_max_gap_ms', 200)

        min_segment_duration_s = float(min_seg_ms) / 1000.0
        max_gap_between_segments_to_merge_s = float(max_gap_ms) / 1000.0

        diarization_results_consolidated = _consolidate_pyannote_turns(
            diarization_results_raw,
            min_segment_duration_s=min_segment_duration_s,
            max_gap_between_segments_to_merge_s=max_gap_between_segments_to_merge_s
        )
        
        num_raw_turns = len(diarization_results_raw)
        num_consolidated_turns = len(diarization_results_consolidated)
        logger.info(f"Pyannote diarization: {num_raw_turns} raw turns -> {num_consolidated_turns} consolidated turns.")

        return diarization_results_consolidated

    except Exception as e:
        logger.error(f"Error during external diarization for {wav_file_path.name}: {e}", exc_info=True)
        # If diarization fails critically, it should raise an exception to be caught by the main handler
        # resulting in "processing_error". Returning None here means the pipeline continues without diarization.
        # This behavior depends on whether diarization failure is considered a full pipeline failure.
        # Based on current structure, returning None allows pipeline to proceed with STT-only speakers or UNKNOWNs.
        # If it *must* be a "processing_error", this should `raise e`.
        # For now, returning None as per original behavior (implies diarization is optional for overall success).
        return None 

# <<< REPLACED FUNCTION >>>
def _combine_asr_diarization(
    transcript_results: List[Dict[str, Any]], 
    diarization_results: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Combines STT results with external diarization by assigning words to speaker
    turns based on maximum temporal overlap. This is more accurate than midpoint logic.
    """
    if not transcript_results:
        logger.warning("CombineASR: No STT results provided. Returning empty list.")
        return []

    # Fallback: If no external diarization, trust the STT's speaker labels.
    if not diarization_results:
        logger.info("CombineASR: No external diarization results. Using speaker labels from STT if available.")
        for segment in transcript_results:
            segment_speaker = segment.get("speaker", "UNKNOWN_SPEAKER")
            if 'words' in segment and isinstance(segment['words'], list):
                for word in segment['words']:
                    if 'speaker' not in word or not word['speaker']:
                        word['speaker'] = segment_speaker
        return transcript_results

    logger.info("CombineASR: Aligning STT words to diarization timeline using maximum temporal overlap.")

    # 1. Prepare data structures for efficient lookup
    all_words = []
    for stt_segment in transcript_results:
        if 'words' in stt_segment and isinstance(stt_segment['words'], list):
            all_words.extend(stt_segment['words'])
    
    # Sort both lists by start time. Diarization results are often already sorted.
    all_words.sort(key=lambda x: x.get('start', float('inf')))
    diarization_results.sort(key=lambda x: x.get('start', float('inf')))

    turn_idx = 0
    # 2. Assign speaker to each word based on max overlap
    for word in all_words:
        word_start = word.get('start')
        word_end = word.get('end')

        if not all(isinstance(v, (float, int)) for v in [word_start, word_end]):
            word['speaker'] = "UNKNOWN_SPEAKER"
            continue

        # Optimization: Start searching for speaker turns from the last used index
        best_speaker = "UNKNOWN_SPEAKER"
        max_overlap = 0.0

        # Move turn_idx to the first turn that could possibly overlap with the current word
        while turn_idx < len(diarization_results) and diarization_results[turn_idx].get('end', 0) < word_start:
            turn_idx += 1
        
        # Check subsequent turns for overlap
        temp_turn_idx = turn_idx
        while temp_turn_idx < len(diarization_results):
            turn = diarization_results[temp_turn_idx]
            turn_start = turn.get('start', float('inf'))
            turn_end = turn.get('end', float('inf'))
            
            # If the turn starts after the word ends, no further turns will overlap
            if turn_start > word_end:
                break

            overlap = max(0, min(word_end, turn_end) - max(word_start, turn_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = turn.get('speaker', "UNKNOWN_SPEAKER")
            
            temp_turn_idx += 1
            
        word['speaker'] = best_speaker

    # 3. Re-group words back into segments and recalculate segment-level speaker
    for segment in transcript_results:
        if 'words' in segment and isinstance(segment['words'], list) and segment['words']:
            speaker_counts = Counter(w.get('speaker', "UNKNOWN_SPEAKER") for w in segment['words'])
            if speaker_counts:
                # Set segment speaker to the most common one among its words
                segment['speaker'] = speaker_counts.most_common(1)[0][0]
            else:
                segment['speaker'] = "UNKNOWN_SPEAKER"

    logger.info(f"CombineASR: Finished combining. Produced {len(transcript_results)} segments based on external diarization.")
    return transcript_results


def _post_process_consolidate_unknowns(
    transcript: List[Dict[str, Any]], 
    known_speaker_labels_from_diar: List[str] 
) -> List[Dict[str, Any]]:
    """
    Post-processes the transcript to re-assign words/segments labeled 'UNKNOWN_SPEAKER'
    to the chronologically closest known speaker from the diarization output.
    """
    if not known_speaker_labels_from_diar:
        logger.info("ConsolidateUnknowns: No known speaker labels from diarization provided. Skipping consolidation.")
        return transcript
    
    valid_known_labels = {str(label).upper() for label in known_speaker_labels_from_diar if label and str(label).strip().upper() != "UNKNOWN_SPEAKER"}
    
    if not valid_known_labels:
        logger.info("ConsolidateUnknowns: No valid (non-UNKNOWN) known speaker labels from diarization. Skipping consolidation.")
        return transcript

    logger.info(f"ConsolidateUnknowns: Attempting to consolidate UNKNOWN_SPEAKER labels using known set: {valid_known_labels}")
    consolidated_word_count = 0 # Initialize counter
    
    all_words_indexed: List[Dict[str, Any]] = []
    for seg_idx, segment_data in enumerate(transcript):
        if 'words' in segment_data and isinstance(segment_data['words'], list):
            for word_idx, word_data in enumerate(segment_data['words']):
                all_words_indexed.append({
                    **word_data, 
                    '_seg_idx': seg_idx, 
                    '_word_idx': word_idx
                })
    
    if not all_words_indexed:
        logger.info("ConsolidateUnknowns: No words in transcript to process.")
        return transcript

    all_words_indexed.sort(key=lambda x: x.get('start', float('inf')))

    for i, current_word_info in enumerate(all_words_indexed):
        current_word_speaker_original_case = str(current_word_info.get('speaker') or "UNKNOWN_SPEAKER")
        current_word_speaker_upper = current_word_speaker_original_case.upper()
        
        current_word_text_debug = str(current_word_info.get("word", "")).strip() 
        current_word_start_debug = current_word_info.get("start") 
        
        if current_word_speaker_upper == "UNKNOWN_SPEAKER": 
            original_problem_speaker = current_word_speaker_original_case 
            current_word_start = current_word_info.get('start')
            current_word_end = current_word_info.get('end')
            
            assigned_new_speaker = "UNKNOWN_SPEAKER" 
            
            time_tolerance = 0.001  # 1 millisecond

            if isinstance(current_word_start, (float, int)) and isinstance(current_word_end, (float, int)):
                prev_known_speaker_label = None
                prev_known_speaker_end_time = -float('inf')
                for j in range(i - 1, -1, -1):
                    prev_w = all_words_indexed[j]
                    prev_w_speaker_original_case = str(prev_w.get('speaker'))
                    if prev_w_speaker_original_case.upper() in valid_known_labels:
                        prev_known_speaker_label = prev_w_speaker_original_case 
                        prev_w_end_val = prev_w.get('end')
                        if isinstance(prev_w_end_val, (float,int)):
                                prev_known_speaker_end_time = float(prev_w_end_val)
                        else: 
                                prev_prev_w_start_val = prev_w.get('start')
                                if isinstance(prev_prev_w_start_val, (float,int)):
                                    prev_known_speaker_end_time = float(prev_prev_w_start_val) 
                                else: 
                                    prev_known_speaker_label = None 
                                    break 
                        break
                
                next_known_speaker_label = None
                next_known_speaker_start_time = float('inf')
                for j in range(i + 1, len(all_words_indexed)):
                    next_w = all_words_indexed[j]
                    next_w_speaker_original_case = str(next_w.get('speaker'))
                    if next_w_speaker_original_case.upper() in valid_known_labels:
                        next_known_speaker_label = next_w_speaker_original_case
                        next_w_start_val = next_w.get('start')
                        if isinstance(next_w_start_val, (float,int)):
                            next_known_speaker_start_time = float(next_w_start_val)
                        else: 
                            next_known_speaker_label = None 
                            break
                        break

                config = get_config() 
                aps_cfg = config.get('audio_suite_settings', {})
                # Task 2: Refine Threshold for _post_process_consolidate_unknowns
                max_gap_to_bridge_s = float(aps_cfg.get('unknown_consolidation_max_bridge_s', 1.0))


                dist_to_prev_gap = current_word_start - prev_known_speaker_end_time if prev_known_speaker_label and isinstance(prev_known_speaker_end_time, (float,int)) else float('inf')
                dist_to_next_gap = next_known_speaker_start_time - current_word_end if next_known_speaker_label and isinstance(next_known_speaker_start_time, (float,int)) else float('inf')
                
                if prev_known_speaker_label and next_known_speaker_label and \
                   prev_known_speaker_label.upper() == next_known_speaker_label.upper():
                    if dist_to_prev_gap >= -time_tolerance and dist_to_next_gap >= -time_tolerance: 
                        if current_word_start >= (prev_known_speaker_end_time - time_tolerance) and \
                           current_word_end <= (next_known_speaker_start_time + time_tolerance): 
                            assigned_new_speaker = prev_known_speaker_label
                
                elif prev_known_speaker_label and \
                         dist_to_prev_gap >= -time_tolerance and dist_to_prev_gap <= max_gap_to_bridge_s and \
                         (not next_known_speaker_label or dist_to_prev_gap <= dist_to_next_gap): 
                    assigned_new_speaker = prev_known_speaker_label
                
                elif next_known_speaker_label and \
                         dist_to_next_gap >= -time_tolerance and dist_to_next_gap <= max_gap_to_bridge_s: 
                    assigned_new_speaker = next_known_speaker_label
                
                if assigned_new_speaker.upper() != "UNKNOWN_SPEAKER" and assigned_new_speaker != original_problem_speaker:
                    seg_idx_to_update = current_word_info['_seg_idx']
                    word_idx_to_update = current_word_info['_word_idx']
                    transcript[seg_idx_to_update]['words'][word_idx_to_update]['speaker'] = assigned_new_speaker
                    consolidated_word_count +=1 # Increment counter
                    # Removed: logger.debug(f"ConsolidateUnknowns: Word '{current_word_info.get('word')}' ...")

            else: 
                logger.debug(f"ConsolidateUnknowns: Word '{current_word_info.get('word')}' (orig speaker: {original_problem_speaker}) "
                                 f"has no valid start/end time, cannot re-assign robustly.")
            
    for seg_idx, segment_data in enumerate(transcript):
        if 'words' in segment_data and isinstance(segment_data['words'], list) and segment_data['words']:
            word_speakers_original_case_in_segment = [str(w['speaker']) for w in segment_data['words']]
            known_word_speakers_original_case = [
                spk for spk in word_speakers_original_case_in_segment if str(spk).upper() in valid_known_labels
            ]
            
            if not known_word_speakers_original_case: 
                current_seg_speaker = str(segment_data.get('speaker', "UNKNOWN_SPEAKER"))
                if current_seg_speaker.upper() == "UNKNOWN_SPEAKER" or current_seg_speaker.upper() not in valid_known_labels:
                    segment_data['speaker'] = "UNKNOWN_SPEAKER"
            else:
                unique_known_speakers_in_seg_original_case = sorted(list(set(known_word_speakers_original_case))) 
                if len(unique_known_speakers_in_seg_original_case) == 1:
                    segment_data['speaker'] = unique_known_speakers_in_seg_original_case[0]
                else: 
                    speaker_counts = Counter(known_word_speakers_original_case)
                    if speaker_counts: 
                        dominant_speaker = speaker_counts.most_common(1)[0][0] 
                        segment_data['speaker'] = dominant_speaker
                    else: 
                        segment_data['speaker'] = "UNKNOWN_SPEAKER"
        elif not segment_data.get('words'): 
            current_seg_speaker = str(segment_data.get('speaker', "UNKNOWN_SPEAKER"))
            if current_seg_speaker.upper() == "UNKNOWN_SPEAKER" or current_seg_speaker.upper() not in valid_known_labels:
                segment_data['speaker'] = "UNKNOWN_SPEAKER" 

    logger.debug(f"ConsolidateUnknowns: Re-assigned {consolidated_word_count} UNKNOWN_SPEAKER words to known speakers based on proximity.") # Added summary log
    logger.info("ConsolidateUnknowns: Consolidation of UNKNOWN_SPEAKER labels complete.")
    return transcript

def process_single_audio_file(
    aac_file_path: Path,
    cfg_similarity_threshold: float,
    cfg_ambiguity_upper_bound: float,
    db_lock: threading.Lock,
    processing_date_utc: datetime,
    active_matter_context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    logger.info(f"Starting processing for audio file: {aac_file_path}")
    chunk_id = str(uuid.uuid4())
    config = get_config() # Keep for other config values
    aps_cfg = config.get('audio_suite_settings', {}) # Keep for other aps_cfg values
    paths_cfg = config.get('paths', {})
    stt_master_cfg = config.get('stt', {})

    job_output_dir: Optional[Path] = None
    processed_wav_path_str: Optional[str] = None

    # Initialize data fields that will be part of the successful/no_dialogue payload
    identified_transcript_data: List[Dict[str, Any]] = []
    speaker_turns_processed_text_data: List[Tuple[Optional[str], str]] = []
    new_speaker_enrollment_data_list: List[Dict[str, Any]] = []
    refinement_data_list: List[Dict[str, Any]] = []
    ambiguous_segments_data_list: List[Dict[str, Any]] = []
    
    no_dialogue_found = False
    # To prevent NameErrors in payload creation if the main processing block is skipped
    recognized_command = None

    # Create job directory before WAV conversion for early exit payloads
    job_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"{aac_file_path.stem}_{job_timestamp}"
    base_temp_dir_path = Path(paths_cfg.get('temp_processing_dir', PROJECT_ROOT / "data" / "temp_audio_processing_default"))
    job_output_dir = base_temp_dir_path / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Temporary job directory created: {job_output_dir}")

    # Convert to WAV for audio analysis.
    wav_file_name = f"{aac_file_path.stem}_{job_timestamp}.wav"
    wav_file_path_in_job_dir = job_output_dir / wav_file_name
    processed_wav_path_str = str(wav_file_path_in_job_dir)
    if not _convert_aac_to_wav(aac_file_path, wav_file_path_in_job_dir, target_sr=16000):
        return {
            "processing_status": "processing_error",
            "chunk_id": chunk_id,
            "error_message": f"FFMPEG conversion failed for '{aac_file_path.name}'",
            "original_file": str(aac_file_path),
            "original_file_path_for_error_handling": str(aac_file_path),
            "job_output_dir": str(job_output_dir) if job_output_dir else None
        }

    # Perform VAD check. If silent, return early.
    if not has_speech(wav_file_path_in_job_dir):
        logger.info(f"VAD detected no speech in {aac_file_path.name}. Bypassing main pipeline.")
        
        # Return the standard 'no_dialogue_detected' payload
        master_daily_transcript_path_str = str(get_master_transcript_path(config))
        return {
            "processing_status": "no_dialogue_detected",
            "chunk_id": chunk_id,
            "original_file": str(aac_file_path),
            "job_output_dir": str(job_output_dir),
            "processed_wav_path": processed_wav_path_str,
            "identified_transcript": [], "speaker_turns_processed_text": [], "new_speaker_enrollment_data": [],
            "refinement_data_for_update": [], "ambiguous_segments_flagged": [],
            "master_daily_transcript_path": master_daily_transcript_path_str,
            "recognized_command": None
        }
    
    try:
        pytorch_processing_device = aps_utils.check_device(config.get('global_ml_device', 'cpu'))

        transcript_results = _run_stt_on_wav(wav_file_path_in_job_dir)
        
        if not transcript_results: 
            logger.info(f"STT processing yielded no transcript segments (empty list) for {wav_file_path_in_job_dir.name}.")
            no_dialogue_found = True
        else:
            has_meaningful_content = False
            for segment in transcript_results:
                segment_text = segment.get("text", "").strip()
                segment_words = segment.get("words", [])
                if segment_text:
                    has_meaningful_content = True
                    break
                if isinstance(segment_words, list):
                    for word_info in segment_words:
                        if isinstance(word_info, dict) and word_info.get("word", "").strip():
                            has_meaningful_content = True
                            break
                    if has_meaningful_content:
                        break
            if not has_meaningful_content:
                logger.info(f"STT processing yielded segments, but all were empty of actual dialogue for {wav_file_path_in_job_dir.name}.")
                no_dialogue_found = True
        
        logger.info(f"STT complete: {len(transcript_results) if transcript_results else 0} segments for '{aac_file_path.name}'.")
        
        # If no_dialogue_found is true here, subsequent steps will largely operate on empty lists or skip.
        # The data variables (identified_transcript_data, etc.) will retain their initial empty values.
        if not no_dialogue_found:
            diarization_results = _run_diarization_on_wav(wav_file_path_in_job_dir)
            if diarization_results is not None: 
                logger.info(f"External diarization (Pyannote, consolidated) complete: {len(diarization_results)} turns for '{aac_file_path.name}'.")

            combined_transcript = _combine_asr_diarization(transcript_results, diarization_results) # Use original transcript_results
            
            known_diar_labels_for_consolidation: List[str] = []
            if diarization_results:
                known_diar_labels_for_consolidation = sorted(list(set(
                    str(dr['speaker']) for dr in diarization_results 
                    if dr.get('speaker') and str(dr.get('speaker')).strip().upper() != "UNKNOWN_SPEAKER" 
                )))
            
            if known_diar_labels_for_consolidation:
                needs_consolidation = False
                for seg in combined_transcript:
                    if str(seg.get('speaker')).upper() == "UNKNOWN_SPEAKER":
                        needs_consolidation = True
                        break
                    if not needs_consolidation and 'words' in seg and isinstance(seg['words'], list):
                        for word_info in seg['words']:
                            if str(word_info.get('speaker')).upper() == "UNKNOWN_SPEAKER":
                                needs_consolidation = True
                                break
                        if needs_consolidation: break
                
                if needs_consolidation:
                    logger.info(f"Attempting to consolidate UNKNOWN_SPEAKER labels in combined transcript using known Pyannote labels: {known_diar_labels_for_consolidation}")
                    combined_transcript = _post_process_consolidate_unknowns(combined_transcript, known_diar_labels_for_consolidation)
                else:
                    logger.info("No UNKNOWN_SPEAKER labels found needing consolidation in combined transcript, or no valid known Pyannote labels for consolidation step.")
            else:
                logger.info("No known speaker labels from Pyannote available for UNKNOWN consolidation in combined transcript, or Pyannote identified no speakers (or only UNKNOWN).")

            # Default to combined_transcript if speaker ID is skipped or fails to return a transcript
            identified_transcript_data = combined_transcript 
            
            embedding_model = _loaded_models_cache.get("embedding_model") # This is correct, model is global
            embedding_dim = _loaded_models_cache.get("embedding_dim")
            

            #Initialize this list here so it always exists
            newly_created_cusids_for_review: List[Dict[str, Any]] = []

            if embedding_model and embedding_dim:
                speaker_db_dir_path_str = paths_cfg.get('speaker_db_dir')
                if not speaker_db_dir_path_str :
                    raise ValueError("Config 'paths.speaker_db_dir' not defined or empty.") # This will become processing_error
                speaker_db_dir_path = Path(speaker_db_dir_path_str)
            
                faiss_index_filename = aps_cfg.get('faiss_index_filename', "global_speaker_embeddings.index")
                speaker_map_filename = aps_cfg.get('speaker_map_filename', "global_speaker_names.json")
                
                faiss_index_path = speaker_db_dir_path / faiss_index_filename
                speaker_map_path = speaker_db_dir_path / speaker_map_filename
                
                logger.info(f"Running Speaker Identification for '{aac_file_path.name}'. Speaker DB: {speaker_db_dir_path}")

                
                logger.info("--- Entering aps_speaker_id.run_speaker_identification call ---")
                audio_context = "in_person"
                logger.info(f"Using placeholder audio context for speaker identification: '{audio_context}'")
                # <<< MODIFICATION END >>>
                
                speaker_id_config_args = {
                    "db_lock": db_lock, # Pass the lock down
                    "combined_transcript": combined_transcript,
                    "embedding_model": embedding_model,
                    "emb_dim": embedding_dim,
                    "audio_path": wav_file_path_in_job_dir,
                    "output_dir": job_output_dir,
                    "processing_device": pytorch_processing_device,
                    "min_segment_duration": float(aps_cfg.get('min_segment_duration_for_embedding', 2.0)),
                    "similarity_threshold": cfg_similarity_threshold, # USE PASSED VALUE
                    "faiss_index_path": faiss_index_path,
                    "speaker_map_path": speaker_map_path,
                    "context": audio_context,
                    "active_matter_context": active_matter_context,
                    "processing_date_utc": processing_date_utc, 
                    "live_refinement_min_similarity": float(aps_cfg.get('live_refinement_min_similarity', 0.85)),
                    "live_refinement_min_segment_duration_s": float(aps_cfg.get('live_refinement_min_segment_duration_s', 3.0)),
                    "ambiguity_similarity_lower_bound": float(aps_cfg.get('ambiguity_similarity_lower_bound', 0.65)),
                    "ambiguity_similarity_upper_bound_for_review": cfg_ambiguity_upper_bound, # USE PASSED VALUE
                    "ambiguity_max_similarity_delta_for_multiple_matches": float(aps_cfg.get('ambiguity_max_similarity_delta_for_multiple_matches', 0.05)),
                }
                speaker_id_results = aps_speaker_id.run_speaker_identification(**speaker_id_config_args)

                # <<< MODIFICATION START >>>
                logger.info("--- Exited aps_speaker_id.run_speaker_identification call ---")
                # <<< MODIFICATION END >>>

                identified_transcript_data = speaker_id_results.get("identified_transcript", combined_transcript)
                raw_new_speaker_candidates = speaker_id_results.get("new_speaker_enrollment_data", []) # Renamed
                refinement_data_list = speaker_id_results.get("refinement_data_for_update", [])
                ambiguous_segments_data_list = speaker_id_results.get("ambiguous_segments_flagged", [])

                # This ensures they don't cause schema errors when mixed with other flag types.
                for ambiguity_flag in ambiguous_segments_data_list:
                    if isinstance(ambiguity_flag, dict) and 'flag_type' not in ambiguity_flag:
                        ambiguity_flag['flag_type'] = 'speaker_ambiguity'
                
                logger.info(f"Speaker ID complete. Identified transcript has {len(identified_transcript_data)} segments. "
                            f"{len(raw_new_speaker_candidates)} raw new speaker candidates. "
                            f"{len(ambiguous_segments_data_list)} ambiguous segments flagged. "
                            f"{len(refinement_data_list)} refinement candidates for '{aac_file_path.name}'.")

                # <<< Two-pass CUSID Consolidation >>>
                # Initialize the variable here so it always exists, even if the 'if' block is skipped.
                newly_created_cusids_for_review: List[Dict[str, Any]] = []
                
                if raw_new_speaker_candidates:
                    logger.info(f"Processing {len(raw_new_speaker_candidates)} raw new speaker candidates for CUSID consolidation.")
                    consolidation_threshold = float(aps_cfg.get('consolidate_new_speaker_threshold', 0.88))

                    # --- Pass 1: Intra-file consolidation ---
                    unconsolidated_speakers = raw_new_speaker_candidates.copy()
                    intra_file_consolidated_groups = []
                    while unconsolidated_speakers:
                        pivot_speaker = unconsolidated_speakers.pop(0)
                        new_group = [pivot_speaker]
                        pivot_embedding = np.array(pivot_speaker['embedding'])
                        remaining_speakers_after_grouping = []
                        for other_speaker in unconsolidated_speakers:
                            other_embedding = np.array(other_speaker['embedding'])
                            similarity = _calculate_similarity(pivot_embedding, other_embedding)
                            if similarity > consolidation_threshold:
                                new_group.append(other_speaker)
                            else:
                                remaining_speakers_after_grouping.append(other_speaker)
                        intra_file_consolidated_groups.append(new_group)
                        unconsolidated_speakers = remaining_speakers_after_grouping
                    
                    logger.info(f"Intra-file consolidation: {len(raw_new_speaker_candidates)} candidates grouped into {len(intra_file_consolidated_groups)} unique new speakers for this file.")

                    unique_new_speakers_this_file = []
                    for group in intra_file_consolidated_groups:
                        group_embeddings = [np.array(spk['embedding']) for spk in group]
                        avg_embedding = np.mean(group_embeddings, axis=0)
                        norm_avg = np.linalg.norm(avg_embedding)
                        if norm_avg > 1e-6: avg_embedding /= norm_avg
                        
                        representative_speaker = group[0].copy()
                        representative_speaker['embedding'] = avg_embedding
                        representative_speaker['merged_original_labels'] = [spk['original_label'] for spk in group]
                        unique_new_speakers_this_file.append(representative_speaker)
                    
                    # --- Pass 2: Consolidate with existing CUSIDs ---
                    target_date_for_log_dt = processing_date_utc # Use the passed-in date
                    
                    target_date_for_flags = target_date_for_log_dt.date()
                    flags_queue_path = _get_daily_flags_queue_path(target_date_for_flags, config)
                    all_flags_today = _load_daily_flags_queue(flags_queue_path)
                    
                    active_cusid_registry_from_file = []
                    other_flags_today = []
                    for flag_item in all_flags_today:
                        if flag_item.get('cusid') and flag_item.get('flag_type') == 'pending_review':
                            if 'consolidated_embedding' in flag_item and isinstance(flag_item['consolidated_embedding'], list):
                                flag_item['consolidated_embedding'] = np.array(flag_item['consolidated_embedding'])
                            active_cusid_registry_from_file.append(flag_item)
                        else:
                            other_flags_today.append(flag_item)
                    
                    logger.info(f"Loaded {len(active_cusid_registry_from_file)} active CUSIDs for consolidation from {flags_queue_path}.")

                    speaker_label_to_cusid_map: Dict[str, str] = {}
                    
                    
                    # This is the CUSID registry that will be modified and saved.
                    final_cusid_registry = active_cusid_registry_from_file.copy()

                    for unique_speaker in unique_new_speakers_this_file:
                        candidate_embedding = unique_speaker['embedding']
                        
                        best_match_cusid_entry = None
                        max_similarity = -1.0
                        for cusid_entry in final_cusid_registry:
                            if not isinstance(cusid_entry.get('consolidated_embedding'), np.ndarray): continue
                            similarity = _calculate_similarity(candidate_embedding, cusid_entry['consolidated_embedding'])
                            if similarity > consolidation_threshold and similarity > max_similarity:
                                max_similarity = similarity
                                best_match_cusid_entry = cusid_entry

                        approx_segment_utc = datetime.combine(target_date_for_flags, datetime.min.time(), tzinfo=timezone.utc) + timedelta(seconds=unique_speaker.get('start_time', 0.0))

                        if best_match_cusid_entry:
                            merged_labels_str = ", ".join(unique_speaker['merged_original_labels'])
                            logger.info(f"Merging new speaker group ({merged_labels_str}) into existing CUSID '{best_match_cusid_entry['cusid']}' (sim: {max_similarity:.3f}).")
                            for label in unique_speaker['merged_original_labels']:
                                speaker_label_to_cusid_map[label] = best_match_cusid_entry['cusid']
                            
                            best_match_cusid_entry['segment_occurrences'].append({
                                'audio_file_stem': aac_file_path.stem,
                                'original_diar_label': merged_labels_str,
                                'relative_start_s': unique_speaker.get('start_time'), 'relative_end_s': unique_speaker.get('end_time'),
                                'embedding': candidate_embedding.tolist(), 'timestamp_segment_utc': approx_segment_utc
                            })
                            all_occurrence_embeddings = [np.array(occ['embedding']) for occ in best_match_cusid_entry['segment_occurrences']]
                            new_average_embedding = np.mean(all_occurrence_embeddings, axis=0)
                            norm_avg = np.linalg.norm(new_average_embedding); 
                            if norm_avg > 1e-6: new_average_embedding /= norm_avg
                            best_match_cusid_entry['consolidated_embedding'] = new_average_embedding
                            best_match_cusid_entry['timestamp_last_updated_utc'] = datetime.now(timezone.utc)
                        else:
                            merged_labels_str = ", ".join(unique_speaker['merged_original_labels'])
                            logger.info(f"No existing CUSID match for new speaker group ({merged_labels_str}). Creating new CUSID.")
                            
                            cusid_timestamp_part = int(time.time() * 1000)
                            existing_cusid_ids_today = {c.get('cusid', "") for c in final_cusid_registry}
                            cusid_unique_suffix = 0
                            new_cusid_id = f"CUSID_{target_date_for_flags.strftime('%Y%m%d')}_{cusid_timestamp_part % 100000}_{cusid_unique_suffix}"
                            while new_cusid_id in existing_cusid_ids_today:
                                cusid_unique_suffix += 1
                                new_cusid_id = f"CUSID_{target_date_for_flags.strftime('%Y%m%d')}_{cusid_timestamp_part % 100000}_{cusid_unique_suffix}"
                            
                            new_cusid_entry = {
                                'cusid': new_cusid_id, 'flag_type': 'pending_review',
                                'timestamp_first_detected_utc': datetime.now(timezone.utc), 'timestamp_last_updated_utc': datetime.now(timezone.utc),
                                'consolidated_embedding': candidate_embedding,
                                'segment_occurrences': [{'audio_file_stem': aac_file_path.stem, 'original_diar_label': merged_labels_str,
                                                         'relative_start_s': unique_speaker.get('start_time'), 'relative_end_s': unique_speaker.get('end_time'),
                                                         'embedding': candidate_embedding.tolist(), 'timestamp_segment_utc': approx_segment_utc}],
                                'assigned_name': None, 'notes': "",
                                'first_occurrence_snippet_path': unique_speaker.get('snippet_path_abs'),
                                'first_occurrence_audio_file_stem': aac_file_path.stem,
                            }
                            final_cusid_registry.append(new_cusid_entry)
                            newly_created_cusids_for_review.append(new_cusid_entry)
                            for label in unique_speaker['merged_original_labels']:
                                speaker_label_to_cusid_map[label] = new_cusid_id
                            logger.info(f"Created new CUSID '{new_cusid_id}' for group ({merged_labels_str}).")
                    
                    if speaker_label_to_cusid_map:
                        logger.info(f"Updating transcript with CUSID mappings: {speaker_label_to_cusid_map}")
                        for segment in identified_transcript_data:
                            original_segment_speaker = str(segment.get("speaker", ""))
                            if original_segment_speaker in speaker_label_to_cusid_map:
                                segment["speaker"] = speaker_label_to_cusid_map[original_segment_speaker]
                            if 'words' in segment and isinstance(segment['words'], list):
                                for word_info in segment['words']:
                                    original_word_speaker = str(word_info.get("speaker", ""))
                                    if original_word_speaker in speaker_label_to_cusid_map:
                                        word_info["speaker"] = speaker_label_to_cusid_map[original_word_speaker]
                        logger.info("Transcript updated with CUSIDs where applicable.")
                    
                    combined_flags_to_save = final_cusid_registry + other_flags_today
                    _save_daily_flags_queue(flags_queue_path, combined_flags_to_save)
                    logger.info(f"Saved updated daily flags queue to {flags_queue_path} with {len(combined_flags_to_save)} items.")
                else: # No raw_new_speaker_candidates
                    logger.info("No raw new speaker candidates to process for CUSID consolidation.")
                # <<< MODIFICATION END >>>

            else: # Embedding model not available
                logger.warning(f"Embedding model/dim not available. Skipping speaker identification and CUSID consolidation for '{aac_file_path.name}'.")
            

            final_new_speaker_data_for_return: List[Dict[str, Any]] = []
            if newly_created_cusids_for_review:
                

                for cusid_review_item in newly_created_cusids_for_review:
                    cusid_id = cusid_review_item['cusid']

                    # Create the base structure for return item FIRST
                    current_return_item = {
                        'flag_type': 'pending_review',
                        'cusid': cusid_id,
                        'temp_id': cusid_id,
                        'original_label': cusid_id, # For consistency with CUSID concept
                        'snippet_path_abs': cusid_review_item.get('first_occurrence_snippet_path'),
                        'start_time': cusid_review_item['segment_occurrences'][0].get('relative_start_s') if cusid_review_item.get('segment_occurrences') else None,
                        'end_time': cusid_review_item['segment_occurrences'][0].get('relative_end_s') if cusid_review_item.get('segment_occurrences') else None,
                        
                    }

                    # Populate and verify embedding BEFORE LLM call for this item
                    consolidated_embedding_data = cusid_review_item.get('consolidated_embedding')
                    if isinstance(consolidated_embedding_data, np.ndarray):
                        current_return_item['embedding'] = consolidated_embedding_data.tolist()
                    elif isinstance(consolidated_embedding_data, list): # If it was already converted (e.g. loaded from JSON and not re-converted to np.array)
                        current_return_item['embedding'] = consolidated_embedding_data
                    
                    if not current_return_item.get('embedding'):
                        logger.warning(f"CUSID {cusid_id} has empty or missing consolidated_embedding. This may lead to issues in flag enrollment if embedding is critical there.")
                    
                    final_new_speaker_data_for_return.append(current_return_item)
                
                
                new_speaker_enrollment_data_list = final_new_speaker_data_for_return 
            else: # No newly_created_cusids_for_review
                
                new_speaker_enrollment_data_list = [] # Ensure it's empty for the return payload
                logger.info(f"No new CUSIDs created in this run for '{aac_file_path.name}'.")

            punctuation_model = _loaded_models_cache.get("punctuation")
            logger.info(f"Reconstructing text from words and applying ML punctuation (if model enabled) for '{aac_file_path.name}'...")
            reconstructed_or_punctuated_turns = aps_text_processing.apply_punctuation(
                identified_transcript_data, # Use data var 
                punctuation_model, 
                int(aps_cfg.get('punctuation_chunk_size', 250))
            )
            # The normalization step is now bypassed. The final text data is the direct output from punctuation.
            speaker_turns_processed_text_data = reconstructed_or_punctuated_turns
            
            # >>> NEW STEP: INTERPRET VOICE COMMANDS FROM FINAL TEXT <<<
            recognized_command = None
            vc_config = config.get('voice_commands', {})
            if vc_config.get('enabled', False) and speaker_turns_processed_text_data:
                user_speaker_name = vc_config.get('user_speaker_name')
                if user_speaker_name:
                    # Format the text exactly as it appears in the final log file
                    # speaker_turns_processed_text_data is a list of (speaker, text) tuples
                    final_transcript_text = "\n\n".join(
                        f"{speaker}: {text}" for speaker, text in speaker_turns_processed_text_data
                    )
                    
                    recognized_command = process_voice_command(
                        full_transcript_chunk_text=final_transcript_text,
                        user_speaker_name=user_speaker_name,
                        config=config
                    )

            # Persistence calls using data vars
            aps_persistence.save_final_transcript(speaker_turns_processed_text_data, job_output_dir / f"FINAL_TRANSCRIPT_{job_timestamp}.txt")
            aps_persistence.save_transcript_srt(identified_transcript_data, job_output_dir / f"FINAL_TRANSCRIPT_{job_timestamp}.srt")
            # ... other persistence calls ...
            timestamp_interval_seconds_cfg = aps_cfg.get('timestamped_transcript_interval_seconds', 60) 
            try: timestamp_interval_seconds_for_job_file = int(timestamp_interval_seconds_cfg)
            except ValueError: timestamp_interval_seconds_for_job_file = 60
            aps_persistence.save_transcript_with_timestamps(
                identified_transcript_data, 
                job_output_dir / f"FINAL_TRANSCRIPT_TIMESTAMPS_{job_timestamp}.txt",
                timestamp_interval_seconds_for_job_file 
            )
            aps_persistence.save_segment_level_json(identified_transcript_data, job_output_dir / f"FINAL_SEGMENTS_{job_timestamp}.json")
            aps_persistence.save_word_level_transcript(identified_transcript_data, job_output_dir / f"DEBUG_WORD_LEVEL_FINAL_{job_timestamp}.json")

            # Final check: if all processing happened but resulted in empty dialogue
            final_dialogue_present = False
            if identified_transcript_data:
                for segment in identified_transcript_data:
                    if isinstance(segment, dict) and (segment.get("text","").strip() or \
                           (isinstance(segment.get("words"),list) and any(w.get("word","").strip() for w in segment.get("words",[]) if isinstance(w,dict)))):
                        final_dialogue_present = True; break
            if not final_dialogue_present and speaker_turns_processed_text_data:
                    for _, text_block in speaker_turns_processed_text_data:
                            if text_block.strip(): final_dialogue_present = True; break
            
            if not final_dialogue_present:
                logger.info(f"Pipeline completed, but final processed transcript/dialogue is empty for {aac_file_path.name}.")
                no_dialogue_found = True # Update flag based on final state
                # Reset data to ensure "no_dialogue_detected" payload is accurate
                identified_transcript_data = []
                speaker_turns_processed_text_data = []
                # Consider if new_speaker_enrollment_data_list etc. should also be cleared if dialogue disappears post-processing.
                # For now, they'll retain values if generated before dialogue vanished. This might need refinement.
                # Let's clear them to be consistent with "no dialogue".
                new_speaker_enrollment_data_list = []
                refinement_data_list = []
                ambiguous_segments_data_list = []


        # Construct final payload based on the final state of no_dialogue_found
        master_daily_transcript_path_str = str(get_master_transcript_path(config))

        # --- Calculate start time ---
        day_start_time_for_chunk = get_day_start_time(processing_date_utc)
        seq_num = extract_sequence_number_from_filename(aac_file_path.stem)
        chunk_start_time_utc_iso = None
        if day_start_time_for_chunk and seq_num is not None:
            chunk_duration_s_cfg = config.get('timings', {}).get('audio_chunk_expected_duration', "10m")
            chunk_duration_s = parse_duration_to_minutes(chunk_duration_s_cfg) * 60.0
            chunk_start_dt = day_start_time_for_chunk + timedelta(seconds=(seq_num - 1) * chunk_duration_s)
            chunk_start_time_utc_iso = chunk_start_dt.isoformat()
        else:
            logger.warning(f"Could not determine chunk start time for {aac_file_path.name}. Day start or sequence number missing. This may affect flagging.")
            # Provide a fallback for safety
            chunk_start_time_utc_iso = datetime.now(timezone.utc).isoformat()

        success_payload = {}

        if no_dialogue_found:
            logger.info(f"Finalizing with 'no_dialogue_detected' for {aac_file_path.name}")
            success_payload = {
                "processing_status": "no_dialogue_detected",
                "chunk_id": chunk_id,
                "original_file": str(aac_file_path),
                "job_output_dir": str(job_output_dir) if job_output_dir else None,
                "processed_wav_path": processed_wav_path_str,
                "identified_transcript": identified_transcript_data, # Should be empty
                "speaker_turns_processed_text": speaker_turns_processed_text_data, # Should be empty
                "new_speaker_enrollment_data": new_speaker_enrollment_data_list, # Should be empty
                "refinement_data_for_update": refinement_data_list, # Should be empty
                "ambiguous_segments_flagged": ambiguous_segments_data_list, # Should be empty
                "master_daily_transcript_path": master_daily_transcript_path_str,
                "recognized_command": recognized_command
            }
        else:
            logger.info(f"Successfully processed audio file: {aac_file_path.name}")
            success_payload = {
                "processing_status": "success",
                "chunk_id": chunk_id,
                "original_file": str(aac_file_path),
                "job_output_dir": str(job_output_dir) if job_output_dir else None,
                "processed_wav_path": processed_wav_path_str,
                "identified_transcript": identified_transcript_data,
                "speaker_turns_processed_text": speaker_turns_processed_text_data,
                "new_speaker_enrollment_data": new_speaker_enrollment_data_list,
                "refinement_data_for_update": refinement_data_list,
                "ambiguous_segments_flagged": ambiguous_segments_data_list,
                "master_daily_transcript_path": master_daily_transcript_path_str,
                "recognized_command": recognized_command,
                "source_file_name": aac_file_path.name,
                "chunk_start_time_utc": chunk_start_time_utc_iso,
                
            }

        logger.info(f"--- Post-processing cleanup for {aac_file_path.name} ---")
        # Explicitly delete large intermediate variables to hint at garbage collection
        try:
            del identified_transcript_data
            del speaker_turns_processed_text_data
            del new_speaker_enrollment_data_list
            del refinement_data_list
            del ambiguous_segments_data_list
            del transcript_results
            if 'diarization_results' in locals(): del diarization_results
            if 'combined_transcript' in locals(): del combined_transcript
            if 'full_audio_waveform_tensor' in locals(): del full_audio_waveform_tensor
            if 'speaker_audio_segments_map' in locals(): del speaker_audio_segments_map
            if 'speaker_embeddings_data_map' in locals(): del speaker_embeddings_data_map
            logger.info("Successfully deleted intermediate variables.")
        except NameError as ne:
            # This is expected if a variable wasn't created (e.g., no dialogue)
            logger.debug(f"Cleanup note: Variable not defined, skipping deletion: {ne}")
        except Exception as e_cleanup:
            logger.warning(f"Error during explicit variable cleanup: {e_cleanup}", exc_info=True)
        
        # Trigger garbage collection and empty CUDA/MPS cache
        aps_utils.cleanup_resources(use_cuda=(pytorch_processing_device == 'cuda'))
        logger.info(f"--- Finished post-processing cleanup for {aac_file_path.name} ---")

        return success_payload

    except Exception as e:
        logger.error(f"PIPELINE FAILED for '{aac_file_path.name}': {e}", exc_info=True)
        
        if job_output_dir and isinstance(job_output_dir, Path) and job_output_dir.exists():
            # Per instructions, do not move to error folder here, just log.
            logger.info(f"Temp job directory {job_output_dir} may be kept for review after error.")
        
        return {
            "processing_status": "processing_error",
            "chunk_id": chunk_id,
            "error_message": f"PIPELINE FAILED for '{aac_file_path.name}': {str(e)}",
            "original_file": str(aac_file_path),
            "original_file_path_for_error_handling": str(aac_file_path),
            "job_output_dir": str(job_output_dir) if job_output_dir else None
        }