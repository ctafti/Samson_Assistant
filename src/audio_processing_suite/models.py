# Samson/src/audio_processing_suite/models.py
from pathlib import Path
import torch
# import whisper # You plan to remove this
from pyannote.audio import Pipeline
from speechbrain.inference.classifiers import EncoderClassifier
from transformers import pipeline as hf_pipeline
from typing import Optional, Any, Tuple

from src.logger_setup import logger
from src.config_loader import get_config

# Remove or comment out ParakeetSTT import if no longer trying Python API for it
# try:
#     from mlx_audio import SpeechToText as ParakeetSTT
#     logger.info("Successfully imported 'SpeechToText' from 'mlx_audio' as ParakeetSTT.")
# except ImportError as e:
#     logger.error(f"Could not import 'SpeechToText' from 'mlx_audio': {e}. ")
#     ParakeetSTT = None # type: ignore


# --- OpenAI Whisper Model Loader (you mentioned you'll remove this later) ---
def load_whisper_model(model_name: str, device: str, download_root: Optional[Path] = None) -> Optional[Any]:
    logger.info(f"Loading OpenAI Whisper model: {model_name} (Device: {device})")
    whisper_download_root_str: Optional[str] = None
    if download_root:
        logger.info(f"Using download root for Whisper model: {download_root}")
        whisper_download_root_str = str(download_root)
    else:
        default_cache = Path.home() / ".cache" / "whisper"
        logger.info(f"Using default download root for Whisper model: {default_cache}")
    try:
        # Ensure whisper is imported if this function is kept
        import whisper
        model = whisper.load_model(
            model_name,
            device=torch.device(device),
            download_root=whisper_download_root_str
        )
        if hasattr(model, 'device'):
            logger.info(f"OpenAI Whisper model '{model_name}' loaded. Effective device: {model.device}")
        else:
            logger.info(f"OpenAI Whisper model '{model_name}' loaded. Device attribute not found on model object.")
        return model
    except ImportError:
        logger.critical("The 'whisper' library is not installed. Cannot load OpenAI Whisper model.")
        return None
    except NotImplementedError as nie:
        logger.critical(f"CRITICAL: NotImplementedError loading OpenAI Whisper model '{model_name}' on device '{device}': {nie}. "
                        "This often happens with MPS. Try setting 'stt.openai_whisper.device' to 'cpu' in config.", exc_info=True)
        return None
    except Exception as e:
        logger.critical(f"CRITICAL: Error loading OpenAI Whisper model '{model_name}': {e}. Cannot proceed.", exc_info=True)
        return None

# --- Parakeet MLX Model Loader (MODIFIED for CLI indication) ---
def load_parakeet_mlx_model(model_identifier: str) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Indicates that Parakeet MLX will be run via CLI.
    Returns a placeholder tuple ("PARAKEET_CLI_MODE", None).
    """
    logger.info(f"Parakeet MLX STT model '{model_identifier}' will be processed via CLI. No in-memory model loaded by this function.")
    # Return a specific string or boolean to indicate CLI mode, and None for the tokenizer part.
    # This allows initialize_audio_models to "succeed" without actually loading a Python object.
    return "PARAKEET_CLI_MODE", None

# --- Other model loaders (diarization, embedding, punctuation) - (unchanged) ---
def load_diarization_pipeline(model_name: str, device: str, hf_token: Optional[str]) -> Optional[Pipeline]:
    if not hf_token:
        samson_config = get_config()
        hf_cfg1 = samson_config.get('huggingface', {})
        hf_cfg2 = samson_config.get('audio_suite_settings', {})
        global_hf_token = hf_cfg1.get('hf_token_auth', hf_cfg1.get('hf_token', hf_cfg2.get('hf_token_auth', hf_cfg2.get('hf_token'))))
        if global_hf_token:
            hf_token = global_hf_token
            logger.info("Using Hugging Face token from Samson global configuration.")
        else:
            logger.warning("No Hugging Face token explicitly provided or found in global config. Pyannote may fail.")
    logger.info(f"Loading Pyannote diarization pipeline: {model_name} (Device: {device})")
    try:
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
        pipeline.to(torch.device(device))
        logger.info(f"Pyannote pipeline loaded successfully to {pipeline.device}.")
        return pipeline
    except Exception as e:
        logger.error(f"Error loading Pyannote pipeline '{model_name}': {e}. Diarization will be skipped.", exc_info=True)
        return None

def load_embedding_model(model_name: str, device: str, cache_dir: Path) -> Tuple[Optional[EncoderClassifier], int]:
    samson_config = get_config()
    aps_config = samson_config.get('audio_suite_settings', {})
    default_emb_dim = aps_config.get('embedding_dim', 192)
    logger.info(f"Loading SpeechBrain embedding model: {model_name} (Device: {device})")
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using cache directory for embedding model: {cache_dir}")
    try:
        model = EncoderClassifier.from_hparams(source=model_name, run_opts={"device": device}, savedir=str(cache_dir))
        model.eval()
        logger.info("SpeechBrain embedding model loaded successfully.")
        emb_dim = default_emb_dim
        try:
            dummy_input = torch.rand(1, 16000).to(torch.device(device))
            with torch.no_grad(): output = model.encode_batch(dummy_input)
            if output.ndim == 3: emb_dim = output.shape[-1]
            elif output.ndim == 2: emb_dim = output.shape[-1]
            else: logger.warning(f"Unexpected output shape from embedding model: {output.shape}. Using default dim {default_emb_dim}.")
            if emb_dim != default_emb_dim: logger.info(f"Dynamically determined embedding dimension: {emb_dim}")
            else: logger.info(f"Using pre-configured/default embedding dimension: {emb_dim}")
        except Exception as emb_e:
            logger.warning(f"Could not dynamically determine embedding dimension for '{model_name}': {emb_e}. Using default {emb_dim}.", exc_info=False)
        return model, emb_dim
    except Exception as e:
        logger.error(f"Error loading SpeechBrain model '{model_name}': {e}. Speaker ID will be skipped.", exc_info=True)
        return None, default_emb_dim

def load_punctuation_model(model_name: str, device: str) -> Optional[Any]:
    logger.info(f"Loading Punctuation model: {model_name} (Device: {device})")
    try:
        effective_device_for_pipeline: Any = -1
        if device == "cuda":
            if torch.cuda.is_available(): effective_device_for_pipeline = 0
            else: logger.warning("Punctuation model requested CUDA but not available, using CPU.")
        elif device == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                effective_device_for_pipeline = "mps"
                logger.info("Attempting to use MPS for punctuation model pipeline.")
            else: logger.warning("Punctuation model requested MPS but not available/built, using CPU.")
        elif device != "cpu": logger.warning(f"Unsupported device '{device}' for punctuation model, defaulting to CPU.")

        punc_pipeline = hf_pipeline("token-classification", model=model_name, device=effective_device_for_pipeline, aggregation_strategy="simple")
        actual_device = punc_pipeline.device
        logger.info(f"Punctuation model '{model_name}' loaded successfully to device: {actual_device}.")
        return punc_pipeline
    except Exception as e:
        logger.error(f"Error loading punctuation model '{model_name}': {e}. Punctuation step will be skipped.", exc_info=True)
        return None