# Samson/src/audio_processing_suite/utils.py
import os
import torch
import gc
from typing import Optional, Any
from pathlib import Path

from src.logger_setup import logger
from src.config_loader import get_config

def check_device(requested_device: str) -> str:
    """Checks GPU/MPS availability and returns the validated device string."""
    logger.debug(f"Requested device: '{requested_device}'")
    if requested_device == "cuda":
        if torch.cuda.is_available():
            logger.info("CUDA is available. Using GPU (cuda).")
            return "cuda"
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
    elif requested_device == "mps":
        # Check for MPS availability
        # Based on https://pytorch.org/docs/stable/notes/mps.html
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Additionally, check if PyTorch was built with MPS support.
            # is_built() is crucial for determining if MPS can actually be used.
            if torch.backends.mps.is_built():
                logger.info("MPS is available and built with this PyTorch. Using Apple Metal Performance Shaders (mps).")
                return "mps"
            else:
                logger.warning("MPS is available but not built with this PyTorch version. Falling back to CPU.")
                return "cpu"
        else:
            logger.warning("MPS requested but not available (torch.backends.mps.is_available() is False). Falling back to CPU.")
            return "cpu"
    elif requested_device == "cpu":
        logger.info("Using CPU.")
        return "cpu"
    else:
        logger.warning(f"Requested device '{requested_device}' is not 'cuda', 'mps', or 'cpu'. Defaulting to CPU.")
        return "cpu"

def cleanup_resources(*args: Any, use_cuda: bool = False) -> None:
    """Attempts to delete variables and clear CUDA cache."""
    logger.info("--- Running cleanup_resources ---")
    deleted_count = 0
    for i, arg in enumerate(args):
        if arg is not None:
            # This doesn't actually delete the object in the caller's scope,
            # but we log it to show the intent and trigger GC.
            logger.debug(f"cleanup_resources: Hinting for cleanup of object at arg index {i} (type: {type(arg)}).")
            deleted_count += 1
    logger.info(f"cleanup_resources: Hinted at {deleted_count} objects for cleanup.")

    # The *args mechanism for deleting variables from the caller's scope isn't
    # directly effective in Python. The primary benefit here is triggering GC
    # and CUDA cache clearing if applicable.
    # Variables passed as args are local copies of references.

    # Trigger garbage collection for Python objects
    gc.collect()
    logger.debug("Garbage collection triggered.")

    # Clear CUDA cache if use_cuda is True and CUDA is available
    # Note: MPS does not have an equivalent explicit cache clearing mechanism like CUDA.
    if use_cuda and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache.")
        except Exception as e:
            logger.warning(f"Failed to clear CUDA cache: {e}", exc_info=True)
            
    logger.info("--- Finished cleanup_resources ---")

def load_huggingface_token() -> Optional[str]:
    """
    Loads the Hugging Face token from Samson's central configuration.
    Searches in `huggingface.hf_token_auth` (or `huggingface.hf_token`),
    then falls back to `audio_suite_settings.hf_token` (or `audio_suite_settings.hf_token_auth`).
    The token value can be the token itself or an environment variable reference like "ENV:MY_HF_TOKEN_VAR".
    """
    samson_config = get_config()
    hf_token_val: Optional[str] = None

    # 1. Try general 'huggingface' section
    hf_general_config = samson_config.get('huggingface', {})
    if hf_general_config: # Ensure the section exists
        hf_token_val = hf_general_config.get('hf_token_auth', hf_general_config.get('hf_token'))

    # 2. If not found, try 'audio_suite_settings' section (legacy or specific override)
    if not hf_token_val:
        hf_audio_suite_config = samson_config.get('audio_suite_settings', {})
        if hf_audio_suite_config: # Ensure the section exists
            hf_token_val = hf_audio_suite_config.get('hf_token_auth', hf_audio_suite_config.get('hf_token'))
    
    if not hf_token_val:
        logger.info("No Hugging Face token key (hf_token_auth/hf_token) found in 'huggingface' or 'audio_suite_settings' sections of Samson config.")
        return None

    logger.debug(f"Found potential Hugging Face token value in config: '{str(hf_token_val)[:10]}...'")

    # Check if the value is intended to be an environment variable name
    if isinstance(hf_token_val, str) and hf_token_val.startswith("ENV:"):
        env_var_name = hf_token_val[4:]
        env_token = os.environ.get(env_var_name)
        if env_token:
            logger.info(f"Hugging Face token loaded from environment variable '{env_var_name}' (specified in Samson config).")
            return env_token
        else:
            logger.warning(f"Hugging Face token specified as ENV:'{env_var_name}' in Samson config, but environment variable not found.")
            return None
    # Heuristic for environment variable name (all caps with underscore, not starting with hf_)
    elif isinstance(hf_token_val, str) and hf_token_val.isupper() and '_' in hf_token_val and not hf_token_val.startswith("hf_"):
        env_token = os.environ.get(hf_token_val)
        if env_token:
            logger.info(f"Hugging Face token loaded from environment variable '{hf_token_val}' (inferred from Samson config value).")
            return env_token
        else:
            # If it looked like an env var name but wasn't found, and it starts with "hf_",
            # it might actually be the token itself.
            if hf_token_val.startswith("hf_"):
                 logger.info("Hugging Face token value from Samson config resembles an actual token and was not found as an env var. Using directly.")
                 return hf_token_val
            logger.warning(f"Hugging Face token value '{hf_token_val}' from Samson config (looked like env var name) not found in environment, and doesn't look like a token itself.")
            return None
    # Actual token (typically starts with "hf_")
    elif isinstance(hf_token_val, str) and hf_token_val.startswith("hf_"):
        logger.info("Hugging Face token loaded directly from Samson config.")
        return hf_token_val
    else:
        # If it doesn't match common patterns for env var or actual token.
        logger.warning(f"Hugging Face token value from Samson config ('{str(hf_token_val)[:10]}...') does not match expected formats (actual token starting with 'hf_', 'ENV:VAR_NAME', or inferred VAR_NAME). Treating as no token.")
        return None

# --- Option B (load_huggingface_token_from_file) removed as Option A is preferred ---