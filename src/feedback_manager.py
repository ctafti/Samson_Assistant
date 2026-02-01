# File: src/feedback_manager.py
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone

from src.logger_setup import logger
from src.speaker_profile_manager import add_dynamic_threshold_feedback_entry
from src.config_loader import get_config

def log_matter_correction(correction_type: str, details: Dict[str, Any]):
    """
    Logs a user correction event specifically for Matter assignments.
    This writes to the ground truth log for future analysis and model training.

    Args:
        correction_type (str): The type of correction (e.g., "vocal_override", "gui_manual_correction").
        details (Dict[str, Any]): A dictionary containing the context. It MUST include:
                                   - 'new_matter_id': The ID the user assigned.
                                   - 'source_chunk_id': The chunk where the correction occurred.
    """
    logger.info(f"Logging Matter correction of type '{correction_type}'.")
    try:
        config = get_config()
        kg_config = config.get('knowledge_graph', {})
        
        if not (kg_config.get('enabled') and kg_config.get('ground_truth_file')):
            logger.debug("Knowledge graph or ground truth file not enabled. Skipping matter correction logging.")
            return

        ground_truth_path = Path(kg_config['ground_truth_file'])
        ground_truth_path.parent.mkdir(parents=True, exist_ok=True)

        chunk_id = details.get('source_chunk_id')
        new_matter_id = details.get('new_matter_id')

        if not chunk_id or not new_matter_id:
            logger.error(f"Cannot log matter correction: 'source_chunk_id' or 'new_matter_id' missing from details: {details}")
            return
            
        ground_truth_entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_chunk_id": chunk_id,
            "entity_type": "Matter",
            "canonical_name": new_matter_id,
            "correction_type": correction_type,
            "source_of_truth": details.get('source', 'unknown_correction_source'),
            "details": details
        }
        
        with open(ground_truth_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(ground_truth_entry) + '\n')
        
        logger.info(f"Logged ground truth for chunk '{chunk_id}': Matter is '{new_matter_id}'.")

    except Exception as e_gt:
        logger.error(f"FeedbackManager: Failed to write matter correction to ground truth file: {e_gt}", exc_info=True)


def log_correction_feedback(correction_type: str, details: Dict[str, Any]):
    """
    Logs a user correction event to the appropriate speaker's profile.
    """
    logger.info(f"Logging correction feedback of type '{correction_type}'.")
    
    faiss_id = details.get('faiss_id_of_correct_speaker')
    if faiss_id is None:
        logger.error("FeedbackManager: Cannot log correction feedback. 'faiss_id_of_correct_speaker' is missing from details.")
        return

    try:
        faiss_id_int = int(faiss_id)
    except (ValueError, TypeError):
        logger.error(f"FeedbackManager: 'faiss_id_of_correct_speaker' ('{faiss_id}') is not a valid integer.")
        return

    try:
        config = get_config()
        kg_config = config.get('knowledge_graph', {})
        if kg_config.get('enabled') and kg_config.get('ground_truth_file'):
            ground_truth_path = Path(kg_config['ground_truth_file'])
            ground_truth_path.parent.mkdir(parents=True, exist_ok=True)

            chunk_id = details.get('chunk_id') or details.get('context', {}).get('chunk_id')
            corrected_name = details.get('corrected_speaker_id')

            if chunk_id and corrected_name:
                ground_truth_entry = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "source_chunk_id": chunk_id,
                    "entity_type": "Person",
                    "canonical_name": corrected_name,
                    "source_of_truth": details.get('source', 'unknown_correction_source'),
                    "original_guess": details.get('original_speaker_id')
                }
                
                with open(ground_truth_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(ground_truth_entry) + '\n')
                
                logger.info(f"Logged ground truth for chunk '{chunk_id}': Person is '{corrected_name}'.")

    except Exception as e_gt:
        logger.error(f"FeedbackManager: Failed to write to ground truth file: {e_gt}", exc_info=True)

    feedback_entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "correction_type": correction_type,
        **details
    }
    
    feedback_entry.pop('faiss_id_of_correct_speaker', None)

    logger.debug(f"Constructed feedback entry for faiss_id {faiss_id_int}: {feedback_entry}")

    try:
        success = add_dynamic_threshold_feedback_entry(faiss_id_int, feedback_entry)
        if success:
            logger.info(f"Successfully added correction feedback to profile for faiss_id: {faiss_id_int}.")
        else:
            logger.warning(f"Failed to add correction feedback to profile for faiss_id: {faiss_id_int} (profile may not exist).")
    except Exception as e:
        logger.error(f"FeedbackManager: An unexpected error occurred while adding feedback entry for faiss_id {faiss_id_int}: {e}", exc_info=True)