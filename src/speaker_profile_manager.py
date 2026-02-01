# src/speaker_profile_manager.py

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import threading

from src.logger_setup import logger

# It's good practice to ensure get_config is available if we need project-relative paths
from src.config_loader import get_config, PROJECT_ROOT
from src.utils.file_io import atomic_json_dump
from src.utils.file_locking import get_lock

# --- Globals ---

SPEAKER_PROFILES_FILENAME = "speaker_profiles.json"
PROFILE_LOCK = get_lock("speaker_profiles_db")

# Initialize _speaker_profiles_path later to allow config loading
_speaker_profiles_path: Optional[Path] = None

# --- Initialization ---

def _initialize_profile_path() -> Path:
    """
    Initializes and returns the absolute path to speaker_profiles.json.
    This function is the single source of truth for the profile file path.
    It relies exclusively on the loaded configuration.
    """
    global _speaker_profiles_path
    if _speaker_profiles_path:
        return _speaker_profiles_path

    config = get_config()
    # The get_config() function already resolves paths to be absolute Path objects.
    speaker_db_dir = config.get('paths', {}).get('speaker_db_dir')

    if not speaker_db_dir or not isinstance(speaker_db_dir, Path):
        # This should not happen if config is loaded correctly, but it's a safe fallback.
        fallback_dir = PROJECT_ROOT / "data" / "speaker_database_fallback"
        logger.critical(f"'paths.speaker_db_dir' is missing or invalid in the config. "
                        f"Using a fallback directory: {fallback_dir}. "
                        "THIS IS NOT RECOMMENDED. Please fix your config.yaml.")
        speaker_db_dir = fallback_dir

    speaker_db_dir.mkdir(parents=True, exist_ok=True)
    _speaker_profiles_path = speaker_db_dir / SPEAKER_PROFILES_FILENAME

    logger.debug(f"Speaker profiles path has been set to: {_speaker_profiles_path}")
    return _speaker_profiles_path


def _load_profiles() -> Dict[str, Dict[str, Any]]:
    """Loads speaker profiles from the JSON file. Returns a dictionary keyed by faiss_id (as string)."""
    path = _initialize_profile_path()
    if not path.exists():
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # Profiles are stored as a list, convert to dict keyed by faiss_id (str) for easier access
            profiles_list = json.load(f)
            profiles_dict = {}
            for profile in profiles_list:
                if 'faiss_id' in profile:
                    profiles_dict[str(profile['faiss_id'])] = profile
                else:
                    logger.warning(f"Profile missing faiss_id during load: {profile}")
            return profiles_dict
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {path}. Returning empty profiles.", exc_info=True)
        # Optionally, backup the corrupted file
        corrupted_backup_path = path.with_name(f"{path.stem}_corrupted_{int(datetime.now(timezone.utc).timestamp())}.json")
        try:
            path.rename(corrupted_backup_path)
            logger.info(f"Corrupted speaker_profiles.json backed up to {corrupted_backup_path}")
        except OSError as e_rename:
            logger.error(f"Could not backup corrupted speaker_profiles.json: {e_rename}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading profiles from {path}: {e}", exc_info=True)
        return {}

def _save_profiles(profiles_dict: Dict[str, Dict[str, Any]]):
    """Saves the speaker profiles (values from the dict) to the JSON file as a list."""
    path = _initialize_profile_path()
    try:
        # Store as a list of profiles in the JSON file
        profiles_list = list(profiles_dict.values())
        atomic_json_dump(profiles_list, path)
        logger.debug(f"Speaker profiles saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save speaker profiles to {path}: {e}", exc_info=True)

def _add_embedding_to_profile_object(profile_obj: Dict[str, Any], embedding: np.ndarray, duration_s: float, confidence_score: float, context: str, processing_timestamp: datetime, chunk_id: str) -> Dict[str, Any]:
    """
    Modifies a profile dictionary in-memory to add a new evolution embedding.
    Does NOT perform any file I/O.
    """
    evolution_key = 'segment_embeddings_for_evolution'
    if evolution_key not in profile_obj or not isinstance(profile_obj.get(evolution_key), dict):
        profile_obj[evolution_key] = {}
    
    if context not in profile_obj[evolution_key]:
        profile_obj[evolution_key][context] = []

    embedding_entry = {
        'embedding': embedding.tolist(),
        'duration_s': duration_s,
        'diarization_confidence': confidence_score,
        'timestamp_utc': processing_timestamp.isoformat(),
        'source_chunk_id': chunk_id
    }
    
    profile_obj[evolution_key][context].append(embedding_entry)
    profile_obj['last_updated_utc'] = datetime.now(timezone.utc).isoformat()
    return profile_obj

# --- CRUD Functions ---

def create_speaker_profile(
    faiss_id: int,
    name: str,
    role: Optional[str] = None,
    llm_summary_all_dialogue: Optional[str] = None,
    last_role_inference_utc: Optional[str] = None,  # ISO format string
    dynamic_threshold_feedback: Optional[List[Dict[str, Any]]] = None,
    lifetime_total_audio_s: Optional[float] = None,
    speaker_relationships: Optional[Dict[str, List[str]]] = None,
    **kwargs: Any  # Accept and ignore extra keyword arguments for compatibility
) -> bool:
    """
    Creates a new speaker profile or updates an existing one if faiss_id matches.
    Ensures 'name' is updated if the profile already exists.
    Other fields are only set if provided (won't nullify existing ones if not provided, unless explicitly None).
    'dynamic_threshold_feedback' replaces the list if provided.
    """
    with PROFILE_LOCK:
        profiles = _load_profiles()
        profile_key = str(faiss_id)

        if profile_key in profiles:
            logger.info(f"Updating existing speaker profile for faiss_id: {faiss_id}. Name: '{profiles[profile_key].get('name')}' -> '{name}'.")
            # Update existing profile
            profiles[profile_key]['name'] = name  # Always update name
            if role is not None: profiles[profile_key]['role'] = role
            if llm_summary_all_dialogue is not None: profiles[profile_key]['llm_summary_all_dialogue'] = llm_summary_all_dialogue
            if last_role_inference_utc is not None: profiles[profile_key]['last_role_inference_utc'] = last_role_inference_utc
            
            # For a full create/update, these lists are overwritten if provided.
            if dynamic_threshold_feedback is not None:
                 profiles[profile_key]['dynamic_threshold_feedback'] = dynamic_threshold_feedback
            
            if speaker_relationships is not None:
                profiles[profile_key]['speaker_relationships'] = speaker_relationships

            profiles[profile_key]['last_updated_utc'] = datetime.now(timezone.utc).isoformat()
        else:
            logger.info(f"Creating new speaker profile for faiss_id: {faiss_id}, name: {name}")
            new_profile = {
                "faiss_id": faiss_id,
                "name": name,
                "role": role,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "last_updated_utc": datetime.now(timezone.utc).isoformat(),
                "llm_summary_all_dialogue": llm_summary_all_dialogue,
                "last_role_inference_utc": last_role_inference_utc,
                "profile_last_evolved_utc": None,
                "dynamic_threshold_feedback": dynamic_threshold_feedback if dynamic_threshold_feedback is not None else [],
                "segment_embeddings_for_evolution": {},
                "associated_matter_ids": [],
                "lifetime_total_audio_s": lifetime_total_audio_s if lifetime_total_audio_s is not None else 0.0,
                "speaker_relationships": speaker_relationships if speaker_relationships is not None else {}
            }
            profiles[profile_key] = new_profile
        
        _save_profiles(profiles)
        return True


def get_speaker_profile(faiss_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a speaker profile by faiss_id."""
    with PROFILE_LOCK:
        profiles = _load_profiles()
        return profiles.get(str(faiss_id))

def update_speaker_profile(faiss_id: int, name: Optional[str] = None, associated_matter_ids: Optional[List[str]] = None, role: Optional[str] = None, **kwargs: Any) -> bool:
    """
    Updates specific fields of an existing speaker profile.
    Only updates fields that are explicitly passed in kwargs.
    To add to a list, use a dedicated 'add' function.
    """
    with PROFILE_LOCK:
        all_profiles = _load_profiles()
        profile_key = str(faiss_id)

        if profile_key not in all_profiles:
            logger.warning(f"Cannot update profile: faiss_id {faiss_id} not found.")
            return False

        profile_index = -1
        for i, p in enumerate(list(all_profiles.values())):
            if str(p.get('faiss_id')) == profile_key:
                profile_index = i
                break
        
        profile_to_update = all_profiles.get(profile_key)
        
        if profile_to_update:
            updated = False
            if name is not None:
                if profile_to_update['name'] != name:
                    profile_to_update['name'] = name
                    updated = True
            if role is not None:
                if profile_to_update.get('role') != role:
                    profile_to_update['role'] = role
                    updated = True
            if associated_matter_ids is not None:
                if isinstance(associated_matter_ids, list) and all(isinstance(i, str) for i in associated_matter_ids):
                    if profile_to_update.get('associated_matter_ids') != associated_matter_ids:
                        profile_to_update['associated_matter_ids'] = associated_matter_ids
                        updated = True
                else:
                    logger.warning(f"update_speaker_profile called with invalid format for associated_matter_ids for speaker ID {faiss_id}. Expected list of strings.")

            for key, value in kwargs.items():
                if key in profile_to_update and profile_to_update[key] != value:
                    profile_to_update[key] = value
                    updated = True
                    logger.debug(f"Updating profile {faiss_id}: set {key} to a new value.")

            if updated:
                profile_to_update['last_updated_utc'] = datetime.now(timezone.utc).isoformat()
                _save_profiles(all_profiles)
                logger.info(f"Speaker profile for faiss_id {faiss_id} updated.")
            else:
                logger.info(f"No changes applied to speaker profile for faiss_id {faiss_id}.")
            
            return updated
        return False


def add_dynamic_threshold_feedback_entry(faiss_id: int, feedback_entry: Dict[str, Any]) -> bool:
    """Adds a new entry to the 'dynamic_threshold_feedback' list for a speaker."""
    with PROFILE_LOCK:
        profiles = _load_profiles()
        profile_key = str(faiss_id)

        if profile_key not in profiles:
            logger.warning(f"Cannot add threshold feedback: faiss_id {faiss_id} not found.")
            return False

        if 'dynamic_threshold_feedback' not in profiles[profile_key] or not isinstance(profiles[profile_key]['dynamic_threshold_feedback'], list):
            profiles[profile_key]['dynamic_threshold_feedback'] = []
        
        profiles[profile_key]['dynamic_threshold_feedback'].append(feedback_entry)
        profiles[profile_key]['last_updated_utc'] = datetime.now(timezone.utc).isoformat()
        _save_profiles(profiles)
        logger.info(f"Added dynamic threshold feedback entry for faiss_id {faiss_id}.")
        return True


def delete_speaker_profile(faiss_id: int) -> bool:
    """Removes a speaker profile by faiss_id."""
    with PROFILE_LOCK:
        profiles = _load_profiles()
        profile_key = str(faiss_id)

        if profile_key in profiles:
            del profiles[profile_key]
            _save_profiles(profiles)
            logger.info(f"Speaker profile for faiss_id {faiss_id} deleted.")
            return True
        else:
            logger.warning(f"Cannot delete profile: faiss_id {faiss_id} not found.")
            return False


def get_all_speaker_profiles() -> List[Dict[str, Any]]:
    """Retrieves all speaker profiles as a list."""
    with PROFILE_LOCK:
        profiles = _load_profiles()
        return list(profiles.values())

def get_enrolled_speaker_names() -> List[str]:
    """Retrieves a sorted list of unique names for all enrolled speakers."""
    with PROFILE_LOCK:
        profiles = _load_profiles()
        unique_names = {
            profile.get('name')
            for profile in profiles.values()
            if profile.get('name')
        }
        sorted_names = sorted(list(unique_names))
        logger.debug(f"Retrieved {len(sorted_names)} unique enrolled speaker names.")
        return sorted_names

# --- Functions for Long-Term Profile Evolution ---

def add_segment_embedding_for_evolution(faiss_id: int, embedding: np.ndarray, duration_s: float, confidence_score: float, context: str, processing_timestamp: datetime, chunk_id: str) -> bool:
    """
    Adds a single segment embedding to a profile, handling file load and save.
    This is suitable for single additions but inefficient for batches.
    """
    with PROFILE_LOCK:
        profiles = _load_profiles()
        profile_key = str(faiss_id)

        if profile_key not in profiles:
            logger.warning(f"Cannot add segment embedding: faiss_id {faiss_id} not found.")
            return False

        # Use the private helper to modify the profile object in memory
        profiles[profile_key] = _add_embedding_to_profile_object(
            profiles[profile_key], embedding, duration_s, confidence_score, context, processing_timestamp, chunk_id
        )

        _save_profiles(profiles)
        logger.debug(f"Added segment embedding for evolution to faiss_id {faiss_id} (context: {context}, duration: {duration_s}s). Saved to file.")
        return True


def get_and_clear_pending_embeddings_for_evolution(faiss_id: int, context: str) -> List[Dict[str, Any]]:
    """
    Atomically retrieves all pending segment embeddings for a speaker and context,
    and then clears the list in their profile. This is for the SpeakerIntelligence service.

    Args:
        faiss_id: The speaker's FAISS ID.
        context: The context to retrieve embeddings for ('voip' or 'in_person').

    Returns:
        A list of embedding entries (each containing 'embedding', 'duration_s', and 'timestamp_utc').
    """
    with PROFILE_LOCK:
        profiles = _load_profiles()
        profile_key = str(faiss_id)

        if profile_key not in profiles:
            logger.warning(f"Cannot get/clear segment embeddings: faiss_id {faiss_id} not found.")
            return []

        profile = profiles[profile_key]
        evolution_data = profile.get('segment_embeddings_for_evolution', {})
        if not isinstance(evolution_data, dict):
            logger.warning(f"Evolution data for faiss_id {faiss_id} is not a dict. Resetting.")
            profile['segment_embeddings_for_evolution'] = {}
            _save_profiles(profiles)
            return []

        pending_embeddings = evolution_data.get(context, [])
        
        # Make a copy to return, then clear the original
        embeddings_to_return = list(pending_embeddings)

        evolution_data[context] = []
        profile['last_updated_utc'] = datetime.now(timezone.utc).isoformat()
        _save_profiles(profiles)
        
        logger.info(f"Retrieved and cleared {len(embeddings_to_return)} pending segment embeddings for faiss_id {faiss_id} (context: {context}).")
        return embeddings_to_return


def get_evolution_statistics(faiss_id: int, context: str) -> Dict[str, Any]:
    """
    Retrieves statistics about the evolution data for a speaker profile in a given context.
    
    Args:
        faiss_id: The speaker's FAISS ID.
        context: The context to get stats for ('voip' or 'in_person').

    Returns:
        Dictionary containing evolution statistics or empty dict if speaker/context not found.
    """
    with PROFILE_LOCK:
        profiles = _load_profiles()
        profile_key = str(faiss_id)
        
        if profile_key not in profiles:
            logger.warning(f"Cannot get evolution statistics: faiss_id {faiss_id} not found.")
            return {}
            
        profile = profiles[profile_key]
        evolution_data_for_context = profile.get('segment_embeddings_for_evolution', {}).get(context, [])
        
        if not evolution_data_for_context:
            return {
                'total_segments': 0,
                'total_duration_s': 0.0,
                'average_duration_s': 0.0,
                'oldest_segment_utc': None,
                'newest_segment_utc': None
            }
        
        total_duration = sum(entry.get('duration_s', 0.0) for entry in evolution_data_for_context)
        timestamps = [entry.get('timestamp_utc') for entry in evolution_data_for_context if entry.get('timestamp_utc')]
        
        stats = {
            'total_segments': len(evolution_data_for_context),
            'total_duration_s': total_duration,
            'average_duration_s': total_duration / len(evolution_data_for_context) if evolution_data_for_context else 0.0,
            'oldest_segment_utc': min(timestamps) if timestamps else None,
            'newest_segment_utc': max(timestamps) if timestamps else None
        }
        
        logger.debug(f"Evolution statistics for faiss_id {faiss_id} (context: {context}): {stats}")
        return stats


def get_all_segment_embeddings(faiss_id: int, context: str) -> List[Dict[str, Any]]:
    """
    Retrieves all segment embeddings for evolution for a specific speaker and context.
    This function is used during the recalculation process.
    
    Args:
        faiss_id: The speaker's FAISS ID.
        context: The context to retrieve embeddings for ('voip' or 'in_person').
        
    Returns:
        List of embedding entries with 'embedding', 'duration_s', and 'timestamp_utc'.
    """
    with PROFILE_LOCK:
        profiles = _load_profiles()
        profile_key = str(faiss_id)

        if profile_key not in profiles:
            logger.warning(f"Cannot get segment embeddings: faiss_id {faiss_id} not found.")
            return []

        profile = profiles.get(profile_key, {})
        evolution_data = profile.get('segment_embeddings_for_evolution', {})
        embeddings = evolution_data.get(context, [])
        logger.debug(f"Retrieved {len(embeddings)} segment embeddings for faiss_id {faiss_id} in context '{context}'")
        return embeddings


def clear_segment_embeddings_for_context(faiss_id: int, context: str) -> bool:
    """
    Clears all segment embeddings for a specific context for a given speaker.
    This is called after successful profile recalculation for that context.
    """
    with PROFILE_LOCK:
        profiles = _load_profiles()
        profile_key = str(faiss_id)

        if profile_key not in profiles:
            logger.warning(f"Cannot clear segment embeddings: faiss_id {faiss_id} not found.")
            return False

        profile = profiles[profile_key]
        
        evolution_key = 'segment_embeddings_for_evolution'
        evolution_data = profile.get(evolution_key, {})

        # Gracefully handle cases where evolution_data is not a dictionary.
        if not isinstance(evolution_data, dict):
            logger.warning(f"Cannot clear segment embeddings: '{evolution_key}' is not a dictionary for faiss_id {faiss_id}. Resetting it.")
            profile[evolution_key] = {}
            # Proceed to set timestamps, but there's nothing to clear.
        
        # This handles both when the context doesn't exist and when it does.
        # If it doesn't exist, this does nothing, preventing a KeyError.
        # If it exists, its value (the list of embeddings) is replaced with an empty list.
        evolution_data[context] = []

        now_iso = datetime.now(timezone.utc).isoformat()
        
        # Ensure the timestamp dict exists and is a dictionary before modification
        if 'profile_last_evolved_utc' not in profile or not isinstance(profile.get('profile_last_evolved_utc'), dict):
            profile['profile_last_evolved_utc'] = {}
        
        # Set the timestamp for the specific context to mark the evolution event
        profile['profile_last_evolved_utc'][context] = now_iso

        # Also update the general last_updated timestamp for the whole profile
        profile['last_updated_utc'] = now_iso
        
        _save_profiles(profiles)
        logger.info(f"Cleared segment embeddings for evolution for faiss_id {faiss_id} (context: {context}) and updated evolution timestamp.")
        return True


def get_speakers_with_pending_evolution_data() -> List[int]:
    """
    Returns a list of faiss_ids for speakers who have pending evolution data
    in any context.
    
    Returns:
        List of faiss_ids that have non-empty evolution_data lists.
    """
    with PROFILE_LOCK:
        profiles = _load_profiles()
        speakers_with_data = []
        
        for profile_key, profile in profiles.items():
            evolution_data = profile.get('segment_embeddings_for_evolution', {})
            # Check if any context has a non-empty list of embeddings
            if isinstance(evolution_data, dict) and any(evolution_data.get(context) for context in evolution_data):
                try:
                    faiss_id = int(profile_key)
                    speakers_with_data.append(faiss_id)
                except ValueError:
                    logger.warning(f"Invalid faiss_id format in profile: {profile_key}")
                    
        logger.debug(f"Found {len(speakers_with_data)} speakers with pending evolution data")
        return speakers_with_data


def get_evolution_data_summary() -> Dict[str, Any]:
    """
    Provides a summary of evolution data across all speakers, aggregating all contexts.
    Useful for monitoring and administrative purposes.
    
    Returns:
        Dictionary with summary statistics.
    """
    with PROFILE_LOCK:
        profiles = _load_profiles()
        total_speakers = len(profiles)
        speakers_with_data = 0
        total_segments = 0
        total_duration = 0.0
        
        for profile in profiles.values():
            evolution_data = profile.get('segment_embeddings_for_evolution', {})
            
            all_embeddings = []
            if isinstance(evolution_data, dict):
                 for context_embeddings in evolution_data.values():
                     if isinstance(context_embeddings, list):
                        all_embeddings.extend(context_embeddings)

            if all_embeddings:
                speakers_with_data += 1
            
            total_segments += len(all_embeddings)
            total_duration += sum(entry.get('duration_s', 0.0) for entry in all_embeddings)
        
        summary = {
            'total_speakers': total_speakers,
            'speakers_with_evolution_data': speakers_with_data,
            'total_pending_segments': total_segments,
            'total_pending_duration_s': total_duration,
            'average_segments_per_speaker': total_segments / speakers_with_data if speakers_with_data > 0 else 0
        }
        
        logger.debug(f"Evolution data summary: {summary}")
        return summary


def update_or_remove_evolution_segment(faiss_id: int, context: str, chunk_id: str, new_embedding: Optional[np.ndarray] = None, new_duration_s: Optional[float] = None) -> tuple[bool, float]:
    """
    Atomically finds an evolution segment by its chunk_id and updates, removes, or creates it.

    If new_embedding and new_duration_s are provided:
    - If a segment with chunk_id exists, it is updated.
    - If no segment with chunk_id exists, a new one is created.
    If new_embedding and new_duration_s are None:
    - If a segment with chunk_id exists, it is removed.
    - If no segment with chunk_id exists, nothing happens.

    Args:
        faiss_id: The speaker's FAISS ID.
        context: The context of the segment ('voip', 'in_person', etc.).
        chunk_id: The unique ID of the chunk to find.
        new_embedding: The new numpy array for the embedding, if creating/updating.
        new_duration_s: The new duration in seconds, if creating/updating.

    Returns:
        A tuple: (True if a change was made, duration_of_the_old_segment)
        The old duration is 0.0 if a new segment was created.
    """
    with PROFILE_LOCK:
        profiles = _load_profiles()
        profile_key = str(faiss_id)

        if profile_key not in profiles:
            logger.warning(f"update_or_remove_evolution_segment: faiss_id {faiss_id} not found.")
            return False, 0.0

        profile = profiles[profile_key]
        evolution_data = profile.get('segment_embeddings_for_evolution', {})
        
        # Ensure the context list exists before getting it
        if context not in evolution_data:
            evolution_data[context] = []
        segments_list = evolution_data[context]

        target_index = -1
        for i, segment in enumerate(segments_list):
            if segment.get('source_chunk_id') == chunk_id:
                target_index = i
                break
        
        if target_index == -1:
            # Segment not found. Check if we are supposed to create a new one.
            if new_embedding is not None and new_duration_s is not None:
                # Create a new segment and append it
                new_segment = {
                    'embedding': new_embedding.tolist(),
                    'duration_s': new_duration_s,
                    'diarization_confidence': 1.0,  # Corrections are high confidence
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    'source_chunk_id': chunk_id
                }
                segments_list.append(new_segment)
                logger.info(f"Created new evolution segment for speaker {faiss_id} from chunk '{chunk_id}' (duration: {new_duration_s:.2f}s).")
                profile['last_updated_utc'] = datetime.now(timezone.utc).isoformat()
                _save_profiles(profiles)
                return True, 0.0
            else:
                # No new data provided and segment not found, so do nothing.
                logger.warning(f"update_or_remove_evolution_segment: No evolution segment with chunk_id '{chunk_id}' found for speaker {faiss_id} in context '{context}'. No new data to create one.")
                return False, 0.0
        
        # --- Segment was found (target_index != -1) ---
        old_duration = segments_list[target_index].get('duration_s', 0.0)

        if new_embedding is not None and new_duration_s is not None:
            # Update the existing segment
            segments_list[target_index]['embedding'] = new_embedding.tolist()
            segments_list[target_index]['duration_s'] = new_duration_s
            segments_list[target_index]['diarization_confidence'] = 1.0 # Also set confidence on update
            segments_list[target_index]['timestamp_utc'] = datetime.now(timezone.utc).isoformat() # Mark as updated
            logger.info(f"Updated evolution segment for speaker {faiss_id} from chunk '{chunk_id}' (new duration: {new_duration_s:.2f}s).")
        else:
            # Remove the segment
            del segments_list[target_index]
            logger.info(f"Removed evolution segment for speaker {faiss_id} from chunk '{chunk_id}'.")
        
        profile['last_updated_utc'] = datetime.now(timezone.utc).isoformat()
        _save_profiles(profiles)
        return True, old_duration


def set_speaker_matter_associations(speaker_faiss_id: int, matter_ids: List[str]) -> bool:
    """
    Sets the associated matter IDs for a specific speaker, overwriting the previous list.
    Called by the GUI backend.

    Args:
        speaker_faiss_id: The FAISS ID of the speaker to update.
        matter_ids: A list of matter ID strings to associate with the speaker.

    Returns:
        True if the update was successful, False otherwise.
    """
    logger.info(f"Setting matter associations for speaker FAISS ID {speaker_faiss_id} to {matter_ids}")
    try:
        # The existing update_speaker_profile function can be used directly
        # It already contains the locking and saving logic.
        update_speaker_profile(faiss_id=speaker_faiss_id, associated_matter_ids=matter_ids)
        return True
    except Exception as e:
        logger.error(f"Failed to set matter associations for speaker ID {speaker_faiss_id}: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # To test this, we need to mock the get_config function.
    # For a simple test run, we can just rely on the actual config.
    _initialize_profile_path()

    print("Running enhanced speaker_profile_manager example...")

    # Test Create/Update with new fields
    create_speaker_profile(faiss_id=1, name="Speaker Alpha", role="Interviewer")
    create_speaker_profile(faiss_id=2, name="Speaker Beta", dynamic_threshold_feedback=[{"info": "test"}])
    print(f"Profile 1 (after create): {get_speaker_profile(1)}")

    update_speaker_profile(1, role="Lead Interviewer", llm_summary_all_dialogue="First summary.")
    print(f"Profile 1 (after update): {get_speaker_profile(1)}")

    # Test Add Feedback
    feedback = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "info": "A new feedback event"}
    add_dynamic_threshold_feedback_entry(1, feedback)
    print(f"Profile 1 (after adding feedback): {get_speaker_profile(1)}")

    # Test new embedding functions with context and timestamp
    test_embedding_1 = np.array([0.1, 0.2, 0.3])
    test_embedding_2 = np.array([0.4, 0.5, 0.6])
    now_utc = datetime.now(timezone.utc)
    add_segment_embedding_for_evolution(faiss_id=1, embedding=test_embedding_1, duration_s=2.5, confidence_score=0.98, context='voip', processing_timestamp=now_utc, chunk_id='chunk1')
    add_segment_embedding_for_evolution(faiss_id=1, embedding=test_embedding_2, duration_s=3.2, confidence_score=0.95, context='in_person', processing_timestamp=now_utc, chunk_id='chunk2')
    profile1_after_add = get_speaker_profile(1)
    print(f"Profile 1 (after adding 2 embeddings in different contexts): {json.dumps(profile1_after_add, indent=2)}") # Use json.dumps for pretty printing

    # Test evolution statistics for a specific context
    stats_voip = get_evolution_statistics(faiss_id=1, context='voip')
    print(f"Evolution statistics for speaker 1 (voip): {stats_voip}")
    stats_in_person = get_evolution_statistics(faiss_id=1, context='in_person')
    print(f"Evolution statistics for speaker 1 (in_person): {stats_in_person}")

    # Test new functions with context
    all_voip_embeddings = get_all_segment_embeddings(faiss_id=1, context='voip')
    print(f"All voip embeddings for speaker 1: {len(all_voip_embeddings)} entries")
    
    speakers_with_data = get_speakers_with_pending_evolution_data()
    print(f"Speakers with pending evolution data: {speakers_with_data}")
    
    summary = get_evolution_data_summary()
    print(f"Evolution data summary (aggregated): {summary}")

    # Test clear functionality for one context
    success = clear_segment_embeddings_for_context(faiss_id=1, context='voip')
    print(f"Cleared voip embeddings for speaker 1: {success}")
    
    profile1_after_clear = get_speaker_profile(1)
    print(f"Profile 1 (after clearing voip embeddings): {json.dumps(profile1_after_clear, indent=2)}")

    enrolled_names = get_enrolled_speaker_names()
    print(f"Enrolled names: {enrolled_names}")

    # Cleanup
    delete_speaker_profile(1)
    delete_speaker_profile(2)
    print("Cleaned up test profiles.")