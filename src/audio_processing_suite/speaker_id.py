import torch
import numpy as np
import faiss
from pathlib import Path
import time
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
import json
import threading
import uuid
from datetime import datetime

from src.logger_setup import logger
from src.config_loader import get_config
from src.speaker_profile_manager import get_all_speaker_profiles, get_speaker_profile
from . import persistence
from . import audio_processing
from src.context_manager import get_active_context

# --- Constants ---

SNIPPET_MAX_DURATION_SEC = 12.0 # Max length of audio snippet for enrollment UI
SNIPPETS_SUBDIR = "snippets" # Subdirectory within job output dir to store snippets
SNIPPETS_SUBDIR_AMBIGUOUS = "ambiguous_review" # Subdir for ambiguous snippets
K_FOR_AMBIGUITY_ANALYSIS = 3 # Number of top matches for ambiguity analysis

# --- get_speaker_embeddings (Assumed from previous iterations) ---

def get_speaker_embeddings(speaker_audio_segments_map: Dict[str, List[torch.Tensor]], embedding_model: Any, device: str) -> Dict[str, Dict[str, Any]]:
    """Generates speaker embeddings from audio segments."""
    # Returns: Dict[str(original_diar_label), {'embedding': np.ndarray, 'total_duration_s': float}]
    embeddings_with_duration: Dict[str, Dict[str, Any]] = {}
    logger.info("Entering get_speaker_embeddings...")
    if not speaker_audio_segments_map:
        logger.warning("No speaker audio segments provided for embedding generation.")
        logger.info("Exiting get_speaker_embeddings (no segments).")
        return embeddings_with_duration
    if not embedding_model:
        logger.error("Embedding model not loaded. Cannot generate embeddings.")
        return embeddings_with_duration

    try:
        model_device = torch.device(device)
        embedding_model.to(model_device)
        embedding_model.eval()

        with torch.no_grad():
            for speaker_label, audio_segments_list in speaker_audio_segments_map.items():
                if not audio_segments_list:
                    continue
                
                # Use a simple list to collect numpy arrays after they are moved off the GPU
                speaker_all_segment_embeddings_np: List[np.ndarray] = []
                total_duration_this_speaker_s = 0.0
                sr_for_duration_calc = 16000 
                
                for idx, audio_tensor in enumerate(audio_segments_list):
                    audio_tensor_device = None
                    segment_embedding_raw = None
                    processed_segment_embedding_1d = None
                    logger.debug(f"  Processing segment {idx+1}/{len(audio_segments_list)} for speaker {speaker_label}...")
                    try:
                        # Move tensor to device
                        audio_tensor_device = audio_tensor.to(model_device)
                        
                        # --- Reshape tensor for the model ---
                        if audio_tensor_device.dim() == 3 and audio_tensor_device.shape[1] == 1:
                            logger.debug(f"     Reshaping tensor for model: squeeze(1) from {audio_tensor_device.shape}")
                            audio_tensor_device = audio_tensor_device.squeeze(1)
                        elif audio_tensor_device.dim() == 2 and audio_tensor_device.shape[0] != 1:
                            logger.debug(f"     Reshaping tensor for model: mean(dim=0) from {audio_tensor_device.shape}")
                            audio_tensor_device = torch.mean(audio_tensor_device, dim=0, keepdim=True)
                        elif audio_tensor_device.dim() == 1:
                            logger.debug(f"     Reshaping tensor for model: unsqueeze(0) from {audio_tensor_device.shape}")
                            audio_tensor_device = audio_tensor_device.unsqueeze(0)
                        
                        if audio_tensor_device.numel() == 0:
                            logger.warning(f"     Segment {idx} for {speaker_label} is empty. Skipping.")
                            continue

                        # Update duration
                        segment_duration_s = audio_tensor_device.shape[-1] / sr_for_duration_calc
                        total_duration_this_speaker_s += segment_duration_s

                        # --- NEW: Pre-call Diagnostic Logging and Tensor Correction ---
                        logger.info(f"      RECALC/EMBED PRE-CHECK for speaker '{speaker_label}', segment {idx}:")
                        logger.info(f"        - Tensor Type: {type(audio_tensor_device)}")
                        logger.info(f"        - Tensor Shape (before correction): {audio_tensor_device.shape}")
                        logger.info(f"        - Tensor DType: {audio_tensor_device.dtype}")
                        logger.info(f"        - Tensor Device: {audio_tensor_device.device}")

                        # The SpeechBrain model expects a 2D tensor of shape [batch, samples].
                        # A common error is passing a 1D tensor [samples]. This check corrects it.
                        if audio_tensor_device.dim() == 1:
                            audio_tensor_device = audio_tensor_device.unsqueeze(0)
                            logger.warning(f"        - CORRECTED Tensor Shape: {audio_tensor_device.shape} (unsqueezed to 2D)")

                        logger.info(f"        - Tensor Min/Max/Mean: {audio_tensor_device.min():.4f} / {audio_tensor_device.max():.4f} / {audio_tensor_device.mean():.4f}")
                        logger.info(f"        - Is Contiguous: {audio_tensor_device.is_contiguous()}")
                        # --- END: New Block ---

                        # --- Get embedding from the model ---
                        segment_embedding_raw = embedding_model.encode_batch(audio_tensor_device) 

                        if not (isinstance(segment_embedding_raw, torch.Tensor) and segment_embedding_raw.numel() > 0):
                            logger.warning(f"     Segment {idx} for {speaker_label} produced no embedding. Skipping.")
                            continue

                        # --- Process raw embedding tensor ---
                        if segment_embedding_raw.dim() == 3: 
                            processed_segment_embedding_1d = torch.mean(segment_embedding_raw, dim=1).squeeze(0) 
                        elif segment_embedding_raw.dim() == 2: 
                            processed_segment_embedding_1d = segment_embedding_raw.squeeze(0) 
                        elif segment_embedding_raw.dim() == 1: 
                            processed_segment_embedding_1d = segment_embedding_raw
                        else:
                            logger.warning(f"     Segment {idx} for {speaker_label} produced embedding with unexpected shape: {segment_embedding_raw.shape}. Skipping.")
                            continue

                        # --- Move to CPU and store as numpy array ---
                        if processed_segment_embedding_1d is not None and processed_segment_embedding_1d.dim() == 1:
                            # --- Normalize each segment embedding before appending ---
                            segment_emb_np = processed_segment_embedding_1d.cpu().numpy()
                            norm = np.linalg.norm(segment_emb_np)
                            if norm > 1e-6:
                                segment_emb_np /= norm
                            # else, it's a zero vector, leave it as is.
                            speaker_all_segment_embeddings_np.append(segment_emb_np)

                    except Exception as e_inner:
                        logger.error(f"Error processing segment {idx} for speaker {speaker_label}: {e_inner}", exc_info=True)
                        # The loop will naturally continue to the next segment
                
                # --- Average and normalize the embeddings for the current speaker ---
                if speaker_all_segment_embeddings_np:
                    final_speaker_embedding_np = np.mean(np.array(speaker_all_segment_embeddings_np), axis=0)
                    
                    # --- VITAL: Re-adding the sanity check. If this gets hit, we know the model produced bad data.
                    if not np.isfinite(final_speaker_embedding_np).all():
                        logger.critical(f"CRITICAL: Final embedding for speaker '{speaker_label}' contains NaN or Inf values after averaging. Skipping this speaker to prevent a hang.")
                        continue # Skip to the next speaker
                    
                    norm_before_l2 = np.linalg.norm(final_speaker_embedding_np)
                    
                    if norm_before_l2 > 1e-6: 
                        final_speaker_embedding_normalized = final_speaker_embedding_np / norm_before_l2
                    else:
                        final_speaker_embedding_normalized = final_speaker_embedding_np
                    
                    # --- Calculate Diarization Clustering Confidence ---
                    diarization_clustering_confidence = 1.0  # Default for single-segment speakers
                    if len(speaker_all_segment_embeddings_np) >= 2:
                        # Calculate the average cosine similarity of each segment's embedding to the cluster's mean embedding.
                        # np.dot is cosine similarity for L2-normalized vectors.
                        similarities = [np.dot(final_speaker_embedding_normalized, seg_emb) for seg_emb in speaker_all_segment_embeddings_np]
                        # Clip values to be within [-1, 1] to handle potential floating point inaccuracies
                        diarization_clustering_confidence = float(np.mean(np.clip(similarities, -1.0, 1.0)))
                    
                    embeddings_with_duration[speaker_label] = {
                        'embedding': final_speaker_embedding_normalized,
                        'total_duration_s': total_duration_this_speaker_s,
                        'diarization_clustering_confidence': diarization_clustering_confidence
                    }
    except Exception as model_e:
        logger.error(f"Error during embedding generation main loop: {model_e}", exc_info=True)

    logger.info(f"Exiting get_speaker_embeddings. Generated {len(embeddings_with_duration)} final speaker embeddings.")
    return embeddings_with_duration

def identify_speakers(
    embeddings_data: Dict[str, Dict[str, Any]],
    faiss_index: faiss.Index,
    speaker_map: Dict[int, Any],
    similarity_threshold: float,
    live_refinement_min_similarity: float,
    ambiguity_similarity_lower_bound: float,
    ambiguity_similarity_upper_bound_for_review: float,
    ambiguity_max_similarity_delta_for_multiple_matches: float,
    context: str,
    active_matter_id: Optional[str] = None
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Identifies speakers for given embeddings against a FAISS index and speaker map.
    Also flags segments for live refinement and ambiguity review.

    Returns a map of original_label to {'name': final_name, 'similarity': score}.
    """
    logger.info("Starting speaker identification process...")
    logger.info(f"Processing {len(embeddings_data)} new embeddings against index with {faiss_index.ntotal} speakers.")


    config = get_config()
    aps_cfg = config.get('audio_suite_settings', {})
    initial_thresholds = aps_cfg.get('initial_similarity_thresholds', {})
    context_threshold = initial_thresholds.get(context, similarity_threshold) # Fallback to old global threshold
    context_bonus = aps_cfg.get('context_match_bonus', 0.0)

    speaker_assignments: Dict[str, Dict[str, Any]] = {}
    new_speaker_embeddings_info: List[Dict[str, Any]] = []
    refinement_candidates: List[Dict[str, Any]] = []
    ambiguous_segments_for_review: List[Dict[str, Any]] = []

    if not embeddings_data:
        logger.warning("No embeddings data provided for identification.")
        logger.info("Speaker identification process finished: No embeddings data.")
        return {}, [], [], []

    if faiss_index is None or speaker_map is None:
        logger.warning("FAISS index or speaker map not available. Marking all as new/unknown.")
        if faiss_index is None: logger.info("FAISS index is unavailable.")
        if speaker_map is None: logger.info("Speaker map is unavailable.")
        for original_label, data_dict in embeddings_data.items():
            embedding_vector = data_dict.get('embedding')
            internal_temp_id = original_label
            diar_confidence = data_dict.get('diarization_clustering_confidence', 0.0)
            speaker_assignments[original_label] = {'name': internal_temp_id, 'identification_similarity': 0.0, 'diarization_confidence': diar_confidence}
            if isinstance(embedding_vector, np.ndarray):
                new_speaker_embeddings_info.append({'temp_id': internal_temp_id, 'embedding': embedding_vector, 'original_label': original_label})
        try:
            new_speaker_embeddings_info.sort(key=lambda x: x.get('original_label', ''))
        except Exception: pass
        logger.info(f"Speaker identification process finished: FAISS index/map unavailable. Generated {len(new_speaker_embeddings_info)} new speaker entries.")
        return speaker_assignments, new_speaker_embeddings_info, [], []

    index_size = faiss_index.ntotal
    logger.info(f"Starting speaker identification for {len(embeddings_data)} embedding entries. Index size: {index_size}.")

    if index_size == 0:
        logger.warning("FAISS index is empty. Marking all speakers as new/unknown.")
        for original_label, data_dict in embeddings_data.items():
            embedding_vector = data_dict.get('embedding')
            internal_temp_id = original_label
            diar_confidence = data_dict.get('diarization_clustering_confidence', 0.0)
            speaker_assignments[original_label] = {'name': internal_temp_id, 'identification_similarity': 0.0, 'diarization_confidence': diar_confidence}
            if isinstance(embedding_vector, np.ndarray) and embedding_vector.ndim == 1:
                new_speaker_embeddings_info.append({'temp_id': internal_temp_id, 'embedding': embedding_vector, 'original_label': original_label})
    else:
        query_embeddings_list = []
        query_original_labels_list = []
        query_durations_list = []
        
        for original_label, data_dict in embeddings_data.items():
            embedding_vector = data_dict.get('embedding')
            segment_duration_s_for_embedding = data_dict.get('total_duration_s', 0.0)
            if isinstance(embedding_vector, np.ndarray) and embedding_vector.ndim == 1:
                query_embeddings_list.append(embedding_vector.astype(np.float32))
                query_original_labels_list.append(original_label)
                query_durations_list.append(segment_duration_s_for_embedding)
            else:
                logger.warning(f"Invalid embedding for {original_label}. Assigning error label.")
                speaker_assignments[original_label] = {'name': f"INVALID_EMBEDDING_{original_label}", 'similarity': 0.0}

        if not query_embeddings_list:
            logger.warning("No valid embeddings found to query FAISS index.")
            try:
                new_speaker_embeddings_info.sort(key=lambda x: x.get('original_label', ''))
            except Exception: pass
            logger.info(f"Speaker identification process finished: No valid query embeddings. Generated {len(new_speaker_embeddings_info)} new speaker entries based on initial processing.")
            return speaker_assignments, new_speaker_embeddings_info, [], []

        query_embeddings_np = np.array(query_embeddings_list, dtype=np.float32)
        query_embeddings_np = np.copy(query_embeddings_np)
        logger.info("--- FAISS search prep: Explicit deep copy of query embeddings created in CPU memory. ---")
        
        k_search = K_FOR_AMBIGUITY_ANALYSIS
        if index_size < k_search: k_search = index_size

        logger.info(f"Querying FAISS with {query_embeddings_np.shape[0]} valid embeddings (k={k_search}).")
        logger.info(f"FAISS search PREP: Query embeddings shape: {query_embeddings_np.shape}, dtype: {query_embeddings_np.dtype}.")
        logger.info(f"FAISS search PREP: First query embedding vector (sample): {query_embeddings_np[0][:5]}...")

        try:
            logger.info("--- Entering FAISS search call ---")
            try:
                similarities_matrix, indices_matrix = faiss_index.search(query_embeddings_np, k=k_search)
                logger.info("--- FAISS search call COMPLETE. Processing results. ---")
            except Exception as e_faiss:
                logger.critical(f"--- FAISS search call FAILED WITH UNEXPECTED EXCEPTION: {e_faiss} ---", exc_info=True)
                raise e_faiss

            for i, original_label_from_query in enumerate(query_original_labels_list):
                logger.info(f"Processing original_label: {original_label_from_query}")
                current_query_embedding = query_embeddings_np[i]
                current_segment_duration_s = query_durations_list[i]
                
                is_identified_as_known_speaker = False
                assigned_speaker_name_for_transcript = ""

                if indices_matrix.size == 0 or i >= indices_matrix.shape[0] or i >= similarities_matrix.shape[0] or indices_matrix[i, 0] < 0:
                    logger.info(f"  - Speaker {original_label_from_query}: FAISS search returned no valid top match.")
                    internal_temp_id = original_label_from_query
                    diar_confidence = embeddings_data.get(original_label_from_query, {}).get('diarization_clustering_confidence', 0.0)
                    speaker_assignments[original_label_from_query] = {'name': internal_temp_id, 'identification_similarity': 0.0, 'diarization_confidence': diar_confidence}
                    new_speaker_embeddings_info.append({'temp_id': internal_temp_id, 'embedding': current_query_embedding, 'original_label': original_label_from_query})
                    continue

                # (ADDITION) Context Bonus Logic
                if indices_matrix.shape[1] > 1 and active_matter_id:
                    top_speaker_faiss_id = indices_matrix[i, 0]
                    top_score = similarities_matrix[i, 0]
                    
                    second_speaker_faiss_id = indices_matrix[i, 1]
                    second_score = similarities_matrix[i, 1]

                    # Check for ambiguity: are the top two scores very close?
                    # ambiguity_max_similarity_delta_for_multiple_matches is a good threshold to use here.
                    if top_score > ambiguity_similarity_lower_bound and (top_score - second_score) < ambiguity_max_similarity_delta_for_multiple_matches:
                        top_speaker_profile = get_speaker_profile(top_speaker_faiss_id)
                        second_speaker_profile = get_speaker_profile(second_speaker_faiss_id)

                        if top_speaker_profile and second_speaker_profile:
                            top_in_context = active_matter_id in top_speaker_profile.get('associated_matter_ids', [])
                            second_in_context = active_matter_id in second_speaker_profile.get('associated_matter_ids', [])

                            # Apply bonus ONLY if one is in context and the other is not, to break the tie.
                            context_bonus_ambiguity = config.get('audio_suite_settings', {}).get('ambiguity_matter_context_bonus', 0.05)
                            if top_in_context and not second_in_context:
                                logger.info(f"Applying +{context_bonus_ambiguity} context bonus to '{top_speaker_profile['name']}' to resolve ambiguity.")
                                similarities_matrix[i, 0] += context_bonus_ambiguity # Increase the similarity score
                            elif second_in_context and not top_in_context:
                                logger.info(f"Applying +{context_bonus_ambiguity} context bonus to '{second_speaker_profile['name']}' to resolve ambiguity.")
                                similarities_matrix[i, 1] += context_bonus_ambiguity
                                # If the bonus changes the ranking, swap them so the subsequent code picks the new top match
                                if similarities_matrix[i, 1] > similarities_matrix[i, 0]:
                                    logger.info(f"Context bonus resulted in a new top match: '{second_speaker_profile['name']}'.")
                                    # Swap scores in the matrix
                                    similarities_matrix[i, 0], similarities_matrix[i, 1] = similarities_matrix[i, 1], similarities_matrix[i, 0]
                                    # Swap indices in the matrix
                                    indices_matrix[i, 0], indices_matrix[i, 1] = indices_matrix[i, 1], indices_matrix[i, 0]
                
                top_match_faiss_id = int(indices_matrix[i, 0])
                top_match_similarity = float(similarities_matrix[i, 0])

                top_match_metadata = speaker_map.get(top_match_faiss_id)
                tentative_speaker_name_for_top_match = top_match_metadata.get('name') if top_match_metadata else None
                matched_speaker_context = top_match_metadata.get('context') if top_match_metadata else None

                effective_similarity = top_match_similarity
                if matched_speaker_context and matched_speaker_context == context:
                    effective_similarity += context_bonus
                    logger.debug(f"Applied context bonus of {context_bonus} for matching context '{context}'.")
                
                logger.info(f"  - Speaker {original_label_from_query}: Top match FAISS ID: {top_match_faiss_id}, Sim: {effective_similarity:.4f} (Raw: {top_match_similarity:.4f}), Tentative Name: {tentative_speaker_name_for_top_match}")

                if tentative_speaker_name_for_top_match and effective_similarity >= context_threshold:
                    assigned_speaker_name_for_transcript = tentative_speaker_name_for_top_match
                    diar_confidence = embeddings_data.get(original_label_from_query, {}).get('diarization_clustering_confidence', 0.0)
                    speaker_assignments[original_label_from_query] = {'name': assigned_speaker_name_for_transcript, 'identification_similarity': effective_similarity, 'diarization_confidence': diar_confidence}
                    is_identified_as_known_speaker = True
                    logger.info(f"  - Speaker {original_label_from_query}: Tentatively IDENTIFIED as '{assigned_speaker_name_for_transcript}' (FAISS ID: {top_match_faiss_id}, Sim: {effective_similarity:.3f})")

                    if effective_similarity >= live_refinement_min_similarity:
                        refinement_candidates.append({
                            'faiss_id': top_match_faiss_id,
                            'speaker_name': assigned_speaker_name_for_transcript,
                            'new_segment_embedding': current_query_embedding,
                            'original_diar_label': original_label_from_query,
                            'segment_duration_s': current_segment_duration_s,
                            'diarization_confidence': diar_confidence
                        })
                else:
                    reason_not_id = "Top match FAISS ID not in map" if not tentative_speaker_name_for_top_match else f"Top match effective sim {effective_similarity:.3f} < context_thresh {context_threshold:.3f}"
                    logger.info(f"  - Speaker {original_label_from_query}: Not identified ({reason_not_id}). Marking as new/unknown.")
                    internal_temp_id = original_label_from_query
                    assigned_speaker_name_for_transcript = internal_temp_id
                    diar_confidence = embeddings_data.get(original_label_from_query, {}).get('diarization_clustering_confidence', 0.0)
                    # When a speaker is not identified, store the similarity of its BEST (but failed) match.
                    speaker_assignments[original_label_from_query] = {'name': assigned_speaker_name_for_transcript, 'identification_similarity': effective_similarity, 'diarization_confidence': diar_confidence}
                    new_speaker_embeddings_info.append({'temp_id': internal_temp_id, 'embedding': current_query_embedding, 'original_label': original_label_from_query})
                
                ambiguity_reasons_list = []
                if is_identified_as_known_speaker:
                    if ambiguity_similarity_lower_bound <= effective_similarity < ambiguity_similarity_upper_bound_for_review:
                        ambiguity_reasons_list.append(f"Moderate confidence match to '{assigned_speaker_name_for_transcript}' (sim: {effective_similarity:.3f})")

                    if k_search > 1 and indices_matrix.shape[1] > 1 and indices_matrix[i, 1] >= 0:
                        second_match_faiss_id = int(indices_matrix[i, 1])
                        second_match_similarity = float(similarities_matrix[i, 1])
                        
                        # CORRECTED LOGIC
                        second_match_metadata = speaker_map.get(second_match_faiss_id) # Not applying bonus to 2nd match for simplicity
                        second_match_display_name = second_match_metadata.get('name') if isinstance(second_match_metadata, dict) else "Unknown"

                        if second_match_metadata: # Use second_match_metadata to check for existence
                            delta_similarity = effective_similarity - second_match_similarity
                            if delta_similarity < ambiguity_max_similarity_delta_for_multiple_matches:
                                ambiguity_reasons_list.append(
                                    f"Multiple close enrolled matches: '{assigned_speaker_name_for_transcript}' (sim: {effective_similarity:.3f}) vs "
                                    f"'{second_match_display_name}' (sim: {second_match_similarity:.3f}), delta: {delta_similarity:.3f}"
                                )
                
                if ambiguity_reasons_list:
                    reason_for_flag_str = "; ".join(ambiguity_reasons_list)
                    logger.info(f"    FLAGGED AMBIGUOUS: {original_label_from_query} (tentatively '{assigned_speaker_name_for_transcript}'). Reason(s): {reason_for_flag_str}")
                    
                    # --- NEW WAY ---
                    # (ADDITION)
                    top_matches = list(zip(indices_matrix[i], similarities_matrix[i]))
                    candidates_for_payload = []
                    
                    # Get the top 2 candidates for the flag
                    for speaker_faiss_id, score in top_matches[:2]:
                        profile = get_speaker_profile(speaker_faiss_id)
                        if profile:
                            is_in_context = active_matter_id in profile.get('associated_matter_ids', []) if active_matter_id else False
                            candidates_for_payload.append({
                                "name": profile['name'],
                                "score": round(float(score), 3),
                                "in_context": is_in_context
                            })

                    ambiguous_segments_for_review.append({
                        "reason_for_flag": "Ambiguous speaker identification",
                        "flag_type": "ambiguous_speaker",
                        "candidates": candidates_for_payload,
                        'original_diar_label': original_label_from_query,
                        'tentative_speaker_name': assigned_speaker_name_for_transcript,
                        'tentative_similarity': round(effective_similarity, 4),
                        'segment_embedding': current_query_embedding,
                        'segment_duration_s': current_segment_duration_s,
                    })

        except Exception as e:
            logger.error(f"Error during FAISS search or ambiguity processing: {e}", exc_info=True)
            processed_in_loop = set(speaker_assignments.keys())
            for idx_err, original_label_error_case in enumerate(query_original_labels_list):
                if original_label_error_case not in processed_in_loop:
                    internal_temp_id = original_label_error_case + "_SEARCH_ERROR"
                    original_embedding_data = embeddings_data.get(original_label_error_case, {})
                    diar_confidence = original_embedding_data.get('diarization_clustering_confidence', 0.0)
                    speaker_assignments[original_label_error_case] = {'name': internal_temp_id, 'identification_similarity': 0.0, 'diarization_confidence': diar_confidence}
                    original_embedding_data = embeddings_data.get(original_label_error_case)
                    if original_embedding_data and isinstance(original_embedding_data.get('embedding'), np.ndarray):
                         new_speaker_embeddings_info.append({
                             'temp_id': internal_temp_id,
                             'embedding': original_embedding_data['embedding'].astype(np.float32),
                             'original_label': original_label_error_case
                        })

    logger.info(f"Speaker identification phase complete. Found {len(new_speaker_embeddings_info)} new speaker candidates, {len(refinement_candidates)} refinement candidates, and {len(ambiguous_segments_for_review)} ambiguous segments.")

    if new_speaker_embeddings_info:
        logger.info(f"New speaker embeddings info (first 2): {json.dumps([{'temp_id': n.get('temp_id'), 'original_label': n.get('original_label')} for n in new_speaker_embeddings_info[:2]], indent=2)}")
    if ambiguous_segments_for_review:
        ambiguous_preview = []
        for amb_seg in ambiguous_segments_for_review[:2]:
            preview_item = {k: v for k, v in amb_seg.items() if k != 'segment_embedding'}
            if 'potential_matches' in preview_item:
                 preview_item['potential_matches'] = preview_item['potential_matches'][:2]
            ambiguous_preview.append(preview_item)
        logger.info(f"Ambiguous segments for review (first 2): {json.dumps(ambiguous_preview, indent=2, default=str)}")

    try:
        new_speaker_embeddings_info.sort(key=lambda x: x.get('original_label', ''))
    except Exception as sort_e:
         logger.warning(f"Could not sort new speakers by original_label: {sort_e}.")

    logger.info("Speaker identification process finished.")
    return speaker_assignments, new_speaker_embeddings_info, refinement_candidates, ambiguous_segments_for_review

def update_transcript_speakers(
    word_level_transcript: List[Dict[str, Any]],
    speaker_assignment_map: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Updates speaker labels and adds speaker similarity in a word-level transcript.

    Args:
        word_level_transcript: The transcript data with 'words' list in segments.
        speaker_assignment_map: Dict mapping original diarization labels (e.g., "SPEAKER_00")
                                to a dictionary containing their final name and similarity score.
                                e.g., {'name': 'Admin', 'similarity': 0.85}

    Returns:
        A new list of transcript segments with updated speaker labels and similarity scores.
    """
    if not word_level_transcript:
        logger.warning("update_transcript_speakers: Input transcript is empty. Returning as is.")
        return []
    if not speaker_assignment_map:
        logger.warning("update_transcript_speakers: Speaker assignment map is empty. No speaker updates will be made.")
        return word_level_transcript

    updated_transcript: List[Dict[str, Any]] = []
    logger.debug(f"Updating transcript speakers with map including similarity scores.")

    for segment_idx, original_segment in enumerate(word_level_transcript):
        updated_segment = original_segment.copy()
        updated_segment_words: List[Dict[str, Any]] = []

        original_segment_speaker = str(original_segment.get("speaker", "UNKNOWN_SPEAKER"))

        if 'words' in original_segment and isinstance(original_segment['words'], list):
            for word_idx, original_word_info in enumerate(original_segment['words']):
                if not isinstance(original_word_info, dict):
                    updated_segment_words.append(original_word_info)
                    continue

                updated_word_info = original_word_info.copy()
                original_word_speaker_label = str(original_word_info.get("speaker", original_segment_speaker))

                assignment_details = speaker_assignment_map.get(original_word_speaker_label)

                if assignment_details and isinstance(assignment_details, dict):
                    # Get name and similarity from the new map structure
                    updated_word_info["speaker"] = assignment_details.get('name', original_word_speaker_label)
                    updated_word_info["speaker_identification_similarity"] = round(float(assignment_details.get('identification_similarity', 0.0)), 4)
                    updated_word_info["diarization_confidence"] = round(float(assignment_details.get('diarization_confidence', 0.0)), 4)
                else:
                    # Fallback for unexpected format or if a label is not in the map
                    updated_word_info["speaker"] = original_word_speaker_label
                    updated_word_info["speaker_identification_similarity"] = 0.0
                    updated_word_info["diarization_confidence"] = 0.0

                # To avoid confusion, delete the old probability and similarity keys
                if "speaker_similarity" in updated_word_info:
                    del updated_word_info["speaker_similarity"]
                if "probability" in updated_word_info:
                    del updated_word_info["probability"]
                
                updated_segment_words.append(updated_word_info)
            updated_segment["words"] = updated_segment_words

        # Update segment-level speaker based on word-level updates
        segment_assignment_details = speaker_assignment_map.get(original_segment_speaker)
        if segment_assignment_details and isinstance(segment_assignment_details, dict):
            final_segment_speaker_for_segment = segment_assignment_details.get('name', original_segment_speaker)
        else:
            final_segment_speaker_for_segment = original_segment_speaker

        if updated_segment_words:
            speaker_counts_in_words = defaultdict(int)
            for w_info in updated_segment_words:
                speaker_counts_in_words[w_info.get("speaker", "UNKNOWN_SPEAKER")] += 1
            
            known_speakers_in_words = {spk: count for spk, count in speaker_counts_in_words.items() if spk != "UNKNOWN_SPEAKER"}
            if known_speakers_in_words:
                final_segment_speaker_for_segment = max(known_speakers_in_words, key=known_speakers_in_words.get)
            elif speaker_counts_in_words:
                final_segment_speaker_for_segment = max(speaker_counts_in_words, key=speaker_counts_in_words.get)

        updated_segment["speaker"] = final_segment_speaker_for_segment
        updated_transcript.append(updated_segment)

    logger.info(f"Transcript speaker labels and similarity scores updated based on speaker_assignment_map.")
    return updated_transcript

def run_speaker_identification(
    combined_transcript: list,
    db_lock: threading.Lock, # Accept the DB lock
    embedding_model: Any,
    emb_dim: int, # Dimension of embeddings
    audio_path: Path,
    output_dir: Path, # Job output directory
    processing_device: str,
    min_segment_duration: float, # Min duration for extracting segments for embedding
    similarity_threshold: float, # For initial identification
    faiss_index_path: Path,
    speaker_map_path: Path,
    context: str,
    active_matter_context: Optional[Dict[str, Any]],
    processing_date_utc: datetime,
    live_refinement_min_similarity: float,
    live_refinement_min_segment_duration_s: float, # Min duration of a segment to be eligible for refinement
    ambiguity_similarity_lower_bound: float,
    ambiguity_similarity_upper_bound_for_review: float,
    ambiguity_max_similarity_delta_for_multiple_matches: float
) -> Dict[str, Any]:
    # (ADDITION) Fetch context at the start of the function
    active_matter_id = active_matter_context.get('matter_id') if active_matter_context else None
    logger.info(f"Speaker Identification running with active matter context: {active_matter_id}")

    logger.debug(f"--- Starting Speaker ID Orchestration for: {audio_path.name} ---")
    logger.debug(f"  - Similarity Thresh: {similarity_threshold}, Ambiguity Upper Bound: {ambiguity_similarity_upper_bound_for_review}")

    results = {
        "identified_transcript": combined_transcript, # Default to original if steps fail
        "new_speaker_enrollment_data": [],
        "refinement_data_for_update": [],
        "ambiguous_segments_flagged": []
    }
    if not embedding_model:
        logger.warning("Skipping Speaker Identification (embedding model not loaded/provided).")
        return results
    if not combined_transcript:
        logger.warning("Skipping Speaker Identification (no transcript data).")
        logger.info(f"Speaker identification orchestration finished early: No transcript data for {audio_path.name}.")
        return results

    start_id_time = time.time()

    full_audio_waveform_tensor: Optional[torch.Tensor] = None
    audio_sample_rate: Optional[int] = None
    speaker_audio_segments_map: Optional[Dict[str, List[torch.Tensor]]] = None
    speaker_word_segments_info_map: Optional[Dict[str, List[Dict[str, Any]]]] = None
    speaker_embeddings_data_map: Optional[Dict[str, Dict[str, Any]]] = None

    try:
        full_audio_waveform_tensor, audio_sample_rate = audio_processing.load_audio(audio_path, target_sr=16000) 
        if full_audio_waveform_tensor is None or audio_sample_rate is None:
            raise ValueError(f"Audio loading failed for {audio_path}.")

        logger.debug(f"Extracting audio segments for speaker ID from '{audio_path.name}'...")
        
        intermediate_snippets_dir = output_dir / SNIPPETS_SUBDIR / "speaker_id_temp_extraction"
        intermediate_snippets_dir.mkdir(parents=True, exist_ok=True)

        speaker_audio_segments_map, speaker_word_segments_info_map = \
            audio_processing.extract_speaker_audio_segments(
                full_audio_tensor=full_audio_waveform_tensor,      
                sample_rate=audio_sample_rate,                     
                word_level_transcript=combined_transcript,
                min_duration_s=min_segment_duration,
                target_sr=16000, 
                output_dir=intermediate_snippets_dir
            )
        logger.debug(f"Extracted {len(speaker_audio_segments_map) if speaker_audio_segments_map else 0} speaker audio segments map items.")
        
        if not speaker_audio_segments_map:
            logger.warning("No sufficient audio segments extracted. Skipping embedding generation and identification.")
        else:
            logger.info("--- Entering get_speaker_embeddings call ---")
            speaker_embeddings_data_map = get_speaker_embeddings(
                speaker_audio_segments_map, embedding_model, processing_device
            )
            logger.info("--- Exited get_speaker_embeddings call ---")
            logger.debug(f"Generated {len(speaker_embeddings_data_map) if speaker_embeddings_data_map else 0} speaker embeddings data map items.")

            if speaker_embeddings_data_map:
                with db_lock:
                    faiss_index = persistence.load_or_create_faiss_index(faiss_index_path, emb_dim)
                    speaker_map = persistence.load_or_create_speaker_map(speaker_map_path)
                    
                    logger.debug(f"  - Using FAISS index instance with {faiss_index.ntotal} vectors.")
                    logger.debug(f"  - Using speaker map instance with {len(speaker_map)} speakers.")
                    logger.info("--- Entering identify_speakers call ---")
                    speaker_assignments_raw, new_speaker_info_raw, raw_refinement_candidates, raw_ambiguous_segments = \
                        identify_speakers(
                            embeddings_data=speaker_embeddings_data_map, 
                            faiss_index=faiss_index, 
                            speaker_map=speaker_map,
                            similarity_threshold=similarity_threshold,
                            live_refinement_min_similarity=live_refinement_min_similarity,
                            ambiguity_similarity_lower_bound=ambiguity_similarity_lower_bound,
                            ambiguity_similarity_upper_bound_for_review=ambiguity_similarity_upper_bound_for_review,
                            ambiguity_max_similarity_delta_for_multiple_matches=ambiguity_max_similarity_delta_for_multiple_matches,
                            context=context,
                            active_matter_id=active_matter_id
                        )
                    logger.info("--- Exited identify_speakers call ---")
                logger.debug(f"Identify_speakers returned: {len(new_speaker_info_raw)} new speaker candidates, {len(raw_refinement_candidates)} refinement candidates, {len(raw_ambiguous_segments)} ambiguous segments.")
                
                filtered_refinement_data_list = []
                logger.debug(f"Processing {len(raw_refinement_candidates)} raw refinement candidates. Min segment duration for refinement: {live_refinement_min_segment_duration_s}s.")
                for cand in raw_refinement_candidates:
                    if cand['segment_duration_s'] >= live_refinement_min_segment_duration_s:
                        # Corrected: Ensure segment_duration_s is passed through for evolution data capture.
                        filtered_refinement_data_list.append({
                            'faiss_id': cand['faiss_id'],
                            'speaker_name': cand['speaker_name'],
                            'new_segment_embedding': cand['new_segment_embedding'],
                            'segment_duration_s': cand['segment_duration_s'],
                            'diarization_confidence': cand['diarization_confidence']
                        })
                results["refinement_data_for_update"] = filtered_refinement_data_list
                logger.debug(f"Collected {len(filtered_refinement_data_list)} candidates for live embedding refinement (after duration filter).")

                # --- AC1: Secure Snippet Pre-generation ---
                processed_ambiguous_segments_list = []
                if raw_ambiguous_segments:
                    logger.debug(f"Processing {len(raw_ambiguous_segments)} raw ambiguous segments to pre-generate snippets...")
                    
                    for amb_idx, amb_item in enumerate(raw_ambiguous_segments):
                        original_diar_label = amb_item['original_diar_label']
                        
                        
                        text_preview = "N/A"
                        word_segment_infos = speaker_word_segments_info_map.get(original_diar_label, [])
                        relative_start_s_in_wav = 0.0
                        # The 'segment_duration_s' on amb_item is the TOTAL duration, which is the source of the bug.
                        # We will recalculate a proper end time based on the FIRST segment as a sample.
                        relative_end_s_in_wav = 12.0 # Default fallback

                        if word_segment_infos:
                            first_segment_info = word_segment_infos[0]
                            relative_start_s_in_wav = first_segment_info.get('start', 0.0)
                            # Calculate end time using the same robust pattern as new speaker enrollment
                            first_segment_end_time = first_segment_info.get('end', relative_start_s_in_wav + first_segment_info.get('duration', 12.0))
                            # Cap the snippet duration to the max allowed
                            segment_actual_duration = first_segment_end_time - relative_start_s_in_wav
                            relative_end_s_in_wav = relative_start_s_in_wav + min(segment_actual_duration, SNIPPET_MAX_DURATION_SEC)
                            text_preview = f"Segment by {original_diar_label}..."
                        else:
                            # Fallback if no detailed segment info is found
                            relative_start_s_in_wav = 0.0 
                            relative_end_s_in_wav = amb_item.get('segment_duration_s', 12.0)
                            text_preview = "N/A"
                        

                        amb_item['text_preview'] = text_preview
                        amb_item['relative_start_s_in_wav'] = relative_start_s_in_wav
                        amb_item['relative_end_s_in_wav'] = relative_end_s_in_wav
                        
                        if 'snippet_path_abs' in amb_item:
                            del amb_item['snippet_path_abs']
                            logger.debug(f"    Removed obsolete 'snippet_path_abs' from ambiguous item {amb_idx}.")
                        
                        # --- REMOVED SNIPPET FILE GENERATION ---
                        
                        processed_ambiguous_segments_list.append(amb_item)
                    
                    results["ambiguous_segments_flagged"] = processed_ambiguous_segments_list

                final_transcript_speaker_map: Dict[str, Dict[str, Any]] = speaker_assignments_raw.copy()
                new_speaker_enrollment_candidates_temp: List[Dict[str, Any]] = []

                for new_speaker_candidate in new_speaker_info_raw:
                    original_diar_label = new_speaker_candidate['original_label']
                    assignment_details = final_transcript_speaker_map.get(original_diar_label)
                    # A speaker is "new" if its assigned name is just its own original diarization label.
                    if assignment_details and assignment_details.get('name') == original_diar_label:
                        new_speaker_enrollment_candidates_temp.append({
                            'temp_id': original_diar_label,
                            'original_label': original_diar_label,
                            'embedding': new_speaker_candidate['embedding']
                        })
                
                new_speaker_enrollment_candidates_temp.sort(key=lambda x: x.get('temp_id', ''))
                logger.debug(f"Identified {len(new_speaker_enrollment_candidates_temp)} candidates for new speaker enrollment (before snippet generation).")

                results["identified_transcript"] = update_transcript_speakers(
                    combined_transcript, final_transcript_speaker_map
                )

                if new_speaker_enrollment_candidates_temp:
                    final_new_speaker_enrollment_data_list: List[Dict[str, Any]] = []
                    for enroll_idx, enroll_cand_item in enumerate(new_speaker_enrollment_candidates_temp):
                        original_diar_label_for_enroll = enroll_cand_item['temp_id']
                        logger.debug(f"  Processing enrollment candidate {enroll_idx}: original_diar_label='{original_diar_label_for_enroll}'")

                        start_time_s = 0.0
                        end_time_s = 0.0
                        
                        word_segment_infos_for_enroll = speaker_word_segments_info_map.get(original_diar_label_for_enroll, [])
                        if word_segment_infos_for_enroll:
                            first_chunk_info_enroll = word_segment_infos_for_enroll[0]
                            start_time_s = first_chunk_info_enroll.get('start', 0.0)
                            end_time_s = first_chunk_info_enroll.get('end', start_time_s + first_chunk_info_enroll.get('duration', 0.0))
                            first_segment_end_time = first_chunk_info_enroll.get('end', start_time_s + first_chunk_info_enroll.get('duration', SNIPPET_MAX_DURATION_SEC))
                            segment_actual_duration = first_segment_end_time - start_time_s
                            # The final end time is the start time plus the capped duration.
                            end_time_s = start_time_s + min(segment_actual_duration, SNIPPET_MAX_DURATION_SEC)
                        
                        logger.debug(f"    Snippet generation for enrollment candidate {enroll_idx} is now handled on-demand by the UI.")
                        
                        final_new_speaker_enrollment_data_list.append({
                            'temp_id': original_diar_label_for_enroll, 
                            'original_label': original_diar_label_for_enroll,
                            'embedding': enroll_cand_item['embedding'],
                            'start_time': round(start_time_s, 3),   
                            'end_time': round(end_time_s, 3),          
                            'snippet_path_abs': None # Set to None as it's no longer generated
                        })
                    results["new_speaker_enrollment_data"] = final_new_speaker_enrollment_data_list
                    logger.debug(f"Finalized {len(final_new_speaker_enrollment_data_list)} new speaker enrollment data entries without pre-generating snippets.")
                else: 
                    logger.debug("No new speakers identified for enrollment after processing.")
            else: 
                logger.warning("No speaker embeddings generated. Skipping further identification steps.")
        
        end_id_time = time.time()
        logger.info(f"Speaker Identification orchestration finished for {audio_path.name}. "
                    f"New speakers for enrollment: {len(results['new_speaker_enrollment_data'])}. "
                    f"Refinement ops: {len(results['refinement_data_for_update'])}. "
                    f"Ambiguous segments flagged: {len(results['ambiguous_segments_flagged'])}. "
                    f"Took {end_id_time - start_id_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"Error during speaker identification orchestration for {audio_path.name}: {e}", exc_info=True)
    return results

def update_faiss_embeddings_for_refinement(
    refinements: List[Dict[str, Any]], # Item: {'faiss_id', 'speaker_name', 'new_segment_embedding'}
    faiss_index_path: Path,
    speaker_map_path: Path,
    db_lock: threading.Lock
) -> bool:
    """
    Updates FAISS embeddings for existing speakers by averaging with new segment embeddings.
    This uses a rebuild strategy for safety.
    RETURNS true if the database was modified, false otherwise.
    """
    if not refinements:
        logger.info("No refinement operations provided. Skipping FAISS update for refinement.")
        return False

    with db_lock:
        # NOTE: get_config() is assumed to be an available function, likely from a central config module.
        # This function is not defined in the current file but is required by the instructions.
        embedding_dim = get_config().get('audio_suite_settings', {}).get('embedding_dim', 192)
        faiss_index = persistence.load_or_create_faiss_index(faiss_index_path, embedding_dim)
        speaker_map = persistence.load_or_create_speaker_map(speaker_map_path)

        logger.debug(f"Starting FAISS embedding refinement (rebuild strategy) for {len(refinements)} items.") # Changed to debug
        changes_made_to_db_structure = False
        current_dimension = faiss_index.d

        # 1. Load all current embeddings and names
        all_current_embeddings_list: List[np.ndarray] = []
        all_current_names_list: List[str] = []
        
        if faiss_index.ntotal > 0:
            try:
                # Reconstruct all embeddings. Note: reconstruct_n might not be efficient for all index types.
                # If index is not "direct_map" type or doesn't support reconstruct_n well,
                # iterative reconstruct might be needed, or a different strategy.
                # Assuming IndexFlatIP/L2 supports reconstruct_n.
                all_current_embeddings_np_array = faiss_index.reconstruct_n(0, faiss_index.ntotal)
                all_current_embeddings_list = [all_current_embeddings_np_array[j] for j in range(faiss_index.ntotal)]

                for i in range(faiss_index.ntotal):
                    # It's crucial that speaker_map keys are contiguous from 0 to ntotal-1 if we iterate like this.
                    # This is true if FAISS IDs are managed carefully (e.g., after rebuilds).
                    # If IDs can be non-contiguous (e.g., after selective `remove_ids` without rebuild), this needs adjustment.
                    # However, a rebuild strategy implies speaker_map keys are indeed 0..N-1.
                    speaker_info = speaker_map.get(i)
                    name = None
                    if isinstance(speaker_info, dict):
                        name = speaker_info.get('name')
                    elif isinstance(speaker_info, str):
                        name = speaker_info

                    if name is None:
                        logger.warning(f"FAISS ID {i} present in index but not in speaker_map during refinement prep. Assigning temp name.")
                        name = f"ORPHANED_FAISS_ID_{i}"
                    all_current_names_list.append(name)
            except Exception as e_recon_all:
                logger.error(f"Failed to reconstruct all embeddings from current FAISS index: {e_recon_all}. Aborting refinement.", exc_info=True)
                return False
        
        # 2. Create a map of speaker_name to list of their embeddings (existing + new refinement ones)
        # This handles cases where a speaker might have multiple existing embeddings (if not already averaged)
        # or multiple new refinement segments.
        name_to_all_embeddings_for_averaging: Dict[str, List[np.ndarray]] = defaultdict(list)
        
        # Add existing embeddings
        for i, name in enumerate(all_current_names_list):
            name_to_all_embeddings_for_averaging[name].append(all_current_embeddings_list[i])
            
        # Add new segment embeddings from refinements
        for item in refinements:
            speaker_name_to_refine = item['speaker_name']
            new_segment_emb_unnormalized = item['new_segment_embedding']

            if not isinstance(new_segment_emb_unnormalized, np.ndarray) or new_segment_emb_unnormalized.ndim != 1:
                logger.warning(f"Invalid new_segment_embedding for '{speaker_name_to_refine}' in refinement. Skipping this item.")
                continue
            
            # Normalize the new segment embedding
            new_segment_emb_normalized = new_segment_emb_unnormalized.astype(np.float32)
            norm_new = np.linalg.norm(new_segment_emb_normalized)
            if norm_new > 1e-6:
                new_segment_emb_normalized /= norm_new
            else:
                logger.warning(f"Norm of new segment embedding for '{speaker_name_to_refine}' is near zero. Using as is.")
                # If it's a zero vector, it will slightly pull the average towards zero.

            name_to_all_embeddings_for_averaging[speaker_name_to_refine].append(new_segment_emb_normalized)
            changes_made_to_db_structure = True # Even if name didn't exist, we are adding it for refinement.

        if not changes_made_to_db_structure: # No valid refinements to process
            logger.info("No valid refinement data to process after initial checks.")
            return False

        # 3. Recompute final (averaged and normalized) embeddings for each speaker
        final_embeddings_to_store_list: List[np.ndarray] = []
        final_names_for_new_map_list: List[str] = []

        logger.info("--- REFINEMENT DIAGNOSTICS: Embeddings collected for averaging ---")
        for name, embs_list in name_to_all_embeddings_for_averaging.items():
            logger.info(f"  - Speaker: '{name}' ({len(embs_list)} embeddings to average)")
            for i, emb in enumerate(embs_list):
                logger.info(f"    - Embedding [{i}]: norm={np.linalg.norm(emb):.6f}, preview={emb[:4]}")
        logger.info("--- END REFINEMENT DIAGNOSTICS ---")

        existing_names_in_order = []
        if speaker_map:
            sorted_items = sorted(speaker_map.items())
            for _, info in sorted_items:
                name = info.get('name') if isinstance(info, dict) else info
                if name:
                    existing_names_in_order.append(name)

        existing_names_set = set(existing_names_in_order)
        new_names_to_add = sorted([
            name for name in name_to_all_embeddings_for_averaging
            if name not in existing_names_set
        ])

        final_name_order = existing_names_in_order + new_names_to_add
        
        for name in final_name_order:
            list_of_embeddings_for_name = name_to_all_embeddings_for_averaging.get(name)
            if list_of_embeddings_for_name:
                # All embeddings in list_of_embeddings_for_name should already be normalized individually
                # (existing ones from a presumably normalized index, new ones normalized above)
                averaged_embedding = np.mean(np.array(list_of_embeddings_for_name), axis=0)

                logger.info(f"--- REFINEMENT DIAGNOSTICS: Averaging for '{name}' ---")
                logger.info(f"  - Averaged (pre-norm): norm={np.linalg.norm(averaged_embedding):.6f}, preview={averaged_embedding[:4]}")
                
                # Normalize the final averaged embedding
                norm_averaged = np.linalg.norm(averaged_embedding)
                if norm_averaged > 1e-6:
                    averaged_embedding /= norm_averaged
                else:
                    logger.warning(f"Norm of final averaged embedding for speaker '{name}' is near zero. Storing as is.")
                    # This might happen if embeddings cancel out or are all zero.
                
                logger.info(f"  - Final (post-norm): norm={np.linalg.norm(averaged_embedding):.6f}, preview={averaged_embedding[:4]}")
                logger.info("--- END REFINEMENT DIAGNOSTICS ---")
                
                final_embeddings_to_store_list.append(averaged_embedding.astype(np.float32))
                final_names_for_new_map_list.append(name)
                
        # 4. Rebuild FAISS index and new speaker map
        if final_embeddings_to_store_list:
            # Create a new, empty FAISS index of the same type and dimension
            # Assuming IndexFlatIP as discussed. If another type, adjust here.
            rebuilt_faiss_index = faiss.IndexFlatIP(current_dimension) 
            
            final_embeddings_np_array = np.array(final_embeddings_to_store_list)
            rebuilt_faiss_index.add(final_embeddings_np_array)

            name_to_metadata = {}
            for old_id, speaker_info in speaker_map.items():
                if isinstance(speaker_info, dict):
                    name_to_metadata[speaker_info.get('name')] = speaker_info
                elif isinstance(speaker_info, str): # Handle legacy string-only format
                    name_to_metadata[speaker_info] = {'name': speaker_info, 'context': 'unknown'}

            new_speaker_map_rebuilt = {}
            for i, name in enumerate(final_names_for_new_map_list):
                # Use existing metadata if found, otherwise create a default entry.
                existing_metadata = name_to_metadata.get(name, {'name': name, 'context': 'unknown'})
                new_speaker_map_rebuilt[i] = existing_metadata
           
            
            # 5. Save the new index and map, overwriting the old ones
            try:
                persistence.save_faiss_index(rebuilt_faiss_index, faiss_index_path)
                persistence.save_speaker_map(new_speaker_map_rebuilt, speaker_map_path)
                logger.debug(f"Successfully saved rebuilt FAISS index ({rebuilt_faiss_index.ntotal} embeddings) and speaker map after refinement.") # Changed to debug
                
                return True
            except Exception as e_save_rebuild:
                logger.error(f"Error saving rebuilt FAISS index or speaker map after refinement: {e_save_rebuild}", exc_info=True)
                return False # Return original instances on save failure
        else:
            # This case means after processing refinements, there are no embeddings left (e.g., all names became empty lists)
            # Or if the original index was empty and no refinements were added for new names.
            # If original index was empty and refinements list was not, this means names in refinement didn't match anything.
            logger.info("No final embeddings to store after refinement process. Database might be emptied if it was non-empty.")
            # If we intend to empty the DB, we should explicitly save an empty index/map.
            # For now, if `final_embeddings_to_store_list` is empty, we don't save, meaning no change to files.
            # However, if `changes_made_to_db_structure` was true, it means we *intended* a change.
            # If `final_embeddings_to_store_list` is empty but we intended changes, it's likely an edge case or error.
            # Let's save an empty DB if that's the outcome of refinement on a previously non-empty DB.
            if faiss_index.ntotal > 0 or speaker_map: # If DB was not empty before
                logger.warning("Refinement process resulted in an empty set of embeddings. Clearing and saving empty database.")
                try:
                    empty_index_to_save = faiss.IndexFlatIP(faiss_index.d if faiss_index.d > 0 else 1) # Ensure valid dim
                    persistence.save_faiss_index(empty_index_to_save, faiss_index_path)
                    persistence.save_speaker_map({}, speaker_map_path)
                    return True # Change made (DB emptied)
                except Exception as e_save_empty:
                    logger.error(f"Error saving empty database after refinement: {e_save_empty}", exc_info=True)
                    return False
            return False # No changes effectively made if started empty and ended empty.

def enroll_speakers_programmatic(
    enrollment_data_list: List[Dict[str, Any]], # Item: {'temp_id', 'name_to_enroll', 'embedding', 'context'}
    faiss_index_path: Path,
    speaker_map_path: Path,
    embedding_dim: int,
    db_lock: threading.Lock # Explicitly take the lock
) -> Tuple[bool, Optional[faiss.Index], Optional[Dict[int, Any]]]:
    """
    Enrolls new speakers or updates existing speakers programmatically based on name.
    If 'name_to_enroll' exists, its embedding is updated by averaging. Otherwise, new entry.
    Uses a load-rebuild-save strategy for safety.
    RETURNS the new index and map instances instead of modifying in-place.
    """
    logger.debug(f"--- Starting Programmatic Speaker Enrollment/Update for {len(enrollment_data_list)} items ---") # Changed to debug
    if not enrollment_data_list:
        logger.info("No programmatic enrollment data provided. Skipping.")
        return False, None, None

    # Load fresh instances from disk under lock
    with db_lock:
        faiss_index = persistence.load_or_create_faiss_index(faiss_index_path, embedding_dim)
        speaker_map = persistence.load_or_create_speaker_map(speaker_map_path)

    current_dimension = faiss_index.d
    if current_dimension <= 0 and enrollment_data_list: # Try to get dimension from first embedding if index is empty
        first_valid_emb = next((item.get('embedding') for item in enrollment_data_list if isinstance(item.get('embedding'), np.ndarray) and item.get('embedding').ndim == 1), None)
        if first_valid_emb is not None:
            current_dimension = first_valid_emb.shape[0]
            logger.info(f"FAISS index was empty, derived dimension {current_dimension} from first enrollment candidate.")
        else: # Still no dimension
             logger.error("FAISS index is empty and no valid embeddings in enrollment data to derive dimension. Cannot proceed.")
             return False, faiss_index, speaker_map


    names_to_replace = {
        item['name_to_enroll'] for item in enrollment_data_list 
        if item.get('update_mode') == 'replace'
    }
    logger.debug(f"Programmatic enroll: Replace mode activated for speakers: {names_to_replace}")

    # --- START OF FIX: Implement Duration-Weighted Averaging ---
    all_profiles = get_all_speaker_profiles()
    name_to_profile_map = {p.get('name'): p for p in all_profiles}

    name_to_weighted_embeddings: Dict[str, List[Tuple[np.ndarray, float]]] = defaultdict(list)

    # 1. Collect all existing embeddings and their weights (durations)
    if faiss_index.ntotal > 0:
        existing_embeddings_arr = faiss_index.reconstruct_n(0, faiss_index.ntotal)
        for i in range(faiss_index.ntotal):
            name = (speaker_map.get(i) or {}).get('name') or speaker_map.get(i)
            if not name or name in names_to_replace: continue
            
            profile = name_to_profile_map.get(name)
            # Use lifetime_total_audio_s as weight, fallback to 1.0 for robustness
            weight = profile.get('lifetime_total_audio_s', 1.0) if profile else 1.0
            name_to_weighted_embeddings[name].append((existing_embeddings_arr[i], weight))

    # 2. Add new embeddings from the enrollment list with their provided weights
    for item in enrollment_data_list:
        name_to_enroll = item.get('name_to_enroll')
        new_embedding = item.get('embedding')
        if not name_to_enroll or not isinstance(new_embedding, np.ndarray): continue
        
        # Use provided duration as weight, fallback to 1.0 to avoid zero-weight issues
        weight = item.get('duration_s', 1.0)
        # Ensure embedding is normalized
        norm = np.linalg.norm(new_embedding)
        if norm > 1e-6: new_embedding /= norm
        name_to_weighted_embeddings[name_to_enroll].append((new_embedding, weight))
    # --- END OF FIX ---

    # Get the stable order of existing speakers from the current map
    existing_names_in_order = []
    if speaker_map:
        sorted_items = sorted(speaker_map.items())
        for _, info in sorted_items:
            name = info.get('name') if isinstance(info, dict) else info
            if name: existing_names_in_order.append(name)

    existing_names_set = set(existing_names_in_order)
    new_names_to_add = sorted([name for name in name_to_weighted_embeddings if name not in existing_names_set])
    final_name_order = existing_names_in_order + new_names_to_add

    # 3. Recompute final embeddings using weighted average
    final_embeddings_to_store_prog: List[np.ndarray] = []
    final_names_for_new_map_prog: List[str] = []

    for name in final_name_order:
        list_of_embs_and_weights = name_to_weighted_embeddings.get(name)
        if list_of_embs_and_weights:
            # --- START OF FIX: Use np.average for weighted calculation ---
            embeddings, weights = zip(*list_of_embs_and_weights)
            valid_indices = [i for i, w in enumerate(weights) if w > 0]
            if not valid_indices:
                averaged_emb_prog = np.mean(np.array(embeddings), axis=0).astype(np.float32)
            else:
                valid_embeddings = [embeddings[i] for i in valid_indices]
                valid_weights = [weights[i] for i in valid_indices]
                averaged_emb_prog = np.average(valid_embeddings, axis=0, weights=valid_weights).astype(np.float32)
            # --- END OF FIX ---

            norm_avg_prog = np.linalg.norm(averaged_emb_prog)
            if norm_avg_prog > 1e-6: averaged_emb_prog /= norm_avg_prog
            
            final_embeddings_to_store_prog.append(averaged_emb_prog)
            final_names_for_new_map_prog.append(name)

    # 4. Rebuild FAISS index and speaker map
    if final_embeddings_to_store_prog:
        if current_dimension <= 0: current_dimension = final_embeddings_to_store_prog[0].shape[0]
        rebuilt_faiss_index_prog = faiss.IndexFlatIP(current_dimension)
        rebuilt_faiss_index_prog.add(np.array(final_embeddings_to_store_prog, dtype=np.float32))
        
        name_to_context_map = {item['name_to_enroll']: item.get('context', 'unknown') for item in enrollment_data_list if 'name_to_enroll' in item}
        new_speaker_map_prog = {i: {"name": name, "context": name_to_context_map.get(name, 'unknown')} for i, name in enumerate(final_names_for_new_map_prog)}
        
        try:
            with db_lock:
                persistence.save_faiss_index(rebuilt_faiss_index_prog, faiss_index_path)
                persistence.save_speaker_map(new_speaker_map_prog, speaker_map_path)
            logger.debug(f"Programmatic enrollment/update successful. Saved rebuilt FAISS index ({rebuilt_faiss_index_prog.ntotal} embs) and map.")
            return True, rebuilt_faiss_index_prog, new_speaker_map_prog
        except Exception as e_save_prog_rebuild:
            logger.error(f"Error saving rebuilt FAISS/map after programmatic enrollment: {e_save_prog_rebuild}", exc_info=True)
            return False, faiss_index, speaker_map
            
    # Handle cases where the database might be emptied
    if faiss_index.ntotal > 0:
        logger.warning("Programmatic enrollment resulted in an empty speaker set. Clearing DB.")
        try:
            empty_idx_prog = faiss.IndexFlatIP(current_dimension if current_dimension > 0 else 1)
            with db_lock:
                persistence.save_faiss_index(empty_idx_prog, faiss_index_path)
                persistence.save_speaker_map({}, speaker_map_path)
            return True, empty_idx_prog, {}
        except Exception as e_save_empty_prog:
            logger.error(f"Error saving empty database after programmatic enrollment: {e_save_empty_prog}", exc_info=True)
            return False, faiss_index, speaker_map

    return False, faiss_index, speaker_map