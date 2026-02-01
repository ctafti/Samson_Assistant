# src/matter_analysis_service.py
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from langchain_core.embeddings import Embeddings
from src.logger_setup import logger
import logging

class MatterAnalysisService:
    def __init__(self, config: Dict[str, Any], embedding_model: Embeddings):
        self.config = config
        self.embedding_model = embedding_model
        self.matter_embeddings_cache: Dict[str, np.ndarray] = {}
        self._cached_matter_ids: List[str] = []
        
        context_cfg = config.get("context_management", {})
        audio_suite_cfg = config.get("audio_suite_settings", {})
        
        # Use audio_suite_cfg as primary, but fall back to context_cfg for key settings
        self.matter_change_threshold = audio_suite_cfg.get("matter_change_threshold", context_cfg.get("matter_change_threshold", 3))
        self.matter_assignment_min_similarity = audio_suite_cfg.get("matter_assignment_min_similarity", context_cfg.get("matter_assignment_min_similarity", 0.75))
        self.enable_stickiness = audio_suite_cfg.get("enable_matter_context_stickiness", context_cfg.get("enable_matter_context_stickiness", False))
        self.stickiness_bonus = audio_suite_cfg.get("matter_context_stickiness_bonus", context_cfg.get("matter_context_stickiness_bonus", 0.05))
        self.stickiness_override_delta = audio_suite_cfg.get("matter_stickiness_override_delta", context_cfg.get("matter_stickiness_override_delta", 0.08))

        # Keyword Bonus settings with fallback to context_cfg for backward compatibility
        self.enable_keyword_bonus = audio_suite_cfg.get("enable_keyword_bonus", context_cfg.get("enable_keyword_bonus", False))
        self.keyword_bonus_value = float(audio_suite_cfg.get("keyword_bonus_value", context_cfg.get("keyword_bonus_value", 0.1)))

        # Smart Flagging settings with fallback
        self.enable_conflict_flagging = audio_suite_cfg.get("enable_matter_conflict_flagging", context_cfg.get("enable_matter_conflict_flagging", False))
        self.conflict_high_conf_thresh = audio_suite_cfg.get("matter_conflict_high_confidence_threshold", context_cfg.get("matter_conflict_high_confidence_threshold", 0.85))
        self.conflict_delta_thresh = audio_suite_cfg.get("matter_conflict_delta_threshold", context_cfg.get("matter_conflict_delta_threshold", 0.03))

    def _update_matter_embeddings_cache(self, all_matters: List[Dict[str, Any]]) -> None:
        """Computes and caches embeddings for all matters."""
        current_matter_ids = sorted([m['matter_id'] for m in all_matters])
        if current_matter_ids == self._cached_matter_ids: return
        logger.info("Updating matter embeddings cache.")
        self.matter_embeddings_cache.clear()
        self._cached_matter_ids = []
        for matter in all_matters:
            text_to_embed = f"{matter.get('name', '')} {matter.get('description', '')} {' '.join(matter.get('keywords', []))}".strip()
            if text_to_embed:
                self.matter_embeddings_cache[matter['matter_id']] = np.array(self.embedding_model.encode(text_to_embed))
        self._cached_matter_ids = current_matter_ids

    def _group_by_speaker_turn(self, word_level_transcript: List[Dict]) -> List[Dict[str, Any]]:
        """Groups consecutive words from the same speaker into turns."""
        all_words = []
        for segment in word_level_transcript:
            if isinstance(segment, dict) and 'words' in segment and isinstance(segment['words'], list):
                all_words.extend(segment['words'])
        
        if not all_words: return []

        turns, current_turn = [], None
        for word in all_words: # Iterate over the flattened list
            speaker, word_text, end_time = word.get('speaker'), word.get('word', '').strip(), word.get('end')
            if not word_text or end_time is None: continue
            if current_turn and speaker == current_turn['speaker']:
                current_turn['text'] += " " + word_text
                current_turn['end_time'] = end_time
            else:
                if current_turn: turns.append(current_turn)
                start_time = word.get('start')
                if start_time is None: continue
                current_turn = {'speaker': speaker, 'text': word_text, 'start_time': start_time, 'end_time': end_time}
        if current_turn: turns.append(current_turn)
        return turns

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        return 0.0 if norm1 == 0 or norm2 == 0 else np.dot(vec1, vec2) / (norm1 * norm2)

    def analyze_chunk(self, word_level_transcript: List[Dict], all_matters: List[Dict], speaker_profiles: List[Dict], previous_turn_matter_id: Optional[str], active_matter_id: Optional[str]) -> Tuple[List[Dict], List[Dict], Optional[str], Optional[str]]:
        """
        Analyzes a chunk of transcript to produce matter segments and smart flags.
        Returns a tuple of (matter_segments, flags_to_create, last_segment_matter_id, last_word_end_time_utc).
        """
        last_word_end_time_utc: Optional[str] = None
        if not word_level_transcript:
            return [], [], previous_turn_matter_id, last_word_end_time_utc

        # --- NEW: Find the absolute end time of the last word in the chunk ---
        all_words_in_chunk = []
        for segment in word_level_transcript:
            if isinstance(segment, dict) and 'words' in segment:
                all_words_in_chunk.extend(segment.get('words', []))
        
        if all_words_in_chunk:
            # Filter out words that might be missing the required timestamp
            words_with_timestamps = [
                w for w in all_words_in_chunk 
                if isinstance(w, dict) and 'absolute_end_utc' in w and w['absolute_end_utc']
            ]
            if words_with_timestamps:
                last_word = max(words_with_timestamps, key=lambda w: w['absolute_end_utc'])
                last_word_end_time_utc = last_word['absolute_end_utc']
        # --- END NEW ---

        self._update_matter_embeddings_cache(all_matters)
        speaker_turns = self._group_by_speaker_turn(word_level_transcript)
        flags_to_create = []
        matter_details = {m['matter_id']: m for m in all_matters}
        # Pass 1: Context-Aware Turn Classification with Conflict Detection
        for turn in speaker_turns:
            turn_embedding = self.embedding_model.encode(turn['text'])
            matter_scores = {mid: self._cosine_similarity(turn_embedding, m_emb) for mid, m_emb in self.matter_embeddings_cache.items()}

            # Apply a bonus for direct keyword matches, if enabled.
            if self.enable_keyword_bonus and self.keyword_bonus_value > 0:
                turn_text_lower = turn['text'].lower()
                for matter_id in matter_scores.keys():
                    matter_data = matter_details.get(matter_id, {})
                    keywords = matter_data.get('keywords', [])
                    if not keywords:
                        continue

                    for keyword in keywords:
                        # Perform case-insensitive matching
                        if keyword.lower() in turn_text_lower:
                            original_score = matter_scores[matter_id]
                            matter_scores[matter_id] += self.keyword_bonus_value
                            logger.debug(f"[Keyword Bonus] Applied to '{matter_data.get('name')}' for keyword '{keyword}'. Score boosted from {original_score:.3f} to {matter_scores[matter_id]:.3f}.")
                            break # Apply bonus only once per matter, even if multiple keywords match.

            # Use a copy of the raw scores for conflict detection before applying the stickiness bonus.
            scores_for_conflict_check = matter_scores.copy()
            
            # Prioritize the immediately preceding matter, but fall back to the globally active one.
            matter_to_make_sticky = previous_turn_matter_id if previous_turn_matter_id is not None else active_matter_id
            
            apply_bonus = True
            sorted_raw_scores = sorted(scores_for_conflict_check.items(), key=lambda item: item[1], reverse=True)

            # --- NEW LOGGING BLOCK 1: Show Raw Scores (DEBUG Level) ---
            if logger.isEnabledFor(logging.DEBUG) and sorted_raw_scores:
                # This detailed log only appears if your logger is set to DEBUG level.
                turn_text_preview = turn['text'][:80].strip()
                score_lines = "\n".join([
                    f"  - {matter_details.get(mid, {}).get('name', 'Unknown')}: {score:.4f}"
                    for mid, score in sorted_raw_scores[:5] # Log top 5 scores
                ])
                logger.debug(
                    f"\n--- [Matter Scoring] ---\n"
                    f"Turn Text: \"{turn_text_preview}...\"\n"
                    f"Raw Scores:\n{score_lines}\n"
                    f"-------------------------"
                )
            
            # Check if a new matter is a much stronger candidate than the sticky one
            if self.enable_stickiness and matter_to_make_sticky and sorted_raw_scores:
                top_raw_matter_id, top_raw_score = sorted_raw_scores[0]
                
                # If the top raw score belongs to a new matter...
                if top_raw_matter_id != matter_to_make_sticky:
                    sticky_matter_raw_score = scores_for_conflict_check.get(matter_to_make_sticky, 0.0)
                    delta = top_raw_score - sticky_matter_raw_score
                    
                    # ...and the new matter is significantly better, don't apply the stickiness bonus.
                    if delta > self.stickiness_override_delta:
                        apply_bonus = False
                        logger.info(
                            f"[Stickiness Override] New matter '{matter_details.get(top_raw_matter_id, {}).get('name')}' "
                            f"(score: {top_raw_score:.3f}) overrides sticky matter '{matter_details.get(matter_to_make_sticky, {}).get('name')}' "
                            f"(score: {sticky_matter_raw_score:.3f}). Delta {delta:.3f} > Threshold {self.stickiness_override_delta:.3f}."
                        )

            
            if self.enable_stickiness and apply_bonus and matter_to_make_sticky and matter_to_make_sticky in matter_scores:
                original_score = matter_scores[matter_to_make_sticky]
                matter_scores[matter_to_make_sticky] += self.stickiness_bonus
                logger.debug(
                    f"[Stickiness Bonus] Applied to '{matter_details.get(matter_to_make_sticky, {}).get('name')}'. "
                    f"Score boosted from {original_score:.3f} to {matter_scores[matter_to_make_sticky]:.3f}."
                )

            sorted_scores = sorted(matter_scores.items(), key=lambda item: item[1], reverse=True) if matter_scores else []
            is_conflict = False

            # Perform conflict check on the unmodified raw scores.
            sorted_conflict_scores = sorted(scores_for_conflict_check.items(), key=lambda item: item[1], reverse=True)
            if self.enable_conflict_flagging and len(sorted_conflict_scores) >= 2:
                top_matter_id, top_score = sorted_conflict_scores[0]
                second_matter_id, second_score = sorted_conflict_scores[1]
                
                if top_score > self.conflict_high_conf_thresh and second_score > self.conflict_high_conf_thresh and (top_score - second_score) < self.conflict_delta_thresh:
                    is_conflict = True
                    flag_payload = {
                        "flag_type": "matter_conflict",
                        "payload": {
                            "flag_type": "matter_conflict",
                            "text": turn['text'],
                            "start_time": turn['start_time'],
                            "end_time": turn['end_time'],
                            "conflicting_matters": [
                                {"matter_id": top_matter_id, "name": matter_details.get(top_matter_id, {}).get('name'), "score": float(round(top_score, 4))},
                                {"matter_id": second_matter_id, "name": matter_details.get(second_matter_id, {}).get('name'), "score": float(round(second_score, 4))}
                            ]
                        }
                    }
                    flags_to_create.append(flag_payload)
                    turn['matter_id'] = None
                    turn['similarity'] = top_score
            
            if not is_conflict:
                best_matter_id, highest_similarity_score = None, 0.0
                # Prioritize the immediately preceding turn's matter, but fall back to the global active matter.
                context_matter_id = previous_turn_matter_id or active_matter_id

                if sorted_scores:
                    best_match_id, highest_score = sorted_scores[0]
                    # Always store the raw top score for logging and potential UI display.
                    highest_similarity_score = highest_score
                    
                    # Condition 1: A matter is clearly identified above the confidence threshold.
                    if highest_similarity_score >= self.matter_assignment_min_similarity or not apply_bonus:
                        best_matter_id = best_match_id
                    # Condition 2: No matter is a clear semantic winner, so fall back to the active context.
                    elif context_matter_id:
                        best_matter_id = context_matter_id
                    # Condition 3: No clear winner and no active context, so it remains unassigned.
                    else:
                        best_matter_id = None
                
                # If there were no matter scores at all (e.g., empty text), but we have a context, use it.
                elif context_matter_id:
                    best_matter_id = context_matter_id
                
                turn['matter_id'] = best_matter_id
                turn['similarity'] = highest_similarity_score
            final_matter_name = matter_details.get(turn.get('matter_id'), {}).get('name', 'Unassigned')
            logger.debug(
                f"[Turn Decision] Final assignment for turn is '{final_matter_name}' "
                f"with score {turn.get('similarity', 0.0):.3f}."
            )
            
            previous_turn_matter_id = turn.get('matter_id')

        # Pass 2: Boundary Smoothing
        i = 0
        while i < len(speaker_turns):
            current_matter_id, j = speaker_turns[i]['matter_id'], i
            while j < len(speaker_turns) and speaker_turns[j]['matter_id'] == current_matter_id: j += 1
            block_duration = speaker_turns[j - 1]['end_time'] - speaker_turns[i]['start_time']
            if i > 0 and j < len(speaker_turns) and block_duration < self.matter_change_threshold:
                if speaker_turns[i - 1]['matter_id'] == speaker_turns[j]['matter_id']:
                    for k in range(i, j): speaker_turns[k]['matter_id'] = speaker_turns[i - 1]['matter_id']
            i = j

        # Pass 3: Final Segment Generation
        matter_segments = []
        if not speaker_turns:
            return [], flags_to_create, previous_turn_matter_id, last_word_end_time_utc
            
        current_segment = {"start_time": speaker_turns[0]['start_time'], "end_time": speaker_turns[0]['end_time'], "matter_id": speaker_turns[0]['matter_id']}
        for i in range(1, len(speaker_turns)):
            turn = speaker_turns[i]
            if turn['matter_id'] == current_segment['matter_id']:
                current_segment['end_time'] = turn['end_time']
            else:
                matter_segments.append(current_segment)
                current_segment = {"start_time": turn['start_time'], "end_time": turn['end_time'], "matter_id": turn['matter_id']}
        if current_segment: matter_segments.append(current_segment)

        # Determine the final matter ID to pass as context to the next chunk.
        last_segment_matter_id = matter_segments[-1]['matter_id'] if matter_segments else previous_turn_matter_id

        logger.info(f"Generated {len(matter_segments)} final matter segments. Found {len(flags_to_create)} conflicts. Final context for next chunk: '{last_segment_matter_id}'. Last word ended at: {last_word_end_time_utc}")
        
        return matter_segments, flags_to_create, last_segment_matter_id, last_word_end_time_utc