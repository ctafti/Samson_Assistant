import threading
import time
import queue
from datetime import datetime, timedelta, timezone, date
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import numpy as np
from src.audio_processing_suite import persistence
from src.audio_processing_suite.speaker_id import update_faiss_embeddings_for_refinement

from src.logger_setup import logger
from src.config_loader import get_config
from src.speaker_profile_manager import (
    get_all_speaker_profiles,
    update_speaker_profile,
    add_dynamic_threshold_feedback_entry,
    create_speaker_profile,
    get_and_clear_pending_embeddings_for_evolution
)
from src.daily_log_manager import get_all_dialogue_for_speaker_id, get_daily_log_data
from src.llm_interface import get_llm_chat_model, execute_llm_chat_prompt

class SpeakerIntelligenceBackgroundService:
    def __init__(self,
                 audio_processing_queue: queue.Queue,
                 global_config: Dict[str, Any],
                 shutdown_event: threading.Event,
                 dynamic_config_store: Dict[str, Any],
                 db_lock: threading.Lock,
                 faiss_index_path_str: str,
                 speaker_map_path_str: str):
        self.config = global_config
        self.speaker_intel_config = self.config.get('speaker_intelligence', {})
        self.audio_processing_queue = audio_processing_queue
        self.shutdown_event = shutdown_event
        self.dynamic_config_store = dynamic_config_store
        self.db_lock = db_lock
        self.faiss_index_path = Path(faiss_index_path_str)
        self.speaker_map_path = Path(speaker_map_path_str)
        self.last_recalculation_check_date: Optional[date] = None

        # Initialize configuration values
        self.enable_automatic_recalculation = self.speaker_intel_config.get('enable_automatic_profile_recalculation', True)
        self.recalculation_time_utc = self.speaker_intel_config.get('profile_recalculation_time_utc', '03:00')
        self.min_segments_threshold = self.speaker_intel_config.get('profile_recalculation_min_new_segments_threshold', 50)

        # Parse recalculation time
        try:
            hour, minute = map(int, self.recalculation_time_utc.split(':'))
            self.recalculation_hour = hour
            self.recalculation_minute = minute
        except (ValueError, AttributeError):
            logger.warning(f"Invalid recalculation_time_utc format: {self.recalculation_time_utc}. Using default 03:00")
            self.recalculation_hour = 3
            self.recalculation_minute = 0

        self.thread = threading.Thread(target=self.run, daemon=True, name="SpeakerIntelligenceService")

        # Last run timestamps for periodic tasks
        self.last_dynamic_threshold_adjustment_utc: Optional[datetime] = None
        self.last_profile_evolution_utc: Optional[datetime] = None
        self.last_role_assignment_utc: Optional[datetime] = None

        # Frequencies from config (parsed to seconds for easier comparison)
        self.dyn_thresh_freq_sec = self.speaker_intel_config.get('dynamic_threshold_adjustment_frequency_hours', 24) * 3600
        self.prof_evolve_freq_sec = self.speaker_intel_config.get('profile_evolution_frequency_days', 7) * 24 * 3600
        self.role_assign_freq_sec = self.speaker_intel_config.get('role_assignment_frequency_days', 14) * 24 * 3600

        logger.info("SpeakerIntelligenceBackgroundService initialized.")

    def start(self):
        if not self.thread.is_alive():
            self.thread.start()
            logger.info("SpeakerIntelligenceBackgroundService thread started.")

    def stop(self):
        logger.info("SpeakerIntelligenceBackgroundService stop requested.")
        self.shutdown_event.set()

    def _should_run_task(self, last_run_utc: Optional[datetime], frequency_seconds: float) -> bool:
        if not last_run_utc:
            return True
        return (datetime.now(timezone.utc) - last_run_utc).total_seconds() >= frequency_seconds

    def _should_run_automatic_recalculation(self) -> bool:
        """Check if automatic profile recalculation should run based on scheduled time."""
        if not self.enable_automatic_recalculation:
            return False

        now_utc = datetime.now(timezone.utc)
        today = now_utc.date()

        # Check if we already ran today
        if self.last_recalculation_check_date == today:
            return False

        # Check if current time is past the scheduled time
        scheduled_time_today = datetime.combine(today, datetime.min.time().replace(
            hour=self.recalculation_hour,
            minute=self.recalculation_minute
        )).replace(tzinfo=timezone.utc)

        return now_utc >= scheduled_time_today

    def _perform_automatic_recalculation(self):
        """Perform automatic profile recalculation for eligible speakers."""
        logger.info("Starting automatic profile recalculation check...")

        try:
            all_profiles = get_all_speaker_profiles()
            if not all_profiles:
                logger.info("No speaker profiles found for automatic recalculation.")
                return

            targets_to_recalculate = []

            # Check each profile for accumulated segments in each context
            for profile in all_profiles:
                faiss_id = profile.get('faiss_id')
                speaker_name = profile.get('name', 'Unknown')

                if faiss_id is None:
                    continue

                # The evolution data is a dictionary keyed by context
                evolution_data = profile.get('segment_embeddings_for_evolution', {})
                if not isinstance(evolution_data, dict):
                    continue

                for context, segments in evolution_data.items():
                    if isinstance(segments, list) and len(segments) >= self.min_segments_threshold:
                        targets_to_recalculate.append({'faiss_id': faiss_id, 'context': context})
                        logger.info(f"Speaker {speaker_name} (ID: {faiss_id}) has {len(segments)} segments for context '{context}', scheduling for recalculation.")

            if targets_to_recalculate:
                # Queue ONE command with all the targets
                recalculation_command = {
                    'type': 'RECALCULATE_SPEAKER_PROFILES',
                    'payload': {'targets': targets_to_recalculate}
                }

                self.audio_processing_queue.put(recalculation_command)
                logger.info(f"Queued automatic recalculation for {len(targets_to_recalculate)} speaker-context pairs.")
            else:
                logger.info("No speakers meet the threshold for automatic recalculation.")

        except Exception as e:
            logger.error(f"Error during automatic profile recalculation: {e}", exc_info=True)
        finally:
            # Mark that we've checked today
            self.last_recalculation_check_date = datetime.now(timezone.utc).date()

    def run(self):
        logger.info("SpeakerIntelligenceBackgroundService run loop started.")

        # Initialize last run times
        self.last_dynamic_threshold_adjustment_utc = None
        self.last_profile_evolution_utc = datetime.now(timezone.utc)
        self.last_role_assignment_utc = datetime.now(timezone.utc)

        while not self.shutdown_event.is_set():
            try:
                now_utc = datetime.now(timezone.utc)

                # Check for automatic profile recalculation
                if self._should_run_automatic_recalculation():
                    try:
                        self._perform_automatic_recalculation()
                    except Exception as e_recalc:
                        logger.error(f"Error during automatic recalculation: {e_recalc}", exc_info=True)

                # 1. Dynamic/Adaptive Confidence Thresholds
                if self.speaker_intel_config.get('enable_dynamic_thresholds', False):
                    if self._should_run_task(self.last_dynamic_threshold_adjustment_utc, self.dyn_thresh_freq_sec):
                        logger.info("Speaker Intelligence: Time to run dynamic threshold adjustment.")
                        try:
                            self._adjust_dynamic_thresholds()
                            self.last_dynamic_threshold_adjustment_utc = now_utc
                        except Exception as e_dyn:
                            logger.error(f"Error during dynamic threshold adjustment: {e_dyn}", exc_info=True)

                # 3. Speaker Role/Trait Assignment via LLM
                if self.speaker_intel_config.get('enable_llm_role_assignment', False):
                    if self._should_run_task(self.last_role_assignment_utc, self.role_assign_freq_sec):
                        logger.info("Speaker Intelligence: Time to run LLM role assignment.")
                        try:
                            self._assign_speaker_roles()
                            self.last_role_assignment_utc = now_utc
                        except Exception as e_role:
                            logger.error(f"Error during LLM role assignment: {e_role}", exc_info=True)

                # Calculate sleep duration until next task
                next_task_times = []

                # Add automatic recalculation check time
                if self.enable_automatic_recalculation:
                    today = now_utc.date()
                    if self.last_recalculation_check_date != today:
                        scheduled_time_today = datetime.combine(today, datetime.min.time().replace(
                            hour=self.recalculation_hour,
                            minute=self.recalculation_minute
                        )).replace(tzinfo=timezone.utc)

                        if now_utc < scheduled_time_today:
                            next_task_times.append(scheduled_time_today)
                        else:
                            # Schedule for tomorrow
                            tomorrow = today + timedelta(days=1)
                            scheduled_time_tomorrow = datetime.combine(tomorrow, datetime.min.time().replace(
                                hour=self.recalculation_hour,
                                minute=self.recalculation_minute
                            )).replace(tzinfo=timezone.utc)
                            next_task_times.append(scheduled_time_tomorrow)

                if self.speaker_intel_config.get('enable_dynamic_thresholds', False):
                    next_task_times.append((self.last_dynamic_threshold_adjustment_utc or now_utc) + timedelta(seconds=self.dyn_thresh_freq_sec))
                if self.speaker_intel_config.get('enable_long_term_profile_evolution', False):
                    next_task_times.append((self.last_profile_evolution_utc or now_utc) + timedelta(seconds=self.prof_evolve_freq_sec))
                if self.speaker_intel_config.get('enable_llm_role_assignment', False):
                    next_task_times.append((self.last_role_assignment_utc or now_utc) + timedelta(seconds=self.role_assign_freq_sec))

                sleep_duration_s = 60 * 5  # Default sleep 5 minutes
                if next_task_times:
                    next_due_time = min(next_task_times)
                    sleep_duration_s = max(1, (next_due_time - now_utc).total_seconds())
                    sleep_duration_s = min(sleep_duration_s, 60 * 15)  # Cap at 15 minutes

                logger.debug(f"Speaker Intelligence: Sleeping for {sleep_duration_s:.2f} seconds.")
                self.shutdown_event.wait(timeout=sleep_duration_s)

            except Exception as e_loop:
                logger.error(f"SpeakerIntelligenceBackgroundService: Unhandled error in main loop: {e_loop}", exc_info=True)
                self.shutdown_event.wait(timeout=60)


    def _adjust_dynamic_thresholds(self):
        """Adjusts speaker-specific, context-aware similarity thresholds based on user feedback."""
        logger.info("Speaker Intelligence: Running dynamic threshold adjustment.")

        # --- 1. Configuration ---
        intel_cfg = self.speaker_intel_config
        min_feedback_entries = intel_cfg.get('dynamic_threshold_min_corrections', 5)
        learning_rate = intel_cfg.get('dynamic_threshold_learning_rate', 0.01)
        adjustment_buffer = intel_cfg.get('dynamic_threshold_adjustment_buffer', 0.02)

        audio_cfg = self.config.get('audio_suite_settings', {})
        global_thresholds = audio_cfg.get('initial_similarity_thresholds', {})
        fallback_threshold = 0.75

        # --- 2. Load profiles and process feedback ---
        all_profiles = get_all_speaker_profiles()
        if not all_profiles:
            logger.info("No speaker profiles found. Skipping threshold adjustment.")
            return

        new_dynamic_thresholds = self.dynamic_config_store.get('dynamic_thresholds', {}).copy()
        profiles_to_clear_feedback = []

        for profile in all_profiles:
            faiss_id = profile.get('faiss_id')
            feedback_entries = profile.get('dynamic_threshold_feedback', [])

            if faiss_id is None or not feedback_entries:
                continue

            # Group feedback by context before processing
            feedback_by_context: Dict[str, Dict[str, List[float]]] = {}
            for entry in feedback_entries:
                context = entry.get('audio_context')
                similarity = entry.get('confidence_score')
                original_id = entry.get('original_speaker_id')
                corrected_id = entry.get('corrected_speaker_id')

                if context is None or similarity is None:
                    logger.debug(f"Skipping feedback entry for profile {faiss_id} due to missing context or similarity.")
                    continue

                is_misidentification = str(original_id) != str(corrected_id)

                if context not in feedback_by_context:
                    feedback_by_context[context] = {'misidentifications': [], 'correct_verifications': []}

                if is_misidentification:
                    feedback_by_context[context]['misidentifications'].append(float(similarity))
                else:
                    feedback_by_context[context]['correct_verifications'].append(float(similarity))

            # Adjust threshold for each context with enough feedback
            speaker_id_str = str(faiss_id)
            for context, scores in feedback_by_context.items():
                misidentification_scores = scores['misidentifications']
                verification_scores = scores['correct_verifications']
                total_feedback = len(misidentification_scores) + len(verification_scores)

                if total_feedback < min_feedback_entries:
                    continue  # Not enough data for this context

                # Mark profile for feedback clearance if at least one context is processed
                profiles_to_clear_feedback.append(faiss_id)

                # Get current threshold, falling back to global, then to a hardcoded default
                current_threshold = (
                    new_dynamic_thresholds.get(speaker_id_str, {}).get(context)
                    or global_thresholds.get(context, fallback_threshold)
                )

                # Calculate target threshold
                target_threshold = current_threshold
                max_bad_score = max(misidentification_scores) if misidentification_scores else -1.0
                min_good_score = min(verification_scores) if verification_scores else 2.0

                if max_bad_score > -1.0 and min_good_score < 2.0:
                    target_threshold = (max_bad_score + min_good_score) / 2.0
                elif max_bad_score > -1.0:
                    target_threshold = max_bad_score + adjustment_buffer
                elif min_good_score < 2.0:
                    target_threshold = min_good_score - adjustment_buffer

                # Apply learning rate and clip to get the new threshold
                new_threshold = float(np.clip(
                    current_threshold * (1 - learning_rate) + target_threshold * learning_rate,
                    0.5, 0.95
                ))

                # Ensure the speaker's entry exists in the dictionary
                if speaker_id_str not in new_dynamic_thresholds:
                    new_dynamic_thresholds[speaker_id_str] = {}

                # Update the threshold if it has changed
                if new_dynamic_thresholds.get(speaker_id_str, {}).get(context) != new_threshold:
                    logger.info(
                        f"Adjusting threshold for speaker ID '{speaker_id_str}' in context '{context}': "
                        f"{current_threshold:.3f} -> {new_threshold:.3f} (target: {target_threshold:.3f})"
                    )
                    new_dynamic_thresholds[speaker_id_str][context] = new_threshold

        # Atomically update the shared dynamic config store
        self.dynamic_config_store['dynamic_thresholds'] = new_dynamic_thresholds
        logger.info("Dynamic thresholds updated in shared store.")

        # Clear processed feedback from profiles
        unique_profiles_to_update = set(profiles_to_clear_feedback)
        if unique_profiles_to_update:
            logger.info(f"Clearing processed feedback for {len(unique_profiles_to_update)} speakers.")
            for profile_faiss_id in unique_profiles_to_update:
                update_speaker_profile(profile_faiss_id, dynamic_threshold_feedback=[])

        logger.info("Dynamic threshold adjustment cycle complete.")

    def _assign_speaker_roles(self):
        """Assign roles to speakers using LLM analysis of their dialogue."""
        logger.info("Speaker Intelligence: Running _assign_speaker_roles...")

        all_profiles = get_all_speaker_profiles()
        if not all_profiles:
            logger.info("No speaker profiles found for role assignment.")
            return

        role_assign_freq_days = self.speaker_intel_config.get('role_assignment_frequency_days', 14)
        llm_model_name_override = self.speaker_intel_config.get('llm_role_assignment_model')

        llm_provider = "ollama"
        llm_temperature = 0.3

        if llm_model_name_override:
            model_to_load_name = llm_model_name_override
            main_llm_cfg = self.config.get('llm', {}).get('main_llm', {})
            llm_temperature = self.speaker_intel_config.get('llm_role_assignment_temperature', main_llm_cfg.get('temperature', 0.3))
        else:
            main_llm_cfg = self.config.get('llm', {}).get('main_llm', {})
            model_to_load_name = main_llm_cfg.get('model_name')
            llm_provider = main_llm_cfg.get('provider', 'ollama')
            llm_temperature = main_llm_cfg.get('temperature', 0.3)

        if not model_to_load_name:
            logger.error("No LLM model specified for role assignment. Skipping.")
            return

        try:
            temp_llm_config_for_role = {
                "provider": llm_provider,
                "model_name": model_to_load_name,
                "temperature": llm_temperature
            }
            llm_model = get_llm_chat_model(self.config, llm_config_dict_override=temp_llm_config_for_role)

        except Exception as e_load_model:
            logger.error(f"Failed to load LLM model '{model_to_load_name}' for role assignment: {e_load_model}", exc_info=True)
            return

        if not llm_model:
            logger.error(f"LLM model '{model_to_load_name}' could not be loaded for role assignment. Skipping.")
            return

        prompt_template = self.speaker_intel_config.get('llm_role_assignment_prompt_template',
            "Based on the following dialogue, what is the primary role of this speaker (e.g., Interviewer, Subject Matter Expert, Assistant, Caller, Host, Guest)? Dialogue: {dialogue_text}")
        now_utc = datetime.now(timezone.utc)

        for profile in all_profiles:
            faiss_id = profile.get('faiss_id')
            speaker_name = profile.get('name')
            if faiss_id is None or speaker_name is None:
                continue

            last_inference_str = profile.get('last_role_inference_utc')
            current_role = profile.get('role')

            assign_role_for_speaker = False
            if current_role is None or last_inference_str is None:
                assign_role_for_speaker = True
                logger.info(f"Speaker {speaker_name} (ID: {faiss_id}) queued for initial role assignment.")
            else:
                try:
                    last_inference_dt = datetime.fromisoformat(last_inference_str)
                    if (now_utc - last_inference_dt).days >= role_assign_freq_days:
                        assign_role_for_speaker = True
                        logger.info(f"Speaker {speaker_name} (ID: {faiss_id}) due for role update.")
                except ValueError:
                    logger.warning(f"Could not parse last_role_inference_utc '{last_inference_str}' for faiss_id {faiss_id}.")
                    assign_role_for_speaker = True

            if not assign_role_for_speaker:
                continue

            # Get dialogue for the speaker
            dialogue_start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
            dialogue_segments = get_all_dialogue_for_speaker_id(speaker_name, dialogue_start_date, now_utc)

            if not dialogue_segments:
                logger.info(f"No dialogue segments found for speaker {speaker_name} (ID: {faiss_id}). Skipping role assignment.")
                update_speaker_profile(faiss_id, last_role_inference_utc=now_utc.isoformat())
                continue

            full_dialogue_text = " ".join([seg.get('text', '') for seg in dialogue_segments]).strip()

            MAX_DIALOGUE_LEN_FOR_LLM = 15000
            if len(full_dialogue_text) > MAX_DIALOGUE_LEN_FOR_LLM:
                logger.warning(f"Dialogue for speaker {speaker_name} is very long ({len(full_dialogue_text)} chars). Truncating.")
                full_dialogue_text = full_dialogue_text[:MAX_DIALOGUE_LEN_FOR_LLM]

            if not full_dialogue_text:
                logger.info(f"Empty dialogue text for speaker {speaker_name} (ID: {faiss_id}). Skipping role assignment.")
                update_speaker_profile(faiss_id, last_role_inference_utc=now_utc.isoformat())
                continue

            prompt = prompt_template.format(dialogue_text=full_dialogue_text)

            logger.info(f"Requesting LLM role assignment for speaker {speaker_name} (ID: {faiss_id}).")

            try:
                inferred_role_raw = execute_llm_chat_prompt(
                    prompt,
                    llm_model,
                    model_name=model_to_load_name,
                    temperature=llm_temperature
                )

                if inferred_role_raw:
                    inferred_role = inferred_role_raw.strip().replace("\n", " ").partition(' ')[0].strip()
                    if len(inferred_role) > 1 and inferred_role.startswith('"') and inferred_role.endswith('"'):
                        inferred_role = inferred_role[1:-1]
                    if len(inferred_role) > 50:
                        inferred_role = inferred_role[:47] + "..."

                    logger.info(f"LLM inferred role for {speaker_name} (ID: {faiss_id}): '{inferred_role}'")
                    update_fields = {
                        "role": inferred_role,
                        "last_role_inference_utc": now_utc.isoformat()
                    }
                    update_speaker_profile(faiss_id, **update_fields)
                else:
                    logger.warning(f"LLM did not return a role for speaker {speaker_name} (ID: {faiss_id}).")
                    update_speaker_profile(faiss_id, last_role_inference_utc=now_utc.isoformat())

            except Exception as e_llm_call:
                logger.error(f"Error during LLM call for role assignment for speaker {speaker_name} (ID: {faiss_id}): {e_llm_call}", exc_info=True)

        logger.info("LLM role assignment cycle finished.")


if __name__ == "__main__":
    # Basic testing structure
    print("Testing SpeakerIntelligenceBackgroundService structure...")

    # Create a test queue
    test_queue = queue.Queue()

    # Create service instance
    service = SpeakerIntelligenceBackgroundService(audio_processing_queue=test_queue)

    print("Service created successfully.")
    print(f"Automatic recalculation enabled: {service.enable_automatic_recalculation}")
    print(f"Recalculation time: {service.recalculation_time_utc}")
    print(f"Minimum segments threshold: {service.min_segments_threshold}")
    print("Test completed.")