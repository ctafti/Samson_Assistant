

import sys
from pathlib import Path
import json
import random
import uuid
import numpy as np
import faiss
import datetime
import shutil
import textwrap
import pytz

# --- Path Setup ---
# Add the project root to the Python path to allow importing from 'src'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.config_loader import get_config, PROJECT_ROOT
    from src.logger_setup import setup_logging, logger
except ImportError as e:
    print("\nERROR: Could not import necessary Samson modules.")
    print("Please run this script from the project's root directory (e.g., 'python tools/setup_test_data.py').")
    print(f"Details: {e}")
    sys.exit(1)

# --- Configuration ---

NUM_SPEAKERS = 5
NUM_MATTERS = 4
NUM_DAYS_OF_DATA = 7
NUM_CHUNKS_PER_DAY_RANGE = (4, 8)
PROBABILITY_NO_MATTER = 0.25

# --- Fake Data Content ---

FAKE_SPEAKER_NAMES = [
    "Dr. Evelyn Reed", "Marcus Thorne", "Agent Kaito", "Javier 'Javi' Rios", "System Administrator"
]
FAKE_MATTERS = [
    {"name": "Project Chimera", "desc": "Investigation into the anomalous energy readings from sub-level 7.", "keywords": ["energy", "anomaly", "sub-level 7", "containment"]},
    {"name": "Quantum Entanglement Comms", "desc": "Developing and testing the FTL communicator prototype.", "keywords": ["FTL", "communication", "quantum", "prototype", "lag"]},
    {"name": "Xenobiology Analysis (Specimen 3B)", "desc": "Analyzing the cellular regeneration of Specimen 3B.", "keywords": ["xenobiology", "specimen 3B", "regeneration", "biopsy"]},
    {"name": "Site Maintenance Review", "desc": "Weekly review of ongoing site maintenance and resource allocation.", "keywords": ["maintenance", "logistics", "power grid", "repairs"]}
]
LOREM_IPSUM_WORDS = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum".split()

# <<< FIX 2: Added a self-contained helper function to get the current date based on the config timezone.

def get_samson_today(config):
    """
    Gets the current date based on the 'assumed_recording_timezone' from the config.
    This is a standalone implementation for this script.
    """
    tz_str = config['timings'].get('assumed_recording_timezone', 'UTC')
    try:
        target_tz = pytz.timezone(tz_str)
    except pytz.UnknownTimeZoneError:
        logger.warning(f"Unknown timezone '{tz_str}' in config. Falling back to UTC.")
        target_tz = pytz.utc
    now_utc = datetime.datetime.now(pytz.utc)
    today_in_tz = now_utc.astimezone(target_tz)
    return today_in_tz.date()

# --- Main Generator Class ---

class TestDataGenerator:
    def __init__(self, config):
        self.config = config
        self.paths = config['paths']
        self.audio_suite_settings = config['audio_suite_settings']
        self.timings = config['timings']
        # <<< FIX 1: Get context_management settings from the correct config section
        self.context_management = config['context_management']
        self.embedding_dim = self.audio_suite_settings.get('embedding_dim', 192) # More robust
        self.speakers = []
        self.matters = []
        self.generated_chunk_ids = []

    def confirm_and_clean_directories(self):
        """Asks for user confirmation and cleans relevant data directories."""
        print("--- Samson Test Data Generator ---")
        print("\nThis script will generate fake data for testing the Samson Cockpit GUI.")
        print("It will DELETE existing data in the following configured directories:")
        
        # --- START MODIFICATION ---
        task_data_path = Path(self.config.get('task_intelligence', {}).get('task_data_file', 'data/tasks/tasks.jsonl'))
        
        dirs_to_clean = [
            self.paths['speaker_db_dir'],
            self.paths['database_folder'], # For master logs and future SQLite DBs
            self.paths['daily_log_folder'],
            self.paths['archived_audio_folder'],
            self.paths['flags_queue_dir'],
            task_data_path.parent  # Add the new directory here
        ]
        # --- END MODIFICATION ---

        for dir_path in dirs_to_clean:
            print(f"  - {dir_path}")

        response = input("\nARE YOU SURE you want to proceed? (yes/no): ").lower()
        if response != 'yes':
            print("Operation cancelled.")
            sys.exit(0)

        for dir_path in dirs_to_clean:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("\nCleaned directories successfully.")

    def generate_speakers_and_matters(self):
        """Generates speaker profiles, FAISS index, speaker map, and matters.json."""
        logger.info("--- Generating Speakers and Matters ---")
        
        # Matters
        for i in range(NUM_MATTERS):
            matter = FAKE_MATTERS[i]
            self.matters.append({
                "matter_id": f"m_{uuid.uuid4().hex[:8]}",
                "name": matter["name"],
                "description": matter["desc"],
                "keywords": matter["keywords"],
                "status": "active",
                "source": "test_data_generator"
            })

        matters_path = self.paths['speaker_db_dir'] / "matters.json"
        with open(matters_path, 'w') as f:
            json.dump(self.matters, f, indent=2)
        logger.info(f"Generated {len(self.matters)} matters in '{matters_path}'.")

        # Speakers
        speaker_profiles = []
        speaker_map = {}
        faiss_embeddings = []
        
        for i in range(NUM_SPEAKERS):
            faiss_id = i
            name = FAKE_SPEAKER_NAMES[i % len(FAKE_SPEAKER_NAMES)]
            
            # Create a plausible random embedding
            embedding = np.random.rand(self.embedding_dim).astype('float32')
            embedding /= np.linalg.norm(embedding)
            faiss_embeddings.append(embedding)

            # Speaker Profile
            # Randomly associate this speaker with 1 or 2 matters
            associated_matters = random.sample(self.matters, k=random.randint(1, 2))
            
            speaker_profiles.append({
                "faiss_id": faiss_id,
                "name": name,
                # --- START MODIFICATION ---
                "role": random.choice(["Owner", "Client", "Colleague", "Assistant", "Team Member", None]),
                # --- END MODIFICATION ---
                "created_utc": (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=random.randint(30, 365))).isoformat(),
                "last_updated_utc": (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=random.randint(1, 29))).isoformat(),
                "profile_last_evolved_utc": None,
                "dynamic_threshold_feedback": [],
                "segment_embeddings_for_evolution": {},
                "lifetime_total_audio_s": random.uniform(500, 5000),
                "associated_matter_ids": [m['matter_id'] for m in associated_matters]
            })
            self.speakers.append(speaker_profiles[-1])
            
            # Speaker Map
            speaker_map[faiss_id] = {"name": name, "context": random.choice(["in_person", "voip"])}

        # Save Speaker Profiles
        profiles_path = self.paths['speaker_db_dir'] / "speaker_profiles.json"
        with open(profiles_path, 'w') as f:
            json.dump(speaker_profiles, f, indent=2)
        logger.info(f"Generated {len(speaker_profiles)} speaker profiles in '{profiles_path}'.")
        
        # Save Speaker Map
        map_path = self.paths['speaker_db_dir'] / self.audio_suite_settings['speaker_map_filename']
        with open(map_path, 'w') as f:
            json.dump(speaker_map, f, indent=2)
        logger.info(f"Generated speaker map with {len(speaker_map)} entries in '{map_path}'.")

        # Save FAISS Index
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(np.array(faiss_embeddings))
        index_path = self.paths['speaker_db_dir'] / self.audio_suite_settings['faiss_index_filename']
        faiss.write_index(index, str(index_path))
        logger.info(f"Generated FAISS index with {index.ntotal} vectors in '{index_path}'.")

    def generate_persistent_state_files(self):
        """Generates context.json and events.jsonl with sample data."""
        logger.info("--- Generating Persistent State Files (Context and Events) ---")
        data_dir = PROJECT_ROOT / "data"
        data_dir.mkdir(exist_ok=True)

        # 1. Create context.json
        if self.matters:
            active_matter = self.matters[0]
            context_data = {
                "matter_id": None,
                "matter_name": None,
                "source": "test_data_generator",
                "last_updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            context_path = data_dir / "context.json"
            with open(context_path, 'w', encoding='utf-8') as f:
                json.dump(context_data, f, indent=2)
            logger.info(f"Generated a null initial context in '{context_path}' (no active matter).")

        events = [] # Ensure the events list is empty.
        
        events_path = data_dir / "events.jsonl"
        # This will now create an empty file, which is the desired behavior.
        with open(events_path, 'w', encoding='utf-8') as f:
            for event in events:
                f.write(json.dumps(event) + '\n')
        logger.info(f"Generated an empty events file in '{events_path}' to ensure no queued matter changes.")

        # 2. Create events.jsonl
        events = []
        if False: # len(self.matters) > 1:
            future_matter = self.matters[1]
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            
            # Event in the past (should be picked up on first scheduler run)
            past_event_time = now_utc - datetime.timedelta(minutes=2)
            events.append({
                "item_id": str(uuid.uuid4()),
                "item_type": "MEETING",
                "status": "SCHEDULED",
                "title": f"Switch to Matter: {future_matter['name']}",
                "description": "Scheduled via test_data_generator.",
                "matter_id": future_matter['matter_id'],
                "start_time_utc": past_event_time.isoformat(),
                "created_utc": (past_event_time - datetime.timedelta(minutes=10)).isoformat(),
                "metadata": { "environmental_context": "voip" },
                "source": {"type": "test_data"}
            })

            # Event in the future
            future_event_time = now_utc + datetime.timedelta(minutes=15)
            events.append({
                "item_id": str(uuid.uuid4()),
                "item_type": "MEETING",
                "status": "SCHEDULED",
                "title": f"Switch to Matter: {self.matters[0]['name']}",
                "description": "Scheduled via test_data_generator.",
                "matter_id": self.matters[0]['matter_id'],
                "start_time_utc": future_event_time.isoformat(),
                "created_utc": now_utc.isoformat(),
                "metadata": { "environmental_context": "in_person" },
                "source": {"type": "test_data"}
            })

        events_path = data_dir / "events.jsonl"
        with open(events_path, 'w', encoding='utf-8') as f:
            for event in events:
                f.write(json.dumps(event) + '\n')
        logger.info(f"Generated {len(events)} scheduled events in '{events_path}'.")

    def generate_daily_data(self):
        """Generates daily logs, flags, archived audio, and master transcripts."""
        logger.info(f"--- Generating Daily Data for the Past {NUM_DAYS_OF_DATA} Days ---")
        
        
        today = get_samson_today(self.config)
        
        for i in range(1, NUM_DAYS_OF_DATA + 1):
            current_date = today - datetime.timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            logger.info(f"Generating data for {date_str}...")

            # Setup for the day
            assumed_tz_str = self.timings.get('assumed_recording_timezone', 'UTC')
            local_tz = pytz.timezone(assumed_tz_str)
            day_start_time_local = local_tz.localize(datetime.datetime.combine(current_date, datetime.time(9, 0, 0)))
            day_start_time_utc = day_start_time_local.astimezone(datetime.timezone.utc)

            daily_log_chunks = {}
            daily_flags = []
            all_words_for_master_log = []
            
            num_chunks = random.randint(NUM_CHUNKS_PER_DAY_RANGE[0], NUM_CHUNKS_PER_DAY_RANGE[1])
            
            for chunk_seq in range(1, num_chunks + 1):
                chunk_id = f"chunk_{date_str.replace('-', '')}_{uuid.uuid4().hex[:8]}"
                # --- START MODIFICATION ---
                self.generated_chunk_ids.append(chunk_id)
                # --- END MODIFICATION ---
                chunk_start_utc = day_start_time_utc + datetime.timedelta(minutes=2 * (chunk_seq - 1))
                
                # Create fake archived audio file
                archive_date_dir = self.paths['archived_audio_folder'] / date_str
                archive_date_dir.mkdir(parents=True, exist_ok=True)
                audio_filename = f"alibi-recording-audio_recordings-{chunk_seq}.aac"
                (archive_date_dir / audio_filename).touch()
                
                # <<< CHANGE START: Handle the new nested structure from the generator
                # Generate dialogue for the chunk
                nested_segments, matter_segments = self._generate_dialogue_for_chunk(chunk_start_utc)
                
                # Create the flat list needed for flags and the master log from the new structure
                flat_word_list = [word for segment in nested_segments for word in segment.get('words', [])]
                all_words_for_master_log.extend(flat_word_list)
                # <<< CHANGE END
                
                # Create a speaker ambiguity flag for the first chunk of the day
                if chunk_seq == 1 and i < 3: # Only for the most recent 3 days
                    # FIX: Pass the flat word list to the flag creation function
                    daily_flags.append(self._create_speaker_flag(flat_word_list, audio_filename, chunk_id, date_str))

                # Create a matter ambiguity flag for the second chunk
                if chunk_seq == 2 and i < 4:
                    matter_flag = self._create_matter_flag(flat_word_list, audio_filename, chunk_id, date_str)
                    if matter_flag:
                        daily_flags.append(matter_flag)

                daily_log_chunks[chunk_id] = {
                    "chunk_id": chunk_id,
                    "entry_creation_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "original_file_name": audio_filename,
                    "file_sequence_number": chunk_seq,
                    "audio_chunk_start_utc": chunk_start_utc.isoformat(),
                    "matter_segments": matter_segments,
                    "processed_data": {

                        "word_level_transcript_with_absolute_times": nested_segments, # <<< CHANGE: Use the correct nested structure for the log
                    },
                    "source_job_output_dir": f"/fake/job/dir/{audio_filename}",
                    "processing_date_utc": date_str
                }

            # Save daily log file.
            log_path = self.paths['daily_log_folder'] / f"{date_str}_samson_log.json"
            daily_log_content = {
                "schema_version": "2.0",
                "day_start_timestamp_utc": day_start_time_utc.isoformat(),
                "chunks": daily_log_chunks,
                "matters": self.matters
            }
            with open(log_path, 'w') as f:
                json.dump(daily_log_content, f, indent=2)

            # Save daily flags to their own correct file, as expected by the system.
            if daily_flags:
                flags_queue_path = self.paths['flags_queue_dir'] / f"{date_str}_flags_queue.json"
                with open(flags_queue_path, 'w') as f:
                    json.dump(daily_flags, f, indent=2)
                logger.info(f"Generated {len(daily_flags)} flags in '{flags_queue_path.name}'.")
            
            # Generate Master Log
            self._generate_master_log(all_words_for_master_log, day_start_time_utc, current_date)

    def _generate_dialogue_for_chunk(self, chunk_start_utc):
        # <<< CHANGE START: The function now builds a list of segments, not a flat word list.
        all_segments = []
        matter_segments = []
        current_time = 0.0
        
        num_turns = random.randint(5, 15)
        
        last_speaker_name = None
        last_matter_id = None
        
        for _ in range(num_turns):
            # Pick a speaker (try not to have the same speaker talk twice in a row)
            speaker = random.choice([s for s in self.speakers if s['name'] != last_speaker_name])
            last_speaker_name = speaker['name']
            
            if random.random() < PROBABILITY_NO_MATTER:
                current_matter_id = None
            else:
                matter = random.choice(self.matters)
                current_matter_id = matter['matter_id']

            # --- Matter Segment Management (for the 'matter_segments' list) ---
            if current_matter_id != last_matter_id:
                if matter_segments:
                    matter_segments[-1]['end_time'] = current_time
                
                if current_matter_id is not None:
                    matter_segments.append({
                        "start_time": current_time,
                        "end_time": -1, # Placeholder
                        "matter_id": current_matter_id
                    })

            # --- Word Generation for this Turn (Segment) ---
            words_for_this_turn = []
            turn_text_parts = []
            turn_start_time = current_time
            
            num_words = random.randint(5, 25)
            for _ in range(num_words):
                word_text = random.choice(LOREM_IPSUM_WORDS)
                turn_text_parts.append(word_text)
                word_duration = random.uniform(0.1, 0.8)
                word_end_time = current_time + word_duration
                
                abs_start = chunk_start_utc + datetime.timedelta(seconds=current_time)
                
                words_for_this_turn.append({
                    "word": word_text,
                    "start": round(current_time, 3),
                    "end": round(word_end_time, 3),
                    "probability": random.uniform(0.85, 0.99),
                    "speaker": speaker['name'],
                    "speaker_name": speaker['name'],
                    "absolute_start_utc": abs_start.isoformat(),
                    "matter_id": current_matter_id # Matter ID is still on the word
                })
                current_time = word_end_time + random.uniform(0.05, 0.2)
            
            # Create the segment object that the real pipeline produces
            new_segment = {
                "start": round(turn_start_time, 3),
                "end": round(current_time, 3),
                "text": " ".join(turn_text_parts),
                "words": words_for_this_turn,
                "speaker": speaker['name']
            }
            all_segments.append(new_segment)
            
            last_matter_id = current_matter_id
        
        if matter_segments:
            matter_segments[-1]['end_time'] = current_time

        # For backward compatibility with flag/master log generation, create the flat list.
        flat_word_list = [word for segment in all_segments for word in segment['words']]

        # The key change is returning 'all_segments' where 'flat_word_list' used to be.
        # But since the caller in `generate_daily_data` expects a flat list for other uses,
        # we now must return the newly created `flat_word_list` for those parts,
        # while ensuring the structure written to the JSON log is the correct nested one.
        # Let's adjust the call site as well.

        # The function now returns the NESTED structure and the matter segments.
        # The flat list will be derived at the call site.
        return all_segments, matter_segments
    # <<< CHANGE END

    def _create_speaker_flag(self, transcript, audio_filename, chunk_id, date_str):
        if not transcript: return {}
        flag_word = random.choice(transcript)

        # Pick two different speakers for the ambiguity flag
        speaker1, speaker2 = random.sample(self.speakers, k=2)

        # Simulate scores that are close enough to be ambiguous
        top_score = random.uniform(0.75, 0.85)
        second_score = top_score - random.uniform(0.01, 0.04)

        return {
            "flag_id": f"FLAG_{date_str.replace('-', '')}_{uuid.uuid4().hex[:12]}",
            "chunk_id": chunk_id,
            "timestamp_logged_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "pending_review",
            "source_file_name": audio_filename,
            "reason_for_flag": "Ambiguous speaker identification",
            "flag_type": "ambiguous_speaker",
            "candidates": [
                {
                    "name": speaker1['name'],
                    "score": round(top_score, 3),
                    "in_context": True  # Simulate this speaker being part of the active matter
                },
                {
                    "name": speaker2['name'],
                    "score": round(second_score, 3),
                    "in_context": False
                }
            ],
            "text_preview": " ".join(w['word'] for w in transcript[5:15]),
            "segment_embedding": np.random.rand(self.embedding_dim).tolist()
        }

    def _create_matter_flag(self, transcript, audio_filename, chunk_id, date_str):
        # Ensure there's enough text and matters to create a realistic flag
        if len(transcript) < 20 or len(self.matters) < 2:
            return None # Return None to indicate no flag could be created

         # --- START OF FIX ---
        # The data structure is now flat, matching the real orchestrator's output.
        # The 'payload' key has been removed.

        start_word = transcript[5]  
        end_word = transcript[19]
        text_snippet = " ".join(w['word'] for w in transcript[5:20])
        start_time = start_word['start']
        end_time = end_word['end']

        matter1, matter2 = random.sample(self.matters, 2)
        top_score = random.uniform(0.86, 0.95)
        second_score = top_score - random.uniform(0.005, 0.029)

        # Construct the flag with the correct flat structure
        return {
            "flag_id": f"FLAG_{date_str.replace('-', '')}_{uuid.uuid4().hex[:12]}",
            "chunk_id": chunk_id,
            "timestamp_logged_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "pending_review",
            "flag_type": "matter_conflict",  
            "source_file_name": audio_filename,
            "text_preview": text_snippet, # Correct key name
            "start_time": start_time,
            "end_time": end_time,
            "conflicting_matters": [
                {"matter_id": matter1['matter_id'], "name": matter1['name'], "score": round(top_score, 4)},
                {"matter_id": matter2['matter_id'], "name": matter2['name'], "score": round(second_score, 4)}
            ]
        }
        # --- END OF FIX ---

        # Simulate scores that would trigger a conflict flag (high confidence, small delta)
        top_score = random.uniform(0.86, 0.95)
        second_score = top_score - random.uniform(0.005, 0.029)

        flag_snippets_dir = self.paths['flag_snippets_dir']
        flag_date_dir = flag_snippets_dir / date_str
        flag_date_dir.mkdir(parents=True, exist_ok=True)

        flag_id = f"FLAG_{date_str.replace('-', '')}_{uuid.uuid4().hex[:12]}"

        # Create the empty snippet file
        (flag_date_dir / f"{flag_id}.wav").touch()

        # Construct the flag with the correct nested payload structure
        return {
            "flag_id": f"FLAG_{date_str.replace('-', '')}_{uuid.uuid4().hex[:12]}",
            "chunk_id": chunk_id,
            "timestamp_logged_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "pending_review",
            "flag_type": "matter_conflict",  
            "source_file_name": audio_filename,
            "payload": {  # The required nested payload object
                "flag_type": "matter_conflict",
                "text": text_snippet,
                "start_time": start_time,
                "end_time": end_time,
                "conflicting_matters": [
                    {"matter_id": matter1['matter_id'], "name": matter1['name'], "score": round(top_score, 4)},
                    {"matter_id": matter2['matter_id'], "name": matter2['name'], "score": round(second_score, 4)}
                ]
            },
        }

    def _generate_master_log(self, all_words, day_start_utc, current_date):
        log_path = self.paths['database_folder'] / f"MASTER_DIALOGUE_{current_date.strftime('%Y-%m-%d')}.txt"
        
        assumed_tz_str = self.timings.get('assumed_recording_timezone', 'UTC')
        display_timezone = pytz.timezone(assumed_tz_str)
        master_log_timestamp_format = self.timings.get('master_log_timestamp_format', "%b%d, %Y - %H:%M")
        line_width = self.audio_suite_settings.get('master_log_line_width', 90)

        lines = []
        day_start_display = day_start_utc.astimezone(display_timezone).strftime(master_log_timestamp_format)
        lines.append(f"## Samson Master Log for {current_date.strftime('%Y-%m-%d')} ##\n")
        lines.append(f"Day recording started around: {day_start_display} ({assumed_tz_str})\n")

        current_speaker = None
        current_turn_text = ""

        for word in sorted(all_words, key=lambda x: x['start']):
            if word['speaker'] != current_speaker:
                if current_turn_text:
                    initial_indent = f"[{current_speaker}]: "
                    subsequent_indent = ' ' * len(initial_indent)
                    wrapped = textwrap.fill(current_turn_text, width=line_width, initial_indent=initial_indent, subsequent_indent=subsequent_indent)
                    lines.append(f"\n{wrapped}\n")
                current_speaker = word['speaker']
                current_turn_text = word['word']
            else:
                current_turn_text += " " + word['word']
        
        # Append last turn
        if current_turn_text:
            initial_indent = f"[{current_speaker}]: "
            subsequent_indent = ' ' * len(initial_indent)
            wrapped = textwrap.fill(current_turn_text, width=line_width, initial_indent=initial_indent, subsequent_indent=subsequent_indent)
            lines.append(f"\n{wrapped}\n")

        with open(log_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        logger.info(f"Generated master log: '{log_path.name}'.")

    def generate_task_data(self):
        """Generates a tasks.jsonl file with sample tasks."""
        logger.info("--- Generating Fake Task Data ---")
        
        tasks_file_path = Path(self.config.get('task_intelligence', {}).get('task_data_file', 'data/tasks/tasks.jsonl'))
        tasks_file_path.parent.mkdir(parents=True, exist_ok=True)
    
        tasks_to_create = []
        num_tasks = 10
    
        # Ensure we have speakers and chunks to reference
        if not self.speakers or not self.generated_chunk_ids:
            logger.warning("No speakers or chunks generated, cannot create realistic tasks.")
            return
    
        owner = next((s for s in self.speakers if s['name'] == "System Administrator"), self.speakers[0])
    
        for i in range(num_tasks):
            status = random.choice(["pending_confirmation", "confirmed", "completed", "cancelled"])
            matter = random.choice(self.matters)
            source_chunk_id = random.choice(self.generated_chunk_ids)
    
            task = {
                "task_id": str(uuid.uuid4()),
                "owner_id": owner['name'],
                "status": status,
                "created_utc": (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=random.randint(1, 48))).isoformat(),
                "last_updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "title": f"Follow-up on {matter['name']} analysis",
                "description": f"This is a sample task generated for testing. It is related to matter '{matter['name']}' and originated from transcript chunk {source_chunk_id[:8]}...",
                "assignee_ids": [owner['name']],
                "matter_id": matter['matter_id'],
                "matter_name": matter['name'],
                "source_references": [
                    {
                        "source_type": "transcript",
                        "chunk_id": source_chunk_id
                    }
                ]
            }
            tasks_to_create.append(task)
        
        with open(tasks_file_path, 'w', encoding='utf-8') as f:
            for task in tasks_to_create:
                f.write(json.dumps(task) + '\n')
                
        logger.info(f"Generated {len(tasks_to_create)} tasks in '{tasks_file_path}'.")

def main():
    """Main function to run the data generator."""
    # Initialize logger and config
    config = get_config()
    setup_logging(
        log_folder=config['paths']['log_folder'],
        log_file_name=config['paths']['log_file_name']
    )
    generator = TestDataGenerator(config)
    generator.confirm_and_clean_directories()
    generator.generate_speakers_and_matters()
    generator.generate_persistent_state_files()
    generator.generate_daily_data()
    generator.generate_task_data()

    print("\n--- ✅ Test Data Generation Complete! ---")
    print("You can now run the Samson Cockpit GUI to view the fake data.")
    print("  streamlit run gui.py")

if __name__ == "__main__":
    main()