import threading
import uuid
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from src.matter_manager import get_matter_by_id

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.logger_setup import logger
from src.config_loader import get_config, PROJECT_ROOT
from src.llm_interface import get_llm_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from src.utils.file_io import atomic_json_dump
from src.utils.file_locking import get_lock

# Module-level locks for thread-safe file operations
_EMBEDDING_MODEL_CACHE: Optional[SentenceTransformer] = None

def _get_embedding_model() -> SentenceTransformer:
    """Loads and caches the embedding model instance from the global config."""
    global _EMBEDDING_MODEL_CACHE
    if _EMBEDDING_MODEL_CACHE is None:
        config = get_config()
        embedding_model_config = config.get('llm', {}).get('text_embedding_model', {})
        model_name = embedding_model_config.get('model_name')
        if not model_name:
            raise ValueError("Configuration for 'llm.text_embedding_model.model_name' is missing.")
        logger.info(f"Task helpers: Loading shared embedding model '{model_name}' for the first time.")
        _EMBEDDING_MODEL_CACHE = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL_CACHE


TASK_LOCK = get_lock("tasks_jsonl")
TASK_INDEX_LOCK = get_lock("tasks_faiss_index")  # Lock specifically for FAISS index files

def _get_tasks_file_path() -> Path:
    """Gets the path to tasks.jsonl from the global config."""
    config = get_config()
    file_path_str = config.get('task_intelligence', {}).get('task_data_file', 'data/tasks/tasks.jsonl')
    return PROJECT_ROOT / file_path_str

def _get_tasks_index_path() -> Path:
    """Gets the path to the task FAISS index from the global config."""
    tasks_file_path = _get_tasks_file_path()
    return tasks_file_path.parent / "tasks.index"

def _get_tasks_map_path() -> Path:
    """Gets the path to the task FAISS ID map from the global config."""
    tasks_file_path = _get_tasks_file_path()
    return tasks_file_path.parent / "tasks_map.json"

def _load_tasks() -> List[Dict[str, Any]]:
    """Loads all tasks from the JSONL file, one JSON object per line."""
    tasks_file = _get_tasks_file_path()
    if not tasks_file.exists():
        return []
    
    tasks = []
    try:
        with open(tasks_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        return tasks
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading or parsing {tasks_file.name}: {e}", exc_info=True)
        return []

def _save_tasks(tasks_data: List[Dict[str, Any]]):
    """Saves a list of tasks to the JSONL file atomically."""
    tasks_file = _get_tasks_file_path()
    try:
        tasks_file.parent.mkdir(parents=True, exist_ok=True)
        # We'll write line by line to a temp file
        temp_file_path = tasks_file.with_suffix(tasks_file.suffix + '.tmp')
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            for task in tasks_data:
                f.write(json.dumps(task) + '\n')
        os.rename(temp_file_path, tasks_file)
        logger.debug(f"Atomically saved {len(tasks_data)} tasks to {tasks_file}")
    except Exception as e:
        logger.error(f"Failed to perform atomic save for tasks file {tasks_file}: {e}", exc_info=True)
        if 'temp_file_path' in locals() and temp_file_path.exists():
            try:
                os.remove(temp_file_path)
            except OSError:
                pass
        raise

class TaskIntelligenceManager:
    def __init__(self, config: Dict[str, Any], embedding_model: SentenceTransformer):
        self.config = config
        self.owner_id = self._get_owner_id()
        self.llm = None  # Lazy load LLM
        self.embedding_model = embedding_model
        self.embedding_model_dim = None
        self._initialize_embedding_model_properties()
        self.similarity_threshold = self.config.get('task_intelligence', {}).get('task_similarity_threshold', 0.80)

    def _get_owner_id(self) -> str:
        """Retrieves the system owner's name from config with a fallback."""
        owner = self.config.get('voice_commands', {}).get('user_speaker_name')
        if not owner:
            logger.warning("`voice_commands.user_speaker_name` is not set. Defaulting owner_id to 'SYSTEM_OWNER'.")
            return "SYSTEM_OWNER"
        return owner

    def _initialize_llm(self):
        """Lazy loader for the LLM to avoid instantiation if not needed."""
        if self.llm is None:
            try:
                # Use a specific model configuration key for task extraction
                self.llm = get_llm_chat_model(self.config, 'llm_task_extractor')
                logger.info("TaskIntelligenceManager: LLM for task extraction initialized.")
            except Exception as e:
                logger.error(f"TaskIntelligenceManager: Failed to initialize LLM: {e}", exc_info=True)
                raise

    def _initialize_embedding_model_properties(self):
        """Caches the embedding model's dimension."""
        if self.embedding_model is None:
            raise ValueError("TaskIntelligenceManager requires a valid SentenceTransformer model.")
        self.embedding_model_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"TaskIntelligenceManager initialized with embedding dimension: {self.embedding_model_dim}")

    def _load_or_create_task_index_and_map(self) -> Tuple[faiss.Index, Dict[int, str]]:
        """Thread-safely loads the FAISS index and ID map, or creates them if they don't exist."""
        with TASK_INDEX_LOCK:
            index_path = _get_tasks_index_path()
            map_path = _get_tasks_map_path()
            if index_path.exists() and map_path.exists():
                try:
                    index = faiss.read_index(str(index_path))
                    with open(map_path, 'r', encoding='utf-8') as f:
                        # JSON keys are strings, must convert them back to int
                        str_map = json.load(f)
                        id_map = {int(k): v for k, v in str_map.items()}
                    return index, id_map
                except Exception as e:
                    logger.error(f"Error loading task index/map, will create new ones. Error: {e}")
            
            logger.info("Creating new task FAISS index and map.")
            index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_model_dim))
            id_map = {}
            return index, id_map

    def _save_task_index_and_map(self, index: faiss.Index, id_map: Dict[int, str]):
        """Thread-safely and atomically saves the FAISS index and ID map."""
        with TASK_INDEX_LOCK:
            index_path = _get_tasks_index_path()
            map_path = _get_tasks_map_path()
            
            temp_index_path = index_path.with_suffix(index_path.suffix + '.tmp')
            faiss.write_index(index, str(temp_index_path))
            os.rename(temp_index_path, index_path)

            atomic_json_dump(id_map, map_path)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generates a normalized embedding for a given text."""
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.astype('float32').reshape(1, -1)

    def _add_task_embedding(self, task_id: str, text_to_embed: str):
        """Adds a new task's embedding to the index using a new sequential ID."""
        index, id_map = self._load_or_create_task_index_and_map()
        embedding = self._get_embedding(text_to_embed)
        
        # Use the next available integer as the FAISS ID
        faiss_id = max(id_map.keys()) + 1 if id_map else 0
        
        index.add_with_ids(embedding, np.array([faiss_id]))
        id_map[faiss_id] = task_id
        
        self._save_task_index_and_map(index, id_map)
        logger.info(f"Added embedding for task {task_id} with FAISS ID {faiss_id}.")

    def _update_task_embedding(self, task_id: str, new_text_to_embed: str):
        """Updates the embedding for an existing task."""
        index, id_map = self._load_or_create_task_index_and_map()
        
        faiss_id_to_update = None
        for fid, tid in id_map.items():
            if tid == task_id:
                faiss_id_to_update = fid
                break
        
        if faiss_id_to_update is None:
            logger.warning(f"Could not find FAISS ID for task {task_id} to update embedding. Adding as new.")
            self._add_task_embedding(task_id, new_text_to_embed)
            return

        new_embedding = self._get_embedding(new_text_to_embed)
        index.remove_ids(np.array([faiss_id_to_update]))
        index.add_with_ids(new_embedding, np.array([faiss_id_to_update]))
        
        self._save_task_index_and_map(index, id_map) # Map is unchanged, but save index
        logger.info(f"Updated embedding for task {task_id} at FAISS ID {faiss_id_to_update}.")

    def _remove_task_embedding(self, task_id: str):
        """Removes a task's embedding from the index and map."""
        index, id_map = self._load_or_create_task_index_and_map()

        faiss_id_to_remove = None
        for fid, tid in id_map.items():
            if tid == task_id:
                faiss_id_to_remove = fid
                break
        
        if faiss_id_to_remove is None:
            logger.warning(f"Could not find FAISS ID for task {task_id} to remove. Index may be out of sync.")
            return

        index.remove_ids(np.array([faiss_id_to_remove]))
        del id_map[faiss_id_to_remove]
        
        self._save_task_index_and_map(index, id_map)
        logger.info(f"Removed embedding for task {task_id} from FAISS ID {faiss_id_to_remove}.")

    def _search_similar_tasks(self, text_to_embed: str, matter_id: Optional[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches for similar, non-completed tasks within the same matter."""
        index, id_map = self._load_or_create_task_index_and_map()
        if not id_map or index.ntotal == 0:
            return []

        query_embedding = self._get_embedding(text_to_embed)
        distances, faiss_ids = index.search(query_embedding, min(top_k, index.ntotal))

        similar_task_ids = set()
        for i, score in enumerate(distances[0]):
            if score >= self.similarity_threshold:
                task_uuid = id_map.get(faiss_ids[0][i])
                if task_uuid:
                    similar_task_ids.add(task_uuid)
        
        if not similar_task_ids:
            return []

        all_tasks = _load_tasks()
        candidate_tasks = []
        for task in all_tasks:
            if task.get('task_id') in similar_task_ids and \
               task.get('matter_id') == matter_id and \
               task.get('status') not in ['completed', 'cancelled']:
                candidate_tasks.append(task)
        
        return candidate_tasks

    def _amend_task(self, task_id: str, new_description_text: str, assigner_id: str) -> Optional[Dict[str, Any]]:
        """Appends information to a task's description and adds a version history entry."""
        with TASK_LOCK:
            all_tasks = _load_tasks()
            task_found = False
            updated_task = None
            
            for i, task in enumerate(all_tasks):
                if task.get("task_id") == task_id:
                    # Version History
                    if 'version_history' not in task:
                        task['version_history'] = []
                    version_entry = {
                        "version_id": str(uuid.uuid4()),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "source": "llm_refinement",
                        "change_author": assigner_id,
                        "change_summary": f"Appended new details: '{new_description_text[:100]}...'"
                    }
                    task['version_history'].append(version_entry)
                    
                    # Append to description
                    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
                    task['description'] += f"\n\n[Update on {timestamp} by {assigner_id}]: {new_description_text}"
                    
                    task["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
                    all_tasks[i] = task
                    task_found = True
                    updated_task = task
                    break
            
            if not task_found:
                logger.warning(f"Could not find task with ID {task_id} to amend.")
                return None
            
            _save_tasks(all_tasks)
            logger.info(f"Amended task {task_id} with new information.")
            return updated_task

    def add_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adds a single new task and its embedding to the index."""
        now_iso = datetime.now(timezone.utc).isoformat()
        
        new_task = {
            "task_id": str(uuid.uuid4()),
            "owner_id": self.owner_id,
            "status": "pending_confirmation",
            "created_utc": now_iso,
            "last_updated_utc": now_iso,
            "version_history": [], # Initialize version history
        }
        new_task.update(task_data)

        with TASK_LOCK:
            all_tasks = _load_tasks()
            all_tasks.append(new_task)
            _save_tasks(all_tasks)
        
        # Add embedding to the index
        text_to_embed = f"{new_task.get('title', '')}: {new_task.get('description', '')}"
        self._add_task_embedding(new_task['task_id'], text_to_embed)
        
        logger.info(f"Added new task with ID: {new_task['task_id']}")
        return new_task

    def update_task(self, task_id: str, updates: Dict[str, Any], author: Optional[str] = "System") -> Optional[Dict[str, Any]]:
        """Updates an existing task and its corresponding embedding if necessary."""
        with TASK_LOCK:
            all_tasks = _load_tasks()
            task_found = False
            updated_task = None
            original_text = ""

            for i, task in enumerate(all_tasks):
                if task.get("task_id") == task_id:
                    original_text = f"{task.get('title', '')}: {task.get('description', '')}"

                    changes = []
                    for key, new_value in updates.items():
                        old_value = task.get(key)
                        # Normalize assignee lists for comparison
                        if key == 'assignee_ids' and isinstance(old_value, list) and isinstance(new_value, list):
                            if sorted(old_value) != sorted(new_value):
                                changes.append(f"Assignees changed from {old_value} to {new_value}.")
                        elif old_value != new_value:
                            changes.append(f"'{key}' changed from '{old_value}' to '{new_value}'.")
                    
                    if changes:
                        if 'version_history' not in task or not isinstance(task['version_history'], list):
                            task['version_history'] = []
                        
                        change_summary = " ".join(changes)
                        version_entry = {
                            "version_id": str(uuid.uuid4()),
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "change_author": author,
                            "change_summary": change_summary
                        }
                        task['version_history'].append(version_entry)
                        logger.info(f"Task {task_id}: Logging update by {author}: {change_summary}")

                    task.update(updates)
                    task["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
                    all_tasks[i] = task
                    task_found = True
                    updated_task = task
                    break
            
            if not task_found:
                logger.warning(f"Could not find task with ID {task_id} to update.")
                return None

            _save_tasks(all_tasks)

        # After saving, handle embedding changes outside the lock
        if 'status' in updates and updates['status'] in ['completed', 'cancelled']:
            self._remove_task_embedding(task_id)
        else:
            new_text = f"{updated_task.get('title', '')}: {updated_task.get('description', '')}"
            if new_text != original_text:
                self._update_task_embedding(task_id, new_text)
        
        logger.info(f"Updated task with ID: {task_id}. Updates: {updates}")
        return updated_task

    def get_tasks_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Retrieves all tasks matching a specific status."""
        all_tasks = _load_tasks()
        return [task for task in all_tasks if task.get("status") == status]

    def get_tasks_by_matter(self, matter_id: str) -> List[Dict[str, Any]]:
        """Retrieves all tasks linked to a specific matter_id."""
        all_tasks = _load_tasks()
        return [task for task in all_tasks if task.get("matter_id") == matter_id]

    def get_task_ids_by_source(self, chunk_id: str) -> List[str]:
        """Critical idempotency check: Finds task IDs from a specific chunk_id."""
        all_tasks = _load_tasks()
        found_ids = []
        for task in all_tasks:
            for ref in task.get("source_references", []):
                if ref.get("source_type") == "transcript" and ref.get("chunk_id") == chunk_id:
                    found_ids.append(task["task_id"])
                    break
        return found_ids

    def _format_transcript_for_prompt(self, wlt: List[Dict]) -> str:
        """
        Formats the word-level transcript into a readable, chronological string for the LLM
        by flattening all words and regrouping them by speaker turn.
        """
        # Flatten all words from all segments into a single list
        all_words = []
        if not wlt:
            return ""

        # Robustly handle both segmented and flat lists of words
        first_item = wlt[0]
        if isinstance(first_item, dict) and 'words' in first_item and isinstance(first_item['words'], list):
            # Data is a list of segments: [{'speaker': 'A', 'words': [...]}, ...]
            logger.debug("Formatting a segmented transcript for LLM prompt.")
            for segment in wlt:
                if isinstance(segment, dict) and 'words' in segment and isinstance(segment.get('words'), list):
                    all_words.extend(segment.get('words', []))
        elif isinstance(first_item, dict) and 'word' in first_item:
            # Data is a flat list of words: [{'word': 'Hello', ...}, ...]
            logger.debug("Formatting a flat list of words for LLM prompt.")
            all_words = wlt
        else:
            logger.warning(f"Unrecognized transcript format in wlt. First item: {first_item}")
            return ""

        if not all_words:
            return ""
            
        # Sort by start time to ensure strict chronological order of the conversation
        all_words.sort(key=lambda w: w.get('start', float('inf')))

        formatted_lines = []
        last_speaker = None
        current_line = ""

        for word in all_words:
            speaker = word.get('speaker', 'Unknown')
            word_text = word.get('word', '').strip()

            if not word_text:
                continue

            if speaker != last_speaker:
                if current_line:
                    # Append the previous speaker's full turn
                    formatted_lines.append(current_line)
                # Start a new turn for the new speaker
                current_line = f"{speaker}: {word_text}"
                last_speaker = speaker
            else:
                # Continue the current turn by appending the word
                current_line += f" {word_text}"

        # Append the very last turn after the loop finishes
        if current_line:
            formatted_lines.append(current_line)
        
        return "\n".join(formatted_lines)

    def _load_prompt_template(self) -> PromptTemplate:
        """Loads the prompt from the specified file."""
        prompt_path = PROJECT_ROOT / "src/prompts/extract_tasks_prompt.txt"
        if not prompt_path.exists():
            logger.error(f"Task extraction prompt not found at {prompt_path}")
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        template_string = prompt_path.read_text(encoding='utf-8')
        return PromptTemplate.from_template(template_string)

    def extract_and_save_tasks(self, transcript_chunk_data: Dict) -> List[str]:
        """
        Main method to extract tasks, refine existing ones, or create new ones.
        Returns a list of created or updated task_ids.
        """
        logger.info("[TASK_DEBUG] TaskIntelligenceManager.extract_and_save_tasks called.")
        chunk_id = transcript_chunk_data.get('chunk_id')
        if not chunk_id:
            logger.error("Cannot extract tasks: transcript_chunk_data is missing 'chunk_id'.")
            return []

        # Idempotency Check
        existing_task_ids = self.get_task_ids_by_source(chunk_id)
        if existing_task_ids:
            logger.warning(f"Task extraction for chunk {chunk_id} skipped: Tasks {existing_task_ids} already exist from this source.")
            return existing_task_ids

        try:
            wlt = transcript_chunk_data.get('processed_data', {}).get('word_level_transcript_with_absolute_times', [])
            if not wlt:
                logger.info(f"No transcript found in chunk {chunk_id}. Skipping task extraction.")
                return []

            full_transcript = self._format_transcript_for_prompt(wlt)
            speakers_present = sorted(list({seg.get('speaker', 'Unknown') for seg in wlt}))
            
            matter_id_counts = Counter()
            for segment in wlt:
                for word in segment.get('words', []):
                    word_matter_id = word.get('matter_id')
                    if word_matter_id:
                        matter_id_counts[word_matter_id] += 1
            
            matter_id = None
            matter_name = "Unassigned"

            if matter_id_counts:
                # Find the most common matter_id from the word-level analysis
                dominant_matter_id = matter_id_counts.most_common(1)[0][0]
                matter_id = dominant_matter_id
                matter_details = get_matter_by_id(matter_id)
                if matter_details:
                    matter_name = matter_details.get('name', 'Unassigned')
            else:
                # Fallback to the original method if no words have a matter_id
                active_matter = transcript_chunk_data.get('active_matter_context') or {}
                matter_id = active_matter.get('matter_id')
                matter_name = active_matter.get('matter_name', 'Unassigned')

            # --- Refinement Loop: Search for similar tasks ---
            similar_tasks = self._search_similar_tasks(full_transcript, matter_id)
            existing_tasks_context = "[]"
            if similar_tasks:
                context_list = [{
                    "task_id": t['task_id'],
                    "title": t['title'],
                    "description": t.get('description', '')[-500:] # Last 500 chars
                } for t in similar_tasks]
                existing_tasks_context = json.dumps(context_list, indent=2)

            # --- LLM Inference ---
            self._initialize_llm()
            if not self.llm: return []

            prompt_template = self._load_prompt_template()
            parser = JsonOutputParser()

            def _strip_think_tags(text_or_message: Any) -> str:
                content = getattr(text_or_message, 'content', str(text_or_message))
                return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            chain = prompt_template | self.llm | RunnableLambda(_strip_think_tags) | parser

            llm_response = chain.invoke({
                "transcript": full_transcript,
                "speakers": ", ".join(speakers_present),
                "matter_name": matter_name,
                "system_owner": self.owner_id,
                "existing_tasks_context": existing_tasks_context,
                "format_instructions": parser.get_format_instructions(),
            })
            
            tasks_from_llm = llm_response.get("tasks", []) if isinstance(llm_response, dict) else []
            if not tasks_from_llm:
                 logger.info(f"LLM analysis complete: No tasks found or updated in chunk {chunk_id}.")
                 return []

            # --- Enrichment and Save/Update ---
            processed_task_ids = []
            for task_data in tasks_from_llm:
                if not isinstance(task_data, dict) or "title" not in task_data:
                    logger.warning(f"Skipping malformed task from LLM: {task_data}")
                    continue

                updated_task_id = task_data.get("updated_task_id")
                
                if updated_task_id and updated_task_id in [t['task_id'] for t in similar_tasks]:
                    # This is an update to an existing task
                    amended_task = self._amend_task(
                        task_id=updated_task_id,
                        new_description_text=task_data.get("description", ""),
                        assigner_id=task_data.get("assigner_id", self.owner_id)
                    )
                    if amended_task:
                        text_to_embed = f"{amended_task.get('title', '')}: {amended_task.get('description', '')}"
                        self._update_task_embedding(updated_task_id, text_to_embed)
                        processed_task_ids.append(updated_task_id)
                else:
                    # This is a new task
                    if not task_data.get("description") and task_data.get("title"):
                        task_data["description"] = task_data["title"]
                    task_data["matter_id"] = matter_id
                    task_data["matter_name"] = matter_name
                    task_data["source_references"] = [{"source_type": "transcript", "chunk_id": chunk_id}]
                    if "assignee_ids" not in task_data or not task_data["assignee_ids"]:
                        task_data["assignee_ids"] = [self.owner_id]

                    new_task = self.add_task(task_data)
                    processed_task_ids.append(new_task['task_id'])

            logger.info(f"Successfully processed {len(processed_task_ids)} tasks from chunk {chunk_id}.")
            return processed_task_ids

        except Exception as e:
            logger.error(f"Error during task extraction for chunk {chunk_id}: {e}", exc_info=True)
            return []


# Helper functions that can be called without instantiating the class
def get_tasks_by_matter(matter_id: str) -> List[Dict]:
    """Convenience function to get tasks for a matter."""
    # In a larger app, might use a singleton pattern, but this is fine for now.
    manager = TaskIntelligenceManager(get_config(), _get_embedding_model())
    return manager.get_tasks_by_matter(matter_id)

def get_pending_task_flags() -> List[Dict]:
    """Convenience function to get pending tasks for the UI."""
    manager = TaskIntelligenceManager(get_config(), _get_embedding_model())
    return manager.get_tasks_by_status("pending_confirmation")