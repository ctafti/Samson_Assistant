# src/folder_monitor.py

import time
from pathlib import Path
from typing import Callable, List, Optional, Set, Dict, Tuple, Any
import threading
import sys
from datetime import datetime, timezone

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from src.logger_setup import logger
from src.daily_log_manager import (
    extract_sequence_number_from_filename,
    parse_duration_to_minutes,
    get_day_start_time,
    # check_if_sequence_processed, # <<< REMOVED as per instruction
    get_highest_processed_sequence
)
from src.config_loader import get_config
from src.signal_interface import send_message

RECENTLY_HANDLED_EVENT_EXPIRY_SECONDS = 5
INITIAL_BATCH_PROMPT_COOLDOWN_S = 300

class AudioFileEventHandler(FileSystemEventHandler):
    def __init__(self,
                 callback_on_new_file: Callable[[Path], None],
                 last_known_processed_sequence: int,
                 file_extensions: Optional[List[str]] = None,
                 config: Optional[Dict[str, any]] = None):
        super().__init__()
        self.callback_on_new_file = callback_on_new_file
        self.config = config if config else get_config()

        if file_extensions:
            self.file_extensions = [ext.lower() if ext.startswith('.') else f".{ext.lower()}"
                                    for ext in file_extensions]
        else:
            self.file_extensions = None

        timings_cfg = self.config.get('timings', {})
        chunk_duration_str = timings_cfg.get('audio_chunk_expected_duration', "10m")
        chunk_duration_minutes = parse_duration_to_minutes(chunk_duration_str, default_minutes=10)
        self.CHUNK_DURATION_SECONDS = chunk_duration_minutes * 60
        self.GAP_TIMEOUT_S = int(self.CHUNK_DURATION_SECONDS * 2.5)
        self.BATCHING_WINDOW_SECONDS = self.CHUNK_DURATION_SECONDS * 0.25

        logger.info(f"AudioFileEventHandler initialized. Watched Extensions: {self.file_extensions or 'all'}.")
        logger.info(f"  - Initial Last Processed Sequence: {last_known_processed_sequence}")
        logger.info(f"  - Chunk Duration: {self.CHUNK_DURATION_SECONDS:.1f}s. Batch Window: {self.BATCHING_WINDOW_SECONDS:.1f}s. Gap Timeout: {self.GAP_TIMEOUT_S}s.")

        self.last_processed_sequence = last_known_processed_sequence
        self.pending_files: Dict[int, Path] = {}
        self.gap_timers: Dict[int, threading.Timer] = {}
        self.recently_seen_by_event_system: Dict[Path, float] = {}
        
        self.batch_window_timer: Optional[threading.Timer] = None
        self.initial_batch_detected_awaiting_time_confirmation: bool = False
        self.initial_batch_prompt_last_sent_time: float = 0.0
        
        # +++ NEW STATE VARIABLE TO FIX THE RACE CONDITION +++
        self.sequential_start_in_progress: bool = False

        self.lock = threading.Lock()

    def _cleanup_stale_seen_events(self):
        now = time.monotonic()
        stale_paths = {path for path, ts in self.recently_seen_by_event_system.items() if now - ts > RECENTLY_HANDLED_EVENT_EXPIRY_SECONDS}
        for path in stale_paths:
            if path in self.recently_seen_by_event_system:
                del self.recently_seen_by_event_system[path]

    def _clear_all_gap_timers(self):
        for seq_num, timer in self.gap_timers.items():
            timer.cancel()
        self.gap_timers.clear()

    def _process_pending_buffer(self):
        """Processes the pending_files buffer sequentially. Assumes lock is held."""
        while True:
            next_expected_seq = self.last_processed_sequence + 1
            if next_expected_seq in self.pending_files:
                successor_seq = next_expected_seq + 1
                if successor_seq in self.pending_files:
                    # Check if the successor_seq had a timer and cancel it
                    if successor_seq in self.gap_timers:
                        timer = self.gap_timers[successor_seq]
                        timer.cancel()
                        del self.gap_timers[successor_seq]
                        logger.info(f"MONITOR: Cancelled gap timer for #{successor_seq} as it has now arrived.")
                    file_to_process = self.pending_files[next_expected_seq]
                    logger.info(f"MONITOR: Successor #{successor_seq} seen. Enqueueing file #{next_expected_seq} ('{file_to_process.name}') for processing.")
                    self.callback_on_new_file(file_to_process)
                    self.last_processed_sequence = next_expected_seq
                    del self.pending_files[next_expected_seq]
                    if next_expected_seq in self.gap_timers:
                        self.gap_timers[next_expected_seq].cancel()
                        del self.gap_timers[next_expected_seq]
                    
                    # NOTE: Do NOT reset sequential_start_in_progress here. It should
                    # persist until the day's start time is confirmed in the daily log.
                    continue
                else:
                    self._start_gap_timer_if_needed(successor_seq)
                    break
            else:
                if any(seq > next_expected_seq for seq in self.pending_files.keys()):
                    self._start_gap_timer_if_needed(next_expected_seq)
                break

    def _on_gap_timeout(self, missing_seq_num: int):
        # <<< MODIFICATION START: Replaced entire function for robust timeout handling >>>
        with self.lock:
            if missing_seq_num not in self.gap_timers:
                return # Timer was already cancelled, do nothing.

            del self.gap_timers[missing_seq_num]
            
            # If we are in the initial batching phase waiting for manual time confirmation, do not process on timeout.
            if self.initial_batch_detected_awaiting_time_confirmation:
                logger.warning(f"MONITOR: GAP TIMEOUT for sequence #{missing_seq_num} occurred, but waiting for SETTIME. Ignoring.")
                return
            
            # The purpose of the timeout is to process the file that was waiting for the timed-out successor.
            predecessor_to_process_seq = missing_seq_num - 1

            logger.warning(f"MONITOR: TIMEOUT waiting for sequence #{missing_seq_num}. Attempting to process predecessor #{predecessor_to_process_seq}.")

            # Check if the predecessor is in the buffer and is the next file we expect to process.
            if predecessor_to_process_seq in self.pending_files and predecessor_to_process_seq == self.last_processed_sequence + 1:
                
                # Process the predecessor since its successor didn't arrive in time.
                file_to_process = self.pending_files[predecessor_to_process_seq]
                self.callback_on_new_file(file_to_process)
                self.last_processed_sequence = predecessor_to_process_seq
                del self.pending_files[predecessor_to_process_seq]

                # Now, determine if the timed-out file was part of a gap or the end of a session.
                has_successor_in_buffer = any(seq > missing_seq_num for seq in self.pending_files.keys())

                if has_successor_in_buffer:
                    # A file with a higher sequence number exists, so the timed-out file was a real gap.
                    # We must advance the state past the gap to allow processing of the successor.
                    logger.warning(f"MONITOR: Successor files exist in buffer. Assuming sequence #{missing_seq_num} is lost and advancing state past it.")
                    self.last_processed_sequence = missing_seq_num
                    # Re-evaluate the buffer, which will now look for `missing_seq_num + 1`.
                    self._process_pending_buffer()
                else:
                    # No successor file is in the buffer. This was the last file of a session.
                    # We are now up-to-date.
                    logger.info(f"MONITOR: Processed #{predecessor_to_process_seq} on timeout. No successors in buffer. Waiting for new files.")
            else:
                # This state can be reached if the predecessor was processed by other means or if the state is inconsistent.
                logger.info(f"MONITOR: GAP TIMEOUT for sequence #{missing_seq_num}, but its predecessor #{predecessor_to_process_seq} was not in a processable state. No action taken. "
                            f"(Last processed: {self.last_processed_sequence}, Pending keys: {list(self.pending_files.keys())})")
        # <<< MODIFICATION END >>>

    def _start_gap_timer_if_needed(self, seq_num_to_wait_for: int):
        """Starts a timer for a file that is missing but expected. Assumes lock is held."""
        if seq_num_to_wait_for not in self.gap_timers:
            logger.info(f"MONITOR: Starting gap timeout ({self.GAP_TIMEOUT_S}s) for missing sequence #{seq_num_to_wait_for}.")
            timer = threading.Timer(self.GAP_TIMEOUT_S, self._on_gap_timeout, args=[seq_num_to_wait_for])
            timer.daemon = True
            self.gap_timers[seq_num_to_wait_for] = timer
            timer.start()

    def _send_settime_prompt(self):
        """Sends the Signal prompt to the user, respecting cooldown. Assumes lock is held."""
        if (time.monotonic() - self.initial_batch_prompt_last_sent_time) > INITIAL_BATCH_PROMPT_COOLDOWN_S:
            recipient = self.config.get('signal', {}).get('recipient_phone_number')
            if recipient:
                logger.info("MONITOR: Sending Signal prompt for initial batch start time.")
                send_message(recipient, "Samson detected an initial batch of audio files for a new day. Please set the recording start time by replying with 'SETTIME HH:MM' or just 'HH:MM' (e.g., 'SETTIME 09:30' or '09:30'). Processing is paused until the time is set.", self.config)
                self.initial_batch_prompt_last_sent_time = time.monotonic()
            else:
                logger.error("MONITOR: Cannot send initial batch prompt - recipient_phone_number not configured.")
        else:
            logger.info("MONITOR: Initial batch prompt cooldown active. Not re-sending Signal message.")

    def _evaluate_batch_buffer(self):
        """Called by a timer to decide if a batch was detected."""
        with self.lock:
            if self.batch_window_timer:
                self.batch_window_timer.cancel()
                self.batch_window_timer = None

            buffer_size = len(self.pending_files)
            logger.info(f"MONITOR: Batch detection window closed. Found {buffer_size} file(s) in the pending buffer.")

            if buffer_size > 1:
                logger.info(f"MONITOR: Confirmed as a batch arrival ({buffer_size} files). Awaiting manual SETTIME command.")
                self.initial_batch_detected_awaiting_time_confirmation = True
                self._send_settime_prompt()
            elif buffer_size == 1:
                logger.info("MONITOR: Confirmed as a normal sequential start. Proceeding with standard N-1 logic.")
                # +++ SETTING THE NEW STATE +++
                self.sequential_start_in_progress = True
                self._process_pending_buffer()

    def _process_event_entrypoint(self, event_path_str: str, event_type: str):
        file_path = Path(event_path_str)
        try: normalized_path = file_path.resolve()
        except FileNotFoundError: return

        with self.lock:
            self._cleanup_stale_seen_events()
            if normalized_path in self.recently_seen_by_event_system: return
            if self.file_extensions and normalized_path.suffix.lower() not in self.file_extensions: return
            if normalized_path.name.startswith(".") or any(normalized_path.name.endswith(s) for s in (".tmp", ".part", ".crdownload", ".syncthing", "~", ".stversions")): return
            
            # <<< MODIFICATION START >>>
            # Check file size. If zero, it might still be writing. Ignore and wait for a modified event.
            try:
                if not normalized_path.exists() or normalized_path.stat().st_size == 0:
                    logger.debug(f"MONITOR: Ignoring event '{event_type}' for '{normalized_path.name}' because it does not exist or has zero size. Waiting for content.")
                    return
            except FileNotFoundError:
                return # File disappeared between event and stat call
            
            seq_num = extract_sequence_number_from_filename(normalized_path.stem)

            # Add a check to see if the sequence number is already in the pending buffer.
            # This makes the handler more robust against multiple events for the same file (e.g., created then modified).
            if seq_num is not None and seq_num in self.pending_files:
                logger.debug(f"MONITOR: Ignoring event '{event_type}' for seq #{seq_num}, as it is already in the pending buffer.")
                return
            # <<< MODIFICATION END >>>

            self.recently_seen_by_event_system[normalized_path] = time.monotonic()
            logger.info(f"MONITOR: Event '{event_type}': Relevant file detected: {normalized_path.name}")
            
            # <<< MODIFICATION: The duplication check (check_if_sequence_processed call and related logic) is removed from here >>>
            # The line "# seq_num was already extracted above" was removed.
            # The line "today_utc = datetime.now(timezone.utc)" was removed.
            # The "if seq_num is not None and check_if_sequence_processed(seq_num, today_utc):" block was removed.

            if self.initial_batch_detected_awaiting_time_confirmation:
                logger.info(f"MONITOR: Awaiting SETTIME confirmation. Buffering '{normalized_path.name}' in main pending queue.")
                if seq_num is not None:
                    if seq_num not in self.pending_files:
                        self.pending_files[seq_num] = normalized_path
                return

            # +++ CHECKING THE NEW STATE VARIABLE +++
            # <<< MODIFICATION: Changed to use datetime.now(timezone.utc) directly as original today_utc was removed >>>
            is_day_start_known = get_day_start_time(datetime.now(timezone.utc)) is not None
            if not is_day_start_known and not self.sequential_start_in_progress:
                if seq_num is not None:
                    if seq_num not in self.pending_files:
                        self.pending_files[seq_num] = normalized_path
                    if self.batch_window_timer is None or not self.batch_window_timer.is_alive():
                        logger.info(f"MONITOR: First file of the day detected ('{normalized_path.name}'). Starting {self.BATCHING_WINDOW_SECONDS:.1f}s batch detection window.")
                        self.batch_window_timer = threading.Timer(self.BATCHING_WINDOW_SECONDS, self._evaluate_batch_buffer)
                        self.batch_window_timer.daemon = True
                        self.batch_window_timer.start()
                    else:
                        logger.debug(f"MONITOR: New file '{normalized_path.name}' arrived during active batching window. Added to pending buffer.")
                else:
                    logger.warning(f"MONITOR: Non-sequenced file '{normalized_path.name}' arrived before day start time was set. Enqueueing directly after 5s delay.")
                    threading.Timer(5.0, self.callback_on_new_file, args=[normalized_path]).start()
                return

            if seq_num is not None:
                if seq_num in self.pending_files: return
                logger.info(f"MONITOR: Day start time is known (or sequential start in progress). Adding file '{normalized_path.name}' (seq #{seq_num}) to main pending buffer.")
                self.pending_files[seq_num] = normalized_path
                self._process_pending_buffer()
            else:
                logger.warning(f"MONITOR: Could not extract sequence number from '{normalized_path.name}'. Enqueueing directly after a 5s delay.")
                threading.Timer(5.0, self.callback_on_new_file, args=[normalized_path]).start()

    def process_initial_batch_after_time_set(self):
        """Called by the orchestrator after a SETTIME command is successfully processed."""
        with self.lock:
            if not self.initial_batch_detected_awaiting_time_confirmation:
                return
            logger.info(f"MONITOR: Day start time confirmed by user. Processing {len(self.pending_files)} buffered files.")
            self.initial_batch_detected_awaiting_time_confirmation = False
            self.sequential_start_in_progress = False # We now have an official start time
            self._process_pending_buffer()

    def on_created(self, event: FileSystemEvent):
        super().on_created(event)
        if event.is_directory: return
        self._process_event_entrypoint(event.src_path, "created")

    def on_moved(self, event: FileSystemEvent):
        super().on_moved(event)
        if event.is_directory: return
        self._process_event_entrypoint(event.dest_path, "moved")

    def on_modified(self, event: FileSystemEvent):
        super().on_modified(event)
        if event.is_directory: return
        # <<< MODIFICATION START >>>
        # Also process 'modified' events to handle cases where files are written to after creation.
        # The debouncing logic in _process_event_entrypoint will prevent duplicate processing.
        self._process_event_entrypoint(event.src_path, "modified")
        # <<< MODIFICATION END >>>

    def stop(self):
        logger.info("MONITOR: Stopping AudioFileEventHandler...")
        with self.lock:
            if self.batch_window_timer and self.batch_window_timer.is_alive():
                self.batch_window_timer.cancel()
            self._clear_all_gap_timers()
            self.pending_files.clear()
        logger.info("MONITOR: AudioFileEventHandler stopped.")


def start_monitoring(folder_to_watch: Path,
                     callback_on_new_file: Callable[[Path], None],
                     last_known_processed_sequence: int,
                     file_extensions: Optional[List[str]] = None,
                     app_config: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Observer], Optional[AudioFileEventHandler]]:
    if not folder_to_watch.exists() or not folder_to_watch.is_dir():
        logger.error(f"Folder to watch '{folder_to_watch}' does not exist or is not a directory.")
        return None, None
    current_config = app_config if app_config is not None else get_config()
    event_handler = AudioFileEventHandler(
        callback_on_new_file,
        last_known_processed_sequence,
        file_extensions,
        config=current_config
    )
    observer = Observer()
    observer.schedule(event_handler, str(folder_to_watch), recursive=False)
    try:
        observer.start()
        logger.info(f"Folder monitoring started for: {folder_to_watch}")
        return observer, event_handler
    except Exception as e:
        logger.error(f"Failed to start folder monitor for {folder_to_watch}: {e}", exc_info=True)
        return None, None


if __name__ == "__main__":
    sys_path_to_add = str(Path(__file__).resolve().parent.parent)
    if sys_path_to_add not in sys.path:
        sys.path.insert(0, sys_path_to_add)
    from src.logger_setup import setup_logging
    try:
        test_config = get_config()
        log_folder_path = Path(test_config['paths']['log_folder'])
        log_file_name = test_config['paths']['log_file_name']
        setup_logging(log_folder=log_folder_path, log_file_name=log_file_name)
        test_monitoring_folder = Path(test_config['paths'].get('monitored_audio_folder'))
        test_monitoring_folder.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error setting up test environment: {e}. Ensure config.yaml is accessible and valid.")
        sys.exit(1)

    def my_test_callback(file_path: Path):
        logger.info(f"++++++++++++ TEST CALLBACK: File enqueued for processing -> {file_path.name} ++++++++++++")

    today_for_test = datetime.now(timezone.utc)
    last_processed_for_test = get_highest_processed_sequence(today_for_test)
    logger.info(f"--- TEST STARTUP: Highest processed sequence for today is #{last_processed_for_test} ---")
    
    test_extensions = ['.aac', '.mp3', '.wav']
    test_observer, test_handler = start_monitoring(test_monitoring_folder, my_test_callback, last_processed_for_test, test_extensions, test_config)
    if not test_observer:
        logger.error("Failed to start monitor in test. Exiting."); sys.exit(1)

    try:
        logger.info(f"Monitoring active. Test by creating files in '{test_monitoring_folder.resolve()}'.")
        logger.info("Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping folder monitor test due to Ctrl+C...")
    finally:
        if test_handler: test_handler.stop()
        if test_observer.is_alive(): test_observer.stop(); test_observer.join()
        logger.info("Folder monitor test stopped.")