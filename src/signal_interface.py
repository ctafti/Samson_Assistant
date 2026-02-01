# File: src/signal_interface.py

import subprocess
import json
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List
import sys 
import psutil
import signal as system_signal

from src.logger_setup import logger

# Default timeout for the signal-cli receive command in seconds
RECEIVE_TIMEOUT_SECONDS = 10  # Increased from 10 to 30 seconds

def check_signal_cli_health(config: Dict[str, Any]) -> bool:
    """
    Check if signal-cli is healthy by running a simple daemon status check.
    
    Args:
        config: The global Samson configuration dictionary.
        
    Returns:
        True if signal-cli appears healthy, False otherwise.
    """
    signal_cfg = config.get('signal', {})
    samson_bot_number = signal_cfg.get('samson_phone_number')
    signal_cli_path = signal_cfg.get('signal_cli_path', 'signal-cli')
    signal_data_path_obj = signal_cfg.get('signal_cli_data_path')
    
    if not all([samson_bot_number, signal_data_path_obj]):
        return False
        
    signal_data_path_str = str(signal_data_path_obj.expanduser().resolve())
    
    # Try a simple listAccounts command with short timeout
    command = [
        str(signal_cli_path),
        "-u", str(samson_bot_number),
        "--config", signal_data_path_str,
        "listAccounts"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False

def kill_existing_signal_cli_processes():
    """
    Kill any existing signal-cli processes that might be hanging.
    Tries to terminate gracefully first, then kills forcefully if needed.
    """
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check both name and command line to be thorough
                is_signal_proc = (proc.info['name'] and 'signal-cli' in proc.info['name'].lower()) or \
                                 (proc.info['cmdline'] and any('signal-cli' in str(arg).lower() for arg in proc.info['cmdline']))

                if is_signal_proc:
                    logger.warning(f"Found existing signal-cli process (PID: {proc.pid}). Attempting graceful termination...")
                    proc.terminate()
                    try:
                        # Wait for the process to terminate.
                        proc.wait(timeout=5)
                        logger.info(f"Process {proc.pid} terminated gracefully.")
                    except psutil.TimeoutExpired:
                        # The process did not terminate within the timeout.
                        logger.warning(f"Process {proc.pid} did not respond to terminate. Forcing kill...")
                        proc.kill()
                        proc.wait(timeout=2) # Wait a moment for the OS to kill it.
                        logger.info(f"Process {proc.pid} forcefully killed.")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process might have disappeared between listing and access, which is fine.
                continue
            except psutil.TimeoutExpired:
                # This would happen on the second proc.wait() if the kill also hangs.
                logger.error(f"Forceful kill of process {proc.pid} also timed out. The process may be unkillable.")
    except Exception as e:
        logger.warning(f"Error while cleaning up signal-cli processes: {e}")

def send_message(
    recipient_phone_number: str,
    message_body: str,
    config: Dict[str, Any],
    attachments: Optional[List[str]] = None,
    retry_count: int = 3
) -> bool:
    """
    Sends a message using signal-cli with retry logic.

    Args:
        recipient_phone_number: The recipient's phone number.
        message_body: The text of the message.
        config: The global Samson configuration dictionary.
        attachments: An optional list of absolute file paths to attach.
        retry_count: Number of retry attempts.

    Returns:
        True if the message was sent successfully, False otherwise.
    """
    signal_cfg = config.get('signal', {})
    samson_bot_number = signal_cfg.get('samson_phone_number')
    signal_cli_path = signal_cfg.get('signal_cli_path', 'signal-cli')
    signal_data_path_obj = signal_cfg.get('signal_cli_data_path')

    if not samson_bot_number:
        logger.error("Signal Send: Samson's bot phone number not configured in 'signal.samson_phone_number'.")
        return False
    if not signal_data_path_obj or not isinstance(signal_data_path_obj, Path):
        logger.error(f"Signal Send: 'signal.signal_cli_data_path' ('{signal_data_path_obj}') not configured or not a Path object.")
        return False
    
    signal_data_path_str = str(signal_data_path_obj.expanduser().resolve())

    command = [
        str(signal_cli_path), 
        "-u", str(samson_bot_number), 
        "--config", signal_data_path_str,
        "send",
        "-m", message_body,
        str(recipient_phone_number) 
    ]

    if attachments:
        for attachment_path_str in attachments:
            attachment_path = Path(attachment_path_str) 
            if attachment_path.exists() and attachment_path.is_file():
                command.extend(["-a", str(attachment_path)])
                logger.info(f"Signal Send: Attaching file: {attachment_path}")
            else:
                logger.warning(f"Signal Send: Attachment path '{attachment_path}' does not exist or is not a file. Skipping attachment.")

    log_command_display = [
        str(signal_cli_path),
        "-u", str(samson_bot_number),
        "--config", signal_data_path_str,
        "send",
        "-m", "[MESSAGE_BODY_REDACTED]", 
        "[RECIPIENT_REDACTED]" 
    ]
    if attachments: log_command_display.extend(["-a", "[ATTACHMENTS_REDACTED]"])

    # Retry logic
    for attempt in range(retry_count):
        if attempt > 0:
            logger.info(f"Signal Send: Retry attempt {attempt + 1}/{retry_count}")
            time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.info(f"Signal Send: Attempting to send. Command structure: {' '.join(log_command_display)}")
        logger.debug(f"Signal Send: Full command for execution (sensitive parts shown): {command[:7]} ...")

        try:
            process = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8', timeout=60)  # Increased timeout
            if process.returncode == 0:
                logger.info(f"Signal Send: Message sent successfully to {str(recipient_phone_number)}.")
                if process.stdout: logger.debug(f"Signal Send STDOUT: {process.stdout.strip()}")
                return True
            else:
                logger.error(f"Signal Send: Failed to send message to {str(recipient_phone_number)}. "
                             f"Return code: {process.returncode} (Attempt {attempt + 1}/{retry_count})")
                if process.stdout: logger.error(f"Signal Send STDOUT: {process.stdout.strip()}")
                
                if process.stderr:
                    stderr_lower = process.stderr.lower()
                    if "unregistereduserexception" in stderr_lower:
                        logger.error(
                            f"Signal Send STDERR (Specific): UnregisteredUserException detected. "
                            f"The bot's number ({samson_bot_number}) or the recipient ({recipient_phone_number}) "
                            f"might be unregistered or experiencing issues with signal-cli linkage."
                        )
                        break  # Don't retry for registration issues
                    elif "ratelimitexception" in stderr_lower:
                        logger.error(
                            f"Signal Send STDERR (Specific): RateLimitException detected. "
                            f"Too many messages sent too quickly. Need to wait."
                        )
                        time.sleep(30)  # Wait longer for rate limits
                    elif "timeoutexception" in stderr_lower:
                        logger.warning(f"Signal Send: Timeout exception detected. Will retry if attempts remain.")
                    logger.error(f"Signal Send STDERR (Full): {process.stderr.strip()}")
                    
                # Don't retry on the last attempt
                if attempt == retry_count - 1:
                    return False
                    
        except FileNotFoundError:
            logger.error(f"Signal Send: signal-cli executable not found at '{signal_cli_path}'.")
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"Signal Send: Command timed out (60s) while trying to send message to {str(recipient_phone_number)} (Attempt {attempt + 1}/{retry_count}).")
            if attempt == retry_count - 1:
                return False
        except Exception as e:
            logger.error(f"Signal Send: An unexpected error occurred (Attempt {attempt + 1}/{retry_count}): {e}", exc_info=True)
            if attempt == retry_count - 1:
                return False

    return False

def _signal_listener_loop(
    callback_on_message: Callable[[Dict[str, Any]], None],
    config: Dict[str, Any],
    shutdown_event: threading.Event
):
    signal_cfg = config.get('signal', {})
    samson_bot_number = signal_cfg.get('samson_phone_number')
    signal_cli_path = signal_cfg.get('signal_cli_path', 'signal-cli')
    signal_data_path_obj = signal_cfg.get('signal_cli_data_path')

    if not samson_bot_number: 
        logger.error("Signal Listener: Samson's bot phone number not configured. Listener cannot start.")
        return
    if not signal_data_path_obj or not isinstance(signal_data_path_obj, Path):
        logger.error(f"Signal Listener: 'signal.signal_cli_data_path' ('{signal_data_path_obj}') not configured or not a Path. Listener cannot start.")
        return
        
    signal_data_path_str = str(signal_data_path_obj.expanduser().resolve())

    logger.info(f"Signal Listener: Thread started for bot {samson_bot_number} using data path {signal_data_path_str}.")

    # Enhanced backoff logic with health checks
    consecutive_specific_rc3_errors = 0
    consecutive_general_errors = 0
    MAX_CONSECUTIVE_SPECIFIC_RC3_ERRORS = 3  # Reduced from 5
    MAX_CONSECUTIVE_GENERAL_ERRORS = 10
    INITIAL_SPECIFIC_RC3_RETRY_DELAY_S = 60  # Increased from 30s
    MAX_SPECIFIC_RC3_RETRY_DELAY_S = 300     # Max delay (5 minutes)
    current_specific_rc3_retry_delay_s = INITIAL_SPECIFIC_RC3_RETRY_DELAY_S
    STANDARD_ERROR_RETRY_DELAY_S = 10  # Increased from 5s
    HEALTH_CHECK_INTERVAL = 300  # Check health every 5 minutes
    last_health_check = 0

    while not shutdown_event.is_set():
        command = [
            str(signal_cli_path),
            "-u", str(samson_bot_number),
            "--config", signal_data_path_str,
            "-o", "json",
            "receive",
            "-t", str(RECEIVE_TIMEOUT_SECONDS)
        ]
        
        process = None 
        try:
            # Kill any existing signal-cli processes before starting
            if consecutive_specific_rc3_errors > 0 or consecutive_general_errors > 5:
                logger.info("Signal Listener: Cleaning up potential zombie processes...")
                kill_existing_signal_cli_processes()
                time.sleep(2)
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, 
                encoding='utf-8',
                bufsize=1,
                preexec_fn=lambda: system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)  # Ignore SIGINT in child
            )

            if process.stdout:
                for line_content in iter(process.stdout.readline, ''): 
                    if shutdown_event.is_set():
                        logger.info("Signal Listener: Shutdown event during stdout read.")
                        break 
                    line_content = line_content.strip()
                    if line_content:
                        try:
                            message_json = json.loads(line_content)
                            if isinstance(message_json, dict) and "envelope" in message_json:
                                logger.debug(f"Signal Listener: Received JSON via Popen: {line_content[:250]}...")
                                callback_on_message(message_json)
                            else:
                                logger.warning(f"Signal Listener: JSON via Popen, but not expected envelope: {line_content[:250]}...")
                        except json.JSONDecodeError as je:
                            logger.error(f"Signal Listener: Popen JSON Decode Error: {je}. Line: '{line_content[:250]}...'")
                        except Exception as cb_e:
                            logger.error(f"Signal Listener: Popen Callback Error for line '{line_content[:250]}...': {cb_e}", exc_info=True)
                
                if not process.stdout.closed:
                    process.stdout.close()

            python_popen_wait_timeout = RECEIVE_TIMEOUT_SECONDS + 15  # Increased buffer
            
            if shutdown_event.is_set():
                if process and process.poll() is None: 
                    logger.info("Signal Listener: Shutdown detected before Popen.wait(). Terminating process.")
                    process.terminate()
                    try:
                        process.wait(timeout=5) 
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Signal Listener: Process (pid {process.pid}) did not terminate gracefully, killing.")
                        process.kill()
                        process.wait() 
            else: 
                try:
                    process.wait(timeout=python_popen_wait_timeout)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Signal Listener: Popen.wait() timed out ({python_popen_wait_timeout}s). Killing signal-cli process (pid {process.pid}).")
                    process.kill() 
                    
                    stderr_output_killed = ""
                    try:
                        _, killed_stderr_data = process.communicate(timeout=5) 
                        if killed_stderr_data:
                             stderr_output_killed = killed_stderr_data.strip()
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Signal Listener: communicate() timed out after killing process (pid {process.pid}). stderr might be incomplete.")
                        process.wait() 
                    except Exception as e_comm:
                        logger.warning(f"Signal Listener: Error during communicate() for killed process (pid {process.pid}): {e_comm}. Forcing wait.")
                        process.wait() 

                    if stderr_output_killed:
                        logger.warning(f"Signal Listener: STDERR from Popen killed process: {stderr_output_killed[:500]}")
                    
                    if process.stderr and not process.stderr.closed:
                        process.stderr.close()
                    
                    consecutive_general_errors += 1
                    if shutdown_event.wait(timeout=1): break
                    continue 

            if process and process.returncode is not None:
                final_stderr_output = ""
                if process.stderr and not process.stderr.closed:
                    final_stderr_output = process.stderr.read().strip()
                    process.stderr.close() 

                if process.returncode == 0:
                    #logger.debug(f"Signal Listener: `signal-cli receive` process ended with RC 0.")
                    if final_stderr_output and "RefreshRecipientsJob - Full CDSI recipients refresh failed" not in final_stderr_output:
                        logger.debug(f"Signal Listener: Popen STDERR (RC 0): {final_stderr_output[:500]}")
                    # Reset error counters on success
                    consecutive_specific_rc3_errors = 0 
                    consecutive_general_errors = 0
                    current_specific_rc3_retry_delay_s = INITIAL_SPECIFIC_RC3_RETRY_DELAY_S
                else: 
                    consecutive_general_errors += 1
                    logger.error(f"Signal Listener: `signal-cli receive` Popen process failed. RC: {process.returncode}")
                    if final_stderr_output:
                        logger.error(f"Signal Listener: Popen STDERR: {final_stderr_output[:500]}")
                    
                    # Enhanced backoff logic for specific RC:3 errors
                    is_specific_rc3_error = False
                    if process.returncode == 3:
                        stderr_lower = final_stderr_output.lower()
                        if any(keyword in stderr_lower for keyword in [
                            "java.util.concurrent.timeoutexception",
                            "closed unexpectedly",
                            "error while checking account",
                            "connection timeout",
                            "read timeout"
                        ]):
                            is_specific_rc3_error = True
                    
                    if is_specific_rc3_error:
                        consecutive_specific_rc3_errors += 1
                        logger.warning(f"Signal Listener: Detected specific signal-cli failure (RC:3, timeout/account check). "
                                       f"Consecutive count: {consecutive_specific_rc3_errors}. "
                                       f"Retrying after {current_specific_rc3_retry_delay_s}s.")
                        
                        if consecutive_specific_rc3_errors >= MAX_CONSECUTIVE_SPECIFIC_RC3_ERRORS:
                            logger.critical(f"Signal Listener: Persistent signal-cli failure (RC:3, timeout/account check) for "
                                            f"{consecutive_specific_rc3_errors} attempts. "
                                            f"Attempting to clean up processes and reset connection...")
                            
                            # Aggressive cleanup
                            kill_existing_signal_cli_processes()
                            time.sleep(10)
                            
                            # Try to check if account is still registered
                            if not check_signal_cli_health(config):
                                logger.error("Signal Listener: Account health check failed after cleanup. "
                                           "Bot may need re-registration or signal-cli may need manual intervention.")
                        
                        if shutdown_event.wait(timeout=current_specific_rc3_retry_delay_s): 
                            logger.info("Signal Listener: Shutdown event during specific RC3 error retry delay.")
                            break 
                        
                        current_specific_rc3_retry_delay_s = min(current_specific_rc3_retry_delay_s * 1.5, MAX_SPECIFIC_RC3_RETRY_DELAY_S)
                    else:
                        # For other errors, reset specific RC3 error tracking and use standard delay
                        consecutive_specific_rc3_errors = 0 
                        current_specific_rc3_retry_delay_s = INITIAL_SPECIFIC_RC3_RETRY_DELAY_S 
                        
                        if consecutive_general_errors >= MAX_CONSECUTIVE_GENERAL_ERRORS:
                            logger.critical(f"Signal Listener: Too many consecutive general errors ({consecutive_general_errors}). "
                                          "Attempting process cleanup...")
                            kill_existing_signal_cli_processes()
                            consecutive_general_errors = 0
                            time.sleep(10)
                        
                        if shutdown_event.wait(timeout=STANDARD_ERROR_RETRY_DELAY_S): 
                            logger.info("Signal Listener: Shutdown event during general error retry delay.")
                            break 

            elif shutdown_event.is_set():
                logger.debug("Signal Listener: Cycle ended due to shutdown event before full Popen completion.")
            
            if process and process.stderr and not process.stderr.closed:
                process.stderr.close()

        except FileNotFoundError:
            logger.error(f"Signal Listener: signal-cli executable not found at '{signal_cli_path}'. Listener stopping.")
            if process: 
                if process.stdout and not process.stdout.closed: process.stdout.close()
                if process.stderr and not process.stderr.closed: process.stderr.close()
            break 
        except Exception as e:
            consecutive_general_errors += 1
            logger.error(f"Signal Listener: Unexpected error in Popen loop: {e}", exc_info=True)
            if process and process.poll() is None: 
                logger.warning("Signal Listener: Killing stray signal-cli process due to unexpected error.")
                process.kill()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Signal Listener: Timeout waiting for stray process to die after kill.")
                    process.wait() 
                except Exception as e_comm_stray:
                    logger.warning(f"Signal Listener: Error communicating with stray process: {e_comm_stray}. Forcing wait.")
                    process.wait()
            if process:
                if process.stdout and not process.stdout.closed: process.stdout.close()
                if process.stderr and not process.stderr.closed: process.stderr.close()
            
            if shutdown_event.wait(timeout=STANDARD_ERROR_RETRY_DELAY_S): break

        if shutdown_event.is_set():
            logger.info("Signal Listener: Shutdown event detected post-cycle, exiting loop.")
            if process and process.poll() is None: 
                logger.info(f"Signal Listener: Terminating active signal-cli process (pid {process.pid}) due to shutdown.")
                process.terminate() 
                try:
                    process.wait(timeout=3) 
                except subprocess.TimeoutExpired:
                    logger.warning(f"Signal Listener: signal-cli (pid {process.pid}) did not terminate gracefully, killing.")
                    process.kill()
                    process.wait()
                finally: 
                    if process.stdout and not process.stdout.closed: process.stdout.close()
                    if process.stderr and not process.stderr.closed: process.stderr.close()
            break
    
    logger.info("Signal Listener: Thread finished.")

def start_signal_listener_thread(
    callback_on_message: Callable[[Dict[str, Any]], None], 
    config: Dict[str, Any],
    shutdown_event: threading.Event
) -> threading.Thread:
    """
    Starts the Signal listener in a separate thread.
    """
    # Clean up any existing processes before starting
    logger.info("Signal Listener: Cleaning up any existing signal-cli processes...")
    kill_existing_signal_cli_processes()
    time.sleep(2)
    
    listener_thread = threading.Thread(
        target=_signal_listener_loop,
        args=(callback_on_message, config, shutdown_event),
        daemon=True 
    )
    listener_thread.start()
    logger.info("Signal Listener: Thread creation initiated and started.") 
    return listener_thread

if __name__ == "__main__":
    sys_path_to_add = str(Path(__file__).resolve().parent.parent)
    shutdown_flag = threading.Event()
    def test_message_callback(msg_data: Dict[str, Any]): 
        logger.info("--- TEST CALLBACK RECEIVED MESSAGE ---")
        logger.info(f"Raw Data: {json.dumps(msg_data, indent=2)[:1000]}...") # Limit log size
        try:
            envelope = msg_data.get("envelope", {}) # noqa: F841
            sender = envelope.get("source") or envelope.get("sourceName") or envelope.get("sourceNumber")
            data_msg = envelope.get("dataMessage", {})
            sync_msg = envelope.get("syncMessage", {}) 
            receipt_msg = envelope.get("receiptMessage", {}) # noqa: F841

            timestamp = data_msg.get("timestamp") or sync_msg.get("sent", {}).get("timestamp") 
            body = data_msg.get("message")

            logger.info(f"From: {sender}")
            logger.info(f"Timestamp: {timestamp}")
            if body: logger.info(f"Body: {body}")
            
            if data_msg.get("groupInfo"):
                logger.info(f"Group: {data_msg.get('groupInfo',{}).get('groupId')}")
            elif sync_msg.get("sent",{}).get("groupInfoV2"): 
                 logger.info(f"Group (v2 from sync): {sync_msg.get('sent',{}).get('groupInfoV2',{}).get('group',{}).get('id')}")

            logger.info("------------------------------------")
        except Exception as e:
            logger.error(f"Error processing message in test callback: {e}", exc_info=True)

    if sys_path_to_add not in sys.path:
        sys.path.insert(0, sys_path_to_add)

    from src.config_loader import get_config, CONFIG_FILE_PATH, ensure_config_exists
    from src.logger_setup import setup_logging 

    print("Signal Interface - Direct Test Mode")

    if not ensure_config_exists(CONFIG_FILE_PATH):
        print(f"Default config created at {CONFIG_FILE_PATH}. "
              "Please CONFIGURE THE SIGNAL SECTION (samson_phone_number, recipient_phone_number, signal_cli_path, signal_cli_data_path) "
              "and ensure signal-cli is set up for the bot number, then run this test again.")
        sys.exit(0)

    try:
        test_config = get_config()
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)

    log_folder_p = test_config.get('paths', {}).get('log_folder')
    log_file_n = test_config.get('paths', {}).get('log_file_name')

    if not isinstance(log_folder_p, Path) or not log_file_n : 
        print("Log folder or file name not found or invalid in config. Exiting test.")
        sys.exit(1)
    setup_logging(log_folder=log_folder_p, log_file_name=log_file_n)

    # Test health check
    logger.info("Testing signal-cli health check...")
    if check_signal_cli_health(test_config):
        logger.info("Signal-cli health check: PASSED")
    else:
        logger.warning("Signal-cli health check: FAILED - Consider checking your signal-cli setup")

    test_recipient = test_config.get('signal', {}).get('recipient_phone_number')
    if not test_recipient:
        logger.error("Test Error: 'signal.recipient_phone_number' not set in config.yaml. Cannot run send test.")
    else:
        logger.info(f"Attempting to send a test message to: {test_recipient}")
        success = send_message(
            recipient_phone_number=test_recipient,
            message_body="Hello from Samson! This is a test message from signal_interface.py (direct test).",
            config=test_config
        )
        if success: logger.info("Test message sent successfully (check Signal).")
        else: logger.error("Failed to send test message.")

        dummy_attachment_path = Path("test_attachment_signal_interface.txt")
        try:
            with open(dummy_attachment_path, "w") as f: f.write("This is a test attachment file from signal_interface.py.")
            
            logger.info(f"Attempting to send a test message with attachment to: {test_recipient}")
            success_attach = send_message(
                recipient_phone_number=test_recipient,
                message_body="Test message with attachment from signal_interface.py.",
                config=test_config,
                attachments=[str(dummy_attachment_path.resolve())]
            )
            if success_attach: logger.info("Test message with attachment sent successfully.")
            else: logger.error("Failed to send test message with attachment.")
        finally:
            if dummy_attachment_path.exists(): 
                dummy_attachment_path.unlink(missing_ok=True)


    if test_config.get('signal',{}).get('samson_phone_number'): 
        logger.info("\nStarting Signal listener test. Send a message to your Samson bot's number.")
        logger.info("Press Ctrl+C to stop the listener test.")

        listener_thread_obj = start_signal_listener_thread(test_message_callback, test_config, shutdown_flag)
        
        join_timeout_listener = RECEIVE_TIMEOUT_SECONDS + 30  # Increased timeout


        try:
            while listener_thread_obj.is_alive():
                time.sleep(1) 
        except KeyboardInterrupt:
            logger.info("Ctrl+C received by main thread. Stopping listener test...")
        finally:
            logger.info("Setting shutdown flag for listener thread...")
            shutdown_flag.set()
            
            logger.info(f"Attempting to join listener thread with timeout: {join_timeout_listener}s")
            listener_thread_obj.join(timeout=join_timeout_listener) 
            if listener_thread_obj.is_alive():
                logger.warning("Listener thread did not terminate gracefully after join timeout.")
            else:
                logger.info("Listener thread has finished.")
            logger.info("Signal listener test concluded.")
    else:
        logger.warning("Skipping listener test as 'signal.samson_phone_number' is not configured.")