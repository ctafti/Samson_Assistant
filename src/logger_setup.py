import logging
import logging.handlers
from pathlib import Path
import os

# This will be the global logger instance for the application
# Other modules can import it: from src.logger_setup import logger
logger = logging.getLogger("SamsonApp")
_test_log_handlers = [] 


class ListLogHandler(logging.Handler):
    """A logging handler that stores log records in a list."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def emit(self, record):
        self.records.append(self.format(record))


def add_test_log_handler(handler: logging.Handler):
    """Attaches a temporary handler to the global logger for testing."""
    global logger, _test_log_handlers
    logger.addHandler(handler)
    _test_log_handlers.append(handler)
    logger.info("Attached new test log handler: %s", handler.__class__.__name__)

def remove_all_test_log_handlers():
    """Removes all temporary test handlers from the global logger."""
    global logger, _test_log_handlers
    for handler in _test_log_handlers:
        logger.info("Removing test log handler: %s", handler.__class__.__name__)
        logger.removeHandler(handler)
    _test_log_handlers.clear()

def setup_logging(log_folder: Path,
                  log_file_name: str,
                  console_level=logging.INFO,
                  file_level=logging.DEBUG,
                  app_name="SamsonApp"): # Allow app_name to be passed for flexibility
    """
    Sets up logging for the application.

    Args:
        log_folder (Path): The folder where log files will be stored.
        log_file_name (str): The name of the log file.
        console_level (int): Logging level for console output.
        file_level (int): Logging level for file output.
        app_name (str): Name for the logger instance.
    """
    global logger
    logger = logging.getLogger(app_name) # Get or create logger with this name

    # Prevent adding multiple handlers if setup_logging is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)  # Set the logger to the lowest level to capture all messages

    # Create log directory if it doesn't exist
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = log_folder / log_file_name

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
    )

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (Rotating)
    # Rotate log file when it reaches 5MB, keep 5 backup files
    try:
        fh = logging.handlers.RotatingFileHandler(
            log_file_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        fh.setLevel(file_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info(f"File logging initialized. Log file: {log_file_path}")
    except PermissionError:
        logger.error(f"Permission denied to write log file at {log_file_path}. File logging disabled.")
    except Exception as e:
        logger.error(f"Failed to initialize file logger at {log_file_path}: {e}. File logging disabled.")


    logger.info(f"Console logging initialized at level {logging.getLevelName(console_level)}.")
    return logger # Return the configured logger instance

# Example usage (for testing within this file if run directly)
if __name__ == "__main__":
    # This test assumes it's run from a context where Samson/ is the root
    # or paths are adjusted.
    current_script_path = Path(__file__).resolve()
    project_root_for_test = current_script_path.parent.parent

    test_log_folder = project_root_for_test / "logs_test_logger_setup"
    test_log_file = "samson_test_logger.log"

    print(f"Setting up test logger. Log folder: {test_log_folder}")
    test_logger_instance = setup_logging(test_log_folder, test_log_file)

    test_logger_instance.debug("This is a DEBUG message for the test logger.")
    test_logger_instance.info("This is an INFO message for the test logger.")
    test_logger_instance.warning("This is a WARNING message for the test logger.")
    test_logger_instance.error("This is an ERROR message for the test logger.")
    test_logger_instance.critical("This is a CRITICAL message for the test logger.")

    print(f"Test log file should be at: {test_log_folder / test_log_file}")
    print("Check the console and the log file for messages.")

    # To use in other modules:
    # from src.logger_setup import logger
    # logger.info("Log message from another module")