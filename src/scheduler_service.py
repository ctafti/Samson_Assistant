# src/scheduler_service.py

import threading
import time
from datetime import datetime
from typing import Dict, Any, Callable
import pytz
import json
import uuid

from .logger_setup import logger
from . import event_manager

class SchedulerService:
    """
    A background service that monitors scheduled events and queues them for execution
    when they become due.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        shutdown_event: threading.Event,
        queue_item_function: Callable
    ):
        """
        Initializes the SchedulerService.

        Inputs:
            config (Dict[str, Any]): The global application configuration.
            shutdown_event (threading.Event): An event to signal the service to stop.
            queue_item_function (Callable): A function to add items to the main worker queue.
                                            (e.g., main_orchestrator._queue_item_with_priority)
        """
        self.config = config
        self.shutdown_event = shutdown_event
        self.queue_item_function = queue_item_function
        
        
        # Get poll interval from config, with a sensible default.
        services_cfg = self.config.get('services', {}).get('scheduler', {})
        self.poll_interval_s = services_cfg.get('poll_interval_s', 15)

    def run(self):
        """
        The main run loop for the service. Continuously checks for due events.
        """
        logger.info(f"Scheduler Service started. Polling every {self.poll_interval_s} seconds.")
        while not self.shutdown_event.is_set():
            try:
                self._check_for_due_events()
            except Exception as e:
                logger.error(f"Error in Scheduler Service run loop: {e}", exc_info=True)
            
            # Wait for the poll interval or until the shutdown event is set.
            self.shutdown_event.wait(self.poll_interval_s)
        
        logger.info("Scheduler Service shutting down.")

    def _check_for_due_events(self):
        """
        Fetches scheduled events, checks if they are due, and queues them for execution.
        """
        # 1. Get all currently scheduled (but not completed/cancelled) events.
        scheduled_items = event_manager.get_all_items(status="SCHEDULED")
        if not scheduled_items:
            return # Nothing to do.

        now_utc = datetime.now(pytz.utc)

        for item in scheduled_items:
            try:
                # 2. Parse the event's start time.
                start_time_str = item.get("start_time_utc")
                if not start_time_str:
                    logger.warning(f"Scheduled item '{item.get('item_id')}' is missing 'start_time_utc'. Skipping.")
                    continue
                
                # fromisoformat requires timezone to be +00:00, not Z.
                event_start_time_utc = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))

                # 3. Check if the event is due.
                if event_start_time_utc <= now_utc:
                    logger.info(f"Event '{item.get('item_id')}' is due. Queuing for execution.")
                    
                    if item.get('item_type') == "MEETING":
                        matter_id = item.get('matter_id')
                        # Get matter name from title for logging, but use ID for the command
                        matter_name = item.get('title', 'Unknown Matter').replace("Switch to Matter: ", "")
                        # Extract environmental_context, defaulting to 'in_person'
                        environmental_context = item.get('metadata', {}).get('environmental_context', 'in_person')

                        if not matter_id:
                            logger.error(f"Cannot process scheduled meeting '{item.get('item_id')}': 'matter_id' is missing.")
                            event_manager.update_item_status(item['item_id'], "ERROR")
                            continue

                        command_to_write = {
                            "type": "APPLY_TIMED_MATTER_UPDATE", # New command type for worker
                            "payload": {
                                "new_matter_id": matter_id,
                                "new_matter_name": matter_name,
                                "environmental_context": environmental_context,
                                "start_time_utc": event_start_time_utc, # Pass the precise datetime object
                                "source": "scheduler",
                                "source_event_id": item['item_id']
                            }
                        }
                        
                        self.queue_item_function(0, command_to_write) # Use the priority queue function
                        logger.info(f"Queued command 'APPLY_TIMED_MATTER_UPDATE' for matter '{matter_name}' (Env: {environmental_context}) for event '{item['item_id']}'.")
                        event_manager.update_item_status(item['item_id'], "QUEUED")

                    elif item.get('item_type') == "TASK_CONFIRMATION":
                        task_id = item.get('metadata', {}).get('task_id')
                        if not task_id:
                            logger.error(f"Cannot process TASK_CONFIRMATION event '{item.get('item_id')}': 'task_id' is missing from metadata.")
                            event_manager.update_item_status(item['item_id'], "ERROR")
                            continue

                        command_to_worker = {
                            "type": "CONFIRM_TASK",
                            "payload": {"task_id": task_id}
                        }
                        self.queue_item_function(1, command_to_worker) # Normal priority
                        logger.info(f"Queued command 'CONFIRM_TASK' for task '{task_id}' from event '{item['item_id']}'.")
                        
                        # Mark as completed so it doesn't run again
                        event_manager.update_item_status(item['item_id'], "COMPLETED")
            except Exception as e:
                logger.error(f"Failed to process scheduled event item {item.get('item_id')}: {e}", exc_info=True)
                # Mark as error to prevent it from being retried constantly.
                event_manager.update_item_status(item['item_id'], "ERROR")
