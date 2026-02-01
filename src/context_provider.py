# src/context_provider.py
from typing import Dict

def get_current_context() -> str:
    """
    Determines the current environmental context (e.g., 'voip', 'in_person').
    STUB: For now, returns a default context. This will later read from a system state,
    e.g., checking if a specific microphone is active.
    """
    # In a real implementation, this would involve logic to check calendar events,
    # active applications, or microphone names.
    return "in_person"