import json
import re
from typing import Dict, Any, Optional
from pathlib import Path

from src.llm_interface import get_llm_chat_model, execute_llm_chat_prompt
from src.logger_setup import logger
from src.config_loader import PROJECT_ROOT

def _extract_json_from_response(text: str) -> Optional[str]:
    """
    Extracts a JSON object from a string, handling markdown code blocks and other text.
    """
    if not text:
        return None
        
    # Pattern to find JSON within ```json ... ``` or just ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Fallback pattern to find the first '{' to the last '}' in the string
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)
        
    # If the response is just the JSON, it might not match the above
    if text.strip().startswith('{') and text.strip().endswith('}'):
        return text.strip()
        
    return None

def process_voice_command(
    full_transcript_chunk_text: str,
    user_speaker_name: str,
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Analyzes the final text transcript of an audio chunk to find and parse a voice command.
    """
    vc_config = config.get('voice_commands', {})
    if not vc_config.get('enabled', False):
        return None

    # 1. Filter for user's speech and check for trigger phrase
    user_lines = []
    speaker_pattern = re.compile(f"^\\s*{re.escape(user_speaker_name)}\\s*:", re.IGNORECASE | re.MULTILINE)
    for line in full_transcript_chunk_text.splitlines():
        if speaker_pattern.match(line):
            # Extract just the text part after the "Speaker Name:" part
            user_lines.append(line.split(':', 1)[1].strip())
    
    if not user_lines:
        return None # User did not speak in this chunk

    user_speech_text = " ".join(user_lines).lower()

    if not any(trigger in user_speech_text for trigger in vc_config.get('trigger_phrases', [])):
        return None

    logger.info("Trigger phrase detected in user speech. Analyzing for commands...")

    # 2. Prepare and execute LLM prompt
    try:
        prompt_path = PROJECT_ROOT / vc_config['llm_prompt_template_path']
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        command_definitions_str = json.dumps(vc_config.get('command_definitions', []), indent=2)
        
        prompt = prompt_template.format(
            command_definitions_json=command_definitions_str,
            user_speaker_name=user_speaker_name,
            full_transcript_chunk_text=full_transcript_chunk_text
        )

        llm_model = get_llm_chat_model(config, 'llm_command_parser')
        if not llm_model:
            logger.error("Could not load command parser LLM ('llm_command_parser' section in config).")
            return None
        
        response_text = execute_llm_chat_prompt(prompt, llm_model, model_name="", temperature=0.0)

        if not response_text or response_text.strip().lower() == "null":
            return None

        # 3. FIX: Robustly extract the JSON string before parsing
        json_str = _extract_json_from_response(response_text)
        if not json_str:
            logger.warning(f"Could not extract a valid JSON block from the LLM response. Full response: '{response_text}'")
            return None

        # 4. Parse the extracted, clean JSON string
        command_json = json.loads(json_str)

        # 5. Validate and return
        if 'command_type' in command_json and 'payload' in command_json:
            logger.info(f"Successfully parsed voice command: {command_json['command_type']}")
            return command_json
        else:
            logger.warning(f"LLM response was valid JSON but lacked required keys: {response_text}")
            return None

    except json.JSONDecodeError:
        logger.warning(f"Failed to decode the extracted JSON string. Raw string was: '{json_str}'", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error during voice command interpretation: {e}", exc_info=True)
        return None