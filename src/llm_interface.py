
import re # Import the regular expression module
import json
from typing import Dict, Any, Optional, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

from src.logger_setup import logger
from src.config_loader import get_config, PROJECT_ROOT
from pathlib import Path


def get_llm_chat_model(
    config: Dict[str, Any],
    llm_config_key: str = "classification_llm"
) -> Optional[BaseChatModel]:
    """
    Initializes and returns a Langchain Chat model based on the application configuration.
    This function supports both 'ollama' and 'lmstudio' providers.

    Args:
        config: The global application configuration dictionary.
        llm_config_key: The key within config['llm'] that specifies which LLM settings to use.

    Returns:
        A Langchain Chat model instance if configuration is valid, otherwise None.
    """
    llm_master_config = config.get('llm')
    if not llm_master_config:
        logger.error("LLM Interface: 'llm' section not found in configuration.")
        return None

    specific_llm_cfg = llm_master_config.get(llm_config_key)
    if not specific_llm_cfg:
        logger.error(f"LLM Interface: Configuration for '{llm_config_key}' not found under 'llm' section.")
        return None

    provider = specific_llm_cfg.get('provider')
    model_name = specific_llm_cfg.get('model_name')
    temperature = specific_llm_cfg.get('temperature', 0.3)
    
    if not provider or not model_name:
        logger.error(f"LLM Interface: 'provider' or 'model_name' missing in '{llm_config_key}' configuration.")
        return None

    logger.info(f"LLM Interface: Initializing model '{model_name}' from provider '{provider}' using config key '{llm_config_key}'.")

    try:
        if provider.lower() == "ollama":
            # For Ollama, base_url is optional and defaults correctly in ChatOllama.
            base_url = specific_llm_cfg.get('base_url') # Can be None
            num_gpu = specific_llm_cfg.get('num_gpu', -1)
            disable_thinking = specific_llm_cfg.get('disable_thinking', False)

            model_params = {
                "model": model_name,
                "temperature": float(temperature),
                "num_gpu": num_gpu
            }
            if base_url:
                model_params["base_url"] = base_url
            if disable_thinking:
                model_params["system"] = "/no_think"

            llm_model = ChatOllama(**model_params)
            logger.info(f"Ollama model '{model_name}' initialized successfully.")
            return llm_model

        elif provider.lower() == "lmstudio":
            base_url = specific_llm_cfg.get('base_url')
            if not base_url:
                logger.error(f"LLM Interface: 'base_url' is required for the 'lmstudio' provider in '{llm_config_key}' configuration.")
                return None

            # LM Studio is OpenAI-compatible, so we can use ChatOllama and just point it to the right server.
            # We must provide a placeholder api_key.
            # LM Studio is OpenAI-compatible, so we must use the ChatOpenAI class.
            model_params = {
                "model_name": model_name,
                "temperature": float(temperature),
                "base_url": base_url,
                "api_key": "lm-studio" # Placeholder key is required
            }
            
            llm_model = ChatOpenAI(**model_params)

            logger.info(f"LM Studio compatible model '{model_name}' initialized successfully (via {base_url}).")
            return llm_model
        
        elif provider.lower() == "google_gemini":
            api_key = specific_llm_cfg.get('api_key')
            if not api_key:
                raise ValueError("'api_key' is required for the 'google_gemini' provider.")
            return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=float(temperature))

        elif provider.lower() == "anthropic":
            api_key = specific_llm_cfg.get('api_key')
            if not api_key:
                raise ValueError("'api_key' is required for the 'anthropic' provider.")
            return ChatAnthropic(model=model_name, anthropic_api_key=api_key, temperature=float(temperature))
            
        else:
            logger.error(f"LLM Interface: Unsupported LLM provider '{provider}' for '{llm_config_key}'. Supported providers: 'ollama', 'lmstudio'.")
            return None

    except ImportError:
        logger.critical("LLM Interface: 'langchain_ollama' or its dependencies not installed. Cannot use local models.")
        return None
    except Exception as e:
        logger.error(f"LLM Interface: Failed to initialize model '{model_name}' from provider '{provider}': {e}", exc_info=True)
        return None


def execute_llm_chat_prompt(
    prompt_text: str,
    llm_model_instance: BaseChatModel, # Expecting a Langchain BaseChatModel instance
    model_name: str, # For logging/tracking, as model is part of instance
    temperature: float, # For logging/tracking, as temp is part of instance
    llm_provider: str = "ollama" # For logging/tracking
) -> Optional[str]:
    """
    Sends a prompt to the specified LLM (Langchain model) and returns the text response.

    Args:
        prompt_text: The full prompt string to send to the LLM.
        llm_model_instance: The loaded Langchain LLM model instance (e.g., ChatOllama).
        model_name: The model name (primarily for logging, assumed to be configured in instance).
        temperature: The temperature (primarily for logging, assumed to be configured in instance).
        llm_provider: The provider of the LLM (primarily for logging).

    Returns:
        The LLM's text response, or None if an error occurs or no response.
    """
    logger.debug(f"Executing LLM chat prompt. Provider: {llm_provider}, Model: {model_name}, Temp: {temperature}, Prompt length: {len(prompt_text)}")

    if not llm_model_instance:
        logger.error("LLM model instance not provided to execute_llm_chat_prompt.")
        return None

    try:
        messages = [HumanMessage(content=prompt_text)]
        response = llm_model_instance.invoke(messages) # Use invoke for Langchain models

        if hasattr(response, 'content') and isinstance(response.content, str):
            llm_response_text = response.content
            logger.debug(f"LLM raw response: {llm_response_text[:200]}...")
            return llm_response_text.strip()
        else:
            logger.warning(f"LLM response format unexpected or empty for model {model_name}. Response: {response}")
            return None
    except Exception as e:
        logger.error(f"Error during LLM API call for model {model_name}: {e}", exc_info=True)
        return None

def summarize_speaker_text(
    dialogue_snippets: List[str],
    llm_model: BaseChatModel,
    max_words: int = 70
) -> str:
    """
    Summarizes a chunk of a transcript using the provided Langchain LLM model.
    The prompt is designed to be simple and direct for smaller models.

    Args:
        dialogue_snippets: A list of text snippets spoken by a single speaker.
        llm_model: An initialized Langchain LLM model (e.g., ChatOllama instance).
        max_words: The desired approximate word count for the summary.

    Returns:
        A narrative summary string, or an error message/empty string on failure.
    """
    if not dialogue_snippets:
        logger.info("Summarize: Input dialogue snippet list is empty. Returning empty summary.")
        return ""
    
    text_to_summarize = "\n".join(dialogue_snippets)

    if not text_to_summarize.strip():
        logger.info("Summarize: Joined dialogue snippets are empty. Returning empty summary.")
        return ""
    if not llm_model:
        logger.error("Summarize: LLM model not provided. Cannot generate summary.")
        return "Error: LLM model not available for summarization."

    # Updated prompt to inform the LLM it's receiving a transcript chunk.
    prompt_template = (
        f"You will receive a chunk from a conversation transcript. "
        f"Concisely summarize the speaker's role and intent based *only* on this chunk. "
        f"The summary should be a single block of text, around {max_words} words, "
        f"with no extra preamble or labels.\n\n"
        f"TRANSCRIPT CHUNK:\n"
        f"\"\"\"\n{text_to_summarize}\n\"\"\"\n\n"
        f"SUMMARY:"
    )

    logger.debug(f"Summarize: Sending {len(dialogue_snippets)} snippets (total length {len(text_to_summarize)}) to LLM for summary. Max words: {max_words}.")
    logger.debug(f"Summarize: Prompt used (first 300 chars):\n{prompt_template[:300]}...")

    try:
        messages = [HumanMessage(content=prompt_template)]
        response = llm_model.invoke(messages)

        if hasattr(response, 'content') and isinstance(response.content, str):
            # The LLM's direct output is the desired summary.
            summary = response.content.strip()
            summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
            logger.info(f"Summarize: LLM generated summary: '{summary}'")
            return summary
        else:
            logger.error(f"Summarize: LLM response was not in the expected format (string content). Response: {response}")
            return "Error: Unexpected LLM response format during summarization."

    except Exception as e:
        logger.error(f"Summarize: An error occurred while invoking the LLM for summarization: {e}", exc_info=True)
        return f"Error: LLM summarization failed ({type(e).__name__})."


def generate_matter_details(dialogue_text: str, llm_model: BaseChatModel) -> Optional[Dict[str, Any]]:
    """
    Uses an LLM to generate a name, description, and keywords for a dialogue chunk.

    Args:
        dialogue_text: The text of the conversation to analyze.
        llm_model: An initialized Langchain LLM model.

    Returns:
        A dictionary with 'name', 'description', and 'keywords', or None on failure.
    """
    prompt = f"""
Analyze the following conversation dialogue and identify the primary topic or "matter" being discussed.
Based on your analysis, generate a concise name for this matter, a one-sentence description, and a list of 3-5 relevant keywords.

Your response MUST be a valid JSON object with the following structure:
{{
  "name": "Example Matter Name",
  "description": "A one-sentence summary of the matter.",
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}

Do not include any other text, explanations, or markdown formatting like ```json before or after the JSON object.

DIALOGUE:
---
{dialogue_text}
---
"""
    logger.debug(f"Generating matter details for dialogue of length {len(dialogue_text)}")
    try:
        response = llm_model.invoke([HumanMessage(content=prompt)])
        response_text = response.content
        if not response_text:
            logger.warning("LLM returned an empty response for matter generation.")
            return None

        # Attempt to parse the JSON directly
        try:
            parsed_json = json.loads(response_text)
            required_keys = {'name', 'description', 'keywords'}
            if required_keys.issubset(parsed_json.keys()):
                logger.info(f"Successfully generated and parsed matter details: {parsed_json.get('name')}")
                return parsed_json
            else:
                logger.warning(f"LLM generated JSON is missing required keys. Found: {list(parsed_json.keys())}")
                return None
        except json.JSONDecodeError:
            logger.warning("Failed to decode LLM response as JSON directly. Attempting regex extraction.")
            # If direct parsing fails, try to extract JSON from markdown or other text
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    required_keys = {'name', 'description', 'keywords'}
                    if required_keys.issubset(parsed_json.keys()):
                        logger.info(f"Successfully generated and parsed matter details from extracted JSON: {parsed_json.get('name')}")
                        return parsed_json
                    else:
                        logger.warning(f"LLM generated JSON (extracted) is missing required keys. Found: {list(parsed_json.keys())}")
                        return None
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode even the extracted JSON string from LLM response. Extracted: {json_str[:200]}...")
                    return None
            else:
                logger.error(f"Could not find any JSON object in the LLM response. Response: {response_text[:200]}...")
                return None

    except Exception as e:
        logger.error(f"An exception occurred during LLM call for matter generation: {e}", exc_info=True)
        return None
    return None


def extract_structured_entities(text: str, llm_model: BaseChatModel) -> Optional[Dict[str, Any]]:
    """
    Parses natural language text to find a potential new matter and its start time
    by loading a prompt template from the configuration file.

    Args:
        text: The input text from the user (e.g., from a Signal message).
        llm_model: An initialized Langchain LLM model.

    Returns:
        A dictionary with 'matter_name' and 'start_time' ("now" or ISO 8601 string), or None.
    """
    config = get_config()
    prompt_path_str = config.get('signal', {}).get('unstructured_command_prompt_path')

    if not prompt_path_str:
        logger.error("Path to signal command prompt not found in config ('signal.unstructured_command_prompt_path').")
        return None

    try:
        prompt_path = PROJECT_ROOT / prompt_path_str
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        # Use str.replace() instead of .format() to avoid issues with JSON examples in the prompt.
        # This is a safer, literal replacement that won't misinterpret curly braces.
        prompt = prompt_template.replace('{text}', text)
        
    except FileNotFoundError:
        logger.error(f"Signal command prompt template not found at path: {prompt_path_str}")
        return None
    except Exception as e:
        logger.error(f"Failed to load or format the signal command prompt: {e}", exc_info=True)
        return None

    logger.debug(f"Extracting structured entities from text of length {len(text)}")
    try:
        response_text = execute_llm_chat_prompt(prompt, llm_model, "", 0.0)
        if not response_text or response_text.strip().lower() == "null":
            return None
        
        # The LLM often includes a <think> block before the JSON. Remove it first.
        # Then, find the start and end of the JSON object in the cleaned text.
        cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
        
        start_index = cleaned_response.find('{')
        end_index = cleaned_response.rfind('}')
        
        if start_index == -1 or end_index == -1 or end_index < start_index:
            logger.error(f"Could not find a valid JSON object in the cleaned LLM response. Response: {cleaned_response[:300]}...")
            return None
        
        json_str = cleaned_response[start_index : end_index + 1]
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode extracted JSON from LLM response even after cleaning. Extracted: '{json_str[:300]}...'")
            return None

        # Basic validation: Only require command_type. The caller will validate specifics.
        if isinstance(data, dict) and 'command_type' in data:
            return data
        else:
            logger.warning(f"LLM response parsed to JSON, but missing required key 'command_type'. Data: {data}")
            return None
    except Exception as e:
        logger.error(f"Error during structured entity extraction: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # --- Minimal Setup for Direct Testing of llm_interface.py ---
    import sys
    from pathlib import Path
    # Ensure the script can find other modules in the 'src' directory if run directly
    sys_path_to_add = str(Path(__file__).resolve().parent.parent)
    if sys_path_to_add not in sys.path:
        sys.path.insert(0, sys_path_to_add)

    from src.config_loader import get_config, CONFIG_FILE_PATH, ensure_config_exists
    from src.logger_setup import setup_logging

    print("LLM Interface - Direct Test Mode")

    if not ensure_config_exists(CONFIG_FILE_PATH):
        print(f"Default config created at {CONFIG_FILE_PATH}. "
              "Please CONFIGURE THE 'llm' SECTION, especially for 'summary_llm'.\n"
              "IMPORTANT: To force GPU use, add 'num_gpu: -1' to the summary_llm config.\n"
              "Then ensure Ollama is running with the specified model and run this test again.")
        sys.exit(0)

    try:
        test_config = get_config()
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)

    log_folder_p = test_config['paths']['log_folder']
    log_file_n = test_config['paths']['log_file_name']
    if not isinstance(log_folder_p, Path) or not log_file_n: # Ensure resolved Path object
        print("Log folder or file name not found or invalid in config. Exiting test.")
        sys.exit(1)
    setup_logging(log_folder=log_folder_p, log_file_name=log_file_n)
    
    llm_key_to_test = "summary_llm"
    if 'llm' not in test_config or llm_key_to_test not in test_config['llm']:
        logger.error(f"'{llm_key_to_test}' not found in config['llm']. Please configure it. Cannot run LLM test.")
        sys.exit(1)


    logger.info(f"Attempting to load LLM model using config key: '{llm_key_to_test}'")
    chat_model_instance = get_llm_chat_model(test_config, llm_config_key=llm_key_to_test)

    if chat_model_instance:
        logger.info("Successfully initialized LLM model instance.")

        sample_snippets_user_request = [
            "I want to describe myself so I would sense that what Salman Rushdi would feel against Islamism"
        ]
        
        simulated_llm_output = (
            "Speaker is a user, expressing their desire to describe themselves and their understanding of the concept "
            "of Salman Rushdi's feelings towards Islamism. The speaker is asking for clarification on whether they "
            "would sense the emotions of someone with a particular worldview."
        )

        logger.info("\n--- Test Case 1: Replicating User's Desired Output (Simulated) ---")
        class MockLLMResponse:
            def __init__(self, content_text):
                self.content = content_text
        
        original_invoke = chat_model_instance.invoke
        chat_model_instance.invoke = lambda messages: MockLLMResponse(simulated_llm_output)
        
        summary_result = summarize_speaker_text(sample_snippets_user_request, chat_model_instance, max_words=70)
        
        logger.info(f"Input snippets for user request: {len(sample_snippets_user_request)} snippets.")
        logger.info(f"Simulated LLM Raw Output:\n{simulated_llm_output}")
        logger.info(f"Processed Summary: '{summary_result}'")

        chat_model_instance.invoke = original_invoke

        logger.info("\n--- Test Case 2: User Request (Actual LLM Call) ---")
        logger.info("This will use your actual LLM and GPU configuration.")
        summary3_actual = summarize_speaker_text(sample_snippets_user_request, chat_model_instance, max_words=70)
        logger.info(f"Input snippets: {len(sample_snippets_user_request)} snippets.")
        logger.info(f"Actual LLM Summary for User Request: '{summary3_actual}'")

    else:
        logger.error("Failed to initialize LLM model instance. Cannot run summarization tests.")
        logger.error(f"Ensure Ollama is running and the model specified in 'llm.{llm_key_to_test}.model_name' is available.")
        logger.error("If using GPU, ensure your system and Ollama are correctly configured for it.")

    logger.info("\nLLM Interface test concluded.")
