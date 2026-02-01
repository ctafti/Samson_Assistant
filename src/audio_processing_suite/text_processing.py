# src/audio_processing_suite/text_processing.py
import re
# import logging # REMOVED
from typing import List, Dict, Any, Tuple, Optional

# Import external libraries (ensure they are installed)
# For fuzzy name matching
# Ensure 'thefuzz' and 'python-Levenshtein' are installed.
# Add "thefuzz[speedup]>=0.19.0" to your requirements.txt
try:
    from thefuzz import fuzz, process
    THEFUZZ_AVAILABLE = True
except ImportError:
    fuzz = None # type: ignore
    process = None # type: ignore
    THEFUZZ_AVAILABLE = False
    # Logger warning will be emitted by get_closest_name_match if called without the library


# import traceback # Not used in the provided code, can be removed
# from collections import defaultdict # Not used in the provided code, can be removed

from src.logger_setup import logger # ADDED

# REMOVED: logger = logging.getLogger(__name__)

# Initialize globally or pass instances if preferred
# Passing instances is generally better for testability
# cucco_instance = Cucco() if Cucco else None
# inflect_engine = inflect.engine() if inflect else None

def reconstruct_text_from_word_objects(
    word_objects: List[Dict[str, Any]],
    speaker_label: Optional[str] = "Unknown" # Added for context
) -> str:
    """
    Reconstructs a text string from a list of word objects,
    handling spacing more intelligently.
    """
    if not word_objects:
        return ""

    full_text_parts: List[str] = []
    for i, word_obj in enumerate(word_objects):
        word_text = str(word_obj.get("word", "")).strip() # Ensure word is stripped here
        
        if not word_text: # Skip empty words
            continue

        full_text_parts.append(word_text)

    reconstructed = " ".join(full_text_parts)
    
    reconstructed = reconstructed.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
    reconstructed = reconstructed.replace(" ' ", "'") 
    reconstructed = reconstructed.replace(" n't", "n't")
    reconstructed = reconstructed.replace(" 're", "'re")
    reconstructed = reconstructed.replace(" 've", "'ve")
    reconstructed = reconstructed.replace(" 'll", "'ll")
    reconstructed = reconstructed.replace(" 's", "'s") 
    reconstructed = reconstructed.replace(" 'm", "'m")

    logger.debug(f"Reconstructed text for speaker {speaker_label or 'Unknown'} ({len(word_objects)} words).") # Removed full text
    return reconstructed

def format_punctuated_output(results: List[Dict[str, Any]]) -> str:
    """
    Reconstructs text from the punctuation pipeline output (token classification).
    Handles subwords and applies punctuation/capitalization based on entity labels.
    Assumes HuggingFace pipeline with aggregation_strategy="simple".
    """
    text = ''
    last_word_end = -1 

    logger.debug(f"Formatting {len(results)} punctuation results using format_punctuated_output.")
    if not results: return ""

    ATTACHING_PUNCTUATION_LABELS = {
        'PERIOD', 'PUNC_PERIOD', 'QUESTION', 'PUNC_QUESTION', 'COMMA', 'PUNC_COMMA',
        'EXCLAMATION', 'PUNC_EXCLAMATION', 'COLON', 'PUNC_COLON', 'SEMICOLON', 'PUNC_SEMICOLON',
    }
    # MODIFIED: Added regex match for generic LABEL_X
    IGNORE_LABELS = {'O'} # Removed LABEL_0, LABEL_1 to be handled by regex

    try:
        for i, item in enumerate(results):
            if not isinstance(item, dict):
                logger.warning(f"Skipping invalid item in punctuation results: {item}")
                continue

            word = item.get('word', '').strip()
            entity = str(item.get('entity_group', item.get('entity', 'O'))).upper()
            score = item.get('score', 0.0)
            start = item.get('start') 
            end = item.get('end')

            if not word: continue 

            is_subword = word.startswith('##') or (start is not None and last_word_end is not None and start == last_word_end)
            if word.startswith('##'):
                word = word[2:]

            if text and not is_subword and not text.endswith(('-', "'")):
                 if entity not in ATTACHING_PUNCTUATION_LABELS:
                      text += ' '

            if i == 0 or (text and re.search(r'[.?!]\s*$', text)):
                 if word: word = word.capitalize()

            text += word

            punct_map = {
                'PERIOD': '.', 'PUNC_PERIOD': '.', 'QUESTION': '?', 'PUNC_QUESTION': '?',
                'COMMA': ',', 'PUNC_COMMA': ',', 'EXCLAMATION': '!', 'PUNC_EXCLAMATION': '!',
                'COLON': ':', 'PUNC_COLON': ':', 'SEMICOLON': ';', 'PUNC_SEMICOLON': ';',
            }
            
            # Check if entity is a generic label like "LABEL_12", otherwise process
            is_generic_label = re.fullmatch(r"LABEL_\d+", entity)

            if entity in punct_map:
                text += punct_map[entity]
            elif entity not in IGNORE_LABELS and not is_generic_label: # MODIFIED
                logger.debug(f"Unhandled punctuation entity label '{entity}' for word '{word}' (Score: {score:.2f}).")

            if end is not None:
                 last_word_end = end

    except Exception as e:
        logger.error(f"Error during punctuation formatting (format_punctuated_output): {e}", exc_info=True)
        return " ".join([item.get('word', '') for item in results if isinstance(item, dict)])

    text = re.sub(r'\s+([.,?!:;])', r'\1', text) 
    text = re.sub(r'([.,?!:;])(?=\S)', r'\1 ', text) 
    text = re.sub(r'\s{2,}', ' ', text).strip() 
    return text


def apply_punctuation(
    word_level_transcript: List[Dict[str, Any]], 
    punctuation_model: Optional[Any] = None,
    chunk_size: int = 250 # Default as per existing code
) -> List[Tuple[Optional[str], str]]:
    logger.debug("\n--- Applying Punctuation / Reconstructing Text Turns ---") # Changed to debug
    if not word_level_transcript:
        logger.warning("Punctuation: Word level transcript is empty.")
        return []

    raw_speaker_texts: List[Tuple[Optional[str], str]] = []
    current_speaker: Optional[str] = None
    current_turn_word_objects: List[Dict[str, Any]] = [] 

    for segment in word_level_transcript:
        if not isinstance(segment, dict) or 'words' not in segment or not isinstance(segment.get('words'), list):
            logger.debug(f"Skipping invalid segment structure: {type(segment)}")
            continue
            
        for word_obj in segment['words']:
            if not isinstance(word_obj, dict):
                logger.debug(f"Skipping invalid word_obj structure: {type(word_obj)}")
                continue

            word_text_val = str(word_obj.get("word", "")).strip() 
            speaker_val = str(word_obj.get("speaker", "UNKNOWN_SPEAKER")) 

            if not word_text_val: 
                continue
            
            if speaker_val != current_speaker:
                if current_turn_word_objects: 
                    reconstructed_text = reconstruct_text_from_word_objects(current_turn_word_objects, current_speaker)
                    if reconstructed_text: 
                        raw_speaker_texts.append((current_speaker, reconstructed_text))
                    current_turn_word_objects = []
                current_speaker = speaker_val
            
            current_turn_word_objects.append(word_obj) 

    if current_turn_word_objects:
        reconstructed_text = reconstruct_text_from_word_objects(current_turn_word_objects, current_speaker)
        if reconstructed_text:
            raw_speaker_texts.append((current_speaker, reconstructed_text))
    
    if not raw_speaker_texts:
        logger.warning("Punctuation: No text turns constructed from word_level_transcript.")
        return []
        
    logger.debug(f"Punctuation: Reconstructed {len(raw_speaker_texts)} raw speaker turns.") # Changed to debug
    for i, (spk, txt) in enumerate(raw_speaker_texts[:3]): 
        logger.debug(f"  Raw Turn {i+1} [{spk if spk else 'None'}] ({len(txt.split())} words).") # Changed to word count

    if not punctuation_model:
        logger.debug("Punctuation: Punctuation model not provided. Returning raw reconstructed text turns.") # Changed to debug
        return raw_speaker_texts 

    logger.debug("Punctuation: Punctuation model provided. Applying punctuation to reconstructed turns.") # Changed to debug
    final_punctuated_turns: List[Tuple[Optional[str], str]] = []
    total_words_processed_for_punctuation = 0 
    
    for speaker, text_block in raw_speaker_texts: # text_block is unpunctuated string here
        if not text_block.strip(): 
            final_punctuated_turns.append((speaker, "")) 
            continue

        punctuated_text_parts: List[str] = []
        words_in_block = text_block.split() 
        
        if not words_in_block: 
            final_punctuated_turns.append((speaker, text_block)) 
            continue

        total_words_processed_for_punctuation += len(words_in_block)

        for i in range(0, len(words_in_block), chunk_size):
            chunk_words_list = words_in_block[i:i + chunk_size]
            chunk_text = " ".join(chunk_words_list) # This is the text to punctuate for the current chunk
            
            if chunk_text.strip(): 
                logger.debug(f"  Punctuating chunk for speaker {speaker} ({len(chunk_words_list)} words).") # Removed text content
                try:
                    # Get raw output from the punctuation model
                    model_output_for_chunk = punctuation_model(chunk_text)
                    
                    formatted_chunk_text: str
                    # Ensure format_punctuated_output is called correctly
                    if isinstance(model_output_for_chunk, list) and \
                       all(isinstance(item, dict) for item in model_output_for_chunk):
                        # This case is when HF pipeline returns a list of dicts directly
                        formatted_chunk_text = format_punctuated_output(model_output_for_chunk)
                    elif isinstance(model_output_for_chunk, str):
                        # Some pipelines might directly return a string
                        formatted_chunk_text = model_output_for_chunk
                        logger.info(f"Punctuation: Punctuation model returned a string directly for speaker {speaker or 'Unknown'} on chunk: '{chunk_text[:50]}...'. This will be used as is.")
                    else:
                        logger.warning(f"Punctuation: Unexpected output type from punctuation model: {type(model_output_for_chunk)} for speaker {speaker or 'Unknown'} on chunk: '{chunk_text[:50]}...'. Using raw chunk text.")
                        formatted_chunk_text = chunk_text # Fallback

                    punctuated_text_parts.append(formatted_chunk_text)
                except Exception as e_punc:
                    logger.error(f"Punctuation: Error during model inference or formatting for speaker {speaker or 'Unknown'} on chunk: '{chunk_text[:50]}...': {e_punc}", exc_info=True)
                    punctuated_text_parts.append(chunk_text) # Fallback to unpunctuated chunk
            elif chunk_words_list: # If chunk was only whitespace but came from actual words (e.g. [" ", " "])
                punctuated_text_parts.append(chunk_text) # Append the whitespace as is

        final_text_for_speaker = " ".join(punctuated_text_parts).strip()
        
        # Apply regex for spacing around punctuation
        final_text_for_speaker = re.sub(r'\s+([.,?!:;])', r'\1', final_text_for_speaker) # Remove space before punctuation
        final_text_for_speaker = re.sub(r'([.,?!:;])(?=\S)', r'\1 ', final_text_for_speaker) # Add space after punctuation if followed by non-space
        final_text_for_speaker = re.sub(r'\s{2,}', ' ', final_text_for_speaker).strip() # Consolidate multiple spaces and strip again
        
        final_punctuated_turns.append((speaker, final_text_for_speaker))

    logger.debug(f"Punctuation: Punctuation application complete. Processed approx {total_words_processed_for_punctuation} words through model for {len(final_punctuated_turns)} turns.") # Changed to debug
    return final_punctuated_turns


# New function for fuzzy name matching as per instructions
def get_closest_name_match(
    input_name: str,
    known_names: List[str],
    threshold_ratio: int = 80  # Default similarity ratio (0-100)
) -> Optional[str]:
    """
    Finds the best fuzzy match for input_name from a list of known_names.
    Returns the best matching known_name if above threshold, else None.
    """
    if not THEFUZZ_AVAILABLE:
        logger.warning(
            "Fuzzy name matching function called, but 'thefuzz' library is not available. "
            "Please install it (e.g., 'pip install \"thefuzz[speedup]>=0.19.0\"'). "
            "No fuzzy matching will be performed."
        )
        return None

    # Check for empty inputs after library availability
    if not known_names or not input_name or not input_name.strip():
        logger.debug(
            f"get_closest_name_match: called with empty known_names or invalid input_name. "
            f"Input: '{input_name}', Known names count: {len(known_names) if known_names else 0}"
        )
        return None

    # process.extractOne returns a tuple: (choice, score) or None if choices is empty
    # The 'type: ignore' can be helpful if your linter struggles with fuzz/process being None
    # despite the THEFUZZ_AVAILABLE check, though the runtime guard is sufficient.
    best_match_tuple = process.extractOne(input_name, known_names, scorer=fuzz.ratio) # type: ignore

    if best_match_tuple:
        match_name, score = best_match_tuple
        logger.debug(f"Fuzzy match for '{input_name}': Best candidate '{match_name}' with score {score} (threshold: {threshold_ratio}).")
        if score >= threshold_ratio:
            return match_name
        else:
            logger.debug(f"Match score {score} for '{input_name}' ('{match_name}') is below threshold {threshold_ratio}.")
            return None # Explicitly return None if below threshold
    else:
        # This case implies process.extractOne returned None, which can happen if choices (known_names) is empty.
        # However, we check for empty known_names earlier.
        # It might also occur if input_name is highly dissimilar or other edge cases in the library.
        logger.debug(f"No fuzzy match candidate returned by process.extractOne for '{input_name}' with the given known_names.")
        return None