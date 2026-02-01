#path: f/workflows/fun_test
# /// script
# dependencies = [
#   "requests",
#   "openai",
#   "PyPDF2"
# ]
# ///

"""
Workflow: PDF Word Replacer and Creative Writer

This script processes a user-provided PDF file. It extracts the text,
replaces a specified word with another, and then uses an LLM to generate
a creative piece based on the modified sentences.

Plan:
1. Validate inputs: Ensure a file is selected and instructions are provided in the correct format.
2. Parse instructions: Extract the word to remove and the word to replace it with from the `additional_instructions` string.
3. Extract text from PDF: Read the specified PDF file from `/wmill/data/` and extract all its text content.
4. Perform word replacement: Go through the extracted text and replace all occurrences of the target word (case-insensitively).
5. Identify modified sentences: Isolate the specific sentences that were altered by the replacement.
6. Fetch LLM configuration: Get the local LLM settings from the Samson API.
7. Generate creative content: Send the modified sentences to the LLM with a prompt to create something fun (e.g., a short story or poem).
8. Return structured results: Output a dictionary containing the replacement summary, the modified sentences, and the LLM's creative response.
"""

import requests
import re
import PyPDF2
from openai import OpenAI
from typing import Dict, List, Any, Tuple

# --- Helper Functions ---

def parse_instructions(instructions: str) -> Tuple[str, str]:
    """
    Parses the replacement instructions from the format (word1 --> word2).
    
    Args:
        instructions: The string containing the replacement rule.
        
    Returns:
        A tuple of (word_to_remove, word_to_replace_with).
        
    Raises:
        ValueError: If the instructions are not in the expected format.
    """
    match = re.search(r"\((.*?)\s*-->\s*(.*?)\)", instructions)
    if not match:
        raise ValueError("Instructions must be in the format: (word_to_remove --> word_to_replace_with)")
    
    word_to_remove = match.group(1).strip()
    word_to_replace_with = match.group(2).strip()
    
    if not word_to_remove or not word_to_replace_with:
        raise ValueError("Both words for replacement must be non-empty.")
        
    return word_to_remove, word_to_replace_with

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts all text content from a given PDF file.
    
    Args:
        file_path: The full path to the PDF file.
        
    Returns:
        A string containing all the text from the PDF.
        
    Raises:
        FileNotFoundError: If the PDF file does not exist.
        Exception: For other PDF processing errors.
    """
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# --- Main Logic ---

def main(selected_file: str = "", additional_instructions: str = "") -> Dict[str, Any]:
    """
    Extracts text from a PDF, replaces a word, and uses an LLM for creative writing.

    Args:
        selected_file: The name of the PDF file in `/wmill/data/`.
        additional_instructions: A string in the format "(word_to_remove --> word_to_replace_with)"
                                 specifying the word replacement rule.

    Returns:
        A dictionary containing the results of the operation, including the
        number of replacements and the creative output from the LLM.
    """
    # Step 1: Validate inputs and parse instructions
    if not selected_file:
        return {"success": False, "error": "No file was selected. Please choose a PDF file."}
    if not additional_instructions:
        return {"success": False, "error": "Additional instructions are required. Please provide them in the format (word_to_remove --> word_to_replace_with)."}

    try:
        word_to_remove, word_to_replace_with = parse_instructions(additional_instructions)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    full_path = f"/wmill/data/{selected_file}"

    # Step 2: Extract text from PDF
    try:
        original_text = extract_text_from_pdf(full_path)
        if not original_text.strip():
            return {"success": False, "error": f"Could not extract any text from '{selected_file}'. The file might be image-based or empty."}
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {full_path}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to process PDF '{selected_file}': {str(e)}"}

    # Step 3: Perform word replacement and identify modified sentences
    # Use re.sub for case-insensitive replacement and to count occurrences
    modified_text, num_replacements = re.subn(
        r'\b' + re.escape(word_to_remove) + r'\b', 
        word_to_replace_with, 
        original_text, 
        flags=re.IGNORECASE
    )

    if num_replacements == 0:
        return {
            "success": True,
            "message": f"The word '{word_to_remove}' was not found in the document. No changes were made.",
            "file_processed": selected_file,
            "replacements_made": 0,
        }

    # Split into sentences to find which ones were changed
    original_sentences = re.split(r'(?<=[.!?])\s+', original_text)
    modified_sentences_all = re.split(r'(?<=[.!?])\s+', modified_text)
    
    changed_sentences = [
        mod_sent 
        for orig_sent, mod_sent in zip(original_sentences, modified_sentences_all) 
        if orig_sent != mod_sent
    ]

    # Step 4: Fetch LLM configuration
    base_api = "http://host.docker.internal:5000/api"
    try:
        llm_response = requests.get(f"{base_api}/llm_config", timeout=10)
        llm_response.raise_for_status()
        llm_config = llm_response.json()
        if not llm_config.get('success') or not llm_config.get('models'):
            return {"success": False, "error": "Failed to fetch a valid LLM configuration from Samson API."}
        base_url = llm_config['base_url_from_windmill']
        model = llm_config['models'][0]['model_name']
    except requests.RequestException as e:
        return {"success": False, "error": f"Could not connect to Samson API for LLM config: {str(e)}"}

    # Step 5: Generate creative content with LLM
    llm_creative_output = "LLM analysis was not performed."
    try:
        client = OpenAI(base_url=base_url, api_key="lm-studio")
        
        sentences_for_prompt = "\n- ".join(changed_sentences)
        prompt = f"""You are a whimsical storyteller. You have been given a list of sentences that were recently and strangely modified by replacing the word '{word_to_remove}' with '{word_to_replace_with}'.

Your task is to write a short, fun, and creative piece (like a micro-story, a poem, or a dramatic monologue) that incorporates some or all of these sentences. Embrace the absurdity!

Here are the modified sentences:
- {sentences_for_prompt}

Now, create your masterpiece.
"""
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a creative and slightly eccentric writer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=500
        )
        llm_creative_output = response.choices[0].message.content
    except Exception as e:
        # If LLM fails, we still return the successful text replacement part
        llm_creative_output = f"LLM creative generation failed: {str(e)}"

    # Step 6: Return structured results
    return {
        "success": True,
        "file_processed": selected_file,
        "word_removed": word_to_remove,
        "word_added": word_to_replace_with,
        "replacements_made": num_replacements,
        "modified_sentences_count": len(changed_sentences),
        "modified_sentences": changed_sentences,
        "llm_creative_output": llm_creative_output
    }