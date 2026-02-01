# /// script
# dependencies = [
#   "requests"
# ]
# ///

"""
Workflow: Fetch Today's Transcript

This script retrieves the simple text transcript for the current day from the Samson API.
It follows a robust, two-step process to ensure data availability before fetching the
full transcript. The transcript is returned directly and also saved to a file in the
output directory for easy access.

Plan:
1.  Define the base API URL for Samson.
2.  Step 1: Call the `/api/daily_log/today` summary endpoint to verify that a log exists
    and to get the correct, API-validated date string.
3.  Handle cases where no log data is available yet.
4.  Step 2: Use the validated date string to call the `/api/daily_log_details/{date}`
    endpoint with `format=simple` to retrieve the plain text transcript.
5.  Handle cases where the transcript is empty.
6.  Create the output directory `/wmill/data/output/` if it doesn't exist.
7.  Save the retrieved transcript to a file named `transcript_{date}.txt`.
8.  Return a success message, the date, the path to the saved file, and the full transcript content.
"""

import requests
import os
from typing import Dict, Any

def main(selected_file: str = "", additional_instructions: str = "") -> Dict[str, Any]:
    """
    Fetches today's transcript from the Samson API.

    This function implements the required two-step data access pattern:
    1. Check the summary endpoint to confirm data exists and get the date.
    2. Fetch the detailed transcript using the validated date.

    Args:
        selected_file (str): Not used in this script.
        additional_instructions (str): Not used in this script.

    Returns:
        A dictionary containing the success status, the date of the transcript,
        the path to the saved output file, and the full transcript text.
        In case of an error, it returns a dictionary with an 'error' key.
    """
    base_api = "http://host.docker.internal:5000/api"

    # STEP 1: Check if today's log exists and get the validated date
    try:
        summary_response = requests.get(f"{base_api}/daily_log/today", timeout=10)
        summary_response.raise_for_status()
        summary_data = summary_response.json()

        if not summary_data.get("success"):
            return {"error": "Could not retrieve today's log summary. Recording may not have started yet."}

        if summary_data.get("chunks_count", 0) == 0:
            return {"error": "No recording data available for today yet.", "suggestion": "Check back after recording starts."}

        # CRITICAL: Use the date from the API, don't construct your own
        date_str = summary_data.get("date")
        if not date_str:
            return {"error": "API did not return a valid date for today's log."}

    except requests.RequestException as e:
        return {"error": f"Failed to check log availability: {str(e)}"}

    # STEP 2: Fetch the transcript using the validated date
    try:
        transcript_url = f"{base_api}/daily_log_details/{date_str}?format=simple"
        transcript_response = requests.get(transcript_url, timeout=30)
        transcript_response.raise_for_status()
        transcript_data = transcript_response.json()

        full_transcript = transcript_data.get("transcript", "")
        if not full_transcript:
            return {"error": f"Transcript exists but is empty for {date_str}.", "suggestion": "Recording may have just started. Wait a few minutes."}

    except requests.RequestException as e:
        return {"error": f"Failed to fetch transcript for date {date_str}: {str(e)}"}

    # STEP 3: Save the transcript to a file
    try:
        output_dir = "/wmill/data/output"
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"transcript_{date_str}.txt"
        output_filepath = os.path.join(output_dir, output_filename)

        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(full_transcript)
    except IOError as e:
        return {
            "error": f"Failed to write transcript to file: {str(e)}",
            "date": date_str,
            "transcript": full_transcript  # Still return the transcript even if file write fails
        }

    # STEP 4: Return the result
    return {
        "success": True,
        "message": f"Successfully fetched and saved transcript for {date_str}.",
        "date": date_str,
        "output_file": output_filepath,
        "transcript": full_transcript
    }