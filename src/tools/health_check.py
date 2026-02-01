# File: tools/health_check.py
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Set

# Add project root to sys.path to allow running from command line
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_loader import get_config, PROJECT_ROOT
from src.logger_setup import logger, setup_logging
from src.daily_log_manager import get_daily_log_data, get_samson_today
from src.speaker_profile_manager import get_all_speaker_profiles
from src.master_daily_transcript import get_master_transcript_path
from src.daily_log_manager import load_daily_flags_queue # Import helper from log manager

def run_health_check(lookback_days: int = 7) -> Dict[str, Any]:
    """
    Performs integrity and health checks on Samson's data files.

    Args:
        lookback_days: The number of recent days to check for logs and transcripts.

    Returns:
        A dictionary containing the status and a list of found issues.
        Example: {'status': 'issues_found', 'issues': ['Issue 1', 'Issue 2']}
    """
    logger.info(f"--- Running Data Health Check (lookback: {lookback_days} days) ---")
    issues: List[str] = []
    
    try:
        # 1. Load baseline data
        all_profiles = get_all_speaker_profiles()
        enrolled_speaker_names: Set[str] = {p.get('name') for p in all_profiles if p.get('name')}
        logger.info(f"Health Check: Found {len(enrolled_speaker_names)} enrolled speaker profiles.")

        # 2. Iterate through recent days to check logs and transcripts
        today = get_samson_today()
        for i in range(lookback_days):
            current_date = today - timedelta(days=i)
            current_datetime = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc)
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Load daily log for the current date
            daily_log = get_daily_log_data(current_datetime)
            
            # Skip if no log exists for this day
            if not daily_log or not daily_log.get('entries'):
                logger.debug(f"Health Check: No daily log entries for {date_str}, skipping checks for this day.")
                continue

            logger.info(f"Health Check: Analyzing data for {date_str}...")
            
            # --- Check 2.1: Speaker Name Consistency ---
            speakers_in_log: Set[str] = set()
            for entry in daily_log.get('chunks', {}).values():
                wlt = entry.get("processed_data", {}).get("word_level_transcript_with_absolute_times", [])
                for segment in wlt:
                    # Check segment-level speaker
                    seg_speaker = segment.get('speaker')
                    if seg_speaker:
                        speakers_in_log.add(seg_speaker)
                    # Check word-level speakers
                    for word in segment.get('words', []):
                        word_speaker = word.get('speaker')
                        if word_speaker:
                            speakers_in_log.add(word_speaker)
            
            unrecognized_speakers = speakers_in_log - enrolled_speaker_names
            # Filter out temporary/internal IDs
            unrecognized_speakers = {
                s for s in unrecognized_speakers 
                if not s.startswith(('SPEAKER_', 'UNKNOWN_', 'CUSID_'))
            }

            if unrecognized_speakers:
                issue_str = (f"On {date_str}, found speaker names in logs that are not in "
                             f"profiles: {', '.join(sorted(list(unrecognized_speakers)))}")
                issues.append(issue_str)
                logger.warning(f"Health Check Issue: {issue_str}")

            # --- Check 2.2: Master Transcript Existence ---
            config = get_config()
            master_log_path = get_master_transcript_path(config, current_datetime)
            if not master_log_path.exists() or master_log_path.stat().st_size == 0:
                issue_str = f"Master transcript for {date_str} is missing or empty."
                issues.append(issue_str)
                logger.warning(f"Health Check Issue: {issue_str}")

        # 3. Check for any pending flags across all recent days
        pending_flags_count = 0
        for i in range(lookback_days):
            target_date_for_flags = today - timedelta(days=i)
            flags_for_date = load_daily_flags_queue(datetime.combine(target_date_for_flags, datetime.min.time()))
            for flag in flags_for_date:
                if flag.get('status') == 'pending_review':
                    pending_flags_count += 1
        
        if pending_flags_count > 0:
            issue_str = f"Found {pending_flags_count} flag(s) pending review. Use the GUI or 'REVIEW FLAGS' command to resolve."
            issues.append(issue_str)
            logger.warning(f"Health Check Issue: {issue_str}")

    except Exception as e:
        error_issue = f"An unexpected error occurred during health check: {e}"
        issues.append(error_issue)
        logger.error(error_issue, exc_info=True)

    # 4. Final report
    if issues:
        report = {
            "status": "issues_found",
            "issues": issues,
            "checked_at_utc": datetime.now(timezone.utc).isoformat()
        }
        logger.warning(f"Health Check complete. Found {len(issues)} issue(s).")
    else:
        report = {
            "status": "ok",
            "issues": [],
            "checked_at_utc": datetime.now(timezone.utc).isoformat()
        }
        logger.info("Health Check complete. All checks passed.")
    
    return report

if __name__ == "__main__":
    # This allows the script to be run directly from the command line for manual checks
    try:
        config = get_config()
        setup_logging(
            log_folder=config['paths']['log_folder'],
            log_file_name=config['paths']['log_file_name']
        )
    except Exception as e:
        print(f"FATAL: Could not initialize backend configuration or logger for manual run: {e}")
        sys.exit(1)
        
    print("\nRunning Samson Data Health Check manually...")
    health_report = run_health_check()
    print("\n--- Health Check Report ---")
    print(f"Status: {health_report['status']}")
    if health_report['issues']:
        print("Details:")
        for idx, issue in enumerate(health_report['issues']):
            print(f"  {idx + 1}. {issue}")
    else:
        print("No issues found.")
    print("---------------------------\n")