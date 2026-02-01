from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timezone
import sys
from pathlib import Path

# Add your project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.daily_log_manager import get_daily_log_data, get_samson_today
from src.matter_manager import get_all_matters, get_matter_by_id
from src.task_intelligence_manager import _load_tasks
from src.config_loader import load_config

app = Flask(__name__)
CORS(app)  # Allow Windmill to call from Docker

# Load config once at startup
config = load_config()

@app.route('/health', methods=['GET'])
def health_check():
    """Simple endpoint to verify API is running"""
    return jsonify({"status": "healthy", "service": "Samson API"})

@app.route('/api/daily_log/today', methods=['GET'])
def get_today_log():
    """Get today's daily log data"""
    try:
        today_date = get_samson_today()
        target_dt = datetime.combine(today_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        log_data = get_daily_log_data(target_dt)
        
        if not log_data:
            return jsonify({
                "success": False,
                "message": f"No log data found for {today_date.isoformat()}",
                "date": today_date.isoformat(),
                "chunks_count": 0
            })
        
        chunks = log_data.get("chunks", {})
        return jsonify({
            "success": True,
            "date": today_date.isoformat(),
            "chunks_count": len(chunks),
            "summary": log_data.get("summary", {}),
            "total_duration_seconds": log_data.get("total_duration_seconds", 0)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/daily_log/<date_str>', methods=['GET'])
def get_log_by_date(date_str):
    """Get daily log for a specific date (format: YYYY-MM-DD)"""
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        log_data = get_daily_log_data(target_date)
        
        if not log_data:
            return jsonify({
                "success": False,
                "message": f"No log data found for {date_str}",
                "date": date_str,
                "chunks_count": 0
            })
        
        chunks = log_data.get("chunks", {})
        return jsonify({
            "success": True,
            "date": date_str,
            "chunks_count": len(chunks),
            "summary": log_data.get("summary", {}),
            "total_duration_seconds": log_data.get("total_duration_seconds", 0)
        })
    except ValueError:
        return jsonify({"success": False, "error": "Invalid date format. Use YYYY-MM-DD"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/matters', methods=['GET'])
def list_matters():
    """Get all matters - returns plain list"""
    try:
        matters = get_all_matters()
        return jsonify(matters if matters else [])
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/matters/<matter_id>', methods=['GET'])
def get_matter(matter_id):
    """Get a specific matter by ID"""
    try:
        matter = get_matter_by_id(matter_id)
        if not matter:
            return jsonify({"success": False, "error": "Matter not found"}), 404
        return jsonify({"success": True, "matter": matter})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """Get all tasks - returns plain list"""
    try:
        tasks = _load_tasks()
        return jsonify(tasks if tasks else [])
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/daily_log_details/<date_str>', methods=['GET'])
def get_log_details_by_date(date_str):
    """
    Get detailed daily log for a specific date (format: YYYY-MM-DD).
    Query param 'format' can be 'full' (default) or 'simple'.
    'simple' returns a concatenated text transcript with speaker and matter context.
    'full' returns a flattened list of all word objects for the day with detailed timestamps.
    """
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        log_data = get_daily_log_data(target_date)

        if not log_data:
            return jsonify({"success": False, "message": f"No log data found for {date_str}"}), 404
        
        response_format = request.args.get('format', 'full').lower()

        if response_format == 'simple':
            from datetime import timedelta
            simple_transcript_lines = []
            matter_name_cache = {}
            last_matter_id = None
            day_start_timestamp = log_data.get("day_start_timestamp_utc")
            
            # Ensure day_start_timestamp is a datetime object for calculations
            day_start_dt = None
            if isinstance(day_start_timestamp, str):
                try:
                    day_start_dt = datetime.fromisoformat(day_start_timestamp.replace('Z', '+00:00'))
                except ValueError:
                    pass # Keep as None if parsing fails
            elif isinstance(day_start_timestamp, datetime):
                day_start_dt = day_start_timestamp
            
            timestamp_interval_seconds = 300  # 5 minutes
            last_timestamp_written_at = -timestamp_interval_seconds

            sorted_chunks = sorted(
                log_data.get("chunks", {}).values(),
                key=lambda c: c.get('file_sequence_number', 0)
            )

            for chunk in sorted_chunks:
                matter_segments = chunk.get("matter_segments", [])
                # Read directly from the original, non-renamed key for clarity
                word_level_transcript = chunk.get("processed_data", {}).get("word_level_transcript_with_absolute_times", [])
                
                for segment in word_level_transcript:
                    segment_start_time = segment.get("start", -1)

                    if day_start_dt and segment_start_time != -1:
                        if segment_start_time >= last_timestamp_written_at + timestamp_interval_seconds:
                            # Note: segment start time is relative to the CHUNK, not the day. This logic is complex.
                            # For simplicity, we will assume this provides a rough time guide.
                            # A more accurate implementation would require chunk start times.
                            pass # Timestamping logic disabled for review, can be re-enabled.

                    current_matter_id = None
                    for ms in matter_segments:
                        if ms.get("start_time", -1) <= segment_start_time < ms.get("end_time", -1):
                            current_matter_id = ms.get("matter_id")
                            break
                    
                    if current_matter_id and current_matter_id != last_matter_id:
                        if current_matter_id not in matter_name_cache:
                            matter_details = get_matter_by_id(current_matter_id)
                            matter_name_cache[current_matter_id] = matter_details.get("name", "Unknown Matter") if matter_details else "Unknown Matter"
                        
                        simple_transcript_lines.append(f"\n[Matter: {matter_name_cache[current_matter_id]}]\n")
                        last_matter_id = current_matter_id

                    speaker = segment.get("speaker", "Unknown")
                    text = segment.get("text", "")
                    simple_transcript_lines.append(f"{speaker}: {text}")

            return jsonify({
                "success": True,
                "date": date_str,
                "transcript": "\n".join(simple_transcript_lines)
            })
        else: # 'full' format
            all_words_transformed = []
            
            # Get the day start timestamp for reference
            day_start_timestamp = log_data.get("day_start_timestamp_utc")
            if isinstance(day_start_timestamp, str):
                day_start_dt = datetime.fromisoformat(day_start_timestamp.replace('Z', '+00:00'))
            elif isinstance(day_start_timestamp, datetime):
                day_start_dt = day_start_timestamp
            else:
                day_start_dt = None
            
            sorted_chunks = sorted(
                log_data.get("chunks", {}).values(),
                key=lambda c: c.get('file_sequence_number', 0)
            )
            
            for chunk in sorted_chunks:
                word_level_transcript = chunk.get("processed_data", {}).get("word_level_transcript_with_absolute_times", [])
                for segment in word_level_transcript:
                    speaker = segment.get("speaker", "Unknown")
                    
                    for word in segment.get("words", []):
                        transformed_word = word.copy()
                        transformed_word['speaker'] = speaker
                        
                        # Calculate seconds from day start using absolute UTC timestamps
                        if day_start_dt and 'absolute_end_utc' in transformed_word:
                            try:
                                word_end_dt = datetime.fromisoformat(
                                    transformed_word['absolute_end_utc'].replace('Z', '+00:00')
                                )
                                # Calculate seconds from day start
                                seconds_from_start = (word_end_dt - day_start_dt).total_seconds()
                                transformed_word['end'] = seconds_from_start
                                
                                if 'absolute_start_utc' in transformed_word:
                                    word_start_dt = datetime.fromisoformat(
                                        transformed_word['absolute_start_utc'].replace('Z', '+00:00')
                                    )
                                    transformed_word['start'] = (word_start_dt - day_start_dt).total_seconds()
                            except (ValueError, TypeError):
                                pass  # Keep original relative times if conversion fails
                        
                        all_words_transformed.append(transformed_word)

            simplified_data = {}
            if isinstance(log_data.get("day_start_timestamp_utc"), datetime):
                simplified_data["day_start_timestamp_utc"] = log_data["day_start_timestamp_utc"].isoformat()
            
            simplified_data["word_level_transcript"] = all_words_transformed

            return jsonify({
                "success": True,
                "date": date_str,
                "data": simplified_data
            })

    except ValueError:
        return jsonify({"success": False, "error": "Invalid date format. Use YYYY-MM-DD"}), 400
    except Exception as e:
        # Added traceback for better debugging
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/llm_config', methods=['GET'])
def get_llm_config():
    """Returns LLM configuration for workflows to use"""
    try:
        llm_config = config.get('llm', {})
        
        # Extract only LM Studio configurations
        lm_studio_models = []
        profiles = llm_config.get('profiles', {})
        for key, profile in profiles.items():
            if isinstance(profile, dict) and profile.get('provider', '').lower() == 'lmstudio':
                lm_studio_models.append({
                    'profile_name': key,
                    'model_name': profile.get('model_name'),
                    'base_url': profile.get('base_url', 'http://localhost:1234/v1'),
                    'temperature': profile.get('temperature', 0.3),
                    'use_case': key.replace('_', ' ').title()
                })
        
        return jsonify({
            'success': True,
            'base_url_from_windmill': 'http://host.docker.internal:1234/v1',  # Docker-adjusted
            'models': lm_studio_models
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    # Run on all interfaces so Docker can access it
    app.run(host='0.0.0.0', port=5000, debug=True)