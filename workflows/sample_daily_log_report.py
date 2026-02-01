#path:f/workflows/sample_daily_log_report

import requests

def main():
    """Generate daily log report by calling Samson API"""
    
    print("--- Daily Log Report Starting ---")
    
    try:
        # Use host.docker.internal to reach your host from Docker
        api_url = "http://host.docker.internal:5000/api/daily_log/today"
        
        print(f"Calling Samson API: {api_url}")
        response = requests.get(api_url, timeout=30)
        
        print(f"API Response Status: {response.status_code}")
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"API returned status {response.status_code}"
            }
        
        data = response.json()
        print(f"API Response: {data}")
        
        if not data.get("success"):
            return data
        
        chunks_count = data.get("chunks_count", 0)
        date = data.get("date", "unknown")
        
        result = {
            "success": True,
            "summary": f"Report generated for {date}: {chunks_count} chunks processed",
            "date": date,
            "chunks_count": chunks_count
        }
        
        print(f"Report: {result['summary']}")
        print("--- Daily Log Report Complete ---")
        
        return result
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return {"success": False, "error": error_msg}