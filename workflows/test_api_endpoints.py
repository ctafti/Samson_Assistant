#path: f/workflows/ai_give_me_a_workflow_that_tests_every_api_endpoint_y_5b5422
import requests
from datetime import datetime

def main() -> dict:
    """
    Tests all available Samson API endpoints and returns a summary of the results.

    This script systematically calls each known endpoint of the Samson API:
    - GET /api/daily_log/today
    - GET /api/daily_log/{YYYY-MM-DD} (using today's date)
    - GET /api/matters
    - GET /api/tasks

    It captures the status, response, and any errors for each call, providing a
    comprehensive health check of the API from within the Windmill environment.

    Returns:
        dict: A dictionary where each key is an endpoint and the value is another
              dictionary containing the test 'status' ('success' or 'error'),
              and the 'response' data or error message.
    """
    # --- Plan ---
    # 1. Define the base URL for the Samson API.
    # 2. Create a dictionary to store the results of each API test.
    # 3. Define a reusable helper function to make API requests and handle common errors.
    # 4. Call the helper for each endpoint:
    #    - /daily_log/today
    #    - /daily_log/DATE (using today's date)
    #    - /matters
    #    - /tasks
    # 5. Populate the results dictionary with the outcome of each call.
    # 6. Return the final results dictionary.
    # --- End Plan ---

    BASE_URL = "http://host.docker.internal:5000/api"
    test_results = {}

    def _make_request(endpoint: str) -> dict:
        """
        A helper function to perform a GET request and handle potential errors.

        Args:
            endpoint: The API endpoint to call (e.g., "/matters").

        Returns:
            A dictionary with the status and response/error.
        """
        url = f"{BASE_URL}{endpoint}"
        try:
            response = requests.get(url, timeout=10)
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()
            
            # Try to parse JSON, handle case where response is empty or not valid JSON
            try:
                data = response.json()
                return {"status": "success", "response": data}
            except requests.exceptions.JSONDecodeError:
                return {"status": "success", "response": "Response was not valid JSON."}

        except requests.exceptions.HTTPError as http_err:
            return {
                "status": "error",
                "response": f"HTTP error occurred: {http_err} - Status Code: {http_err.response.status_code}"
            }
        except requests.exceptions.ConnectionError as conn_err:
            return {
                "status": "error",
                "response": f"Connection error occurred: {conn_err}"
            }
        except requests.exceptions.Timeout as timeout_err:
            return {
                "status": "error",
                "response": f"Timeout error occurred: {timeout_err}"
            }
        except requests.exceptions.RequestException as req_err:
            return {
                "status": "error",
                "response": f"An unexpected error occurred: {req_err}"
            }

    # 1. Test GET /api/daily_log/today
    print("Testing GET /api/daily_log/today...")
    test_results["/daily_log/today"] = _make_request("/daily_log/today")

    # 2. Test GET /api/daily_log/DATE
    today_str = datetime.now().strftime("%Y-%m-%d")
    date_endpoint = f"/daily_log/{today_str}"
    print(f"Testing GET {date_endpoint}...")
    test_results[date_endpoint] = _make_request(date_endpoint)

    # 3. Test GET /api/matters
    print("Testing GET /api/matters...")
    test_results["/matters"] = _make_request("/matters")

    # 4. Test GET /api/tasks
    print("Testing GET /api/tasks...")
    test_results["/tasks"] = _make_request("/tasks")

    print("API endpoint testing complete.")
    return test_results