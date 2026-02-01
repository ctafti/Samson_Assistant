#!/bin/bash

# Samson AI Assistant - Setup Script
# ---------------------------------
# This script checks for dependencies, guides setup, and can attempt
# non-interactive installations/updates.
#
# MODIFICATIONS:
# - Added Environment Sanity Check to prevent running with active Conda envs.
# - Added strict Python Version Check to ensure a compatible version (3.11) is used,
#   which fixes the 'pip install' errors for locked dependencies.

echo "Samson AI Assistant - Setup Script"
echo "------------------------------------"

# --- Configuration ---
NON_INTERACTIVE=false
if [[ "$1" == "--non-interactive" ]]; then
    NON_INTERACTIVE=true
    echo "[INFO] Running in non-interactive mode. Will attempt default actions."
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE_RELATIVE="config/config.yaml" # Relative to script location
CONFIG_FILE="$SCRIPT_DIR/$CONFIG_FILE_RELATIVE" # Absolute path to config
CAPTCHA_TOKEN_FILE="$SCRIPT_DIR/captcha_token.txt" # Path for the captcha token file

# --- NEW: Environment Sanity Check ---
# Prevents running inside a Conda environment which can cause conflicts with venv.
if [ -n "$CONDA_PREFIX" ]; then
    echo ""
    echo "[FATAL ERROR] Conflicting environments detected."
    echo "A Conda environment ('${CONDA_DEFAULT_ENV:-}') appears to be active."
    echo "To avoid severe installation and runtime issues, this script must be run from a clean shell."
    echo ""
    echo "SOLUTION: Please deactivate Conda fully by running 'conda deactivate' in your terminal"
    echo "          (you may need to run it multiple times) until your shell prompt no longer"
    echo "          shows a Conda environment name like '(base)'."
    echo "          Then, start from a clean terminal and run this script again."
    exit 1
fi
# --- END: Environment Sanity Check ---

# Function to prompt user, respecting NON_INTERACTIVE mode
prompt_user() {
    local message="$1"
    local default_action_is_yes="$2" # true or false

    if [ "$NON_INTERACTIVE" = true ]; then
        if [ "$default_action_is_yes" = true ]; then
            echo "$message (Assuming 'yes' in non-interactive mode)"
            return 0 # Represents 'yes'
        else
            echo "$message (Assuming 'no' in non-interactive mode)"
            return 1 # Represents 'no'
        fi
    else
        read -p "$message (y/N): " response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            return 0 # Yes
        else
            return 1 # No
        fi
    fi
}

# Function to check if a command exists
command_exists () {
    command -v "$1" >/dev/null 2>&1
}

# Function to robustly get a value from config.yaml (simple key: value)
get_config_value() {
    local key_name_raw="$1"
    local config_f="$2"
    local value_final="" # Initialize to empty

    # FIX: Escape dots in key_name to prevent regex wildcard matching
    local key_name=$(echo "$key_name_raw" | sed 's/\./\\./g')

    if [ ! -f "$config_f" ]; then
        echo ""
        return
    fi

    # Get the first non-commented line matching the key
    local line_content=$(grep -E "^\s*${key_name}\s*:" "$config_f" | grep -v "^\s*#" | head -n 1)

    if [ -z "$line_content" ]; then
        echo ""
        return
    fi

    # Extract value part after colon, strip leading/trailing whitespace and comments
    local value_part=$(echo "$line_content" | sed -E 's/^[^:]*://' | sed -E 's/^[[:space:]]*//') # Get part after ':' and trim leading space
    value_part=$(echo "$value_part" | sed -E 's/\s*#.*$//') # Remove comments from value part
    value_part="${value_part#"${value_part%%[![:space:]]*}"}" # Trim leading whitespace again (more robust)
    value_part="${value_part%"${value_part##*[![:space:]]}"}" # Trim trailing whitespace

    # Remove quotes if present
    if [[ "$value_part" =~ ^\"(.*)\"$ ]] || [[ "$value_part" =~ ^\'(.*)\'$ ]]; then
        value_final="${BASH_REMATCH[1]}"
    else
        value_final="$value_part"
    fi

    echo "$value_final"
}


# --- OS Check ---
IS_MACOS=false
if [[ "$OSTYPE" == "darwin"* ]]; then
    IS_MACOS=true
    echo "[INFO] Detected macOS."
else
    echo "[INFO] Detected non-macOS system ($OSTYPE)."
fi
echo ""

# --- Homebrew Check & Update (macOS only) ---
if [ "$IS_MACOS" = true ]; then
    if ! command_exists brew; then
        echo "[WARNING] Homebrew not found."
        if prompt_user "Install Homebrew now?" true; then
            echo "Installing Homebrew... Please follow prompts if any."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            # Ensure Homebrew is in PATH after install (for current session)
            if [[ "$(uname -m)" == "arm64" ]]; then # Apple Silicon
                 eval "$(/opt/homebrew/bin/brew shellenv)"
            else # Intel
                 eval "$(/usr/local/bin/brew shellenv)"
            fi
            if ! command_exists brew; then echo "[ERROR] Homebrew installation failed."; exit 1; fi
            echo "Homebrew installed."
        else echo "[INFO] Skipping Homebrew install."; fi
    else
        echo "[OK] Homebrew is installed."
        if prompt_user "Update Homebrew and check formulas?" true; then
            echo "Updating Homebrew..."; brew update; echo "Homebrew updated.";
        else echo "[INFO] Skipping Homebrew update."; fi
    fi
    echo ""
fi

# --- Node.js & npm Check (for Windmill CLI) ---
echo "Checking for Node.js and npm..."
if ! command_exists node || ! command_exists npm; then
    echo "[WARNING] Node.js or npm not found, which is required for the Windmill CLI."
    if [ "$IS_MACOS" = true ] && command_exists brew; then
        if prompt_user "Install Node.js via Homebrew?" true; then
            echo "Installing Node.js..."
            brew install node
        fi
    else
        echo "Please install Node.js and npm from https://nodejs.org/"
    fi
fi

if command_exists npm && [ -f "$SCRIPT_DIR/package.json" ]; then
    echo "[OK] Node.js is installed and package.json found."
    if prompt_user "Install Node.js dependencies (for Windmill CLI)?" true; then
        echo "Running 'npm install'..."
        npm install
        if [ $? -eq 0 ]; then
            echo "[OK] Node.js dependencies installed successfully."
        else
            echo "[ERROR] 'npm install' failed. The Windmill CLI may not be available."
        fi
    fi
else
    echo "[INFO] Skipping Node.js dependency installation (npm not found or package.json missing)."
fi
echo ""

# --- Docker Check (for Windmill Services) ---
echo "Checking for Docker..."
if ! command_exists docker; then
    echo "[ERROR] Docker command not found. Docker is required to run Windmill."
    if [ "$IS_MACOS" = true ]; then
        echo "Please download and install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    else
        echo "Please install Docker for your system from: https://docs.docker.com/get-docker/"
    fi
    read -p "Press Enter to continue after Docker has been installed and started..."
fi

if command_exists docker; then
    if ! docker ps > /dev/null 2>&1; then
        echo "[ERROR] The Docker daemon is not running."
        echo "Please start Docker Desktop (macOS/Windows) or the Docker service (Linux) and then re-run this script."
        exit 1
    else
        echo "[OK] Docker is installed and the daemon is running."
    fi
fi
echo ""

# --- Windmill Workflow Engine Setup ---
echo "Windmill Workflow Engine Setup:"
echo "--------------------------------"

# Create required .env file for docker-compose if it doesn't exist
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "[INFO] Creating default .env file for Windmill..."
    cat > "$SCRIPT_DIR/.env" << EOF
DATABASE_URL=postgres://postgres:changeme@db:5432/windmill?sslmode=disable
WM_IMAGE=ghcr.io/windmill-labs/windmill:main
EOF
    echo "[OK] .env file created."
fi

# Create Caddyfile if it doesn't exist
if [ ! -f "$SCRIPT_DIR/Caddyfile" ]; then
    echo "[INFO] Creating default Caddyfile for Windmill reverse proxy..."
    cat > "$SCRIPT_DIR/Caddyfile" << EOF
{
    admin off
    auto_https off
}

:80 {
    reverse_proxy windmill_server:8000
}
EOF
    echo "[OK] Caddyfile created."
fi

# Dynamically generate wmill.yaml from config.yaml if it doesn't exist
if [ ! -f "$SCRIPT_DIR/wmill.yaml" ]; then
    echo "[INFO] Dynamically generating wmill.yaml from config.yaml settings..."
    WM_REMOTE=$(get_config_value "windmill.remote" "$CONFIG_FILE")
    WM_WORKSPACE=$(get_config_value "windmill.workspace" "$CONFIG_FILE")
    WM_TOKEN=$(get_config_value "windmill.api_token" "$CONFIG_FILE")

    cat > "$SCRIPT_DIR/wmill.yaml" << EOF
remotes:
  production:
    remote: "${WM_REMOTE:-http://localhost:80}"
    workspace: "${WM_WORKSPACE:-samson}"
    token: "${WM_TOKEN:-YOUR_WINDMILL_API_TOKEN_HERE}"
EOF
    echo "[OK] wmill.yaml created."
fi

# Verify Docker Compose shared volume configuration
echo "[INFO] Verifying Docker Compose configuration for shared volume..."
DOCKER_COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yaml"
if [ -f "$DOCKER_COMPOSE_FILE" ]; then
    HOST_SHARED_FOLDER_PATH=$(get_config_value "windmill_shared_folder" "$CONFIG_FILE")

    if [ -n "$HOST_SHARED_FOLDER_PATH" ]; then
        # Use grep's exit code directly. -q is for quiet, -- ensures paths with '-' are not treated as options.
        if grep -q -- "- ./${HOST_SHARED_FOLDER_PATH}:/wmill/data" "$DOCKER_COMPOSE_FILE"; then
            echo "[OK] Verified shared data volume ('${HOST_SHARED_FOLDER_PATH}') is correctly mounted in docker-compose.yaml."
        else
            echo ""
            echo "[!! CRITICAL CONFIGURATION ERROR !!]"
            echo "--------------------------------------------------------------------------"
            echo "The required shared volume for Windmill is MISSING from your 'docker-compose.yaml'."
            echo ""
            echo "Your config.yaml specifies the shared folder as: '$HOST_SHARED_FOLDER_PATH'"
            echo ""
            echo "CONSEQUENCE: Any AI-generated workflow that needs to read a file"
            echo "             (like a PDF) will fail with a 'FileNotFoundError'."
            echo ""
            echo "SOLUTION: Please open the 'docker-compose.yaml' file and add the"
            echo "          following line under the 'windmill_worker:' -> 'volumes:' section:"
            echo ""
            echo "          services:"
            echo "            windmill_worker:"
            echo "              # ... other settings"
            echo "              volumes:"
            echo "                - ./${HOST_SHARED_FOLDER_PATH}:/wmill/data  # <--- ADD THIS LINE"
            echo "                - /var/run/docker.sock:/var/run/docker.sock"
            echo "                # ... other volumes"
            echo ""
            echo "--------------------------------------------------------------------------"
            if [ "$NON_INTERACTIVE" = false ]; then
                 read -p "Press Enter to continue the script, but you MUST apply this fix for workflows to function."
            fi
        fi
    else
        echo "[WARNING] Could not read 'windmill_shared_folder' from '$CONFIG_FILE_RELATIVE'. Skipping Docker volume verification."
    fi
else
    echo "[WARNING] 'docker-compose.yaml' not found. Skipping volume verification."
fi
echo ""


# Check if Windmill containers are running
if docker ps | grep -q "windmill_server"; then
    echo "[OK] Windmill containers appear to be running."
else
    if prompt_user "Windmill services are not running. Start them now with Docker?" true; then
        echo "Starting Windmill services via docker-compose... This may take a minute."
        cd "$SCRIPT_DIR" && docker-compose up -d
        echo "Waiting for services to initialize..."
        sleep 20 # Give containers time to start up

        if docker ps | grep -q "windmill_server"; then
            echo "[OK] Windmill started successfully."
            echo "    > Access the Windmill UI at: http://localhost"
            echo "    > Default login: admin@windmill.dev / changeme"
        else
            echo "[ERROR] Windmill services failed to start. Please check the logs using:"
            echo "        cd \"$SCRIPT_DIR\" && docker-compose logs"
        fi
    fi
fi
echo ""

# Helper for Homebrew formulas
manage_brew_formula() {
    local formula_name="$1"
    local pretty_name="$2"
    local cmd_to_check="${3:-$formula_name}"

    if ! command_exists "$cmd_to_check" && ! (command_exists brew && brew list "$formula_name" &>/dev/null); then
        echo "[WARNING] $pretty_name ($formula_name) not found."
        if [ "$IS_MACOS" = true ] && command_exists brew; then
            if prompt_user "Install $pretty_name via Homebrew?" true; then
                echo "Installing $pretty_name..."
                brew install "$formula_name"
                if ! command_exists "$cmd_to_check" && ! (command_exists brew && brew list "$formula_name" &>/dev/null); then
                    echo "[ERROR] $pretty_name install failed."; return 1;
                fi
                echo "[OK] $pretty_name installed."
            else echo "[INFO] Skipping $pretty_name install."; return 1; fi
        else echo "Please install $pretty_name manually."; return 1; fi
    else
        echo "[OK] $pretty_name ($cmd_to_check) is installed/available."
        if [ "$IS_MACOS" = true ] && command_exists brew && brew list "$formula_name" &>/dev/null; then
            if brew outdated "$formula_name" &>/dev/null; then
                echo "[INFO] Newer $pretty_name version available via Homebrew."
                if prompt_user "Upgrade $pretty_name?" true; then
                    echo "Upgrading $pretty_name..."; brew upgrade "$formula_name"; echo "[OK] $pretty_name upgraded.";
                else echo "[INFO] Skipping $pretty_name upgrade."; fi
            else echo "[INFO] $pretty_name is up to date (Homebrew)."; fi
        elif command_exists "$cmd_to_check"; then
             echo "[INFO] $pretty_name found. Manual update may be needed if not managed by a system package manager."
        fi
    fi
    return 0
}

# --- Java Check & Update ---
echo "Checking for Java..."
JAVA_CMD_FOUND=false
if command_exists java && java -version 2>&1 | grep -q "version"; then
    JAVA_CMD_FOUND=true
fi

if ! $JAVA_CMD_FOUND ; then
    manage_brew_formula "openjdk" "OpenJDK (Java)" "java"
    if command_exists java && java -version 2>&1 | grep -q "version"; then
        echo "[OK] Java (OpenJDK) found post-install:"
        java -version
    else
        echo "[ERROR] Java still not found or configured after attempting install. signal-cli will likely fail."
        echo "       Ensure Java is correctly in your PATH. For Homebrew OpenJDK, symlinking might be needed:"
        echo "       e.g., sudo ln -sfn \$(brew --prefix openjdk)/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk"
        echo "       Or ensure your shell profile sources Homebrew's environment: eval \"\$($(brew --prefix)/bin/brew shellenv)\""
    fi
else
    echo "[OK] Java found:"
    java -version
    if [ "$IS_MACOS" = true ] && command_exists brew && brew list openjdk &>/dev/null; then
        if brew outdated openjdk &>/dev/null; then
            echo "[INFO] Newer OpenJDK version available via Homebrew."
            if prompt_user "Upgrade OpenJDK?" true; then
                echo "Upgrading OpenJDK..."; brew upgrade openjdk; echo "[OK] OpenJDK upgraded.";
            else
                echo "[INFO] Skipping OpenJDK upgrade.";
            fi
        else
            echo "[INFO] OpenJDK (Homebrew) is up to date.";
        fi
    fi
fi
echo ""

# --- ffmpeg Check & Update ---
echo "Checking for ffmpeg..."
manage_brew_formula "ffmpeg" "ffmpeg"
FFMPEG_PATH_CONFIG_SUGGESTION=$(command_exists ffmpeg && which ffmpeg || echo "ffmpeg_or_path")
echo ""

# --- Python Virtual Environment and Dependencies ---
echo "Python Setup:"
echo "-------------"

# --- NEW: Strict Python Version Check ---
# This project's requirements.txt is locked and requires a specific Python series.
# Using a newer Python (like 3.12 or 3.13) will cause 'pip install' to fail.
RECOMMENDED_PYTHON_VERSION="3.11"
PYTHON_CMD_TO_USE="python${RECOMMENDED_PYTHON_VERSION}"

if [[ -z "$VIRTUAL_ENV" ]]; then # Only check system python if we are not already in a venv
    if ! command_exists "$PYTHON_CMD_TO_USE"; then
        echo "[FATAL ERROR] Python ${RECOMMENDED_PYTHON_VERSION} is required but not found."
        echo "The project dependencies in 'requirements.txt' are not compatible with newer Python versions."
        echo "Your system must have Python ${RECOMMENDED_PYTHON_VERSION} available to proceed."
        echo ""
        echo "SOLUTION (macOS with Homebrew):"
        echo "1. Install Python ${RECOMMENDED_PYTHON_VERSION}: brew install python@${RECOMMENDED_PYTHON_VERSION}"
        echo "2. Once installed, re-run this setup script."
        echo ""
        echo "If you have Python ${RECOMMENDED_PYTHON_VERSION} installed but it is not in your PATH as '${PYTHON_CMD_TO_USE}',"
        echo "you must resolve this before continuing."
        exit 1
    fi
    echo "[OK] Found required '${PYTHON_CMD_TO_USE}'."
fi
# --- END: Strict Python Version Check ---

VENV_DIR_STANDARD=".venv"

# Check for venv and guide user
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "[OK] Currently in an active Python virtual environment: $VIRTUAL_ENV"
    # Optional: could add a check here to see if the venv python version is correct
    ACTIVE_PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    if [[ ! "$ACTIVE_PYTHON_VERSION" == "$RECOMMENDED_PYTHON_VERSION."* ]]; then
        echo "[WARNING] The active virtual environment is using Python ${ACTIVE_PYTHON_VERSION},"
        echo "          but this project requires Python ${RECOMMENDED_PYTHON_VERSION}.x for its dependencies."
        echo "          Installation will likely fail. It is recommended to exit, run 'deactivate',"
        echo "          remove the '.venv' directory, and let this script recreate it."
    fi
elif [ -d "$SCRIPT_DIR/$VENV_DIR_STANDARD" ] && [ -f "$SCRIPT_DIR/$VENV_DIR_STANDARD/bin/activate" ]; then
    echo "[INFO] A Python virtual environment named '$VENV_DIR_STANDARD' exists in '$SCRIPT_DIR'."
    echo "       IMPORTANT: Activate it before running the app: source $SCRIPT_DIR/$VENV_DIR_STANDARD/bin/activate"
else
    echo "[RECOMMENDATION] Using a Python virtual environment is highly recommended."
    if prompt_user "Create Python virtual environment '$SCRIPT_DIR/$VENV_DIR_STANDARD' using '${PYTHON_CMD_TO_USE}' now?" true; then
        echo "Creating virtual environment with '${PYTHON_CMD_TO_USE} -m venv'..."
        $PYTHON_CMD_TO_USE -m venv "$SCRIPT_DIR/$VENV_DIR_STANDARD"
        if [ $? -eq 0 ]; then
            echo "[OK] Virtual environment created. Activate it with: source $SCRIPT_DIR/$VENV_DIR_STANDARD/bin/activate"
            echo "     After activating, re-run this script to install dependencies into it."
        else
            echo "[ERROR] Failed to create virtual environment."
        fi
    else
        echo "[INFO] Skipping virtual environment creation. Dependencies will be installed globally if you proceed."
    fi
fi
echo ""

# Check for requirements.txt and install
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then # Corrected to lowercase 'r'
    echo "[INFO] Found 'requirements.txt' for Python dependencies."
    if [[ -n "$VIRTUAL_ENV" ]]; then
        if prompt_user "Install/update Python dependencies into the active venv now?" true; then
            echo "Installing/updating dependencies from requirements.txt... This may take several minutes."
            pip install -r "$SCRIPT_DIR/requirements.txt" # Corrected to lowercase 'r'
            PIP_EXIT_CODE=$?
            if [ $PIP_EXIT_CODE -eq 0 ]; then
                echo "[OK] Python dependencies installed successfully."
            else
                echo "[ERROR] 'pip install' failed with exit code $PIP_EXIT_CODE."
                echo "        The application will likely fail with a 'ModuleNotFoundError'."
                echo "        >>> THIS IS VERY LIKELY DUE TO AN INCOMPATIBLE PYTHON VERSION. <<<"
                echo "        This project requires Python ${RECOMMENDED_PYTHON_VERSION}.x. Please ensure your virtual environment was created with it."
                echo "        If the problem persists, review the pip errors above for other issues like missing system build tools."
            fi
        else
            echo "[WARNING] Skipping automatic dependency installation."
            echo "          You MUST install dependencies manually before running the application:"
            echo "          pip install -r $SCRIPT_DIR/requirements.txt"
        fi
    else
        echo "[WARNING] Not in an active Python virtual environment."
        echo "          Dependencies will NOT be installed automatically to avoid polluting your global Python environment."
        echo "          Please create and/or activate your virtual environment and re-run this script."
    fi
else
    echo "[ERROR] 'requirements.txt' not found at '$SCRIPT_DIR/requirements.txt'. Cannot install dependencies."
fi
echo ""

# --- Configuration File & Directory Setup ---
echo "Ensuring all configuration files and data directories exist..."

# 1. Create default config.yaml using the logic from config_loader.py
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[INFO] Main config.yaml not found. Creating default..."
    python -c "from src.config_loader import ensure_config_exists; ensure_config_exists()"
    if [ $? -eq 0 ]; then
        echo "[OK] Default config.yaml created. Please review and customize it."
    else
        echo "[ERROR] Failed to create default config.yaml."
    fi
fi

# 2. Create Streamlit config file
STREAMLIT_CONFIG_DIR="$SCRIPT_DIR/.streamlit"
STREAMLIT_CONFIG_FILE="$STREAMLIT_CONFIG_DIR/config.toml"
if [ ! -f "$STREAMLIT_CONFIG_FILE" ]; then
    echo "[INFO] Creating Streamlit config.toml..."
    mkdir -p "$STREAMLIT_CONFIG_DIR"
    echo '[server]' > "$STREAMLIT_CONFIG_FILE"
    echo 'fileWatcherType = "none"' >> "$STREAMLIT_CONFIG_FILE"
    echo "[OK] .streamlit/config.toml created."
fi

# 3. Create all data directories defined in the default config
if [ -f "$CONFIG_FILE" ]; then
    echo "[INFO] Creating data directories specified in config.yaml..."
    # This python snippet reads the yaml, finds all values under 'paths' that are directories, and prints them
    python -c "
import yaml, os
from pathlib import Path

config_path = '$CONFIG_FILE'
project_root = Path('$SCRIPT_DIR')
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

paths_to_create = cfg.get('paths', {})
for key, rel_path in paths_to_create.items():
    if isinstance(rel_path, str) and ('file' not in key):
        abs_path = project_root / rel_path
        print(f'Creating directory: {abs_path}')
        os.makedirs(abs_path, exist_ok=True)
"
    echo "[OK] Data directories checked/created."
fi
echo ""


# --- signal-cli Check, Install, Register, Verify ---
# ... (The rest of your script remains unchanged) ...
echo "signal-cli Setup:"
echo "-----------------"
SIGNAL_CLI_CMD="signal-cli"
SIGNAL_CLI_INSTALLED_PATH=""
manage_brew_formula "signal-cli" "signal-cli"
if command_exists signal-cli; then
    SIGNAL_CLI_INSTALLED_PATH=$(which signal-cli)
fi
SIGNAL_CLI_PATH_CONFIG_SUGGESTION="${SIGNAL_CLI_INSTALLED_PATH:-signal-cli_or_path}"

if command_exists "$SIGNAL_CLI_CMD" || [ -n "$SIGNAL_CLI_INSTALLED_PATH" ]; then
    EFFECTIVE_SIGNAL_CLI_CMD="${SIGNAL_CLI_INSTALLED_PATH:-$SIGNAL_CLI_CMD}"
    echo "[INFO] Will use '$EFFECTIVE_SIGNAL_CLI_CMD' for Signal operations."

    if [ -f "$CONFIG_FILE" ]; then
        echo "[INFO] Checking '$CONFIG_FILE' for Signal settings..."
        SAMSON_PHONE=$(get_config_value "samson_phone_number" "$CONFIG_FILE")
        SIGNAL_DATA_PATH_RAW=$(get_config_value "signal_cli_data_path" "$CONFIG_FILE")

        SIGNAL_DATA_PATH_EXPANDED=""
        if [ -n "$SIGNAL_DATA_PATH_RAW" ]; then
            # Expand tilde manually
            eval SIGNAL_DATA_PATH_EXPANDED="$SIGNAL_DATA_PATH_RAW" # Use eval for robust tilde expansion
            echo "[INFO] Using Signal data path from config: '$SIGNAL_DATA_PATH_RAW' (expanded to '$SIGNAL_DATA_PATH_EXPANDED')"
        else
            echo "[WARNING] 'signal_cli_data_path' not found or empty in $CONFIG_FILE. Defaulting to '~/.local/share/signal-cli'."
            SIGNAL_DATA_PATH_EXPANDED="$HOME/.local/share/signal-cli"
        fi

        if ! mkdir -p "$SIGNAL_DATA_PATH_EXPANDED"; then
             echo "[ERROR] Could not create Signal data directory: '$SIGNAL_DATA_PATH_EXPANDED'"
             echo "        Please check permissions or create it manually."
        else
             # Directory exists or was created
             :
        fi

        if [ -n "$SAMSON_PHONE" ]; then
            echo "[INFO] Samson bot phone number from config: '$SAMSON_PHONE'"

            echo "Checking if '$SAMSON_PHONE' is already registered with signal-cli at data path '$SIGNAL_DATA_PATH_EXPANDED'..."
            if "$EFFECTIVE_SIGNAL_CLI_CMD" -u "$SAMSON_PHONE" --config "$SIGNAL_DATA_PATH_EXPANDED" listDevices >/dev/null 2>&1; then
                echo "[OK] '$SAMSON_PHONE' appears to be already registered and accessible."
            else
                echo "[WARNING] '$SAMSON_PHONE' does not appear to be registered or accessible with current config."
                echo "          This could also be due to a new/uninitialized data directory or incorrect phone number in config."
                if prompt_user "Attempt to register '$SAMSON_PHONE' with signal-cli now?" false; then
                    echo "Ensure you have a DEDICATED number for this bot. DO NOT use your personal Signal number."
                    echo "You will receive a verification code via SMS or voice call to '$SAMSON_PHONE'."

                    CAPTCHA_TOKEN=""
                    REG_STATUS=1 # Assume registration is needed

                    if [ -f "$CAPTCHA_TOKEN_FILE" ]; then
                        CAPTCHA_TOKEN=$(cat "$CAPTCHA_TOKEN_FILE" | tr -d '\n\r')
                        if [ -n "$CAPTCHA_TOKEN" ]; then
                            echo "[INFO] Found existing CAPTCHA token in '$CAPTCHA_TOKEN_FILE'. Attempting registration with this token."
                            REG_OUTPUT_CAPTCHA=$("$EFFECTIVE_SIGNAL_CLI_CMD" -u "$SAMSON_PHONE" --config "$SIGNAL_DATA_PATH_EXPANDED" register --captcha "$CAPTCHA_TOKEN" 2>&1)
                            CURRENT_REG_STATUS=$?
                            if [ $CURRENT_REG_STATUS -ne 0 ]; then
                                echo "$REG_OUTPUT_CAPTCHA"
                                echo "[ERROR] Registration with pre-existing CAPTCHA token from '$CAPTCHA_TOKEN_FILE' failed (Exit code: $CURRENT_REG_STATUS)."
                                echo "        The token might be old or invalid. Please remove '$CAPTCHA_TOKEN_FILE' and try again."
                                CAPTCHA_TOKEN=""
                            else
                                echo "[INFO] Registration with pre-existing CAPTCHA token appears successful. Proceeding to verification..."
                                REG_STATUS=0
                            fi
                        else
                            echo "[WARNING] '$CAPTCHA_TOKEN_FILE' exists but is empty. Ignoring."
                        fi
                    fi

                    if [ $REG_STATUS -ne 0 ]; then # If pre-existing token didn't work or wasn't there
                        echo "Attempting initial registration for '$SAMSON_PHONE'..."
                        REG_OUTPUT=$("$EFFECTIVE_SIGNAL_CLI_CMD" -u "$SAMSON_PHONE" --config "$SIGNAL_DATA_PATH_EXPANDED" register 2>&1)
                        REG_STATUS=$?
                    fi

                    if [ $REG_STATUS -ne 0 ]; then
                        # Combine potential error outputs for CAPTCHA check
                        COMBINED_REG_ERROR_OUTPUT="${REG_OUTPUT_CAPTCHA:-}${REG_OUTPUT:-}"

                        if [[ "$COMBINED_REG_ERROR_OUTPUT" == *"Captcha required"* || "$COMBINED_REG_ERROR_OUTPUT" == *"CAPTCHA"* ]]; then
                            if [ -n "$REG_OUTPUT" ]; then echo "$REG_OUTPUT"; fi # Print initial error if it happened
                            if [ -n "$REG_OUTPUT_CAPTCHA" ] && [ "$REG_OUTPUT_CAPTCHA" != "$REG_OUTPUT" ]; then echo "$REG_OUTPUT_CAPTCHA"; fi # Print captcha error if different

                            echo ""
                            echo "---------------------------------------------------------------------------------"
                            echo "CAPTCHA REQUIRED FOR SIGNAL REGISTRATION:"
                            # ... (CAPTCHA instructions remain the same) ...
                            echo "1. Open this URL in your browser: https://signalcaptchas.org/registration/generate.html"
                            echo "   (If that fails, try: https://signalcaptchas.org/challenge/generate.html )"
                            echo "2. Solve the CAPTCHA displayed on the page."
                            echo "3. After solving, a button/link like 'Open Signal' or similar will appear."
                            echo "   RIGHT-CLICK on this button/link and select 'Copy Link Address'."
                            echo "4. The copied link will look like 'signalcaptcha://signal-hcaptcha...'. This is your token."
                            echo "5. Create a file named 'captcha_token.txt' in the Samson directory:"
                            echo "   $CAPTCHA_TOKEN_FILE"
                            echo "6. Paste the ENTIRE token (starting with 'signalcaptcha://') into this file and save it."
                            echo "   Ensure it's the only content in the file, with no extra lines or spaces."
                            echo "---------------------------------------------------------------------------------"
                            read -p "Once you have created and saved '$CAPTCHA_TOKEN_FILE' with the token, press Enter to continue..."

                            if [ -f "$CAPTCHA_TOKEN_FILE" ]; then
                                CAPTCHA_TOKEN=$(cat "$CAPTCHA_TOKEN_FILE" | tr -d '\n\r')
                                if [ -n "$CAPTCHA_TOKEN" ]; then
                                    echo "Attempting registration with CAPTCHA token from '$CAPTCHA_TOKEN_FILE'..."
                                    REG_OUTPUT_CAPTCHA_AGAIN=$("$EFFECTIVE_SIGNAL_CLI_CMD" -u "$SAMSON_PHONE" --config "$SIGNAL_DATA_PATH_EXPANDED" register --captcha "$CAPTCHA_TOKEN" 2>&1)
                                    REG_STATUS=$?
                                    if [ $REG_STATUS -ne 0 ]; then
                                        echo "$REG_OUTPUT_CAPTCHA_AGAIN"
                                        echo "[ERROR] signal-cli registration with CAPTCHA token from file failed (Exit code: $REG_STATUS)."
                                    else
                                        echo "[INFO] Registration with CAPTCHA token from file appears successful. Proceeding to verification..."
                                    fi
                                else
                                    echo "[ERROR] '$CAPTCHA_TOKEN_FILE' was found but is empty."
                                    REG_STATUS=1
                                fi
                            else
                                echo "[ERROR] '$CAPTCHA_TOKEN_FILE' not found."
                                REG_STATUS=1
                            fi
                        else # Not a CAPTCHA error, or other failure
                            if [ -n "$COMBINED_REG_ERROR_OUTPUT" ]; then
                                echo "$COMBINED_REG_ERROR_OUTPUT"
                            fi
                            echo "[ERROR] Signal registration failed (Exit code: $REG_STATUS). Reason might not be CAPTCHA. Please check logs or try manually."
                        fi
                    fi

                    if [ $REG_STATUS -eq 0 ]; then
                        echo "[INFO] Registration successful or initiated. Please enter the verification code you receive for '$SAMSON_PHONE':"
                        read -p "Verification code (e.g., 123-456): " VERIFY_CODE
                        if [ -n "$VERIFY_CODE" ]; then
                            echo "Verifying '$SAMSON_PHONE' with code '$VERIFY_CODE'..."
                            VERIFY_OUTPUT=$("$EFFECTIVE_SIGNAL_CLI_CMD" -u "$SAMSON_PHONE" --config "$SIGNAL_DATA_PATH_EXPANDED" verify "$VERIFY_CODE" 2>&1)
                            if [ $? -eq 0 ]; then
                                echo "[OK] '$SAMSON_PHONE' verified successfully!"
                                echo "Attempting to set profile name 'Samson Assistant'..."
                                "$EFFECTIVE_SIGNAL_CLI_CMD" -u "$SAMSON_PHONE" --config "$SIGNAL_DATA_PATH_EXPANDED" updateProfile --name "Samson"
                                if [ -f "$CAPTCHA_TOKEN_FILE" ]; then
                                    echo "[INFO] Removing temporary CAPTCHA token file: $CAPTCHA_TOKEN_FILE"
                                    rm "$CAPTCHA_TOKEN_FILE"
                                fi
                            else
                                echo "$VERIFY_OUTPUT"
                                echo "[ERROR] Verification failed. Please check the code and try manually."
                            fi
                        else
                            echo "[INFO] No verification code entered. Please complete verification manually."
                        fi
                    else
                        echo "[INFO] Skipping verification step due to registration failure or incomplete CAPTCHA process."
                    fi
                else
                    echo "[INFO] Skipping signal-cli registration. Please do it manually if needed."
                fi
            fi
        else
            echo "[WARNING] 'samson_phone_number' not found or empty in $CONFIG_FILE. Cannot automate Signal registration."
        fi
    else
        echo "[WARNING] '$CONFIG_FILE_RELATIVE' not found. Cannot check for Signal numbers or guide registration."
    fi
else
    echo "[ERROR] signal-cli command not found. Cannot proceed with Signal setup checks/registration."
fi
echo ""

# --- Optional - LM Studio for Local LLMs ---
echo "Local LLM Setup (LM Studio):"
echo "--------------------------------"
echo "[INFO] This project is configured to use LM Studio for local Large Language Models."
echo "[INFO] Please download and install from: https://lmstudio.ai/"

if command_exists curl; then
    echo "Checking if LM Studio server is running on default port (1234)..."
    if curl -s -o /dev/null http://localhost:1234/v1; then
        echo "[OK] LM Studio server is running and accessible."
        echo "       Please ensure you have downloaded and loaded the required models within the app."
    else
        echo "[WARNING] Could not connect to LM Studio server at http://localhost:1234/v1."
        echo "          After installing LM Studio, please start the local server from the UI."
    fi
else
    echo "[INFO] 'curl' not found. Cannot check for running LM Studio server. Please ensure it is running manually."
fi
echo ""


echo "Configuration Reminders:"
echo "------------------------"
echo "[ACTION] Ensure '$CONFIG_FILE_RELATIVE' is up-to-date:"
echo "  - tools.ffmpeg_path: Suggested value is '$FFMPEG_PATH_CONFIG_SUGGESTION'"
echo "  - signal.samson_phone_number: (Should be set for Signal to work)"
echo "  - signal.recipient_phone_number: (Your personal number for Samson to message)"
echo "  - signal.signal_cli_path: Suggested value is '$SIGNAL_CLI_PATH_CONFIG_SUGGESTION'"
echo "  - signal.signal_cli_data_path: (Should match what was used for registration, e.g., \"${SIGNAL_DATA_PATH_EXPANDED:-$HOME/.local/share/signal-cli}\")"
echo "  - windmill.remote: (Should be 'http://localhost:80' for this setup)"
echo "  - windmill.api_token: (Generate from Windmill UI: Settings -> Tokens)"
echo "  (Review all other paths, model names, API keys, etc.)"
echo ""
echo "[ACTION] After generating an API token from the Windmill UI, update 'wmill.yaml'"
echo "         so the Windmill command-line tool (wmill) can authenticate."
echo ""

echo "Setup script finished."
echo "----------------------"
echo "Review output for any errors or manual steps still required."
echo "If you created a new virtual environment, activate it ('source .venv/bin/activate') and re-run this script to install dependencies."

# ... (The final "HOW TO RUN" section is good as is) ...

echo ""
echo "========================================================================"
echo "      >>> IMPORTANT: HOW TO RUN THE APPLICATION <<<"
echo "========================================================================"
echo ""
echo "1. ENSURE DOCKER & WINDMILL ARE RUNNING."
echo "   If you opted to start Windmill, make sure Docker Desktop is running."
echo "   You can verify services are up with: cd \"$SCRIPT_DIR\" && docker-compose ps"
echo ""
echo "2. ACTIVATE THE VIRTUAL ENVIRONMENT:"
echo "   From the Samson project directory, run:"
echo "   $ source .venv/bin/activate"
echo ""
echo "3. VERIFY YOUR ENVIRONMENT (Optional but Recommended):"
echo "   After activating, your prompt should change. Then run:"
echo "   $ which python"
echo "   The output should point to the python executable INSIDE your project's .venv/bin/ directory."
echo ""
echo "4. RUN THE APPLICATION:"
echo "   To run the main orchestrator:"
echo "   $ python main_orchestrator.py"
echo ""
echo "   To run the Streamlit GUI:"
echo "   $ streamlit run gui.py"
echo ""
echo "========================================================================"