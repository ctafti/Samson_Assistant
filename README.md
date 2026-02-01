

# Samson AI Assistant

## Overview

Samson is an automated audio ingestion and analysis pipeline designed to convert a constant stream of raw recordings into structured, queryable data. IUnlike cloud-based transcription services, Samson operates as a fully autonomous, locally hosted pipeline optimized for Apple Silicon. It monitors synchronized audio directories in real-time, ingesting raw recordings to generate speaker-attributed transcripts, semantic embeddings, and actionable task data with near-zero latency.

Samson is architected on a "privacy-first" philosophy, ensuring sensitive conversations never leave local hardware. By orchestrating a suite of advanced machine learning models—including Voice Activity Detection (VAD), Diarization, and Vector Analysis—Samson functions without external dependencies. Beyond standard transcription, the system serves as a persistent contextual memory. It utilizes vector databases to classify conversations into long-term projects ("Matters") and automatically extracts structured project management artifacts, effectively bridging the gap between ephemeral dialogue and concrete data.

## System Architecture

The application utilizes a hub-and-spoke architecture managed by a **Main Orchestrator**. This Python-based orchestrator manages concurrency using threaded services, effectively decoupling data ingestion from computationally intensive machine learning tasks.

### 1. Audio Processing Pipeline
The core of the system is a linear, multi-stage pipeline designed to maximize throughput while maintaining data consistency.

*   **Ingestion:** A **Folder Monitor** service utilizes `watchdog` to detect file system events in specific directories (e.g., Syncthing targets). It implements file stability checks and sequence enforcement to ensure chunked audio files are processed in strict chronological order.
*   **Voice Activity Detection (VAD):** Pre-processing energy filters analyze the waveform to detect speech activity, preventing the engagement of heavy ML models for silent audio segments.
*   **Speech-to-Text (STT):** The system implements an abstraction layer for STT engines. It defaults to **Parakeet MLX** for hardware-accelerated transcription on macOS, with an automated fallback to **OpenAI Whisper** for compatibility.
*   **Speaker Diarization:** **Pyannote Audio** is employed to segment the audio stream by speaker turns. A custom consolidation algorithm post-processes these segments to bridge micro-silences and merge rapid turn-taking, resulting in a coherent conversational flow.
*   **Speaker Identification:** A custom implementation using **SpeechBrain** generates 192-dimensional embeddings for every speaker segment. These vectors are queried against a local **FAISS (Facebook AI Similarity Search)** index to identify enrolled speakers in real-time.
*   **Voice Command Parsing:** The transcript is scanned for specific trigger phrases. If detected, the relevant text chunk is passed to a local Large Language Model (LLM) to parse the syntax and execute system commands vocally (e.g., "Samson, force the matter to Project Phoenix").

### 2. Adaptive Speaker Intelligence
Samson includes a background service dedicated to the unsupervised evolution of speaker profiles, allowing identification accuracy to improve over time without manual model retraining.

*   **Profile Evolution:** During scheduled maintenance windows, the system performs a batch recalculation of speaker profiles. It aggregates high-confidence audio segments collected across different recording contexts (VoIP vs. In-Person) and re-calculates the master vector embedding using a recency-weighted average.
*   **Dynamic Thresholds:** The system implements a feedback loop based on user corrections. When a user manually reassigns a speaker segment via the UI, the system recalculates the optimal similarity confidence threshold for that specific speaker and context.
*   **Role Inference:** An LLM periodically analyzes the aggregate dialogue history of a speaker to infer their role (e.g., "Interviewer," "Subject Matter Expert"), providing semantic metadata for the knowledge graph.

### 3. Contextual Awareness Engine
Samson maintains a persistent state of the active "Matter" (project or topic), ensuring that isolated audio chunks are attributed to the correct long-term initiative.

*   **Semantic Analysis:** Incoming transcripts are vectorized using **SentenceTransformers**. The system calculates the cosine similarity between the conversation vector and a database of known Matter definitions.
*   **Context Hysteresis:** To prevent rapid context switching during brief digressions, the system implements "stickiness" logic. A new Matter must exceed the similarity score of the current active Matter by a significant delta before a context switch is triggered.
*   **Smart Flagging:** If the similarity scores of two different Matters are within a narrow margin of error, the system flags the segment as a "Conflict." These flags are serialized to a review queue for human adjudication.

### 4. Task Intelligence Manager
This subsystem functions as a structured data extractor, converting unstructured dialogue into project management artifacts.

*   **Extraction:** An LLM processes transcripts to identify commitments, deadlines, and delegations based on semantic intent.
*   **Vector-Based Deduplication:** Extracted tasks are embedded and stored in a secondary FAISS index. Before creating a new task, the system queries this index to determine if the detected item is an update to an existing task or a duplicate, preventing database clutter.
*   **Version Control:** Tasks utilize an append-only version history, tracking how the parameters of a deliverable evolve over multiple conversations.

### 5. Workflow Automation (Windmill Integration)
Samson integrates with **Windmill**, an open-source workflow engine, to execute arbitrary code based on natural language requests.

*   **Generative Workflows:** Users can request complex data operations via voice or text (e.g., "Summarize last week's tasks and email them to the client"). The system uses an LLM to generate a Python script, validate dependencies, and push it to Windmill for execution.
*   **Code-as-Action:** This architecture allows the platform to extend its own capabilities dynamically at runtime without requiring core codebase deployments.

### 6. Event Scheduling Service
To handle temporal commands and deferred actions, the system includes a dedicated **Scheduler Service**.
*   **Event Loop:** A background thread polls an atomic `events.jsonl` ledger to identify actions that must occur at specific timestamps.
*   **Deferred Execution:** This service enables temporal commands such as "Set the matter to Project X at 2:00 PM," ensuring context switching occurs automatically at the scheduled time, independent of user interaction.

## Reliability and Redundancy Systems

A primary focus of the system design was ensuring data integrity, fault tolerance, and redundant access patterns for critical operations.

### Dual-Path Command Execution
To ensure responsiveness and robustness, system commands are handled via two redundant pathways:
*   **File-Based Queue:** Internal services and the GUI write command objects (JSON) to a monitored directory. A dedicated thread processes these commands sequentially. This ensures that even if the API layer fails or is overloaded, commands are persisted to disk and processed eventually.
*   **API Server:** A Flask-based REST API runs in parallel to accept commands from external tools (such as Windmill Docker containers) that cannot access the host file system directly.

### Data Persistence Redundancy
The system implements a dual-write strategy for transcript data to protect against corruption and ensure accessibility:
*   **Structured Store:** All data is serialized into daily JSONL (JSON Lines) files. This format allows for efficient appending and machine parsing without loading the entire dataset into memory.
*   **Human-Readable Store:** Simultaneously, a "Master Transcript" text file is generated and appended to. This serves as a fail-safe, format-agnostic backup that remains readable even if the structured data parsers fail.
*   **Atomic Write Operations:** All database interactions utilize an atomic write pattern (write to temporary file, flush, OS rename) to prevent partial data writes or file corruption in the event of power loss.
*   **Absolute Timestamp Reconstruction:** The logging manager calculates absolute UTC timestamps for every individual word based on the source file creation time and audio offset. This ensures temporal accuracy is maintained even if files are processed out of order or renamed.

### Process Management
*   **Signal Interface Resilience:** The external messaging bridge includes health-check logic to detect zombie `signal-cli` processes or connection drops. It automatically restarts the interface with exponential backoff strategies upon failure.
*   **Concurrency Locking:** A custom file-locking mechanism utilizing `filelock` ensures that the Main Orchestrator, Background Workers, and GUI components can access shared data stores simultaneously without race conditions.
*   **Automated Health Monitoring:** A dedicated health check utility runs daily to validate data integrity. It scans for inconsistencies such as orphaned speaker IDs, missing transcript files, or pending flags that have exceeded their review window, generating a report for the administrator.

## User Interface (Cockpit)

The user interface is built with **Streamlit**, offering a "Human-in-the-loop" workflow for data validation and system management.

*   **Transcript Correction:** A custom component allows for precise, word-level timestamp corrections and speaker reassignment. It supports multi-word selection and batch editing.
*   **Secure Media Delivery:** To comply with browser security sandboxing, the platform spins up an ephemeral, local HTTP server solely for streaming audio snippets to the frontend interface, preventing direct file system access vulnerabilities.
*   **Flag Review:** Ambiguous events (unknown speakers, conflicting topics) are presented in a unified queue. Users can resolve these with single-click actions that trigger backend model updates.
*   **Speaker Management:** Users can merge profiles, rename speakers, and view long-term participation statistics.
*   **Workflow Management:** A dedicated interface for managing AI-generated Windmill scripts, allowing users to inspect code, rename workflows, and manually trigger executions.

## External Integrations

*   **Signal Messenger:** A bi-directional interface allows the system to push notifications to the user's phone and receive commands remotely via a CLI bridge. It supports interactive sessions, such as reviewing flags or confirming entity resolution via chat.
*   **Local LLM Inference:** The system abstraction layer allows it to interface with various local inference servers (e.g., LM Studio or Ollama) or cloud providers (Anthropic, Google), ensuring flexibility in model selection.

## Setup

### Prerequisites
Before installing Samson, ensure your system meets the following requirements:
* **Operating System**: macOS (Apple Silicon optimized)
* **Python**: Version 3.11
* **System Tools**:
    * `ffmpeg` (Required for audio processing)
    * `signal-cli` (Required for Signal messaging integration)
    * `docker` & `docker-compose` (Required for the Windmill workflow engine)
* **LLM Provider**: A local inference server like **LM Studio** or **Ollama** running and accessible (default configuration expects `http://localhost:1234/v1`).

### Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/ctafti/Samson_Assistant.git](https://github.com/ctafti/Samson_Assistant.git)
    cd Samson_Assistant
    ```

2.  **Run the Setup Script**
    The included `setup.sh` script automates the environment initialization and installs the required Python dependencies defined in `requirements.txt`.
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

3.  **Start Infrastructure Services**
    Samson uses Docker to run the Windmill workflow engine and its database.
    ```bash
    docker-compose up -d
    ```

### Configuration

1.  **Locate Configuration**: Navigate to the `config/` directory. Ensure `config.yaml` exists.
2.  **Edit Settings**: Open `config/config.yaml` and customize the following key parameters:
    * **Paths**: Update `monitored_audio_folder` to point to the directory where your new audio files will be synced (e.g., your Syncthing target folder).
    * **Signal**: Set the `samson_phone_number` (the bot's number) and `recipient_phone_number` (your admin number).
    * **LLM**: Verify that the `base_url` under the `llm` section matches your local provider.
    * **Tools**: Verify the `ffmpeg_path` and `signal_cli_path` if they are not in your system's global PATH.

### Running Samson

The system requires two concurrent processes to function: the backend orchestrator and the frontend user interface.

**1. Start the Backend Orchestrator**
This service monitors the file system, processes incoming audio, and executes commands.
```bash
python main_orchestrator.py
```
**1. Start the Frontend Streamlit Application**
In a separate terminal window, start the Streamlit interface:
```bash
streamlit run gui.py
```

You can then access the Samson Cockpit in your browser at http://localhost:8501.
## Technical Stack

*   **Language:** Python 3.11
*   **Machine Learning:** PyTorch, FAISS, SentenceTransformers, Pyannote, SpeechBrain, MLX
*   **Orchestration:** Threading, Watchdog, Subprocess management, FileLock
*   **Interface:** Streamlit
*   **Data:** JSONL (Logs/Events), SQLite (Vector Metadata), FAISS (Vector Data)
*   **Workflow:** Windmill (Dockerized)