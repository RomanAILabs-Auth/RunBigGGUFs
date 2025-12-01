README: Adaptive LLM Runtime Tuner (Spacetime LLM Tuner v2)

🌌 Spacetime LLM Tuner v2

Adaptive LLM Runtime Tuner (FIXED & UPGRADED)

The Spacetime LLM Tuner v2 is a specialized, production-grade tool designed for dynamically optimizing the performance of Large Language Models (LLMs) served via llama-cpp-python (GGUF format). It uses an Online Low-Rank Model (a form of adaptive PCA) to analyze real-time system telemetry and instantly adjust core LLM runtime parameters for peak efficiency.

This version is a significant upgrade, featuring REAL dynamic runtime adjustments previously unavailable in older LLM frontends.

✨ Key Features

    True Dynamic Tuning: Adjusts core parameters (n_batch, n_ctx) at runtime without requiring a model reload (requires llama-cpp-python >= 0.2.85).

    Adaptive Performance Scoring: Utilizes an Online Low-Rank Model to calculate a real-time Tuning Score (0-100) based on CPU load, memory usage, I/O rates, and tokens-per-second (TPS).

    Optimized Resource Allocation: Maps the Tuning Score to tiered performance settings, dynamically adjusting context size (n_ctx), batch size (n_batch), and thread count.

    CPU Affinity Binding: Pins the LLM process to the fastest physical CPU cores to minimize context switching overhead and ensure rock-solid performance (Non-Windows systems).

    Real KV Cache Pruning: Executes a true KV cache clear using llm.reset() when performance throttling is required.

    GUI Interface: A simple, modern tkinter-based interface for loading GGUF models, observing the tuning score, and interacting with the LLM.

    Auto-Dependency Install: Automatically installs necessary libraries (numpy, scipy, llama-cpp-python) with CPU optimizations.

🛠️ Prerequisites

    Python 3.8+

    A GGUF format Large Language Model file (e.g., model.gguf).

🚀 Installation and Usage

1. Save the File

Save the provided script content as LatestModelTuner.py.

2. Run the Script

Execute the file from your terminal. The script will automatically check for and install the required dependencies (including llama-cpp-python with optimized CPU flags).
Bash

python3 LatestModelTuner.py

3. Using the GUI

    Load Model: Click the "Load GGUF" button and select your .gguf model file.

    Observe Tuning: The main bar and label at the top will dynamically update, showing:

        Tuning Score: The real-time adaptive performance rating.

        Status: The current performance tier (e.g., "Peak Performance," "Balanced").

        Applied Params: The current n_batch and n_ctx applied to the LLM.

    Generate Text: Enter your prompt in the input box and press "Send" or Ctrl+Return. The tuning system will continue to adapt in the background even during generation.

⚙️ Configuration

The core tuning logic is managed by the Config class within the file. You can adjust the sensitivity of the adaptive model and the performance tiers here:
Python

class Config:
    # ... other settings

    @staticmethod
    def get_tuning_tiers(score: int) -> Dict[str, Any]:
        cpu_physical = psutil.cpu_count(logical=False) or 1
        # Defines the mapping from the 0-100 score to LLM parameters
        if score > 90:
            return {"n_batch": 4096, "n_ctx": 32768, "threads": max(4, cpu_physical), "rope_scale": 1.00}
        if score > 70:
            return {"n_batch": 3072, "n_ctx": 24576, "threads": max(3, cpu_physical - 1), "rope_scale": 1.00}
        # ... and so on for lower scores
        return {"n_batch": 1024, "n_ctx": 12288, "threads": 2, "rope_scale": 0.95}

🛑 Shutting Down

Use the "Unload" button or close the application window. The _on_close handler will ensure the LLM resources are properly released, and the background telemetry thread is cleanly shut down.
