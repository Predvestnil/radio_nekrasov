# config.py
import os

# Use environment variables or defaults
OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://localhost:8001")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8002")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://localhost:8003")

# Directory for generated audio files (relative to tts_service.py)
GENERATED_AUDIO_DIR = "generated_audio"
# Static path prefix for accessing generated audio from the webapp
GENERATED_AUDIO_STATIC_PATH = "/static/generated_audio"