"""
Simple configuration file for the RAG system
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "Geschaeftsberichte"
OUTPUT_DIR = BASE_DIR / "Extrahierter_Text_Markdown"

# Vector database files
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "chunks_metadata.pkl"

# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-1.5-flash"

# Processing settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_K_RESULTS = 5

# API keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = "pipeline.log"

# FAISS settings
FAISS_INDEX_TYPE = "IndexFlatIP"  # for cosine similarity

def get_api_key():
    """
    Get Google API key from environment variables.
    
    Returns:
        str or None: API key if found, None otherwise
    """
    return GOOGLE_API_KEY or GEMINI_API_KEY

def ensure_directories():
    """
    Ensure required directories exist.
    Creates DATA_DIR and OUTPUT_DIR if they don't exist.
    """
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)