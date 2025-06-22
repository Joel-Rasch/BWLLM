"""
Configuration module for the RAG system
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Paths
    MARKDOWN_FOLDER = "./Extrahierter_Text_Markdown"
    VECTOR_STORE_PATH = "./faiss_index"
    
    # Text processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Models
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gemini-2.5-flash"
    LLM_TEMPERATURE = 0.1
    
    # Retrieval
    TOP_K = 5
    MAX_QUERY_VARIANTS = 4
