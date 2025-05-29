import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PDF_FOLDER = "./data/pdfs"
    VECTOR_DB_PATH = "./data/vector_db"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

config = Config()
