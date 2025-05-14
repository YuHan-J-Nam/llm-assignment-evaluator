"""
Configuration settings for the RAG system.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Text Embedding Configuration
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI embedding model
CHUNK_SIZE = 1000  # Size of text chunks for processing
CHUNK_OVERLAP = 200  # Overlap between chunks

# Vector Database Configuration
VECTOR_DB_TYPE = "chroma"  # Options: "chroma", "faiss", etc.
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "vector_db")

# PDF Processing
PDF_DATA_FOLDER = os.path.join(os.path.dirname(__file__), "pdf_data")

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(os.path.dirname(__file__), "rag_system.log")

# Quality Check Configuration
SIMILARITY_THRESHOLD = 0.75  # Threshold for similarity checks
QUALITY_CHECK_SAMPLES = 5  # Number of samples to check for quality 