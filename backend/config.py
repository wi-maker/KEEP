# Config settings for KEEP backend
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in the same directory as this file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    # API
    PROJECT_NAME: str = "KEEP API"
    API_V1_STR: str = "/api/v1"
    
    # Database
    # FORCE system temp for DB to avoid file watchers (ignores .env value)
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(tempfile.gettempdir(), 'keep.db')}")
    
    # ===========================================
    # AI PROVIDERS (Priority: OpenRouter -> Gemini)
    # ===========================================
    
    # Primary: OpenRouter API (access to 400+ models)
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # Fallback: Google Gemini (also used for embeddings)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
    
    # Embedding model (always uses Google API for ChromaDB compatibility)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
    
    # Provider priority (comma-separated: openrouter,gemini)
    AI_PROVIDER_PRIORITY: str = os.getenv("AI_PROVIDER_PRIORITY", "openrouter,gemini")
    
    # ChromaDB
    # FORCE system temp for ChromaDB to avoid file watchers (ignores .env value)
    CHROMA_PERSIST_DIR: str = os.path.join(tempfile.gettempdir(), "keep_chroma_db")
    
    # File Storage
    # FORCE system temp directory to avoid triggering file watchers (ignores .env value)
    UPLOAD_DIR: str = os.path.join(tempfile.gettempdir(), "keep_uploads")
    
    # Auth
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 43200  # 30 days
    
    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    
    def __init__(self):
        # Validate at least one AI provider is configured
        if not self.OPENROUTER_API_KEY and not self.GOOGLE_API_KEY:
            raise ValueError(
                "No AI provider configured. Please set at least one of:\n"
                "- OPENROUTER_API_KEY (primary provider)\n"
                "- GOOGLE_API_KEY (fallback provider + embeddings)\n"
                "in your .env file."
            )
        
        # Warn if no Google API key (needed for embeddings)
        if not self.GOOGLE_API_KEY:
            import logging
            logging.warning(
                "GOOGLE_API_KEY not set. Embeddings will not work. "
                "The RAG pipeline requires Google API for ChromaDB embeddings."
            )

settings = Settings()
