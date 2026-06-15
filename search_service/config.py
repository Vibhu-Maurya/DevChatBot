import os
from dotenv import load_dotenv

# Load env variables from .env file if present
# Check both search_service/ directory and project root
env_paths = [
    os.path.join(os.path.dirname(__file__), ".env"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
]
for path in env_paths:
    if os.path.exists(path):
        load_dotenv(path)
        break
else:
    load_dotenv()

class Settings:
    # Server configuration
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    # Qdrant configuration
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_PATH: str = os.getenv("QDRANT_PATH", "../crawler/qdrant_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "docs_v1")

    # LLM configurations
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.5-flash")

settings = Settings()
