import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
APP_HOST = os.getenv("APP_HOST") or os.getenv("FLASK_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT") or os.getenv("FLASK_PORT", "8000"))
