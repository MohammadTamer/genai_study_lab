from langchain_ollama import ChatOllama, OllamaEmbeddings
from .config import OLLAMA_MODEL, OLLAMA_EMBED_MODEL

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
