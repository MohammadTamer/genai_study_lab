from pathlib import Path
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .model import embeddings

def load_markdown_documents(data_dir="data"):
    docs = []
    for path in sorted(Path(data_dir).glob("*.md")):
        docs.append(Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.name},
        ))
    return docs

def build_vector_store(data_dir="data"):
    raw_docs = load_markdown_documents(data_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        add_start_index=True,
    )
    chunks = splitter.split_documents(raw_docs)
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(chunks)
    return store, chunks

def get_retriever(data_dir="data", k=4):
    store, chunks = build_vector_store(data_dir)
    retriever = store.as_retriever(search_kwargs={"k": k})
    return retriever, store, chunks
