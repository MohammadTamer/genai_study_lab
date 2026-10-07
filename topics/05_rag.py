"""
05_rag.py
Topic: Retrieval-Augmented Generation (RAG) using InMemoryVectorStore.
Run: python topics/05_rag.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()


def load_docs():
    """Load documents from data/ or fallback to sample documents."""
    data_dir = Path("data")
    if data_dir.exists() and list(data_dir.glob("*.md")):
        return [
            Document(page_content=f.read_text(encoding="utf-8"), metadata={"source": f.name})
            for f in sorted(data_dir.glob("*.md"))
        ]
    return [
        Document(
            page_content=(
                "Company Policy:\n"
                "Employees receive 25 days of paid annual leave per year.\n"
                "Standard working hours are 9:00 AM to 5:00 PM."
            ),
            metadata={"source": "policy.md"},
        ),
        Document(
            page_content=(
                "Benefits Guide:\n"
                "Every engineer has an annual education budget of $1,500.\n"
                "Health insurance covers optical and dental care."
            ),
            metadata={"source": "benefits.md"},
        ),
    ]


def format_docs(docs):
    return "\n\n".join(f"[{d.metadata.get('source', 'doc')}]: {d.page_content}" for d in docs)


def main():
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=0.0)
    embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))

    # 1. Load and split documents
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} documents, split into {len(chunks)} chunks.")

    # 2. Build local vector store and retriever
    try:
        store = InMemoryVectorStore(embedding=embeddings)
        store.add_documents(chunks)
        retriever = store.as_retriever(search_kwargs={"k": 2})
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        print("Make sure Ollama is running and nomic-embed-text is installed.")
        return

    # 3. Grounded RAG prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer the question ONLY from the provided context.\n"
            "If the information is not in the context, respond: 'Information not found.'\n\n"
            "Context:\n{context}",
        ),
        ("human", "Question: {question}"),
    ])

    chain = (
        RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "How many annual leave days do employees get?",
        "What is the annual education budget?",
        "What is the company dress code on Friday?",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        try:
            answer = chain.invoke(q)
            print(f"Answer: {answer.strip()}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
