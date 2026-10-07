"""
05b_rag_classic_chains.py
Topic: Classic LangChain RAG style using RetrievalQA & ConversationalRetrievalChain with Memory.
Uses the same local model (Ollama) as 05_rag.py.

Run:
python topics/05b_rag_classic_chains.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()


def main():
    # 1. Same local model and embeddings used in 05_rag.py
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=0.0)
    embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))

    # 2. Document Loading using TextLoader
    filename = Path("data/company_guide.md")
    if not filename.exists():
        # Fallback if file is missing
        filename = Path("sample_guide.txt")
        filename.write_text(
            "Company Mobile Policy:\n"
            "Employees may use company mobile devices for business purposes.\n"
            "Points: 1. Keep devices password protected. 2. Do not install unauthorized apps.\n"
            "Aim: Ensure data security and smooth business communication.",
            encoding="utf-8",
        )

    loader = TextLoader(str(filename), encoding="utf-8")
    documents = loader.load()

    # 3. Splitting with CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Chunks count: {len(texts)}")

    # 4. Vector Store ingestion
    try:
        docsearch = InMemoryVectorStore.from_documents(texts, embeddings)
        print("Document ingested successfully.")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        print("Make sure Ollama is running and nomic-embed-text is available.")
        return

    retriever = docsearch.as_retriever(search_kwargs={"k": 2})

    # -------------------------------------------------------------
    # Part 1: Classic RetrievalQA Chain (Single Turn)
    # -------------------------------------------------------------
    print("\n--- Part 1: Classic RetrievalQA (Single Turn) ---")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )

    query = "What is the annual leave and training budget policy?"
    print(f"Query: {query}")
    try:
        res = qa.invoke(query)
        answer = res.get("result", res) if isinstance(res, dict) else res
        print(f"Answer:\n{answer}\n")
    except Exception as e:
        print(f"Error in RetrievalQA: {e}")

    # -------------------------------------------------------------
    # Part 2: ConversationalRetrievalChain with Memory (Multi Turn)
    # -------------------------------------------------------------
    print("--- Part 2: ConversationalRetrievalChain with Memory ---")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    conv_qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
    )

    # Multi-turn conversation demonstrating follow-up questions
    dialogue_questions = [
        "What is the policy on remote work?",
        "How many days are permitted?",  # Follow-up question relying on memory
        "Who needs to approve it?",       # Follow-up question relying on memory
    ]

    for q in dialogue_questions:
        print(f"User: {q}")
        try:
            result = conv_qa.invoke({"question": q})
            print(f"Assistant: {result['answer'].strip()}\n")
        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    main()
