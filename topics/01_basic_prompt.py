"""
01_basic_prompt.py
Topic: Basic & Zero-Shot Prompting using ChatPromptTemplate.
Run: python topics/01_basic_prompt.py
"""

import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def main():
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
    llm = ChatOllama(model=model_name, temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI tutor. Explain concepts briefly with a simple example."),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    question = "What is Retrieval-Augmented Generation (RAG)?"
    print(f"Question: {question}\n")

    try:
        response = chain.invoke({"question": question})
        print("Response:\n" + response)
    except Exception as e:
        print(f"Error running model: {e}")
        print("Make sure Ollama is running (e.g., ollama run llama3.1).")


if __name__ == "__main__":
    main()
