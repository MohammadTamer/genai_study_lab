"""
04_parallel_chain.py
Topic: Parallel Execution with RunnableParallel.
Run: python topics/04_parallel_chain.py
"""

import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()


def main():
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
    llm = ChatOllama(model=model_name, temperature=0.2)
    parser = StrOutputParser()

    # Branch 1: One-sentence summary
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the text in exactly one sentence."),
        ("human", "{text}"),
    ])
    summary_chain = summary_prompt | llm | parser

    # Branch 2: Extract 3 key bullet points
    points_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract 3 key takeaways from the text as short bullet points."),
        ("human", "{text}"),
    ])
    points_chain = points_prompt | llm | parser

    # Combine both branches in parallel
    parallel_chain = RunnableParallel({
        "summary": summary_chain,
        "takeaways": points_chain,
    })

    sample_text = (
        "AI agents are software systems that use large language models to reason, "
        "plan tasks, and call external tools to achieve specific goals autonomously."
    )

    print(f"Input Text:\n{sample_text}\n")
    print("Running parallel branches (summary + takeaways)...")

    try:
        results = parallel_chain.invoke({"text": sample_text})
        print("\nSummary:\n" + results["summary"].strip())
        print("\nTakeaways:\n" + results["takeaways"].strip())
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running.")


if __name__ == "__main__":
    main()
