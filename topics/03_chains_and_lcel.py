"""
03_chains_and_lcel.py
Topic: LangChain Expression Language (LCEL) and Sequential Runnables.
Run: python topics/03_chains_and_lcel.py
"""

import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()


def normalize_input(raw_input):
    """Clean extra spaces and format as dictionary for the prompt."""
    if isinstance(raw_input, dict):
        topic = raw_input.get("topic", "")
    else:
        topic = str(raw_input)
    return {"topic": topic.strip().title()}


def main():
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
    llm = ChatOllama(model=model_name, temperature=0.3)
    parser = StrOutputParser()

    # Chain 1: Generate a startup idea
    prompt_idea = ChatPromptTemplate.from_messages([
        ("system", "Suggest one startup idea for this industry in under 15 words."),
        ("human", "{topic}"),
    ])
    idea_chain = RunnableLambda(normalize_input) | prompt_idea | llm | parser

    # Chain 2: Generate a short slogan for the idea
    prompt_slogan = ChatPromptTemplate.from_messages([
        ("system", "Write a 5-word catchy slogan for this startup idea."),
        ("human", "{idea}"),
    ])
    slogan_chain = prompt_slogan | llm | parser

    user_industry = "  artificial intelligence in healthcare  "
    print(f"Input industry: '{user_industry}'")

    try:
        startup_idea = idea_chain.invoke(user_industry)
        print(f"Generated Idea: {startup_idea.strip()}")

        slogan = slogan_chain.invoke({"idea": startup_idea})
        print(f"Generated Slogan: {slogan.strip()}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running.")


if __name__ == "__main__":
    main()
