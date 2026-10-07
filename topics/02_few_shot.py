"""
02_few_shot.py
Topic: Few-Shot Prompting (In-Context Learning) using FewShotChatMessagePromptTemplate.
Run: python topics/02_few_shot.py
"""

import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def main():
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
    llm = ChatOllama(model=model_name, temperature=0.0)

    # 1. Provide few-shot demonstration examples
    examples = [
        {"input": "I really love using this software, it saves me hours!", "output": "positive"},
        {"input": "This update broke everything and support is not responding.", "output": "negative"},
        {"input": "The package arrived on Tuesday as expected.", "output": "neutral"},
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )

    # 2. Build full prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify sentiment. Return exactly ONE word: positive, neutral, or negative."),
        few_shot_prompt,
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()

    # 3. Test on new unseen inputs
    test_inputs = [
        "The interface is so confusing and I lost my work.",
        "The new speed improvement is incredible!",
        "The application started in 3 seconds.",
    ]

    print("Running few-shot sentiment classification:\n")
    try:
        for text in test_inputs:
            result = chain.invoke({"input": text})
            print(f"Input: {text}")
            print(f"Sentiment: {result.strip()}\n")
    except Exception as e:
        print(f"Error running model: {e}")
        print("Make sure Ollama is running.")


if __name__ == "__main__":
    main()
