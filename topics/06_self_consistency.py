"""
06_self_consistency.py
Topic: Self-Consistency Prompting (Sample multiple reasoning paths and majority vote).
Run: python topics/06_self_consistency.py
"""

import os
import re
from collections import Counter
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def extract_answer(text: str) -> str:
    """Extract answer from 'FINAL: <answer>' or return the first line."""
    match = re.search(r"FINAL:\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else text.strip().splitlines()[0].strip()


def main():
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=0.7)

    prompt = PromptTemplate.from_template(
        """Solve this math/logic question step by step.
On the FIRST line of your response, write your final answer in this exact format:
FINAL: <answer>

Then write your brief reasoning.
Problem: {problem}
"""
    )

    chain = prompt | llm | StrOutputParser()

    problem = "When I was 6 years old, my sister was half my age. Now I am 70. How old is my sister?"
    print(f"Problem:\n{problem}\n")

    candidates = []
    num_samples = 3

    print(f"Generating {num_samples} reasoning samples...")
    try:
        for i in range(1, num_samples + 1):
            response = chain.invoke({"problem": problem})
            ans = extract_answer(response)
            candidates.append(ans)
            print(f"Sample {i} Answer: {ans}")

        winner, count = Counter(candidates).most_common(1)[0]
        print(f"\nFinal Consensus Answer: {winner} ({count}/{num_samples} votes)")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running.")


if __name__ == "__main__":
    main()
