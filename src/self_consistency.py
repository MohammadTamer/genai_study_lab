from collections import Counter
import re
from langchain_core.output_parsers import StrOutputParser
from .model import llm
from .prompts import self_consistency_prompt

def _extract_final(text):
    match = re.search(r"FINAL:\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else text.strip().splitlines()[0].strip()

def self_consistent_answer(problem, samples=3):
    candidates = []
    raw = []
    chain = self_consistency_prompt | llm | StrOutputParser()
    for _ in range(samples):
        response = chain.invoke({"problem": problem})
        raw.append(response)
        candidates.append(_extract_final(response))
    winner, count = Counter(candidates).most_common(1)[0]
    return {"answer": winner, "votes": count, "samples": raw, "candidates": candidates}
