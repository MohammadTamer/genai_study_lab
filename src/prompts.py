from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, FewShotChatMessagePromptTemplate

basic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful GenAI tutor. Explain concepts clearly and use small examples."),
    ("human", "{question}"),
])

rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a grounded assistant. Answer ONLY from the supplied context. "
        "If the context does not contain the answer, say you do not have enough information. "
        "Do not invent policies or facts. Mention source names when possible.",
    ),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
])

few_shot_examples = [
    {"input": "I love this product.", "output": "positive"},
    {"input": "This is disappointing.", "output": "negative"},
    {"input": "It is okay.", "output": "neutral"},
]

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=few_shot_examples,
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ]),
)

classification_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify sentiment. Return exactly one word: positive, neutral, or negative."),
    few_shot_prompt,
    ("human", "{input}"),
])

self_consistency_prompt = PromptTemplate.from_template(
    """Solve the problem independently.
Return the final answer on the FIRST line using this exact format:
FINAL: <answer>

Then give a brief explanation.
Problem:
{problem}
"""
)
