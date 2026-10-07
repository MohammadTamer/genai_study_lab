from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from .model import llm
from .prompts import basic_prompt, rag_prompt, classification_prompt

basic_chain = basic_prompt | llm | StrOutputParser()
few_shot_chain = classification_prompt | llm | StrOutputParser()

def normalize_input(data):
    if isinstance(data, str):
        return {"question": data}
    return data

normalizer = RunnableLambda(normalize_input)
basic_chain_with_lambda = normalizer | basic_chain

def build_rag_chain(retriever):
    retrieve_and_pass = RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough(),
    })
    return retrieve_and_pass | rag_prompt | llm | StrOutputParser()

def build_parallel_demo_chain():
    translation_prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate the user's text into Arabic. Return only the translation."),
        ("human", "{question}"),
    ])
    return RunnableParallel({
        "summary": basic_prompt | llm | StrOutputParser(),
        "translation": translation_prompt | llm | StrOutputParser(),
    })
