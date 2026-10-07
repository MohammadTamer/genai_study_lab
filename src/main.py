import sys
if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr:
    sys.stderr.reconfigure(encoding="utf-8")

from .chains import basic_chain, few_shot_chain, build_rag_chain, build_parallel_demo_chain
from .rag import get_retriever
from .agents import build_single_agent, build_multi_agent_system
from .self_consistency import self_consistent_answer

def run_demo():
    print("\n=== 1) Zero-shot/basic prompting ===")
    print(basic_chain.invoke({"question": "Explain RAG in simple terms."}))

    print("\n=== 2) Few-shot/in-context learning ===")
    print(few_shot_chain.invoke({"input": "I really like the new LangChain release."}))

    print("\n=== 3) RAG ===")
    retriever, _, chunks = get_retriever()
    print(f"Indexed {len(chunks)} chunks.")
    rag_chain = build_rag_chain(retriever)
    print(rag_chain.invoke("How many annual leave days do employees receive?"))

    print("\n=== 4) RunnableParallel ===")
    print(build_parallel_demo_chain().invoke({"question": "What is a foundation model?"}))

    print("\n=== 5) Agent + tools ===")
    agent = build_single_agent(retriever)
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "How much is the training budget, and what is 18 * 7?"
        }]
    })
    print(result["messages"][-1].content)

    print("\n=== 6) Multi-agent ===")
    manager = build_multi_agent_system(retriever)
    result = manager.invoke({
        "messages": [{
            "role": "user",
            "content": "Find the annual leave policy and calculate 25 * 4."
        }]
    })
    print(result["messages"][-1].content)

    print("\n=== 7) Self-consistency ===")
    sc = self_consistent_answer(
        "When I was 6, my sister was half my age. Now I am 70. How old is my sister?"
    )
    print("Final answer:", sc["answer"])
    print("Votes:", sc["votes"], "/", len(sc["candidates"]))

if __name__ == "__main__":
    run_demo()
