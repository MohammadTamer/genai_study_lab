"""
09_multi_agent.py
Topic: Multi-Agent System (Manager coordinating specialized sub-agents).
Run: python topics/09_multi_agent.py
"""

import ast
import operator as op
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

# --- Tools for Specialists ---
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    raise ValueError("Only basic arithmetic is allowed.")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression (e.g. '1500 * 8')."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return str(_safe_eval(tree))
    except Exception as e:
        return f"Calculation error: {e}"


KB = {
    "budget": "The personal learning budget is $1,500 per engineer each year.",
    "leave": "Employees get 25 annual leave days.",
}


@tool
def search_kb(topic: str) -> str:
    """Search internal company knowledge base."""
    for key, text in KB.items():
        if key in topic.lower():
            return text
    return f"No document found for '{topic}'."


def _get_text(res):
    messages = res.get("messages", [])
    if not messages:
        return str(res)
    last = messages[-1]
    return getattr(last, "content", str(last))


def main():
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=0.1)

    # 1. Specialized Agents
    research_agent = create_agent(
        model=llm,
        tools=[search_kb],
        system_prompt="You are a research specialist. Search facts and return concise answers.",
        name="research_agent",
    )

    math_agent = create_agent(
        model=llm,
        tools=[calculator],
        system_prompt="You are a math specialist. Use the calculator and return the exact calculation.",
        name="math_agent",
    )

    # 2. Expose Specialists as Tools for Manager
    @tool
    def ask_research_specialist(question: str) -> str:
        """Ask research specialist for company policies or facts."""
        res = research_agent.invoke({"messages": [{"role": "user", "content": question}]})
        return _get_text(res)

    @tool
    def ask_math_specialist(problem: str) -> str:
        """Ask math specialist to calculate a numeric problem."""
        res = math_agent.invoke({"messages": [{"role": "user", "content": problem}]})
        return _get_text(res)

    # 3. Manager Agent
    manager = create_agent(
        model=llm,
        tools=[ask_research_specialist, ask_math_specialist],
        system_prompt=(
            "You are the team manager. Delegate policy questions to 'ask_research_specialist' "
            "and arithmetic to 'ask_math_specialist'. Combine results clearly."
        ),
        name="manager_agent",
    )

    query = "What is the annual learning budget, and what is the total budget for 8 engineers?"
    print(f"Query: {query}\n")

    try:
        response = manager.invoke({"messages": [{"role": "user", "content": query}]})
        print("Manager Response:\n" + _get_text(response))
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running.")


if __name__ == "__main__":
    main()
