"""
08_single_agent.py
Topic: Single Autonomous Agent equipped with tools.
Run: python topics/08_single_agent.py
"""

import ast
import operator as op
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

# --- Safe Calculator Tool ---
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
    """Calculate arithmetic expressions such as 18 * 7 or (25 * 4) / 2."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return str(_safe_eval(tree))
    except Exception as e:
        return f"Calculation error: {e}"


# --- Knowledge Search Tool ---
POLICY_DATA = {
    "leave": "Employees get 25 days of paid annual leave each year.",
    "budget": "The annual education budget per employee is $1,500.",
}


@tool
def search_policy(topic: str) -> str:
    """Search company policy documents for leave or budget."""
    for key, val in POLICY_DATA.items():
        if key in topic.lower():
            return val
    return "Policy topic not found."


def main():
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=0.1)

    agent = create_agent(
        model=llm,
        tools=[calculator, search_policy],
        system_prompt=(
            "You are a helpful assistant. Use 'calculator' for any math. "
            "Use 'search_policy' for policy questions. Synthesize the final answer clearly."
        ),
        name="study_agent",
    )

    query = "Find how many annual leave days employees get, and calculate 19 * 8."
    print(f"User Query: {query}\n")

    try:
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
        messages = response.get("messages", [])
        final_answer = messages[-1].content if messages else str(response)
        print(f"Final Agent Answer:\n{final_answer}")
    except Exception as e:
        print(f"Error running agent: {e}")
        print("Make sure Ollama is running.")


if __name__ == "__main__":
    main()
