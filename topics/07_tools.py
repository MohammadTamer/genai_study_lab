"""
07_tools.py
Topic: Custom Tools with the @tool decorator and AST-safe calculator.
Run: python topics/07_tools.py
"""

import ast
import operator as op
from langchain.tools import tool

# Allowed operators for safe mathematical evaluation
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
    """Calculate basic arithmetic expressions like '12 * (7 + 3)' or '100 / 4'."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return str(_safe_eval(tree))
    except Exception as e:
        return f"Calculation error: {e}"


LOCAL_DB = {
    "rag": "Retrieval-Augmented Generation combines retrieval with an LLM.",
    "agent": "An AI Agent uses an LLM to choose actions and execute tools.",
    "langchain": "LangChain is a framework for developing applications with LLMs.",
}


@tool
def search_tech_term(term: str) -> str:
    """Search internal dictionary for a technical term."""
    return LOCAL_DB.get(term.strip().lower(), f"Term '{term}' not found in knowledge base.")


def main():
    print("Testing calculator tool:")
    for expr in ["15 * (4 + 6)", "2 ** 8", "1000 / 8"]:
        result = calculator.invoke({"expression": expr})
        print(f"  {expr} = {result}")

    print("\nTesting search_tech_term tool:")
    for term in ["rag", "agent", "docker"]:
        result = search_tech_term.invoke({"term": term})
        print(f"  [{term}]: {result}")


if __name__ == "__main__":
    main()
