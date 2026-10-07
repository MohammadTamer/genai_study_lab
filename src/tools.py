import ast
import operator as op
from langchain.tools import tool

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
    """Calculate a basic arithmetic expression such as 12*(7+3)."""
    tree = ast.parse(expression, mode="eval")
    return str(_safe_eval(tree))

def make_search_tool(retriever):
    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the project knowledge base and return relevant passages."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents were found."
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] Source: {source}\n{doc.page_content}")
        return "\n\n".join(parts)
    return search_knowledge_base
