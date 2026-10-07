from langchain.agents import create_agent
from langchain.tools import tool
from .model import llm
from .tools import calculator, make_search_tool

def _last_content(result):
    messages = result.get("messages", [])
    if not messages:
        return str(result)
    content = getattr(messages[-1], "content", messages[-1])
    return content if isinstance(content, str) else str(content)

def build_single_agent(retriever):
    search_tool = make_search_tool(retriever)
    return create_agent(
        model=llm,
        tools=[calculator, search_tool],
        system_prompt=(
            "You are a study assistant. Use tools when needed. "
            "For company/course questions, search the knowledge base first. "
            "For arithmetic, use the calculator. Never invent facts absent from tools."
        ),
        name="study_agent",
    )

def build_multi_agent_system(retriever):
    search_tool = make_search_tool(retriever)

    research_agent = create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt="You are the research specialist. Search the knowledge base and return grounded facts only.",
        name="research_agent",
    )

    math_agent = create_agent(
        model=llm,
        tools=[calculator],
        system_prompt="You are the math specialist. Use the calculator for arithmetic and return the final result clearly.",
        name="math_agent",
    )

    @tool
    def research_specialist(question: str) -> str:
        """Ask the research specialist about the knowledge base."""
        result = research_agent.invoke({"messages": [{"role": "user", "content": question}]})
        return _last_content(result)

    @tool
    def math_specialist(question: str) -> str:
        """Ask the math specialist to solve a numeric problem."""
        result = math_agent.invoke({"messages": [{"role": "user", "content": question}]})
        return _last_content(result)

    return create_agent(
        model=llm,
        tools=[research_specialist, math_specialist],
        system_prompt=(
            "You manage a small specialist team. "
            "Use research_specialist for knowledge-base questions and "
            "math_specialist for arithmetic. Synthesize their results."
        ),
        name="manager_agent",
    )
