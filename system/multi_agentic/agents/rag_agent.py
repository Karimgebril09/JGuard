from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from defenders.tools.rag.src.rag_pipeline import rag
from defenders.tools.rag.src.jailbreak_scanner import InjectionScanner

from .llm import llm


class RagAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_rag_agent(rag_protection: bool = True):

    @tool
    def search_knowledge_base(query: str) -> str:
        """
        Searches the company knowledge base using semantic similarity.
        Use this whenever the user asks about company policies, products, or procedures.
        Returns the most relevant passages found.
        """
        chunks_tuples = rag.retrieve(query)
        print("[retrieved]")
        chunks = [c[0] for c in chunks_tuples]
        for c in chunks:
            print(f" - {c.text[:10]}... (source: {c.source})")

        if rag_protection:
            scanner = InjectionScanner()
            print("[scanner]")
            safe_chunks = scanner.check_jailbreak(chunks, query)
            print("[safe chunks]")
            for c in safe_chunks:
                print(f" - {c.text[:10]}... (source: {c.source})")
        else:
            safe_chunks = chunks

        return "\n\n".join(c.text for c in safe_chunks)

    llm_with_tools = llm.bind_tools([search_knowledge_base])

    def rag_agent_function(state: RagAgentState):
        system_prompt = SystemMessage(content=(
            "You are a helpful company assistant with access to the company knowledge base. "
            "When needed, use the search_knowledge_base tool to find relevant information. "
            "After receiving tool results, answer the user directly and do not call the tool again. "
            "Base your answer strictly on what the tool returns. "
            "If the tool finds nothing relevant, say so clearly — never make up information. "
            "Be concise and cite which source the information came from."
        ))

        for m in state["messages"]:
            content = str(m.content) if m.content is not None else "<None>"
            print(f" - {content[:10]}... ({type(m).__name__})")

        messages = state.get("messages", [])
        has_tool_result = any(isinstance(m, ToolMessage) for m in messages)

        if has_tool_result:
            response = llm.invoke([SystemMessage(content="answer only based on the provided tool result")] + messages)
        else:
            response = llm_with_tools.invoke([system_prompt] + messages)

        return {"messages": [response]}

    workflow = StateGraph(RagAgentState)
    workflow.add_node("agent", rag_agent_function)
    workflow.add_node("tools", ToolNode([search_knowledge_base]))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


rag_agent = build_rag_agent()
