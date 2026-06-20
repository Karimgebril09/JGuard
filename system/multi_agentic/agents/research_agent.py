from langgraph.graph import StateGraph, START, END
from typing import Literal
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

from .llm import llm
from defenders.tools.web_search.WebSearchGuard import WebSearchGuard


load_dotenv()
assert os.getenv("TAVILY_API_KEY"), "TAVILY_API_KEY is not set!"

tavily_search = TavilySearch(max_results=3, topic="general")
researcher_llm = llm.bind_tools([tavily_search])


class ResearchAgentState(MessagesState):
    research_topic_from_user: str
    remaining_available_steps: int
    past_search_queries: list[str]


def research_agent(state: ResearchAgentState):
    topic = state["research_topic_from_user"]
    prior_messages = state.get("messages", [])

    system_prompt = (
        "You are a research agent. Your task is to research the following topic: "
        f"{topic}. "
        "Use the Tavily Search tool to find relevant, up-to-date information. "
        "Call the tool with a query that is directly related to the topic above. "
        "Once you have search results, provide a structured final answer and stop calling the tool."
    )

    if not prior_messages:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=topic)]
    else:
        messages = [SystemMessage(content=system_prompt), *prior_messages]

    response = researcher_llm.invoke(messages)

    return {
        "messages": [response],
        "remaining_available_steps": state["remaining_available_steps"] - 1,
    }


def _router_to_guard(state: ResearchAgentState) -> Literal["end", "search_guard"]:
    last_msg = state["messages"][-1]
    has_tool_calls = getattr(last_msg, "tool_calls", None)
    if state["remaining_available_steps"] <= 0 or not has_tool_calls:
        return "end"
    return "search_guard"


def _router_to_tool(state: ResearchAgentState) -> Literal["end", "tavily_tool"]:
    last_msg = state["messages"][-1]
    has_tool_calls = getattr(last_msg, "tool_calls", None)
    if state["remaining_available_steps"] <= 0 or not has_tool_calls:
        return "end"
    return "tavily_tool"


def _search_guard_router(state: ResearchAgentState) -> Literal["tavily_tool", "research_agent"]:
    # search_guard injects a ToolMessage when it blocks; pass-through leaves last msg as AIMessage
    if isinstance(state["messages"][-1], ToolMessage):
        return "research_agent"
    return "tavily_tool"


def build_research_agent(web_search_protection: bool = True):
    graph = StateGraph(ResearchAgentState)

    graph.add_node("research_agent", research_agent)
    graph.add_node("tavily_tool", ToolNode([tavily_search]))

    graph.add_edge(START, "research_agent")

    if web_search_protection:
        guard = WebSearchGuard()

        def search_guard(state: ResearchAgentState) -> dict:
            last_msg = state["messages"][-1]
            tool_calls = getattr(last_msg, "tool_calls", [])
            if not tool_calls:
                return {}
            query: str = tool_calls[0].get("args", {}).get("query", "")
            tool_call_id: str = tool_calls[0].get("id", "")
            original_request: str = state["research_topic_from_user"]
            past_queries: list[str] = state.get("past_search_queries", [])  # type: ignore[assignment]

            _, state_update = guard.check_query(
                query, original_request, past_queries, tool_call_id
            )
            return state_update

        graph.add_node("search_guard", search_guard)

        graph.add_conditional_edges(
            "research_agent",
            _router_to_guard,
            {"end": END, "search_guard": "search_guard"},
        )
        graph.add_conditional_edges(
            "search_guard",
            _search_guard_router,
            {"tavily_tool": "tavily_tool", "research_agent": "research_agent"},
        )
    else:
        graph.add_conditional_edges(
            "research_agent",
            _router_to_tool,
            {"end": END, "tavily_tool": "tavily_tool"},
        )

    graph.add_edge("tavily_tool", "research_agent")

    return graph.compile()
