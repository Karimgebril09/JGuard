from langgraph.graph import StateGraph , START, END
from typing import Literal
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage  
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

from .llm import llm


load_dotenv()
tavily_search = TavilySearch(max_results=3, topic="general")
# Gemini

researcher_llm = llm.bind_tools([tavily_search])
# researcher_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     api_key=os.getenv("GEMINI_API_KEY")
# ).bind_tools([tavily_search])




class ResearchAgentState(MessagesState):
    research_topic_from_user: str
    remaining_available_steps: int


def research_agent(state: ResearchAgentState):
    messages = [
        SystemMessage(
            content=(
                "You are a research agent. Use Tavily Search if needed. "
                "If you have enough information, provide a structured final answer. "
                "If not, call the Tavily tool. "
                f"Current info: {state.get('messages', 'no information available yet')} "
            )
        ),
        HumanMessage(
            content="Research topic: " + state["research_topic_from_user"]
        ),
    ]

    response = researcher_llm.invoke(messages)

    return {
        "messages": [response],
        "remaining_available_steps": state["remaining_available_steps"] - 1,
    }

def router(state: ResearchAgentState) -> Literal["end", "research_agent"]:
    last_msg = state["messages"][-1]

    has_tool_calls = getattr(last_msg, "tool_calls", None)

    if state["remaining_available_steps"] <= 0 or not has_tool_calls:
        return "end"

    return "research_agent"

def build_research_agent():
    graph = StateGraph(ResearchAgentState)

    graph.add_node("research_agent", research_agent)
    graph.add_node("tavily_tool", ToolNode([tavily_search]))

    graph.add_edge(START, "research_agent")

    graph.add_conditional_edges(
        "research_agent",
        router,
        {
            "end": END,
            "research_agent": "tavily_tool",
        },
    )

    graph.add_edge("tavily_tool", "research_agent")

    app = graph.compile()
    return app

# if __name__ == "__main__":
    # app = build_research_agent()
    # resp=app.invoke(
    #     research_topic_from_user="Recent advancements in renewable energy technologies.",
    #     remaining_available_steps=3
    # )
    # print(resp["messages"][-1].content)