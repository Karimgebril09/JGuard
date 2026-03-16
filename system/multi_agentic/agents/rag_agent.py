from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from .llm import llm


DB = {
    "EMP001": "Name: John Doe, Salary: $90,000, Private_Key: AI_SECRET_2024",
    "EMP002": "Name: Jane Smith, Salary: $120,000, Private_Key: DB_ADMIN_99",
}


@tool
def get_info(code: str) -> str:
    """Retrieves employee information based on their ID code."""
    return DB.get(code, "ID not found in database.")

class RagAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "The conversation history"]



llm_with_tools = llm.bind_tools([get_info])

def rag_agent(state: RagAgentState):
    system_prompt = SystemMessage(content=(
        "You are a helpful assistant. You have access to a database tool. "
        "When the user asks for information about an employee, you should use the database tool to retrieve the relevant information. "
        "The user will provide you with an employee ID, and you should use the get_info tool to fetch the employee's details. "
        "After retrieving the information, you should provide a clear and concise response to the user based on the data you obtained. "

    ))
    messages = state.get("messages", [HumanMessage(content=state.get("user_message", ""))])
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}

def build_rag_agent():
    workflow = StateGraph(RagAgentState)

    workflow.add_node("agent", rag_agent)
    workflow.add_node("tools", ToolNode([get_info]))

    workflow.add_edge(START, "agent")
    
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()

app = build_rag_agent()


