
from .coding_agent import build_coding_agent
from .document_parser import build_document_processor
from ..checkpointer.evaluator import JailbreakEvaluator
from ..checkpointer.safety_checkpointer import SafetyCheckpointer
from .research_agent import build_research_agent
from .rag_agent import build_rag_agent
from .email_agent import build_email_agent, run_email_agent, display_result
from .llm import llm
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from typing import Literal
from langgraph.graph.message import MessagesState
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph,START,END
import os

coder_agent = build_coding_agent()
research_agent = build_research_agent()
document_agent = build_document_processor()
rag_agent = build_rag_agent()
email_agent = build_email_agent(use_gmail_service=True)


class orch_messages(BaseModel):
    Next_action: Literal["code", "research", "document", "email", "end", "rag"] = Field(..., description="The next action to take")
    final_response: str = Field(..., description="The final response to the user if the next action is 'end' keep it empty otherwise")


def create_llms(local:bool=True):
    orch_agent = llm.with_structured_output(orch_messages)
   
    return orch_agent

orch_agent = create_llms(local=True)
class AgentState(MessagesState):
    user_message: str
    response: str
    next_action: Literal["code", "research", "document", "email", "end","rag"] | None = None

def run_coder_agent(state: AgentState) -> None:
    print("Running coder agent...")
    response = coder_agent.invoke({"problem_description": state["user_message"]})
    return{
        "messages": [AIMessage(content=response["code"])],
    }
def run_research_agent(state: AgentState) -> None:
    print("Running research agent...")
    response = research_agent.invoke({"research_topic_from_user": state["user_message"],"remaining_available_steps":3})
    return{
        "messages": [response["messages"][-1]],
    }

def run_document_agent(state: AgentState) -> None:
    """ file path should be revised how it is handeled"""
    print("Running document agent...")
    response = document_agent.invoke({"request": state["user_message"]})
    return {
        "messages": [AIMessage(response["messages"][-1].content)],
    }
def run_rag_agent(state: AgentState) -> None:
    """Run RAG agent to get employee information from database based on user query"""
    print("Running RAG agent...")
    # Pass messages to RAG agent - convert user_message to message format if needed
    rag_input = {
        "messages": [HumanMessage(content=state["user_message"])]
    }
    response = rag_agent.invoke(rag_input)
    return {
        "messages": [AIMessage(content=response["messages"][-1].content)],
       
    }

def run_email_agent_node(state: AgentState) -> None:
    """Run the email agent for reading/sending emails"""
    print("Running email agent...")
    result = run_email_agent(email_agent, state["user_message"])
    response_content = f"Action: {result['action']}\nResponse: {result['response']}"
    if result.get('result'):
        response_content += f"\nResult: {result['result']}"
    return {
        "messages": [AIMessage(content=response_content)],
    }

def orch_agent_node(state: AgentState) -> None:
    print("Running orchestration agent...")
    messages=[SystemMessage(content="You are an orchestration agent that decides " \
    "whether to delegate tasks to " \
    "rag: helps you find information about employees agent have access to employee database " \
    ", research: helps you find information from web if you need or dont have the knowledge" \
    ", document processing: handles tasks related to processing and managing documents like creating documentation and save in pdf" \
    "or reading docs based on user input if it returned file path required end."\
    ", email: handles tasks related to reading emails from inbox or sending emails to recipients."\
    "you should keep using the agents until the user's request is fulfilled."
    
     "1-generate a final response to the user using all info you collected"\
     "2-end the conversation by choosing the 'end' action"
    ),
    HumanMessage(content=f"user message: {state['user_message']} \n current collected info: {state.get('messages',[])}" )]
    response = orch_agent.invoke(messages)
    return{
        "next_action": response.Next_action,
        "response": response.final_response,
        "messages":AIMessage(content=f"{response.final_response} next action: {response.Next_action}")
    }



def route(state:AgentState) -> str:
    if state["next_action"] == "code":
        return "code"
    elif state["next_action"] == "research":
        return "research"
    elif state["next_action"] == "document":
        return "document"
    elif state["next_action"] == "email":
        return "email"
    elif state["next_action"] == "rag":
        return "rag"
    else:
        return "end"
    

graph = StateGraph(AgentState)

graph.add_node("orch_agent", orch_agent_node)
graph.add_node("run_coder_agent", run_coder_agent)
graph.add_node("run_research_agent", run_research_agent)
graph.add_node("run_document_agent", run_document_agent)
graph.add_node("run_email_agent", run_email_agent_node)
graph.add_node("run_rag_agent", run_rag_agent)


graph.add_conditional_edges("orch_agent", route, {
    "code": "run_coder_agent",
    "research": "run_research_agent",
    "document": "run_document_agent",
    "email": "run_email_agent",
    "rag":"run_rag_agent",
    "end": END
    })

graph.add_edge(START, "orch_agent")
graph.add_edge("run_coder_agent", "orch_agent")
graph.add_edge("run_research_agent", "orch_agent")
graph.add_edge("run_document_agent", "orch_agent")
graph.add_edge("run_email_agent", "orch_agent")
graph.add_edge("run_rag_agent", "orch_agent")



evaluator = JailbreakEvaluator()
safe_checkpointer = SafetyCheckpointer(evaluator)

app=graph.compile(checkpointer=safe_checkpointer)

