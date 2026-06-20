
from .coding_agent import build_coding_agent
from .document_parser import build_document_processor
from ..checkpointer.evaluator import JailbreakEvaluator
from ..checkpointer.safety_checkpointer import SafetyCheckpointer
from .research_agent import build_research_agent
from .rag_agent import build_rag_agent
from .email_agent import build_email_agent, run_email_agent, display_result
from .llm import llm
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from typing import Any, Literal, cast
from langgraph.graph.message import MessagesState
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph,START,END
import os

orchestrator_system_prompt = (
        "You are an orchestration agent. Your job is to choose exactly ONE next action "
        "for the user's request from this set: 'code', 'research', 'document', 'email', 'end'.\n\n"

        "Use 'code' for programming-related work: writing, fixing, explaining, or testing code when the main goal "
        "is to work on the code itself. Examples: 'write a Python function', 'fix this TypeError', 'generate unit "
        "tests', 'explain this code and give an example'. If the main goal is to produce documentation or a PDF "
        "about the code (for example, API documentation), prefer 'document' instead of 'code'.\n\n"

        "Use 'research' when the user asks for up-to-date information or broad knowledge that "
        "typically requires searching the web or external sources. Examples: 'latest advancements', "
        "'compare frameworks in 2026', 'find recent research and summarize it', 'give trustworthy sources'.\n\n"

        "Use 'document' when the task is about reading from or writing to documents (PDFs, files, "
        "documentation). Whenever the user asks to create or update documentation, manuals, or PDF files (for example, "
        "'create API documentation and save as PDF', 'generate API documentation for my Flask app and save it as a PDF'), "
        "treat this as 'document' even if code is involved. Other examples: 'read this PDF and summarize it', "
        "'write pipeline documentation to the docs folder', 'read user_manual.pdf and list features'.\n\n"

        "Use 'email' only when the task clearly involves email or an inbox: reading emails, summarizing emails, "
        "or drafting/sending emails to recipients. The user should mention words like 'email', 'mail', 'inbox', "
        "'Gmail', or similar. Examples: 'check my inbox', 'summarize unread emails', 'draft a polite email', "
        "'send a follow-up email'. Never choose 'email' if the user is just greeting you, asking you to say hello or "
        "goodbye, asking for a recap of what you did, or asking for a short motivational or closing message, and the "
        "request does not explicitly mention email or an inbox.\n\n"

        "Use 'end' when the user explicitly indicates the conversation should end, or when you can fully answer "
        "the request yourself without delegating to any other agent. This includes simple greetings, friendly goodbyes, "
        "brief recaps of what you did together, and short motivational or closing messages, as well as pure explanation "
        "questions that do not require tools (for example, short descriptions of what this orchestration or multi-agent "
        "system does in one paragraph). In these cases, answer the user directly in 'final_response' and choose 'end' "
        "instead of any tool-using action.\n\n"

        "In particular, simple greeting requests (for example, anything equivalent to 'Just say hello to me and do nothing "
        "else'), short friendly goodbyes (for example, 'Just tell me a friendly goodbye message and then stop responding'), "
        "brief recaps of the session (for example, 'Give a brief recap of what we did today and then wrap up'), and short "
        "motivational closing messages (for example, 'Wrap up with a short motivational message and then stop') should all "
        "be answered directly in 'final_response' with Next_action set to 'end'. Do not route these cases to 'email', "
        "'document', 'code', or 'research'.\n\n"

        "You should keep using agents (code, research, document, email) only while they are truly needed to fulfill "
        "the user's request. When you no longer need any more actions: (1) generate a final response to the user using "
        "all collected information, and (2) choose the 'end' action."
    )

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

def run_coder_agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    print("Running coder agent...")
    mas_guard = config.get("configurable", {}).get("mas_guard")
    if mas_guard is not None and mas_guard.check_multi_turn("coder", state["user_message"]):
        print("BLOCKED: Multi-turn attack detected for coder agent.")
        blocked = "I cannot fulfill that request."
        mas_guard.update_last_response("coder", blocked)
        return {"messages": [AIMessage(content=blocked)]}

    response = coder_agent.invoke({"problem_description": state["user_message"]})
    content = response["code"]
    if mas_guard is not None:
        mas_guard.update_last_response("coder", content)
    return {"messages": [AIMessage(content=content)]}


def run_research_agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    print("Running research agent...")
    mas_guard = config.get("configurable", {}).get("mas_guard")
    if mas_guard is not None and mas_guard.check_multi_turn("research", state["user_message"]):
        print("BLOCKED: Multi-turn attack detected for research agent.")
        blocked = "I cannot fulfill that request."
        mas_guard.update_last_response("research", blocked)
        return {"messages": [AIMessage(content=blocked)]}

    response = research_agent.invoke(cast(Any, {"research_topic_from_user": state["user_message"], "remaining_available_steps": 3}))
    last_msg = response["messages"][-1]
    content = getattr(last_msg, "content", str(last_msg))
    if mas_guard is not None:
        mas_guard.update_last_response("research", content)
    return {"messages": [last_msg]}


def run_document_agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    print("Running document agent...")
    mas_guard = config.get("configurable", {}).get("mas_guard")
    if mas_guard is not None and mas_guard.check_multi_turn("document", state["user_message"]):
        print("BLOCKED: Multi-turn attack detected for document agent.")
        blocked = "I cannot fulfill that request."
        mas_guard.update_last_response("document", blocked)
        return {"messages": [AIMessage(content=blocked)]}

    response = document_agent.invoke({"request": state["user_message"]})
    content = response["messages"][-1].content
    if mas_guard is not None:
        mas_guard.update_last_response("document", content)
    return {"messages": [AIMessage(content=content)]}


def run_rag_agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    print("Running RAG agent...")
    mas_guard = config.get("configurable", {}).get("mas_guard")
    if mas_guard is not None and mas_guard.check_multi_turn("rag", state["user_message"]):
        print("BLOCKED: Multi-turn attack detected for RAG agent.")
        blocked = "I cannot fulfill that request."
        mas_guard.update_last_response("rag", blocked)
        return {"messages": [AIMessage(content=blocked)]}

    response = rag_agent.invoke({"messages": [HumanMessage(content=state["user_message"])],
                                 'enable_scanner': False})  ###  7ot mkanha al var men al backend
    content = response["messages"][-1].content
    if mas_guard is not None:
        mas_guard.update_last_response("rag", content)
    return {"messages": [AIMessage(content=content)]}


def run_email_agent_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    print("Running email agent...")
    mas_guard = config.get("configurable", {}).get("mas_guard")
    if mas_guard is not None and mas_guard.check_multi_turn("email", state["user_message"]):
        print("BLOCKED: Multi-turn attack detected for email agent.")
        blocked = "I cannot fulfill that request."
        mas_guard.update_last_response("email", blocked)
        return {"messages": [AIMessage(content=blocked)]}

    result = run_email_agent(email_agent, state["user_message"])
    content = f"Action: {result['action']}\nResponse: {result['response']}"
    if result.get("result"):
        content += f"\nResult: {result['result']}"
    if mas_guard is not None:
        mas_guard.update_last_response("email", content)
    return {"messages": [AIMessage(content=content)]}

def orch_agent_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    print("Running orchestration agent...")

    mas_guard = config.get("configurable", {}).get("mas_guard")

    # Multi-turn check: detect attack patterns across conversation turns
    if mas_guard is not None and mas_guard.check_multi_turn("orchestrator", state["user_message"]):
        print("BLOCKED: Multi-turn attack detected for orchestrator.")
        blocked = "I cannot fulfill that request."
        mas_guard.update_last_response("orchestrator", blocked)
        return {"next_action": "end", "response": blocked, "messages": [AIMessage(content=blocked)]}

    # Check the last AI message coming INTO the orchestrator (previous agent output)
    if mas_guard is not None:
        msgs = state.get("messages", [])
        if msgs:
            last_msg = msgs[-1]
            if getattr(last_msg, "type", "") == "ai":
                content = getattr(last_msg, "content", "")
                if isinstance(content, str) and content.strip():
                    checked_content, blocked_flag, defenses = mas_guard.check_message(content)
                    if blocked_flag:
                        print(f"BLOCKED: Incoming message to orchestrator blocked by {defenses}")
                        return {
                            "next_action": "end",
                            "response": checked_content,
                            "messages": [AIMessage(content=checked_content)],
                        }

    messages = [
        SystemMessage(content=orchestrator_system_prompt),
        HumanMessage(content=f"user message: {state['user_message']} \n current collected info: {state.get('messages', [])}"),
    ]
    response = cast(orch_messages, orch_agent.invoke(messages))

    final_response = response.final_response

    # Check the orchestrator's outgoing final response before it reaches the next agent or user
    if mas_guard is not None and final_response.strip():
        checked_response, blocked_flag, defenses = mas_guard.check_message(final_response)
        if blocked_flag:
            print(f"BLOCKED: Orchestrator outgoing response blocked by {defenses}")
            mas_guard.update_last_response("orchestrator", checked_response)
            return {
                "next_action": "end",
                "response": checked_response,
                "messages": [AIMessage(content=checked_response)],
            }
        final_response = checked_response

    # Update last response only when the orchestrator produces a meaningful reply to the user
    if mas_guard is not None and response.Next_action == "end" and final_response.strip():
        mas_guard.update_last_response("orchestrator", final_response)

    return {
        "next_action": response.Next_action,
        "response": final_response,
        "messages": [AIMessage(content=f"{final_response} next action: {response.Next_action}")],
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

