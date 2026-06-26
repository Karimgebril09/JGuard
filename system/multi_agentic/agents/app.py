
import datetime
import os
import json
from evaluation.evaluator import Evaluator
from evaluation.multi_agent_eval.multi_agnet_evaluator import MultiAgentEvaluator


from .coding_agent import build_coding_agent
from .document_parser import build_document_processor
from .research_agent import build_research_agent
from .rag_agent import build_rag_agent
from .email_agent import build_email_agent, run_email_agent, display_result
from .llm import llm
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from typing import Any, Literal, cast
from langgraph.graph.message import MessagesState
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END


orchestrator_system_prompt = (
        "You are an orchestration agent. Your job is to choose exactly ONE next action "
        "for the user's request from this set: 'code', 'research', 'rag', 'document', 'email', 'end'.\n\n"

        "Use 'rag' when the user asks a question that should be answered from the local "
        "knowledge base or previously ingested documents — this includes any question "
        "about employee onboarding topics such as company policies, HR procedures, leave "
        "and benefits, IT account setup, security procedures, or employee records/data "
        "contained in the indexed onboarding documentation."
        "This is different from 'research' (which searches the live web for external, "
        "real-time, or general information not contained in our documents) and from "
        "'document' (which reads a specific file the user uploads or names directly in "
        "the current turn, rather than querying the pre-ingested index).\n\n"
        "Do NOT use 'rag' for questions about current events, things outside the "
        "company's internal documents, or a file the user just uploaded in this "
        "conversation — route those to 'research' or 'document' instead."
    
        "Use 'code' for programming-related work: writing, fixing, explaining, or testing code when the main goal "
        "is to work on the code itself. Examples: 'write a Python function', 'fix this TypeError', 'generate unit "
        "tests', 'explain this code and give an example'. If the main goal is to produce documentation or a PDF "
        "about the code (for example, API documentation), prefer 'document' instead of 'code'.\n\n"

        "Use 'research' when the user asks for up-to-date information or broad knowledge that "
        "typically requires searching the web or external sources. Examples: 'latest advancements', "
        "'compare frameworks in 2026', 'find recent research and summarize it', 'give trustworthy sources'. "
        "Also use 'research' whenever the user explicitly says 'use the research agent' regardless of the topic.\n\n"


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
        "'document', 'code', 'research', or 'rag'.\n\n"

        "You should keep using agents (code, research, rag, document, email) only while they are truly needed to fulfill "
        "the user's request. When you no longer need any more actions: (1) generate a final response to the user using "
        "all collected information, and (2) choose the 'end' action.\n\n"

        "IMPORTANT: If an agent has already returned a result (visible in the sub-agent outputs), do NOT call that same "
        "agent again unless the result was clearly insufficient. One successful result from an agent is normally enough — "
        "summarize it and choose 'end'."
    )

evaluator = MultiAgentEvaluator( Evaluator(
            type_judge="ollama",
            judge="qwen2.5:3b-instruct",
            base_url_judge=None,
            api_key_judge=None,
        ))

class orch_messages(BaseModel):
    Next_action: Literal["code", "research", "document", "email", "end", "rag"] = Field(..., description="The next action to take")
    final_response: str = Field(..., description="The final response to the user if the next action is 'end' keep it empty otherwise")


class AgentState(MessagesState):
    user_message: str
    response: str
    next_action: Literal["code", "research", "document", "email", "end", "rag"] | None


def create_llms(local: bool = True):
    return llm.with_structured_output(orch_messages)


def route(state: AgentState) -> str:
    returned_action = ""
    if state["next_action"] == "code":
        returned_action = "code"
    elif state["next_action"] == "research":
        returned_action = "research"
    elif state["next_action"] == "document":
        returned_action = "document"
    elif state["next_action"] == "email":
        returned_action = "email"
    elif state["next_action"] == "rag":
        returned_action = "rag"
    else:
        returned_action = "end"
    print(f"Routing to: {returned_action}") 
    return returned_action



def build_mas_app(
    web_search_protection: bool = True,
    code_execution_protection: bool = True,
    code_deep_check: bool = True,
    rag_protection: bool = True,
    email_protection: bool = True,
    document_protection: bool = True,
) -> Any:
    coder_agent = build_coding_agent(code_execution_protection=code_execution_protection, code_deep_check=code_deep_check)
    research_agent = build_research_agent(web_search_protection=web_search_protection)
    document_agent = build_document_processor(document_protection=document_protection)
    rag_agent = build_rag_agent(rag_protection=rag_protection)
    email_agent = build_email_agent(use_gmail_service=True, email_protection=email_protection)

    orch_agent = create_llms(local=True)

    def run_coder_agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        print("Running coder agent...")
        mas_guard = config.get("configurable", {}).get("mas_guard")
        if mas_guard is not None and mas_guard.check_multi_turn("coder", state["user_message"]):
            print("BLOCKED: Multi-turn attack detected for coder agent.")
            blocked = "I cannot fulfill that request."
            mas_guard.update_last_response("coder", blocked)
            return {"messages": [AIMessage(content=blocked)]}

        response = coder_agent.invoke(cast(Any, {"problem_description": state["user_message"]}))
        content = response["code"]
        if mas_guard is not None:
            mas_guard.update_last_response("coder", content)
        return {"messages": [AIMessage(content=content, name="coder_agent")]}

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
        return {"messages": [AIMessage(content=content, name="research_agent")]}

    def run_document_agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        print("Running document agent...")
        mas_guard = config.get("configurable", {}).get("mas_guard")
        if mas_guard is not None and mas_guard.check_multi_turn("document", state["user_message"]):
            print("BLOCKED: Multi-turn attack detected for document agent.")
            blocked = "I cannot fulfill that request."
            mas_guard.update_last_response("document", blocked)
            return {"messages": [AIMessage(content=blocked)]}

        response = document_agent.invoke(cast(Any, {"request": state["user_message"]}))
        content = response["messages"][-1].content
        if mas_guard is not None:
            mas_guard.update_last_response("document", content)
        return {"messages": [AIMessage(content=content, name="document_agent")]}

    def run_rag_agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        print("RAG ENTER")
        print("messages before:", len(state["messages"]))
        print("Running RAG agent...")
        mas_guard = config.get("configurable", {}).get("mas_guard")
        if mas_guard is not None and mas_guard.check_multi_turn("rag", state["user_message"]):
            print("BLOCKED: Multi-turn attack detected for RAG agent.")
            blocked = "I cannot fulfill that request."
            mas_guard.update_last_response("rag", blocked)
            return {"messages": [AIMessage(content=blocked)]}

        response = rag_agent.invoke({"messages": [HumanMessage(content=state["user_message"])]})
        content = response["messages"][-1].content
        if mas_guard is not None:
            mas_guard.update_last_response("rag", content)
        return {"messages": [AIMessage(content=content, name="rag_agent")]}

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
        return {"messages": [AIMessage(content=content, name="email_agent")]}

        
    def orch_agent_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        print("Running orchestration agent...")

        mas_guard = config.get("configurable", {}).get("mas_guard")

        if mas_guard is not None and mas_guard.check_multi_turn("orchestrator", state["user_message"]):
            print("BLOCKED: Multi-turn attack detected for orchestrator.")
            blocked = "I cannot fulfill that request."
            mas_guard.update_last_response("orchestrator", blocked)
            return {"next_action": "end", "response": blocked, "messages": [AIMessage(content=blocked)]}

        if mas_guard is not None:
            msgs = state.get("messages", [])
            if msgs:
                last_msg = msgs[-1]
                if getattr(last_msg, "type", "") == "ai":
                    content = getattr(last_msg, "content", "")
                    if isinstance(content, str) and content.strip():
                        evaluator.evaluate_and_log(
                            node_name=getattr(last_msg, "name", "unknown_agent"),
                            user_message=state["user_message"],
                            ai_content=content
                        )
                        checked_content, blocked_flag, defenses = mas_guard.check_message(content)
                        if blocked_flag:
                            print(f"BLOCKED: Incoming message to orchestrator blocked by {defenses}")
                            return {
                                "next_action": "end",
                                "response": checked_content,
                                "messages": [AIMessage(content=checked_content)],
                            }

        collected_msgs = state.get("messages", [])
        if collected_msgs:
            parts = []
            for m in collected_msgs:
                content = getattr(m, "content", "")
                if isinstance(content, str) and content.strip():
                    label = getattr(m, "name", None) or "agent"
                    parts.append(f"[{label}]: {content[:1000]}")
            formatted_info = "\n---\n".join(parts) if parts else "(none yet)"
        else:
            formatted_info = "(none yet)"

        messages = [
            SystemMessage(content=orchestrator_system_prompt),
            HumanMessage(content=f"user message: {state['user_message']}\nSub-agent outputs collected so far:\n{formatted_info}"),
        ]
        response = cast(orch_messages, orch_agent.invoke(messages))

        final_response = response.final_response

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

        if mas_guard is not None and response.Next_action == "end" and final_response.strip():
            mas_guard.update_last_response("orchestrator", final_response)

        if response.Next_action == "end":
            evaluator.evaluate_and_log(
                node_name="orch_agent",
                user_message=state["user_message"],
                ai_content=final_response,
             
            )
            return {
                "next_action": "end",
                "response": final_response,
                "messages": [AIMessage(content=final_response)] if final_response.strip() else [],
            }
            
        print("=== ORCH ===")
        print("messages count:", len(state["messages"]))
        print("next_action:", response.Next_action)
        print("final_response:", response.final_response[:100])
        
        
        return {
            "next_action": response.Next_action,
            "response": "",
        }

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
        "rag": "run_rag_agent",
        "end": END,
    })

    graph.add_edge(START, "orch_agent")
    graph.add_edge("run_coder_agent", "orch_agent")
    graph.add_edge("run_research_agent", "orch_agent")
    graph.add_edge("run_document_agent", "orch_agent")
    graph.add_edge("run_email_agent", "orch_agent")
    graph.add_edge("run_rag_agent", "orch_agent")

    return graph.compile()
