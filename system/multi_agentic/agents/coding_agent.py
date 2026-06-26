from langgraph.graph import StateGraph, START, END
from typing import Literal
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import MessagesState
from defenders.tools.code.src.code_defender import CodeDefender
import traceback
import warnings
import os
warnings.filterwarnings("ignore")


@tool
def execute_code(code: str) -> str:
    """
    executes python code containing both code and tests.
    Tests will be expected to raise assertions errors if a test fails
    and message with the failed assertion will be return
    if it pass nothing should appear
    """
    try:
        exec(code, {}, {}) # empty global and local for isolation
        return "ALL_TEST_CASES_PASSED"
    except AssertionError as e: # if testing assertion fails
        return str(e)
    except Exception: # other errors raised
        return traceback.format_exc()


class CodingAgentState(MessagesState):
    problem_description: str
    code: str


def create_llms():
    model_id = os.environ.get("OLLAMA_MODEL","qwen3:4b-instruct")
    coding_llm = ChatOllama(model=model_id).bind_tools([execute_code]) # make the llm know it have this tool
    feedback_llm = ChatOllama(model=model_id)
    return coding_llm, feedback_llm

coding_llm, feedback_llm = create_llms()


def coding_agent(state: CodingAgentState) -> CodingAgentState:
    messages = [
        SystemMessage(
            content=(
                "You are a coding agent:"
                "Write ONLY code with No explanations. "
                "use Python only and generate functional, efficient code "
                "Make sure only code is the output dont add any thing around it "
                "dont do debugging prints or any thing else just the code "
                "you should generate only one solution for the problem described below "
                "you should not generate anything related to markdown or any other format just pure code "
                "MAKE SURE THE FINAL RESPONSE IS ONLY CODE AND NOTHING ELSE "
                "also if there are any feedbacks given by feedback agent consider them and improve the code and fix issues "
                "if not ignore the feedback part"
                "you have the choice to either make a test functions and call the tool to test it or "
                "directly generate the final code without making test cases and calling the tool"
                "if you feel something suspicious about the code you generated exit"
                f"feedback:{state['messages'][-1].content if len(state['messages']) > 0 else 'no feedbacks given yet'}"
            )
        ),
        HumanMessage(
            content=f"Problem description:\n {state['problem_description']} "
        ),
    ]
    resp = coding_llm.invoke(messages)
    tool_calls = getattr(resp, "tool_calls", None)
    if tool_calls:
        content = tool_calls[0]["args"].get("code", "")
    elif type(resp.content) == str:
        content = resp.content
    else: # gemini return different format
        content = resp.content[0].get("text", str(resp.content))
    return {
        "code": content.strip(),
        "messages": [resp],
    }


def feedback_agent(state: CodingAgentState) -> dict:
    messages = [
        SystemMessage(
            content="You are a feedback agent. Analyze test results and explain how to improve or fix bugs in the code."
        ),
        HumanMessage(
            content=f"""
            Problem: \n {state['problem_description']} \n
            Current code:\n{state['code']} \n
            Test result: \n {state['messages'][-1].content}
            """
        ),
    ]
    resp = feedback_llm.invoke(messages)
    return {
        "code": resp.content.strip(),
        "messages": state["messages"] + [resp],
    }


def router1(state: CodingAgentState) -> Literal["cont", "end"]:
    if "ALL_TEST_CASES_PASSED" in state["messages"][-1].content:
        return "end"
    return "cont"

def router2(state: CodingAgentState) -> Literal["execute Tool", "end"]:
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", None)
    if tool_calls:
        return "execute Tool"
    return "end"


def build_coding_agent(code_execution_protection: bool = True, code_deep_check: bool = True):
    defender = CodeDefender(deep_check=code_deep_check) if code_execution_protection else None

    def code_defense_node(state: CodingAgentState) -> dict:
        if defender is None:
            return {}
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        if tool_calls:
            code_to_execute = tool_calls[0]["args"]["code"]
            safe, reason = defender.is_safe(code_to_execute)
            if not safe:
                print(f"[CodeDefender] execution blocked: {reason}")
                return {
                    "code": f"[EXECUTION BLOCKED] {reason}",
                }
        return {}

    graph = StateGraph(CodingAgentState)
    graph.add_node("coding", coding_agent)
    graph.add_node("code_defense", code_defense_node)
    graph.add_node("execute Tool", ToolNode([execute_code]))
    graph.add_node("feedback", feedback_agent)

    graph.add_edge(START, "coding")
    graph.add_edge("coding", "code_defense")
    graph.add_edge("feedback", "coding")

    graph.add_conditional_edges(
        "execute Tool",
        router1,
        {
            "cont": "feedback",
            "end": END
        },
    )
    graph.add_conditional_edges(
        "code_defense",
        router2,
        {
            "execute Tool": "execute Tool",
            "end": END
        },
    )

    return graph.compile()


if __name__ == "__main__":
    app = build_coding_agent(code_execution_protection=True)
    initial_state = {
        "problem_description": (
            "i want you to give me a python code that do the following it takes a list of values and start to sum the even values"
            "use code execution tools"
        ),
        "code": "",
        "messages": [],
    }
    final_state = app.invoke(initial_state)
    print("final code:", final_state["code"])
