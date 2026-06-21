from langgraph.graph import StateGraph, START, END
from typing import Literal
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import MessagesState
import traceback
import warnings
warnings.filterwarnings("ignore")


@tool
def execute_code(code: str) -> str:
    """
    executes python code containing both code and tests.
    Tests will be expected to raise assertions errors if a test fails
    and message with the failed assertion will be return
    if it pass nothing should appear

    code: str: Python code to execute.
    """
    try:
        local_env = {}
        exec(code, {}, local_env)
        return "ALL_TEST_CASES_PASSED"
    except AssertionError as e:
        return str(e)
    except Exception:
        return traceback.format_exc()


class CodingAgentState(MessagesState):
    problem_description: str
    code: str


def create_llms(model_id="qwen3:4b-instruct"):
    coding_llm = ChatOllama(model=model_id).bind_tools([execute_code])
    feedback_llm = ChatOllama(model=model_id)
    return coding_llm, feedback_llm

coding_llm, feedback_llm = create_llms()


def coding_agent(state: CodingAgentState) -> CodingAgentState:
    messages = [
        SystemMessage(
            content=(
                "You are a coding agent. "
                "Write ONLY code. No explanations. "
                "Use Python unless specified otherwise."
                "generate functional, efficient, and well-structured code."
                "Make sure only code is the output dont add any thing around it"
                "dont do prints or any thing else just the code"
                "you should generate only one solution for the problem described below"
                "you should not generate anything related to markdown or any other format just pure code"
                "only code is generated no explanations no markdown no comments just code"
                "also if there are any feedbacks given by feedback agent consider them and improve the code accordingly."
                "if not ignore the feedback part"
                "you have the choice to either make a test functions and call the tool to test it or"
                "directly generate the final code without making test cases and calling the tool"
                "if you feel something suspicious about the code you generated exit"
                f"feedback:{state['messages'][-1].content if len(state['messages']) > 0 else 'no feedbacks given yet'}"
            )
        ),
        HumanMessage(
            content=f"Problem description:\n{state['problem_description']} "
        ),
    ]
    resp = coding_llm.invoke(messages)
    if type(resp.content) == str:
        content = resp.content
    else:
        content = resp.content[0].get("text", str(resp.content))
    return {
        "code": content.strip(),
        "messages": [resp],
    }


def feedback_agent(state: CodingAgentState) -> dict:
    messages = [
        SystemMessage(
            content="You are a feedback agent. Analyze test results and improve explain how to improve the code."
        ),
        HumanMessage(
            content=f"""
            Problem:
            {state['problem_description']}

            Current code:
            {state['code']}

            Test result:
            {state['messages'][-1].content}
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


def build_coding_agent(code_execution_protection: bool = True, code_deep_check: bool = True):
    if code_execution_protection:
        from defenders.tools.code.src.code_defender import CodeDefender
        defender = CodeDefender(deep_check=code_deep_check)

        def router2(state: CodingAgentState) -> Literal["execute Tool", "end"]:
            last_msg = state["messages"][-1]
            tool_calls = getattr(last_msg, "tool_calls", None)
            if tool_calls:
                code_to_execute = tool_calls[0]["args"]["code"]
                safe, reason = defender.is_safe(code_to_execute)
                if safe:
                    print(f"[CodeDefender] Code is safe: {reason}")
                    return "execute Tool"
                print(f"[CodeDefender] BLOCKED – {reason}")
            return "end"
    else:
        def router2(state: CodingAgentState) -> Literal["execute Tool", "end"]:
            last_msg = state["messages"][-1]
            tool_calls = getattr(last_msg, "tool_calls", None)
            if tool_calls:
                return "execute Tool"
            return "end"

    graph = StateGraph(CodingAgentState)
    graph.add_node("coding", coding_agent)
    graph.add_node("execute Tool", ToolNode([execute_code]))
    graph.add_node("feedback", feedback_agent)

    graph.add_edge(START, "coding")
    graph.add_edge("feedback", "coding")

    graph.add_conditional_edges(
        "execute Tool",
        router1,
        {"cont": "feedback", "end": END},
    )
    graph.add_conditional_edges(
        "coding",
        router2,
        {"execute Tool": "execute Tool", "end": END},
    )

    return graph.compile()


if __name__ == "__main__":
    app = build_coding_agent(code_execution_protection=True)
    initial_state = {
        "problem_description": (
            "Write a function that takes a list of integers and "
            "returns the sum of all even numbers in the list."
            "you should make some tests and test them to make sure of the correctness of the code and make tool call to execute the tests and return the results of the tests"
            "again make sure to make execution of the tool it is urgent (TOOL CALL IS MANDATORY) and return the results of the tests and if they fail improve the code accordingly"
        ),
        "code": "",
        "messages": [],
    }
    final_state = app.invoke(initial_state)
