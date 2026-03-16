from langgraph.graph import StateGraph , START, END
from typing import Literal,TypedDict
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage ,AIMessage 
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import traceback


from .llm import llm


@tool
def execute_code(code: str) -> str:
    """
    Executes python code containing both implementation + tests.
    Tests MUST raise AssertionError on failure.

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
    test_code: str

def create_llms():
    
    coding_llm =llm
    tester_llm = llm.bind_tools([execute_code])
    feedback_llm = llm

    return coding_llm, tester_llm, feedback_llm

coding_llm, tester_llm, feedback_llm = create_llms()


def coding_agent(state: CodingAgentState) -> CodingAgentState:
    print("coding_agent state")
    messages = [
        SystemMessage(
            content=(
                "You are a coding agent. "
                "Write ONLY code. No explanations. "
                "Use Python unless specified otherwise."
                "generate functional, efficient, and well-structured code."\
                "Make sure only code is the output dont add any thing around it"
                "dont do prints or any thing else just the code"
                "you should generate only one solution for the problem described below"
                "you should not generate anything related to markdown or any other format just pure code"
                "only code is generated no explanations no markdown no comments just code"
                "also if there are any feedbacks given by feedback agent consider them and improve the code accordingly."\
                "if not ignore the feedback part"
                f"feedback:{state['messages'][-1].content if len(state['messages'])>0 else 'no feedbacks given yet'}"
            )
        ),
        HumanMessage(
            content=f"Problem description:\n{state['problem_description']} "\
        ),
    ]


    resp = coding_llm.invoke(messages)
    # Handle both string and list content (Gemini returns list)
    content = resp.content if isinstance(resp.content, str) else resp.content[0].get("text", str(resp.content))
    return {
        "code": content.strip(),
        "messages": [resp]
    }

def testing_agent(state: CodingAgentState) -> CodingAgentState:
    print("testing_agent state")
    messages = [
        SystemMessage(
            content=(
                "You are a testing agent. "
                "Write Python code that tests the given solution. "
                "Generate test cases that cover various scenarios including edge cases. "
                "Use assertions to validate the output of the code for each test case. "
                "Make sure only code is the output dont add any thing around it"
                "dont do prints or any thing else just the code"
                "you should generate only code that tests the provided solution"
                "the code should be executable and should raise AssertionError if any test case fails with details about the failure"\
                "you should not generate anything related to markdown or any other format just pure code"
            )
        ),
        HumanMessage(
            content=f"""
                Problem:
                {state['problem_description']}

                Code to test:
                {state['code']}
                """
        ),
    ]

    resp = tester_llm.invoke(messages)
    
    return {
        "test_code": resp.content.strip(),
        "messages": [resp]
    }

def feedback_agent(state: CodingAgentState) -> CodingAgentState:
    print("feedback_agent state")
    messages = [
        SystemMessage(
            content=(
                "You are a feedback agent. "
                "you should analyze the test results and provide constructive feedback to improve the code. "
            )
        ),
        HumanMessage(
            content=f"""
                Problem:
                {state['problem_description']}

                Current code:
                {state['code']}

                Test result:
                {state['messages'][-1].content}
                """),]

    resp = feedback_llm.invoke(messages)
    state["code"] = resp.content.strip()
    return state

def router(state: CodingAgentState) -> Literal["feedback", "end"]:
    print("#################################################################")
    print("Router State:")
    for msg in state["messages"]:
        print(msg)
    if "ALL_TEST_CASES_PASSED" in state["messages"][-1].content:
        return "end"
    return "cont"

def build_coding_agent():
    graph = StateGraph(CodingAgentState)

    graph.add_node("coding", coding_agent)
    graph.add_node("testing", testing_agent)
    graph.add_node("execute Tool", ToolNode([execute_code]))
    graph.add_node("feedback", feedback_agent)

    graph.add_edge(START, "coding")
    graph.add_edge("coding", "testing")
    graph.add_edge("testing", "execute Tool")
    graph.add_edge("feedback", "coding")

    graph.add_conditional_edges(
        "execute Tool",
        router,
        {
            "cont": "feedback",
            "end": END,
        },
    )

    app = graph.compile()
    return app


# if __name__ == "__main__":
#     app = build_coding_agent()
#     initial_state = {
#         "problem_description": "Write a function that takes a list of integers and returns the sum of all even numbers in the list.",
#         "code": "",
#         "test_code": "",
#         "messages": [],
#     }
#     final_state = app.invoke(initial_state)
#     print("Final State:")
#     print(final_state)