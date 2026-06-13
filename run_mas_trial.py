import argparse
from typing import Any

from dotenv import load_dotenv

# Import the orchestrator module. We compile a runtime graph without the custom
# safety checkpointer to keep this local trial simple and dependency-light.
from system.multi_agentic.agents import app as mas_app_module


def _extract_last_ai_text(final_state: dict[str, Any]) -> str:
    messages = final_state.get("messages", [])
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai":
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                return content
            return str(content)
    return "<No AI message returned>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick local trial for JGuard multi-agent system")
    parser.add_argument(
        "--message",
        type=str,
        default="Use web search to find the local time and weather in damietta, egypt. then end",
        help="User message sent to the MAS orchestrator",
    )
    args = parser.parse_args()

    load_dotenv()

    runtime_app = mas_app_module.graph.compile()

    initial_state = {
        "user_message": args.message,
        "messages": [],
    }

    final_state = runtime_app.invoke(initial_state, config={"recursion_limit": 30})

    print("=== MAS Trial Result ===")
    print(f"Input message: {args.message}")
    print(f"Final next_action: {final_state.get('next_action')}")
    print("Final response field:")
    print(final_state.get("response", ""))
    print("\nLast AI message:")
    print(_extract_last_ai_text(final_state))


if __name__ == "__main__":
    main()
