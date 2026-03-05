from dotenv import load_dotenv
from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import os
from langchain_openai import ChatOpenAI
load_dotenv()


class JudgeState(BaseModel):
    user_prompt: str
    llm_response: str
    max_trials: int
    default_val: int
    result: int
    reason: str


class RefusalClassification(BaseModel):
    harm: int = Field(..., description="1 = harmful, 0 = safe")
    reason: str = Field(..., description="Reason for the classification")


class SentinelJudge:

    def __init__(
        self,
        model_type: str,
        model_name: str,
        system_prompt_path: str,
        base_url: str | None = None,
        temperature: float = 0,
    ):

        self.model_type = model_type
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature

        self.sys_message = self._load_system_prompt(system_prompt_path)

        self.llm = self._load_llm()

        self.graph = self._build_graph()
        self.judge = self.graph.compile()

    # Load system prompt
    def _load_system_prompt(self, path: str) -> SystemMessage:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return SystemMessage(content=content)

    # Load LLM dynamically
    def _load_llm(self):

        if self.model_type == "gemini":

            api_key = os.getenv("GEMINI_API_KEY")

            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                api_key=api_key,
                temperature=self.temperature,
            )

        elif self.model_type == "ollama":

            llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
            )
        elif self.model_type=="openai":

            

            api_key = os.getenv("OPENAI_API_KEY")

            llm = ChatOpenAI(
                model=self.model_name,
                api_key=api_key,
                temperature=self.temperature,
            )
        


        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        return llm.with_structured_output(RefusalClassification)

    # Prediction Node
    def pred_node(self, state: JudgeState) -> JudgeState:

        prompt = f"""
        User Prompt: {state.user_prompt}

        LLM Response: {state.llm_response}
        """

        response = self.llm.invoke(
            [self.sys_message, HumanMessage(content=prompt)]
        )

        state.result = response.harm
        state.reason = response.reason

        state.max_trials -= 1

        if state.max_trials <= 0:
            state.result = state.default_val

        return state

    # Router
    def route_fn(self, state: JudgeState):

        if state.result in (0, 1):
            return "end"

        if state.max_trials > 0:
            return "predict"

        return "end"

    # Graph Builder
    def _build_graph(self):

        graph = StateGraph(JudgeState)

        graph.add_node("predict", self.pred_node)

        graph.add_edge(START, "predict")

        graph.add_conditional_edges(
            "predict",
            self.route_fn,
            {
                "predict": "predict",
                "end": END,
            },
        )

        return graph

    # Run Judge
    def run(self, user_prompt, llm_response, max_trials=3, default_val=0):

        state = JudgeState(
            user_prompt=user_prompt,
            llm_response=llm_response,
            max_trials=max_trials,
            default_val=default_val,
            result=-1,
            reason="",
        )

        return self.judge.invoke(state)