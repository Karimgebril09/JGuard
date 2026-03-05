from langchain_ollama import ChatOllama

class TargetLLM:
    def __init__(self,model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url

    def get_model(self):
        return ChatOllama(
            model=self.model_name,
            base_url=self.base_url
        )

