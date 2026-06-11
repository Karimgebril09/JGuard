from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from collections import deque
import os
from dotenv import load_dotenv
load_dotenv("./.env")

class LLM:
    def __init__(self,model_name: str, model_type : str, base_url: str=None,temperature: float = 0.7,\
                 system_prompt: str = None, api_key: str = None, history_length: int = 5):
        self.model_name =model_name 
        self.model=None
        self.base_url = base_url
        self.temperature = temperature
        self.system_prompt = system_prompt or ""   
        self.model_type = model_type
        self.api_key = api_key
        self.buffer = deque(maxlen=history_length) 


        if self.model_type == "ollama":
            self.model = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                base_url=self.base_url,
            )

        elif self.model_type == "gemini":
            self.model= ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,   
                google_api_key=self.api_key,
            )

        elif self.model_type == "openai":
            self.model= ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
            )

    def get_model(self):
        return self.model
    
    def generate_response(self, prompt):
        response = self.model.invoke([self.system_prompt]+[prompt])
        return response
    
    def generate_response_buffered(self, prompt):
        self.buffer.append(prompt)
        response = self.model.invoke([self.system_prompt]+list(self.buffer))
        return response

# TESTING

# if __name__ == "__main__":
#     print(os.getenv("NGROK_ATTACK_ENDPOINT"))
#     llm = LLM(model_name="qwen2.5:3b-instruct", model_type="ollama", base_url=os.getenv("NGROK_SYSTEM_ENDPOINT"))
#     response = llm.generate_response("What is the capital of France?")
#     print(response)