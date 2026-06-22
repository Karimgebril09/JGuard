import os
from langchain_ollama import ChatOllama

model_id = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct")
llm = ChatOllama(model=model_id,
                )