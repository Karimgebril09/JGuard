from langgraph.graph import StateGraph , START, END
from typing import Literal, Optional
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage  ,ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.tools import tool
import re
import os
import pypdf
from fpdf import FPDF

from .llm import llm




def clean_text(text: str) -> str:
    "clean the input text by removing special characters, URLs, emails, and extra spaces. "
    text = text.lower() 
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)    # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # remove emails
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)  # Remove special characters 
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces/newlines
    
    return text.strip()

@tool
def read_pdf_file(file_path: str) -> str:
    """
    Reads the text content of a PDF file given its path.
    
    Args:
        file_path (str): The local system path to the PDF file.
        
    Returns:
        str: The extracted text from all pages, or an error message if the file cannot be read.
    """
    try:
        text_content = []
        
        with open(file_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text_content.append(extracted_text)
        
        return clean_text("\n".join(text_content))

    except FileNotFoundError:
        return f"Error: The file at {file_path} was not found."
    except Exception as e:
        return f"Error: An unexpected error occurred while reading the PDF: {str(e)}"
    


@tool
def write_documentation_to_pdf(file_path: str, documentation: str) -> str:
    """
    Writes a string of documentation text into a PDF file.
    
    Args:
        file_path (str): The destination path where the PDF will be saved.
        documentation (str): The text content to be written into the PDF.
        
    Returns:
        str: A success message with the file path or an error message.
    """
    try:
        # Initialize FPDF object
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        pdf.set_font("Arial", size=12)
        
        # Multi_cell allows for automatic line wrapping
        # w=0 means the cell stretches to the right margin
        pdf.multi_cell(w=0, h=10, txt=documentation)
        
        # Output the file
        pdf.output(file_path)
        
        return f"Successfully created PDF documentation at: {file_path}"

    except PermissionError:
        return f"Error: Permission denied. Close the file if it is open in another program."
    except Exception as e:
        return f"Error creating PDF: {str(e)}"
    

def create_llms(local:bool=True):
    
    load_dotenv()
    # if local:
    doc_agent = llm.bind_tools([read_pdf_file, write_documentation_to_pdf])
    # else:
        # gemini_api_key = os.getenv("GEMINI_API_KEY")
        # doc_agent = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=gemini_api_key).bind_tools([read_pdf_file, write_documentation_to_pdf])
   
    return doc_agent

document_processor_llm = create_llms(local=True)


class DocumentProcessorAgentState(MessagesState):
    request: str
    document:str

def document_processor_agent(state: DocumentProcessorAgentState):
    messages = [
        SystemMessage(
            content=(
                "you are a document processor agent. "
                "you are supposed to do either [1] read a documentation file from a given file path, "
                "or [2] generate documentation based on the provided code and write it to a PDF file  "
                "you should pick your actions according to user's request. "
                "you have access to two tools: read_pdf_file(file_path) and write_documentation_to_pdf(file_path, documentation). "
                "your output should be only the documentation no other additional text if you choose to write documentation. "
                "if user didn't provide a file path, you should give a file name for generating the documentation PDF file. "
                "for reading documentation dont use any tool and return message with content that path is required. "
            )
        ),
        HumanMessage(
            content="user request: " + state["request"] 
        ),
    ]

    response = document_processor_llm.invoke(messages)

    return {
        "messages": [response],
    }

def route(state: DocumentProcessorAgentState) -> Literal["tools_output", "end"]:
    print("fififi")
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "tools_output"


def build_document_processor():
    graph = StateGraph(DocumentProcessorAgentState)

    graph.add_node("document_processor",document_processor_agent)
    graph.add_node("tools", ToolNode([read_pdf_file, write_documentation_to_pdf]))

    graph.add_edge(START, "document_processor")
    graph.add_conditional_edges("document_processor", route, {
        "tools_output": "tools",
        "end": END,
    })
    graph.add_edge("tools", END)

    app = graph.compile()
    return app

# if __name__ == "__main__":
#     app = build_document_processor()
#     app.run(