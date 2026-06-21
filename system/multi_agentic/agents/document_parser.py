from langgraph.graph import StateGraph, START, END
from typing import Literal
from langchain.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import MessagesState
from dotenv import load_dotenv
from langchain.tools import tool
import re
import pypdf
from fpdf import FPDF
from .llm import llm
from defenders.tools.document_parser import DocumentGuard, ALLOWED_INPUT_DIR, ALLOWED_OUTPUT_DIR


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.strip()


class DocumentProcessorAgentState(MessagesState):
    request: str
    document: str


def build_document_processor(document_protection: bool = True):
    load_dotenv()

    guard = DocumentGuard() if document_protection else None

    @tool
    def read_pdf_file(file_path: str) -> str:
        """
        Reads the text content of a PDF file.
        The file must be located inside the designated input directory.
        Provide only the filename (e.g. 'report.pdf'), not an absolute path.
        """
        if guard is not None:
            safe, reason, file_path = guard.check_read(file_path)
            if not safe:
                print(f"[DocumentGuard] READ BLOCKED – {reason}")
                return f"BLOCKED: {reason}"

        resolved_path = ALLOWED_INPUT_DIR / file_path
        try:
            text_content = []
            with open(resolved_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_content.append(extracted)
            return clean_text("\n".join(text_content))
        except FileNotFoundError:
            return f"Error: '{file_path}' not found in the input directory."
        except Exception as e:
            return f"Error reading PDF: {e}"

    @tool
    def write_documentation_to_pdf(file_path: str, documentation: str) -> str:
        """
        Writes documentation text into a PDF file in the designated output directory.
        Provide only the filename (e.g. 'api-docs.pdf'), not an absolute path.
        The file must not already exist.
        """
        if guard is not None:
            safe, reason, file_path = guard.check_write(file_path, documentation)
            if not safe:
                print(f"[DocumentGuard] WRITE BLOCKED – {reason}")
                return f"BLOCKED: {reason}"

        resolved_path = ALLOWED_OUTPUT_DIR / file_path
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(w=0, h=10, text=documentation)
            pdf.output(str(resolved_path))
            return f"Successfully created PDF at: {resolved_path}"
        except PermissionError:
            return "Error: Permission denied. Close the file if it is open in another program."
        except Exception as e:
            return f"Error creating PDF: {e}"

    doc_agent_llm = llm.bind_tools([read_pdf_file, write_documentation_to_pdf])

    def document_processor_agent(state: DocumentProcessorAgentState):
        messages = [
            SystemMessage(content=(
                "You are a document processor agent. "
                "You can either (1) read a PDF file using read_pdf_file(file_path), "
                "or (2) generate documentation and save it as a PDF using write_documentation_to_pdf(file_path, documentation). "
                "For reading: the file must already exist in the input directory — provide only the filename (e.g. 'report.pdf'). "
                "For writing: provide only the output filename (e.g. 'api-docs.pdf') and the full documentation text. "
                "Do not use absolute paths. Do not invent file paths that were not given or implied by the user."
            )),
            HumanMessage(content="user request: " + state["request"]),
        ]
        response = doc_agent_llm.invoke(messages)
        return {"messages": [response]}

    def route(state: DocumentProcessorAgentState) -> Literal["tools_output", "end"]:
        last_message = state["messages"][-1]
        if not getattr(last_message, "tool_calls", None):
            return "end"
        return "tools_output"

    graph = StateGraph(DocumentProcessorAgentState)
    graph.add_node("document_processor", document_processor_agent)
    graph.add_node("tools", ToolNode([read_pdf_file, write_documentation_to_pdf]))
    graph.add_edge(START, "document_processor")
    graph.add_conditional_edges("document_processor", route, {
        "tools_output": "tools",
        "end": END,
    })
    graph.add_edge("tools", END)

    return graph.compile()
