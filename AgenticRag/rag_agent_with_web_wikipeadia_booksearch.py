from langchain_community.document_loaders import PyPDFLoader 
from models import DocumentModel
from db import get_db
from fastapi import HTTPException , status , UploadFile , File , APIRouter , Depends
from schemas import DocumentSchema
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import ArxivAPIWrapper , WikipediaAPIWrapper 
from langchain_community.tools import ArxivQueryRun , WikipediaQueryRun
from langchain_openai import ChatOpenAI , OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI , GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
from langgraph.prebuilt import tools_condition ,ToolNode
from langchain_core.messages import BaseMessage , SystemMessage , HumanMessage , AIMessage
from typing import TypedDict , List , Optional , Annotated , Sequence , Literal
from langgraph.graph import START , StateGraph , END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from langchain_tavily import TavilySearch
from pydantic import BaseModel
import os

load_dotenv()

router = APIRouter()
tavily_api_key = os.getenv("TAVILY_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=api_key,
    temperature=0
)

embeddings = OpenAIEmbeddings(
     model="text-embedding-3-small",
    api_key=api_key
)

def setup_vector_store():
    """Setup vector store from PDF documents"""
    if os.path.exists("vectordatas"):
        vector_store = Chroma(
            persist_directory="vectordatas",
            embedding_function=embeddings
        )
        print("Loaded existing vector store")
    else:
        loader = PyPDFLoader(file_path="documents/vigneshwaran_k.pdf")
        pages = loader.load()
        
        split_docs = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = split_docs.split_documents(documents=pages)
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="vectordatas"
        )
        print("Created new vector store")
    
    return vector_store

# Initialize vector store
vector_store = setup_vector_store()

tavily_tool = TavilySearch(api_key=tavily_api_key)
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper, description="Query papers")
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=200)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

tools = [tavily_tool, arxiv_tool, wikipedia_tool]
llm_with_tools = llm.bind_tools(tools=tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class QuestionSchema(BaseModel):
    question: str


def rag_node(state: AgentState) -> AgentState:
    """
    Search documents and try to answer from RAG.
    If no relevant info found, let agent use external tools.
    """
    messages = state["messages"]
    user_question = messages[-1].content
    
    print(f"🔍 RAG: Searching documents for: {user_question}")
    
    try:
        docs = vector_store.similarity_search(user_question, k=3)
        
        if not docs:
            print("RAG: No relevant documents found, routing to external tools...")
         
            no_info_message = AIMessage(
                content="[RAG_NO_INFO] No relevant information found in documents."
            )
            return {"messages": [no_info_message]}
        
        context = "\n\n".join([
            f"[Chunk {i+1}]\n{doc.page_content}" 
            for i, doc in enumerate(docs)
        ])
        
        # Create RAG prompt
        rag_prompt = f"""You are answering based on document context. 

Context from documents:
{context}

User question: {user_question}

Instructions:
- If the context contains relevant information to answer the question, provide a complete answer.
- If the context does NOT contain relevant information, respond with EXACTLY: "[NO_RELEVANT_INFO]"
- Do not make up information not in the context.

Answer:"""
        
        # Get RAG response
        response = llm.invoke([HumanMessage(content=rag_prompt)])
        
      
        if "[NO_RELEVANT_INFO]" in response.content:
            print(" RAG: Context not relevant, routing to external tools...")
            no_info_message = AIMessage(
                content="[RAG_NO_INFO] Context not relevant to question."
            )
            return {"messages": [no_info_message]}
        
        print("RAG: Found relevant information in documents")
        return {"messages": [response]}
        
    except Exception as e:
        print(f"❌ RAG Error: {str(e)}")
        error_message = AIMessage(content=f"[RAG_ERROR] {str(e)}")
        return {"messages": [error_message]}


def agent_node(state: AgentState) -> AgentState:
    """
    Agent that uses external tools (Tavily, ArXiv, Wikipedia)
    """
    messages = state["messages"]
    
    print(" Agent: Using external tools...")
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def route_after_rag(state: AgentState) -> Literal["agent", "end"]:
    """
    Check if RAG answered the question or if we need external tools
    """
    messages = state["messages"]
    last_message = messages[-1].content
    
    if "[RAG_NO_INFO]" in last_message or "[RAG_ERROR]" in last_message or "[NO_RELEVANT_INFO]" in last_message:
        print("→ Routing to Agent (external tools)")
        return "agent"
    
    print("→ RAG answered successfully, ending")
    return "end"


# Build the graph
builder = StateGraph(state_schema=AgentState)

# Add nodes
builder.add_node("rag", rag_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools=tools))

# Start with RAG
builder.add_edge(START, "rag")

# After RAG, decide: end or use external tools
builder.add_conditional_edges(
    "rag",
    route_after_rag,
    {
        "agent": "agent",
        "end": END
    }
)

# Agent can use tools
builder.add_conditional_edges(
    "agent",
    tools_condition
)

# Tools go back to agent
builder.add_edge("tools", "agent")

graph = builder.compile()


# Save graph visualization
try:
    tool_png = graph.get_graph().draw_mermaid_png()
    with open("graphimage.png", "wb") as f:
        f.write(tool_png)
    print(" Graph visualization saved to graphimage.png")
except Exception as e:
    print("Error while creating image:", str(e))


# Interactive chat loop
print("\n" + "="*60)
print(" AI Assistant with RAG + External Tools")
print("="*60)
print("Flow: RAG first → If no answer → External tools")
print("Type 'exit' to quit\n")
print("="*60 + "\n")

user_input = input("You: ")

while user_input.lower() != "exit":
    try:
        response = graph.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
        
        final_answer = response["messages"][-1].content
        
        if "[RAG_NO_INFO]" in final_answer or "[RAG_ERROR]" in final_answer or "[NO_RELEVANT_INFO]" in final_answer:
            final_answer = "I couldn't find information in the documents or external sources."
        
        print("\n" + "─"*60)
        print("Assistant:", final_answer)
        print("─"*60 + "\n")
        
    except Exception as e:
        print(f"\n Error: {str(e)}\n")
    
    user_input = input("You: ")

print("\n Chat ended. Goodbye!\n")


# FastAPI endpoint
@router.post("/ask-ai")
def ask_ai(data: QuestionSchema):
    try:
        response = graph.invoke({
            "messages": [HumanMessage(content=data.question)]
        })
        
        answer = response["messages"][-1].content
        
        if "[RAG_NO_INFO]" in answer or "[RAG_ERROR]" in answer or "[NO_RELEVANT_INFO]" in answer:
            answer = "I couldn't find relevant information to answer your question."
        
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

