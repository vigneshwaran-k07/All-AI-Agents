from langchain_openai import ChatOpenAI , OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph , START , END 
from typing import TypedDict
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import create_retriever_tool
# from langchain_classic.tools.retriever import create_retriever_tool 
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=api_key,
    temperature=0.5
)

embeddings = OpenAIEmbeddings()

loader = PyPDFLoader("D:\AI-Projects\AgenticRag/vigneshwaran_k_.pdf").load()
# print(loader)
# state

splitter =  RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

docs = splitter.split_documents(loader)

vectore_store = FAISS.from_documents(documents=docs , embedding=embeddings)
retriver = vectore_store.as_retriever(search_kwargs={"k":3})

retriver_doc = create_retriever_tool(retriever=retriver,name="pdf_search",
    description="Search the resume PDF",)

def ask_pdf(question: str):
    docs = retriver.invoke(question)   # <-- Updated API

    context = "\n\n".join([d.page_content for d in docs])

    response = llm.invoke(
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    return response.content



print(ask_pdf("is he worked on any ai project?"))
