from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper , WikipediaAPIWrapper 
from langchain_community.tools import ArxivQueryRun , WikipediaQueryRun
from langchain_tavily import TavilySearch
from langchain_google_genai import  ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict , Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState , START , END , add_messages , StateGraph
from langgraph.prebuilt import tools_condition , ToolNode
import os
from langchain_core.messages import HumanMessage


load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")


groq_model = ChatGoogleGenerativeAI(
    api_key=gemini_api_key,
    model="gemini-2.5-pro",
    temperature=0.5
)

arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=200) 
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api_wrapper , description="Query pappers")

wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=300)

wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)


tavily_tool = TavilySearch(api_key = tavily_api_key)


tools = [arxiv_tool,wikipedia_tool,tavily_tool]

tool_llm=groq_model.bind_tools(tools=tools)


class State(TypedDict):
    messages : Annotated[list[AnyMessage],add_messages]

def tool_call_llm(state: State) -> State:
    result = tool_llm.invoke(state["messages"])
    return {"messages": [result]}


builder = StateGraph(state_schema=State)

builder.add_node("tool_call_llm",tool_call_llm)
builder.add_node("tools",ToolNode(tools=tools))

builder.add_edge(START,"tool_call_llm")
builder.add_conditional_edges("tool_call_llm",tools_condition)

builder.add_edge("tools",END)

graph= builder.compile()

try:
    tool_png = graph.get_graph().draw_mermaid_png()
    with open("graphimage.png","wb") as f:
        f.write(tool_png)
except Exception as e:
    print("error while create image",str(e))



# response = graph.invoke({"messages": [HumanMessage(content="what is the latest news about tamilnadu election?")]})
response = graph.invoke({"messages": [HumanMessage(content="explain about apj abdul kalam")]})

for res in response["messages"]:
    print(res.pretty_print())
