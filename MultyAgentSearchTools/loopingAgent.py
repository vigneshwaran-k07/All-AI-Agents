from langgraph.graph import StateGraph , START , END
from langchain_core.messages import HumanMessage , AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition , ToolNode 
from langchain_core.messages import BaseMessage , ToolMessage , SystemMessage 
from langgraph.graph.message import add_messages
from typing import TypedDict , Annotated , Sequence
import os

load_dotenv()


api_key = os.getenv('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(
    api_key=api_key,
    model="gemini-2.5-pro",
    temperature=0
)

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage] , add_messages]

@tool
def add(a:int , b:int):
    """add the following two numbers"""
    return a+b
@tool
def calculator(expr:str):
    """calculate values and return answer"""
    return eval(expr)

tools = [add , calculator] 

llm_with_tool = llm.bind_tools(tools=tools)

def model_call(state:AgentState)->AgentState:
    prompt = SystemMessage(content="you are good ai assistent.pls answer to my query correctly")
    response = llm_with_tool.invoke([prompt]+state["messages"])
    print("model response \n :",response)
    return {"messages":[response]}


def should_countinue(state:AgentState):
    message = state['messages']
    last_message = message[-1]
    if not last_message.tool_calls:
        return "exit"
    else:
        return "countinue"
    

builder = StateGraph(state_schema=AgentState)
builder.add_node("model",model_call)
builder.add_node("tool",ToolNode(tools=tools))

builder.add_edge(START , "model")
builder.add_conditional_edges(
    "model",
    should_countinue,
    {
        "countinue":"tool",
        "exit":END
    }
)
builder.add_edge("tool","model")

graph = builder.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message , tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages":[HumanMessage(content="add 5+7 and multiply the result by 6 . add 45+5 and multiply the result by 2. devide 12/6 and add the result with 50")]}

print_stream(graph.stream(input=inputs,stream_mode="values"))

try:
    image = graph.get_graph().draw_mermaid_png()
    with open("graphImage.png","wb") as f:
        f.write(image)
    print("image created successfully..🤡")
except Exception as e:
    print("error handling",str(e))



