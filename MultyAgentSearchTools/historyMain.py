from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage , AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict , List , Union
import os

load_dotenv()



api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(
    api_key=api_key,
    model="openai/gpt-oss-20b",
    temperature=0
)


class AgentState(TypedDict):
    message : List[ Union[ HumanMessage , AIMessage ]]


def process(state:AgentState)->AgentState:
    """this node will solve the request input"""
    print("\n state message :\n",state["message"])
    response = llm.invoke(state['message'])
    print("\n AI Response \n :",response.content)
    state['message'].append(AIMessage(content=response.content))
    print("\n state message after ai response: \n" , state["message"])
    return state

builder = StateGraph(state_schema=AgentState)
builder.add_node("process",process)
builder.set_entry_point("process")
builder.set_finish_point("process")

graph = builder.compile()


try:
    image = graph.get_graph().draw_mermaid_png()
    with open("graphImage.png","wb") as f:
        f.write(image)
    print("image created successfully..🤡")
except Exception as e:
    print("error handling",str(e))




conversation_history = []

user_input = input("enter request :")

while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = graph.invoke(AgentState(message=conversation_history))
    print("\n final answer result : \n ",result["message"])
    conversation_history = result["message"]
    print("\n new conversation history : \n" , conversation_history)
    print("=============================================================================================================================================================")
    user_input = input("Enter request :")
