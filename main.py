import os
from typing import Annotated,TypedDict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model

class State(TypedDict):
    messages: Annotated[list,add_messages]

builder = StateGraph(State)


model = init_chat_model("qwen-plus-latest", model_provider="openai",base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def chatbox(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}
builder.add_node("chatbox", chatbox)

builder.add_edge(START,"chatbox")
builder.add_edge("chatbox",END)
graph = builder.compile(checkpointer=MemorySaver())
thread1 = {"configurable": {"thread_id": "1"}}
result_1 = graph.invoke({"messages":[HumanMessage("hi, my name is Jack!")]}, thread1)
print(result_1)
result_2 = graph.invoke(
    { "messages": [HumanMessage("what is my name?")] },
    thread1
)
print(result_2)