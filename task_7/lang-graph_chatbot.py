from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def chatbot(state: State) -> State:
    response = llm.invoke(state["messages"])
    if isinstance(response, BaseMessage):
        return {"messages": [response]}
    else:
        return {"messages": [AIMessage(content=response)]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    output = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config,
    )
    print("AI:", output["messages"][-1].content)
