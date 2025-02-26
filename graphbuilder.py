from langgraph.graph import START, StateGraph, END
from manager_agent import manager_agent, intent_classifier
from langchain_core.messages import HumanMessage
from rag_agent import rag_agent
from chat_agent import chat_agent
from GlobalState import GlobalState
from langgraph.checkpoint.memory import MemorySaver
import asyncio


workflow = StateGraph(GlobalState)


workflow.add_node("manager", manager_agent)
workflow.add_node("rag_agent", rag_agent)
workflow.add_node("chat_agent", chat_agent)
workflow.add_edge(START, "manager")


workflow.add_conditional_edges(
    "manager",
    intent_classifier,
    {"rag_agent": "rag_agent",
     "chat_agent": "chat_agent",
     "END": END}
)

workflow.add_edge("rag_agent", END)
workflow.add_edge("chat_agent", END)
inmemory = MemorySaver()

ss_agent = workflow.compile(checkpointer=inmemory)

if __name__ == "__main__":
    query = "Transaction failed during gift card"
    query_input = {"messages": [HumanMessage(content=query)], "query": query}
    config = {"configurable": {"thread_id": 1}}
    response = asyncio.run(ss_agent.ainvoke(query_input, config=config))
    print(response["messages"][-1].content)

