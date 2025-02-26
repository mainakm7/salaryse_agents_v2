from langgraph.graph import START, StateGraph, END
from manager_agent import manager_agent, intent_classifier
from langchain_core.messages import HumanMessage
from rag_agent import rag_agent
from chat_agent import chat_agent
from summarize_coversations import summarize_conversations, summarization_intent
from GlobalState import GlobalState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import asyncio
import aiosqlite


workflow = StateGraph(GlobalState)

workflow.add_node("summarize_conversations", summarize_conversations)
workflow.add_node("manager", manager_agent)
workflow.add_node("rag_agent", rag_agent)
workflow.add_node("chat_agent", chat_agent)

workflow.add_conditional_edges(
    START,
    summarization_intent,
    {"summarize_conversations": "summarize_conversations",
     "manager": "manager"}
)
workflow.add_edge("summarize_conversations", "manager")
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

async def init_memory():
    conn = await aiosqlite.connect("db/chat_memory.db")
    memory = AsyncSqliteSaver(conn)
    return memory


asqlmemory = asyncio.run(init_memory())

ss_agent = workflow.compile(checkpointer=asqlmemory)

if __name__ == "__main__":
    query = "Tell me about our previous conversation"
    query_input = {"messages": [HumanMessage(content=query)], "query": query}
    config = {"configurable": {"thread_id": "1"}}
    
    try:
        response = asyncio.run(ss_agent.ainvoke(query_input, config=config))
        print(response["messages"][-1].content)
        print("----------------\n")
        print(response["messages"])
        print("----------------\n")
        print(response["summary"])
    except Exception as e:
        print(f"Error invoking agent: {str(e)}")
    finally:
        asyncio.run(asqlmemory.conn.close())


