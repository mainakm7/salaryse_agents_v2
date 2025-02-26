from graphbuilder import workflow
from fastapi import FastAPI, status, HTTPException, Depends
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import HumanMessage
from contextlib import asynccontextmanager
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import asyncio
import aiosqlite


async def init_memory():
    conn = await aiosqlite.connect("db/thread_id_memory.db")
    return AsyncSqliteSaver(conn)

class AppInput(BaseModel):
    thread_id: str = Field("1", description="Thread ID for the current user session.")
    query: str = Field(..., description="User query for the SalarySe assistant.")
    
memory = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles the startup and shutdown of the SQLite connection."""
    global memory
    print("Opening SQLite connection.")
    memory = await init_memory()  
    try:
        yield
    finally:
        print("Closing SQLite connection.")
        await memory.conn.close()


app = FastAPI(lifespan=lifespan)

async def get_memory():
    """Dependency to provide the initialized memory object."""
    if memory is None:
        raise HTTPException(status_code=500, detail="Memory not initialized.")
    return memory

@app.post("/ask", status_code=status.HTTP_201_CREATED)
async def ask_agent(input: AppInput, memory=Depends(get_memory)):
    """
    Endpoint to query the SalarySe assistant.

    Args:
        input (AppInput): The input containing the thread ID and user query.
    
    Returns:
        Response containing the assistant's answer.
    """
    query_payload = {
        "messages": [HumanMessage(content=input.query)],
        "query": input.query
    }
    config = {"configurable": {"thread_id": input.thread_id}}

    try:
        ss_agent = workflow.compile(checkpointer=memory)
        response = await ss_agent.ainvoke(query_payload, config=config)

        if (response and 
            isinstance(response, dict) and 
            "messages" in response and 
            len(response["messages"]) > 0):
            return {"response": response["messages"][-1].content}
        else:
            return {"response": "No output was generated."}
    
    except Exception as e:
        import traceback
        error_details = f"Error processing query: {str(e)}\n{traceback.format_exc()}"
        print(error_details)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request."
        )