from graphbuilder import ss_agent
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import HumanMessage
import asyncio

app = FastAPI()

class AppInput(BaseModel):
    thread_id: str = Field("1", description="Thread ID for the current user session.")
    query: str = Field(..., description="User query for the SalarySe assistant.")

@app.post("/ask", status_code=status.HTTP_201_CREATED)
async def ask_agent(input: AppInput):
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