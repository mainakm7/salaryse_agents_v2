from langchain_ollama import ChatOllama
from langchain_aws import BedrockLLM, ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
import asyncio

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GlobalState import GlobalState

load_dotenv(find_dotenv())


investment_llm = ChatBedrock(credentials_profile_name="default",
                  model_id="meta.llama3-8b-instruct-v1:0",
                  model_kwargs=dict(temperature=0))

async def investment_api(state: GlobalState) -> GlobalState:
    
    query = state.get("query")
    user_info = state.get("user_info", {})
    user_id = user_info.get("user_id", "")
    
    user_id = "740ad7d0-0b8c-4bde-a861-a97a5f2d3f52"
        
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an AI assistant representing SalarySe, a professional organization. 
        Your primary role is to provide api access to the user queries.
        
        You have knowledge about the apis listed below:
        1.'https://api.dev.salaryse.com/gw/v1/investment/fd/upswing/initiate'. This API is used to initiate investment upswing.
        2.'https://api.dev.salaryse.com/gw/v1/investment/fd/dashboard'. This API is used to access the investment dashboard for the user.
        
        based on the query you have to decide which api to call.
        
        **Your response should be formatted as a Json object with a key 'api' and the value as the name of the api or 'No API Found' if no api is found.**
        
        Query: {query}
        user_id: {user_id}
        Response:
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["query"]
    )
    
    investment_chain = prompt | investment_llm | JsonOutputParser()
    try:
        generation = await investment_chain.ainvoke({"query": query, "user_id": user_id})
    except Exception as e:
        generation = f"I'm sorry, an error occurred while generating the response: {str(e)}"
        
    updated_state = state.copy()
    updated_state["api"] = generation["api"].strip()
    updated_state["messages"] = [AIMessage(content="API provided".strip())]
    return updated_state

workflow = StateGraph(GlobalState)
workflow.add_node("investment", investment_api)
workflow.set_entry_point("investment")
workflow.add_edge("investment", END)
investment_agent = workflow.compile()

if __name__ == "__main__":
    query = "Can you direct me to my investment dashboard?"
    query_input = {"messages": [HumanMessage(content=query)], "query": query}
    config = {"configurable": {"thread_id": "1"}}
    response = asyncio.run(investment_agent.ainvoke(query_input, config=config))
    print(response["messages"][-1].content)
    print(response["api"])