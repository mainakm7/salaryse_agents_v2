from langchain_ollama import ChatOllama
from langchain_aws import BedrockLLM, ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.types import Command
from typing import Literal
from GlobalState import GlobalState
import asyncio


manager_llm = ChatBedrock(credentials_profile_name="default",
                  model_id="meta.llama3-8b-instruct-v1:0",
                  model_kwargs=dict(temperature=0.3))


async def manager_agent(state: GlobalState) -> GlobalState:
    """
    Manager agent that routes queries to worker agents based on their content.

    Args:
        state (GlobalState): The current global state containing the user query.

    Returns:
        Updated state
    """
    query = state.get("query", "")
    
    manager_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a multi-agent AI assistant for the company SalarySe.
        Your job is to act as a manager agent that routes queries to worker agents based on their content. 
        Your task is to decide query routing based on its content.
        - If the query is a general conversational query (eg. "How are you?" or anything unrelated to SalarySe which you have answers to already), respond with "chat_agent".
        - If the query is about the company SalarySe or its policies or its products or the data, respond with "rag_agent".
        - For all other cases, respond with "END".

        Query: {query}
        Answer:
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["query"],
    )

    manager_chain = manager_prompt |manager_llm | StrOutputParser()
    try:
        response = await manager_chain.ainvoke({"query": query})
    except Exception as e:
        response = "END"  

    valid_responses = {"chat_agent", "rag_agent", "END"}
    if response not in valid_responses:
        response = "END" 


    return {"messages": AIMessage(content=response), "intent":response}
  
def intent_classifier(state: GlobalState) -> str:
    intent = state.get("intent", "")
    return intent


if __name__ == "__main__":
    query = "What are scoins?"

    query_input = {"messages": [HumanMessage(content=query)], "query": query}

    response = asyncio.run(manager_agent(query_input))

    print("routing to :",response["intent"])