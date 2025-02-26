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
    query = state.get("query", "")
    conversation_history = state.get("summary", "")

    manager_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a multi-agent AI assistant for the company SalarySe.
        Your job is to act as a manager agent that routes queries to worker agents based on their content. 
        Your task is to decide query routing based on its content.
        
        - If the query is a general conversational query (e.g., "How are you?", "What did I ask previously?", "Tell me something", or anything unrelated to SalarySe), respond with "chat_agent".
        - If the query is about the company SalarySe, its policies, products, or data, respond with "rag_agent".
        - If the query asks for context about previous queries (e.g., "What did I ask previously?" or similar), respond with "chat_agent".
        - For any other cases, respond with "END" if you cannot determine a specific action.

        **Respond only with a single word: "chat_agent", "rag_agent", or "END".** 
        Do not provide any extra explanation or context.
        
        You can access the previous conversation history through the 'summary' key in the state.

        Query: {query}
        summary: {summary}
        Answer:
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["query", "messages"],
    )


    manager_chain = manager_prompt | manager_llm | StrOutputParser()
    
    try:
        response = await manager_chain.ainvoke({"query": query, "summary": conversation_history})
    except Exception as e:
        response = "END"

    valid_responses = {"chat_agent", "rag_agent", "END"}
    if response not in valid_responses:
        response = "END" 

    updated_state = {
        "intent": response
    }

    return updated_state



def intent_classifier(state: GlobalState) -> str:
    intent = state.get("intent", "")
    return intent


if __name__ == "__main__":
    query = "What did I ask previously?"

    query_input = {"messages": [HumanMessage(content=query)], "query": query}

    response = asyncio.run(manager_agent(query_input))

    print("routing to :", response["intent"])
