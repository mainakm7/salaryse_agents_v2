from langchain_ollama import ChatOllama
from langchain_aws import BedrockLLM, ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from GlobalState import GlobalState
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
import asyncio


load_dotenv(find_dotenv())


chat_llm = ChatBedrock(credentials_profile_name="default",
                  model_id="meta.llama3-8b-instruct-v1:0",
                  model_kwargs=dict(temperature=0.7))

async def chat(state: GlobalState) -> GlobalState:
    
    query = state.get("query")
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an AI assistant representing SalarySe, a professional organization. 
        Your primary role is to provide accurate, concise, and professional responses to user queries. 
        You can also carry out general conversations.
        Use "we" to refer to SalarySe, and always maintain a respectful and informative tone.
        Ensure responses are tailored to the user's input and avoid irrelevant details.

        Query: {query}
        Response:
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["query"]
    )
    
    chat_chain = prompt | chat_llm | StrOutputParser()
    try:
        generation = await chat_chain.ainvoke({"query": query})
    except Exception as e:
        generation = f"I'm sorry, an error occurred while generating the response: {str(e)}"
        
    updated_state = state.copy()
    updated_state["generation"] = generation.strip()
    updated_state["messages"] = AIMessage(content=generation.strip())
    return updated_state

workflow = StateGraph(GlobalState)
workflow.add_node("chat", chat)
workflow.set_entry_point("chat")
workflow.add_edge("chat", END)
inmemory = MemorySaver()
chat_agent = workflow.compile()

if __name__ == "__main__":
    query = "How are you today?"
    query_input = {"messages": [HumanMessage(content=query)], "query": query}

    response = asyncio.run(chat_agent.ainvoke(query_input))
    print(response["messages"][-1].content)