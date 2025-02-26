from langchain_aws import BedrockLLM, ChatBedrock
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from dotenv import load_dotenv, find_dotenv
import os
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.tools import TavilySearchResults
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from GlobalState import GlobalState
from rag_retriever_chroma import retriever_chroma
import asyncio

load_dotenv(find_dotenv())

# llm = BedrockLLM(
#     credentials_profile_name="default", model_id="meta.llama3-8b-instruct-v1:0"
# )

# llm = ChatBedrock(credentials_profile_name="default",
#                   model_id="meta.llama3-8b-instruct-v1:0",
#                   model_kwargs=dict(temperature=0))

llm = ChatOllama(model="llama3:8b", temperature=0.0)
async def retrieve(state: GlobalState) -> GlobalState:
    
    """
    Retrieve documents from the Chroma vectorstore.

    Args:
        state (GlobalState): Current state containing the user query.

    Returns:
        GlobalState: Updated state with retrieved documents.
    """
    question = state["query"]
    rag_docs = await retriever_chroma.ainvoke(question)

    
    retrieved_docs = [doc.page_content for doc in rag_docs]

    updated_state = state.copy()
    updated_state["documents"] = retrieved_docs
    return updated_state 


async def generate(state: GlobalState) -> GlobalState:
    """
    Generate an answer using RAG (Retrieve and Generate) on retrieved documents.

    Args:
        state (GlobalState): Current state containing the user query and retrieved documents.

    Returns:
        GlobalState: Updated state with a new key `generation` containing the generated response.
    """
    query = state["query"]
    documents = state["documents"]
    
    context = "\n\n".join(documents) if documents else "No relevant context available."

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an AI assistant representing SalarySe, specializing in answering questions about our company, products, and services from our perspective. 
        Speak as if you are part of the company, using "we" to represent SalarySe. 
        Provide clear and concise answers with a maximum of 10 lines. 
        If the information is not available or unclear, respond with "I'm sorry, I don't have that information." 
        Tailor responses to maintain a professional and informative tone.

        Query: {query}
        Context: {context}
        Answer:
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["query", "context"]
    )

    rag_chain = prompt | llm | StrOutputParser()

    try:
        generation = await rag_chain.ainvoke({"context": context, "query": query})
    except Exception as e:
        generation = f"I'm sorry, an error occurred while generating the response: {str(e)}"

    updated_state = state.copy()
    updated_state["generation"] = generation.strip()
    updated_state["messages"] = [AIMessage(content=generation.strip())]
    return updated_state



workflow = StateGraph(GlobalState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")

workflow.add_edge("generate", END)

rag_agent = workflow.compile()

if __name__ == "__main__":
    query = "RBL card blocked"
    initial_state = GlobalState({"query": query})
    config = {"configurable": {"thread_id": "1"}}
    final_state = asyncio.run(rag_agent.ainvoke(initial_state, config=config))
    print("retrived docs: \n")
    for doc in final_state.get("documents", "No response generated."):
        print("\n",doc)
    print("-------------------------------------------\n")
    print("Final response: \n")
    
    print(final_state.get("generation", "No response generated."))