from GlobalState import GlobalState
from langchain_core.messages import RemoveMessage, AIMessage
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END

load_dotenv(find_dotenv())

summary_llm = ChatBedrock(credentials_profile_name="default",
                          model_id="meta.llama3-8b-instruct-v1:0",
                          model_kwargs=dict(temperature=0.7))

async def summarize_conversations(state: GlobalState) -> GlobalState:
    """
    Summarize the conversation history.
    """
    
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an AI assistant proficient in summarizing conversation history.
        You will be provided with the conversation history and your task is to generate a concise summary.
        Ensure the summary captures the essence of the conversation and is clear and concise.
        The previous conversation history can be accessed through the 'messages' key in the state.

        messages: {messages}
        Response:
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["messages"]
    )
    
    summary_chain = prompt | summary_llm | StrOutputParser()
    messages = state.get("messages", [])
    try:
        response = await summary_chain.ainvoke({"messages": messages})
    except Exception as e:
        response = f"I'm sorry, an error occurred while generating the response: {str(e)}"
    
    summary = response.strip()
    
    updated_state = state.copy()
    updated_state["messages"] = updated_state.get("messages", []) + [AIMessage(content=summary)]
    
    delete_messages = [RemoveMessage(id=m.id) for m in updated_state["messages"][:-4]]
    
    return {"summary": summary, "messages": delete_messages}

def summarization_intent(state: GlobalState) -> str:
    """Return the next node to execute."""
    
    messages = state.get("messages", [])
    
    if len(messages) > 6:
        return "summarize_conversations"
    
    return "manager"
