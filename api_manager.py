from langgraph_supervisor import create_supervisor
from GlobalState import GlobalState
from langchain_core.messages import RemoveMessage, AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END
from api_agents.investment_agent import investment_agent
from api_agents.credit_card_agent import credit_card_agent
from api_agents.credit_score_agent import credit_score_agent
from api_agents.dashboard_agent import dashboard_agent
import asyncio
from langgraph.graph import START, StateGraph, END

api_supervisor_llm = ChatBedrock(credentials_profile_name="default",
                          model_id="meta.llama3-8b-instruct-v1:0",
                          model_kwargs=dict(temperature=0))


async def api_supervisor(state: GlobalState) -> GlobalState:
    query = state.get("query", "")

    manager_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a multi-agent AI assistant for the company SalarySe.
        Your job is to act as a manager agent that routes queries to worker agents based on their content. 
        The worker agents specialize in different api executions.
        Your task is to decide query routing based on its content.
        
        - If the query is about credit cards, respond with "credit_card_agent".
        - If the query is about user's personal data or information, respond with "dashboard_agent".
        - If the query is about investment, respond with "investment_agent".
        - If the query is about user's credit score, respond with "credit_score_agent".
        - For any other cases, respond with "END" if you cannot determine a specific action.

        **Respond only with a single word: "credit_card_agent", "dashboard_agent", "investment_agent", "credit_score_agent", or "END".** 
        Do not provide any extra explanation or context.
        

        Query: {query}
        Answer:
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["query", "messages"],
    )


    api_supervisor_chain = manager_prompt | api_supervisor_llm | StrOutputParser()
    
    try:
        response = await api_supervisor_chain.ainvoke({"query": query})
    except Exception as e:
        response = "END"

    valid_responses = {"credit_card_agent", "dashboard_agent", "investment_agent", "credit_score_agent", "END"}
    if response not in valid_responses:
        response = "END" 

    updated_state = {
        "api_intent": response
    }

    return updated_state



def api_intent_classifier(state: GlobalState) -> str:
    api_intent = state.get("api_intent", "")
    return api_intent


workflow = StateGraph(GlobalState)
workflow.add_node("api_supervisor", api_supervisor)
workflow.add_node("credit_card_agent", credit_card_agent)
workflow.add_node("credit_score_agent", credit_score_agent)
workflow.add_node("investment_agent",investment_agent)
workflow.add_node("dashboard_agent", dashboard_agent)
workflow.set_entry_point("api_supervisor")

workflow.add_conditional_edges(
    "api_supervisor",
    api_intent_classifier,
    {"credit_card_agent": "credit_card_agent",
     "credit_score_agent": "credit_score_agent",
     "investment_agent": "investment_agent",
     "dashboard_agent": "dashboard_agent",
     "END": END}
)

workflow.add_edge("credit_card_agent", END)
workflow.add_edge("credit_score_agent", END)
workflow.add_edge("investment_agent", END)
workflow.add_edge("dashboard_agent", END)

api_supervisor_agent = workflow.compile()


if __name__ == "__main__":
    query = "How do i check my credit card application status?"

    query_input = {"messages": [HumanMessage(content=query)], "query": query, "user_info": {"user_id": "740ad7d0-0b8c-4bde-a861-a97a5f2d3f52"}}

    response = asyncio.run(api_supervisor_agent.ainvoke(query_input))

    print("routing to :", response["api"])