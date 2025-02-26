from typing import Annotated, List, TypedDict, Union, Dict, Any
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

class GlobalState(TypedDict):
  messages: Annotated[List[AnyMessage], add_messages]
  intent: str
  query: str
  context : str
  generation: Union[str, List[Any]]
  documents: List[str]
  sql_query: str
  sql_result: str