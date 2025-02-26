from typing import Annotated, List, TypedDict, Union, Dict, Any
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages, MessagesState

class GlobalState(MessagesState):
  # messages: Annotated[List[AnyMessage], add_messages]
  intent: str
  query: str
  context : str
  generation: Union[str, List[Any]]
  documents: List[str]
  summary: str
  config: Dict[str, Any]