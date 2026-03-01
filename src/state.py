from typing import List, TypedDict

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str # "yes" or "no"
    documents: List[str]