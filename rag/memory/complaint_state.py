from typing import TypedDict


# The shared data passed between the RAG and the agents
class ComplaintState(TypedDict):
    complaint: str
    id: str
    time: str
    category: str
    sub_category: str
    similar_complaints: list[dict]
    complaint_embedding: list[float]

