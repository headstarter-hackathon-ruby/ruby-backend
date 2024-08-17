from typing import TypedDict


# The shared data passed between the RAG and the agents
class ComplaintState(TypedDict):
    complaint: str
    summary: str
    id: str
    category: str
    sub_category: str
    resolved: bool
    admin_text: str
    similar_complaints: list[dict]
    complaint_embedding: list[float]

