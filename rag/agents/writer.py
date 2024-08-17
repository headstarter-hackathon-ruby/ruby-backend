import os
import uuid

from openai import OpenAI
from pinecone import Pinecone

from rag.memory.complaint_state import ComplaintState


async def write(state: ComplaintState):
    """
    Write the complaint to Pinecone
    :param state:
    :return:
    """
    openai = OpenAI()
    id = state['id']
    summary = state['summary']
    complaint = state['complaint']
    category = state['category']
    sub_category = state['sub_category']
    resolved = state['resolved']
    admin_text = state['admin_text']

    print(f"Writing complaint to Pinecone: {complaint} with category: {category} and sub-category: {sub_category}")
    model_name = "text-embedding-3-small"
    pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pinecone.Index("complaints")

    raw_embedding = openai.embeddings.create(
        input=[complaint],
        model=model_name
    )
    embedding = raw_embedding.data[0].embedding

    # Insert the complaint into the Pinecone index
    index.upsert(
        vectors=[
            {
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": {
                    "userID": id,
                    "summary": summary,
                    "product": category,
                    "subcategory": sub_category,
                    "text": complaint,
                    "resolved": resolved,
                    "admin_text": admin_text
                }
            },
        ],
        namespace="rag_complaints"
    )

    return {
        "id": id,
        "complaint": complaint,
        "category": category,
        "sub_category": sub_category,
        "resolved": resolved,
        "admin_text": admin_text
    }
