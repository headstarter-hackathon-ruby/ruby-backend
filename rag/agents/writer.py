import os

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
    complaint = state['complaint']
    category = state['category']
    sub_category = state['sub_category']

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
                "id": id,
                "values": embedding,
                "metadata": {"text": complaint, "product": category,
                             "sub_product": sub_category}
            },
        ],
        namespace="rag_complaints"
    )

    return {
        "id": id,
        "complaint": complaint,
        "category": category,
        "sub_category": sub_category,
        "time": "2021-09-01"
    }
