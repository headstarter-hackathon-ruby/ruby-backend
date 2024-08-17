import os

from openai import OpenAI
from pinecone import Pinecone

from rag.memory.complaint_state import ComplaintState


async def match(state: ComplaintState):
    """
    Match the complaint text to a response
    :param state:
    :return:
    """
    openai = OpenAI()
    pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    pinecone_index = pinecone.Index("complaints")
    namespace = "complaints"

    complaint = state['complaint']

    raw_query_embedding = openai.embeddings.create(
        input=[complaint],
        model="text-embedding-3-small"
    )

    complaint_embedding = raw_query_embedding.data[0].embedding

    top_matches = pinecone_index.query(vector=complaint_embedding, top_k=10, include_metadata=True, namespace=namespace)

    similar_complaints = [item['metadata']['text'] for item in top_matches['matches']]

    # Query Pinecone for the most similar complaint with complaint_embedding

    return {"similar_compaints": similar_complaints}
