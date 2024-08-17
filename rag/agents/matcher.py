import os

from openai import OpenAI
from pinecone import Pinecone

from rag.memory.complaint_state import ComplaintState


async def match(state: ComplaintState):
    """
    Match the complaint text to a similar complaints with its vector embedding
    :param state:
    :return:
    """
    openai = OpenAI()
    pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pinecone.Index("complaints")
    namespace = "rag_complaints"
    model_name = "text-embedding-3-small"

    complaint = state['complaint']
    raw_embedding = openai.embeddings.create(
        input=[complaint],
        model=model_name
    )
    try:
        complaint_embedding = raw_embedding.data[0].embedding

        # Query Pinecone for the most similar complaint with complaint_embedding
        top_matches = index.query(
            namespace=namespace,
            vector=complaint_embedding,
            top_k=3,
            include_values=True,
            include_metadata=True,
        )

        similar_complaints = [
            {
                'product': match['metadata']['product'],
                'sub_product': match['metadata'].get('subcategory', 'General-purpose credit card or charge card'),
                'text': match['metadata']['text']
            }
            for match in top_matches['matches']
        ]

        print(f"Similar complaints: {similar_complaints}")
        return {
            "complaint_embedding": complaint_embedding,
            "similar_complaints": similar_complaints,
        }
    except Exception as e:
        print(f"Error: {e}")

