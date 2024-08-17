import asyncio
import json
import os
import uuid

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from rag.utils.graph import stream


def load_sample_data():
    load_dotenv()

    openai = OpenAI()
    pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    pinecone_index = "complaints"
    namespace = "rag_complaints"
    index = pinecone.Index(pinecone_index)

    model_name = "text-embedding-3-small"

    data = json.loads(open("ruby_hackathon_data.json").read())

    for complaint in data:
        # Inserts the complaint into the Pinecone index with the complaint text as the vector
        id = complaint['_id']
        complaint_text = complaint['_source']['complaint_what_happened']
        metadata = {
            'product': complaint['_source']['product'],
            'sub_product': complaint['_source']['sub_product'],
            'response': complaint['_source']['company_public_response']
        }
        # Create the embedding for the complaint text
        raw_embedding = openai.embeddings.create(
            input=[complaint_text],
            model=model_name
        )
        embedding = raw_embedding.data[0].embedding

        response = metadata['response']
        resolved = True if response else False
        admin_text = response if resolved else ' '
        # Insert the complaint into the Pinecone index
        index.upsert(
            vectors=[
                {
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": {
                        "userID": id,
                        "text": complaint_text,
                        "summary": complaint_text,
                        "product": metadata['product'],
                        "subcategory": metadata['sub_product'],
                        "resolved": resolved,
                        "admin_text": admin_text,
                    }
                },
            ],
            namespace=namespace
        )


async def main():
    """
    Test function to run the RAG model
    Will NOT work if while Pinecone is not setup
    :return:
    """
    sample_complaint = "I am not happy with the product as I have bought it with my credit card and found defects. I want a refund"
    inputs = {
        'complaint': sample_complaint,
        'id': '1',
        'time': 'start',
        'product_category': '',
        'sub_category': '',
        'similar_complaints': []
    }

    complaint = await stream(inputs)
    print(complaint)

if __name__ == "__main__":
    asyncio.run(main())
