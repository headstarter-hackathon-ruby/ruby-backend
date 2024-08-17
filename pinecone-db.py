from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(PINECONE_API_KEY)

if "pinecone-db" not in pc.list_indexes().names():
    pc.create_index(
        name="pinecone-db",
        dimension=2,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-2'
        ) 
    ) 