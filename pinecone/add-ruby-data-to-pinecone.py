from pinecone import Pinecone, ServerlessSpec
import json
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
file_path='./ruby_hackathon_data.json'
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

data = json.loads(Path(file_path).read_text())

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

if "complaints" not in pc.list_indexes().names():
    pc.create_index(
        name="complaints",
        dimension=2,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    )