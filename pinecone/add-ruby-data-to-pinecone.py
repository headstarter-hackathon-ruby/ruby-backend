#https://docs.pinecone.io/integrations/openai
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import json
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

file_path='./ruby_hackathon_data.json'
index_name = "complaints"

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

data = json.loads(Path(file_path).read_text())

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=2,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    )

index = Pinecone.Index(index_name)

client = OpenAI(api_key=OPENAI_API_KEY)