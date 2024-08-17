import os
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

from pinecone import Pinecone
from rag.utils.graph import invoke_graph

load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

app = FastAPI()

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "complaints"
NAME_SPACE = "rag_complaints"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Generate some example data

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "https://ruby-frontend-five.vercel.app/"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def generate_example_data(numOfEntries):
    data = []
    for i in range(numOfEntries):
        example_data = {
            "id": i + 1,
            "product_category": "Credit card",
            "sub_product_category": "General purpose credit card or charge card",
            "issue": "Sample issue",
            "sub_issue": "Sample sub issue",
            "complaint_what_happened": "Sample complaint",
            "date_sent": (datetime.now() + timedelta(days=i)).isoformat()
        }
        data.append(example_data)
    return data


data = generate_example_data(5)


@app.get("/")
def read_root():
    return data


@app.get("/items/{item_id}")
def read_item(item_id: int):
    for item in data:
        if item["id"] == item_id:
            return item
    return {"error": "Item not found"}


# Define the request body formats


class MessageFormat(BaseModel):
    summary: str
    complaint: bool
    category: str
    subcategory: str
    textResponse: str


class PromptFormat(BaseModel):
    prompt: str
    userID: str


@app.post("/textPrompt", description="This endpoint will post and use GPT to classify a text prompt")
async def text_prompt(request: PromptFormat):
    try:
        # Use the OpenAI API to get a completion
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system",
                 "content": "You are a helpful and friendly chat support agent. Your job is to assist users with their complaints and provide troubleshooting tips. Using the given prompt, determine if it is a complaint or not. If it is a complaint, classify it as its appropriate complaint and subcategory, alongside a summary. If it isn't a complaint, please tell the user in a text response. If it is a complaint, say sorry and you have documented and sent it to the support team in the text response with some common troubleshooting tips."},
                {"role": "user", "content": request.prompt},
            ],
            response_format=MessageFormat,
        )

        # Extract and return the content of the response
        print(completion)
        event = completion.choices[0].message.parsed
        if event.complaint:
            # Invoke the RAG model and insert to Pinecone
            data = {
                'complaint': request.prompt,
                'summary': event.summary,
                'id': request.userID,
                'category': event.category,
                'sub_category': event.subcategory,
                'resolved': False,
                'admin_text': ' ',
                'similar_complaints': []
            }
            await invoke_graph(data)
            return {"result": event}

        else:
            print("Not a complaint")

        return {"result": event}

    except Exception as e:
        return {"error": str(e)}


# uvicorn app:app --reload# New functions for complaint queries


def get_all_complaints():
    dummy_vector = np.zeros(1536).tolist()
    results = index.query(
        vector=dummy_vector,
        top_k=10000,
        include_metadata=True,
        namespace=NAME_SPACE
    )
    return [{'id': match['id'], 'metadata': match['metadata']} for match in results['matches']]


def get_all_categories(complaints):
    categories = [complaint['metadata'].get(
        'product', 'Unknown') for complaint in complaints]
    return dict(Counter(categories))


def count_resolved_unresolved(complaints):
    resolved = sum(
        1 for c in complaints if c['metadata'].get('resolved', False))
    return resolved, len(complaints) - resolved


async def get_similar_complaints(complaint: str, limit: int):
    raw_embedding = client.embeddings.create(
        input=[complaint],
        model="text-embedding-3-small"
    )
    embedding = raw_embedding.data[0].embedding

    top_matches = index.query(
        namespace=NAME_SPACE,
        vector=embedding,
        top_k=limit,
        include_values=True,
        include_metadata=True,
    )

    similar_complaints = [
        {
            'product': match['metadata']['product'],
            'subcategory': match['metadata'].get('subcategory', 'General-purpose credit card or charge card'),
            'text': match['metadata']['text'],
            'resolved': match['metadata']['resolved'],
            'admin_text': match['metadata']['admin_text'],
            'summary': match['metadata']['summary'],
            'userID': match['metadata']['userID'],

        }
        for match in top_matches['matches']
    ]
    return similar_complaints


# New endpoints


@app.get("/complaints/all", description="Returns all complaints")
async def read_all_complaints():
    """
    This function returns all complaints. It returns the metadata of all complaints.
    """
    complaints = get_all_complaints()
    # Return only first 5 for brevity
    return {"total_complaints": len(complaints), "complaints": complaints[:5]}


@app.get("/complaints/categories", description="Returns the categories of all complaints")
async def read_categories():
    """
    This function returns the categories of all complaints. It calculates the number of complaints in each category.
    """
    complaints = get_all_complaints()
    categories = get_all_categories(complaints)
    return {"categories": categories}


@app.get("/complaints/resolution_status", description="Returns the resolution status of all complaints")
async def read_resolution_status():
    """
    This function returns the resolution status of all complaints. It calculates the number of resolved and unresolved complaints,
    """
    complaints = get_all_complaints()
    resolved, unresolved = count_resolved_unresolved(complaints)
    return {
        "resolved": resolved,
        "unresolved": unresolved,
        "resolution_rate": resolved / len(complaints) if complaints else 0
    }


@app.get("/complaints/similar", description="Returns similar complaints")
async def get_similar_complaints_with_solution(complaint: str, limit: int = 3):
    """
    This function returns similar complaints to the given complaint with an optional limit of 3.
    """
    return await get_similar_complaints(complaint, limit)


@app.get("/complaints/open")
async def read_resolution_status():
    complaints = get_all_complaints()
    resolved, unresolved = count_resolved_unresolved(complaints)
    return {
        "unresolved": unresolved,
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
