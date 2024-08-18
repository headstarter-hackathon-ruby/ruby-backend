import os
from collections import Counter
from datetime import datetime, timedelta
import requests
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
import whisper
from pinecone import Pinecone
from rag.utils.graph import invoke_graph
from rag.utils.llm import invoke_model
import mimetypes


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
        'sub_product', 'Unknown') for complaint in complaints]
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
    # Skip the first one as it is the same as the input complaint
    return similar_complaints[1:]


async def get_solution(complaint: str, limit: int):
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
        filter={'resolved': True}
    )

    similar_solutions = [
        {
            'product': match['metadata']['product'],
            'subcategory': match['metadata'].get('subcategory', 'General-purpose credit card or charge card'),
            'text': match['metadata']['text'],
            'admin_text': match['metadata']['admin_text'],
            'summary': match['metadata']['summary'],

        }
        for match in top_matches['matches']
    ]
    text_contexts = [complaint['text'] for complaint in similar_solutions]
    product_category_contexts = [complaint['product']
                                 for complaint in similar_solutions]
    sub_category_contexts = [complaint['subcategory']
                             for complaint in similar_solutions]
    solutions_contexts = [complaint['admin_text']
                          for complaint in similar_solutions]
    summary_contexts = [complaint['summary']
                        for complaint in similar_solutions]

    # Combine all contexts into a single string
    contexts = [f"Text: {text} Product: {product}, Sub-Product: {sub_product}, Solution: {solution}, Summary: {summary}"
                for text, product, sub_product, solution, summary in
                zip(text_contexts, product_category_contexts, sub_category_contexts, solutions_contexts,
                    summary_contexts)]

    query = f"Given the complain: {complaint} \n" \
        f"You have one task: identify a plausible and potential solution using previous similar examples \n" \
        f"Please provide a solution based on the context while ensuring the response is human readable and\n" \
        f"understandable to the user. It should be short, sweet, and succint.\n" \

    # Augment the query with the context
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(
        contexts) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
    print(augmented_query)

    prompt = [{
        "role": "system",
        "content": "You are a expert at identifying product categories of credit/cash and its subcategories"
    }, {
        "role": "user",
        "content": augmented_query

    }]
    response = invoke_model(prompt, 'gpt-3.5-turbo')
    return {
        "solution": response
    }


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
async def get_similar_complaints_with_solution(complaint: str, limit: int = 4):
    """
    This function returns similar complaints to the given complaint with an optional limit of 4.
    """
    return await get_similar_complaints(complaint, limit)


@app.get("/complaints/open")
async def read_resolution_status():
    complaints = get_all_complaints()
    resolved, unresolved = count_resolved_unresolved(complaints)
    return {
        "unresolved": unresolved,
    }


@app.get("/complaints/solutions", description="Returns solutions given a complaint")
async def get_solutions(complaint: str, limit: int = 3):
    """
    This function returns similar complaints to the given complaint with an optional limit of 3.
    """
    return await get_solution(complaint, limit)


class TranscriptionReq(BaseModel):
    audio: str
    userID: str


# Manual mapping of MIME types to file extensions
mime_extension_map = {
    "audio/mpeg": ".mp3",       # MPEG audio
    "audio/mp4": ".m4a",        # MP4 audio (used by .m4a files)
    "audio/m4a": ".m4a",        # M4A audio (same as audio/mp4)
    "audio/mp3": ".mp3",        # MP3 audio (same as audio/mpeg)
    "audio/wav": ".wav",        # WAV audio
    "audio/mpga": ".mp3",       # MP3 (audio/mpga is another MIME type for MP3)
    "audio/webm": ".webm",      # WEBM audio format
    "audio/x-mp4": ".m4a",      # MP4 audio (non-standard MIME type for .m4a)
    "audio/x-m4a": ".m4a",      # M4A audio (non-standard MIME type for .m4a)
}


@app.post("/transcribe/audio", description="Transcribe an audio file to text")
async def transcribe(request: TranscriptionReq):
    '''
    This function transcribes an audio file to text
    '''
    try:
        # Download the audio file from the provided URL
        audio_response = requests.get(request.audio)
        if audio_response.status_code != 200:
            return {"error": "Failed to download audio file from the provided URL"}

        # Determine the audio file format using mimetypes
        content_type = audio_response.headers.get('Content-Type')

        ext = mimetypes.guess_extension(content_type)
        if not ext:
            ext = mime_extension_map.get(content_type)

        if not ext:
            return {"error": f"Unsupported file format: {content_type}"}

        file_path = f"/tmp/audio_file{ext}"
        audio_data = audio_response.content
        with open(file_path, "wb") as temp_audio_file:
            temp_audio_file.write(audio_data)

        # Use OpenAI Whisper to transcribe the downloaded audio file
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        print(transcription)
        print(type(transcription))

        # Return the transcription and userID
        return {
            "userID": request.userID,
            "transcription": transcription.text
        }

    except Exception as e:
        return {"error": str(e)}


# uvicorn app:app --reload

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
