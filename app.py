import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag.utils.graph import invoke_graph

load_dotenv()
print(os.environ.get("OPENAI_API_KEY"))
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

app = FastAPI()


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
                 "content": "Using the given prompt, determine if it is a complaint or not. If it is a complaint, classify it as its appropriate complaint and subcategory, alongside a summary. If it isnt a complaint, please tell the user in text response. If it is a complaint, say sorry and you have documented and sent it to the support team in the text response with some common trouble shooting tips"},
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

            return {"result": event.textResponse}

        else:
            print("Not a complaint")
        return {"result": event}

    except Exception as e:
        return {"error": str(e)}

# uvicorn app:app --reload
# If running the script directly, start the FastAPI server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
