from typing import List
from scipy import stats
from datetime import date, datetime, timedelta
from fastapi import FastAPI, HTTPException
import os
from collections import Counter
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from pinecone import Pinecone
from rag.utils.graph import invoke_graph
from rag.utils.llm import invoke_model
import mimetypes
from sklearn.linear_model import LinearRegression
from datetime import date
from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing


load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
supabase_key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")


supabase: Client = create_client(supabase_url, supabase_key)

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
                   "https://ruby-frontend-five.vercel.app"],  # Allows all origins
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
        'subcategory', 'Unknown') for complaint in complaints]
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
 \
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
async def read_all_complaints(page: int = Query(1, ge=1), page_size: int = Query(50, ge=1)):
    """
    This function returns paginated complaints. It returns the metadata of all complaints.
    """
    complaints = get_all_complaints()
    total_complaints = len(complaints)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_complaints = complaints[start:end]

    if not paginated_complaints and page != 1:
        raise HTTPException(status_code=404, detail="Page not found")

    return {
        "total_complaints": total_complaints,
        "page": page,
        "page_size": page_size,
        "complaints": paginated_complaints
    }


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


@app.get("/complaints/admin_message", description="Updates the admin message of a complaint")
async def update_admin_note(note: str, id: str):
    """
    This function updates the admin note of a complaint.
    """
    index.update(
        id=id,
        set_metadata={"admin_text": note},
        namespace="rag_complaints",
    )

    return


@app.get("/complaints/resolution", description="Updates the resolution status of a complaint")
async def update_resolution(id: str):
    """
    This function updates the resolution status of a complaint.
    """
    complaint_data = index.fetch(ids=[id], namespace="rag_complaints")
    print(complaint_data)
    resolved = complaint_data['vectors'][id]['metadata'].get('resolved', False)
    index.update(
        id=id,
        set_metadata={"resolved": not resolved},
        namespace="rag_complaints",
    )

    return


@app.get("/complaints/current", description="Returns the current complaint")
async def get_current_complaints(id: str):
    """
    This function returns the current complaint's metadata by the id.
    """
    complaint = index.fetch(ids=[id], namespace="rag_complaints")
    print(complaint)
    return complaint['vectors'][id]['metadata']


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
    "audio/mpeg": ".mp3",  # MPEG audio
    "audio/mp4": ".m4a",  # MP4 audio (used by .m4a files)
    "audio/m4a": ".m4a",  # M4A audio (same as audio/mp4)
    "audio/mp3": ".mp3",  # MP3 audio (same as audio/mpeg)
    "audio/wav": ".wav",  # WAV audio
    "audio/mpga": ".mp3",  # MP3 (audio/mpga is another MIME type for MP3)
    "audio/webm": ".webm",  # WEBM audio format
    "audio/x-mp4": ".m4a",  # MP4 audio (non-standard MIME type for .m4a)
    "audio/x-m4a": ".m4a",  # M4A audio (non-standard MIME type for .m4a)
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
        modelReq = transcription.text
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system",
                 "content": "You are a helpful and friendly chat support agent. Your job is to assist users with their complaints and provide troubleshooting tips. Using the given prompt, determine if it is a complaint or not. If it is a complaint, classify it as its appropriate complaint and subcategory, alongside a summary. If it isn't a complaint, please tell the user in a text response. If it is a complaint, say sorry and you have documented and sent it to the support team in the text response with some common troubleshooting tips."},
                {"role": "user", "content": modelReq},
            ],
            response_format=MessageFormat,
        )

        # Extract and return the content of the response
        print(completion)
        event = completion.choices[0].message.parsed
        if event.complaint:
            # Invoke the RAG model and insert to Pinecone
            data = {
                'complaint': modelReq,
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

    except Exception as e:
        return {"error": str(e)}

# Manual mapping of MIME types to file extensions
# GPT only accepts these types
image_mime_extension_map = {
    "image/jpeg": ".jpg",       # JPEG image
    "image/jpg": ".jpg",        # JPG image
    "image/png": ".png",        # PNG image
    "image/webp": ".webp",      # WEBP image
}


class ImageTranscriptionReq(BaseModel):
    image: str
    userID: str


@app.post("/transcribe/image", description="Analyze an image and provide a description")
async def transcribe_image(request: ImageTranscriptionReq):
    '''
    This function analyzes an image and provides a description, then classifies it as a complaint or not.
    '''
    try:
        # Using the image URL directly without downloading
        image_url = request.image

        # Using OpenAI's vision model to analyze the image
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image and provide a detailed description. If it appears to be a complaint or issue related to a product or service, please summarize it."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        # Extract the description from the response
        description = response.choices[0].message.content

        # Process the description to determine if it's a complaint
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system",
                 "content": "You are a helpful and friendly chat support agent. Your job is to assist users with their complaints and provide troubleshooting tips. Using the given prompt, determine if it is a complaint or not. If it is a complaint, classify it as its appropriate complaint and subcategory, alongside a summary. If it isn't a complaint, please tell the user in a text response. If it is a complaint, say sorry and you have documented and sent it to the support team in the text response with some common troubleshooting tips."},
                {"role": "user", "content": description},
            ],
            response_format=MessageFormat,
        )

        # Extract and return the content of the response
        print(completion)
        event = completion.choices[0].message.parsed
        if event.complaint:
            # Invoke the RAG model and insert to Pinecone
            data = {
                'complaint': description,
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

    except Exception as e:
        return {"error": str(e)}


class TransactionCreate(BaseModel):
    user_id: str
    transaction_name: str
    amount: float
    date: date


@app.post("/add_transaction", description="Add a transaction")
async def add_transaction(transaction: TransactionCreate):
    """
    This function adds a transaction to the user's account.
    """
    try:
        data = {
            'user_id': transaction.user_id,
            'transaction_name': transaction.transaction_name,
            'amount': transaction.amount,
            'date': transaction.date.isoformat()  # Convert date to ISO 8601 string
        }
        result = supabase.table('Transactions').insert(data).execute()

        # Check if data is in the result
        if result.data:
            return {"message": "Transaction added successfully", "data": result.data[0]}
        else:
            # If no data is returned but no exception was raised, assume success
            return {"message": "Transaction added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_transactions", description="Get all transactions")
async def get_transactions(user_id: str):
    """
    This function returns all transactions for the user.
    /get_transactions?user_id=123
    """
    try:
        result = supabase.table('Transactions').select(
            '*').eq('user_id', user_id).execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class FinancialGoalCreate(BaseModel):
    user_id: str
    date: date
    current_balance: float
    goal_balance: float


@app.post("/add_financial_goal", description="Add a financial goal")
async def add_financial_goal(goal: FinancialGoalCreate):
    """
    This function adds a financial goal to the user's account.
    """
    try:
        data = {
            'user_id': goal.user_id,
            'date': goal.date.isoformat(),  # Convert date to ISO 8601 string
            'current_balance': goal.current_balance,
            'goal_balance': goal.goal_balance
        }
        result = supabase.table('financial Goals').upsert(data).execute()

        # Check if data is in the result
        if result.data:
            return {"message": "Financial goal added successfully", "data": result.data[0]}
        else:
            # If no data is returned but no exception was raised, assume success
            return {"message": "Financial goal added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_financial_goals", description="Get all financial goals")
async def get_financial_goals(user_id: str):
    """
    This function returns all financial goals for the user.
    /get_financial_goals?user_id=123
    """
    try:
        result = supabase.table('financial Goals').select(
            '*').eq('user_id', user_id).execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TimeMachinePrediction(BaseModel):
    date: date
    predicted_balance: float


app = FastAPI()

# Assuming you have supabase client initialized somewhere
# from your_supabase_module import supabase


class TimeMachinePrediction(BaseModel):
    date: date
    predicted_balance: float
    upper_balance: float
    lower_balance: float


@app.get("/get_prediction", description="Get adaptive financial prediction")
async def get_prediction(user_id: str, prediction_days: int = 30):
    """
    This function provides an adaptive financial prediction based on available data.
    /get_prediction?user_id=123&prediction_days=30
    """
    try:
        transactions = supabase.table('Transactions').select(
            '*').eq('user_id', user_id).execute()
        df = pd.DataFrame(transactions.data)

        if df.empty:
            return []

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        df['cumulative_balance'] = df['amount'].cumsum()

        last_date = df['date'].max()
        future_dates = [last_date + timedelta(days=i)
                        for i in range(1, prediction_days + 1)]

        base_prediction = get_adaptive_prediction(df, prediction_days)
        recurring_transactions = identify_recurring_transactions(df)
        adjusted_prediction = adjust_for_recurring(
            base_prediction, recurring_transactions, future_dates)
        final_prediction = incorporate_recent_trend(df, adjusted_prediction)
        final_prediction_with_volatility = add_volatility(df, final_prediction)
        upper_lower_bounds = calculate_bounds(
            df, final_prediction_with_volatility)

        # Prepare response
        predictions = [
            TimeMachinePrediction(
                date=date,
                predicted_balance=float(balance),
                upper_balance=float(upper),
                lower_balance=float(lower)
            )
            for date, balance, (upper, lower) in zip(future_dates, final_prediction_with_volatility, upper_lower_bounds)
        ]

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def calculate_bounds(df, prediction, confidence_level=0.95):
    if len(df) >= 7:
        # Calculate the standard deviation of daily changes
        daily_changes = df['amount'].diff().dropna()
        std_dev = daily_changes.std()

        # Calculate the margin of error
        margin = stats.norm.ppf((1 + confidence_level) / 2) * \
            std_dev * np.sqrt(np.arange(1, len(prediction) + 1))

        upper_bound = prediction + margin
        lower_bound = prediction - margin

        return list(zip(upper_bound, lower_bound))
    else:
        # For very limited data, use a simple percentage-based approach
        upper_bound = prediction * 1.1  # 10% above prediction
        lower_bound = prediction * 0.9  # 10% below prediction
        return list(zip(upper_bound, lower_bound))


def get_adaptive_prediction(df, prediction_days):
    data_points = len(df)

    if data_points >= 14:
        model = ExponentialSmoothing(
            df['cumulative_balance'], trend='add', seasonal='add', seasonal_periods=7)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(prediction_days)
    elif data_points >= 7:  # At least 1 week of data
        model = ExponentialSmoothing(
            df['cumulative_balance'], trend='add', seasonal=None)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(prediction_days)
    else:
        X = np.array(range(len(df))).reshape(-1, 1)
        y = df['cumulative_balance'].values
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.array(
            range(len(df), len(df) + prediction_days)).reshape(-1, 1)
        forecast = model.predict(future_X)

    return forecast


def identify_recurring_transactions(df):
    if len(df) < 2:
        return {}

    grouped = df.groupby('transaction_name')
    recurring = {}

    for name, group in grouped:
        if len(group) > 1:
            intervals = group['date'].diff().mean().days
            if 25 <= intervals <= 35:  # Monthly
                recurring[name] = {'frequency': 'monthly',
                                   'amount': group['amount'].mean()}
            elif 6 <= intervals <= 8:  # Weekly
                recurring[name] = {'frequency': 'weekly',
                                   'amount': group['amount'].mean()}

    return recurring


def adjust_for_recurring(base_prediction, recurring_transactions, future_dates):
    adjusted = base_prediction.copy()

    for i, date in enumerate(future_dates):
        for transaction, details in recurring_transactions.items():
            if details['frequency'] == 'monthly' and date.day == 1:
                adjusted[i] += details['amount']
            elif details['frequency'] == 'weekly' and date.weekday() == 0:
                adjusted[i] += details['amount']

    return adjusted


def incorporate_recent_trend(df, prediction):
    if len(df) >= 7:
        recent_trend = df['amount'].tail(7).mean()
        weights = np.linspace(1, 0, len(prediction))
        trend_adjustment = recent_trend * weights
        return prediction + trend_adjustment
    return prediction


def add_volatility(df, prediction):
    if len(df) >= 7:
        volatility = df['amount'].std()
        # Reduced volatility impact
        noise = np.random.normal(0, volatility * 0.5, len(prediction))
        return prediction + noise
    return prediction


# uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
