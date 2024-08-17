import os
from dotenv import load_dotenv
from pinecone import Pinecone
import numpy as np

# Load environment variables
load_dotenv()

# Get the API key and print it (be careful with this in production!)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
print(f"API Key: {PINECONE_API_KEY}")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in the environment variables")

INDEX_NAME = "complaints"
NAME_SPACE = "rag_complaints"

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)  # Fixed: apiKey -> api_key
    index = pc.Index(INDEX_NAME)  # Fixed: use INDEX_NAME instead of "pinecone-index"
    print(f"Successfully connected to Pinecone index: {INDEX_NAME}")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    raise

def get_all_complaints():
    dummy_vector = np.zeros(1536).tolist()

    try:
        results = index.query(
            vector=dummy_vector,
            top_k=10000,  # can adjust based on our vector size
            include_metadata=True,
            namespace=NAME_SPACE
        )
        print(f"Query successful. Number of matches: {len(results['matches'])}")
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

    complaints = []
    for match in results['matches']:
        complaints.append({
            'id': match['id'],
            'score': match['score'],
            'metadata': match['metadata']
        })

    return complaints

if __name__ == "__main__":
    all_complaints = get_all_complaints()
    print(f"Total complaints retrieved: {len(all_complaints)}")
    for complaint in all_complaints[:5]:  # Print first 5 as an example
        print(f"ID: {complaint['id']}")
        print(f"Summary: {complaint['metadata'].get('summary', 'N/A')}")
        print(f"Category: {complaint['metadata'].get('product', 'N/A')}")
        print(f"Sub-category: {complaint['metadata'].get('sub_product', 'N/A')}")
        print("---")