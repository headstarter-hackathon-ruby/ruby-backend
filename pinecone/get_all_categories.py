import os
from dotenv import load_dotenv
from pinecone import Pinecone
import numpy as np
from collections import Counter

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "complaints"
NAME_SPACE = "rag_complaints"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def get_all_complaints():
    dummy_vector = np.zeros(1536).tolist()

    results = index.query(
        vector=dummy_vector,
        top_k=10000,  # Adjust based on your total number of vectors
        include_metadata=True,
        namespace=NAME_SPACE
    )
    complaints = []
    for match in results['matches']:
        complaints.append({
            'id': match['id'],
            'score': match['score'],
            'metadata': match['metadata']
        })

    return complaints

def get_all_categories(complaints):
    categories = [complaint['metadata'].get('product', 'Unknown') for complaint in complaints]
    category_counts = Counter(categories)
    return category_counts

if __name__ == "__main__":
    all_complaints = get_all_complaints()
    print(f"Total complaints retrieved: {len(all_complaints)}")

    all_categories = get_all_categories(all_complaints)
    print("\nAll Categories:")
    for category, count in all_categories.items():
        print(f"{category}: {count}")

    print("\nTop 5 complaints as example:")
    for complaint in all_complaints[:5]:  # Print first 5 as an example
        print(f"ID: {complaint['id']}")
        print(f"Category: {complaint['metadata'].get('product', 'N/A')}")
        print(f"Sub-category: {complaint['metadata'].get('sub_product', 'N/A')}")
        print("---")