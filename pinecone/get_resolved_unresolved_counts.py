import os
from dotenv import load_dotenv
from pinecone import Pinecone
import numpy as np

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

def count_resolved_unresolved(complaints):
    resolved = 0
    unresolved = 0
    
    for complaint in complaints:
        if complaint['metadata'].get('resolved', False):
            resolved += 1
        else:
            unresolved += 1
    
    return resolved, unresolved

if __name__ == "__main__":
    all_complaints = get_all_complaints()
    print(f"Total complaints retrieved: {len(all_complaints)}")

    resolved, unresolved = count_resolved_unresolved(all_complaints)
    print(f"\nResolved complaints: {resolved}")
    print(f"Unresolved complaints: {unresolved}")

    print("\nResolution rate: {:.2f}%".format(resolved / len(all_complaints) * 100))

    print("\nExample complaints:")
    for complaint in all_complaints[:5]:  # Print first 5 as an example
        print(f"ID: {complaint['id']}")
        print(f"Resolved: {complaint['metadata'].get('resolved', 'Unknown')}")
        print(f"Category: {complaint['metadata'].get('product', 'N/A')}")
        print("---")