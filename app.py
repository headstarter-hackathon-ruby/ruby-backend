# main.py
from fastapi import FastAPI
import random
from datetime import datetime, timedelta

app = FastAPI()

def generate_example_data(numOfEntries):
    data = []
    for i in range(numOfEntries):
        example_data = {
            "id": i+1,
            "product_category": "Credit card",
            "sub_product_category": "General purpose credit cardor charge card",
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
    


# uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
