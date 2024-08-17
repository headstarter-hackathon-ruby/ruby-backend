import json

from rag.memory.complaint_state import ComplaintState
from rag.utils.llm import invoke_model


async def categorize(state: ComplaintState):
    """
    Categorize the complaint text to a product category and sub category
    :param state:
    :return:
    """
    complaint = state['complaint']

    prompt = [{
        "role": "system",
        "content": "You are a expert at identifying product categories of credit/cash and its subcategories"
    }, {
        "role": "user",
        "content": f"Given the complain: {complaint}\n"
                   f"You have two tasks. The first is to identify the product category as credit/cash\n"
                   f"The second is to identify the subcategory where it can look like General-purpose credit card or charge card\n"
                   f"You must return nothing but a JSON with the field 'category' (str) and 'subcategory' (str)\n"

    }]
    response = invoke_model(prompt, 'gpt-3.5-turbo', response_format='json')
    response_json = json.loads(response)

    return {
        "product_category": response_json.get("category"),
        "sub_category": response_json.get("subcategory")
    }
