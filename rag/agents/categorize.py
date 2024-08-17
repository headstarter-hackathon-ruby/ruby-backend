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
    similar_complaints = state['similar_complaints']

    text_contexts = [complaint['text'] for complaint in similar_complaints]
    product_category_contexts = [complaint['product'] for complaint in similar_complaints]
    sub_category_contexts = [complaint['sub_product'] for complaint in similar_complaints]

    # Combine product_category and sub_category into a single context string
    contexts = [f"Text: {text} Product: {product}, Sub-Product: {sub_product}" for text, product, sub_product in
                zip(text_contexts, product_category_contexts, sub_category_contexts)]

    query = f"Given the complain: {complaint} \n" \
            f"You have two tasks. The first is to identify the product category. The product must be isolated and can be identified on the most common category based on context\n" \
            f"The second is to identify the subcategory\n" \
            f"Please provide the product category and subcategory based on the context\n" \
            f"You must return nothing but a JSON with the field 'category' (str) and 'subcategory' (str)\n"

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
    response = invoke_model(prompt, 'gpt-3.5-turbo', response_format='json')
    response_json = json.loads(response)

    return {
        "category": response_json.get("category"),
        "sub_category": response_json.get("subcategory")
    }
