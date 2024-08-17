import asyncio

from rag.utils.graph import build, stream

async def main():
    """
    Test function to run the RAG model
    Will NOT work if while Pinecone is not setup
    :return:
    """
    sample_complaint = "I am not happy with the product as I have bought it with my credit card and found defects. I want a refund"
    graph = build()
    graph.compile()
    inputs = {
        'complaint': sample_complaint,
        'time': '2021-09-01',
        'product_category': '',
        'sub_category': '',
        'similar_complaints': []
    }

    complaint = await stream(inputs)
    print(complaint)

if __name__ == "__main__":
    asyncio.run(main())