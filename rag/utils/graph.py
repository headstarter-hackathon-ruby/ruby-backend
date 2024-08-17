from dotenv import load_dotenv
from langgraph.graph import StateGraph

from rag.agents.categorize import categorize
from rag.agents.matcher import match
from rag.agents.writer import write
from rag.memory.complaint_state import ComplaintState

load_dotenv()


def build():
    """
    Build the RAG model, which is a state graph that needs to be compiled
    """
    workflow = StateGraph(ComplaintState)
    workflow.add_node('Match', match)
    workflow.add_node('Categorize', categorize)
    workflow.add_node('Write', write)

    workflow.add_edge('Match', 'Categorize')
    workflow.add_edge('Categorize', 'Write')

    workflow.set_entry_point('Match')
    workflow.set_finish_point('Write')

    return workflow


async def stream(inputs: dict):
    """
    Stream the inputs through the RAG model
    :param inputs:
    :return:
    """
    workflow = build()
    graph = workflow.compile()

    async for event, chunk in graph.astream(inputs, stream_mode=["updates", "debug"]):
        print(f"Receiving new event of type: {event}...")
        print(chunk)
        print("\n\n")


async def invoke_graph(inputs: dict):
    """
    Run the RAG model
    :param inputs:
    :return:
    """
    workflow = build()
    graph = workflow.compile()

    result = await graph.ainvoke(inputs)

    return {
        "id": result['id'],
        "summary": result['summary'],
        "complaint": result['complaint'],
        "category": result['category'],
        "sub_category": result['sub_category'],
        "resolved": result['resolved'],
        "admin_text": result['admin_text'],
        "similar_complaints": result['similar_complaints']
    }
