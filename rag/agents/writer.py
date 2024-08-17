from rag.memory.complaint_state import ComplaintState


async def write(state: ComplaintState):
    """
    Write the complaint to Pinecone
    :param state:
    :return:
    """
    complaint = state['complaint']
    category = state['category']
    sub_category = state['sub_category']

    # Write the complaint to Pinecone
    return {
        'time': 'Finished',
    }