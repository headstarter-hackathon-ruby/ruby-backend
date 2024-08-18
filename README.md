
# Ruby Dashboard's Backend

Developed with FastAPI and deployed to Render.

The source of all AI/RAG workflows. Finding similar complaints, making predictions, and any other server-side functionality is here.
 


## Frameworks/Libraries/APIs

- FastAPI for backend routing
- OpenAI for embedding, audio transcription (Whisper), image recognition
- Supabase for storing transactions and audio/img files
- Langgraph for agentic RAG
- SciKit Learn for statistical prediction

## RAG Pipeline

Sample data received from the Ruby Hackathon was first inserted to Pinecone with metadata of its text, category, and subcategory. If their was an existing, public company response, we marked this complaint as resolved.

### Submitting Complaints
    1. Complaints are first filtered by whether it is a complaint. If it is not detected as a complaint, the workflow will end (audio/image complaints are first transcribed). 
    2. Valid complaints are fed into a graph with three agents
        - Matcher: finds three similar complaints based on vector embedding of complaint text
        - Categorizer: categorizes complaint based on similar complaints
        - Writer: writes newly categorized complaint to Pinecone
    3. Once inserted to Pinecone, users will receive a text response offering condolences and advice.


### Suggesting solutions
    1. The vector embedding of the plain complaint text is found
    2. Finds the most similar resolved complaints, where there is an existing resolution already present
    3. Using these similar, but already resolved complaints, we augment the llm with new context to suggest an appropriate solution to the complaint.








## Authors

- [@Naman](https://github.com/namanNagelia)
- [@Maisha](https://github.com/maishaSupritee)
- [@Evan](https://github.com/Colexeco)
- [@Humzah](https://github.com/hdeejay)
- [@Harvey](https://github.com/Verdenroz)


