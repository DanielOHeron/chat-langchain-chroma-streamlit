# Chat Langchain with OpenAI GPT 3.5 16k, Chroma Documentation Embeddings, and Streamlit Frontend
This repo takes the official chat langchain app [https://github.com/langchain-ai/chat-langchain.git] and ports the vector database backend to chroma instead of weaviate. Also recreates the frontend with streamlit.

## Running locally
1. Add your openai api to the env.sh file and source the enviroment variables in bash. source ./env.sh
2. Run python ingest.py to embed the documentation from the langchain documentation website, the api documentation website, and the langsmith documentation website.
3. Run python chain.py to chat the docs via terminal
4. Run streamlit run streamlit_frontend.py for the streamlit front end

## Work in Progess
Streamlit frontend needs links added back to the source documentation for the retrieval part
