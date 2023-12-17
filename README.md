#Chat Langchain with OpenAI GPT 3.5 16k, Chroma Documentation Embeddings, and Streamlit Frontend
This repo takes the official chat langchain app [https://github.com/langchain-ai/chat-langchain.git] and ports the vector database backend to chroma instead of weaviate. Also recreates the frontend with streamlit.

##Running locally
1. Add your openai api to the env.sh file and source the enviroment variables in bash. source ./env.sh
2. Run python ingest.py to embed the documentation from the langchain documentation website, the api documentation website, and the langsmith documentation website.
3. Run python chain.py to chat the docs
4. Streamlit frontend coming soon

##Work in Progess
Currently the project will work by adding your openai api to the environment variables env.sh, running ingest.py to create the vector database of the documentation, and using python chain.py to chat in the command line. Streamlit frontend coming soon!
