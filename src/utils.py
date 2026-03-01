import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# Updated modern import
from langchain_tavily import TavilySearch

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./data/vector_db",
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def get_web_search_tool():
    # Modern class name
    return TavilySearch(max_results=3)