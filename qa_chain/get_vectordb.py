import sys 
import os
from database.create_db import create_db, load_knowledge_db
from embedding.call_embedding import get_embedding

def get_vectordb(file_path:str=None, persist_path:str=None, embedding = "m3e", embedding_key:str=None):
    """
    Creates or loads a vector database and returns a vector database object that can be used for retrieval.
    
    Args:
        file_path: Path to the knowledge base files (only needed when creating a new vector database)
        persist_path: Path for persistent storage of the vector database
        embedding: Selected embedding model
        embedding_key: API Key used to access the embedding model

    Return:
        vectordb: Vector database
    """
    embedding = get_embedding(embedding=embedding, embedding_key=embedding_key)  # Get embedding model
    
    if os.path.exists(persist_path):  # Check if the persistence directory exists
        contents = os.listdir(persist_path)
        if len(contents) == 0:  # If the directory exists but is empty
            vectordb = create_db(file_path, persist_path, embedding, merge=False)
            vectordb = load_knowledge_db(persist_path, embedding)
        else:
            vectordb = load_knowledge_db(persist_path, embedding)
    else:  # If the directory does not exist, create a new vector database from scratch
        vectordb = create_db(file_path, persist_path, embedding, merge=False)
        vectordb = load_knowledge_db(persist_path, embedding)

    return vectordb