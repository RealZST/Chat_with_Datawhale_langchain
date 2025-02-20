'''
Used for index construction (Document loading → Splitting → Vectorization → Storing in database)
'''

import os
import sys
import re
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tempfile
from dotenv import load_dotenv, find_dotenv
from embedding.call_embedding import get_embedding
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma


DEFAULT_DB_PATH = "../knowledge_db"
DEFAULT_PERSIST_PATH = "../vector_db/chroma"


# def get_files(dir_path):
#     file_list = []
#     for filepath, dirnames, filenames in os.walk(dir_path):
#         for filename in filenames:
#             file_list.append(os.path.join(filepath, filename))
#     return file_list


def file_loader(file, loaders):
    '''
    Document loading
    '''
    if isinstance(file, tempfile._TemporaryFileWrapper):
        file = file.name
    if not os.path.isfile(file):
        [file_loader(os.path.join(file, f), loaders) for f in  os.listdir(file)]
        return
    file_type = file.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file))
    elif file_type == 'md':
        pattern = r"不存在|风控"
        match = re.search(pattern, file)
        if not match:
            loaders.append(UnstructuredMarkdownLoader(file))
    elif file_type == 'txt':
        loaders.append(UnstructuredFileLoader(file))
    return


def create_db_info(files=DEFAULT_DB_PATH, embeddings="m3e", persist_directory=DEFAULT_PERSIST_PATH):
    if embeddings == 'openai' or embeddings == 'm3e' or embeddings =='zhipuai':
        vectordb = create_db(files, persist_directory, embeddings)
    return ""


def create_db(files=DEFAULT_DB_PATH, persist_directory=DEFAULT_PERSIST_PATH, embeddings="m3e"):
    """
    This function loads files, splits documents, generates document embeddings, 
    and creates a vector database.
   
    Args:
        files: Path to the stored files.
        embeddings: Embedding model used for vectorization.

    Return:
        vectordb: The created vector database.
    """
    if files == None:
        return "can't load empty file"
    if type(files) != list:
        files = [files]
    
    # Load documents
    loaders = []
    [file_loader(file, loaders) for file in files]
    docs = []
    for loader in loaders:
        if loader is not None:
            docs.extend(loader.load())
    
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)
    
    # Select the embedding model
    if type(embeddings) == str:
        embeddings = get_embedding(embedding=embeddings)
    
    # Create a vector database and store vectors
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    ) 

    # Persist the vector database to ensure it remains available after a restart 
    # without needing to be rebuilt, suitable for ChromaDB.
    vectordb.persist()
    
    return vectordb

def load_knowledge_db(path, embeddings):
    """
    This function loads an existing vector database.

    Args:
        path: Path to the vector database.
        embeddings: The embedding model used by the vector database.

    Return:
        vectordb: The loaded vector database.
    """
    # Load an existing vector database
    vectordb = Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )
    return vectordb


if __name__ == "__main__":
    create_db(embeddings="m3e")
