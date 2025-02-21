import sys 
import os
from database.create_db import create_db, load_knowledge_db
from embedding.call_embedding import get_embedding

def get_vectordb(file_path:str=None, persist_path:str=None, embedding = "m3e", embedding_key:str=None):
    """
    用于创建或加载一个向量数据库，并返回 一个可以用于检索的向量数据库对象。
    
    Args:
        file_path: 知识库文件的路径(仅在新建向量数据库时需要)
        persist_path: 向量数据库的持久化存储路径
        embedding: 选择的 embedding 模型
        embedding_key: API Key，用于访问 embedding 模型

    Return:
        vectordb: 向量数据库
    """
    embedding = get_embedding(embedding=embedding, embedding_key=embedding_key)  # Get embedding model
    
    if os.path.exists(persist_path):  #持久化目录存在
        contents = os.listdir(persist_path)
        if len(contents) == 0:  #但是下面为空
            #print("目录为空")
            vectordb = create_db(file_path, persist_path, embedding, merge=False)
            vectordb = load_knowledge_db(persist_path, embedding)
        else:
            #print("目录不为空")
            vectordb = load_knowledge_db(persist_path, embedding)
    else: #目录不存在，从头开始创建向量数据库
        vectordb = create_db(file_path, persist_path, embedding, merge=False)
        vectordb = load_knowledge_db(persist_path, embedding)

    return vectordb