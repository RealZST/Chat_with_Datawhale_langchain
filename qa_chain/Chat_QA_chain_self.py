from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb
import re

class Chat_QA_chain_self:
    """"
    QA chain with conversation history 
    """
    def __init__(self,model:str, temperature:float=0.0, top_k:int=4, chat_history:list=[], file_path:str=None, persist_path:str=None, appid:str=None, api_key:str=None, Spark_api_secret:str=None, Wenxin_secret_key:str=None, embedding = "m3e", embedding_key:str=None):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k  # Number of top similar documents retrieved
        self.chat_history = chat_history  # List storing conversation history
        self.file_path = file_path  # Path to the knowledge base files
        self.persist_path = persist_path  # Path to store the vector database
        self.appid = appid
        self.api_key = api_key
        self.Spark_api_secret = Spark_api_secret
        self.Wenxin_secret_key = Wenxin_secret_key
        self.embedding = embedding  # Selected embedding model
        self.embedding_key = embedding_key

        # Load/Create vector database
        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding,self.embedding_key)

        # LLM instance
        self.llm = model_to_llm(self.model, self.temperature, self.appid, self.api_key, self.Spark_api_secret, self.Wenxin_secret_key)

        # Create a retriever from the vector database
        self.retriever = self.vectordb.as_retriever(search_type="similarity",   
                                        search_kwargs={'k': self.top_k})  # Default: similarity search, k=4
        
    def answer(self, question:str=None, temperature=None, top_k=4):
        """"
        Call the QA chain to return the final answer
        
        Args:
            question: User's question
        
        Return: 
            All QA records
        """
        
        if len(question) == 0:
            return self.chat_history
        
        if temperature == None:
            temperature = self.temperature

        # The QA chain is not placed in __init__() because ConversationalRetrievalChain needs to dynamically manage chat_history
        # ConversationalRetrievalChain is not a static LLM but a dynamically constructed chain. 
        # It needs to be reconstructed each time with the latest chat_history.
        qa = ConversationalRetrievalChain.from_llm(
            llm = self.llm,
            retriever = self.retriever
        )
        
        result = qa({"question": question, "chat_history": self.chat_history})  
        
        # result contains question, chat_history, and answer
        answer = result['answer']
        answer = re.sub(r"\\n", '<br/>', answer)

        # Update conversation history
        self.chat_history.append((question, answer)) 

        # Return updated conversation history (including the current QA)
        return self.chat_history  


    def clear_history(self):
        """ Clear the conversation history """
        return self.chat_history.clear()

    def change_history_length(self, history_len:int=1):
        """
        Retain conversation history for a specified number of turns
        
        Args:
            history_len: Number of recent conversation turns to retain
            chat_history: Current conversation history
        
        Returns:
            The last `history_len` conversations
        """
        n = len(self.chat_history)
        return self.chat_history[n-history_len:]

