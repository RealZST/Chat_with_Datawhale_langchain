"""
Gradio-based interactive web UI for LLM applications, allowing users to:
1. Select LLM and Embedding models
2. Input questions and receive AI-generated responses
3. Upload custom knowledge base files
4. Perform knowledge-based Q&A (with or without chat history)
5. Adjust parameters such as temperature and top-K
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import IPython.display
import io
import gradio as gr  # Gradio for UI interaction
from dotenv import load_dotenv, find_dotenv  # dotenv loads .env file (API Key, etc.)
from llm.call_llm import get_completion
from database.create_db import create_db_info
from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self
from qa_chain.QA_chain_self import QA_chain_self
import re


# Load environment variables (e.g., API Key) from .env into os.environ
_ = load_dotenv(find_dotenv())

# Dictionary storing available LLMs 
LLM_MODEL_DICT = {
    "openai": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"],
    "wenxin": ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"],
    "xinhuo": ["Spark-1.5", "Spark-2.0"],
    "zhipuai": ["chatglm_pro", "chatglm_std", "chatglm_lite"]
}
LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()),[])

# Default model settings
INIT_LLM = "gpt-3.5-turbo"
EMBEDDING_MODEL_LIST = ['zhipuai', 'openai', 'm3e']
INIT_EMBEDDING_MODEL = "m3e"
DEFAULT_DB_PATH = "../knowledge_db"
DEFAULT_PERSIST_PATH = "../vector_db/chroma"

# File paths for images/logos
AIGC_AVATAR_PATH = "../figures/aigc_avatar.png"
DATAWHALE_AVATAR_PATH = "../figures/datawhale_avatar.png"
AIGC_LOGO_PATH = "../figures/aigc_logo.png"
DATAWHALE_LOGO_PATH = "../figures/datawhale_logo.png"


class Model_center():
    """
    Stores Q&A Chain objects.

    - chat_qa_chain_self: Stores Q&A chains with history, indexed by (model, embedding).
    - qa_chain_self: Stores Q&A chains without history, indexed by (model, embedding).
    """
    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self = {}

    def chat_qa_chain_self_answer(self, question: str, chat_history: list = [], model: str = "openai", embedding: str = "m3e", 
                                    temperature: float = 0.0, top_k: int = 4, history_len: int = 3, file_path: str = DEFAULT_DB_PATH, 
                                    persist_path: str = DEFAULT_PERSIST_PATH):
        """
        Calls the Q&A chain with history.
        """
        if question == None or len(question) < 1:
            return "", chat_history
        
        try:
            # Create a new chain if it does not exist
            if (model, embedding) not in self.chat_qa_chain_self:
                self.chat_qa_chain_self[(model, embedding)] = Chat_QA_chain_self(model=model, temperature=temperature,
                                                                                    top_k=top_k, chat_history=chat_history, file_path=file_path, persist_path=persist_path, embedding=embedding)
            chain = self.chat_qa_chain_self[(model, embedding)]
            return "", chain.answer(question=question, temperature=temperature, top_k=top_k)
        
        except Exception as e:
            return e, chat_history

    def qa_chain_self_answer(self, question: str, chat_history: list = [], model: str = "openai", embedding="m3e", 
                                temperature: float = 0.0, top_k: int = 4, file_path: str = DEFAULT_DB_PATH, 
                                persist_path: str = DEFAULT_PERSIST_PATH):
        """
        Calls the Q&A chain without history.
        """
        if question == None or len(question) < 1:
            return "", chat_history
        
        try:
            if (model, embedding) not in self.qa_chain_self:
                self.qa_chain_self[(model, embedding)] = QA_chain_self(model=model, temperature=temperature,
                                                                       top_k=top_k, file_path=file_path, persist_path=persist_path, embedding=embedding)
            chain = self.qa_chain_self[(model, embedding)]
            chat_history.append((question, chain.answer(question, temperature, top_k)))
            return "", chat_history  # Returning chat_history here because the frontend UI needs to display the previous conversation.
        
        except Exception as e:
            return e, chat_history

    def clear_history(self):
        if len(self.chat_qa_chain_self) > 0:
            for chain in self.chat_qa_chain_self.values():
                chain.clear_history()


def format_chat_prompt(message, chat_history):
    """
    Format the chat prompt.

    Args:
        message: Current user message.
        chat_history: List of chat history.

    Return:
        prompt: Formatted prompt.
    """
    # Initialize an empty string to store the formatted chat prompt.
    prompt = ""

    # Iterate through the chat history.
    for turn in chat_history:
        # Extract user and assistant messages from the chat history.
        user_message, bot_message = turn
        # Update the prompt by appending previous user and assistant messages.
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    
    # Append the current user message to the prompt and leave space for the assistant's response.
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    
    return prompt


def respond(message, chat_history, llm, history_len=3, temperature=0.1, max_tokens=2048):
    """
    Generate the assistant's response.

    Args:
        message: Current user input.
        chat_history: List of chat history.
        llm: The language model used to generate responses.
        history_len: The length of conversation history to keep.
        temperature: LLM temperature setting for response variation.
        max_tokens: Maximum number of tokens for response generation.

    Returns:
        "": An empty string indicating no additional text needs to be displayed in the UI.
        chat_history: Updated chat history.
    """
    if message == None or len(message) < 1:
            return "", chat_history
    
    try:
        # Limit the length of conversation history.
        chat_history = chat_history[-history_len:] if history_len > 0 else []
        # Structure the user message and chat history into a prompt.
        formatted_prompt = format_chat_prompt(message, chat_history)
        # Generate a response using the LLM.
        bot_message = get_completion(formatted_prompt, llm, temperature=temperature, max_tokens=max_tokens)
        # Replace "\n" in bot_message with "<br/>" for proper HTML display.
        bot_message = re.sub(r"\\n", '<br/>', bot_message)
        # Append the user message and assistant response to the chat history.
        chat_history.append((message, bot_message))
        # Return an empty string and the updated chat history 
        # (the empty string can be replaced with the actual response if needed in the UI).
        return "", chat_history
    
    except Exception as e:
        return e, chat_history

model_center = Model_center()

block = gr.Blocks()

with block as demo:
    with gr.Row(equal_height=True):           
        gr.Image(value=AIGC_LOGO_PATH, scale=1, min_width=10, show_label=False, show_download_button=False, container=False)
   
        with gr.Column(scale=2):
            gr.Markdown("""<h1><center>Hands-on LLM Application Development</center></h1>
                        <center>LLM-UNIVERSE</center>
                        """)
        
        gr.Image(value=DATAWHALE_LOGO_PATH, scale=1, min_width=10, show_label=False, show_download_button=False, container=False)

    with gr.Row():
        with gr.Column(scale=4):
            # Create a chatbot interface
            chatbot = gr.Chatbot(height=400, show_copy_button=True, show_share_button=True, avatar_images=(AIGC_AVATAR_PATH, DATAWHALE_AVATAR_PATH))
            # Create a user input box
            msg = gr.Textbox(label="Prompt/query")

            with gr.Row():
                # Create submission buttons
                db_with_his_btn = gr.Button("Chat db with history (RAG)")
                db_wo_his_btn = gr.Button("Chat db without history (RAG)")
                llm_btn = gr.Button("Chat with llm")
            
            with gr.Row():
                # Create a clear button to reset the chatbot component
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")

        with gr.Column(scale=1):
            # Allow users to upload files as the knowledge base
            file = gr.File(label='Select Knowledge Base Directory', file_count='directory',
                           file_types=['.txt', '.md', '.docx', '.pdf'])
            with gr.Row():
                init_db = gr.Button("Vectorize Knowledge Base")
            
            model_argument = gr.Accordion("Parameter Configuration", open=False)
            with model_argument:
                temperature = gr.Slider(0,
                                        1,
                                        value=0.01,
                                        step=0.01,
                                        label="llm temperature",
                                        interactive=True)

                top_k = gr.Slider(1,
                                  10,
                                  value=3,
                                  step=1,
                                  label="vector db search top k",
                                  interactive=True)

                history_len = gr.Slider(0,
                                        5,
                                        value=3,
                                        step=1,
                                        label="history length",
                                        interactive=True)

            model_select = gr.Accordion("Model Selection")
            with model_select:
                llm = gr.Dropdown(
                    LLM_MODEL_LIST,
                    label="large language model",
                    value=INIT_LLM,
                    interactive=True)

                embeddings = gr.Dropdown(EMBEDDING_MODEL_LIST,
                                         label="Embedding model",
                                         value=INIT_EMBEDDING_MODEL)

        # Set up the click event for the knowledge base initialization button. 
        # When clicked, it calls `create_db_info()`, passing in the uploaded file and the selected embedding model.
        init_db.click(create_db_info,
                      inputs=[file, embeddings], outputs=[msg])

        # Set up button click events. When clicked, it calls `chat_qa_chain_self_answer()`, 
        # passing in the user's query and chat history, then updates the UI components.
        db_with_his_btn.click(model_center.chat_qa_chain_self_answer, inputs=[
                              msg, chatbot,  llm, embeddings, temperature, top_k, history_len],
                              outputs=[msg, chatbot])
        
        # Set up button click events. When clicked, it calls `qa_chain_self_answer()`, 
        # passing in the user's query and chat history, then updates the UI components.
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot, llm, embeddings, temperature, top_k], outputs=[msg, chatbot])
        
        # Set up button click events. When clicked, it calls `respond()`, 
        # passing in the user's query and chat history, then updates the UI components.
        llm_btn.click(respond, inputs=[
                      msg, chatbot, llm, history_len, temperature], outputs=[msg, chatbot], show_progress="minimal")

        # Set up a text box submission event (triggered when pressing Enter). 
        # Functions the same as the `llm_btn` click event.
        msg.submit(respond, inputs=[
                   msg, chatbot,  llm, history_len, temperature], outputs=[msg, chatbot], show_progress="hidden")
        
        # Clicking the button will clear the stored chat history in the backend.
        clear.click(model_center.clear_history)
    
    gr.Markdown("""Reminder:<br>
    1. Please upload your knowledge base files before use; otherwise, the system will use the default knowledge base.<br>
    2. Initializing the database may take some time; please be patient.<br>
    3. If an error occurs, it will be displayed in the text input box.<br>
    """)


# Close all running Gradio instances to prevent port conflicts.
gr.close_all()

# Start the Gradio Web UI, available at http://127.0.0.1:7860/
demo.launch()  

