from chains.OG_PerunaBot_chain import Original_PerunaBot_chain
from chains.chain_0 import base_retriever_chain_0
from chains.chain_1 import parent_retriever_chain_1
from chains.chain_2 import ensemble_retriever_chain_2
from recreate_chains import get_random_chains, get_all_chains
from langchain.schema import AIMessage, HumanMessage
from dotenv import load_dotenv, find_dotenv
import time
import gradio as gr 
import os


# Load environment variables from the .env file using 'from dotenv import find_dotenv, load_dotenv'
load_dotenv(find_dotenv(filename='SURF-Project_Optimizing-PerunaBot/Setup/.env'))
open_ai_api_key = os.environ['OPENAI_API_KEY']

def new_chat(chain):
    def chat_with_chain(user_input, chat_history):
        chat_history = []
        
        response = chain.invoke({
                    "chat_history": chat_history, 
                    "input": user_input})
        
        chat_history.append(AIMessage(content=response["answer"]))
        chat_history.append(HumanMessage(content=user_input))
        
        answer = response["answer"]
        stream = ''
        for character in answer:
            time.sleep(0.01)
            stream += character
            yield stream

OG_chain_convo = new_chat(Original_PerunaBot_chain)
PerunaBot_0_convo = new_chat(base_retriever_chain_0)
PerunaBot_1_convo = new_chat(parent_retriever_chain_1)
PerunaBot_2_convo = new_chat(ensemble_retriever_chain_2)

gr.ChatInterface(fn=OG_chain_convo).launch()
