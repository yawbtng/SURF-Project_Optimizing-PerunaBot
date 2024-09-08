from chains.OG_PerunaBot_chain import Original_PerunaBot_chain
from chains.chain_0 import base_retriever_chain_0
from chains.chain_1 import parent_retriever_chain_1
from chains.chain_2 import ensemble_retriever_chain_2
from langchain.schema import AIMessage, HumanMessage
import random
import time

def chat_with_OG_chain(user_input, chat_history):
    chat_history = []
    
    response = Original_PerunaBot_chain.invoke({
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

def chat_with_chain_0(user_input, chat_history):
    chat_history = []
    
    response = base_retriever_chain_0.invoke({
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

def chat_with_chain_1(user_input, chat_history):
    chat_history = []
    
    response = parent_retriever_chain_1.invoke({
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

def chat_with_chain_2(user_input, chat_history):
    chat_history = []
    
    response = ensemble_retriever_chain_2.invoke({
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

def chatting_with_chain_(chain, user_input, chat_history):
    
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


def get_all_chains():
    return [
        {
            "name": "Original PerunaBot (from Jan 2024) 🌟",
            "chain": Original_PerunaBot_chain,
            "chat_function": chat_with_OG_chain
        },
        {
            "name": "PerunaBot v0 🤖",
            "chain": base_retriever_chain_0,
            "chat_function": chat_with_chain_0
        },
        {
            "name": "PerunaBot v1 🚀",
            "chain": parent_retriever_chain_1,
            "chat_function": chat_with_chain_1
        },
        {
            "name": "PerunaBot v2 🔥",
            "chain": ensemble_retriever_chain_2,
            "chat_function": chat_with_chain_2
        }
    ]

def get_random_chains(number: int = 2):
    return random.sample(get_all_chains(), number)

if __name__ == "__main__":
    random_chains = get_random_chains()
    for chain in random_chains:
        print(chain["name"])

