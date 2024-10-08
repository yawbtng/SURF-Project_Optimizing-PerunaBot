import os
from dotenv import find_dotenv, load_dotenv
from langsmith import Client
from langchain_qdrant import QdrantVectorStore as Qdrant
from qdrant_client import qdrant_client
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
import json
import shelve
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain import hub


# Load environment variables from the .env file using 'from dotenv import find_dotenv, load_dotenv'
load_dotenv(find_dotenv(filename='SURF-Project_Optimizing-PerunaBot/setup/.env'))

# Initialize API keys and environment variables using 'import os'
# Qdrant vector db
qdrant_host = os.environ['QDRANT_HOST']
qdrant_api_key = os.environ['QDRANT_API_KEY']
qdrant_collection_0 = os.environ['QDRANT_COLLECTION_0']

# OpenAI API
openai_api_key = os.environ['OPENAI_API_KEY']

# langsmith
langsmith_api_key = os.environ["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"]
langchain_endpoint = os.environ["LANGCHAIN_ENDPOINT"]
langsmith_project = os.environ["LANGCHAIN_PROJECT"]

# Initialize LangSmith Client using 'from langsmith import Client'
langsmith_client = Client()

# Define a function to get vector store using 'from langchain_qdrant import Qdrant' and 'from qdrant_client import qdrant_client'
def get_vectorstore(qdrant_collection_name):
    # Ensure Qdrant Client connection and return vector store
    client = qdrant_client.QdrantClient(
        url=qdrant_host, 
        api_key=qdrant_api_key,
    )

    vector_store = Qdrant(
        client=client, 
        collection_name=qdrant_collection_name, 
        embedding=OpenAIEmbeddings(), # Uses 'from langchain_openai import OpenAIEmbeddings'
    )
    return vector_store

# Initialize vector store for collection 0 using 'get_vectorstore' function
vector_store_0 = get_vectorstore(qdrant_collection_0)
vector_store_0_retriever = vector_store_0.as_retriever()


# Load the LangChain documents from the shelve file
with shelve.open("../Common/serialized_data/data_preprocessing_langchain_docs.db") as db:
    langchain_docs_loaded = {key: db[key] for key in db}

csv_docs = langchain_docs_loaded['csv_docs']
normal_split_docs = langchain_docs_loaded["normal_split_docs"]


# Load the prompts from the JSON file
with open("prompts.json", "r") as json_file:
    prompts = json.load(json_file)

# Access the prompts as Python objects
condense_question_system_template = prompts["condense_question_system_template"]

template = """Use the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {input}
"""


# Create prompt templates using 'from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder'
condense_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_question_system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)


# Configure language model using 'from langchain_openai import ChatOpenAI'
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.25, max_tokens=750, timeout=None, max_retries=2)

#____________________________________________________________________________

# Define a function to create a chain based on each retriever using 'from langchain.chains import create_history_aware_retriever, create_retrieval_chain' 
# and 'from langchain.chains.combine_documents import create_stuff_documents_chain'
def create_convo_chain(vector_store_retriever):
   
   #  Create a chain based on retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, vector_store_retriever, condense_question_prompt
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return convo_qa_chain

# Create chain for collection 0 
Original_PerunaBot_chain = create_convo_chain(vector_store_0_retriever)
Original_PerunaBot_chain = Original_PerunaBot_chain.with_config({"run_name": "OG PerunaBot"})
Original_PerunaBot_chain = Original_PerunaBot_chain.with_config({
    "tags": ["OG PerunaBot"],
    "metadata": {
        "retriever": "base retriever (aka vector store as retriever)",
        "collection": "smu_data-0",
        "llm": "gpt-3.5-turbo"
    }
})


import uuid
def chat_convo():
    # session id for the convo
    config = {"metadata": {"session_id": str(uuid.uuid4())}}    

    # Define a function to process chat input and return response using 'from langchain_core.messages import HumanMessage, AIMessage'
    def process_chat(chain, question, chat_history):
        # Process chat input and return response
        response = chain.invoke({
            "chat_history": chat_history,
            "input": question,
        }, config=config)
        return response["answer"]
    
    # Initialize chat history
    chat_history_0 = []

    # Start chat with PerunaBot 0
    print("You are talking with OG PerunaBot that uses vector store 0 and the base retriever")

    check_0 = True
    while check_0:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            check_0 = False
            chat_history_0.clear()
            print("OG PerunaBot: Goodbye! Have a great day!")
        else:
            response = process_chat(Original_PerunaBot_chain, user_input, chat_history_0)
            chat_history_0.append(HumanMessage(content=user_input)) # Uses 'from langchain_core.messages import HumanMessage'
            chat_history_0.append(AIMessage(content=response)) # Uses 'from langchain_core.messages import AIMessage'
            print("User: ", user_input)
            print("OG PerunaBot: ", response)

if __name__ == '__main__':
    chat_convo()
# ____________________________________________________________________________
# Chain without history for evaluation

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

base_prompt = hub.pull("rlm/rag-prompt")


generation_chain = base_prompt | llm | StrOutputParser()

Original_PerunaBot_eval_chain = (
    {"context": itemgetter("question") | vector_store_0_retriever,
     "question": itemgetter("question")} 
    | RunnablePassthrough.assign(output = generation_chain))

# Configure the chain
Original_PerunaBot_eval_chain = Original_PerunaBot_eval_chain.with_config({"run_name": "OG PerunaBot Eval"})
Original_PerunaBot_eval_chain = Original_PerunaBot_eval_chain.with_config({
    "tags": ["OG_PerunaBot_eval_chain"],
    "metadata": {
        "retriever": "base retriever (aka vector store as retriever)",
        "collection": "smu_data-0",
        "llm": "gpt-3.5-turbo"
    }
})
# Original_PerunaBot_eval_chain.invoke({"question": "What is a good place to study?"})
#________________________________________________________________

new_llm = ChatOpenAI(model="gpt-4o", temperature=0.25, max_tokens=750, timeout=None, max_retries=2)

generation_chain = base_prompt | new_llm | StrOutputParser()

Original_PerunaBot_eval_chain_v1 = (
    {"context": itemgetter("question") | vector_store_0_retriever,
     "question": itemgetter("question")} 
    | RunnablePassthrough.assign(output = generation_chain))

# Configure the chain
Original_PerunaBot_eval_chain_v1 = Original_PerunaBot_eval_chain_v1.with_config({"run_name": "OG PerunaBot Eval"})
Original_PerunaBot_eval_chain_v1 = Original_PerunaBot_eval_chain_v1.with_config({
    "tags": ["OG_PerunaBot_eval_chain_v1"],
    "metadata": {
        "retriever": "base retriever (aka vector store as retriever)",
        "collection": "smu_data-0",
        "llm": "gpt-4o"
    }
})