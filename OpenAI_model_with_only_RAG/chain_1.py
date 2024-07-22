import os
from dotenv import find_dotenv, load_dotenv
from langsmith import Client
from langchain_qdrant import Qdrant
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.parent_document_retriever import CustomParentDocumentRetriever
from langchain.storage import InMemoryStore

# Load environment variables from the .env file using 'from dotenv import find_dotenv, load_dotenv'
load_dotenv(find_dotenv(filename='SURF-Project_Optimizing-PerunaBot/setup/.env'))

# Initialize API keys and environment variables using 'import os'
# Qdrant vector db
qdrant_host = os.environ['QDRANT_HOST']
qdrant_api_key = os.environ['QDRANT_API_KEY']
qdrant_collection_1 = os.environ['QDRANT_COLLECTION_1']

# OpenAI API
openai_api_key = os.environ['OPENAI_API_KEY']

#langsmith
langsmith_api_key = os.environ["LANGSMITH_API_KEY"]
langchain_endpoint = os.environ["LANGCHAIN_ENDPOINT"]
langsmith_project = os.environ["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_TRACING_V2"]

# Initialize LangSmith Client using 'from langsmith import Client'
langsmith_client = Client()


# Load the LangChain documentation from the shelve file
with shelve.open("../Common/serialized_data/data_preprocessing_langchain_docs.db") as db:
    langchain_docs_loaded = {key: db[key] for key in db}

pdf_docs = langchain_docs_loaded['pdf_docs']
csv_docs = langchain_docs_loaded['csv_docs']


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
        embeddings=OpenAIEmbeddings(), # Uses 'from langchain_openai import OpenAIEmbeddings'
    )
    return vector_store

# Initialize vector store for collection 0 using 'get_vectorstore' function
vector_store_1 = get_vectorstore(qdrant_collection_1)

# Configure text splitters and in-memory storage using 'from langchain.text_splitter import RecursiveCharacterTextSplitter' and 'from langchain.storage import InMemoryStore'
child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25, length_function=len, add_start_index=True)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=50, length_function=len, add_start_index=True)
store_in_memory = InMemoryStore()

# Define a function to create a parent document retriever using 'from langchain.retrievers import ParentDocumentRetriever'
def create_parent_retriever():
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vector_store_1, 
        docstore=store_in_memory, 
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 8}
    )
    return parent_retriever

# Function to create a custom parent document retriever using 'from langchain.retrievers.parent_document_retriever import CustomParentDocumentRetriever'
def create_custom_parent_retriever():
    parent_retriever = CustomParentDocumentRetriever(
        vectorstore=vector_store_1, 
        docstore=store_in_memory, 
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 10}
    )
    return parent_retriever

# Create and configure custom parent document retriever
parent_retriever = create_custom_parent_retriever()
parent_retriever.add_documents(pdf_docs + csv_docs, add_to_vectorstore=False)

# Load the prompts from the JSON file
with open("prompts.json", "r") as json_file:
    prompts = json.load(json_file)

# Access the prompts as Python objects
condense_question_system_template = prompts["condense_question_system_template"]
chatbot_personality = prompts["chatbot_personality"]


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
        ("system", chatbot_personality),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

# Configure language model using 'from langchain_openai import ChatOpenAI'
llm = ChatOpenAI(model="gpt-4o", temperature=0.25, max_tokens=750, timeout=None, max_retries=2)

# Define a function to create a chain based on each retriever using 'from langchain.chains import create_history_aware_retriever, create_retrieval_chain' 
# and 'from langchain.chains.combine_documents import create_stuff_documents_chain'
def create_chain(vector_store_retriever):
    # Create a chain based on retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, vector_store_retriever, condense_question_prompt
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return convo_qa_chain

# Create chain for collection 1
parent_retriever_chain_1 = create_chain(parent_retriever)
parent_retriever_chain_1 = parent_retriever_chain_1.with_config({"run_name": "PerunaBot 1"})
parent_retriever_chain_1 = parent_retriever_chain_1.with_config({"tags": ["chain_1"], 
                                                                "metadata": {"retriever": "parent retriever", 
                                                                             "collection": "smu_data-1", 
                                                                             "llm": "gpt-4o"}})
# Define a function to process chat input and return response using 'from langchain_core.messages import HumanMessage, AIMessage'
def process_chat(chain, question, chat_history):
    # Process chat input and return response
    response = chain.invoke({
        "chat_history": chat_history,
        "input": question,
    })
    return response["answer"]

if __name__ == '__main__':
    # Initialize chat history
    chat_history_1 = []

    # Start chat with PerunaBot 1
    print("You are talking with PerunaBot 1 that uses vector store 1 and the custom parent document retriever")

    check_1 = True
    while check_1:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            check_1 = False
            chat_history_1.clear()
        else:
            response = process_chat(parent_retriever_chain_1, user_input, chat_history_1)
            chat_history_1.append(HumanMessage(content=user_input)) # Uses 'from langchain_core.messages import HumanMessage'
            chat_history_1.append(AIMessage(content=response)) # Uses 'from langchain_core.messages import AIMessage'
            print("User: ", user_input)
            print("PerunaBot 1: ", response)

# ____________________________________________________________________________
# Chain with stateful chat message history

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import uuid

### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

parent_retriever_chain_1_rag = RunnableWithMessageHistory(
    parent_retriever_chain_1,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def run_chain_1(question):
    chat_history_0 = []
    response = parent_retriever_chain_1_rag.invoke(
        {"input": question},
        config = {"configurable": {"session_id": uuid.uuid4().hex}}
    )
    return response["answer"]

# ____________________________________________________________________________
# Chain without history for evaluation
from langchain_core.output_parsers import MarkdownListOutputParser
from langchain_core.runnables import RunnablePassthrough

generation_chain = qa_prompt | llm | MarkdownListOutputParser()
parent_retriever_eval_chain_1 = {
    "context": parent_retriever,
    "question": RunnablePassthrough(),
} | RunnablePassthrough.assign(output = generation_chain)

# Configure the chain
parent_retriever_eval_chain_1 = parent_retriever_eval_chain_1.with_config({"run_name": "PerunaBot 1 Eval"})
parent_retriever_eval_chain_1 = parent_retriever_eval_chain_1.with_config({"tags": ["chain_1"], 
                                                                "metadata": {"retriever": "parent retriever", 
                                                                             "collection": "smu_data-1", 
                                                                             "llm": "gpt-4o"}})
 # ____________________________________________________________________________