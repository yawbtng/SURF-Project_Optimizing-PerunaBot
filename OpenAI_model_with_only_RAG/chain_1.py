import os
from dotenv import find_dotenv, load_dotenv
from langsmith import Client
from langchain_qdrant import Qdrant
from qdrant_client import qdrant_client
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
import json
from langchain_core.load import load
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
qdrant_host = os.environ['QDRANT_HOST']
qdrant_api_key = os.environ['QDRANT_API_KEY']
openai_api_key = os.environ['OPENAI_API_KEY']
qdrant_collection_1 = os.environ['QDRANT_COLLECTION_1']
langsmith_api_key = os.environ["LANGSMITH_API_KEY"]
langchain_endpoint = os.environ["LANGCHAIN_ENDPOINT"]
langsmith_project = os.environ["LANGCHAIN_PROJECT"]

# Initialize LangSmith Client using 'from langsmith import Client'
langsmith_client = Client()

# Load serialized data from JSON files using 'import json' and 'from langchain_core.load import load'
with open('C:/Users/yawbt/OneDrive/Documents/GitHub/SURF-Project_Optimizing-PerunaBot/Common/data_preprocessing_langchain_objects.json', 'r') as file:
    serialized_data = json.load(file)

# Revive the LangChain docs from the serialized data
revived_data = load(serialized_data)
pdf_docs = revived_data['pdf_docs']
csv_docs = revived_data['csv_docs']

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
store = InMemoryStore()

# Define a function to create a parent document retriever using 'from langchain.retrievers import ParentDocumentRetriever'
def create_parent_retriever():
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vector_store_1, 
        docstore=store, 
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 8}
    )
    return parent_retriever

# Function to create a custom parent document retriever using 'from langchain.retrievers.parent_document_retriever import CustomParentDocumentRetriever'
def create_custom_parent_retriever():
    parent_retriever = CustomParentDocumentRetriever(
        vectorstore=vector_store_1, 
        docstore=store, 
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 10}
    )
    return parent_retriever

# Create and configure custom parent document retriever
parent_retriever = create_custom_parent_retriever()
parent_retriever.add_documents(pdf_docs + csv_docs, add_to_vectorstore=False)

# Load serialized prompts from JSON file using 'import json'
with open('C:/Users/yawbt/OneDrive/Documents/GitHub/SURF-Project_Optimizing-PerunaBot/OpenAI_model_with_only_RAG/prompts.json', 'r') as file:
    serialized_prompts = json.load(file)

# Retrieve the prompts using 'from langchain_core.load import load'
retrieved_prompts = load(serialized_prompts)
condense_question_system_template = retrieved_prompts["condense_question_system_template"]
chatbot_personality = retrieved_prompts["chatbot_personality"]

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
llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=750, timeout=None, max_retries=2)

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
chain_1 = create_chain(parent_retriever)

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
            response = process_chat(chain_1, user_input, chat_history_1)
            chat_history_1.append(HumanMessage(content=user_input)) # Uses 'from langchain_core.messages import HumanMessage'
            chat_history_1.append(AIMessage(content=response)) # Uses 'from langchain_core.messages import AIMessage'
            print("User: ", user_input)
            print("PerunaBot 1: ", response)