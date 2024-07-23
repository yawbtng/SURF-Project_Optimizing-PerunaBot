# python script of data preprocesssing step

# Set up to initialize API keys from .env file into the
import os
from dotenv import find_dotenv, load_dotenv

# Load environment variables from the .env files
load_dotenv(find_dotenv(filename='SURF-Project_Optimizing-PerunaBot/setup/.env'))

#________________________________________________________________________________________________________________________________

# Import the Client class from the langsmith package for tracing and tracking

from langsmith import Client

# Retrieve Langsmith API key and other related environment variables
langsmith_api_key = os.environ["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"]
langchain_endpoint = os.environ["LANGCHAIN_ENDPOINT"]
langsmith_project = os.environ["LANGCHAIN_PROJECT"]

# Initialize a Langsmith Client instance
langsmith_client = Client()

# Test section (commented out)
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI()
# llm.invoke("What can you do?")

#________________________________________________________________________________________________________________________________

# Import the PyPDFLoader from langchain_community.document_loaders for PDF document loading
from langchain_community.document_loaders import PyPDFLoader

# file paths of PDFs to be used
# file paths of PDFs to be used
pdf_paths = ['../Data/RAG Knowledge Base/20232024 Undergraduate Catalog91123.pdf',
             '../Data/RAG Knowledge Base/Official University Calendar 2023-2024.pdf',
             '../Data/RAG Knowledge Base/2023_PerunaPassport.pdf',
             '../Data/RAG Knowledge Base/SMU Student Handbook 23-24.pdf',
             '../Data/RAG Knowledge Base/SMUCampusGuideFactsMap.pdf',
             ]

# Function to load PDFs using LangChain's PyPDFLoader
def load_pdfs_with_langchain(pdf_paths):
    documents = []
    for path in pdf_paths:
        try:
            # Use LangChain's PyPDFLoader to load the PDF
            loader = PyPDFLoader(path)
            # Load and pase the PDF into document instances
            pdf_doc = loader.load()
            # Insert the parsed PDF documents into the documents list
            documents.extend(pdf_doc)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return documents

# Load PDF documents using the function
docs = load_pdfs_with_langchain(pdf_paths)

print(len(docs))
print(docs[0].page_content[0:100])
print(docs[0].metadata)

#________________________________________________________________________________________________________________________________

# Import the pandas library to work with the Excel file and convert it to a data frame
import pandas as pd

# Load the Excel file
excel_path = '../Data/RAG Knowledge Base/SMU FAQs.xlsx'
xlsx = pd.ExcelFile(excel_path)

# checking to see if loading the file worked
print(xlsx.sheet_names)

# Iterate through each sheet and save as a CSV file
csv_files = []
for sheet_name in xlsx.sheet_names:
    # Read the entire sheet to extract the metadata from cell A1
    sheet_df = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)
    
    # Get the link of the webpage to include as metadata
    metadata = sheet_df.iat[0, 0]
    
    # Read the sheet into a DataFrame starting from the second row
    df = pd.read_excel(xlsx, sheet_name=sheet_name, skiprows=1)
    
    # Save the DataFrame to a CSV file
    csv_path = f'../Data/RAG Knowledge Base/{sheet_name}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')
    csv_files.append((csv_path, metadata))

# Display the list of generated CSV files and their metadata
print(csv_files)

#________________________________________________________________________________________________________________________________

# Import CSVLoader from langchain_community.document_loaders to load CSV documents
from langchain_community.document_loaders import CSVLoader

# Create LangChain documents from CSV files with metadata
csv_documents = []

for csv_path, metadata in csv_files:
    loader = CSVLoader(file_path=csv_path, encoding='utf-8')
    csv_docs = loader.load()
    for csv_doc in csv_docs:
        csv_doc.metadata['source'] = metadata
    csv_documents.extend(csv_docs)

# Display the first document as an example
print(csv_documents[0])

#________________________________________________________________________________________________________________________________

# Import Qdrant client for vector database cloud store
from qdrant_client import qdrant_client
from qdrant_client.http import models

# Initialize Qdrant host URL and API key from environment variables
qdrant_host = os.environ['QDRANT_HOST']
qdrant_api_key = os.environ['QDRANT_API_KEY']

# Initialize Qdrant Client
client = qdrant_client.QdrantClient(
    url=qdrant_host, 
    api_key = qdrant_api_key,
)

#________________________________________________________________________________________________________________________________

# Import OpenAIEmbeddings and Qdrant from respective langchain modules

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant

# Retrieve OpenAI API key from environment variables
openai_api_key = os.environ['OPENAI_API_KEY']

# Function to create a vector store based on the collection name
def create_vectorstore(qdrant_collection_name):
    
    # Ensuring Qdrant Client connection
    client = qdrant_client.QdrantClient(
        url=qdrant_host, 
        api_key = qdrant_api_key,
    )

    vectors_config = models.VectorParams(
        size=1536, #for OpenAI
        distance=models.Distance.COSINE
   )
    # Create a Qdrant collection with the specified name and vectors configuration
    client.create_collection(
        collection_name = qdrant_collection_name,
        vectors_config=vectors_config,   
    )

    # Initialize the vector store with the created Qdrant collection
    vector_store = Qdrant(
        client=client, 
        collection_name=qdrant_collection_name, 
        embeddings=OpenAIEmbeddings(),
    )
  
    return vector_store

# Function to return the vector store if it already exists

def get_vectorstore(qdrant_collection_name):
    
    # Ensuring Qdrant Client connection
    client = qdrant_client.QdrantClient(
    url=qdrant_host, 
    api_key = qdrant_api_key,
    )

    vector_store = Qdrant(
        client=client, 
        collection_name=qdrant_collection_name, 
        embeddings=OpenAIEmbeddings(),
    )

    return vector_store

#________________________________________________________________________________________________________________________________

# create 1st collection of vectors
qdrant_collection_1 = os.environ['QDRANT_COLLECTION_1']

# Checking if the collection already exists
collection_check_1 = False

if client.collection_exists(qdrant_collection_1):
    vector_store_1 = get_vectorstore(qdrant_collection_1)
    collection_check_1 = True
    print(qdrant_collection_1 + " already exists")
else:
    vector_store_1 = create_vectorstore(qdrant_collection_1)
    print(qdrant_collection_1 + " was just created")

#________________________________________________________________________________________________________________________________

# Parent Document Retriever Method
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
# created a custom class in ParentDocumentRetriever that adds the documents to the docstore but not to the vectorstore


from langchain.storage import InMemoryStore

child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25, 
                                                length_function=len, add_start_index=True) 
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=50, 
                                                length_function=len, add_start_index=True)  

# storage for parent splitter
store = InMemoryStore()

# retriever
def create_parent_retriever():
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vector_store_1, 
        docstore=store, 
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs = {"k": 8}
        )
    return parent_retriever

parent_retriever = create_parent_retriever()

#________________________________________________________________________________________________________________________________

# Check the status of the collection and add documents to the vector store if necessary
if collection_check_1 == False:
    # if collection is just created and empty
    if client.get_collection(qdrant_collection_1).vectors_count == None:
    # Add documents to the Qdrant vector database and parent store
        parent_retriever.add_documents(docs)
        parent_retriever.add_documents(csv_documents)
        print("PDF docs and CSV docs added to doc store and vectorstore")

elif collection_check_1 == True:  
    # if collection was already there and empty
    if client.get_collection(qdrant_collection_1).vectors_count == None: 
        # Add documents to the Qdrant vector database and parent store
        parent_retriever.add_documents(docs)
        parent_retriever.add_documents(csv_documents)
        print("PDF docs and CSV docs added to doc store and vectorstore")

# testing the retriever
parent_retriever.invoke("What is SMU")

#________________________________________________________________________________________________________________________________

# semantic text splitting method
# Import SemanticChunker from langchain_experimental.text_splitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# Initialize the semantic text splitter with OpenAI embeddings
semantic_text_splitter = SemanticChunker(
    OpenAIEmbeddings(), 
    breakpoint_threshold_type="percentile")
# Split documents using the semantic text splitter
semantic_docs = semantic_text_splitter.split_documents(docs)

print(semantic_docs[0].page_content)
print(len(semantic_docs))

#________________________________________________________________________________________________________________________________

# Create another instance of a vector store with a new collection using the function created earlier
qdrant_collection_2 = os.environ['QDRANT_COLLECTION_2']

# Check if the second collection already exists
collection_check_2 = False

# creating the second vector store and retriever
if client.collection_exists(qdrant_collection_2):
    vector_store_2 = get_vectorstore(qdrant_collection_2)
    print(qdrant_collection_2 + " already exists")
    collection_check_2 = True
else:
    vector_store_2 = create_vectorstore(qdrant_collection_2)
    print(qdrant_collection_2 + " was just created")

#________________________________________________________________________________________________________________________________

# Function to create a retriever for the second vector store
def create_vector_store_2_retriever():
    vector_store_2_retriever = vector_store_2.as_retriever(search_type="similarity_score_threshold",
                                                            search_kwargs = {"k": 8, "score_threshold" : 0.75})
    return vector_store_2_retriever

vector_store_2_retriever = create_vector_store_2_retriever()

# Add documents to the second vector store if necessary
if collection_check_2 == False:
        vector_store_2_retriever.add_documents(semantic_docs) # adding the semantically split docs into the vector store if not there already
        vector_store_2_retriever.add_documents(csv_documents) # adding csv docs to vectorstore 
elif collection_check_2 == True:
    if client.get_collection(qdrant_collection_2).vectors_count == None:
      vector_store_2_retriever.add_documents(semantic_docs) # adding the semantically split docs into the vector store if not there already
      vector_store_2_retriever.add_documents(csv_documents) # adding csv docs to vectorstore

#________________________________________________________________________________________________________________________________

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Initialize BM25 retriever from combined semantic and CSV documents
bm25_retriever = BM25Retriever.from_documents(semantic_docs+csv_documents)

# Initialize the ensemble retriever with BM25 and vector store retrievers
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_store_2_retriever], 
    weights=[0.5, 0.5]
)

# Test the ensemble retriever
ensemble_retriever.invoke("How many credit hours should I take my first year?")

#________________________________________________________________________________________________________________________________

# Initialize a base text splitter for normal splitting of documents
base_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, 
                                                length_function=len, add_start_index=True)  

# Split documents using the base text splitter
normal_split_docs = base_text_splitter.split_documents(docs)

# Check and print the result of the normal splitting
print(normal_split_docs[0].page_content)
print(len(normal_split_docs))

#________________________________________________________________________________________________________________________________

# getting the collection name of the third vector store
qdrant_collection_0 = os.environ['QDRANT_COLLECTION_0']

# Check if the third collection already exists
collection_check_0 = False

# creating the third vector store and retriever
if client.collection_exists(qdrant_collection_0):
    vector_store_0 = get_vectorstore(qdrant_collection_0)
    collection_check_0 = True
    print(qdrant_collection_0 + " is already there")
else:
    vector_store_0 = create_vectorstore(qdrant_collection_0)
    print(qdrant_collection_0 + " was just created")

#________________________________________________________________________________________________________________________________

# Initialize the retriever for the third vector store
vector_store_0_retriever = vector_store_0.as_retriever(search_kwargs = {"k": 8, "score_threshold" : 0.75})

# Add documents to the third vector store if necessary
if collection_check_0 == False:
        vector_store_0_retriever.add_documents(normal_split_docs) # adding the semantically split docs into the vector store if not there already
        vector_store_0_retriever.add_documents(csv_documents) # adding csv docs to vectorstore 
elif collection_check_0 == True:
    if client.get_collection(qdrant_collection_0).vectors_count == None:
      vector_store_2_retriever.add_documents(normal_split_docs) # adding the semantically split docs into the vector store if not there already
      vector_store_2_retriever.add_documents(csv_documents) # adding csv docs to vectorstore

# Test the third vector store retriever
vector_store_0_retriever.invoke("Can I do study abroad?")

#________________________________________________________________________________________________________________________________

# Function to get all LangChain documents
def get_all_langchain_docs():
  return {
    "pdf_docs": docs,
    "csv_docs": csv_documents
  }

# Function to get all vector stores
def get_all_vectorstores():
  return {
      "vector_store_0": vector_store_0, # collection smu-data_0
      "vector_store_1": vector_store_1, # collection smu-data_1
      "vector_store_2": vector_store_2, # collection smu-data_2
  }

# Function to get all retrievers
def get_all_retrievers():
  return {
      "vector_store_0_retriever": vector_store_0_retriever, # collection smu-data_0
      "parent_retriever": parent_retriever, # collection smu-data_1
      "ensemble_retriever": ensemble_retriever, # collection smu-data_2
  }


#________________________________________________________________________________________________________________________________

# Collecting needed langchain objects into a dictionary
all_data = {
    'pdf_docs': docs,
    'csv_docs': csv_documents,
    'semantic_docs': semantic_docs,
    'normal_split_docs': normal_split_docs,
}

import shelve

# Serialize the LangChain documentation to a JSON file
with shelve.open("data_preprocessing_langchain_docs.db") as db:
    for key, value in all_data.items():
        db[key] = value