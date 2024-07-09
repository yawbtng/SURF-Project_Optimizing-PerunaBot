# python script of data preprocesssing step

# Set up to initialize API keys from .env file into the
import os
from dotenv import find_dotenv, load_dotenv

# Load environment variables from the .env files
load_dotenv(find_dotenv(filename='SURF-Project_Optimizing-PerunaBot/setup/.env'))

# --------------------------------------------------------------------------------------------------------------------------------

# Here we will initialize langmsith for tracing and tracking

from langsmith import Client
langsmith_api_key = os.environ["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"]
langchain_endpoint = os.environ["LANGCHAIN_ENDPOINT"]
langsmith_project = os.environ["LANGCHAIN_PROJECT"]

langmsiht_client = Client()

# test
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
llm.invoke("What can you do?")

# --------------------------------------------------------------------------------------------------------------------------------

# langchain imports
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

# file paths to the two PDFs we're using
pdf_paths = ['C:/Users/yawbt/OneDrive/Documents/GitHub/SURF-Project_Optimizing-PerunaBot/Data/20232024 Undergraduate Catalog91123.pdf',
             'C:/Users/yawbt/OneDrive/Documents/GitHub/SURF-Project_Optimizing-PerunaBot/Data/Official University Calendar 2023-2024.pdf',
             'C:/Users/yawbt/OneDrive/Documents/GitHub/SURF-Project_Optimizing-PerunaBot/Data/2023_PerunaPassport.pdf',
             'C:/Users/yawbt/OneDrive/Documents/GitHub/SURF-Project_Optimizing-PerunaBot/Data/SMU Student Handbook 23-24.pdf',
             'C:/Users/yawbt/OneDrive/Documents/GitHub/SURF-Project_Optimizing-PerunaBot/Data/SMUCampusGuideFactsMap.pdf'
             ]

def load_pdfs_with_langchain(pdf_paths):
    documents = []
    for path in pdf_paths:
        try:
            # Use LangChain's PyPDFLoader to load the PDF
            loader = PyPDFLoader(path)
            # Load and pase the PDF into document instances
            pdf_doc = loader.load()
            # Insert pdf into documents list variable
            documents.extend(pdf_doc)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return documents

#Load PDF documents using the function
docs = load_pdfs_with_langchain(pdf_paths)

print(len(docs))
print(docs[0].page_content[0:100])
print(docs[0].metadata)

# --------------------------------------------------------------------------------------------------------------------------------

# importing qdrant
from qdrant_client import qdrant_client
from qdrant_client.http import models

# Initializing Qdrant host URL and API key
qdrant_host = os.environ['QDRANT_HOST']
qdrant_api_key = os.environ['QDRANT_API_KEY']

#Initialize Qdrant Client
client = qdrant_client.QdrantClient(
    url=qdrant_host, 
    api_key = qdrant_api_key,
)

# --------------------------------------------------------------------------------------------------------------------------------

# function to create a vector store based on the collection name
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant

# Initializing OpenAI API key for embeddings and later use
openai_api_key = os.environ['OPENAI_API_KEY']

# creating the vector store
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
    
    client.create_collection(
   collection_name = qdrant_collection_name,
   vectors_config=vectors_config,   
    )

    vector_store = Qdrant(
        client=client, 
        collection_name=qdrant_collection_name, 
        embeddings=OpenAIEmbeddings(),
    )
  
    return vector_store

# function to return the vectorstore if you have to rerun the code for any reason and don't want to recreate the vector store everytime
# in this case, the vector store was probably already created in data-preprocessing.ipynb so we are going to use this function so...
# we don't have to chunk and upload all the documents again bc that can take like 15 mins!
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

# --------------------------------------------------------------------------------------------------------------------------------

# create 1st collection of vectors
qdrant_collection_1 = os.environ['QDRANT_COLLECTION_1']


collection_check_1 = False

if client.get_collection(qdrant_collection_1):
    vector_store_1 = get_vectorstore(qdrant_collection_1)
    collection_check_1 = True
else:
    vector_store_1 = create_vectorstore(qdrant_collection_1)

# --------------------------------------------------------------------------------------------------------------------------------

# Parent Document Retriever Method
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
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
        search_kwargs = {"k": 10, "score_threshold" : 0.8}
        )
    return parent_retriever

parent_retriever = create_parent_retriever()

if collection_check_1 == False:
# adding  documents into the Qdrant vector database in the 1st collection if not already tehre
    parent_retriever.add_documents(docs)

# testing the retriever
parent_retriever.invoke("What is SMU?")

# --------------------------------------------------------------------------------------------------------------------------------

# semantic text splitting method
# do '%pip install langchain_experimental' if needed
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

semantic_text_splitter = SemanticChunker(
    OpenAIEmbeddings(), 
    breakpoint_threshold_type="percentile")

semantic_docs = semantic_text_splitter.split_documents(docs)
print(semantic_docs[0].page_content)
print(len(semantic_docs))

# --------------------------------------------------------------------------------------------------------------------------------

# creating another instance of a vector store with a new collection using the function we made earlier
qdrant_collection_2 = os.environ['QDRANT_COLLECTION_2']

collection_check_2 = False

# creating the third vector store and retriever
if client.get_collection(qdrant_collection_2):
    vector_store_2 = get_vectorstore(qdrant_collection_2)
    collection_check_2 = True
else:
    vector_store_2 = create_vectorstore(qdrant_collection_2)

# --------------------------------------------------------------------------------------------------------------------------------

def create_vector_store_2_retriever():
    vector_store_2_retriever = vector_store_2.as_retriever(search_type="similarity_score_threshold",
                                                            search_kwargs = {"k": 5, "score_threshold" : 0.75})
    return vector_store_2_retriever

vector_store_2_retriever = create_vector_store_2_retriever()

if collection_check_2 == False:
    vector_store_2_retriever.add_documents(semantic_docs) # adding the semantically split docs into the vector store if not there already

# --------------------------------------------------------------------------------------------------------------------------------

from langchain.retrievers import EnsembleRetriever, BM25Retriever

bm25_retriever = BM25Retriever.from_documents(semantic_docs)

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_store_2_retriever], 
    weights=[0.7, 0.3]
)

ensemble_retriever.invoke("How many credit hours is a major in Computer Science?")

# --------------------------------------------------------------------------------------------------------------------------------

base_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, 
                                                length_function=len, add_start_index=True)  
normal_split_docs = base_text_splitter.split_documents(docs)

# checking result
print(normal_split_docs[0].page_content)
print(len(normal_split_docs))

# --------------------------------------------------------------------------------------------------------------------------------

# getting the collection name of the third vector store
qdrant_collection_0 = os.environ['QDRANT_COLLECTION_0']

collection_check_0 = False

# creating the third vector store and retriever
if client.get_collection(qdrant_collection_0):
    vector_store_0 = get_vectorstore(qdrant_collection_0)
    collection_check_0 = True
else:
    vector_store_0 = create_vectorstore(qdrant_collection_0)

# --------------------------------------------------------------------------------------------------------------------------------

def create_vector_store_0_retriever():
    vector_store_0_retriever = vector_store_0.as_retriever(search_kwargs = {"k": 10, "score_threshold" : 0.8})
    return vector_store_0_retriever

vector_store_0_retriever = create_vector_store_0_retriever()

if collection_check_0 == False:
    vector_store_0_retriever.add_documents(normal_split_docs) # adding split docs into the vector store

# testing the retriever
vector_store_0_retriever.invoke("How many credit hours is a major in Computer Science?")

# --------------------------------------------------------------------------------------------------------------------------------

# using the pandas library to work with excel file and convert it to a data frame
import pandas as pd

# Load the Excel file
excel_path = 'C:/Users/yawbt/OneDrive/Documents/GitHub/SURF-Project_Optimizing-PerunaBot/Data/SMU FAQs.xlsx'
xlsx = pd.ExcelFile(excel_path)

# checking to see if loading the file worked
print(xlsx.sheet_names)

# Iterate through each sheet and save as a CSV file
csv_files = []
for sheet_name in xlsx.sheet_names:
    # Read the entire sheet to extract the metadata from cell A1
    sheet_df = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)
    
    # getting the link of the webpage to include as the metadata 
    metadata = sheet_df.iat[0, 0]
    
    # Read the sheet into a DataFrame starting from the second row
    df = pd.read_excel(xlsx, sheet_name=sheet_name, skiprows=1)
    
    # Save the DataFrame to a CSV file
    csv_path = f'C:/Users/yawbt/OneDrive/Documents/GitHub/SURF-Project_Optimizing-PerunaBot/Data/{sheet_name}.csv'
    df.to_csv(csv_path, index=False)
    csv_files.append((csv_path, metadata))

# Display the list of generated CSV files and their metadata
print(csv_files)

# --------------------------------------------------------------------------------------------------------------------------------

# Now turning each csv into a langchain document
from langchain.document_loaders import CSVLoader

# Create LangChain documents from CSV files with metadata
csv_documents = []

for csv_path, metadata in csv_files:
    loader = CSVLoader(file_path=csv_path)
    csv_docs = loader.load()
    for csv_doc in csv_docs:
        csv_doc.metadata['source'] = metadata
    csv_documents.extend(csv_docs)

# Display the first document as an example
print(csv_documents[0])

# --------------------------------------------------------------------------------------------------------------------------------

# vector store collection 1 - uses parent/child text splitter with parent retriever
if collection_check_1 == False:
    parent_retriever.add_documents(csv_documents)

# --------------------------------------------------------------------------------------------------------------------------------

# vector store collection 2 - uses semantic text splitter (or chunker) with the ensemble retriever (BM25 + vector store as retriever)
# uploaded to vector store using vector store as the retriever
if collection_check_2 == False:
    vector_store_2_retriever.add_documents(csv_documents)

# --------------------------------------------------------------------------------------------------------------------------------

# vector stoer collection 0 - uses the recursive chatacter text splitter with vector store as the retriever
# base option from last project
if collection_check_0 == False:
    vector_store_0_retriever.add_documents(csv_documents)

# --------------------------------------------------------------------------------------------------------------------------------

def get_all_langchain_docs():
  return {
    "pdf_docs": docs,
    "csv_docs": csv_documents
  }

def get_all_vectorstores():
  return {
      "vector_store_0": vector_store_0, # collection smu-data_0
      "vector_store_1": vector_store_1, # collection smu-data_1
      "vector_store_2": vector_store_2, # collection smu-data_2
  }

def get_all_retrievers():
  return {
      "vector_store_0_retriever": vector_store_0_retriever, # collection smu-data_0
      "parent_retriever": parent_retriever, # collection smu-data_1
      "ensemble_retriever": ensemble_retriever, # collection smu-data_2
  }