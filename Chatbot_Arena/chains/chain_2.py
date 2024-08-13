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
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Load environment variables from the .env file using 'from dotenv import find_dotenv, load_dotenv'
load_dotenv(find_dotenv(filename='SURF-Project_Optimizing-PerunaBot/setup/.env'))

# Initialize API keys and environment variables using 'import os'
#Qdrant vector db
qdrant_host = os.environ['QDRANT_HOST']
qdrant_api_key = os.environ['QDRANT_API_KEY']
qdrant_collection_2 = os.environ['QDRANT_COLLECTION_2']

#OpenAI API
openai_api_key = os.environ['OPENAI_API_KEY']

# langsmith
langsmith_api_key = os.environ["LANGSMITH_API_KEY"]
langchain_endpoint = os.environ["LANGCHAIN_ENDPOINT"]
langsmith_project = os.environ["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_TRACING_V2"]

# Initialize LangSmith Client using 'from langsmith import Client'
langsmith_client = Client()


# Update the path to the correct location of your .db file
db_path = "C:/Users/yawbt/Documents/GitHub/SURF-Project_Optimizing-PerunaBot/Common/serialized_data/data_preprocessing_langchain_docs.db"

# Load the LangChain documents from the shelve file
with shelve.open(db_path) as db:
    langchain_docs_loaded = {key: db[key] for key in db}

csv_docs = langchain_docs_loaded['csv_docs']
semantic_docs = langchain_docs_loaded['semantic_docs']


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

# Initialize vector store for collection 2 using 'get_vectorstore' function
vector_store_2 = get_vectorstore(qdrant_collection_2)

# Initialize vector store retriever with specific search parameters
vector_store_2_retriever = vector_store_2.as_retriever(search_type="similarity_score_threshold",
                                                      search_kwargs={"k": 8, "score_threshold": 0.75})

# Initialize BM25 retriever using 'from langchain_community.retrievers import BM25Retriever'
bm25_retriever = BM25Retriever.from_documents(semantic_docs + csv_docs)

# Initialize the ensemble retriever using 'from langchain.retrievers import EnsembleRetriever'
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_store_2_retriever], 
    weights=[0.5, 0.5]
)

# ensemble_retriever.invoke("What if I don't know what to major in?")

# Load the prompts from the JSON file
prompts_path = "C:/Users/yawbt/Documents/GitHub/SURF-Project_Optimizing-PerunaBot/OpenAI_model_with_only_RAG/prompts.json"

with open(prompts_path, "r") as json_file:
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

# Create chain for collection 2 using 'ensemble_retriever'
ensemble_retriever_chain_2 = create_chain(ensemble_retriever)
ensemble_retriever_chain_2 = ensemble_retriever_chain_2.with_config({"run_name": "PerunaBot 2"})
ensemble_retriever_chain_2 = ensemble_retriever_chain_2.with_config({
    "tags": ["chain_2"], 
    "metadata": {
        "retriever": "ensemble retriever", 
        "components & weights": "(bm25 + vector store) [0.5, 0.5]",
        "collection": "smu_data-2", 
        "llm": "gpt-4o"
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
    chat_history_2 = []

    # Start chat with PerunaBot 2
    print("You are talking with PerunaBot 2 that uses vector store 2 and the ensemble retriever")

    check_2 = True
    while check_2:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            check_2 = False
            chat_history_2.clear()
        else:
            response = process_chat(ensemble_retriever_chain_2, user_input, chat_history_2)
            chat_history_2.append(HumanMessage(content=user_input)) # Uses 'from langchain_core.messages import HumanMessage'
            chat_history_2.append(AIMessage(content=response)) # Uses 'from langchain_core.messages import AIMessage'
            print("User: ", user_input)
            print("PerunaBot 2: ", response)

if __name__ == '__main__':
    chat_convo()
# ____________________________________________________________________________

# Chain without history for evaluation
from langchain_core.output_parsers import MarkdownListOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

new_qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", chatbot_personality),
        ("user", "{question}"),
    ]
)

generation_chain = new_qa_prompt | llm | StrOutputParser()
ensemble_retriever_eval_chain_2 = (
    {"context": itemgetter("question") | ensemble_retriever,
     "question": itemgetter("question")} 
     | RunnablePassthrough.assign(output = generation_chain))


# Configure the chain
ensemble_retriever_eval_chain_2 = ensemble_retriever_eval_chain_2.with_config({"run_name": "PerunaBot 2 Eval"})
ensemble_retriever_eval_chain_2 = ensemble_retriever_eval_chain_2.with_config({
    "tags": ["chain_2"], 
    "metadata": {
        "retriever": "ensemble retriever", 
        "collection": "smu_data-2", 
        "llm": "gpt-4o"
        }
})

# ensemble_retriever_eval_chain_2.invoke({"question": "What if I can't afford to go to SMU?"})
 # ____________________________________________________________________________

new_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.25, max_tokens=750, timeout=None, max_retries=2)

generation_chain = new_qa_prompt | new_llm | StrOutputParser()
ensemble_retriever_eval_chain_2_v1 = (
    {"context": itemgetter("question") | ensemble_retriever,
     "question": itemgetter("question")} 
     | RunnablePassthrough.assign(output = generation_chain))


# Configure the chain
ensemble_retriever_eval_chain_2_v1 = ensemble_retriever_eval_chain_2_v1.with_config({"run_name": "PerunaBot 2 Eval"})
ensemble_retriever_eval_chain_2_v1 = ensemble_retriever_eval_chain_2_v1.with_config({
    "tags": ["chain_2"], 
    "metadata": {
        "retriever": "ensemble retriever", 
        "collection": "smu_data-2", 
        "llm": "gpt-3.5-turbo"
        }
})

 # ____________________________________________________________________________

new_llm_2 = ChatOpenAI(model="gpt-4o-mini", temperature=0.25, max_tokens=750, timeout=None, max_retries=2)

generation_chain = new_qa_prompt | new_llm_2 | StrOutputParser()
ensemble_retriever_eval_chain_2_v2 = (
    {"context": itemgetter("question") | ensemble_retriever,
     "question": itemgetter("question")} 
     | RunnablePassthrough.assign(output = generation_chain))


# Configure the chain
ensemble_retriever_eval_chain_2_v2 = ensemble_retriever_eval_chain_2_v2.with_config({"run_name": "PerunaBot 2 Eval"})
ensemble_retriever_eval_chain_2_v2 = ensemble_retriever_eval_chain_2_v2.with_config({
    "tags": ["chain_2_v2"], 
    "metadata": {
        "retriever": "ensemble retriever", 
        "collection": "smu_data-2", 
        "llm": "gpt-4o-mini"
        }
})