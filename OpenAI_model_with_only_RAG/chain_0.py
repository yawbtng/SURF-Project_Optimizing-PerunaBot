import os
from dotenv import find_dotenv, load_dotenv
from langsmith import Client
from langchain_qdrant.vectorstores import Qdrant
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
        embeddings=OpenAIEmbeddings(), # Uses 'from langchain_openai import OpenAIEmbeddings'
    )
    return vector_store

# Initialize vector store for collection 0 using 'get_vectorstore' function
vector_store_0 = get_vectorstore(qdrant_collection_0)
vector_store_0_retriever = vector_store_0.as_retriever(search_kwargs = {"k": 8, "score_threshold" : 0.75})


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

#____________________________________________________________________________

# Define a function to create a chain based on each retriever using 'from langchain.chains import create_history_aware_retriever, create_retrieval_chain' 
# and 'from langchain.chains.combine_documents import create_stuff_documents_chain'
def create_chain(vector_store_retriever):
   
   #  Create a chain based on retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, vector_store_retriever, condense_question_prompt
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return convo_qa_chain

# Create chain for collection 0
base_retriever_chain_0 = create_chain(vector_store_0_retriever)
base_retriever_chain_0 = base_retriever_chain_0.with_config({"run_name": "PerunaBot 0"})
base_retriever_chain_0 = base_retriever_chain_0.with_config({
    "tags": ["chain_0"], 
    "metadata": {
        "retriever": "base retriever (aka vector store as retriever)", 
        "collection": "smu_data-0", 
        "llm": "gpt-4o"
        }
})

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
    chat_history_0 = []

    # Start chat with PerunaBot 0
    print("You are talking with PerunaBot 0 that uses vector store 0 and the base retriever")

    check_0 = True
    while check_0:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            check_0 = False
            chat_history_0.clear()
        else:
            response = process_chat(base_retriever_chain_0, user_input, chat_history_0)
            chat_history_0.append(HumanMessage(content=user_input)) # Uses 'from langchain_core.messages import HumanMessage'
            chat_history_0.append(AIMessage(content=response)) # Uses 'from langchain_core.messages import AIMessage'
            print("User: ", user_input)
            print("PerunaBot 0: ", response)


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
base_retriever_eval_chain_0 = (
    {"context": itemgetter("question") | vector_store_0_retriever,
     "question": itemgetter("question")} 
     | RunnablePassthrough.assign(output = generation_chain))

base_retriever_eval_chain_0 = base_retriever_eval_chain_0.with_config({"run_name": "PerunaBot 0 Eval"})
base_retriever_eval_chain_0 = base_retriever_eval_chain_0.with_config({
    "tags": ["chain_0"], 
    "metadata": {
        "retriever": "base retriever (aka vector store as retriever)", 
        "collection": "smu_data-0", 
        "llm": "gpt-4o"
        }
})
base_retriever_eval_chain_0.invoke({"question": "What are some good resources on campus?"})
 # ____________________________________________________________________________