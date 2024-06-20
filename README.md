# PerunaBot OG 01/24
This repository is for tracking end to end LLM changes in PerunaBot that was created in Jan 2024.

## Project Description
PerunaBot is a sophisticated and user-friendly chatbot designed to assist students and faculty at Southern Methodist University. This project leverages the power of OpenAI's technology and integrates with the Qdrant vector database to provide accurate and relevant information. It's built using: Python, ChainLit, Langchain Qdrant, and Streamlit.

## Key features of PerunaBot include:
-Providing information on the SMU catalog

-Utilizing LangChain for enhanced language understanding.

-An interactive chatbot interface created with Chainlit.

-Advanced search and retrieval powered by Qdrant.

-This project aims to create an accessible and helpful resource for the SMU community, streamlining access to critical university information.

## Technical aspect
This version of PerunaBot was built in Jan 2024 in a different Github repository and utilizes...
-the base OpenAI large language model without any fine-tuning  

-a RAG pipeline with access to the SMU catalog through a Qdrant vectorstore

-Chainlit for the UI

-Literal AI for tracking generations, threads, runs, and responses (which will be switched to LangSmith)

The point of this repository is to rebuild the version of PerunaBot from [the original repository](https://github.com/yawbtng/SMUChatBot_Project) in a more organized way that allows for better experimentation, research, and documentation

## Experimentation/Research
This LLM application will be compared in performance to two other versions of PerunaBot that are in different repositories:

-A fine-tuned LLM with OpenAI model gpt-4 or gpt-3.5-turbo

-A fine-tuned LLM with open source models from HuggingFace Hub



