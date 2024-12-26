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
This LLM application was compared in performance to three other versions of PerunaBot within the same repository:

- PerunaBot v0: An upgraded version using GPT-4 with a more detailed personality template.

- PerunaBot v1: A version using a Parent-Child document structure for improved context-aware retrieval.

- PerunaBot v2: The most advanced version, combining semantic chunking and ensemble retrieval techniques.

These variations were implemented to evaluate different approaches in answering questions about SMU, allowing for direct comparison of performance and effectiveness. The project includes a "Model Descriptions" section in the user interface, providing detailed explanations of each version's technical implementation and purpose.
