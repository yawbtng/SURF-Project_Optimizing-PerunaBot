from chains.OG_PerunaBot_chain import Original_PerunaBot_chain
from chains.chain_0 import base_retriever_chain_0
from chains.chain_1 import parent_retriever_chain_1
from chains.chain_2 import ensemble_retriever_chain_2
import random

def get_all_chains():
    return [
        {
            "name": "Original PerunaBot (from  2024)",
            "chain": Original_PerunaBot_chain
        },
        {
            "name": "PerunaBot v0",
            "chain": base_retriever_chain_0
        },
        {
            "name": "PerunaBot v1",
            "chain": parent_retriever_chain_1
        },
        {
            "name": "PerunaBot v2",
            "chain": ensemble_retriever_chain_2
        }

    ]

def get_random_chains(number: int = 2):
    return random.sample(get_all_chains(), number)