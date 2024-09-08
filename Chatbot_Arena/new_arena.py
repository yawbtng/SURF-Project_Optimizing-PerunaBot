from chains.OG_PerunaBot_chain import Original_PerunaBot_chain
from chains.chain_0 import base_retriever_chain_0
from chains.chain_1 import parent_retriever_chain_1
from chains.chain_2 import ensemble_retriever_chain_2
from recreate_chains import get_random_chains, get_all_chains
from recreate_chains import chat_with_OG_chain, chat_with_chain_0, chat_with_chain_1, chat_with_chain_2
from dotenv import load_dotenv, find_dotenv
import gradio as gr 
import os

# Load environment variables from the .env file using 'from dotenv import find_dotenv, load_dotenv'
load_dotenv(find_dotenv(filename='SURF-Project_Optimizing-PerunaBot/Setup/.env'))
open_ai_api_key = os.environ['OPENAI_API_KEY']

all_chains = get_all_chains()
     


js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""


def create_textbox():
    return gr.Textbox(
        show_label=False,
        placeholder="Enter your query and press ENTER",
        elem_id="input_box",
        scale=4,
    )

def create_regenerate_button():
    return gr.Button(
        value="🔄 Regenerate", 
        interactive=True, 
        elem_id="regenerate_btn"
    )

def create_clear_button():
    return gr.ClearButton(
        value="🎲 New Round", 
        interactive=True, 
        elem_id="clear_btn"
    )


def update_models(selection):
    if selection == "Original PerunaBot (from 2024) vs PerunaBot v0":
        return [all_chains[0], all_chains[1]]

    elif selection == "Original PerunaBot (from 2024) vs PerunaBot v1":
        return [all_chains[0], all_chains[2]]

    elif selection == "Original PerunaBot (from 2024) vs PerunaBot v2":
        return [all_chains[0], all_chains[3]]

    elif selection == "PerunaBot v0 vs PerunaBot v1":
        return [all_chains[1], all_chains[2]]

    elif selection == "PerunaBot v2 vs PerunaBot v0":
        return [all_chains[3], all_chains[1]]

    elif selection == "PerunaBot v1 vs PerunaBot v2":
        return [all_chains[2], all_chains[3]]

    elif selection == "Random":
        return get_random_chains()

with gr.Blocks(
    theme=gr.Theme.from_hub("gradio/soft"),
    title="Welcome to Chatbot Arena!!",
) as demo:
    with gr.Tab("Arena⚔️"):
        gr.Markdown(
            """
            # ⚔️ Test and Compare two different Chatbots on the Same Question ⚔️

            ## Rules
            - Ask any questions to the two chatbots that are shown below
            - You can continue to ask questions until you identify a winner
            - Start a new chat by selecting a different set of models from the dropdown
            - See how the models compare to each other and find out which you think is the best 🥇

            ## Arena Options
            ### Original version versus new versions
            - Original PerunaBot (from Jan 2024) 🌟 vs PerunaBot v0 🤖
            - Original PerunaBot (from Jan 2024) 🌟 vs PerunaBot v1 🚀
            - Original PerunaBot (from Jan 2024) 🌟 vs PerunaBot v2 🔥
            ### New versions versus each other
            - PerunaBot v0 🤖 vs PerunaBot v1 🚀
            - PerunaBot v2 🔥 vs PerunaBot v0 🤖
            - PerunaBot v1 🚀 vs PerunaBot v2 🔥

            ## 👇 Chat now!
            """
        )   
        num_sides = 2
        chatbots = [None] * num_sides

        
        with gr.Tabs() as tabs:
            with gr.Tab("(from Jan 2024) 🌟 vs v0 🤖"):
                with gr.Row():
                    textbox_1 = create_textbox()
            
                with gr.Row():
                    examples = gr.Examples(
                    [
                        "What are SMU's most popular majors?",
                        "How does SMU support international students?",
                        "What is the history behind SMU's mascot, Peruna?",
                        "Can you describe SMU's campus architecture?",
                        "What research opportunities are available for undergraduates at SMU?",
                    ],
                    inputs=[textbox_1], #type: ignore
                    label="Example inputs"
                )   

                chains_accordion_1 = update_models("Original PerunaBot (from 2024) vs PerunaBot v0")

                with gr.Row():
                    for i in range(num_sides):
                        label = chains_accordion_1[i]["name"] # type: ignore
                        with gr.Column():
                            chatbots[i] = gr.ChatInterface( # type: ignore
                                chatbot=gr.Chatbot(
                                    label=label, 
                                    elem_id=f"chatbot_{i}",
                                    height=400,
                                    show_copy_button=True
                                    ),
                                
                                fn=chains_accordion_1[i]["chat_function"], # type: ignore
                                textbox=textbox_1, 
                                submit_btn=None,  
                            )
                
            with gr.Tab("(from Jan 2024) 🌟 vs v1 🚀"):
                with gr.Row():
                    textbox_2 = create_textbox()
            
                with gr.Row():
                    examples = gr.Examples(
                    [
                        "What is SMU's approach to liberal arts education?",
                        "How does SMU promote diversity and inclusion on campus?",
                        "What are some unique traditions at SMU?",
                        "Can you explain SMU's honor code system?",
                        "What study abroad programs does SMU offer?",
                    ],
                    inputs=[textbox_2], #type: ignore
                    label="Example inputs"
                )   

                chains_accordion_2 = update_models("Original PerunaBot (from 2024) vs PerunaBot v1")
                with gr.Row():
                    for i in range(num_sides):
                        label = chains_accordion_2[i]["name"] # type: ignore
                        with gr.Column():
                            chatbots[i] = gr.ChatInterface( # type: ignore
                                chatbot=gr.Chatbot(
                                    label=label, 
                                    elem_id=f"chatbot_{i}",
                                    height=400,
                                    show_copy_button=True
                                    ),
                                
                                fn=chains_accordion_2[i]["chat_function"], # type: ignore
                                textbox=textbox_2, 
                                submit_btn=None,  
                            )

            with gr.Tab("(from Jan 2024) 🌟 vs v2 🔥"):
                with gr.Row():
                    textbox_3 = create_textbox()
            
                with gr.Row():
                    examples = gr.Examples(
                    [
                        "What is the student-to-faculty ratio at SMU?",
                        "How does SMU support entrepreneurship among students?",
                        "What are some notable alumni from SMU?",
                        "Can you describe SMU's athletic programs?",
                        "What sustainability initiatives does SMU have?",
                    ],
                    inputs=[textbox_3], #type: ignore
                    label="Example inputs"
                )   

                chains_accordion_3 = update_models("Original PerunaBot (from 2024) vs PerunaBot v2")
                with gr.Row():
                    for i in range(num_sides):
                        label = chains_accordion_3[i]["name"] # type: ignore
                        with gr.Column():
                            chatbots[i] = gr.ChatInterface( # type: ignore
                                chatbot=gr.Chatbot(
                                    label=label, 
                                    elem_id=f"chatbot_{i}",
                                    height=400,
                                    show_copy_button=True
                                    ),
                                
                                fn=chains_accordion_3[i]["chat_function"], # type: ignore
                                textbox=textbox_3, 
                                submit_btn=None,  
                            )
            
            with gr.Tab("v0 🤖 vs v1 🚀"):
                with gr.Row():
                    textbox_4 = create_textbox()
            
                with gr.Row():
                    examples = gr.Examples(
                    [
                        "What career services does SMU provide?",
                        "How does SMU support students' mental health?",
                        "What is the Engaged Learning program at SMU?",
                        "Can you describe SMU's residential commons system?",
                        "What are some popular student organizations at SMU?",
                    ],
                    inputs=[textbox_4], #type: ignore
                    label="Example inputs"
                )

                chains_accordion_4 = update_models("PerunaBot v0 vs PerunaBot v1")
                with gr.Row():
                    for i in range(num_sides):
                        label = chains_accordion_4[i]["name"] # type: ignore
                        with gr.Column():
                            chatbots[i] = gr.ChatInterface( # type: ignore
                                chatbot=gr.Chatbot(
                                    label=label, 
                                    elem_id=f"chatbot_{i}",
                                    height=400,
                                    show_copy_button=True
                                    ),
                                
                                fn=chains_accordion_4[i]["chat_function"], # type: ignore
                                textbox=textbox_4, 
                                submit_btn=None,  
                            )

            with gr.Tab("v2 🔥 vs v0 🤖"):
                with gr.Row():
                    textbox_5 = create_textbox()
            
                with gr.Row():
                    examples = gr.Examples(
                    [
                        "What is the Dedman College of Humanities and Sciences?",
                        "How does SMU support first-generation college students?",
                        "What is the SMU-in-Taos program?",
                        "Can you explain SMU's core curriculum?",
                        "What are some unique features of SMU's libraries?",
                    ],
                    inputs=[textbox_5], #type: ignore
                    label="Example inputs"
                )

                chains_accordion_5 = update_models("PerunaBot v2 vs PerunaBot v0")
                with gr.Row():
                    for i in range(num_sides):
                        label = chains_accordion_5[i]["name"] # type: ignore
                        with gr.Column():
                            chatbots[i] = gr.ChatInterface( # type: ignore
                                chatbot=gr.Chatbot(
                                    label=label, 
                                    elem_id=f"chatbot_{i}",
                                    height=400,
                                    show_copy_button=True
                                    ),
                                
                                fn=chains_accordion_5[i]["chat_function"], # type: ignore
                                textbox=textbox_5, 
                                submit_btn=None,  
                            )
                          
            with gr.Tab("v1 🚀 vs v2 🔥"):
                with gr.Row():
                    textbox_6 = create_textbox()
            
                with gr.Row():
                    examples = gr.Examples(
                    [
                        "What is the Lyle School of Engineering known for?",
                        "How does SMU support students' professional development?",
                        "What is the significance of Dallas Hall to SMU?",
                        "Can you describe SMU's commitment to the arts?",
                        "What research centers does SMU have?",
                    ],
                    inputs=[textbox_6], #type: ignore
                    label="Example inputs"
                )

                chains_accordion_6 = update_models("PerunaBot v1 vs PerunaBot v2")
                with gr.Row():
                    for i in range(num_sides):
                        label = chains_accordion_6[i]["name"] # type: ignore
                        with gr.Column():
                            chatbots[i] = gr.ChatInterface( # type: ignore
                                chatbot=gr.Chatbot(
                                    label=label, 
                                    elem_id=f"chatbot_{i}",
                                    height=400,
                                    show_copy_button=True
                                    ),
                                
                                fn=chains_accordion_6[i]["chat_function"], # type: ignore
                                textbox=textbox_6, 
                                submit_btn=None,  
                            )

    with gr.Tab("Model Descriptions🛈"):
        gr.Markdown(
            """
            # Model Descriptions

            Welcome to the PerunaBot Arena! Here, we showcase different versions of our SMU AI assistant, each with its own unique approach to answering your questions about SMU.

            ## Original PerunaBot (from Jan 2024) 🌟

            ### Simple Explanation
            This is the original version of PerunaBot made in January 2024. It is the first version and uses a simple approach to answering questions.

            ### Technical Explanation
            - **Data Chunking:** RecursiveCharacterTextSplitter (chunk size: 500, overlap: 50)
            - **Retrieval:** Simple vector store retriever
            - **Prompt:** Custom template focused on context-based question answering
            - **Model:** GPT-3.5-turbo (with options for GPT-4 and GPT-4-mini in evaluation chains)

            ## PerunaBot v0 🤖

            ### Simple Explanation
            This version is very similar to the original but with a brain upgrade and a more detailed personality.

            ### Technical Explanation
            - **Data Chunking:** RecursiveCharacterTextSplitter (chunk size: 500, overlap: 50)
            - **Retrieval:** Vector store retriever with similarity search (k=8, score threshold: 0.75)
            - **Prompt:** Detailed chatbot personality template
            - **Model:** GPT-4

            ## PerunaBot v1 🚀

            ### Simple Explanation
            This version is like having a librarian who not only knows where the books are but also understands the relationships between different topics.

            ### Technical Explanation
            - **Data Chunking:** Parent-Child structure
                - Parent: RecursiveCharacterTextSplitter (chunk size: 750, overlap: 50)
                - Child: RecursiveCharacterTextSplitter (chunk size: 250, overlap: 25)
            - **Retrieval:** ParentDocumentRetriever
            - **Prompt:** Detailed chatbot personality template
            - **Model:** GPT-4

            ## PerunaBot v2 🔥

            ### Simple Explanation
            This is the most advanced version, combining different techniques to understand and answer questions.

            ### Technical Explanation
            - **Data Chunking:** SemanticChunker for meaning-based splits
            - **Retrieval:** EnsembleRetriever
                - Combines BM25Retriever and vector store retriever
                - Weights: [0.5, 0.5]
            - **Prompt:** Detailed chatbot personality template
            - **Model:** GPT-4

            ## Why These Differences Matter

            Each version of PerunaBot is designed to handle questions about SMU in slightly different ways:

            - The **Original PerunaBot** provides a baseline performance using simpler techniques.
            - **PerunaBot v0** upgrades the language model while maintaining the original chunking strategy.
            - **PerunaBot v1** introduces a hierarchical document structure for more context-aware retrieval.
            - **PerunaBot v2** combines keyword-based and semantic search for potentially more accurate information retrieval.

            These variations allow us to compare different approaches and find the most effective way to answer your questions about SMU. Feel free to try each version and see which one works best for you!

            """
        )

if __name__ == "__main__":
    demo.launch()