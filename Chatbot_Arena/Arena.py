from chains.OG_PerunaBot_chain import Original_PerunaBot_chain
from chains.chain_0 import base_retriever_chain_0
from chains.chain_1 import parent_retriever_chain_1
from chains.chain_2 import ensemble_retriever_chain_2
from recreate_chains import get_random_chains, get_all_chains
from langchain.schema import AIMessage, HumanMessage
from dotenv import load_dotenv, find_dotenv
import time
import gradio as gr 
import os

# Load environment variables from the .env file using 'from dotenv import find_dotenv, load_dotenv'
load_dotenv(find_dotenv(filename='SURF-Project_Optimizing-PerunaBot/Setup/.env'))
open_ai_api_key = os.environ['OPENAI_API_KEY']


all_chains = get_all_chains()
     
#figuring out gradio and understanding streaming
def chat_with_OG_chain(user_input, chat_history):
    chat_history = []
    
    response = parent_retriever_chain_1.invoke({
                "chat_history": chat_history, 
                "input": user_input})
    
    chat_history.append(AIMessage(content=response["answer"]))
    chat_history.append(HumanMessage(content=user_input))
    
    answer = response["answer"]
    stream = ''
    for character in answer:
        time.sleep(0.01)
        stream += character
        yield stream

def chat_with_chain_0(user_input, chat_history):
    chat_history = []
    
    response = base_retriever_chain_0.invoke({
                "chat_history": chat_history, 
                "input": user_input})
    
    chat_history.append(AIMessage(content=response["answer"]))
    chat_history.append(HumanMessage(content=user_input))
    
    answer = response["answer"]
    stream = ''
    for character in answer:
        time.sleep(0.01)
        stream += character
        yield stream

def chat_with_chain_1(user_input, chat_history):
    chat_history = []
    
    response = parent_retriever_chain_1.invoke({
                "chat_history": chat_history, 
                "input": user_input})
    
    chat_history.append(AIMessage(content=response["answer"]))
    chat_history.append(HumanMessage(content=user_input))
    
    answer = response["answer"]
    stream = ''
    for character in answer:
        time.sleep(0.01)
        stream += character
        yield stream

def chat_with_chain_2(user_input, chat_history):
    chat_history = []
    
    response = ensemble_retriever_chain_2.invoke({
                "chat_history": chat_history, 
                "input": user_input})
    
    chat_history.append(AIMessage(content=response["answer"]))
    chat_history.append(HumanMessage(content=user_input))
    
    answer = response["answer"]
    stream = ''
    for character in answer:
        time.sleep(0.01)
        stream += character
        yield stream

def chatting_with_chain_(chain, user_input, chat_history):
    
    response = chain.invoke({
                "chat_history": chat_history, 
                "input": user_input})
    
    chat_history.append(AIMessage(content=response["answer"]))
    chat_history.append(HumanMessage(content=user_input))
    
    answer = response["answer"]
    stream = ''
    for character in answer:
        time.sleep(0.01)
        stream += character
        yield stream

# gr.ChatInterface(fn=chat_with_chain_1).launch()

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

def activate_chat_buttons():
    regenerate_btn = gr.Button(
        value="🔄  Regenerate", interactive=True, elem_id="regenerate_btn"
    )
    clear_btn = gr.Button(
        value="🎲 New Round", 
        interactive=True, 
    )
    return regenerate_btn, clear_btn

def deactivate_chat_buttons():
    regenerate_btn = gr.Button(
        value="🔄  Regenerate", interactive=False, elem_id="regenerate_btn"
    ) 
    clear_btn = gr.Button(
        value="🎲 New Round", 
        interactive=True, 
    )
    return regenerate_btn, clear_btn

def chat_with_chain(chain, user_input, chat_history):
    
    response = chain.invoke({
                "chat_history": chat_history, 
                "input": user_input})
    answer = response["answer"]
    return answer

def new_handle_message_and_process_responses(chains, states1, states2, user_input):
    history1 = states1.value if states1 else []
    history2 = states2.value if states2 else []
    states = [states1, states2]
    history = [history1, history2]

    generators = [
        chains[i]
        for i in range(2)
    ]
    histories = [
        history[i]
        for i in range(2)
    ]
    
    done = [False, False]
    while not all(done):
        for i in range(2):
            if not done[i]:
                try: 
                    response = next(chat_with_chain(generators[i], user_input, histories[i]))
                    if response:
                        history[i].append({"type": "humanMessage", "content": user_input})
                        history[i].append({"type": "aiMessage", "content": response})
                        states[i] = gr.State(history[i])
                        yield history[0], history[1], states[0], states[1]
                        stream = ''
                    for character in response:
                            time.sleep(0.01)
                            stream += character
                            yield stream
                except StopIteration:
                    done[i] = True
    yield history[0], history[1], states[0], states[1]

def regenerate_message(chains, states1, states2):
    history1 = states1.value if states1 else []  
    history2 = states2.value if states2 else []  

    if history1 and isinstance(history1[-1], dict) and history1[-1].get("type") == "humanMessage":
        user_input = history1.pop().content

    for history in [history1, history2]:
        if history and isinstance(history[-1], dict) and history[-1].get("type") == "aiMessage":
            history.pop() 

    states = [states1, states2]  
    history = [history1, history2]  

    for (
        updated_history1,
        updated_history2,
        updated_states1,
        updated_states2,
    ) in new_handle_message_and_process_responses(
        chains, history, states, user_input
    ):
        yield updated_history1, updated_history2, updated_states1, updated_states2




with gr.Blocks(
    js=js_func,
    theme=gr.Theme.from_hub("gradio/soft"),
    title="Welcome to Chatbot Arena!!",
) as demo:
    with gr.Tab("Arena"):
        gr.Markdown(
            """
            # ⚔️ Test and Compare two different Chatbots on the Same Question ⚔️

            ## Rules
            - Ask any questions to the two chatbots that are shown below
            - You can continue to ask questions until you identify a winner as both have chat history
            - The chat history will be reset after each round and two random models will be chosen
            - Choose which response you think is better, if it's a tie, or if both chatbots responded poorly
            - Start a new round with two randomly selected models by clicking '🎲 New Round'
            - Ask the same question again by clicking '🔄 Regenerate'

            ## 🏆 Leaderboard
            - See how the models compare to each other and find out which 
            is the best 🥇

            ## 👇 Chat now!
            """
        )   
        num_sides = 2
        states = [gr.State() for _ in range(num_sides)]
        chatbots = []
        chains = gr.State(get_random_chains)
        all_chains = get_all_chains()

        with gr.Row():
            for i in range(num_sides):
                label = chains.value[i]["name"]
                with gr.Column():
                    chatbots.append(gr.Chatbot(
                        label=label,
                        elem_id=f"chatbot",
                        height=400,
                        show_copy_button=True
                    ))
    with gr.Row():  
        textbox = gr.Textbox(
            show_label=False,
            placeholder="Enter your query and press ENTER",
            elem_id="input_box",
            scale=4,
        ) 
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row() as button_row:  # We make a row for some action buttons below the chat.
        clear_btn = gr.ClearButton(
            value="🎲 New Round",
            elem_id="clear_btn",
            interactive=False,
            components=chatbots + states,
        )

        regenerate_btn = gr.Button(
            value="🔄 Regenerate", interactive=False, elem_id="regenerate_btn"
        )

    with gr.Row():  # We add a row to show example questions people can click on.
        examples = gr.Examples(
            [
                "Tell me about SMU",
                "What resources does SMU provide to support student success and engagement?",
                "Can you tell me about SMU's academic programs and their reputation?",
                "Tell me an interesting fact about SMU",
                "Tell me a unique joke about SMU",
            ],
            inputs=[textbox],
            label="Example inputs",
        )

    textbox.submit(
        new_handle_message_and_process_responses,
        inputs=[
            chains,
            states[0],
            states[1],
            textbox,
        ],
        outputs=[chatbots[0], chatbots[1], states[0], states[1]]
        ).then(
            activate_chat_buttons,
            inputs=[],
            outputs=[regenerate_btn, clear_btn]
        )

    send_btn.click(
        new_handle_message_and_process_responses,
        inputs=[
            chains,
            states[0],
            states[1],
            textbox,
        ],
        outputs=[chatbots[0], chatbots[1], states[0], states[1]]
        ).then(
            activate_chat_buttons,
            inputs=[],
            outputs=[regenerate_btn, clear_btn]
        )
    
    regenerate_btn.click(
        regenerate_message,
        inputs=[
            chains,
            states[0],
            states[1],
        ],
        outputs=[chatbots[0], chatbots[1], states[0], states[1]],
    )
    
    clear_btn.click(
        deactivate_chat_buttons,
        inputs=[],
        outputs=[regenerate_btn, clear_btn]
    ).then(lambda: get_random_chains(), inputs=None, outputs=[chains])
    
    
       

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=10) 
    demo.launch()
