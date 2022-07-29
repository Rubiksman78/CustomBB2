from configs.opt_summsc import OPT
import streamlit as st
from configs.summsc_config import DEFAULT_CONFIG
from streamlit_chat import message
import os
os.system("cd ParlAI")

from parlai.core.agents import create_agent_from_model_file

def load_model():
    #Setup of the model
    opt = OPT
    opt['no_cuda'] = not use_cuda

    chat_agent = create_agent_from_model_file(model, opt)

    #Reset agent
    chat_agent.reset()
    return chat_agent

def speak_loop(n_turns, agent,context):
    if "history" not in st.session_state:
        st.session_state.history = []

    st.title("Hello Chatbot")

    def generate_answer():
        user_message = st.session_state.input_text
        turn = {'text': context[:-5]+user_message, 'episode_done': False}
        agent.observe(turn)
        response = agent.act()['text']
    
        st.session_state.history.append({"message": user_message, "is_user": True})
        st.session_state.history.append({"message": response, "is_user": False})

    st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)
    for chat in st.session_state.history:
        
        message(**chat)  

if __name__ == '__main__':
    use_cuda = DEFAULT_CONFIG["USE_CUDA"]
    n_turns = DEFAULT_CONFIG["N_TURNS"]
    model = DEFAULT_CONFIG["MODEL"]
    print_terminal = DEFAULT_CONFIG["PRINT_TERMINAL"]
    reset_history = DEFAULT_CONFIG["RESET_HISTORY"]

    chat_agent = load_model()
    
    #Use_context
    if reset_history:
        with open("history.txt","w") as history:
            history.write("")
    with open("history.txt","r") as history:
        context = ' '.join(history.readlines())
    if print_terminal:
        print("--------------------------------")
        print("This is the history of the previous conversations:")
        print(context)
        print("--------------------------------")

    speak_loop(n_turns, chat_agent,context)
    
