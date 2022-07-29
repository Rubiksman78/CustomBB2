from configs.opt_summsc import OPT
from configs.summsc_config import DEFAULT_CONFIG
import os
import deepl
import gradio as gr
from scripts.summarizer import summarize_text
from scripts.safety import filter_unsafe
os.system("cd ParlAI")
from parlai.core.agents import create_agent_from_model_file

def predict(query_fr,session=[]):
    global context, chat_agent
    """ Prediction function to speak to the agent """
    query_en = deepl.translate(
        text=query_fr, 
        target_language='EN', 
        source_language='FR'
        )
    if not session:
        turn = {
            'text': context+query_en, 
            'episode_done': False}
    else:
        turn = {
            'text': query_en, 
            'episode_done': False
            }
   
    chat_agent.observe(turn)

    response_en = chat_agent.act()
    #print(response_en['beam_texts'])
    response_en = filter_unsafe(
        response_en['text'],
        response_en['beam_texts'])
    response_fr = deepl.translate(
        text=response_en, 
        target_language='FR', 
        source_language='EN')
    context = context + \
            "partner's persona: " + query_en + "\n" \
            + "your persona: " + response_en + "\n"
    session.append((query_fr,response_fr))
    return session

if __name__ == '__main__':
    use_cuda = DEFAULT_CONFIG["USE_CUDA"]
    n_turns = DEFAULT_CONFIG["N_TURNS"]
    model = DEFAULT_CONFIG["MODEL"]
    print_terminal = DEFAULT_CONFIG["PRINT_TERMINAL"]
    reset_history = DEFAULT_CONFIG["RESET_HISTORY"]
    history_path = DEFAULT_CONFIG["HISTORY_PATH"]

    #Setup of the model
    opt = OPT
    opt['no_cuda'] = not use_cuda
    chat_agent = create_agent_from_model_file(model, opt)
   
    #Reset agent
    chat_agent.reset()

    #Use_context
    if reset_history:
        with open(history_path,"w") as history:
            history.write("")
    with open(history_path,"r") as history:
        context = ''.join(history.readlines())
    context = "partner's persona: " + summarize_text(context) + '\n'
    gr.Interface(fn=predict,
                 inputs="text",
                 outputs="chatbot",
                 title="Chatbot",
                 description="Chatbot",
                 ).launch(share=True,
                height=500,
                width=500,)
    #Write context to file
    with open(history_path,"w") as history:
        history.write(context)
    history.close()