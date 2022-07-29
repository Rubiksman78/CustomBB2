from configs.blend_config import DEFAULT_CONFIG
from scripts.utils import return_persona_string,reorganize_persona
from scripts.safety import filter_unsafe
use_small = DEFAULT_CONFIG['USE_SMALL']
import os
import deepl
import gradio as gr
if use_small:
    from configs.opt_blender_small import OPT #import small config (400M parameters)
else:
    from configs.opt_blender import OPT #or import normal config (3B)
os.system("cd ParlAI")
from parlai.core.agents import create_agent_from_model_file

def predict(query_fr,session=[]):
    global context, chat_agent, context_for_bot
    query_en = deepl.translate(
        text=query_fr,
        target_language='EN',
        source_language='FR'
        ) # translate query to english

    if not session: #append context for the first turn only)
        turn = {
            'text': context_for_bot+query_en,
            'episode_done': False
            } #text and episode_done mandatory
    else:
        turn = {'text': query_en, 'episode_done': False}
   
    chat_agent.observe(turn) #observe the turn

    response_en_complete = chat_agent.act() #get response from agent
    #print(response_en_complete['memories'])
    response_en = filter_unsafe(
        response_en_complete['text'],
        response_en_complete['beam_texts']
        ) #filter unsafe responses (warning: response_en is just the text response)
    response_fr = deepl.translate(
        text=response_en, 
        target_language='FR', 
        source_language='EN'
        ) # translate response to french
    if use_small: #if small delimiter '\n' is used
        context = context + \
            "partner's persona: " + query_en + "\n" + \
            "your persona: " + response_en + "\n" 
    else: #if normal delimiter '  ' is used
        context = context + \
            "partner's persona: " + query_en + "  " \
            + "your persona: " + response_en + "  "
    session.append((query_fr,response_fr)) #append query/answer to session for printing
    return session

if __name__ == '__main__':
    #Import parameters
    use_cuda = DEFAULT_CONFIG["USE_CUDA"]
    model = DEFAULT_CONFIG["MODEL"]
    reset_history = DEFAULT_CONFIG["RESET_HISTORY"]
    history_path = DEFAULT_CONFIG["HISTORY_PATH"]
    user_id = DEFAULT_CONFIG["ID"]
    persona_path = DEFAULT_CONFIG["PERSONA_PATH"]
    include_persona = DEFAULT_CONFIG["INCLUDE_PERSONA"]

    #Setup of the model
    HOST = "0.0.0.0:1111" #BB2 refuses to have None server for search so just put a dummy one
    opt = OPT
    opt['no_cuda'] = not use_cuda
    opt['search_server'] = "$HOST"
    chat_agent = create_agent_from_model_file(model, opt) #built-in function to create agent from model file

    #Reset agent
    chat_agent.reset()

    #Use_context
    if reset_history: #if we want to reset the history, just write an empty string
        with open(history_path,"w") as history:
            history.write("")
    with open(history_path,"r") as history:
        context = ''.join(history.readlines())

    context_for_bot = reorganize_persona(context) #context without persona tokens (for model)
    context_for_bot = context_for_bot + '\n'
    context = context + '\n' #context with persona tokens (to save)

    #Use persona
    if include_persona:
        persona = return_persona_string(persona_path,user_id)
        context_for_bot = context_for_bot + "partner's persona: " + persona + '\n'

    gr.Interface(
                fn=predict,
                inputs="text",
                outputs="chatbot",
                title="Chatbot",
                description="Chatbot",
                ).launch(
                share=True,
                height=500,
                width=500,
                server_name='0.0.0.0',
                 )

    #Write context to file
    with open(history_path,"w") as history:
        history.write(context)
    history.close()