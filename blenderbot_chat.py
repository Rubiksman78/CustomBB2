from configs.opt_blender import OPT
from configs.blend_config import DEFAULT_CONFIG
import os
import deepl
from scripts.utils import reorganize_persona
from scripts.safety import filter_unsafe
os.system("cd ParlAI")
from ParlAI.parlai.core.agents import create_agent_from_model_file

#Same as blender_gradio.py but with a loop for terminal chat
def speak_loop(n_turns, agent,context_for_bot):
    """ Loop to speak to the agent """
    global context
    for i in range(n_turns):
        query_fr = input("Person speaks: ")
        query_en = deepl.translate(
            text=query_fr, 
            target_language='EN', 
            source_language='FR'
            )
        if i == 0:
            turn = {
                'text': context_for_bot+query_en, 
                'episode_done': False
                }
        else:
            turn = {
                'text': query_en, 
                'episode_done': False
                }

        if query_fr == "STOP": #Put any token you want to end the loop
                break
        agent.observe(turn)

        response_en = agent.act()
        #print(response_en['beam_texts'])
        response_en = filter_unsafe(
            response_en['text'],
            response_en['beam_texts'])
        response_fr = deepl.translate(
            text=response_en, 
            target_language='FR', 
            source_language='EN')
        if print_terminal:
            print("Chatbot replied: {}".format(response_fr))
            print()
        context = context + \
            "partner's persona: " + query_en + "  " \
            + "your persona: " + response_en + "  "
    return context
      
if __name__ == '__main__':
    use_cuda = DEFAULT_CONFIG["USE_CUDA"]
    n_turns = DEFAULT_CONFIG["N_TURNS"]
    model = DEFAULT_CONFIG["MODEL"]
    print_terminal = DEFAULT_CONFIG["PRINT_TERMINAL"]
    reset_history = DEFAULT_CONFIG["RESET_HISTORY"]
    history_path = DEFAULT_CONFIG["HISTORY_PATH"]
    
    #Setup of the model
    HOST = "0.0.0.0:1111"
    opt = OPT
    opt['no_cuda'] = not use_cuda
    opt['search_server'] = "$HOST"
    chat_agent = create_agent_from_model_file(model, opt)
    
    #Reset agent
    chat_agent.reset()
    
    #Use_context
    if reset_history:
        with open(history_path,"w") as history:
            history.write("")
    with open(history_path,"r") as history:
        context = ''.join(history.readlines())
    context_for_bot = reorganize_persona(context)
    context_for_bot = context_for_bot + '\n'
    if print_terminal:
        print("--------------------------------")
        print("This is the history of the previous conversations:")
        print(context_for_bot)
        print("--------------------------------")

    context = speak_loop(n_turns, chat_agent,context_for_bot)
    
    #Write context to file
    with open(history_path,"w") as history:
        history.write(context)
    history.close()