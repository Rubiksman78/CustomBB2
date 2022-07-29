from configs.opt_summsc import OPT
from configs.summsc_config import DEFAULT_CONFIG
import os
import deepl
from scripts.summarizer import summarize_text
from scripts.safety import filter_unsafe
os.system("cd ParlAI")
from parlai.core.agents import create_agent_from_model_file

def speak_loop(n_turns, agent,context):
    """ Loop to speak to the agent """
    for i in range(n_turns):
        query_fr = input("Person speaks: ")
        query_en = deepl.translate(
            text=query_fr, 
            target_language='EN', 
            source_language='FR'
            )
        if i == 0:
            turn = {
                'text': context+query_en, 
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
            "partner's persona: " + query_en + "\n" \
            + "your persona: " + response_en + "\n"
    return context
      

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
    if print_terminal:
        print("--------------------------------")
        print("This is the history of the previous conversations:")
        print(context)
        print("--------------------------------")

    context = speak_loop(n_turns, chat_agent,context)
    
    #Write context to file
    with open(history_path,"w") as history:
        history.write(context)
    history.close()