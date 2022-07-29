from configs.opt_summarizer import SUM_OPT
from configs.sum_config import SUM_CONFIG
import os

os.system("cd ParlAI")

from parlai.core.agents import create_agent_from_model_file

use_cuda = SUM_CONFIG["USE_CUDA"]
model = SUM_CONFIG["MODEL"]

#Setup of the model
opt = SUM_OPT
opt['no_cuda'] = not use_cuda
sum_agent = create_agent_from_model_file(model, opt)

#Reset agent
sum_agent.reset()

def summarize_text(text):
    """ Summarize text using the Summarizer agent """
    text = text.replace("your persona:","").replace("partner's persona:","")[:-2] 
    turn = {'text': text, 'episode_done': True}
    sum_agent.observe(turn)
    return sum_agent.act()['text']

