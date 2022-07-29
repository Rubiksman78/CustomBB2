from parlai.utils.safety import OffensiveStringMatcher, OffensiveLanguageClassifier
from configs.sensitive_opt import SAFE_CONFIG
import os
os.system("cd ParlAI")
from ParlAI.parlai.core.agents import create_agent_from_model_file

# Load the Sensitive topic model and the offensive language classifier
offensive_classifier = OffensiveLanguageClassifier()
safe_opt = SAFE_CONFIG
model = "zoo:sensitive_topics_classifier/model"
safe_agent = create_agent_from_model_file(model, safe_opt)
safe_agent.reset()

### Safe filter ###
def filter_potentially(sentence):
    # Check if the sentence is potentially offensive based only on the tokens returned by BB2
    word_list = sentence.split()
    unsafe = False
    for i in range(len(word_list)):
        if word_list[i].startswith(
            ('_POTENTIALLY_','_Potentially_')
            ):
            unsafe = True
    return unsafe

def filter_with_model(sentence):
    # Check if the sentence is potentially offensive based on the Offensive language classifier
    text_is_unsafe, proba = offensive_classifier.contains_offensive_language(sentence)
    #print(f"{sentence} is {'unsafe' if text_is_unsafe else 'safe'} with probability {proba}")
    if text_is_unsafe and proba < 0.8: #Tune proba threshold for unsafe
        text_is_unsafe = not text_is_unsafe
    return text_is_unsafe

def filter_topic(sentence):
    # Check if the sentence is potentially offensive based on the Sensitive topic model
    safe_agent.observe({'text':sentence,'episode_done': True})
    response = safe_agent.act()
    topic = response['text']
    print(f"topic: {topic}")
    if topic in ["politics","religion","nsfw","drugs"]:
        return True
    else:
        return False

def filter_unsafe(sentence,texts,with_model=True):
    """
    Args:
        sentence (str): sentence to check if unsafe
        texts (str): list of alternative texts 

    Returns:
        to_check(str): checked sentence 
    """
    to_check = sentence
    i = 0
    n = len(texts)
    filter_function = filter_with_model if with_model else filter_potentially
    while i < n-1 and filter_function(to_check) and filter_topic(to_check):
        to_check = texts[i][0]
        i += 1
    if i >= n-1:
        return sentence
    else:
        return to_check
