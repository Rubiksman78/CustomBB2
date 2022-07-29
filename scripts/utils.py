from copy import copy
import pandas
import numpy as np

### Persona insertion ###
def phrasage(word,prefix):
    """
    Args:
        word (str or list): direct object complements of the verb
        prefix (str): subject + verb to put in front of the word
    Raises:
        ValueError: if word not a string or list

    Returns:
        prefix + word + ". " : sentence with the prefix and the word
    """
    if isinstance(word,str):
        return prefix + word + ". "
    elif isinstance(word,list):
        return prefix + " ".join(word) + ". "
    else:
        raise ValueError("word must be a string or list")

def return_persona_string(persona_path,user_id):
    """
    Args:
        persona_path (str): path of the persona database
        user_id (int): id of the user

    Returns:
        sentence_persona(str): sentences with the persona of the user
    """
    df = pandas.read_csv(persona_path,dtype=str)
    persona = df.iloc[user_id].values.tolist()
    sentence_persona = ["a" for i in range(len(persona)-1)]
    sentence_persona[0] = phrasage(persona[0]+persona[1],"My name is ")
    sentence_persona[1] = phrasage(persona[2],"I am a ")
    sentence_persona[2] = phrasage(persona[3],"I am ")
    sentence_persona[3] = phrasage(persona[4],"I like ")
    sentence_persona[4] = phrasage(persona[5],"My job is ")
    sentence_persona[5] = phrasage(persona[6],"I live in ")
    sentence_persona[6] = phrasage(persona[7],"I have ")
    sentence_persona = " ".join(''.join(sentence_persona).split()).replace(';','')
    return sentence_persona

### Persona organization ###
def reorganize_one_session(session_context,delimiter,is_persona,is_last_session):
    """
    Args:
        session_context (str): context of one session
        delimiter (str): delimiter for personas
        is_persona (bool): if there is a persona summary appended at the end
        is_last_session (bool): if it is the last session

    Returns:
        res(str): cleaned context of one session
    """
    history_list = session_context.split(delimiter)
    persona_list = []
    if is_persona and is_last_session:
        last_to_have_persona = "your persona:" 
        last = len(history_list) - 2
    else:  
        last_to_have_persona = "partner's persona:"
        last = len(history_list) - 1
    
    while not history_list[last].startswith(last_to_have_persona) and last>0:
        history_list.pop()
        last -= 1
    for i in range(len(history_list)-1,0,-1): 
        if history_list[i].startswith("your persona:") and \
            history_list[i-1].startswith("partner's persona:") or \
                history_list[i].startswith("partner's persona:") and \
                    history_list[i-1].startswith("your persona:"):
            persona_list.append(history_list[i])
    if len(history_list) > 1: 
        if history_list[0].startswith("your persona:") and \
            history_list[1].startswith("partner's persona:") or \
                history_list[0].startswith("partner's persona:") and \
                    history_list[1].startswith("your persona:"):
            persona_list.append(history_list[0])
    persona_list.reverse()
    res = delimiter.join(persona_list).replace("your persona: ","").replace("partner's persona: ","")
    return res

def reorganize_persona(history,delimiter="  ",session_delimiter='\n',is_persona=False):
    session = history.split(session_delimiter)
    session_res = []
    for i,session_context in enumerate(session):
        is_last_session = i == len(session)-1
        res = reorganize_one_session(session_context,delimiter,is_persona,is_last_session)
        session_res.append(res)
    return session_delimiter.join(session_res)