# Chatbot with long term memory

This project aims to implement a simple interface to use Facebook ParlAI's chatbot based on the paper [Beyond Goldfish Memory: Long-Term Open-Domain Conversation](https://arxiv.org/abs/2107.07567).

It provides multiple scripts to discuss with Summary-Memory model and BlenderBot2 or eventually train them.

## Installation :computer:

The ParlAI repository recquires a lot of requirements. :fearful:

It is recommended to use a Conda environment this time because there can be some bugs on Windows with pip or on Google Colab.
It is also highly recommended to be on a Linux setup (WSL or whatsoever) for some installations like fairseq or FAISS.
Just execute these commands 
```
git clone https://github.com/Rubiksman78/Memory_chatbot.git
```
If you want to use virtualenv:
```
pip install virtualenv
virtualenv venv

source venv/bin/activate #or source venv/Scripts/activate on Windows
```
If you want to use conda venvs:
```
conda create -n venv python=3.9
conda activate venv
```
And now install all the requirements for ParlAI (the setup can be quite long).
```
bash setup.sh
```

## Interact with META AI browserchat :email:
If you want to chat with the model on the browser_chat API provided by FB you can execute
```
bash scripts/chat.sh
```
You can now connect from your browser on "IP:8080".
You can get the IP of your remote machine with:
```
hostname -I
```
There might be some issues if the download of the model is interrupted during the process (be sure to have a stable internet connection or you would have to download the model several times). Just delete the folder ParlAI/data/models/problematic_model and relaunch the command.

## Interact with the simplified script :outbox_tray:

You can specify which way of chatting you want to use in `chat_config.py`. You can also take a look at the opt files in the `configs` folders which can allow you to custom the models (for inference or to fine tune them).

You can store history conversations in the `text_files` folder and change the related paths in the configs. You can also create a database of personas in a `.csv` file to load them at the end of the context. 

### Interact in command line (easier control on data) :smile:
There is a script to interact directly with the model from a simple python script
```
python summsc_chat.py
```

Or if you want to use BB2 model directly without search engine
```
python blender_chat.py
```

You may have to delete models in ParlAI/data if there are some downloading issues.

### Interact with Gradio API :grin:

To use the gradio chat interface launch the server on the remote machine with
```
python summsc_gradio.py
```

Or for the BB2 model
```
python blender_gradio.py
```

You can now access to the public server just by clicking the link. 

You can also use localhost: On your local machine create a tunnel to this server with 

```
ssh -L local_port:remote_server_ip:remote_port remote_name@remote_host
ssh -L 8080:192.168.10.40:7860 remote_name@remote_host
```
The local terminal will ask you your ID for your remote account and you can now access to the chatbot API on `localhost:8080`.