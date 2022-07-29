#!/bin/bash

conda activate chatvenv
cd ParlAI/
python parlai/chat_service/services/browser_chat/run.py --config-path config.yml
python parlai/chat_service/services/browser_chat/client.py --host 0.0.0.0