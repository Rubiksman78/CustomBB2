#!/bin/bash

pip install -r requirements.txt
cd ParlAI
python setup.py develop
pip install -r requirements.txt
apt install libopenblas-base libomp-dev #if not already installed