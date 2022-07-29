#!/bin/bash

python -m parlai train_model --mf zoo:zoo:blenderbot2/blenderbot2_3B/model --model transformer/generator --task msc --model-file 'tmp/model' --batchsize 32