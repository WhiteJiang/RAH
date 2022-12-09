#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python run.py --dataset aircraft --root /storage/jx_data/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 12,24 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info 'Aircraft-SA-FBSM-V2-CRA0.5-V2-CA' --momen 0.91