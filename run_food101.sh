#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python run.py --dataset food101 --root /storage/jx_data/food101 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 10,25 --num-samples 2000 --info 'Food101-SA-FBSM-V2-CRA0.5-V2-CA-CLS-GL' --momen 0.91
