#!/bin/bash

python run.py --dataset vegfru --root /opt/data/private/ML_DATA/vegfru/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 12,24,32,48 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 4000 --info 'VegFru-SEMICON' --momen 0.91