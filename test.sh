#!/bin/bash

 python run.py --dataset cub-2011 --root /opt/data/private/ML_DATA/CUB_200_2011 --gpu 0 --arch test --batch-size 16 --code-length 12,24,32,48 --wd 1e-4 --info 'CUB-SEMICON'