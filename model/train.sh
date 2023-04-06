#!/bin/sh

python train.py --loss=CE --opt=Adam --log_name=top_view --batch-size=32 --epochs=200 --lr=0.001 --ese=20 --root_dir=/home/gc28692/Projects/data/nvgesture/nvGesture_v1