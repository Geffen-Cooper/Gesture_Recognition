#!/bin/sh

python train.py --loss=CE --opt=Adam --log_name=Baseline --batch-size=32 --epochs=100 --lr=0.001 --ese=5