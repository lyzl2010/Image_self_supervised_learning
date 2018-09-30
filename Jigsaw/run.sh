#!/bin/bash
pip install pytorch==0.4.0 torchvision
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR
python train_imagenet.py
