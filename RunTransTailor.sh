#!/bin/sh

TransTailor_Path=$1

python $TransTailor_Path/TransTailor.py --root ./output_directory --batchsize 128 --logdir logs
