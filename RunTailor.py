#!/bin/sh

TransTailor_Path=$1

python $TransTailor_Path/TransTailorV3.py --root $1 --numworker 8 --batchsize 32