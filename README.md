# TF_TransTailor


This repo is TensorFlow version of this repo: https://github.com/locth/TransTailor


### 1. How to Run the Script Correctly
You need to provide the root directory when running the script. This directory will be used to store checkpoints and logs. 

```bash
python TransTailor.py --root ./output_directory --batchsize 64 --logdir logs
```

### 2. Tensorboard

```bash
tensorboard --logdir logs


### 3. What the Script Does
This script is implementing a neural network model pruning technique with TensorFlow.
1. Loads a pre-trained ResNet50 model
2. Fine-tunes it on the CIFAR-10 dataset
3. Uses scaling factors to identify and prune less important filters in the convolutional layers
4. Performs importance-aware fine-tuning after pruning
5. Continues pruning until the accuracy drops below a threshold
