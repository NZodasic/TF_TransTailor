# TF_TransTailor


This repo is TensorFlow version of this repo: https://github.com/locth/TransTailor


### 1. How to Run the Script Correctly
You need to provide the root directory when running the script. This directory will be used to store checkpoints and logs. 

```bash
python tf_trans_tailor.py --root /path/to/your/output/directory
```
==> Short form:
```bash
python tf_trans_tailor.py -r /path/to/your/output/directory
```

### 2. Additional Optional Arguments
The script also accepts these optional arguments:
```
-c or --checkpoint: Path to a checkpoint file if you want to resume training from a previous state
-b or --batchsize: Batch size for training (defaults to 32)
```

For example:
```bash
python tf_trans_tailor.py -r /path/to/output -b 64 -c /path/to/checkpoint.h5
```

### 3. What the Script Does
This script is implementing a neural network model pruning technique with TensorFlow.
1. Loads a pre-trained ResNet50 model
2. Fine-tunes it on the CIFAR-10 dataset
3. Uses scaling factors to identify and prune less important filters in the convolutional layers
4. Performs importance-aware fine-tuning after pruning
5. Continues pruning until the accuracy drops below a threshold
