import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tf_pruner import Prunerz

import numpy as np
import argparse
import os
import logging
import time
import matplotlib.pyplot as plt

# Thiết lập logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TEST_NAME = "TA5_IA10_DROP5_ResNet50_TF"

TA_EPOCH = 5

# Old LR 0.005
TA_LR = 0.001
TA_MOMENTUM = 0.9

IA_EPOCH = 10

# Old LR 0.005
IA_LR = 0.001
IA_MOMENTUM = 0.9

ACC_DROP = 5
PRUNING_PERCENTAGE = 5  # The percentage of least important filters that need to be pruned

def LoadModel():
    """Load the ResNet50 model with weights pre-trained on ImageNet"""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    return model

def load_data(batch_size):
    """Load CIFAR10 dataset and preprocess it for ResNet50"""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Preprocess data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # Normalize
    mean = np.mean(x_train, axis=(0, 1, 2))
    std = np.std(x_train, axis=(0, 1, 2))
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    # One-hot encode targets
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Split train into train and validation
    val_size = int(0.2 * x_train.shape[0])
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]
    
    # Create data generators with resizing
    train_datagen = ImageDataGenerator(
        preprocessing_function=lambda x: tf.image.resize(x, [224, 224]).numpy()
    )
    val_datagen = ImageDataGenerator(
        preprocessing_function=lambda x: tf.image.resize(x, [224, 224]).numpy()
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=lambda x: tf.image.resize(x, [224, 224]).numpy()
    )
    
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)
    test_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size)
    
    return train_generator, val_generator, test_generator

def LoadArguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Config cli params")
    parser.add_argument("-r","--root", help="Root directory")
    parser.add_argument("-c","--checkpoint", default="", help="Checkpoint path")
    parser.add_argument("-n", "--numworker", default=1, help="Number of worker")
    parser.add_argument("-b", "--batchsize", default=32, help="Batch size")

    args = parser.parse_args()
    ROOT_DIR = args.root
    CHECKPOINT_PATH = args.checkpoint
    NUM_WORKER = int(args.numworker)
    BATCH_SIZE = int(args.batchsize)

    return ROOT_DIR, CHECKPOINT_PATH, NUM_WORKER, BATCH_SIZE

def calculate_accuracy(model, data_generator):
    """Calculate the accuracy of the model on a given dataset"""
    steps = len(data_generator)
    correct = 0
    total = 0
    
    for i in range(steps):
        x_batch, y_batch = data_generator.next()
        predictions = model.predict(x_batch)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_batch, axis=1)
        
        total += len(true_classes)
        correct += np.sum(predicted_classes == true_classes)
    
    accuracy = 100 * correct / total
    return accuracy

def time_log():
    """Log current time"""
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    print("Time log:", curr_time)

if __name__ == "__main__":
    # LOAD ARGUMENTS
    logger.info("START MAIN PROGRAM!")
    ROOT_DIR, CHECKPOINT_PATH, BATCH_SIZE = load_arguments()
    
    RESULT_PATH = os.path.join(ROOT_DIR, "checkpoint", "optimal", TEST_NAME + "_optimal_model")
    SAVED_PATH = os.path.join(ROOT_DIR, "checkpoint", "pruner", TEST_NAME + "_checkpoint_{pruned_count}.h5")
    FIG_PATH = os.path.join(ROOT_DIR, "train_log", TEST_NAME + ".png")
    
    # ENSURE DIRECTORIES EXIST
    os.makedirs(os.path.join(ROOT_DIR, "checkpoint", "optimal"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "checkpoint", "pruner"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "train_log"), exist_ok=True)
    
    # CHECK FOR GPU
    logger.info("GET DEVICE INFORMATION")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"DEVICE: {gpus[0].name}")
        # Enable memory growth to prevent allocation errors
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.info("DEVICE: CPU")
    
    # LOAD DATASET
    logger.info("LOAD DATASET: CIFAR10")
    train_generator, val_generator, test_generator = load_data(BATCH_SIZE)
    
    # LOAD MODEL
    logger.info("LOAD PRETRAINED MODEL: ResNet50 (ImageNet)")
    model = load_model()
    
    # Compile model with SGD optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=TA_LR, momentum=TA_MOMENTUM)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # INIT PRUNING SCHEME
    pruner = Pruner(model, train_generator, val_generator, test_generator)
    
    if os.path.isfile(CHECKPOINT_PATH):
        logger.info("Load model and pruning info from checkpoint...")
        pruner.load_state(CHECKPOINT_PATH)
    else:
        logger.info("Finetuning the model...")
        pruner.finetune(40, TA_LR, TA_MOMENTUM, 0)
        
        logger.info("Initializing scaling factors...")
        pruner.init_scaling_factors()
        pruner.save_state(SAVED_PATH.format(pruned_count=0))
    
    # Evaluate original model
    opt_accuracy = calculate_accuracy(pruner.model, test_generator)
    print(f"Accuracy of finetuned model: {opt_accuracy:.2f}%")
    logger.info(f"Accuracy of finetuned model: {opt_accuracy:.2f}%")
    logger.info("===DONE EVALUATE===")
    
    # START PRUNING PROCESS
    while True:
        time_log()
        logger.info("Training scaling factors...")
        pruner.train_scaling_factors(IA_EPOCH, IA_LR, IA_MOMENTUM)
        
        time_log()
        logger.info("Generating importance scores...")
        pruner.generate_importance_scores()
        
        time_log()
        logger.info("Finding filters to prune...")
        filters_to_prune = pruner.find_filters_to_prune(PRUNING_PERCENTAGE)
        
        time_log()
        logger.info("Pruning and restructuring model...")
        pruner.prune_and_restructure(filters_to_prune)
        
        time_log()
        logger.info("Pruning scaling factors...")
        pruner.prune_scaling_factors(filters_to_prune)
        
        time_log()
        logger.info("Pruning importance scores...")
        pruner.prune_importance_scores(filters_to_prune)
        
        time_log()
        logger.info("Performing importance-aware fine-tuning...")
        pruner.importance_aware_fine_tuning(IA_EPOCH, IA_LR, IA_MOMENTUM)
        
        # Count pruned filters
        sum_filters = 0
        for layer in filters_to_prune:
            number_of_filters = len(filters_to_prune[layer])
            sum_filters += number_of_filters
        
        print(f"===Number of pruned filters is: {sum_filters}")
        logger.info(f"===Number of pruned filters is: {sum_filters}")
        
        pruned_count = len(pruner.pruned_filters)
        
        if pruned_count % 5 == 0:
            pruner.save_state(SAVED_PATH.format(pruned_count=pruned_count))
        
        time_log()
        logger.info("Fine-tuning pruned model...")
        pruner.finetune(TA_EPOCH, TA_LR, TA_MOMENTUM, 0)
        
        time_log()
        pruned_accuracy = calculate_accuracy(pruner.model, test_generator)
        
        print(f"Accuracy of pruned model: {pruned_accuracy:.2f}%")
        logger.info(f"Accuracy of pruned model: {pruned_accuracy:.2f}%")
        
        # Check if accuracy drop exceeds threshold
        if abs(opt_accuracy - pruned_accuracy) > ACC_DROP:
            print("Optimization done!")
            pruner.model.save(RESULT_PATH)
            break
        else:
            print("Update optimal model")
            pruner.plot_losses(FIG_PATH)