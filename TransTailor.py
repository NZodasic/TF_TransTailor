import tensorflow as tf
import numpy as np
import os
import logging
import time
import argparse
import matplotlib.pyplot as plt
from Pruner import Pruner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Constants
TEST_NAME = "TF_ResNet50_CIFAR10_TA5_IA10_DROP5"

# Training parameters
TA_EPOCH = 5  # Number of epochs for TA fine-tuning
TA_LR = 0.001  # Learning rate for TA
TA_MOMENTUM = 0.9  # Momentum for TA

IA_EPOCH = 10  # Number of epochs for IA fine-tuning
IA_LR = 0.001  # Learning rate for IA
IA_MOMENTUM = 0.9  # Momentum for IA

ACC_DROP = 5  # Maximum acceptable accuracy drop (percentage)
PRUNING_PERCENTAGE = 5  # Percentage of filters to prune in each iteration

def load_model():
    """
    Load a pre-trained ResNet50 model and adapt it for CIFAR-10
    
    Returns:
        A TensorFlow Keras ResNet50 model adapted for CIFAR-10
    """
    # Create a ResNet50 model with input shape for CIFAR-10
    input_shape = (32, 32, 3)  # CIFAR-10 image size with 3 channels
    
    # Build ResNet50 without top layer, with CIFAR-10 input shape
    base_model = tf.keras.applications.ResNet50(
        include_top=False, 
        weights='imagenet', 
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Add custom classification head for CIFAR-10
    x = base_model.output
    predictions = tf.keras.layers.Dense(10)(x)  # No activation - outputs raw logits
    
    # Create the full model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    
    return model

def load_data(batch_size):
    """
    Load and preprocess the CIFAR-10 dataset
    
    Args:
        batch_size: Batch size for datasets
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Convert to float32 and normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Preprocess inputs for ResNet50
    x_train = tf.keras.applications.resnet50.preprocess_input(x_train * 255)
    x_test = tf.keras.applications.resnet50.preprocess_input(x_test * 255)
    
    # Convert targets to single dimension and one-hot
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    
    # Split train into train and validation
    val_size = int(0.2 * len(x_train))
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]
    
    # Create TF datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, test_dataset

def load_arguments():
    """
    Parse command-line arguments
    
    Returns:
        root_dir, checkpoint_path, batch_size
    """
    parser = argparse.ArgumentParser(description="Config cli params")
    parser.add_argument("-r", "--root", help="Root directory")
    parser.add_argument("-c", "--checkpoint", default="", help="Checkpoint path")
    parser.add_argument("-b", "--batchsize", default=64, help="Batch size")
    parser.add_argument("--logdir", default="logs", help="TensorBoard log directory")

    args = parser.parse_args()
    root_dir = args.root
    checkpoint_path = args.checkpoint
    batch_size = int(args.batchsize)
    
    return root_dir, checkpoint_path, batch_size, args.logdir

def calculate_accuracy(model, dataset):
    """
    Calculate accuracy of model on dataset
    
    Args:
        model: TensorFlow model
        dataset: TensorFlow dataset with images and labels
        
    Returns:
        Accuracy as a percentage
    """
    correct = 0
    total = 0
    
    for images, labels in dataset:
        predictions = model(images, training=False)
        predicted_classes = tf.argmax(predictions, axis=1)
        correct += tf.reduce_sum(tf.cast(tf.equal(predicted_classes, labels), tf.float32))
        total += len(labels)
    
    accuracy = 100 * correct / total
    return accuracy.numpy()

def time_log():
    """Log current time"""
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    logger.info(f"Time log: {curr_time}")

if __name__ == "__main__":
    # Load arguments
    logger.info("START MAIN PROGRAM!")
    ROOT_DIR, CHECKPOINT_PATH, BATCH_SIZE, LOG_DIR = load_arguments()
    
    # Setup paths
    RESULT_PATH = os.path.join(ROOT_DIR, "checkpoint", "optimal", f"{TEST_NAME}_optimal_model")
    SAVED_PATH = os.path.join(ROOT_DIR, "checkpoint", "pruner", f"{TEST_NAME}_checkpoint_")
    FIG_PATH = os.path.join(ROOT_DIR, "train_log", f"{TEST_NAME}.png")
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(ROOT_DIR, "checkpoint", "optimal"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "checkpoint", "pruner"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "train_log"), exist_ok=True)
    
    # Set memory growth for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Using GPU: {len(gpus)} GPUs available")
        except RuntimeError as e:
            logger.error(f"Error setting memory growth: {e}")
    
    # Load data
    logger.info("LOADING CIFAR-10 DATASET")
    train_dataset, val_dataset, test_dataset = load_data(BATCH_SIZE)
    
    # Load model
    logger.info("LOADING PRE-TRAINED RESNET50 MODEL")
    model = load_model()
    
    # Initialize pruner
    pruner = Pruner(model, train_dataset, val_dataset, test_dataset)
    
    print("Writing test log")
    test_writer = tf.summary.create_file_writer("logs/test")
    with test_writer.as_default():
        tf.summary.scalar("dummy_scalar", 0.5, step=0)
    test_writer.flush()
    
    # Load from checkpoint or train initial model
    if os.path.isfile(CHECKPOINT_PATH):
        logger.info("Loading model and pruning info from checkpoint...")
        pruner.load_state(CHECKPOINT_PATH)
    else:
        logger.info("Fine-tuning initial model...")
        pruner.finetune(10, TA_LR, TA_MOMENTUM, 0)
        
        logger.info("Initializing scaling factors...")
        pruner.init_scaling_factors()
        pruner.save_state(f"{SAVED_PATH}0")
    
    # Evaluate initial model
    opt_accuracy = calculate_accuracy(pruner.model, test_dataset)
    logger.info(f"Accuracy of finetuned model: {opt_accuracy:.2f}%")
    
    # Start pruning process
    iteration = 1
    while True:
        logger.info(f"STARTING PRUNING ITERATION {iteration}")
        time_log()
        
        # Train scaling factors
        logger.info("Training scaling factors...")
        pruner.train_scaling_factors(TA_EPOCH, TA_LR, TA_MOMENTUM, log_dir=LOG_DIR)

        # Generate importance scores
        logger.info("Generating importance scores...")
        pruner.generate_importance_scores()
        
        # Find filters to prune
        logger.info(f"Finding {PRUNING_PERCENTAGE}% of filters to prune...")
        filters_to_prune = pruner.find_filters_to_prune(PRUNING_PERCENTAGE)
        
        # Prune model
        logger.info("Pruning filters and restructuring model...")
        pruner.prune_and_restructure(filters_to_prune)
        
        # Update scaling factors and importance scores
        pruner.prune_scaling_factors(filters_to_prune)
        pruner.prune_importance_scores(filters_to_prune)
        
        # Fine-tuning with importance-aware gradients
        logger.info("Fine-tuning with importance-aware gradients...")
        pruner.importance_aware_fine_tuning(IA_EPOCH, IA_LR, IA_MOMENTUM, log_dir=LOG_DIR)
        
        # Regular fine-tuning
        logger.info("Regular fine-tuning...")
        pruner.finetune(TA_EPOCH, TA_LR, TA_MOMENTUM, 0, log_dir=LOG_DIR)
        
        # Evaluate pruned model
        curr_accuracy = calculate_accuracy(pruner.model, test_dataset)
        logger.info(f"Accuracy after pruning iteration {iteration}: {curr_accuracy:.2f}%")
        
        # Save current state
        pruner.save_state(f"{SAVED_PATH}{iteration}")
        
        # Check if accuracy drop is acceptable
        accuracy_drop = opt_accuracy - curr_accuracy
        logger.info(f"Accuracy drop: {accuracy_drop:.2f}%")
        
        if accuracy_drop > ACC_DROP:
            logger.info(f"Accuracy drop {accuracy_drop:.2f}% exceeds threshold {ACC_DROP}%")
            logger.info(f"Stopping pruning process after {iteration} iterations")
            
            # Load previous iteration which still meets accuracy requirements
            if iteration > 1:
                logger.info(f"Loading optimal model from iteration {iteration-1}")
                pruner.load_state(f"{SAVED_PATH}{iteration-1}")
                
                # Save as optimal model
                pruner.save_state(RESULT_PATH)
                
                # Generate final plot
                pruner.plot_losses(FIG_PATH)
                
                # Final evaluation
                final_accuracy = calculate_accuracy(pruner.model, test_dataset)
                logger.info(f"Final model accuracy: {final_accuracy:.2f}%")
            break
        
        iteration += 1
        
    logger.info("TRAINING COMPLETE!")
    time_log()