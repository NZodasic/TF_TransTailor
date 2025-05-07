# tf_resnet50_cifar10_pruner.py
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import argparse
import time
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TEST_NAME = "TA5_IA10_DROP5_v3"

TA_EPOCH = 5
TA_LR = 0.005

IA_EPOCH = 10
IA_LR = 0.005

ACC_DROP = 5
PRUNING_PERCENTAGE = 5

AUTOTUNE = tf.data.AUTOTUNE


def preprocess(image, label):
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.per_image_standardization(image)
    return image, label


def load_data(batch_size):
    (train_ds, val_ds, test_ds), ds_info = tfds.load(
        "cifar10",
        split=["train[:60%]", "train[60%:80%]", "train[80%:]"],
        as_supervised=True,
        with_info=True,
    )

    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


def load_model():
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg"
    )
    base_model.trainable = True

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model


def calculate_accuracy(model, dataset):
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    for images, labels in dataset:
        preds = model(images, training=False)
        acc.update_state(labels, preds)
    return acc.result().numpy() * 100


def time_log():
    print("Time log:", time.strftime("%H:%M:%S", time.localtime()))


def load_arguments():
    parser = argparse.ArgumentParser(description="Config cli params")
    parser.add_argument("-r", "--root", help="Root directory")
    parser.add_argument("-c", "--checkpoint", default="", help="Checkpoint path")
    parser.add_argument("-b", "--batchsize", default=32, type=int, help="Batch size")
    args = parser.parse_args()
    return args.root, args.checkpoint, args.batchsize


if __name__ == "__main__":
    logger.info("START MAIN PROGRAM!")
    ROOT_DIR, CHECKPOINT_PATH, BATCH_SIZE = load_arguments()

    RESULT_PATH = os.path.join(ROOT_DIR, "checkpoint", "optimal", TEST_NAME + "_optimal_model.h5")
    FIG_PATH = os.path.join(ROOT_DIR, "train_log", TEST_NAME + ".png")

    logger.info("LOAD DATASET: CIFAR10")
    train_ds, val_ds, test_ds = load_data(BATCH_SIZE)

    logger.info("LOAD MODEL: ResNet50")
    model = load_model()

    optimizer = tf.keras.optimizers.SGD(learning_rate=TA_LR, momentum=0.9)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_ds, epochs=TA_EPOCH, validation_data=val_ds)

    opt_accuracy = calculate_accuracy(model, test_ds)
    print(f"Accuracy of finetuned model: {opt_accuracy:.2f}%")

    model.save(RESULT_PATH)
    logger.info("Saved optimal model")

    # PRUNING STEP PLACEHOLDER
    # The actual pruning logic similar to PyTorch's Pruner class must be implemented manually or via TensorFlow Model Optimization Toolkit.

    logger.info("===DONE EVALUATE===")
