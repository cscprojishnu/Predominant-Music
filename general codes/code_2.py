import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import image_dataset_from_directory

# Set seed for reproducibility
SEED = 42
keras.utils.set_random_seed(SEED)

# DATA
BATCH_SIZE = 128
BUFFER_SIZE = BATCH_SIZE * 2
AUTO = tf.data.AUTOTUNE
IMAGE_SIZE = 48  # Resize input images to this size
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 11  # Number of classes in the dataset

# OPTIMIZER
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# PRETRAINING
EPOCHS = 5

# Path to the extracted database directory
dataset_path = r"D:\DJT\Downloads\PAPER\PAPER 3\database\mel_spectrum_all_train"

# Load datasets from the directory
train_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
)

val_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
)

# Use the same directory or separate test directory if available
test_ds = image_dataset_from_directory(
    r"D:\DJT\Downloads\PAPER\PAPER 3\database\melspectrum_all_test",  # Replace with the test dataset path if separate
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
)

# Optional: Shuffle and prefetch for performance optimization
train_ds = train_ds.shuffle(BUFFER_SIZE).prefetch(AUTO)
val_ds = val_ds.prefetch(AUTO)
test_ds = test_ds.prefetch(AUTO)

def get_preprocessing():
    model = keras.Sequential(
        [
            keras.layers.Rescaling(1 / 255.0),
            keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        ],
        name="preprocessing",
    )
    return model

def get_train_augmentation_model():
    model = keras.Sequential(
        [
            keras.layers.Rescaling(1 / 255.0),
            keras.layers.Resizing(INPUT_SHAPE[0] + 20, INPUT_SHAPE[0] + 20),
            keras.layers.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
            keras.layers.RandomFlip("horizontal"),
        ],
        name="train_data_augmentation",
    )
    return model

def build_convolutional_stem(dimensions):
    """Build the convolutional stem.

    Args:
        dimensions: The embedding dimension of the patches (d in paper).

    Returns:
        The convolutional stem as a keras sequential model.
    """
    config = {
        "kernel_size": (3, 3),
        "strides": (2, 2),
        "activation": tf.nn.gelu,
        "padding": "same",
    }

    convolutional_stem = keras.Sequential(
        [
            keras.layers.Conv2D(filters=dimensions // 2, **config),
            keras.layers.Conv2D(filters=dimensions, **config),
        ],
        name="convolutional_stem",
    )

    return convolutional_stem

class SqueezeExcite(keras.layers.Layer):
    """Applies squeeze and excitation to input feature maps.

    Args:
        ratio: The ratio with which the feature map needs to be reduced in
        the reduction phase.

    Inputs:
        Convolutional features.

    Outputs:
        Attention modified feature maps.
    """

    def __init__(self, ratio, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

    def build(self, input_shape):
        filters = input_shape[-1]
        self.squeeze = keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.reduction = keras.layers.Dense(
            units=filters // self.ratio,
            activation="relu",
            use_bias=False,
        )
        self.excite = keras.layers.Dense(units=filters, activation="sigmoid", use_bias=False)
        self.multiply = keras.layers.Multiply()

    def call(self, x):
        shortcut = x
        x = self.squeeze(x)
        x = self.reduction(x)
        x = self.excite(x)
        x = self.multiply([shortcut, x])
        return x

# Define the Trunk, AttentionPooling, and PatchConvNet classes (similar to the original code)
# These are omitted here for brevity but remain unchanged.

# Assemble preprocessing and training models
train_augmentation_model = get_train_augmentation_model()
preprocessing_model = get_preprocessing()
conv_stem = build_convolutional_stem(dimensions=256)

# Compile and pretrain the model
patch_conv_net = keras.Sequential([
    preprocessing_model,
    train_augmentation_model,
    conv_stem
])

# Assemble the optimizer
optimizer = keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

# Compile the model
patch_conv_net.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)

# Train the model
patch_conv_net = keras.Sequential([
    preprocessing_model,
    train_augmentation_model,
    conv_stem,
    keras.layers.GlobalAveragePooling2D(),  # Pool the spatial dimensions
    keras.layers.Dense(NUM_CLASSES)  # Output the class probabilities
])


patch_conv_net.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)

# Train the model
history = patch_conv_net.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
)

#  Evaluate the model and get final accuracy
evaluation_results = patch_conv_net.evaluate(test_ds)
loss = evaluation_results[0]
accuracy = evaluation_results[1] * 100  # Top-1 accuracy as percentage
top5_accuracy = evaluation_results[2] * 100  # Top-5 accuracy as percentage

print(f"Final Evaluation Results:")
print(f"Loss: {loss:.2f}")
print(f"Top-1 Accuracy: {accuracy:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")