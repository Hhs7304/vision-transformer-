import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Flatten the labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return (X_train, y_train), (X_test, y_test)

# Define class names
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
