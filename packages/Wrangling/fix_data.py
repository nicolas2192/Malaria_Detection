import numpy as np
import tensorflow as tf
import random


def shuffle_data(data):
    """
    Shuffling the data to prevent over-fitting
    :param data: List. Data coming from the load_data function or loaded from a binary file.
    :return: Same list but shuffled.
    """
    random.shuffle(data)
    print("Data shuffled")
    return data


def prep_data(data, pixels=130):
    """
    Splitting data and turning it into arrays, X will be the training set while y it corresponding label
    :param data: Shuffled data
    :param pixels: image dimension, default 130
    :return: Returns X training data in a keras array format and y in a numpy array format.
        Both variables are ready to train the CNN model.
    """
    # Splitting training data and label
    X, y = [], []
    for image, category in data:
        X.append(image)
        y.append(category)
    print("Data was split into training set and label")

    # Turning lists into arrays
    X = np.array(X).reshape(-1, pixels, pixels, 3)  # 1 grayscale, 3 colored images
    y = np.array(y)
    print("Data was turned into a keras readable array format")

    # Normalizing training data, X
    X = tf.keras.utils.normalize(X, axis=1)
    print("Training data was normalized")

    print(f"Training set X shape: {X.shape}")
    print(f"Label y size: {y.shape}")

    return X, y
