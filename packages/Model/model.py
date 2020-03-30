from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import os


def model_setup(X, y):
    """
    Setting up the model (layers and compiler) and fitting it
    :param X: Training data set to train the model.
    :param y: Label corresponding to each element in the training data set.
    :return: model instance ready to predict new data.
    """

    # Model's name
    NAME = f"Malaria-CNN-{int(time.time())}"
    print(f"Model's name: {NAME}")

    # Model instantiation and layers definition
    model = Sequential()

    # Adding first 2D convolution layer, activation relu
    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:], activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding second 2D convolution layer
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding third 2D convolution layer
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding flattened dense layer
    model.add(Flatten())

    # Adding output layer, 1 unit, activation sigmoid
    model.add(Dense(1, activation="sigmoid"))

    # Setting up tensorboard logs
    tb = TensorBoard(log_dir=f"data/logs/{NAME}")

    # Model parameters
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Model Summary
    model.summary()

    # Model fitting
    model.fit(X, y, batch_size=32, epochs=10, validation_split=0.2, callbacks=[tb])

    return model


def model_performance(model, X, y):
    """
    Prints model performance. Whether trained or loaded.
    :param model: Model to check
    :param X: Training data set
    :param y: Corresponding label
    :return: Prints model performance
    """
    loss, acc = model.evaluate(X, y, verbose=1)
    print(f"Loss: {round(loss, 2)}, Accuracy: {round(acc, 2)}")


def save_model_h5(model, model_path="data/model", model_name="model"):
    """
    Saves model to a .h5 file
    :param model: Model to be saved as a .h5 file in the specified path
    :param model_path: Path where the .h5 file model will be saved to. Default: ../data/model
    :param model_name: Model's name. Default: model
    :return: Saves the model to the specified path under the specified name
    """
    model.save(os.path.join(model_path, f"{model_name}.h5"))
    print(f"Saved model to {model_path}/{model_name}")


def load_model_h5(model_path="data/model", model_name="model"):
    """
    Loads a previously saved model in a .h5 file
    :param model_path: Path where the .h5 file model will be read from. Default: ../data/model
    :param model_name: Model's name. Default: model
    :return: Loaded model
    """
    try:
        # Loading H5 file
        loaded_model = load_model(os.path.join(model_path, f"{model_name}.h5"))
        print(f"Model loaded successfully -> {model_name}.h5")
        return loaded_model
    except Exception as e:
        print("Model couldn't be loaded")
        exit()


def prediction(model, image_path, categories, pixels=130):
    """
    Analyses new cell images given a loaded or just trained model
    :param model: Model to use
    :param image_path: Image to analyse
    :param categories: List. Defined previously
    :param pixels: Image dimension. Default 130
    :return: Shows up the original cell image with a caption, Parasitized or Uninfected
    """
    try:
        img = cv.imread(image_path)
        # test_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        resized_img = cv.resize(img, (pixels, pixels))
        print("Image loaded successfully")
    except Exception as e:
        print(f"Error reading file {image_path}")
        exit()

    to_predict = np.array(resized_img).reshape(-1, pixels, pixels, 3)  # 1 grayscale, 3 colored images
    y = model.predict(to_predict)

    # Display results
    result = categories[int(y[0][0])]
    print(f"The following image: {image_path} was categorized as: {result}")

    fig, ax = plt.subplots()
    label_font = {"fontname": "Arial", "fontsize": 12}
    plt.imshow(img)
    fig.suptitle(result, fontsize=18)
    ax.set_title(image_path, fontdict=label_font)
    # plt.show()

    return img, result, image_path
