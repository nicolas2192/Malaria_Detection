import cv2 as cv
import os
import pickle


def load_data(directory, categories, pixels):
    """
    Loads each image and stores it ina list, where the first element will be the image 8(array) and the second
        its corresponding label (Parasitized or Uninfected)
    :param directory: Path where images are located.
    :param categories: List, each element will be a label. Parasitized or Uninfected.
    :param pixels: All images must have the same dimensions. 130x130 was determined to be the most sensible option
    :return: List of all images. Each element is comprised by another list where the first element is the image (array)
        and the second one it corresponding label (Parasitized is 1 and Uninfected 0)
    """

    data = []
    for category in categories:
        print(f"Loading data, category: {category}")
        path = os.path.join(directory, category)

        # Checking if there are files to load. Checking path.
        if not os.path.isdir(os.path.join(directory, category)):
            print(f"The following directory could not be found: {path}")
            exit()
        else:
            files = os.listdir(path)

        # for i in range(len(files) - 1): # Uncomment this line to load all available images.
        for i in range(3000):
            # Skip corrupt images
            try:
                img = cv.imread(os.path.join(path, files[i]))  # read image into an array
                new_img = cv.resize(img, (pixels, pixels))  # images must be alike when it comes to dimensions
                data.append([new_img, categories.index(category)])
            except:
                pass

    if len(categories) == 2:
        clean = [x[1] for x in data]
        print(f"{len(clean)} samples loaded.\
        {categories[0]}: {len(clean ) -sum(clean)} and {categories[1]}: {sum(clean)}")
    else:
        print(f"{len(data)} samples loaded.")
    return data


def save_binary(data, path):
    """
    Saving data into a binary file
    :param data: List generated using the load_data function.
    :param path: Path where the binary file will be saved to.
    :return: Saved binary file
    """
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Data binary file saved to {path}")


def load_binary(path):
    """
    Loading data from a binary file
    :param path: Path where the binary file will be read from.
    :return: Loaded binary file
    """
    try:
        with open(path, "rb") as f:
            loaded_data = pickle.load(f)
        print(f"Data binary file loaded from {path}")
        return loaded_data
    except Exception as e:
        print("Binary file couldn't be loaded.")
