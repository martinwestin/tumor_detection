import os
import PIL
import image_processing
import numpy as np


IMAGE_HEIGHT, IMAGE_WIDTH = (32, 32)


def load_data():
    """
    load all of the data set images and split them into.
    two categories, features and labels.
    :returns: np.array of the features (x) and np.array of the labels (y).
    """
    x = []
    y = []
    for directory in image_processing.RAW_DIRECTORIES:
        for sub_dir in image_processing.CLASS_NAMES:
            for file in os.listdir(os.path.join(directory, sub_dir)):
                try:
                    path = os.path.join(directory, sub_dir, file)
                    img = image_processing.Img(path, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    x.append(img.get_grayscale_values())
                    y.append(sub_dir)

                except PIL.UnidentifiedImageError:
                    os.remove(os.path.join(directory, sub_dir, file))

    return np.array(x), np.array(y)


def modify_images(x: np.ndarray, y: np.ndarray, augmentations: int=15):
    """
    augments given images by turning and modifying them
    in certain ways so that the model will have more data
    to train on. standard augmentation value (the amount) of
    "new" images that this function will return is 10, but
    this can be changed by changing parameter "augmentations".
    :param x: all the features (images) in the form of
    a numpy array.
    :param y: all the labels.
    :param augmentations: the amount of "new" modified images
    to produce.
    :return: a new np.array of all the images (including the "new" ones)
    together with the corresponding labels.
    """
    images = []
    labels = []
    if len(x) == len(y):
        for i in range(len(x)):
            variation = image_processing.ImageVariations(x[i])

            for image in variation.variate_image(augmentations):
                images.append(image)
                labels.append(y[i])

        print("Finished modifying images")
        return np.array(images), np.array(labels)

    raise Exception("make sure that length of x and y is the same. current x length: " + str(len(x)) +
                    " current y length: " + str(len(y)))
