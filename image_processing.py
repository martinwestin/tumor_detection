import PIL
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os
import sqlite3


IMAGE_HEIGHT, IMAGE_WIDTH = (32, 32)
DATA_GEN = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)


class DBModels:
    """
    store temporary paths to hold images together with
    their id (autoincrement), in order to organize
    given images.
    """
    def __init__(self):
        self.con = sqlite3.connect("data.db", check_same_thread=False)
        self.cur = self.con.cursor()

        self.cur.execute("""CREATE TABLE IF NOT EXISTS images (
            id integer primary key autoincrement
        )""")

    def add_image(self, img_data):
        self.cur.execute("INSERT INTO images VALUES (?)", (None,))
        self.con.commit()
        with open(os.path.join("mod_images", f"{self.cur.lastrowid}.png"), "wb") as f:
            f.write(img_data)

        return self.cur.lastrowid

    def remove(self, img_id):
        self.cur.execute("DELETE FROM images WHERE id = (?)", (img_id,))
        self.con.commit()
        os.remove(os.path.join("mod_images", f"{img_id}.png"))


class ImageVariations:
    def __init__(self, img: np.ndarray):
        self.img = image.img_to_array(img)  # convert image to numpy array
        self.img = self.img.reshape((1,) + self.img.shape)  # reshape image
        self.images = []
        self.index = 0

    def variate_image(self, augmentations):
        for batch in DATA_GEN.flow(self.img, save_prefix="test", save_format="jpeg"):
            self.images.append(image.img_to_array(batch[0]))
            self.index += 1

            if self.index == augmentations:
                break

        return np.array(self.images)


class Img:
    def __init__(self, path: str, dimensions: tuple):
        self.path = path
        self.width, self.height = dimensions

    def resize(self):
        img = Image.open(self.path)
        img = img.resize((self.width, self.height))
        img.save(self.path)

    def get_grayscale_values(self):
        img = Image.open(self.path)
        img = img.convert("LA")
        pixel_values = np.array(list(img.getdata()))
        pixel_values = np.invert(np.array(list(map(lambda x: list(x)[0], pixel_values))))
        pixel_values = pixel_values.reshape(IMAGE_WIDTH, IMAGE_HEIGHT) / 255
        return pixel_values

    @staticmethod
    def fetch_path(img_id):
        return os.path.join("mod_images", f"{img_id}.png")


CLASS_NAMES = {"glioma_tumor": "gt", "meningioma_tumor": "mt", "no_tumor": "nt", "pituitary_tumor": "pt"}
RAW_DIRECTORIES = [os.path.join("data", "Testing"), os.path.join("data", "Training")]


def rename_images():
    for directory in RAW_DIRECTORIES:
        for sub_dir in CLASS_NAMES:
            index = 1
            for image in os.listdir(os.path.join(directory, sub_dir)):
                try:
                    path = os.path.join(directory, sub_dir, image)
                    new_name = os.path.join(directory, sub_dir, f"{CLASS_NAMES[sub_dir]}{index}.png")
                    os.rename(path, new_name)
                    index += 1
                except PIL.UnidentifiedImageError:
                    os.remove(os.path.join(directory, sub_dir, image))


def resize_images():
    for directory in RAW_DIRECTORIES:
        for sub_dir in CLASS_NAMES:
            for file in os.listdir(os.path.join(directory, sub_dir)):
                try:
                    path = os.path.join(directory, sub_dir, file)
                    img = Image.open(path)
                    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                    img.save(path)
                except PIL.UnidentifiedImageError:
                    os.remove(os.path.join(directory, sub_dir, file))
