import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import data
import image_processing


IMAGE_HEIGHT, IMAGE_WIDTH = (32, 32)

x, y = data.load_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
class_names = list(image_processing.CLASS_NAMES.keys())

y_train = np.array(list(map(lambda x: class_names.index(x), y_train)))
y_test = np.array(list(map(lambda x: class_names.index(x), y_test)))
model = None

x_test = x_test.reshape(x_test.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, 1)
print(x_test.shape)


def process_data():
    global x_train, y_train
    variate_train_images = data.modify_images(x_train, y_train)
    x_train, y_train = variate_train_images[0], variate_train_images[1]
    print(x_train.shape, y_train.shape)


def new_model():
    global model
    model = tf.keras.models.Sequential()

    # convolutional layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))

    # dense layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(10))

    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    model.save("model.h5")
    return history.history["accuracy"]


def ask_retrain_model():
    global model
    try:
        model = tf.keras.models.load_model("model.h5")
        print("Model found")
        re_train = input("Would you like to train the model again (Y for yes, anything else for no)? ").lower() == "y"
        if re_train:
            process_data()
            new_model()

    except IOError:
        print("No model found, beginning to train...")
        process_data()
        new_model()
        model = tf.keras.models.load_model("model.h5")


if __name__ == '__main__':
    ask_retrain_model()

    predicted = model.predict(x_test)

    for i in range(10):
        plt.figure()
        plt.imshow(x_test[i], cmap=plt.cm.binary)
        plt.title(f"Predicted: {class_names[np.argmax(predicted[i])]}")
        plt.xlabel(f"Actual: {class_names[y_test[i]]}")

    plt.show()
