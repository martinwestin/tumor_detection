from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import tensorflow as tf
import image_processing
import numpy as np


model = tf.keras.models.load_model("model.h5")
db_models = image_processing.DBModels()
class_names = list(image_processing.CLASS_NAMES.keys())


app = Flask(__name__)
socketio = SocketIO(app)


def classify_new_image(path):
    image = image_processing.Img(path, (image_processing.IMAGE_WIDTH, image_processing.IMAGE_HEIGHT))
    image.resize()

    pixel_values = np.array([image.get_grayscale_values()])
    pixel_values = pixel_values.reshape(pixel_values.shape[0], image_processing.IMAGE_WIDTH, image_processing.IMAGE_HEIGHT, 1)
    new_prediction = model.predict(pixel_values)
    return class_names[np.argmax(new_prediction[0])]


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("classify_image")
def classify_image(msg):
    img = msg["data"]
    img_id = db_models.add_image(img)
    rv = classify_new_image(image_processing.Img.fetch_path(img_id))
    emit("classification_response", {"result": " ".join(rv.split("_")).title()}, room=request.sid)
    db_models.remove(img_id)


if __name__ == '__main__':
    socketio.run(app, debug=True)
