# Tumor Detection
A brain tumor detection using a convolutional neural network with Tensorflow in Python. 
This is a program that can classify CT scan images of the brain into one of the following categories:
Meningioma tumor, Glioma tumor, Pituitary tumor or No tumor. 
#
# Folders and files
The "data" folder contains a couple
thousand images of CT scan images to train the model on. The file "model.h5" is the pretrained model
that can be used to classify new images, so don't change this one. In order to retrain the model or
change it, run file "model.py".
#
The "website.py" file is the file responsable for the web page itself, which makes it easy for
users to interact and use the model. If you decide to use this model to clasify your own images,
remember that it is not 100% accurate and don't use the results as medical advice.
# About the model
The model is, as previously stated, a convolutional neural network using Tensorflow, and it
consists of 3 tf.keras.layers.Conv2D layers, 2 tf.keras.layers.MaxPooling2D layers, 1 tf.keras.layers.Flatten layer,
and 2 tf.keras.layers.Dense layers. When I have trained the model myself, I have reached about 82% accuracy on
test samples.
