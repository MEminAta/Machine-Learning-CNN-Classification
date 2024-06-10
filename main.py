import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2

# Modeli yükle
model = tf.keras.models.load_model('model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    print(prediction)
    if(prediction[0][0] > prediction[0][1]):
        print("kedi")
    else:
        print("köpek")

predict_image("data/IMG_4297.png")