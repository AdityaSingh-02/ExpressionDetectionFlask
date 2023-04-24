from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import numpy as np
import cv2
import tensorflow as tf
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import tensorflow_hub as hub
import base64
import io

def load_model(model_path):
  print(f"Loading Saved Model From: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects = {"KerasLayer": hub.KerasLayer})
  return model

model1 = load_model("brain.h5")
model2 = load_model("pneumonia.h5")


pic_size = 224

app = Flask(__name__)



# Define prediction route
# @app.route('/api/tumor/<path:img_url>', methods=['POST', 'GET'])
# def tumor(img_url):
#     fetched_Image = img_url
#     response = requests.get(fetched_Image)
#     image = Image.open(BytesIO(response.content))
#     if response.status_code == 200:
#         image = image.resize((224, 224))
#         image = tf.keras.preprocessing.image.img_to_array(image)
#         image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

#         prediction = model1.predict(tf.expand_dims(image, axis=0))
#         expression = ['no_tumor', 'meningioma_tumor','Glioma_tumor','pituitary_tumor' ][prediction.argmax()]
#         return jsonify({'expression': expression})
#     else:
#         return "Failure"


# @app.route('/api/pneumonia/<path:img_url>', methods=['POST', 'GET'])
# def pneumonia(img_url):
#     fetched_Image = img_url
#     response = requests.get(fetched_Image)
#     image = Image.open(BytesIO(response.content))
#     if response.status_code == 200:
#         image = image.resize((224, 224))
#         image = tf.keras.preprocessing.image.img_to_array(image)
#         image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

#         prediction = model2.predict(tf.expand_dims(image, axis=0))
#         expression = ['no pneumonia','pneumonia'][prediction.argmax()]
#         return jsonify({'expression': expression})
#     else:
#         return "Failure"

@app.route('/uploads', methods=['POST'])
def uploads():
    data = request.get_json()
    image_str = data['image_rec']
    num_padding_chars = 4 - len(image_str) % 4
    image_str += "=" * num_padding_chars
    image_bytes = base64.b64decode(image_str)
    image = Image.open(io.BytesIO(image_bytes))
    # image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    prediction = model2.predict(tf.expand_dims(image, axis=0))
    expression = ['no pneumonia','pneumonia'][prediction.argmax()]
    return jsonify({'expression': expression})

# Define home page route
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

