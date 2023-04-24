from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import numpy as np
import cv2
import tensorflow as tf
import requests
from PIL import Image, UnidentifiedImageError,ImageFile
from io import BytesIO
import tensorflow_hub as hub
import base64
import io

ImageFile.LOAD_TRUNCATED_IMAGES = True
def load_model(model_path):
  print(f"Loading Saved Model From: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects = {"KerasLayer": hub.KerasLayer})
  return model

model1 = load_model("brain.h5")
model2 = load_model("pneumonia.h5")


pic_size = 224

app = Flask(__name__)



def prepare_detail(image,model,info):
    image = image.resize((pic_size, pic_size))
    image = np.array(image)
    image = image / 255
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    prediction = model.predict(tf.expand_dims(image, axis=0))
    expression = info[prediction.argmax()]
    return jsonify({'expression': expression})

@app.route('/Xray', methods=['POST'])
def xray():
    data = request.get_json()
    image_str = data['image_rec']
    num_padding_chars = 4 - len(image_str) % 4
    image_str += "=" * num_padding_chars
    image_bytes = base64.b64decode(image_str)
    image = Image.open(io.BytesIO(image_bytes))
    label = ['pneumonia','no pneumonia']
    return prepare_detail(image,model2,label)

@app.route('/Mri', methods=['POST'])
def Mri():
    data = request.get_json()
    image_str = data['image_rec']
    num_padding_chars = 4 - len(image_str) % 4
    image_str += "=" * num_padding_chars
    image_bytes = base64.b64decode(image_str)
    image = Image.open(io.BytesIO(image_bytes))
    label = ['no_tumor', 'meningioma_tumor','Glioma_tumor','pituitary_tumor' ]
    return prepare_detail(image,model1,label)


# Define home page route
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

