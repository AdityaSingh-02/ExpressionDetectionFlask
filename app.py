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
model3 = load_model("skin.h5")
model4 = load_model("retina.h5")


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


@app.route('/Infection', methods=['POST'])
def infection():
    data = request.get_json()
    image_str = data['image_rec']
    num_padding_chars = 4 - len(image_str) % 4
    image_str += "=" * num_padding_chars
    image_bytes = base64.b64decode(image_str)
    image = Image.open(io.BytesIO(image_bytes))
    label = ['Light Diseases and Disorders of Pigmentation','Lupus and other Connective Tissue diseases','Acne and Rosacea Photos','Systemic Disease',
 'Poison Ivy Photos and other Contact Dermatitis',
 'Vascular Tumors',
 'Urticaria Hives',
 'Atopic Dermatitis Photos',
 'Bullous Disease Photos',
 'Hair Loss Photos Alopecia and other Hair Diseases',
 'Tinea Ringworm Candidiasis and other Fungal Infections',
 'Psoriasis pictures Lichen Planus and related diseases',
 'Melanoma Skin Cancer Nevi and Moles',
 'Nail Fungus and other Nail Disease',
 'Scabies Lyme Disease and other Infestations and Bites',
 'Eczema Photos',
 'Exanthems and Drug Eruptions',
 'Herpes HPV and other STDs Photos',
 'Seborrheic Keratoses and other Benign Tumors',
 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
 'Vasculitis Photos',
 'Cellulitis Impetigo and other Bacterial Infections',
 'Warts Molluscum and other Viral Infections']
    return prepare_detail(image,model3,label)


@app.route('/Retina', methods=['POST'])
def Retina():
    data = request.get_json()
    image_str = data['image_rec']
    num_padding_chars = 4 - len(image_str) % 4
    image_str += "=" * num_padding_chars
    image_bytes = base64.b64decode(image_str)
    image = Image.open(io.BytesIO(image_bytes))
    label = ['NO_DR', 'Mild','Moderate','Severe','Proliferate_DR' ]
    return prepare_detail(image,model4,label)

# Define home page route
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

