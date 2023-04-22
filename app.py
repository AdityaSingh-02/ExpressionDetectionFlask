from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import numpy as np
import cv2
import tensorflow as tf
import requests
from PIL import Image
from io import BytesIO

model = keras.models.load_model('keras_model.h5')

# Define model and image parameters
class_names = ['Surprise', 'Sad','Neutral','Happy', 'Angry', ]
pic_size = 224


# Initialize Flask app
app = Flask(__name__)


# Define prediction route
@app.route('/predict/<path:img_url>', methods=['POST', 'GET'])
def predict(img_url):
    # Get image file from request
    # img_file = request.files['image']
    fetched_Image = img_url
    response = requests.get(fetched_Image)
    image = Image.open(BytesIO(response.content))
    if response.status_code == 200:
        image = image.resize((224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

        prediction = model.predict(tf.expand_dims(image, axis=0))
        expression = ['Surprise', 'Sad','Neutral','Happy', 'Angry', ][prediction.argmax()]
        return jsonify({'expression': expression})
    else:
        return "Failure"

        
    # print(type(img_file))

    # # Read image file as numpy array
    # img = cv2.imdecode(np.frombuffer(fetched_Image.read(), np.uint8), cv2.IMREAD_COLOR)
    # print(type(img))
    # print(img.shape)


    # img = image.resize(img,(pic_size, pic_size))
    # # print(img.shape)
    # img = img[np.newaxis, ...]  # add channel dimension
    # # print(img.shape)
    # img = img.astype('float32') / 255.0  # normalize pixel values
    # # print(img.shape)


    # # Make prediction
    # pred = model.predict(img)

    # # Get predicted class name
    # class_idx = np.argmax(pred)
    # class_name = class_names[class_idx]

    # # Return prediction as JSON response

    # return jsonify({'class': class_name })


# Define home page route
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
