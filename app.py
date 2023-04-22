from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import numpy as np
import cv2
import tensorflow as tf

model = keras.models.load_model('keras_model.h5')

# Define model and image parameters
class_names = ['Surprise', 'Sad','Neutral','Happy', 'Angry', ]
pic_size = 224


# Initialize Flask app
app = Flask(__name__)


# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get image file from request
    img_file = request.files['image']
    print(type(img_file))

    # Read image file as numpy array
    img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
    print(type(img))
    print(img.shape)


    img = cv2.resize(img, (pic_size, pic_size))
    print(img.shape)
    img = img[np.newaxis, ...]  # add channel dimension
    print(img.shape)
    img = img.astype('float32') / 255.0  # normalize pixel values
    # print(img.shape)


    # Make prediction
    pred = model.predict(img)

    # Get predicted class name
    class_idx = np.argmax(pred)
    class_name = class_names[class_idx]

    # Return prediction as JSON response
    return jsonify({'class': class_name })


# Define home page route
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
