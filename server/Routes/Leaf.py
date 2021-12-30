import base64
import numpy as np
from tensorflow import keras
from flask import Flask, request, abort, redirect, url_for, jsonify
import io, sys
import cv2
import json
import requests

app = Flask(__name__)

#server URL
url = 'http://localhost:8501/v1/models/leaves_classifier:predict'

def get_disease():

def make_prediction():
    img = keras.preprocessing.image.load_img('test.jpeg', target_size=(256, 256))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image = np.vstack([x])
    headers = {"content-type": "application/json"}
    data = json.dumps({"instances": image.tolist()})
    json_response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']

    return predictions
@app.route("/leaf", methods=['post'])
def upload_file():
    file = request.files['image'].read()
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.imwrite('test.jpeg', img)
    prediction = make_prediction()
    return jsonify({'result': prediction})

@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response

if __name__ == "__main__":
	app.run()

