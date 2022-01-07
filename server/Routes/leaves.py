import numpy as np
from tensorflow import keras
from flask import Flask, request, jsonify
import sys
import cv2
import json
import requests
# tensorflow serving URL
url = 'http://localhost:8501/v1/models/leaves_classifier:predict'

app = Flask(__name__)

def check_result(predictions):

    f = open('diseases.json', encoding='utf8')
    diseases_data = json.load(f)
    result = np.amax(predictions)
    disease = predictions.tolist().index(result)

    if disease == 0:
        predicted_disease = diseases_data['first_disease']
    elif disease == 1:
        predicted_disease = diseases_data['second_disease']
    elif disease == 2:
        predicted_disease = 'healthy'
    elif disease == 3:
        predicted_disease = diseases_data['third_disease']
    elif disease == 4:
        predicted_disease = diseases_data['fourth_disease']
    elif disease == 5:
        predicted_disease = diseases_data['fifth_disease']
    else:
        predicted_disease = 0
    f.close()

    response = {"accuracy": result, "disease_info": predicted_disease}
    return response
def make_prediction():
    img = keras.preprocessing.image.load_img('test.jpeg', target_size=(256, 256))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image = np.vstack([x])
    headers = {"content-type": "application/json"}
    data = json.dumps({"instances": image.tolist()})
    json_response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    predictions = np.reshape(predictions, -1)
    response = check_result(predictions)
    return response
@app.route("/leaf", methods=['post'])
def upload_file():
    file = request.files['image'].read()
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.imwrite('test.jpeg', img)
    prediction = make_prediction()
    return jsonify(result=prediction)
@app.after_request
def after_request(response):
    print("log: setting cors", file=sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response


if __name__ == "__main__":
    app.run()
