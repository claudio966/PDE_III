import numpy as np
from tensorflow import keras
from flask import Flask, request, jsonify
import sys
import cv2
import json
import os
import time
import requests
# tensorflow serving URL
url = 'http://tensorflow-serving:8501/v1/models/leaves_classifier:predict'

app = Flask(__name__)

def check_result(predictions, file_name):
    # abre o arquivo json contendo dados sobre cada doença classificável pelo sistema.
    f = open('diseases.json', encoding='utf8')
    # carrega o arquivo em uma variável na forma de dicionário.
    diseases_data = json.load(f)
    # o resultado final será aquele que apresentar maior acurácia.
    result = np.amax(predictions)
    # obtém a posição em que o maior valor se encontra.
    disease = predictions.tolist().index(result)
    # verifica a validade do resultado. Para que esse tenha valor, alguns
    # pré-requisitos devem ser cumpridos. Senão, temos um resultado
    # inconclusivo.
    if np.round(result) < 0.5:
        predicted_disease = 'inconclusive'
    # verificados os requisitos, podemos carregar as informações necessárias.
    # cada número corresponde a uma certa posição na array de resultados e
    # a verificação é necessária para
    else:
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
    # após a verificação, o arquivo salvo pode ser removido.
    os.remove(file_name)
    # formata a resposta para envio.
    response = {"accuracy": result, "disease_info": predicted_disease}
    return response
def make_prediction(file_name):
    # carrega a imagem salva
    img = keras.preprocessing.image.load_img(file_name, target_size=(256, 256))
    # converte a imagem para array
    x = keras.preprocessing.image.img_to_array(img)
    # acrescenta uma dimensão à array original.
    x = np.expand_dims(x, axis=0)
    # "empilha" os itens da array
    image = np.vstack([x])
    # define o cabeçalho a ser utilizado na requisição ao tf serving
    headers = {"content-type": "application/json"}
    # converte os dados para a forma de lista e define os dados na forma de json
    data = json.dumps({"instances": image.tolist()})
    # envia os dados ao tensorflow serving.
    json_response = requests.post(url, data=data, headers=headers)
    # carrega a informação recebida.
    predictions = json.loads(json_response.text)['predictions']
    # converte a array de formato (n, 1) para (n, none)
    predictions = np.reshape(predictions, -1)
    # chama a função responsável por verificar a validade do resultado obtido
    # tal como obter os dados necessários para o usuário
    response = check_result(predictions, file_name)
    return response
# rota principal da api
@app.route("/leaf", methods=['post'])
def upload_file():
    # lê o arquivo enviado por meio do método POST.
    file = request.files['image'].read()
    # codifica a string lida em uma array de valores unsigned int de 8 bits(0-255).
    npimg = np.fromstring(file, np.uint8)
    # converte a array em uma imagem colorida.
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # atribui um nome aleatório ao arquivo.
    file_name = str(time.time()) + '.jpeg'
    # salva o arquivo para uso futuro.
    cv2.imwrite(file_name, img)
    # faz a conexão da api com o tensorflow serving e obtém o resultado.
    prediction = make_prediction(file_name)
    # retorna a resposta ao usuário
    return jsonify(result=prediction)
# habilita a troca de dados com domínios distintos, essencial para o funcionamento da api.
@app.after_request
def after_request(response):
    print("log: setting cors", file=sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response
if __name__=="__main__":
    app.run()
