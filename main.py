import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from model_files.ml_model import predict_score

app = Flask('score_prediction')

# API GET sur numéro du client
# ----------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    client=0
    prediction=np.array([0])
    if request.method == 'GET':
        #client = request.get_json()
        client = request.args.get('client')
        print (client)
        with open('./model_files/model_clf_rf.bin', 'rb') as f_in:
            model = pickle.load(f_in)
            f_in.close()
    
        prediction = predict_score(client, model)
        print ('Prediction',prediction)
        pred = list(prediction)
        print (type(pred))
        print (pred)
        proba = pred[0]

    result = {
        'cli_prediction 0': str(proba[0]),
        'cli_prediction 1': str(proba[1])
    }
    return jsonify(result)

# Ping pour voir si serveur actif
# -------------------------------
@app.route('/ping', methods=['GET'])
def ping():
    return "Pinging Model!!"

# Saisie via une page HTML
# ------------------------
@app.route('/', methods=['GET', 'POST'])
def accueil():
    client=0
    prediction=np.array([0])
    if request.method == 'POST':
        client = request.form['Client']
        print (client)
        with open('./model_files/model_clf_rf.bin', 'rb') as f_in:
            model = pickle.load(f_in)
            f_in.close()
    
        prediction = predict_score(client, model)
        print ('Prediction',prediction)
        
    return render_template("index.html", cli=client, predict=prediction.tolist())



