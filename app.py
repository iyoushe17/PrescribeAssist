import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle


modelPath = "models\PrescribeAssistModel.sav"
vectPath = "models\PrescribeAssistVectorizer.sav"
prescribeDBPath = "data\prescribeDB.csv"

prescriberModel = pickle.load(open(modelPath, 'rb'))
prescriberVectorizer = pickle.load(open(vectPath, 'rb'))

df = pd.read_csv(prescribeDBPath)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptomEntered = request.form['message']
        conditionPredicted = prescriberModel.predict(prescriberVectorizer.transform([symptomEntered]))[0]

    df_result = df.loc[df.condition_cat==conditionPredicted]

    drugNameFetched = df_result.iloc[0,0]
    conditionFetchedFromPrediction = df_result.iloc[0,1]
    return render_template('home.html', dispCondition= f"condition: {conditionFetchedFromPrediction}", dispDrugname = f"drug name: {drugNameFetched}")

if __name__ == '__main__':
	app.run(debug=True)