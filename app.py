import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

modelPath = "models/PrescribeAssistModel.sav"
vectPath = "models/PrescribeAssistVectorizer.sav"
prescribeDBPath = "data/prescribeDB_small.csv"

prescriberModel = pickle.load(open(modelPath, 'rb'))
prescriberVectorizer = pickle.load(open(vectPath, 'rb'))

df = pd.read_csv(prescribeDBPath)

app = Flask(__name__)

class Results:
    def __init__(self):
        self.symptom = ""
        self.drug = ""
        self.condition = ""
        self.conditionStatus = -1
        self.drugStatus = -1

    def get_symptom(self):
        return self.symptom

    def set_symptom(self, symptom):
        self.symptom = symptom
        self.predict_condition()

    def predict_condition(self):
        conditionNumerics = prescriberModel.predict(prescriberVectorizer.transform([self.symptom]))[0]
        df_result = df.loc[df.condition_cat == conditionNumerics]
        self.drug = df_result.iloc[0,0]
        self.condition = df_result.iloc[0,1]

    def get_drug(self):
        return self.drug

    def get_condition(self):
        return self.condition

    def reset_params(self):
        self.symptom = ""
        self.drug = ""
        self.condition = ""
        self.conditionStatus = -1
        self.drugStatus = -1

    def set_condition(self, condition):
        self.condition = condition

    def set_drug(self, drug):
        self.drug = drug

    def set_condition_status(self, status):
        self.conditionStatus = status

    def set_drug_status(self, status):
        self.drugStatus = status

    def get_condition_status(self):
        return self.conditionStatus

    def get_drug_status(self):
        return self.drugStatus


result = Results()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        result.reset_params()
        return render_template("home.html", data = result)
    else:
        return render_template("home.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptomEntered = request.form['symptom']
        result.set_symptom(symptomEntered)
    return render_template('result.html', data = result)


@app.route('/conditionCorrect_yes')
def conditionCorrect_yes():
    result.set_condition_status(1)
    return render_template('result.html', data = result, render = True)


@app.route('/drugCorrect_yes')
def drugCorrect_yes():
    result.set_drug_status(1)
    return render_template('result.html', data = result, render = True)


@app.route('/drugCorrect_no')
def drugCorrect_no():
    result.set_drug_status(0)
    return render_template('result.html', data = result, render = True)


@app.route('/changeWrongDrug', methods = ['POST'])
def changeWrongDrug():
    if request.method == 'POST':
        result.set_drug(request.form['drugCorrection'])
        result.set_drug_status(1)
    return render_template('result.html', data = result, render = True)


@app.route('/conditionCorrect_no')
def conditionCorrect_no():
    result.set_condition_status(0)
    return render_template('result.html', data = result, render = True)


@app.route('/changeBothConditionDrug', methods = ['POST'])
def changeBothConditionDrug():
    result.set_condition(request.form['conditionCorrection'])
    result.set_drug(request.form['drugCorrection'])
    result.set_condition_status(1)
    result.set_drug_status(1)
    return render_template('result.html', data = result)


if __name__ == '__main__':
	app.run(debug=True)
