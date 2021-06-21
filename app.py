import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle


modelPath = "models/PrescribeAssistModel.sav"
vectPath = "models/PrescribeAssistVectorizer.sav"
prescribeDBPath = "data/prescribeDB.csv"

prescriberModel = pickle.load(open(modelPath, 'rb'))
prescriberVectorizer = pickle.load(open(vectPath, 'rb'))

df = pd.read_csv(prescribeDBPath)

app = Flask(__name__)

class Results:
    def __init__(self):
        self.symptom = ""
        self.drug = ""
        self.condition = ""
        self.conditionCheck = False
        self.drugCheck = False

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
    
    def validate_condition(self):
        self.conditionCheck = True

    def validate_drug(self):
        self.drugCheck= True

    def set_condition(self, condition):
        self.condition = condition

    def set_drug(self, drug):
        self.drug = drug

    def get_condition_status(self):
        return self.conditionCheck

    def get_drug_status(self):
        return self.drugCheck

result = Results()

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptomEntered = request.form['symptom']
        result.set_symptom(symptomEntered)
    return render_template('result.html', data = result)

@app.route('/conditionCorrect_yes')
def conditionCorrect_yes():
    result.validate_condition()
    return render_template('result.html', data = result)

@app.route('/drugCorrect_yes')
def drugCorrect_yes():
    result.validate_drug()
    return render_template('feedback_no.html', data = result)

@app.route('/drugCorrect_no')
def drugCorrect_no():
    return render_template('feedback_no.html', data = result)

@app.route('/changeWrongDrug', methods = ['POST'])
def changeWrongDrug():
    if request.method == 'POST':
        result.set_drug(request.form['drugCorrection'])
        result.validate_drug()
    return render_template('feedback_no.html', data = result)


@app.route('/changeBothConditionDrug', methods = ['POST'])
def changeBothConditionDrug():
    result.set_condition(request.form['conditionCorrection'])
    result.set_drug(request.form['drugCorrection'])
    result.validate_condition()
    result.validate_drug()
    return render_template('feedback_no.html', data = result)

if __name__ == '__main__':
	app.run(debug=True)
