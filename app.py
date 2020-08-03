from flask import Flask, render_template, request, redirect, url_for
import cloudpickle
import json
from json import JSONEncoder
import numpy as np

app = Flask(__name__,static_url_path='', static_folder='static', template_folder='templates')
model = cloudpickle.load(open('finalModel.sav', 'rb'))


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/forms')
def forms():
    return render_template('Predict.html')


@app.route("/predict", methods=['POST', 'GET'])
def do_post_search():
    bmi = request.args.get('BMI')
    diabetes = request.args.get('diabetes')
    diaBp = request.args.get('diaBp')
    sysBp = request.args.get('sysBp')
    totChol = request.args.get('totChol')
    glucose = request.args.get('glucose')
    bpmeds = request.args.get('bpmeds')
    stroke = request.args.get('stroke')
    hypertension = request.args.get('hypertension')
    cigs = request.args.get('cigs')
    age = request.args.get('age')
    sex = request.args.get('sex')

    int_features = [bmi, diabetes, diaBp, sysBp, totChol, glucose, bpmeds, stroke, hypertension, cigs, age, sex]
    final = [np.array(int_features)]
    print(final)
    predict = model.predict(final)
    prediction_prob = model.predict_proba(final)
    print(prediction_prob)
    output = {
        "positive prediction": (prediction_prob[0][1]*100).item(),
        "negative prediction": (prediction_prob[0][0]*100).item()
    }
    json_output = json.dumps(output)
    return json_output
    # return render_template('home.html')


if __name__ == "_main_":
    app.run(debug=True)
