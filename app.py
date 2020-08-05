from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
import cloudpickle
import json
from json import JSONEncoder
import numpy as np

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# cors = CORS(app, resources={
#     r"/": {
#         "origins": "*"
#     }
# })
model = cloudpickle.load(open('finalModel.sav', 'rb'))


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/forms')
def forms():
    return render_template('Predict.html')


@app.route("/predict", methods=['POST', 'GET'])
def do_post_search():
    data = request.data
    datadict = json.loads(data)

    weight = float(datadict["weight"])
    height = float(datadict["height"])
    bmi = weight / (height * height)
    diabetes = float(datadict["diabetes"])
    diaBp = float(datadict["diaBp"])
    sysBp = float(datadict["sysBp"])
    totChol = float(datadict["totChol"])
    glucose = float(datadict["glu"])
    bpmeds = float(datadict["bpmeds"])
    stroke = float(datadict["stroke"])
    hypertension = float(datadict["hypertension"])
    cigs = float(datadict["cigs"])
    age = float(datadict["age"])
    sex = float(datadict["sex"])

    int_features = [bmi, diabetes, diaBp, sysBp, totChol, glucose, bpmeds, stroke, hypertension, cigs, age, sex]
    final = [np.array(int_features)]
    predict = model.predict(final)
    prediction_prob = model.predict_proba(final)
    output = {
        "positive prediction": (prediction_prob[0][1] * 100).item(),
        "negative prediction": (prediction_prob[0][0] * 100).item()
    }

    bmr = bmr_calculator(sex, weight, height, age)

    json_output = json.dumps(output)
    return json_output

    # output = {
    #     "positive prediction": "gg",
    #     "negative prediction": "d"
    # }
    # json_output = json.dumps(output)
    # return json_output


def bmr_calculator(sex, weight, height, age):
    if sex == 1:
        BMR = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        return BMR
    elif sex == 0:
        BMR = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        return BMR


if __name__ == "_main_":
    app.run(debug=True)
