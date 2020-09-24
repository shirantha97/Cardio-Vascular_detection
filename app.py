from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
import pickle
import json
from json import JSONEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# cors = CORS(app, resources={
#     r"/": {
#         "origins": "*"
#     }
# })

model = pickle.load(open('finalModel.sav', 'rb'))


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/forms')
def forms():
    return render_template('Predict.html')


@app.route("/predict", methods=['POST', 'GET'])
def do_post_search():
    # print(request.query_string)
    data = request.data
    datadict = json.loads(data)

    weight = float(datadict["weight"])
    height = float(datadict["height"])
    bmi = weight / ((height / 100) * (height / 100))
    cigs = float(datadict["cigs"])
    age = float(datadict["age"])
    sex = float(datadict["sex"])
    diaBp = float(datadict["diaBp"])
    sysBp = 120.0
    totChol = float(datadict["totChol"])
    glucose = float(datadict["glu"])
    alco = 1.0 if cigs == 1.0 else 0
    diabetes = float(datadict["diabetes"])
    glucose_level = glucose_level_check(glucose, diabetes)
    cholesterol_level = cholesterol_check(age, totChol, sex)
    cardio = 0 if cholesterol_level == 3 else 1
    exerciseRate = float(datadict["exerciseRate"])

    bmr = bmr_calculator(sex, weight, height, age)
    tdee = TDEEcalculator(bmr, exerciseRate)

    # bpmeds = float(datadict["bpmeds"])
    # stroke = float(datadict["stroke"])
    # hypertension = float(datadict["hypertension"])

    int_features = [(age * 365), sex, height, weight, diaBp, sysBp, cholesterol_level, glucose_level, cigs, alco,
                    cardio, bmi]
    print(int_features)
    final = [np.array(int_features)]
    predict = model.predict(final)
    prediction_prob = model.predict_proba(final)

    output = {
        "positive_prediction": (prediction_prob[0][1] * 100),
        "negative_prediction": (prediction_prob[0][0] * 100),
        "bmr": bmr,
        "tdee": tdee
    }

    json_output = json.dumps(output)
    return json_output

    # output = {
    #     "positive prediction": "gg",
    #     "negative prediction": "d"
    # }
    # json_output = json.dumps(output)
    # return json_output


def glucose_level_check(glucose, diabetes):
    glucose_normal = 1
    glucose_above_normal = 2
    glucose_well_above = 3
    if diabetes == 0:
        if 70.0 <= glucose <= 85.0:
            return glucose_normal
        elif 85.0 < glucose <= 99.0:
            return glucose_above_normal
        elif 99.0 < glucose:
            return glucose_well_above
    elif diabetes == 1:
        if 80.0 <= glucose <= 120.0:
            return glucose_normal
        elif 120.0 < glucose <= 130.0:
            return glucose_above_normal
        elif 130.0 < glucose:
            return glucose_well_above


def cholesterol_check(age, totChol, sex):
    chol_normal = 1
    chol_above_normal = 2
    chol_well_above = 3
    if age <= 19:
        if totChol <= 170.0:
            return chol_normal
        elif 200.0 < totChol <= 239.0:
            return chol_above_normal
        elif 239.0 < totChol:
            return chol_well_above
    elif age > 20:
        if 125.0 < totChol <= 170.0:
            return chol_normal
        elif 170.0 < totChol <= 239.0:
            return chol_above_normal
        elif 239.0 < totChol:
            return chol_well_above


def bmr_calculator(sex, weight, height, age):
    if sex == 1:
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        return bmr
    elif sex == 0:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        return bmr


def TDEEcalculator(bmr, exerciseRate):
    if exerciseRate == 0:
        return bmr * 1.2
    elif 1 <= exerciseRate <= 3:
        return bmr * 1.2
    elif 3 < exerciseRate <= 5:
        return bmr * 1.375
    elif 6 <= exerciseRate <= 7:
        return bmr * 1.725


if __name__ == "_main_":
    app.run(debug=True)
