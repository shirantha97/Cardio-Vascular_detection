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

    # if request.method == 'POST':
    #     bmi = request.form.get("bmi")
    #     print(bmi)

    data = request.data
    datadict = json.loads(data)
    bmi = float(datadict["bmi"])
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
    print(final)
    predict = model.predict(final)
    prediction_prob = model.predict_proba(final)
    print(predict)
    output = {
        "positive prediction": (prediction_prob[0][1]*100).item(),
        "negative prediction": (prediction_prob[0][0]*100).item()
    }
    json_output = json.dumps(output)
    return json_output

    # output = {
    #     "positive prediction": "gg",
    #     "negative prediction": "d"
    # }
    # json_output = json.dumps(output)
    # return json_output


if __name__ == "_main_":
    app.run(debug=True)
