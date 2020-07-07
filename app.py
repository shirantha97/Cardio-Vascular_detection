from flask import Flask, render_template, request, redirect
import cloudpickle
import json
import numpy as np

app = Flask(__name__, template_folder='templates')
model = cloudpickle.load(open('finalModel.sav', 'rb'))


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/forms')
def forms():
    return render_template('Predict.html')


@app.route("/predict", methods=['POST', 'GET'])
def do_post_search():
    int_features = [float(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(final)
    predict = model.predict(final)
    print(predict)
    return render_template('home.html')


if __name__ == "_main_":
    app.run(debug=True)
