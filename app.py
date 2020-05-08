import numpy as np
from flask import Flask, request, jsonify, render_template
import math
import pickle

app = Flask(__name__)
model = pickle.load(open('taxi.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    inputs = [int(x) for x in request.form.values()]
    final_inputs = [np.array(inputs)]
    prediction = model.predict(final_inputs)
    output = round(prediction[0],2)
    return render_template('index.html', prediction_text = "Predicted weekly rides = {}".format(math.floor(output)))

if __name__ == '__main__':
    app.run(debug=True)
