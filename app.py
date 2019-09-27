import pickle
from flask import Flask, request
import numpy as np
import pandas as pd
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

with open('model.pkl', 'rb') as model_pickle:
    model = pickle.load(model_pickle)

@app.route('/predict', methods=["GET"])
def predict_iris():
    """Example endpoint returning prediction of Iris flowers.
    ---
    parameters:
      - name: S_length
        in: query
        type: string
        required: true
      - name: S_width
        in: query
        type: string
        required: true
      - name: P_length
        in: query
        type: string
        required: true
      - name: P_width
        in: query
        type: string
        required: true
    """

    s_length = request.args.get('S_length')
    s_width = request.args.get('S_width')
    p_length = request.args.get('P_length')
    p_width = request.args.get('S_length')

    
    # the prediction
    prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))
    return str(prediction[0])

# prediction from file
@app.route('/predict_file', methods=["POST"])
def predict_file():
    """Example endpoint returning prediction of Iris Flower prediction from file.
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    """
    
    # Get the file
    file = pd.read_csv(request.args.get("input_file"), header=None)
    
    # the prediction
    prediction = model.predict(np.array(file))
    return str(list(prediction))


if __name__ == '__main__':
    app.run(debug=True)