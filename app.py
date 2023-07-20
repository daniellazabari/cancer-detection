import datetime
from flask import Flask, request, render_template, jsonify
import pandas as pd 
from ms import model
from ms.functions import get_model_response, predict
import numpy as np

app = Flask(__name__)

model_name = "Breast Cancer Wisconsin (Diagnostic)"
model_file = "cancer_detection_model.dat.gz"
version = "v1.0.0"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/info", methods=["GET"])
def info():
    """
    Return model information, version, how to call
    """
    result = {}
    result["name"] = model_name
    result["version"] = version
    return result

@app.route("/about", methods=["GET"])
def about():
    """
    Return about information
    """
    return render_template("about.html")

@app.route("/health", methods=["GET"])
def health():
    """
    Return service health
    """
    return "ok"

@app.route("/predict", methods=["POST"])
def predict():
    features = request.form.to_dict()
    response = get_model_response(features)
    label = response["label"]
    if label == "M":
        label = "malignant"
    else:
        label = "benign"
    return render_template("index.html", prediction_text="The patient is more likely to have {} cancer.".format(label))
    
@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run()

