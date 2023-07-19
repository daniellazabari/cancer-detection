import datetime
from flask import Flask, request, render_template
import pandas as pd 
# from ms import app
from ms.functions import get_model_response

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
    feature_dict = request.get_json(force=True)
    if not feature_dict:
        return {"erro": "Body is empty."}, 500
    try:
        response = get_model_response(feature_dict)
    except ValueError as e:
        return {"erro": str(e).split('\n')[-1].strip()}, 500
    
    return response, 200

if __name__ == "__main__":
    app.run(debug=True)