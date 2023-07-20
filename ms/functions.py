import pandas as pd
from ms import model

def predict(X, model):
    prediction = model.predict(X)[0]
    return prediction

def get_model_response(data):
    X = pd.DataFrame(data, index=[0])
    prediction = predict(X, model)
    if prediction == 1:
        label = "M"
    else:
        label = "B"
    return {
        "status": 200,
        "label": label,
        "prediction": int(prediction)
    }