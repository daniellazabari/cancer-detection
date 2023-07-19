from flask import Flask
import joblib


# Load models
model = joblib.load("model/cancer_detection_model.dat.gz")