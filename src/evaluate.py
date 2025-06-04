# Import required libraries
import pandas as pd
import pickle
import yaml
import os
import mlflow
from sklearn.metrics import accuracy_score

from urllib.parse import urlparse

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/sl-93/MachineLearningPipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "sl-93"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "93f57a140660e90e516e05097eed357d2b9bfd62"

params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Load the model from the disk
    model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    # Log metrics to mlflow
    mlflow.log_metric("accuracy", accuracy)

    print(f"Model Accuracy: {accuracy}")

if __name__ == "__main__":
    evaluate(params["data"], params["model"])
