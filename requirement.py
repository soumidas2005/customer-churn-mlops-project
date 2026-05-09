import os
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

import mlflow
import mlflow.sklearn
import joblib
import numpy as np

model = joblib.load("churn_model.pkl")

X_dummy = np.random.rand(10, model.n_features_in_)

y_prob = model.predict_proba(X_dummy)[:, 1]

with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("dummy_auc", float(np.mean(y_prob)))
    mlflow.sklearn.log_model(model, "model")

print("MLflow run completed")