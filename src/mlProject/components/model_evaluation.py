
import os
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
from mlProject.utils.common import save_json
import dagshub

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        cm = confusion_matrix(actual, pred)
        cr = classification_report(actual, pred)
        return accuracy, cm, cr

    def log_into_mlflow(self):
        # Authenticate with DagsHub for MLflow
        dagshub.init(repo_owner='narraranjith22', repo_name='cibil_score_prediction', mlflow=True)

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]  # Ensure 1D array for sklearn metrics

        # Set MLflow tracking URI and registry URI if provided
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)

            accuracy, cm, cr = self.eval_metrics(test_y, predicted_qualities)

            # Saving metrics as local
            scores = {
                "accuracy": accuracy,
                "confusion_matrix": cm.tolist(),
                "classification_report": cr
            }
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("accuracy", accuracy)
            # Log confusion matrix and classification report as artifacts or text, not as metrics
            mlflow.log_text(str(cm), "confusion_matrix.txt")
            mlflow.log_text(cr, "classification_report.txt")

            # Model registry does not work with file store
            #mlflow.sklearn.log_model(model, "model")