import joblib
import numpy as np
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        # Load model and (optionally) a saved label encoder
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        encoder_path = Path('artifacts/data_transformation/score_category_label_encoder.joblib')
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
        else:
            self.label_encoder = None

    def predict(self, data):
        prediction = self.model.predict(data)
        # If label encoder is available, return the text label
        if self.label_encoder is not None:
            try:
                text_label = self.label_encoder.inverse_transform([int(prediction[0])])[0]
                return text_label
            except Exception:
                return prediction[0]
        return prediction[0]