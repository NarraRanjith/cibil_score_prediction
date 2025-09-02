
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from mlProject import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)


        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        # Dynamic model selection
        MODEL_MAP = {
            "RandomForest": RandomForestClassifier,
            "KNN": KNeighborsClassifier,
            "DecisionTree": DecisionTreeClassifier
        }
        model_cls = MODEL_MAP[self.config.model_type]
        model = model_cls(**self.config.model_params)
        model.fit(train_x, train_y)
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))

