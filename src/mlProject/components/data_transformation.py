import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import LabelEncoder
import joblib

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)
        data = data.drop(columns=["Name","Occupation","Bank"],axis=1) 
        # Encode the score category column to integers
        if 'Score_Category' in data.columns:
            le = LabelEncoder()
            data['Score_Category'] = le.fit_transform(data['Score_Category'])
            # Save the fitted encoder for use in prediction
            
            encoder_path = os.path.join(self.config.root_dir, 'score_category_label_encoder.joblib')
            joblib.dump(le, encoder_path)
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        