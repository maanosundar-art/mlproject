import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transform_object(self):
        logging.info('Data transformation initiated')
        try:
            num_col = ['reading score', 'writing score']
            cat_col = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy = 'median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy= 'most_frequent')),
                    ('one_hot_encoder', OneHotEncoder())                    
                ]
            )
            logging.info('Pipeline for numerical and categorical columns created')
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_col),
                ('cat_pipeline', cat_pipeline, cat_col)
            ])
            return preprocessor
        except Exception as e:
            logging.info('Exception occured in data transformation stage')
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df =pd.read_csv(test_path)
            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessor object')
            preprocessor_obj = self.get_data_transform_object()
            target_col_name = 'math score'
            input_feature_train_df = train_df.drop(columns=[target_col_name], axis=1)
            target_feature_train_df = train_df[target_col_name]
            #input feature test dataframe
            input_feature_test_df = test_df.drop(columns=[target_col_name], axis=1)
            target_feature_test_df = test_df[target_col_name]
            logging.info('Applying preprocessing object on training and testing dataframe')
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info('Saved preprocessing object')
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            return (
                train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
