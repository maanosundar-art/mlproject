import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from sklearn.metrics import r2_score
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test input data')
            X_train,y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'XGBRegressor': XGBRegressor(),
                'SVR': SVR(),
                'KNN': KNeighborsRegressor(),
                'CatBoosting Regressor': CatBoostRegressor(verbose=False)
            }
            params = {
                'Decision Tree': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                'Random Forest': {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'Gradient Boosting': {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'AdaBoost': {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'XGBRegressor': {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'SVR': {
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
                },
                'KNN': {
                    'n_neighbors': [5, 7, 9, 11, 13, 15, 20]
                },
                'CatBoosting Regressor': {
                    'depth': [6, 8, 10],
                    'learning_rate': [.1, .01, .05, .001],
                    'iterations': [30, 50, 100]
                }
            }

            model_report:dict = evaluate_models(X_train, y_train, x_test, y_test, models=models, param=params)
            #best_model_score = max(sorted(model_report.values()))
            #best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model_name, best_model_info = max(model_report.items(), key=lambda x: x[1]['score'])
            best_model_score = model_report[best_model_name]['score']
            best_model = model_report[best_model_name]['model']
            #best_model_name = max(model_report, key=lambda x: model_report[x]["score"])
            #best_model_score = model_report[best_model_name]["score"]
            #best_model = model_report[best_model_name]["model"]


            #best_model = models[best_model_name]
            if best_model_score < 0.6:
                logging.info('No best model found')
                raise CustomException('No best model found')
            logging.info('Best model found, on both training and testing dataset')
        
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            return(best_model_name, best_model_score)
        except Exception as e:
            logging.info('Exception occured at model training stage')
            raise CustomException(e, sys)

