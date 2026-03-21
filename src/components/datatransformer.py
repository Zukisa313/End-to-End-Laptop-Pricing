import os 
import sys
from dataclasses import dataclass
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.exception import CustomException
import numpy as np
import pandas as pd
from src.logger import logging
from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class DataTransformationConfig:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    preprocessor_obj_file_path = os.path.join(root_dir, 'artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_tranformer_config = DataTransformationConfig()
    
    def get_data_tranformer_object(self):
        """"
        This function will transform our features
        """

        try:
            numerical_columns = ['Inches', 'Ram', 'Weight', 'Screen_width',
             'Screen_height']
            categorical_columns = ['Company', 'TypeName',
             'Cpu', 'Memory', 'OpSys']

            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]

             )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )

            # fit transform
            logging.info("One hot encoding standard scaling implemented")

            preprocessor =ColumnTransformer([
                ("num_pipeline", num_pipeline,numerical_columns),
                ("cat_pipeline", cat_pipeline,categorical_columns)
            ]
                
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor = self.get_data_tranformer_object()
            target_columm = "Price_euros"

            input_train_feature = train_df.drop(columns=[target_columm])
            target_train_feature= train_df[target_columm]

            input_test_feature = test_df.drop(columns=[target_columm])
            target_test_feature= test_df[target_columm]


            input_train_feature_arr = preprocessor.fit_transform(input_train_feature).toarray()
            input_test_feature_arr = preprocessor.transform(input_test_feature).toarray()
            
            target_train_feature = target_train_feature.to_numpy().reshape(-1, 1)
            target_test_feature = target_test_feature.to_numpy().reshape(-1, 1)

            train_arr = np.c_[input_train_feature_arr, target_train_feature]
            test_arr = np.c_[input_test_feature_arr, target_test_feature]

            logging.info(f"Train and test array created")
            save_object(
                file_path = self.data_tranformer_config.preprocessor_obj_file_path,
                obj = preprocessor
            )
            return (
                train_arr,
                test_arr,
                self.data_tranformer_config.preprocessor_obj_file_path,
            )



        except Exception as e:
            raise CustomException(e,sys)