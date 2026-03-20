import os 
import sys



from src.exception import CustomException
from src.logger import logging
from src.components.datatransformer import DataTransformation
from src.components.model_trainer import ModelTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass #built in dont install
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
@dataclass
class DataIngestionConfig:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    train_data_path: str = os.path.join(root_dir, 'artifact', 'train.csv')
    test_data_path: str = os.path.join(root_dir, 'artifact', 'test.csv')
    raw_data_path: str = os.path.join(root_dir, 'artifact', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method function")
        try:
            data_path = os.path.join(ROOT_DIR, "notebooks", "data", "laptop_price_edited.csv")
            df = pd.read_csv(data_path)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set , test_set  = train_test_split(df,test_size=0.3, random_state=47)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    bj = DataIngestion()
    train_path , test_path= bj.initiate_data_ingestion()
    datatransformer = DataTransformation()
    train_arr, test_arr, preprocess_path = datatransformer.initiate_data_transformation(train_path,test_path)
    model_trainer = ModelTrainer()
    v = model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)
    print(v)
