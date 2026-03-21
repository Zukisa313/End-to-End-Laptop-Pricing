import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..")))

import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = 'artifact\\model.pkl'
            preprocessor_path = 'artifact\\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocess = load_object(file_path=preprocessor_path)
            data_scaled = preprocess.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 Company:str,
                 TypeName:str,
                 Inches:float,
                 Cpu:str,
                 Ram:str,
                 Memory:str,
                 OpSys:str,
                 Weight:float,
                 Screen_width:float,
                 Screen_height:float):
        self.Company = Company
        self.TypeName = TypeName
        self.Inches = Inches
        self.Cpu = Cpu
        self.Ram = Ram
        self.Memory = Memory
        self.OpSys = OpSys
        self.Weight = Weight
        self.Screen_width = Screen_width
        self.Screen_height = Screen_height
    def convert_to_dataFrame(self):
        try:

            custom_data_input = {
            "Company": self.Company,
            "TypeName": self.TypeName,
            "Inches": self.Inches,
            "Cpu": self.Cpu,
            "Ram": self.Ram,
            "Memory": self.Memory,
            "OpSys": self.OpSys,
            "Weight": self.Weight,
            "Screen_width": self.Screen_width,
            "Screen_height": self.Screen_height,

        }
            return pd.DataFrame([custom_data_input])
        except Exception as e:
            raise CustomException(e,sys)
            
        
