import os
import pandas as pd
import pathlib
import joblib
from pathlib import Path
#from prediction_model.config import config
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config

#Load the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)

    #read the dataset
    _data = pd.read_csv(filepath)
    return _data

#Perform Serialization
def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    #print('Model has been saved under the name: {}'.format(config.MODEL_NAME))
    print(f"Model has been saved under the name {config.MODEL_NAME}")

#Perform Deserialization
def load_pipeline(pipeline_to_load):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model_loaded = joblib.load(save_path)
    print('Model has been loaded')
    return model_loaded