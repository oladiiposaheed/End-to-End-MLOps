import pandas as pd
import numpy as np 
import joblib
from pathlib import Path
import os
import sys

current_directory = os.path.dirname(os.path.realpath(__file__))
PACKAGE_ROOT = os.path.dirname(current_directory)
#PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline, load_dataset
#from sklearn.metrics import predictions

classification_pipeline = load_pipeline(pipeline_to_load=config.MODEL_NAME) #To load the pipeline

def generate_predictions(data_input):
   data = pd.DataFrame(data_input)
   pred = classification_pipeline.predict(data[config.FEATURES])
   output = np.where(pred==1, 'Y', 'N')
   result = {'predictions': output}
   return result

# def generate_predictions():
#     test_data = load_dataset(config.TEST_FILE)
#     pred = classification_pipeline.predict(test_data[config.FEATURES])
#     output = np.where(pred==1, 'Y', 'N')
#     print(output)
#     result = {'Predictions': output}
#     return output

if __name__=='__main__':
    generate_predictions()