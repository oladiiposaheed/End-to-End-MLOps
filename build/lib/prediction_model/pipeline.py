from sklearn.pipeline import Pipeline
 
import joblib
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing import preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

#Create pipeline to perform transformation 
classification_pipeline = Pipeline(
    [
       ('Mean Imputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
       ('Mode Imputation', pp.ModeImputer(variables=config.CAT_FEATURES)),
       ('DomainProcessing', pp.DomainProcessing(variable_to_modify=config.FEATURE_TO_MODIFY, variable_to_add=config.FEATURE_TO_ADD)),
       ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
       ('LabelEncoder', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
       ('LogTransform', pp.LogTransform(variables=config.LOG_FEATURES)),
       ('MinMaxScale', MinMaxScaler()),
       ('LogisticClassifier', LogisticRegression(random_state=0))

    ]
)
