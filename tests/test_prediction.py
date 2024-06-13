import pytest
from pathlib import Path
import os
import sys

# Adding the below path to avoid module not found errorcurrent_directory = os.path.dirname(os.path.realpath(__file__))
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.predict import generate_predictions


#Note
#i. Output from predict script must not be null
#ii. Output from predict script should be str data type
#iii. The output is Y for an example data

@pytest.fixture
def single_prediction():
    test_dataset = load_dataset(config.TEST_FILE)
    single_row = test_dataset[:1]
    result = generate_predictions(single_row)
    return result

#Check the output is not None
def test_single_pred_not_null(single_prediction):
    assert single_prediction is not None

#Check the output data type is a string
def test_single_pred_str_type(single_prediction):
    assert isinstance(single_prediction.get('predictions')[0], str)

#Check the output is Y
def test_single_pred_validate(single_prediction):
    assert single_prediction.get('predictions')[0] == 'Y'