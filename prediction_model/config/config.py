import pathlib
import os
import prediction_model

#PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

current_directory = os.path.dirname(os.path.realpath(__file__))
PACKAGE_ROOT = os.path.dirname(current_directory)

DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")


TRAIN_FILE = 'train_loan.csv'
TEST_FILE = 'test_loan.csv'

MODEL_NAME = 'classification.pkl'

SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'trained_models')

TARGET = 'Loan_Status'

#Final features used for the model
FEATURES = ['Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area']


NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

CAT_FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 
                'Self_Employed', 'Credit_History', 'Property_Area']

#Same as categorical features
FEATURES_TO_ENCODE = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                      'Credit_History', 'Property_Area']

FEATURE_TO_MODIFY = ['ApplicantIncome']
FEATURE_TO_ADD = 'CoapplicantIncome'

DROP_FEATURES = ['CoapplicantIncome']

LOG_FEATURES = ['ApplicantIncome', 'LoanAmount'] #Taking log of numerical columns


