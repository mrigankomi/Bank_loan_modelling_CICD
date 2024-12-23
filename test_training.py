import pytest
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from joblib import load
from bank_loan_model_training import (
    clean_df, preprocess_data,
    train_model, get_classification_report, load_model_artifact,
)
import pandas as pd

@pytest.fixture
def dummy_data():
    # Prepare dummy data for testing
    data = {
        'ID': [30, 40, 50],
        'AGE': [55,38,40],
        'Exprience': [3,6,7],
        'Income': [49,81,63],
        'ZIP Code': [91107,90089,94720],
        'Family': [4,3,1],
        'CCAvg': [1.6,1.5,2.7],
        'Education': [2,1,3],
        'Mortgage': [0,155,104],
        'Personal Loan': [0,1,0],
        'Securities Account': [1,1,0],
        'CD Account': [1,0,0],
        'Online': [0,1,1],
        'CreditCard': [0,1,0],

    }
    return pd.DataFrame(data)


def test_preprocess_features(dummy_data):
    df = dummy_data
    X,y = clean_df(df)
    X_train, X_test, y_train, y_test = preprocess_data(X,y)

    assert X.shape == (3, 12)  
    assert y.shape == (3,)


def test_get_classification_report(dummy_data):
    df = dummy_data
    X,y = clean_df(df)
    X_train, X_test, y_train, y_test = preprocess_data(X,y)
    model = load_model_artifact('xgboost_model.joblib')
    report = get_classification_report(model, X, y)
    assert isinstance(report, dict)
    assert '0' in report.keys()  
    

def test_train_model(dummy_data):
    df = dummy_data
    X,y = clean_df(df)
    X_train, X_test, y_train, y_test = preprocess_data(X,y)
    model = train_model('xgboost', X_train, y_train)
    assert isinstance(model, Pipeline)
